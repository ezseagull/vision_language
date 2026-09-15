import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_cra.only_trans import LanModel
from modules.m_tempclip.transformer import VideoEmbedding
from modules.m_cra.only_trans import Decoder as BaselineDecoder
from modules.m_cra.refine import Decoder as RefineDecoder
from modules.m_cra.causal_module import LinguisticInterventionGraph



class CRA(nn.Module):
    """
    CRA wrapper used as the downstream VideoQA testbed for Q-Bound V2.1.0.

    V2.1 scientific contract
    ------------------------
    Q-Bound itself is answer-free and receives only:

        local/generic video features + question-only TOKEN features

    It constructs one query-specific latent contiguous evidence event from
    local temporal primitives with one shared segment energy E(s,e).

    CRA is deliberately kept as a downstream answer-space testbed/reference:

      1) CRA temporal grounding is detached when Q-Bound is active.
      2) Q-Bound NEVER receives CRA x_bar as temporal input.
      3) Active Q-Bound answer evidence is pooled from temporally LOCAL
         Q-Bound content tokens, not globally contextualized CRA x_bar.
      4) Existing CRA QA candidates [Question SEP Answer] remain unchanged for
         the ordinary benchmark answer branch.
      5) V2.1 additionally accepts ANSWER-ONLY candidates. They are used only
         after Q-Bound has already produced V+Q evidence, providing a cleaner
         training signal for temporal evidence without feeding the GT answer or
         candidate answer semantics into the segment scorer.
      6) Answer-only embeddings are stop-gradient in this auxiliary path so the
         auxiliary objective must improve visual/temporal evidence rather than
         co-adapt the language target.
      7) V2.0 KEEP/MERGE and boundary-equilibrium verifiers are removed.

    Inference grounding remains:
        H* = QBound(Video, Question)

    Temporal GT / IoU / IoP are never consumed by this wrapper.
    """

    def __init__(self, cfgs, tokenizer) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.tokenizer = tokenizer

        model_cfg = cfgs["model"]
        dataset_cfg = cfgs["dataset"]

        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        self.baseline = model_cfg["baseline"]
        self.qmax_words = dataset_cfg["qmax_words"]
        self.max_feats = dataset_cfg["max_feats"]
        self.n_negs = model_cfg["n_negs"]

        self.qbound_enabled = bool(
            qbound_cfg.get(
                "enabled",
                model_cfg.get(
                    "qbound_enabled",
                    model_cfg.get("gpeb_enabled", False),
                ),
            )
        )
        self.qbound_version = str(
            qbound_cfg.get(
                "version",
                model_cfg.get(
                    "qbound_version",
                    model_cfg.get("gpeb_version", "2.1.0"),
                ),
            )
        )

        if self.qbound_enabled and self.qbound_version != "2.1.0":
            raise ValueError(
                "Q-Bound is enabled, but this CRA wrapper expects "
                f"model.qbound.version='2.1.0'; got {self.qbound_version!r}."
            )

        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get("qbound_require_question_only", True),
            )
        )
        if self.qbound_enabled and not self.qbound_require_question_only:
            raise ValueError(
                "Controlled Q-Bound V2.1 requires "
                "model.qbound.require_question_only=true."
            )

        self.qbound_require_token_question = bool(
            qbound_cfg.get("require_token_question", True)
        )
        if self.qbound_enabled and not self.qbound_require_token_question:
            raise ValueError(
                "Controlled Q-Bound V2.1 requires token-level question "
                "conditioning. Set model.qbound.require_token_question=true."
            )

        if self.qbound_enabled and bool(model_cfg.get("do", False)):
            raise ValueError(
                "Controlled Q-Bound V2.1 requires model.do=false. "
                "VisualIntervention would confound the temporal-composition "
                "experiment."
            )

        # Repository compatibility aliases.
        self.gpeb_enabled = self.qbound_enabled
        self.gpeb_version = self.qbound_version
        self.gpeb_require_question_only = self.qbound_require_question_only

        d_model = model_cfg["d_model"]

        self.lang_model = LanModel(
            tokenizer,
            lan=model_cfg["lan"],
            out_dim=d_model,
        )
        self.video_model = VideoEmbedding(cfgs)

        # Kept for repository/checkpoint compatibility only.
        self.do = LinguisticInterventionGraph(cfgs)

        if self.baseline == "baseline":
            self.decoder = BaselineDecoder()
            self.forward = self.baseline_forward
        elif self.baseline == "refine":
            self.decoder = RefineDecoder(
                num_layer=model_cfg["num_layers"],
                cfgs=cfgs,
            )
            if self.qbound_enabled:
                decoder_version = str(
                    getattr(self.decoder, "qbound_version", "")
                )
                if decoder_version != self.qbound_version:
                    raise RuntimeError(
                        "CRA/RefineDecoder Q-Bound version mismatch: "
                        f"CRA={self.qbound_version!r}, "
                        f"decoder={decoder_version!r}."
                    )
            self.forward = self.refine_forward
        else:
            raise ValueError(
                "Unknown model.baseline value: "
                f"{self.baseline!r}. Expected 'baseline' or 'refine'."
            )

    # ------------------------------------------------------------------
    # Language / input helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_answer_indices(
        answer_id,
        batch_size: int,
        device: torch.device,
    ):
        if answer_id is None:
            return None

        answer_indices = answer_id.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).reshape(-1)
        if answer_indices.numel() != batch_size:
            raise ValueError(
                "answer_id must contain one candidate index per sample. "
                f"Expected {batch_size} values, got "
                f"{answer_indices.numel()}."
            )
        return answer_indices

    @staticmethod
    def _select_candidate_features(
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must be [B,C,D], got "
                f"{tuple(candidate_features.shape)}."
            )
        if candidate_indices.ndim != 1:
            raise ValueError(
                "candidate_indices must be [B], got "
                f"{tuple(candidate_indices.shape)}."
            )

        batch_indices = torch.arange(
            candidate_features.size(0),
            device=candidate_features.device,
        )
        return candidate_features[batch_indices, candidate_indices]

    def _encode_candidates(self, tokens, device: torch.device, name: str):
        if tokens is None:
            return None, None

        tokens = tokens.to(device=device, non_blocking=True)
        if tokens.ndim != 3:
            raise ValueError(
                f"{name} tokens must have shape [B,C,L], got "
                f"{tuple(tokens.shape)}."
            )

        if hasattr(self.lang_model, "encode_candidates"):
            candidate_global, candidate_tokens = (
                self.lang_model.encode_candidates(tokens)
            )
        else:
            candidate_global, candidate_tokens = self.lang_model(tokens)

        if candidate_global.ndim != 3:
            raise ValueError(
                "LanModel must return candidate features [B,C,D], got "
                f"{tuple(candidate_global.shape)} for {name}."
            )

        return candidate_global, candidate_tokens

    def _encode_answers(self, answer, device: torch.device):
        """
        Existing CRA candidates:
            [Question SEP Answer_i]
        """
        if answer is None:
            raise ValueError("answer tokens must not be None.")
        return self._encode_candidates(
            answer, device=device, name="answer"
        )

    def _encode_answer_only(self, answer_only, device: torch.device):
        """
        V2.1 auxiliary candidates:
            [Answer_i]

        The representation is detached before it supervises Q-Bound evidence.
        """
        if answer_only is None:
            return None, None
        return self._encode_candidates(
            answer_only, device=device, name="answer_only"
        )

    @staticmethod
    def _question_mask_from_padded_ids(question_ids: torch.Tensor):
        """
        The repository collate function pads question ids with literal zeros.
        Mask those padded positions, while keeping the first token valid because
        some tokenizers also use id=0 for a real BOS/<s> token.
        """
        if question_ids.ndim != 2:
            raise ValueError("question_ids must be [B,L].")

        mask = question_ids.ne(0)
        if mask.size(1) > 0:
            mask[:, 0] = True

        if not torch.all(mask.any(dim=-1)):
            raise ValueError(
                "Every question must contain at least one valid token."
            )
        return mask

    def _encode_question(self, question, device: torch.device):
        """
        Encode question-only tokens and return the token mask needed by V2.1.

        Returns
        -------
        question_global : [B,D]
        question_tokens : [B,L,D]
        question_mask   : [B,L] bool
        """
        if question is None:
            return None, None, None

        question = question.to(device=device, non_blocking=True)
        if question.ndim == 3 and question.size(1) == 1:
            question = question.squeeze(1)
        if question.ndim != 2:
            raise ValueError(
                "Question-only tokens must have shape [B,L] or [B,1,L], got "
                f"{tuple(question.shape)}."
            )

        question_mask = self._question_mask_from_padded_ids(question)

        if hasattr(self.lang_model, "encode_question"):
            question_global, question_tokens = (
                self.lang_model.encode_question(question)
            )
        else:
            question_global, question_tokens = self.lang_model(question)

        if question_global.ndim != 2:
            raise ValueError(
                "LanModel must return question features [B,D], got "
                f"{tuple(question_global.shape)}."
            )
        if question_tokens.ndim != 3:
            raise ValueError(
                "LanModel must return token-level question features [B,L,D], "
                f"got {tuple(question_tokens.shape)}."
            )
        if question_tokens.shape[:2] != question_mask.shape:
            raise ValueError(
                "Question token features and question mask do not align."
            )

        return question_global, question_tokens, question_mask

    @staticmethod
    def _move_video_mask(video_mask, device: torch.device):
        if video_mask is None:
            return None
        return video_mask.to(device=device, non_blocking=True)

    @staticmethod
    def _scalar_flag(mapping, name):
        if name not in mapping:
            raise KeyError(f"Missing Q-Bound flag {name!r}.")
        value = mapping[name]
        if not torch.is_tensor(value):
            raise TypeError(f"Q-Bound flag {name!r} must be a tensor.")
        if value.numel() != 1:
            raise ValueError(f"Q-Bound flag {name!r} must be scalar.")
        return float(value.detach().item())

    def _aggregate_prop_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Preserve the repository's prop_num aggregation when applicable.

        Supported shapes:
            [B,C_total]
            [B,S,C_total]
        """
        if logits.ndim not in {2, 3}:
            raise ValueError(
                "Candidate logits must be [B,C] or [B,S,C], got "
                f"{tuple(logits.shape)}."
            )

        prop_num = int(self.cfgs["model"].get("prop_num", 1))
        if prop_num < 1:
            raise ValueError("model.prop_num must be positive.")

        total_candidates = logits.size(-1)
        if total_candidates % prop_num != 0:
            raise ValueError(
                "Candidate dimension is not divisible by prop_num: "
                f"{total_candidates} vs {prop_num}."
            )
        if prop_num == 1:
            return logits

        answer_count = total_candidates // prop_num
        if logits.ndim == 2:
            return logits.reshape(
                logits.size(0), prop_num, answer_count
            ).mean(dim=1)

        return logits.reshape(
            logits.size(0),
            logits.size(1),
            prop_num,
            answer_count,
        ).mean(dim=2)

    def _score_answer_only(
        self,
        video_feature: torch.Tensor,
        answer_only_proj: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score one visual evidence vector against answer-only candidate features.

        Candidate embeddings are expected to be detached by the caller. This
        makes the auxiliary signal train temporal/visual evidence rather than
        directly co-adapting the candidate language representation.
        """
        if video_feature.ndim != 2:
            raise ValueError(
                "video_feature must be [B,D], got "
                f"{tuple(video_feature.shape)}."
            )
        if answer_only_proj.ndim != 3:
            raise ValueError(
                "answer_only_proj must be [B,C,D], got "
                f"{tuple(answer_only_proj.shape)}."
            )
        if (
            video_feature.size(0) != answer_only_proj.size(0)
            or video_feature.size(-1) != answer_only_proj.size(-1)
        ):
            raise ValueError(
                "Visual evidence and answer-only candidates have incompatible "
                "batch/feature dimensions."
            )

        logits = torch.einsum(
            "bd,bcd->bc",
            video_feature,
            answer_only_proj,
        )
        return self._aggregate_prop_logits(logits)

    # ------------------------------------------------------------------
    # Q-Bound V2.1 contract validation
    # ------------------------------------------------------------------
    def _validate_v210_contract(
        self,
        time_param,
        qbound_aux,
        video_proj,
        batch_size,
        time_steps,
    ):
        if not isinstance(time_param, dict):
            raise TypeError(
                "RefineDecoder must return time_param as a dict."
            )
        if not isinstance(qbound_aux, dict):
            raise TypeError(
                "Q-Bound V2.1 must expose "
                "time_param['qbound_aux'] as a dict."
            )

        required_aux = {
            "qbound_version",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_answer_evidence_uses_local_tokens",
            "qbound_answer_evidence_uses_cra_context",
            "functional_verifier_used",
            "boundary_equilibrium_used",
            "teacher_distribution_used",

            "latent_segment_composition_enabled",
            "question_token_interaction_enabled",
            "segment_energy_enabled",
            "fixed_event_partition_enabled",
            "dynamic_event_count_enabled",
            "event_count_cap_enabled",
            "cut_field_enabled",
            "boundary_gated_reasoning_enabled",
            "independent_endpoint_heads_enabled",
            "hypothesis_competition_enabled",
            "teacher_distillation_enabled",
            "answer_features_used_in_core",
            "temporal_gt_used_in_core",

            "content_tokens",
            "primitive_states",
            "query_conditioned_states",
            "node_relevance",
            "adjacent_relation",
            "segment_energy",
            "segment_prob",
            "segment_entropy_ratio",
            "segment_top1_prob",
            "soft_inside_prob",
            "soft_temporal_prob",
            "training_temporal_prob",
            "main_temporal_prob",
            "main_temporal_logits",
            "main_soft_support",

            "selected_span_start_frame",
            "selected_span_end_frame",
            "selected_span_frame_mask",
            "selected_span_frame_weights",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",
            "single_frame_prob_mass",
            "short_span_prob_mass",
            "selected_single_frame_indicator",
            "selected_full_video_indicator",

            "qbound_hard_video_proj",
            "qbound_training_video_proj",
            "qbound_soft_video_proj",
            "qbound_st_hard_feature_max_abs_diff",

            "cra_key",
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
            "main_video_proj",
            "qbound_video_proj",
            "cra_video_proj",
        }
        missing_aux = sorted(required_aux.difference(qbound_aux))
        if missing_aux:
            raise KeyError(
                "Q-Bound V2.1 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_time = {
            "grounding_feature",
            "qbound_grounding_feature",
            "qbound_temporal_main_feature",
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
            "qbound_selected_span_start_frame",
            "qbound_selected_span_end_frame",
            "qbound_selected_span_mask",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",
            "qbound_soft_expected_span_length",
            "qbound_soft_expected_span_fraction",
            "qbound_latent_segment_composition_enabled",
            "qbound_question_token_interaction_enabled",
            "qbound_segment_energy_enabled",
            "qbound_fixed_event_partition_enabled",
            "qbound_dynamic_event_count_enabled",
            "qbound_event_count_cap_enabled",
            "qbound_cut_field_enabled",
            "qbound_boundary_gated_reasoning_enabled",
            "qbound_independent_endpoint_heads_enabled",
            "qbound_hypothesis_competition_enabled",
            "qbound_teacher_distillation_enabled",
            "qbound_answer_features_used_in_core",
            "qbound_temporal_gt_used_in_core",
            "qbound_smoothing_applied",
        }
        missing_time = sorted(required_time.difference(time_param))
        if missing_time:
            raise KeyError(
                "RefineDecoder/Q-Bound V2.1 time_param output is missing: "
                + ", ".join(missing_time)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "CRA/RefineDecoder/QBoundCore version mismatch: "
                f"CRA={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        expected_flags = {
            "qbound_input_is_generic": 1.0,
            "qbound_input_uses_cra_embedding": 0.0,
            "qbound_answer_evidence_uses_local_tokens": 1.0,
            "qbound_answer_evidence_uses_cra_context": 0.0,
            "functional_verifier_used": 0.0,
            "boundary_equilibrium_used": 0.0,
            "teacher_distribution_used": 0.0,

            "latent_segment_composition_enabled": 1.0,
            "question_token_interaction_enabled": 1.0,
            "segment_energy_enabled": 1.0,

            "fixed_event_partition_enabled": 0.0,
            "dynamic_event_count_enabled": 0.0,
            "event_count_cap_enabled": 0.0,
            "cut_field_enabled": 0.0,
            "boundary_gated_reasoning_enabled": 0.0,
            "independent_endpoint_heads_enabled": 0.0,
            "hypothesis_competition_enabled": 0.0,
            "teacher_distillation_enabled": 0.0,
            "answer_features_used_in_core": 0.0,
            "temporal_gt_used_in_core": 0.0,
        }
        for name, expected in expected_flags.items():
            actual = self._scalar_flag(qbound_aux, name)
            if actual != expected:
                raise RuntimeError(
                    f"V2.1 contract violation: {name}={actual}, "
                    f"expected {expected}."
                )

        if qbound_aux["main_video_proj"].shape != video_proj.shape:
            raise ValueError(
                "Q-Bound main_video_proj shape does not match "
                "returned video_proj."
            )

        for name in (
            "main_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
            "selected_span_frame_mask",
        ):
            value = qbound_aux[name]
            if value.shape != (batch_size, time_steps):
                raise ValueError(
                    f"{name} must be [B,T]={batch_size,time_steps}, got "
                    f"{tuple(value.shape)}."
                )

        if qbound_aux["node_relevance"].shape != (
            batch_size,
            time_steps,
        ):
            raise ValueError(
                "node_relevance must be [B,T]."
            )

        expected_edge = (batch_size, max(time_steps - 1, 0))
        if qbound_aux["adjacent_relation"].shape != expected_edge:
            raise ValueError(
                "adjacent_relation must be [B,T-1], got "
                f"{tuple(qbound_aux['adjacent_relation'].shape)}."
            )

        if qbound_aux["segment_prob"].shape != (
            batch_size,
            time_steps,
            time_steps,
        ):
            raise ValueError(
                "segment_prob must be [B,T,T]."
            )

        start = qbound_aux["selected_span_start_frame"]
        end = qbound_aux["selected_span_end_frame"]
        if start.shape != (batch_size,) or end.shape != (batch_size,):
            raise ValueError(
                "selected span endpoints must be [B]."
            )
        if torch.any(start > end):
            raise RuntimeError(
                "Q-Bound V2.1 produced start > end."
            )

        # The straight-through answer feature must have the exact hard forward
        # numeric value.  A large nonzero value means integration is wrong.
        st_diff = qbound_aux[
            "qbound_st_hard_feature_max_abs_diff"
        ]
        if not torch.isfinite(st_diff).all():
            raise RuntimeError(
                "Q-Bound ST/hard feature diagnostic is non-finite."
            )
        if float(st_diff.detach().max().item()) > 1e-5:
            raise RuntimeError(
                "Q-Bound training feature no longer matches hard H* in the "
                "forward pass. This breaks the V2.1 soft/hard contract."
            )

    # ------------------------------------------------------------------
    # Q-Bound V2.1 refine forward
    # ------------------------------------------------------------------
    def refine_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
        question=None,
        qbound_mix=1.0,
        answer_only=None,
    ):
        """
        Refined CRA/Q-Bound V2.1 forward.

        ``answer``
            Existing CRA [Question SEP Answer] candidates.

        ``answer_only``
            New V2.1 answer-only candidates.  They NEVER enter Q-BoundCore.
            When supplied, the wrapper scores hard/soft/training Q-Bound
            evidence against detached answer-only embeddings and surfaces those
            logits for the training pipeline.

        ``qbound_mix``
            Kept only for repository call-signature compatibility. V2.1 does
            not mix CRA and Q-Bound temporal keys.
        """
        del qbound_mix

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(video_mask, device)

        # Existing CRA candidate-QA representation.
        answer_g, _ = self._encode_answers(answer, device)
        batch_size = answer_g.size(0)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            batch_size,
            answer_g.device,
        )

        # V2.1 answer-only representation.  It is detached before use in the
        # temporal-evidence auxiliary branch.
        answer_only_g, answer_only_tokens = self._encode_answer_only(
            answer_only,
            device,
        )
        if (
            answer_only_g is not None
            and answer_only_g.size(0) != batch_size
        ):
            raise ValueError(
                "answer_only and answer batch sizes do not match."
            )

        video_f = self.video_model(video)
        qsn_gt = None

        if mode == "VQ" and answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g, answer_indices
            )
            qsn_gt = qsn_g
        else:
            # Preserve CRA's historical candidate-QA aggregate as its own
            # reference conditioning. Q-Bound receives question-only tokens.
            qsn_g = answer_g.max(dim=1).values
            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(
                    answer_g, answer_indices
                )

        use_qbound = bool(
            self.qbound_enabled and mode == "VQA"
        )

        question_g = None
        question_tokens = None
        question_mask = None

        if use_qbound:
            if question is None:
                raise ValueError(
                    "Q-Bound V2.1 requires question-only token ids. "
                    "Pass question=... through CRA.forward()."
                )

            (
                question_g,
                question_tokens,
                question_mask,
            ) = self._encode_question(question, device)

            if question_g.size(0) != batch_size:
                raise ValueError(
                    "Question and answer batch sizes do not match: "
                    f"{question_g.size(0)} vs {batch_size}."
                )

            if self.qbound_require_token_question:
                if question_tokens is None or question_tokens.ndim != 3:
                    raise RuntimeError(
                        "V2.1 requires token-level question features."
                    )

        # V2.1 uses the token-level question sequence.
        question_for_qbound = (
            question_tokens if use_qbound else None
        )

        video_proj, video_attn, time_param = self.decoder(
            video_f,
            qsn_g,
            video_mask,
            gsub=gsub,
            do=self.cfgs["model"].get("do", False),
            window_sizes=self.cfgs["model"].get(
                "window_sizes", [1, 3, 5]
            ),
            use_qbound=use_qbound,
            question_feature=question_for_qbound,
            qbound_video_tokens=(video if use_qbound else None),
            qbound_mix=0.0,
            question_mask=(question_mask if use_qbound else None),
        )

        answer_proj = answer_g
        qbound_aux = (
            time_param.get("qbound_aux")
            if isinstance(time_param, dict)
            else None
        )

        answer_only_proj_detached = None
        answer_only_logits = {}

        if use_qbound:
            time_steps = video.shape[1]
            self._validate_v210_contract(
                time_param=time_param,
                qbound_aux=qbound_aux,
                video_proj=video_proj,
                batch_size=batch_size,
                time_steps=time_steps,
            )

            # ----------------------------------------------------------
            # Answer-only temporal-evidence supervision interface.
            # ----------------------------------------------------------
            # This branch happens AFTER H(V,Q) has been constructed.  It cannot
            # affect which segment is proposed in the forward computation.
            if answer_only_g is not None:
                answer_only_proj_detached = answer_only_g.detach()

                hard_logits = self._score_answer_only(
                    qbound_aux["qbound_hard_video_proj"],
                    answer_only_proj_detached,
                )
                training_logits = self._score_answer_only(
                    qbound_aux["qbound_training_video_proj"],
                    answer_only_proj_detached,
                )
                soft_logits = self._score_answer_only(
                    qbound_aux["qbound_soft_video_proj"],
                    answer_only_proj_detached,
                )

                # Training feature has hard forward numerics. These logits
                # should therefore also agree with hard logits numerically.
                train_hard_logit_diff = (
                    training_logits - hard_logits
                ).abs().amax(dim=-1)

                answer_only_logits = {
                    "qbound_answer_only_hard_logits": hard_logits,
                    "qbound_answer_only_training_logits": training_logits,
                    "qbound_answer_only_soft_logits": soft_logits,
                    "qbound_answer_only_train_hard_logit_max_abs_diff":
                        train_hard_logit_diff,
                    "qbound_answer_only_supervision_available": torch.ones(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
                    "qbound_answer_only_embeddings_stop_gradient": torch.ones(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
                    "qbound_answer_only_affects_segment_forward": torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
                }

                if float(
                    train_hard_logit_diff.detach().max().item()
                ) > 1e-5:
                    raise RuntimeError(
                        "Answer-only training logits no longer match hard "
                        "forward logits. The V2.1 ST contract is broken."
                    )
            else:
                answer_only_logits = {
                    "qbound_answer_only_supervision_available": torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
                    "qbound_answer_only_embeddings_stop_gradient": torch.ones(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
                    "qbound_answer_only_affects_segment_forward": torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
                }

            qbound_aux.update(answer_only_logits)
            for key, value in answer_only_logits.items():
                time_param[key] = value

            # Paper-level invariants.
            invariants = {
                "grounding_requires_answer_at_inference": torch.zeros(
                    (),
                    dtype=video_proj.dtype,
                    device=video_proj.device,
                ),
                "temporal_gt_used_in_training": torch.zeros(
                    (),
                    dtype=video_proj.dtype,
                    device=video_proj.device,
                ),
                "answer_only_used_inside_qbound_core": torch.zeros(
                    (),
                    dtype=video_proj.dtype,
                    device=video_proj.device,
                ),
            }
            qbound_aux.update(invariants)
            for key, value in invariants.items():
                time_param[f"qbound_{key}"] = value

        # --------------------------------------------------------------
        # Shared outputs consumed by CRA pipeline/evaluation.
        # --------------------------------------------------------------
        outs = {
            "fatt": video_attn,
            "time_param": time_param,
            "align_gt": qsn_gt,
            "qa_condition": qsn_g,
            "question_condition": question_g,
            "question_tokens": question_tokens,
            "question_mask": question_mask,
            "question_only_conditioning": torch.as_tensor(
                question_g is not None,
                device=video_proj.device,
                dtype=video_proj.dtype,
            ),
            "qbound_align_question": question_g,
            "qbound_align_qa": qsn_gt,
            "qbound_aux": qbound_aux,
            "qbound_active": use_qbound,
            "qbound_version": self.qbound_version,
            "gpeb_active": use_qbound,

            # Keep candidate representations explicit.
            "answer_proj": answer_proj,
            "answer_only_proj": answer_only_proj_detached,
            "answer_only_tokens": answer_only_tokens,
        }

        if isinstance(time_param, dict):
            for key, value in time_param.items():
                if key.startswith("qbound_"):
                    outs[key] = value

        if isinstance(qbound_aux, dict):
            direct_aux_names = (
                # Local/query-conditioned representation.
                "generic_input_adapter_enabled",
                "generic_input_token_cosine",
                "generic_input_delta_ratio",
                "primitive_states",
                "delta_magnitude",
                "accel_magnitude",
                "relative_position",
                "query_conditioned_states",
                "question_interaction_delta_ratio",

                # Query-specific primitive / relation evidence.
                "node_relevance_raw",
                "node_relevance",
                "node_relevance_centered",
                "adjacent_relation_raw",
                "adjacent_relation",
                "adjacent_relation_centered",

                # Segment energy / posterior.
                "segment_node_support",
                "segment_relation_support",
                "segment_boundary_separation",
                "segment_energy",
                "segment_logits",
                "segment_valid_mask",
                "segment_prob",
                "segment_entropy",
                "segment_entropy_ratio",
                "segment_top1_prob",

                # Soft/hard grounding.
                "soft_inside_prob",
                "soft_temporal_prob",
                "training_temporal_prob",
                "main_temporal_prob",
                "main_temporal_logits",
                "main_soft_support",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_mask",
                "selected_span_frame_weights",
                "selected_span_length",
                "selected_span_fraction",
                "selected_segment_energy",
                "soft_expected_span_length",
                "soft_expected_span_fraction",

                # Explicit shortcut diagnostics.
                "single_frame_prob_mass",
                "short_span_prob_mass",
                "selected_single_frame_indicator",
                "selected_full_video_indicator",

                # Architecture flags.
                "latent_segment_composition_enabled",
                "question_token_interaction_enabled",
                "segment_energy_enabled",
                "fixed_event_partition_enabled",
                "dynamic_event_count_enabled",
                "event_count_cap_enabled",
                "cut_field_enabled",
                "boundary_gated_reasoning_enabled",
                "independent_endpoint_heads_enabled",
                "hypothesis_competition_enabled",
                "teacher_distillation_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",

                # Local-evidence provenance / features.
                "qbound_input_is_generic",
                "qbound_input_uses_cra_embedding",
                "qbound_answer_evidence_uses_local_tokens",
                "qbound_answer_evidence_uses_cra_context",
                "functional_verifier_used",
                "boundary_equilibrium_used",
                "teacher_distribution_used",
                "qbound_hard_video_proj",
                "qbound_training_video_proj",
                "qbound_soft_video_proj",
                "qbound_st_hard_feature_max_abs_diff",

                # Answer-only auxiliary interface.
                "qbound_answer_only_hard_logits",
                "qbound_answer_only_training_logits",
                "qbound_answer_only_soft_logits",
                "qbound_answer_only_train_hard_logit_max_abs_diff",
                "qbound_answer_only_supervision_available",
                "qbound_answer_only_embeddings_stop_gradient",
                "qbound_answer_only_affects_segment_forward",

                # Paper-level invariants.
                "grounding_requires_answer_at_inference",
                "temporal_gt_used_in_training",
                "answer_only_used_inside_qbound_core",
            )
            for name in direct_aux_names:
                if name in qbound_aux:
                    outs[name] = qbound_aux[name]

            # Main answer/key aliases for repository evaluation.
            outs["qbound_video_proj"] = qbound_aux.get(
                "qbound_video_proj"
            )
            outs["qbound_hard_video_proj"] = qbound_aux.get(
                "qbound_hard_video_proj"
            )
            outs["qbound_soft_video_proj"] = qbound_aux.get(
                "qbound_soft_video_proj"
            )
            outs["qbound_training_video_proj"] = qbound_aux.get(
                "qbound_training_video_proj"
            )

            outs["qbound_calibrated_video_proj"] = qbound_aux.get(
                "qbound_video_proj"
            )
            outs["qbound_refined_video_proj"] = qbound_aux.get(
                "qbound_video_proj"
            )

            outs["qbound_key"] = qbound_aux.get("qbound_key")
            outs["qbound_training_key"] = qbound_aux.get(
                "qbound_training_key"
            )
            outs["qbound_soft_key"] = qbound_aux.get(
                "qbound_soft_key"
            )
            outs["qbound_calibrated_key"] = qbound_aux.get(
                "qbound_key"
            )
            outs["qbound_refined_key"] = qbound_aux.get(
                "qbound_key"
            )

            outs["cra_video_proj"] = qbound_aux.get(
                "cra_video_proj"
            )
            outs["cra_key"] = qbound_aux.get("cra_key")

        return video_proj, answer_proj, outs

    # ------------------------------------------------------------------
    # Original CRA baseline path
    # ------------------------------------------------------------------
    def baseline_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
        question=None,
        qbound_mix=1.0,
        answer_only=None,
    ):
        # answer_only is accepted only to keep one repository-level call
        # signature. The baseline CRA path does not use V2.1 supervision.
        del gsub, question, qbound_mix, answer_only

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask, device
        )

        answer_g, answer_tokens = self._encode_answers(
            answer, device
        )
        answer_indices = self._prepare_answer_indices(
            answer_id,
            answer_g.size(0),
            answer_g.device,
        )

        video_f = self.video_model(video)
        if answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g, answer_indices
            )
        else:
            qsn_g = answer_g.max(dim=1).values

        video_proj, video_attn = self.decoder(
            video_f,
            answer_tokens,
            video_mask,
        )
        outs = {
            "fatt": video_attn,
            "qa_condition": qsn_g,
        }
        return video_proj, answer_g, outs

    # ------------------------------------------------------------------
    # Existing weak alignment objective
    # ------------------------------------------------------------------
    def align_loss(
        self,
        qsn,
        grounding_feature,
        neg_sample=None,
    ):
        def _sample_negatives(x_pos, k):
            batch_size, feature_dim = x_pos.size()
            if batch_size < 2:
                raise ValueError(
                    "InfoNCE negative sampling requires batch_size >= 2."
                )

            x_neg = torch.zeros(
                batch_size,
                k,
                feature_dim,
                device=x_pos.device,
                dtype=x_pos.dtype,
            )
            for index in range(batch_size):
                candidate_indices = torch.cat(
                    [
                        torch.arange(
                            0, index, device=x_pos.device
                        ),
                        torch.arange(
                            index + 1,
                            batch_size,
                            device=x_pos.device,
                        ),
                    ]
                )
                sampled_positions = torch.randint(
                    low=0,
                    high=candidate_indices.numel(),
                    size=(k,),
                    device=x_pos.device,
                )
                x_neg[index] = x_pos[
                    candidate_indices[sampled_positions]
                ]
            return x_neg

        if neg_sample is None:
            negative_count = max(
                1,
                self.cfgs["dataset"]["batch_size"] // 2,
            )
            x_neg = _sample_negatives(
                grounding_feature,
                negative_count,
            )
        else:
            x_neg = neg_sample

        return self._info_nce_loss(
            qsn,
            grounding_feature,
            x_neg,
        )

    @staticmethod
    def _info_nce_loss(
        x,
        x_pos,
        x_neg,
        temperature=0.07,
    ):
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_neg = F.normalize(x_neg, dim=-1)

        pos_sim = torch.sum(
            x * x_pos,
            dim=-1,
            keepdim=True,
        ) / temperature
        neg_sim = torch.bmm(
            x.unsqueeze(1),
            x_neg.transpose(1, 2),
        ).squeeze(1) / temperature

        logits = torch.cat(
            [pos_sim, neg_sim],
            dim=-1,
        )
        labels = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device,
        )
        return F.cross_entropy(
            logits,
            labels,
        )
