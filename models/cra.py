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
    CRA wrapper used as the downstream VideoQA testbed for Q-Bound V2.0.0.

    V2.0.0 is a representation-level reset from the V1.x family.  Q-Bound no
    longer consumes a fixed event partition, enumerates 528 temporal spans, or
    learns from a CRA teacher distribution.  Its controlled path is:

        generic video + question-only semantics
            -> temporal dynamics
            -> video-only transition prior
            -> bounded query-adaptive functional cut field
            -> boundary-gated temporal reasoning
            -> direct start/end grounding
            -> exact hard span H*

    CRA has two deliberately separated roles.

    1) Downstream factual consumer / benchmark testbed
       ------------------------------------------------
       Q-Bound's selected temporal evidence pools CRA ``x_bar`` and is scored
       against the ordinary candidate answer embeddings.  This preserves a
       comparable NExT-GQA / CRA answer-space testbed without feeding CRA
       temporal representations back into Q-Bound.

    2) Training-only stop-gradient functional verifier
       ------------------------------------------------
       Candidate-answer semantics are permitted to measure *local functional
       consequences* after Q-Bound has already produced its V+Q grounding:

         * KEEP vs MERGE for each adjacent temporal pair;
         * H* vs trim-left/right and expand-left/right views.

       These verifier logits are deterministic and detached.  They are not a
       teacher distribution and they never select the grounding in the forward
       path.  The training pipeline will convert GT-answer margin differences
       into detached functional-composition and boundary-equilibrium targets.

    Controlled V2.0.0 invariants
    ----------------------------
      * Q-Bound input is the explicit generic pre-CRA visual sequence.
      * Q-Bound core is answer-free and question-only at inference.
      * There is no fixed event partition and no semantic event-count cap.
      * Event count emerges from the query-specific soft cut field.
      * Composition is causal: the cut field gates temporal information flow.
      * There is no exhaustive temporal-hypothesis bank or listwise teacher.
      * Hard H* is the official inference/evaluation grounding.
      * A straight-through training key keeps hard forward numerics while
        allowing weak QA loss to reach the start/end heads.
      * CRA temporal grounding is reference/diagnostic only when Q-Bound is on.
      * VisualIntervention is OFF for the controlled V2.0.0 experiment.
      * Temporal GT / IoU / IoP are never consumed by this wrapper.
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
                    model_cfg.get("gpeb_version", "2.0.0"),
                ),
            )
        )

        if self.qbound_enabled and self.qbound_version != "2.0.0":
            raise ValueError(
                "Q-Bound is enabled, but this CRA wrapper expects "
                f"model.qbound.version='2.0.0'; got {self.qbound_version!r}."
            )

        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get("qbound_require_question_only", True),
            )
        )
        if self.qbound_enabled and not self.qbound_require_question_only:
            raise ValueError(
                "Controlled Q-Bound V2.0.0 requires "
                "model.qbound.require_question_only=true."
            )

        if self.qbound_enabled and bool(model_cfg.get("do", False)):
            raise ValueError(
                "Controlled Q-Bound V2.0.0 requires model.do=false. "
                "VisualIntervention would confound the functional-composition "
                "experiment."
            )

        # Compatibility aliases used elsewhere in the repository.
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

        # Kept only for repository/checkpoint compatibility.  The controlled
        # V2.0.0 Q-Bound run requires model.do=false.
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
                decoder_version = str(getattr(self.decoder, "qbound_version", ""))
                if decoder_version != self.qbound_version:
                    raise RuntimeError(
                        "CRA/RefineDecoder Q-Bound version mismatch: "
                        f"CRA={self.qbound_version!r}, "
                        f"decoder={decoder_version!r}."
                    )
                for required_method in (
                    "score_functional_composition_pairs",
                    "score_boundary_equilibrium_views",
                ):
                    if not hasattr(self.decoder, required_method):
                        raise RuntimeError(
                            "Q-Bound V2.0.0 RefineDecoder must implement "
                            f"{required_method}()."
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
    def _prepare_answer_indices(answer_id, batch_size: int, device: torch.device):
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
                f"Expected {batch_size} values, got {answer_indices.numel()}."
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

    def _encode_answers(self, answer, device: torch.device):
        """Encode candidate question-answer sequences: [B,C,L] -> [B,C,D]."""
        if answer is None:
            raise ValueError("answer tokens must not be None.")

        answer = answer.to(device=device, non_blocking=True)
        if answer.ndim != 3:
            raise ValueError(
                "Candidate answer tokens must have shape [B,C,L], got "
                f"{tuple(answer.shape)}."
            )

        if hasattr(self.lang_model, "encode_candidates"):
            answer_global, answer_tokens = self.lang_model.encode_candidates(answer)
        else:
            answer_global, answer_tokens = self.lang_model(answer)

        if answer_global.ndim != 3:
            raise ValueError(
                "LanModel must return candidate features [B,C,D], got "
                f"{tuple(answer_global.shape)}."
            )
        return answer_global, answer_tokens

    def _encode_question(self, question, device: torch.device):
        """
        Encode question-only tokens.

        V2.0.0 deliberately feeds the global question representation to the
        first functional-composition implementation, keeping language changes
        controlled.  Token-level outputs are still surfaced for diagnostics and
        future ablations.
        """
        if question is None:
            return None, None

        question = question.to(device=device, non_blocking=True)
        if question.ndim == 3 and question.size(1) == 1:
            question = question.squeeze(1)
        if question.ndim != 2:
            raise ValueError(
                "Question-only tokens must have shape [B,L] or [B,1,L], got "
                f"{tuple(question.shape)}."
            )

        if hasattr(self.lang_model, "encode_question"):
            question_global, question_tokens = self.lang_model.encode_question(question)
        else:
            question_global, question_tokens = self.lang_model(question)

        if question_global.ndim != 2:
            raise ValueError(
                "LanModel must return question features [B,D], got "
                f"{tuple(question_global.shape)}."
            )
        return question_global, question_tokens

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
        Apply the repository's ``prop_num`` candidate aggregation.

        Supported shapes are [B,C_total] and [B,S,C_total].  V2 functional
        verifier logits therefore live in exactly the same answer space as the
        factual CRA scoring rule, without defining any teacher distribution.
        """
        if logits.ndim not in {2, 3}:
            raise ValueError(
                "Verifier logits must be [B,C] or [B,S,C], got "
                f"{tuple(logits.shape)}."
            )

        prop_num = int(self.cfgs["model"].get("prop_num", 1))
        if prop_num < 1:
            raise ValueError("model.prop_num must be positive.")

        total_candidates = logits.size(-1)
        if total_candidates % prop_num != 0:
            raise ValueError(
                "Verifier candidate dimension is not divisible by prop_num: "
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
            logits.size(0), logits.size(1), prop_num, answer_count
        ).mean(dim=2)

    # ------------------------------------------------------------------
    # V2.0.0 contract validation
    # ------------------------------------------------------------------
    def _validate_v200_contract(
        self,
        time_param,
        qbound_aux,
        video_proj,
        batch_size,
        time_steps,
    ):
        if not isinstance(time_param, dict):
            raise TypeError("RefineDecoder must return time_param as a dict.")
        if not isinstance(qbound_aux, dict):
            raise TypeError(
                "Q-Bound V2.0.0 must expose time_param['qbound_aux'] as a dict."
            )

        required_aux = {
            "qbound_version",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "generic_input_adapter_enabled",
            "functional_composition_enabled",
            "dynamic_event_count_enabled",
            "fixed_event_partition_enabled",
            "event_count_cap_enabled",
            "hypothesis_competition_enabled",
            "teacher_distillation_enabled",
            "answer_features_used_in_core",
            "boundary_equilibrium_interface_enabled",
            "functional_answer_verifier_ready",
            "boundary_equilibrium_verifier_ready",
            "functional_verifier_consumer_stop_gradient",
            "teacher_distribution_used",
            "video_boundary_prior_prob",
            "functional_cut_prob",
            "functional_merge_gate",
            "hard_event_count",
            "soft_effective_event_count",
            "one_event_indicator",
            "all_split_indicator",
            "functional_pair_valid_mask",
            "start_prob",
            "end_prob",
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
            "trim_left_frame_weights",
            "trim_right_frame_weights",
            "expand_left_frame_weights",
            "expand_right_frame_weights",
            "trim_left_valid_mask",
            "trim_right_valid_mask",
            "expand_left_valid_mask",
            "expand_right_valid_mask",
            "cra_key",
            "qbound_key",
            "qbound_training_key",
            "final_key",
            "main_video_proj",
            "qbound_video_proj",
            "cra_video_proj",
        }
        missing_aux = sorted(required_aux.difference(qbound_aux))
        if missing_aux:
            raise KeyError(
                "Q-Bound V2.0.0 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_time = {
            "grounding_feature",
            "qbound_grounding_feature",
            "qbound_temporal_main_feature",
            "qbound_key",
            "qbound_training_key",
            "qbound_selected_span_start_frame",
            "qbound_selected_span_end_frame",
            "qbound_selected_span_mask",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",
            "qbound_functional_composition_enabled",
            "qbound_dynamic_event_count_enabled",
            "qbound_fixed_event_partition_enabled",
            "qbound_event_count_cap_enabled",
            "qbound_hypothesis_competition_enabled",
            "qbound_teacher_distillation_enabled",
            "qbound_answer_features_used_in_core",
            "qbound_boundary_equilibrium_interface_enabled",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_smoothing_applied",
        }
        missing_time = sorted(required_time.difference(time_param))
        if missing_time:
            raise KeyError(
                "RefineDecoder/Q-Bound V2.0.0 time_param output is missing: "
                + ", ".join(missing_time)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "CRA/RefineDecoder/QBoundCore version mismatch: "
                f"CRA={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        # Architectural invariants.
        expected_flags = {
            "qbound_input_is_generic": 1.0,
            "qbound_input_uses_cra_embedding": 0.0,
            "functional_composition_enabled": 1.0,
            "dynamic_event_count_enabled": 1.0,
            "fixed_event_partition_enabled": 0.0,
            "event_count_cap_enabled": 0.0,
            "hypothesis_competition_enabled": 0.0,
            "teacher_distillation_enabled": 0.0,
            "answer_features_used_in_core": 0.0,
            "boundary_equilibrium_interface_enabled": 1.0,
            "functional_answer_verifier_ready": 1.0,
            "boundary_equilibrium_verifier_ready": 1.0,
            "functional_verifier_consumer_stop_gradient": 1.0,
            "teacher_distribution_used": 0.0,
        }
        for name, expected in expected_flags.items():
            actual = self._scalar_flag(qbound_aux, name)
            if actual != expected:
                raise RuntimeError(
                    f"V2.0.0 contract violation: {name}={actual}, "
                    f"expected {expected}."
                )

        if qbound_aux["main_video_proj"].shape != video_proj.shape:
            raise ValueError(
                "Q-Bound main_video_proj shape does not match returned video_proj."
            )

        for name in (
            "main_temporal_prob",
            "training_temporal_prob",
            "start_prob",
            "end_prob",
            "selected_span_frame_mask",
        ):
            value = qbound_aux[name]
            if value.shape != (batch_size, time_steps):
                raise ValueError(
                    f"{name} must be [B,T]={batch_size,time_steps}, got "
                    f"{tuple(value.shape)}."
                )

        cut_prob = qbound_aux["functional_cut_prob"]
        if cut_prob.shape != (batch_size, time_steps - 1):
            raise ValueError(
                "functional_cut_prob must be [B,T-1], got "
                f"{tuple(cut_prob.shape)}."
            )
        if qbound_aux["functional_pair_valid_mask"].shape != cut_prob.shape:
            raise ValueError(
                "functional_pair_valid_mask must match functional_cut_prob."
            )

        start = qbound_aux["selected_span_start_frame"]
        end = qbound_aux["selected_span_end_frame"]
        if start.shape != (batch_size,) or end.shape != (batch_size,):
            raise ValueError("selected span endpoints must be [B].")
        if torch.any(start > end):
            raise RuntimeError("Q-Bound V2.0.0 produced start > end.")

    # ------------------------------------------------------------------
    # Q-Bound V2.0.0 refine forward
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
    ):
        """
        Refined CRA/Q-Bound V2.0.0 forward.

        ``qbound_mix`` remains only for repository call-signature compatibility.
        V2.0.0 does not mix CRA and Q-Bound temporal keys.
        """
        del qbound_mix

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(video_mask, device)

        answer_g, _ = self._encode_answers(answer, device)
        batch_size = answer_g.size(0)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            batch_size,
            answer_g.device,
        )

        video_f = self.video_model(video)
        qsn_gt = None

        if mode == "VQ" and answer_indices is not None:
            qsn_g = self._select_candidate_features(answer_g, answer_indices)
            qsn_gt = qsn_g
        else:
            # CRA's own reference path keeps its historical candidate-QA
            # aggregate.  Q-Bound receives a separately encoded question-only
            # feature below.
            qsn_g = answer_g.max(dim=1).values
            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(answer_g, answer_indices)

        use_qbound = bool(self.qbound_enabled and mode == "VQA")

        question_g = None
        question_tokens = None
        if use_qbound:
            if question is None:
                raise ValueError(
                    "Q-Bound V2.0.0 requires question-only token ids. "
                    "Pass question=... through CRA.forward()."
                )
            question_g, question_tokens = self._encode_question(question, device)
            if question_g.size(0) != batch_size:
                raise ValueError(
                    "Question and answer batch sizes do not match: "
                    f"{question_g.size(0)} vs {batch_size}."
                )

        # V2.0.0 deliberately uses the global question state in the first
        # controlled implementation.  Token-level features are exposed in outs
        # but do not silently change the language model compared with V1.x.
        question_for_qbound = question_g

        video_proj, video_attn, time_param = self.decoder(
            video_f,
            qsn_g,
            video_mask,
            gsub=gsub,
            do=self.cfgs["model"].get("do", False),
            window_sizes=self.cfgs["model"].get("window_sizes", [1, 3, 5]),
            use_qbound=use_qbound,
            question_feature=question_for_qbound,
            qbound_video_tokens=(video if use_qbound else None),
            qbound_mix=0.0,
        )

        answer_proj = answer_g
        qbound_aux = (
            time_param.get("qbound_aux")
            if isinstance(time_param, dict)
            else None
        )

        if use_qbound:
            time_steps = video.shape[1]
            self._validate_v200_contract(
                time_param=time_param,
                qbound_aux=qbound_aux,
                video_proj=video_proj,
                batch_size=batch_size,
                time_steps=time_steps,
            )

            # ----------------------------------------------------------
            # Local functional KEEP-vs-MERGE verifier.
            # ----------------------------------------------------------
            # This branch is fully stop-gradient and runs only after the V+Q
            # grounding has already been produced.  It therefore cannot select
            # or alter Q-Bound's forward grounding.
            with torch.no_grad():
                functional = self.decoder.score_functional_composition_pairs(
                    answer_proj=answer_proj,
                    qbound_aux=qbound_aux,
                )
                equilibrium = self.decoder.score_boundary_equilibrium_views(
                    answer_proj=answer_proj,
                    qbound_aux=qbound_aux,
                )

            required_functional = {
                "functional_keep_answer_logits",
                "functional_merge_answer_logits",
                "functional_left_answer_logits",
                "functional_right_answer_logits",
                "functional_full_video_answer_logits",
                "functional_pair_valid_mask",
                "functional_verifier_stop_gradient",
                "teacher_distribution_used",
            }
            missing = sorted(required_functional.difference(functional))
            if missing:
                raise KeyError(
                    "V2.0.0 functional verifier output is missing: "
                    + ", ".join(missing)
                )

            required_equilibrium = {
                "equilibrium_selected_answer_logits",
                "equilibrium_trim_left_answer_logits",
                "equilibrium_trim_right_answer_logits",
                "equilibrium_expand_left_answer_logits",
                "equilibrium_expand_right_answer_logits",
                "equilibrium_trim_left_valid_mask",
                "equilibrium_trim_right_valid_mask",
                "equilibrium_expand_left_valid_mask",
                "equilibrium_expand_right_valid_mask",
                "equilibrium_verifier_stop_gradient",
                "teacher_distribution_used",
            }
            missing = sorted(required_equilibrium.difference(equilibrium))
            if missing:
                raise KeyError(
                    "V2.0.0 boundary-equilibrium verifier output is missing: "
                    + ", ".join(missing)
                )

            # Convert verifier outputs to the same grouped answer space used by
            # the factual VQA scorer.
            for key in (
                "functional_keep_answer_logits",
                "functional_merge_answer_logits",
                "functional_left_answer_logits",
                "functional_right_answer_logits",
                "functional_full_video_answer_logits",
            ):
                functional[key] = self._aggregate_prop_logits(functional[key])

            for key in (
                "equilibrium_selected_answer_logits",
                "equilibrium_trim_left_answer_logits",
                "equilibrium_trim_right_answer_logits",
                "equilibrium_expand_left_answer_logits",
                "equilibrium_expand_right_answer_logits",
            ):
                equilibrium[key] = self._aggregate_prop_logits(equilibrium[key])

            prop_num = int(self.cfgs["model"].get("prop_num", 1))
            if answer_proj.size(1) % prop_num != 0:
                raise ValueError(
                    "Candidate answer count is not divisible by prop_num."
                )
            num_candidates = answer_proj.size(1) // prop_num

            expected_pair_shape = (batch_size, time_steps - 1, num_candidates)
            for key in (
                "functional_keep_answer_logits",
                "functional_merge_answer_logits",
                "functional_left_answer_logits",
                "functional_right_answer_logits",
            ):
                if functional[key].shape != expected_pair_shape:
                    raise ValueError(
                        f"{key} must be {expected_pair_shape}, got "
                        f"{tuple(functional[key].shape)}."
                    )

            expected_view_shape = (batch_size, num_candidates)
            if functional["functional_full_video_answer_logits"].shape != expected_view_shape:
                raise ValueError(
                    "functional_full_video_answer_logits must be "
                    f"{expected_view_shape}, got "
                    f"{tuple(functional['functional_full_video_answer_logits'].shape)}."
                )
            for key in (
                "equilibrium_selected_answer_logits",
                "equilibrium_trim_left_answer_logits",
                "equilibrium_trim_right_answer_logits",
                "equilibrium_expand_left_answer_logits",
                "equilibrium_expand_right_answer_logits",
            ):
                if equilibrium[key].shape != expected_view_shape:
                    raise ValueError(
                        f"{key} must be {expected_view_shape}, got "
                        f"{tuple(equilibrium[key].shape)}."
                    )

            if self._scalar_flag(functional, "functional_verifier_stop_gradient") != 1.0:
                raise RuntimeError("Functional verifier unexpectedly carries gradients.")
            if self._scalar_flag(equilibrium, "equilibrium_verifier_stop_gradient") != 1.0:
                raise RuntimeError("Boundary-equilibrium verifier unexpectedly carries gradients.")
            if self._scalar_flag(functional, "teacher_distribution_used") != 0.0:
                raise RuntimeError("V2.0.0 must not restore a teacher distribution.")
            if self._scalar_flag(equilibrium, "teacher_distribution_used") != 0.0:
                raise RuntimeError("V2.0.0 must not restore a teacher distribution.")

            # Store verifier outputs canonically in qbound_aux and time_param.
            verifier_mapping = {
                **functional,
                **equilibrium,
                "functional_verifier_uses_answer_information": torch.ones(
                    (), dtype=video_proj.dtype, device=video_proj.device
                ),
                "functional_verifier_affects_grounding_forward": torch.zeros(
                    (), dtype=video_proj.dtype, device=video_proj.device
                ),
                "grounding_requires_answer_at_inference": torch.zeros(
                    (), dtype=video_proj.dtype, device=video_proj.device
                ),
                "temporal_gt_used_in_training": torch.zeros(
                    (), dtype=video_proj.dtype, device=video_proj.device
                ),
            }
            # Both verifier dicts contain the same zero-valued
            # teacher_distribution_used key; force one canonical tensor.
            verifier_mapping["teacher_distribution_used"] = torch.zeros(
                (), dtype=video_proj.dtype, device=video_proj.device
            )

            qbound_aux.update(verifier_mapping)
            for key, value in verifier_mapping.items():
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
        }

        if isinstance(time_param, dict):
            for key, value in time_param.items():
                if key.startswith("qbound_"):
                    outs[key] = value

        if isinstance(qbound_aux, dict):
            # Surface the V2 interfaces directly for the training pipeline.  The
            # detailed canonical object remains qbound_aux.
            direct_aux_names = (
                # Composition / anti-collapse diagnostics.
                "video_boundary_prior_logits",
                "video_boundary_prior_prob",
                "question_cut_residual_raw",
                "question_cut_residual",
                "functional_cut_logits",
                "functional_cut_prob",
                "functional_merge_gate",
                "hard_cut",
                "hard_event_index",
                "hard_event_count",
                "soft_effective_event_count",
                "cut_mean",
                "cut_std",
                "prior_cut_mean",
                "question_residual_abs_mean",
                "one_event_indicator",
                "all_split_indicator",
                "functional_pair_valid_mask",
                # Direct grounding.
                "start_logits",
                "end_logits",
                "start_prob",
                "end_prob",
                "start_entropy",
                "end_entropy",
                "soft_span_support",
                "soft_temporal_prob",
                "training_temporal_prob",
                "frame_evidence_logits",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_mask",
                "selected_span_frame_weights",
                "selected_span_length",
                "selected_span_fraction",
                "selected_pair_score",
                "main_temporal_prob",
                "main_temporal_logits",
                "main_soft_support",
                # Equilibrium views.
                "trim_left_frame_weights",
                "trim_right_frame_weights",
                "expand_left_frame_weights",
                "expand_right_frame_weights",
                "trim_left_valid_mask",
                "trim_right_valid_mask",
                "expand_left_valid_mask",
                "expand_right_valid_mask",
                # Architecture flags.
                "functional_composition_enabled",
                "dynamic_event_count_enabled",
                "fixed_event_partition_enabled",
                "event_count_cap_enabled",
                "hypothesis_competition_enabled",
                "teacher_distillation_enabled",
                "answer_features_used_in_core",
                "boundary_equilibrium_interface_enabled",
                # Functional verifier logits.
                "functional_keep_answer_logits",
                "functional_merge_answer_logits",
                "functional_left_answer_logits",
                "functional_right_answer_logits",
                "functional_full_video_answer_logits",
                "functional_verifier_stop_gradient",
                # Boundary-equilibrium verifier logits.
                "equilibrium_selected_answer_logits",
                "equilibrium_trim_left_answer_logits",
                "equilibrium_trim_right_answer_logits",
                "equilibrium_expand_left_answer_logits",
                "equilibrium_expand_right_answer_logits",
                "equilibrium_trim_left_valid_mask",
                "equilibrium_trim_right_valid_mask",
                "equilibrium_expand_left_valid_mask",
                "equilibrium_expand_right_valid_mask",
                "equilibrium_verifier_stop_gradient",
                # Paper-level invariants.
                "teacher_distribution_used",
                "functional_verifier_uses_answer_information",
                "functional_verifier_affects_grounding_forward",
                "grounding_requires_answer_at_inference",
                "temporal_gt_used_in_training",
            )
            for name in direct_aux_names:
                if name in qbound_aux:
                    outs[f"qbound_{name}"] = qbound_aux[name]

            # Main answer/key aliases for generic repository evaluation.
            outs["qbound_video_proj"] = qbound_aux.get("qbound_video_proj")
            outs["qbound_calibrated_video_proj"] = qbound_aux.get(
                "qbound_video_proj"
            )
            outs["qbound_refined_video_proj"] = qbound_aux.get(
                "qbound_video_proj"
            )
            outs["qbound_key"] = qbound_aux.get("qbound_key")
            outs["qbound_training_key"] = qbound_aux.get("qbound_training_key")
            outs["qbound_calibrated_key"] = qbound_aux.get("qbound_key")
            outs["qbound_refined_key"] = qbound_aux.get("qbound_key")
            outs["cra_video_proj"] = qbound_aux.get("cra_video_proj")
            outs["cra_key"] = qbound_aux.get("cra_key")

            # Candidate-answer features are exposed only for functional-verifier
            # diagnostics/bookkeeping.  They never enter QBoundCore.
            outs["qbound_functional_verifier_answer_proj"] = answer_proj.detach()

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
    ):
        del gsub, question, qbound_mix

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(video_mask, device)

        answer_g, answer_tokens = self._encode_answers(answer, device)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            answer_g.size(0),
            answer_g.device,
        )

        video_f = self.video_model(video)
        if answer_indices is not None:
            qsn_g = self._select_candidate_features(answer_g, answer_indices)
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
    def align_loss(self, qsn, grounding_feature, neg_sample=None):
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
                        torch.arange(0, index, device=x_pos.device),
                        torch.arange(index + 1, batch_size, device=x_pos.device),
                    ]
                )
                sampled_positions = torch.randint(
                    low=0,
                    high=candidate_indices.numel(),
                    size=(k,),
                    device=x_pos.device,
                )
                x_neg[index] = x_pos[candidate_indices[sampled_positions]]
            return x_neg

        if neg_sample is None:
            negative_count = max(1, self.cfgs["dataset"]["batch_size"] // 2)
            x_neg = _sample_negatives(grounding_feature, negative_count)
        else:
            x_neg = neg_sample

        return self._info_nce_loss(qsn, grounding_feature, x_neg)

    @staticmethod
    def _info_nce_loss(x, x_pos, x_neg, temperature=0.07):
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_neg = F.normalize(x_neg, dim=-1)

        pos_sim = torch.sum(x * x_pos, dim=-1, keepdim=True) / temperature
        neg_sim = torch.bmm(
            x.unsqueeze(1),
            x_neg.transpose(1, 2),
        ).squeeze(1) / temperature

        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device,
        )
        return F.cross_entropy(logits, labels)
