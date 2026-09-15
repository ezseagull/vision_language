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
    CRA wrapper used as the downstream VideoQA testbed for Q-Bound V1.12.

    V1.12 is intentionally different from the V1.11 family.  Q-Bound no
    longer exposes Composer -> BAPGR -> PPBR -> PSRC -> donor-CF as its main
    route.  The temporal method is now:

        generic video + question-only semantics
                    -> dynamic DECOMPOSE
                    -> ordered event RELATE
                    -> frame-exact temporal hypothesis bank
                    -> intra-video hypothesis competition
                    -> hard winning hypothesis H*
                    -> exact temporal grounding

    CRA has two strictly separated roles.

    1) Factual downstream consumer
       ----------------------------
       The winning Q-Bound hypothesis pools CRA ``x_bar`` and produces the
       ordinary answer-space video representation returned by this wrapper.
       This keeps the experimental testbed comparable with prior CRA runs.

    2) Privileged training-time answer critic
       ---------------------------------------
       Candidate QA embeddings are allowed to evaluate *all* temporal
       hypotheses through RefineDecoder.score_privileged_hypothesis_bank().
       The critic is stop-gradient with respect to:

         * CRA x_bar;
         * candidate answer/QA embeddings;
         * CRA video projection weights/bias;
         * temporal hypothesis masks/weights.

       Hence the critic can provide ranking targets to the training pipeline,
       but answer features NEVER enter QBoundCore and NEVER determine the
       winning hypothesis in the forward grounding path.  The intended method
       is therefore:

           answer-aware learning, answer-free grounding.

    The training pipeline owns the actual privileged-ranking objective.  This
    wrapper only validates and exposes:

        qbound_teacher_hypothesis_answer_logits   [B,S,C]
        qbound_teacher_complement_answer_logits   [B,S,C]
        qbound_teacher_full_video_answer_logits   [B,C]

    plus the hypothesis neighbor indices (trim/expand) already produced by the
    Q-Bound core.  Temporal GT / IoU / IoP are never consumed here.

    Controlled V1.12 invariants
    ---------------------------
      * Q-Bound input is the explicit generic pre-CRA visual sequence.
      * Q-Bound core is question-only; candidate answers do not enter it.
      * Event count is dynamic up to a capacity ceiling.
      * Event boundaries guide reasoning but do not constrain final precision:
        the hypothesis bank is frame-exact.
      * Final grounding is the hard winning hypothesis, not an all-span
        differentiable marginalization.
      * Gaussian/fixed-window smoothing is not applied to V1.12 MAIN.
      * CRA temporal grounding is reference/diagnostic only when Q-Bound is on.
      * VisualIntervention is OFF for the controlled V1.12 run.
      * No CRA->Q-Bound handover and no direct reasoned-event residual.
      * Legacy Composer/Refiner/PSRC/donor-counterfactual branches are disabled.
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
                    model_cfg.get("gpeb_version", "1.12"),
                ),
            )
        )

        if self.qbound_enabled and self.qbound_version != "1.12":
            raise ValueError(
                "Q-Bound is enabled, but this CRA wrapper expects "
                f"model.qbound.version='1.12'; got {self.qbound_version!r}."
            )

        # V1.12's student scorer must be strictly (V,Q)-conditioned.  Allowing
        # a candidate-QA fallback here would invalidate the claimed generic
        # answer-free grounding interface.
        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get("qbound_require_question_only", True),
            )
        )
        if self.qbound_enabled and not self.qbound_require_question_only:
            raise ValueError(
                "Controlled Q-Bound V1.12 requires "
                "model.qbound.require_question_only=true."
            )

        if self.qbound_enabled and bool(model_cfg.get("do", False)):
            raise ValueError(
                "Controlled Q-Bound V1.12 requires model.do=false. "
                "VisualIntervention would confound hypothesis competition and "
                "the privileged answer-aware critic."
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

        # Kept for repository/checkpoint compatibility.  The controlled V1.12
        # Q-Bound path requires model.do=false and does not use this object to
        # construct temporal hypotheses.
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
                if not hasattr(
                    self.decoder,
                    "score_privileged_hypothesis_bank",
                ):
                    raise RuntimeError(
                        "Q-Bound V1.12 RefineDecoder must implement "
                        "score_privileged_hypothesis_bank()."
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
        """Return candidate labels as a robust [B] LongTensor."""
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
        """Select one [D] candidate feature from [B,C,D] per sample."""
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
        """
        Encode candidate question-answer sequences.

        Expected:
            answer: [B,C,L]

        Returns:
            answer_global: [B,C,D]
            answer_tokens: [B,C,L,D]
        """
        if answer is None:
            raise ValueError("answer tokens must not be None.")

        answer = answer.to(device=device, non_blocking=True)
        if answer.ndim != 3:
            raise ValueError(
                "Candidate answer tokens must have shape [B,C,L], got "
                f"{tuple(answer.shape)}."
            )

        if hasattr(self.lang_model, "encode_candidates"):
            answer_global, answer_tokens = self.lang_model.encode_candidates(
                answer
            )
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
        Encode question-only tokens for Q-Bound.

        Accepted:
            [B,L] or [B,1,L]

        Returns:
            question_global: [B,D]
            question_tokens: [B,L,D]
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
            question_global, question_tokens = self.lang_model.encode_question(
                question
            )
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
        Apply the repository's ``prop_num`` candidate aggregation to logits.

        Accepted shapes are [B,C_total] and [B,S,C_total].  This keeps the
        privileged critic in exactly the same answer space as the factual CRA
        scoring rule used by the pipeline.
        """
        if logits.ndim not in {2, 3}:
            raise ValueError(
                "Teacher logits must be [B,C] or [B,S,C], got "
                f"{tuple(logits.shape)}."
            )

        prop_num = int(self.cfgs["model"].get("prop_num", 1))
        if prop_num < 1:
            raise ValueError("model.prop_num must be positive.")

        total_candidates = logits.size(-1)
        if total_candidates % prop_num != 0:
            raise ValueError(
                "Privileged-critic candidate dimension is not divisible by "
                f"prop_num: {total_candidates} vs {prop_num}."
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
    # V1.12 contract validation
    # ------------------------------------------------------------------
    def _validate_v112_contract(
        self,
        time_param,
        qbound_aux,
        video_proj,
        batch_size,
    ):
        if not isinstance(time_param, dict):
            raise TypeError("RefineDecoder must return time_param as a dict.")
        if not isinstance(qbound_aux, dict):
            raise TypeError(
                "Q-Bound V1.12 must expose time_param['qbound_aux'] as a dict."
            )

        required_aux = {
            "qbound_version",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "generic_input_adapter_enabled",
            "generic_input_context_reuse",
            "dynamic_event_count_enabled",
            "event_capacity",
            "hypothesis_competition_enabled",
            "answer_features_used_in_core",
            "legacy_composer_enabled",
            "legacy_refiner_enabled",
            "legacy_counterfactual_donor_enabled",
            "privileged_answer_teacher_ready",
            "critic_consumer_stop_gradient",
            "critic_projection_dropout_enabled",
            "event_count",
            "event_assignment",
            "event_valid_mask",
            "event_visual_tokens",
            "evidence_visual_token",
            "reasoned_event_tokens",
            "decomp_loss",
            "hypothesis_start_frame",
            "hypothesis_end_frame",
            "hypothesis_valid_mask",
            "hypothesis_frame_weights",
            "hypothesis_complement_frame_weights",
            "hypothesis_complement_valid_mask",
            "hypothesis_trim_left_index",
            "hypothesis_trim_right_index",
            "hypothesis_expand_left_index",
            "hypothesis_expand_right_index",
            "hypothesis_trim_left_valid_mask",
            "hypothesis_trim_right_valid_mask",
            "hypothesis_expand_left_valid_mask",
            "hypothesis_expand_right_valid_mask",
            "hypothesis_logits",
            "hypothesis_posterior",
            "hypothesis_entropy",
            "hypothesis_effective_count",
            "hypothesis_top1_posterior",
            "hypothesis_expected_span_length",
            "selected_hypothesis_index",
            "selected_hypothesis_start_frame",
            "selected_hypothesis_end_frame",
            "selected_hypothesis_frame_mask",
            "selected_hypothesis_frame_weights",
            "selected_hypothesis_span_length",
            "selected_hypothesis_span_fraction",
            "main_temporal_prob",
            "main_temporal_logits",
            "main_soft_support",
            "main_pair_posterior",
            "cra_key",
            "qbound_key",
            "final_key",
            "main_video_proj",
            "qbound_video_proj",
            "qbound_selected_video_proj",
            "cra_video_proj",
            "qbound_smoothing_applied",
            "qbound_handover_enabled",
            "qbound_direct_reasoning_residual_enabled",
        }
        missing_aux = sorted(required_aux.difference(qbound_aux))
        if missing_aux:
            raise KeyError(
                "Q-Bound V1.12 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_time = {
            "grounding_feature",
            "qbound_grounding_feature",
            "qbound_temporal_main_feature",
            "qbound_key",
            "qbound_selected_video_proj",
            "qbound_selected_hypothesis_index",
            "qbound_selected_hypothesis_start_frame",
            "qbound_selected_hypothesis_end_frame",
            "qbound_selected_hypothesis_mask",
            "qbound_selected_hypothesis_span_length",
            "qbound_selected_hypothesis_span_fraction",
            "qbound_dynamic_event_count_enabled",
            "qbound_hypothesis_competition_enabled",
            "qbound_answer_features_used_in_core",
            "qbound_legacy_composer_enabled",
            "qbound_legacy_refiner_enabled",
            "qbound_legacy_counterfactual_donor_enabled",
            "qbound_privileged_answer_teacher_ready",
            "qbound_critic_consumer_stop_gradient",
            "qbound_critic_projection_dropout_enabled",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_smoothing_applied",
        }
        missing_time = sorted(required_time.difference(time_param))
        if missing_time:
            raise KeyError(
                "RefineDecoder/Q-Bound V1.12 time_param output is missing: "
                + ", ".join(missing_time)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "CRA/RefineDecoder/QBoundCore version mismatch: "
                f"CRA={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        if self._scalar_flag(qbound_aux, "qbound_input_is_generic") != 1.0:
            raise RuntimeError(
                "Q-Bound V1.12 must consume the explicit generic visual stream."
            )
        if self._scalar_flag(qbound_aux, "qbound_input_uses_cra_embedding") != 0.0:
            raise RuntimeError(
                "Q-Bound V1.12 unexpectedly reports CRA-embedded input."
            )
        if self._scalar_flag(qbound_aux, "generic_input_context_reuse") != 1.0:
            raise RuntimeError(
                "Controlled V1.12 expects one adapted generic stream to be "
                "reused for Q-Bound raw/context states."
            )
        if self._scalar_flag(qbound_aux, "dynamic_event_count_enabled") != 1.0:
            raise RuntimeError("V1.12 requires dynamic event-count decomposition.")
        if self._scalar_flag(qbound_aux, "hypothesis_competition_enabled") != 1.0:
            raise RuntimeError("V1.12 requires temporal hypothesis competition.")
        if self._scalar_flag(qbound_aux, "answer_features_used_in_core") != 0.0:
            raise RuntimeError(
                "Q-Bound V1.12 core must remain answer-free at grounding time."
            )
        if self._scalar_flag(qbound_aux, "privileged_answer_teacher_ready") != 1.0:
            raise RuntimeError("V1.12 privileged answer critic is not ready.")
        if self._scalar_flag(qbound_aux, "critic_consumer_stop_gradient") != 1.0:
            raise RuntimeError(
                "V1.12 privileged answer critic must be stop-gradient."
            )
        if self._scalar_flag(qbound_aux, "critic_projection_dropout_enabled") != 0.0:
            raise RuntimeError(
                "V1.12 privileged critic must be deterministic (dropout off)."
            )
        if self._scalar_flag(qbound_aux, "qbound_smoothing_applied") != 0.0:
            raise RuntimeError(
                "V1.12 MAIN must not Gaussian-smooth the exact winning hypothesis."
            )
        if self._scalar_flag(qbound_aux, "qbound_handover_enabled") != 0.0:
            raise RuntimeError("V1.12 must not use CRA->Q-Bound temporal handover.")
        if self._scalar_flag(
            qbound_aux,
            "qbound_direct_reasoning_residual_enabled",
        ) != 0.0:
            raise RuntimeError(
                "V1.12 must not inject a direct event-reasoning residual into "
                "the answer representation."
            )

        for legacy_name in (
            "legacy_composer_enabled",
            "legacy_refiner_enabled",
            "legacy_counterfactual_donor_enabled",
        ):
            if self._scalar_flag(qbound_aux, legacy_name) != 0.0:
                raise RuntimeError(
                    f"V1.12 requires {legacy_name}=0."
                )

        if qbound_aux["main_video_proj"].shape != video_proj.shape:
            raise ValueError(
                "Q-Bound main_video_proj shape does not match returned video_proj."
            )
        if qbound_aux["hypothesis_logits"].ndim != 2:
            raise ValueError("hypothesis_logits must be [B,S].")
        if qbound_aux["hypothesis_logits"].size(0) != batch_size:
            raise ValueError("hypothesis_logits batch dimension mismatch.")
        if qbound_aux["hypothesis_frame_weights"].ndim != 3:
            raise ValueError("hypothesis_frame_weights must be [B,S,T].")
        if qbound_aux["hypothesis_frame_weights"].shape[:2] != (
            batch_size,
            qbound_aux["hypothesis_logits"].size(1),
        ):
            raise ValueError(
                "hypothesis_frame_weights and hypothesis_logits disagree on [B,S]."
            )

    # ------------------------------------------------------------------
    # V1.12 refine forward
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
        Refined CRA/Q-Bound V1.12 forward.

        ``qbound_mix`` remains in the repository call signature only.  V1.12
        does not mix CRA and Q-Bound temporal keys.
        """
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
            # Preserve the original CRA testbed condition.  This candidate-QA
            # aggregate is used only by CRA's own decoder/reference path;
            # Q-Bound receives a separately encoded question-only feature.
            qsn_g = answer_g.max(dim=1).values
            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(
                    answer_g,
                    answer_indices,
                )

        use_qbound = bool(self.qbound_enabled and mode == "VQA")

        question_g = None
        question_tokens = None
        if use_qbound:
            if question is None:
                raise ValueError(
                    "Q-Bound V1.12 requires question-only token ids. "
                    "Pass question=... through CRA.forward()."
                )
            question_g, question_tokens = self._encode_question(
                question,
                device,
            )
            if question_g.size(0) != batch_size:
                raise ValueError(
                    "Question and answer batch sizes do not match: "
                    f"{question_g.size(0)} vs {batch_size}."
                )

        video_proj, video_attn, time_param = self.decoder(
            video_f,
            qsn_g,
            video_mask,
            gsub=gsub,
            do=self.cfgs["model"].get("do", False),
            window_sizes=self.cfgs["model"].get("window_sizes", [1, 3, 5]),
            use_qbound=use_qbound,
            question_feature=question_g,
            qbound_video_tokens=(video if use_qbound else None),
            qbound_mix=(qbound_mix if use_qbound else 0.0),
        )

        # Candidate QA embeddings remain the normal answer interface used by
        # the repository's factual VQA scoring code.
        answer_proj = answer_g

        qbound_aux = (
            time_param.get("qbound_aux")
            if isinstance(time_param, dict)
            else None
        )

        if use_qbound:
            self._validate_v112_contract(
                time_param=time_param,
                qbound_aux=qbound_aux,
                video_proj=video_proj,
                batch_size=batch_size,
            )

            # ----------------------------------------------------------
            # Privileged answer-aware critic.
            # ----------------------------------------------------------
            # This branch cannot alter the already-selected grounding.  It is
            # evaluated only after Q-Bound has produced its answer-free
            # hypothesis bank/scorer outputs.  RefineDecoder detaches every
            # critic input and CRA projection parameter.
            with torch.no_grad():
                teacher = self.decoder.score_privileged_hypothesis_bank(
                    answer_proj=answer_proj,
                    qbound_aux=qbound_aux,
                )

            required_teacher = {
                "hypothesis_answer_logits",
                "complement_answer_logits",
                "full_video_answer_logits",
                "hypothesis_valid_mask",
                "complement_valid_mask",
                "critic_stop_gradient",
                "critic_projection_dropout_enabled",
            }
            missing_teacher = sorted(required_teacher.difference(teacher))
            if missing_teacher:
                raise KeyError(
                    "V1.12 privileged critic output is missing: "
                    + ", ".join(missing_teacher)
                )

            # Match the repository's ordinary VQA answer space exactly.
            # RefineDecoder scores raw candidate embeddings; CRA performs the
            # same prop_num grouping/mean used by the pipeline's factual scorer.
            hypothesis_answer_logits = self._aggregate_prop_logits(
                teacher["hypothesis_answer_logits"]
            )
            complement_answer_logits = self._aggregate_prop_logits(
                teacher["complement_answer_logits"]
            )
            full_video_answer_logits = self._aggregate_prop_logits(
                teacher["full_video_answer_logits"]
            )

            num_hypotheses = qbound_aux["hypothesis_logits"].size(1)
            prop_num = int(self.cfgs["model"].get("prop_num", 1))
            num_candidates = answer_proj.size(1) // prop_num
            expected_h_shape = (
                batch_size,
                num_hypotheses,
                num_candidates,
            )
            if hypothesis_answer_logits.shape != expected_h_shape:
                raise ValueError(
                    "V1.12 teacher hypothesis logits shape mismatch: expected "
                    f"{expected_h_shape}, got "
                    f"{tuple(hypothesis_answer_logits.shape)}."
                )
            if complement_answer_logits.shape != expected_h_shape:
                raise ValueError(
                    "V1.12 teacher complement logits shape mismatch: expected "
                    f"{expected_h_shape}, got "
                    f"{tuple(complement_answer_logits.shape)}."
                )
            if full_video_answer_logits.shape != (
                batch_size,
                num_candidates,
            ):
                raise ValueError(
                    "V1.12 teacher full-video logits shape mismatch."
                )

            if float(teacher["critic_stop_gradient"].item()) != 1.0:
                raise RuntimeError(
                    "V1.12 privileged critic unexpectedly carries gradients."
                )
            if float(
                teacher["critic_projection_dropout_enabled"].item()
            ) != 0.0:
                raise RuntimeError(
                    "V1.12 privileged critic unexpectedly enables projection dropout."
                )

            # Store teacher outputs under explicit names.  The training pipeline
            # will turn them into sufficiency / irredundancy / completeness /
            # exclusivity utilities and a detached listwise target.
            teacher_mapping = {
                "teacher_hypothesis_answer_logits": hypothesis_answer_logits,
                "teacher_complement_answer_logits": complement_answer_logits,
                "teacher_full_video_answer_logits": full_video_answer_logits,
                "teacher_hypothesis_valid_mask": teacher[
                    "hypothesis_valid_mask"
                ],
                "teacher_complement_valid_mask": teacher[
                    "complement_valid_mask"
                ],
                "teacher_critic_stop_gradient": teacher[
                    "critic_stop_gradient"
                ],
                "teacher_critic_projection_dropout_enabled": teacher[
                    "critic_projection_dropout_enabled"
                ],
            }
            qbound_aux.update(teacher_mapping)
            for key, value in teacher_mapping.items():
                time_param[f"qbound_{key}"] = value

            # Explicit paper-level invariants.
            qbound_aux["teacher_uses_answer_information"] = torch.ones(
                (), dtype=video_proj.dtype, device=video_proj.device
            )
            qbound_aux["teacher_affects_grounding_forward"] = torch.zeros(
                (), dtype=video_proj.dtype, device=video_proj.device
            )
            qbound_aux["grounding_requires_answer_at_inference"] = torch.zeros(
                (), dtype=video_proj.dtype, device=video_proj.device
            )
            time_param["qbound_teacher_uses_answer_information"] = qbound_aux[
                "teacher_uses_answer_information"
            ]
            time_param["qbound_teacher_affects_grounding_forward"] = qbound_aux[
                "teacher_affects_grounding_forward"
            ]
            time_param["qbound_grounding_requires_answer_at_inference"] = qbound_aux[
                "grounding_requires_answer_at_inference"
            ]

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

        # Surface every decoder-level qbound_* item directly for convenient
        # pipeline logging, while keeping qbound_aux as the canonical detailed
        # interface.  This avoids another brittle hand-maintained V1.11-sized
        # list of keys.
        if isinstance(time_param, dict):
            for key, value in time_param.items():
                if key.startswith("qbound_"):
                    outs[key] = value

        if isinstance(qbound_aux, dict):
            # Canonical explicit interfaces used by the V1.12 pipeline.
            direct_aux_names = (
                "hypothesis_logits",
                "hypothesis_posterior",
                "hypothesis_valid_mask",
                "hypothesis_start_frame",
                "hypothesis_end_frame",
                "hypothesis_frame_weights",
                "hypothesis_complement_frame_weights",
                "hypothesis_complement_valid_mask",
                "hypothesis_trim_left_index",
                "hypothesis_trim_right_index",
                "hypothesis_expand_left_index",
                "hypothesis_expand_right_index",
                "hypothesis_trim_left_valid_mask",
                "hypothesis_trim_right_valid_mask",
                "hypothesis_expand_left_valid_mask",
                "hypothesis_expand_right_valid_mask",
                "hypothesis_entropy",
                "hypothesis_effective_count",
                "hypothesis_top1_posterior",
                "hypothesis_expected_span_length",
                "selected_hypothesis_index",
                "selected_hypothesis_start_frame",
                "selected_hypothesis_end_frame",
                "selected_hypothesis_frame_mask",
                "selected_hypothesis_span_length",
                "selected_hypothesis_span_fraction",
                "event_count",
                "event_assignment",
                "event_valid_mask",
                "event_capacity",
                "decomp_loss",
                "event_visual_tokens",
                "evidence_visual_token",
                "reasoned_event_tokens",
                "dynamic_event_count_enabled",
                "hypothesis_competition_enabled",
                "answer_features_used_in_core",
                "legacy_composer_enabled",
                "legacy_refiner_enabled",
                "legacy_counterfactual_donor_enabled",
                "teacher_hypothesis_answer_logits",
                "teacher_complement_answer_logits",
                "teacher_full_video_answer_logits",
                "teacher_hypothesis_valid_mask",
                "teacher_complement_valid_mask",
                "teacher_critic_stop_gradient",
                "teacher_critic_projection_dropout_enabled",
                "teacher_uses_answer_information",
                "teacher_affects_grounding_forward",
                "grounding_requires_answer_at_inference",
            )
            for name in direct_aux_names:
                if name in qbound_aux:
                    outs[f"qbound_{name}"] = qbound_aux[name]

            # MAIN answer/key aliases.  V1.12's "calibrated/refined" aliases
            # refer to the exact winning hypothesis for compatibility only;
            # there is no PPBR/PSRC calibration path anymore.
            outs["qbound_selected_video_proj"] = qbound_aux.get(
                "qbound_selected_video_proj"
            )
            outs["qbound_calibrated_video_proj"] = qbound_aux.get(
                "qbound_selected_video_proj"
            )
            outs["qbound_refined_video_proj"] = qbound_aux.get(
                "qbound_selected_video_proj"
            )
            outs["qbound_key"] = qbound_aux.get("qbound_key")
            outs["qbound_calibrated_key"] = qbound_aux.get("qbound_key")
            outs["qbound_refined_key"] = qbound_aux.get("qbound_key")
            outs["cra_video_proj"] = qbound_aux.get("cra_video_proj")
            outs["cra_key"] = qbound_aux.get("cra_key")

            # Explicit detached candidate-answer features are exposed only for
            # diagnostics/teacher bookkeeping.  The pipeline should normally
            # use the precomputed teacher logits above rather than rescoring.
            outs["qbound_privileged_answer_proj"] = answer_proj.detach()

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
    def align_loss(
        self,
        qsn,
        grounding_feature,
        neg_sample=None,
    ):
        def _sample_negatives(x_pos, k):
            """Sample k negatives for each item, with replacement."""
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
            x_neg = _sample_negatives(grounding_feature, negative_count)
        else:
            x_neg = neg_sample

        return self._info_nce_loss(qsn, grounding_feature, x_neg)

    @staticmethod
    def _info_nce_loss(
        x,
        x_pos,
        x_neg,
        temperature=0.07,
    ):
        """Compute InfoNCE with one positive and K negatives."""
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
