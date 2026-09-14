import math

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
    CRA wrapper with Q-Bound Version 1.11.

    Q-Bound V1.11:

        DECOMPOSE -> RELATE -> COMPOSE
                    |
                    +-> differentiable coarse MAIN component
                    |
                    +-> detached Composer posterior -> boundary-anchored prior
                                                       -> bounded Refiner correction
                                                       -> reliability calibration
                                                       -> GROUND

    CRA remains only the downstream VideoQA testbed. The official returned
    ``video_proj`` is the reliability-calibrated V1.11 MAIN representation.

    V1.11 input decoupling:
      * ``video`` is the raw/generic sampled visual stream (CLIP-L in NextGQA);
      * ``video_f = CRA.VideoEmbedding(video)`` remains CRA-side only;
      * the raw ``video`` tensor is passed separately as ``qbound_video_tokens``;
      * RefineDecoder/QBoundCore fail fast if Q-Bound falls back to CRA-embedded
        visual tokens.

    Three answer-space branches are exposed:

        qbound_calibrated_video_proj
            Official V1.11 MAIN branch.

        qbound_raw_refined_video_proj
            Boundary-anchored posterior-guided fine REFINE branch, diagnostics/ablation only.

        qbound_coarse_video_proj
            Detached hard-top1 COMPOSE branch, diagnostics/ablation only.

    Compatibility:
      * qbound_refined_video_proj aliases the calibrated MAIN branch.
      * legacy V1.8 calibration names remain available, but
        ``calibration_confidence`` now aliases the detached V1.10.2 reliability.

    Gradient/routing controls:
      * question-only semantics define Q-Bound temporal reasoning;
      * REFINE no longer receives a deterministic hard top-1 Composer region;
      * the Composer posterior routed into REFINE is explicitly detached;
      * V1.10.1 BAPGR anchors start probability toward the first-event left
        boundary and end probability toward the last-event right boundary;
      * V1.10.2 PPBR bounds the learned Refiner correction so it can re-rank
        legal pairs without overriding the structural prior without limit;
      * V1.6 differentiable all-span REFINE marginalization remains disabled;
      * V1.7 minimal-sufficient routing remains disabled;
      * the MAIN probability contains differentiable coarse and refined
        components, so the task can train Composer and Refiner while the
        Composer->Refiner routing prior stays stop-gradient;
      * V1.10.2 reliability is detached before coarse/fine fusion;
      * no CRA->Q-Bound temporal handover;
      * no direct reasoned-event residual into answer space;
      * no temporal GT, IoU/IoP loss, or target span length is consumed here.

    Hard top-1 Composer outputs are intentionally retained as diagnostics and
    as a future selected-event-token interface; they no longer define the
    Refiner search domain.
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
                    model_cfg.get("gpeb_version", "1.11"),
                ),
            )
        )

        # This wrapper is the controlled V1.11 contract. Fail early instead of
        # silently pairing the V1.11 decoder with an older core/config.
        if self.qbound_enabled and self.qbound_version != "1.11":
            raise ValueError(
                "Q-Bound is enabled, but this CRA wrapper expects "
                f"model.qbound.version='1.11'; got {self.qbound_version!r}."
            )
        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get(
                    "qbound_require_question_only",
                    model_cfg.get("gpeb_require_question_only", True),
                ),
            )
        )

        # Compatibility aliases for old checkpoints/logging.
        self.gpeb_enabled = self.qbound_enabled
        self.gpeb_version = self.qbound_version
        self.gpeb_require_question_only = (
            self.qbound_require_question_only
        )

        d_model = model_cfg["d_model"]

        self.lang_model = LanModel(
            tokenizer,
            lan=model_cfg["lan"],
            out_dim=d_model,
        )
        self.video_model = VideoEmbedding(cfgs)
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
        """Select one [D] feature from [B,C,D] for every sample."""
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must be [B,C,D], "
                f"got {tuple(candidate_features.shape)}."
            )

        if candidate_indices.ndim != 1:
            raise ValueError(
                "candidate_indices must be [B], "
                f"got {tuple(candidate_indices.shape)}."
            )

        batch_indices = torch.arange(
            candidate_features.size(0),
            device=candidate_features.device,
        )

        return candidate_features[
            batch_indices,
            candidate_indices,
        ]

    def _encode_answers(
        self,
        answer,
        device: torch.device,
    ):
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

        answer = answer.to(
            device=device,
            non_blocking=True,
        )

        if answer.ndim != 3:
            raise ValueError(
                "Candidate answer tokens must have shape [B,C,L], "
                f"got {tuple(answer.shape)}."
            )

        if hasattr(self.lang_model, "encode_candidates"):
            answer_global, answer_tokens = (
                self.lang_model.encode_candidates(answer)
            )
        else:
            answer_global, answer_tokens = self.lang_model(answer)

        if answer_global.ndim != 3:
            raise ValueError(
                "LanModel must return candidate features [B,C,D], "
                f"got {tuple(answer_global.shape)}."
            )

        return answer_global, answer_tokens

    def _encode_question(
        self,
        question,
        device: torch.device,
    ):
        """
        Encode question-only tokens for Q-Bound.

        Accepted:
            [B,L]
            [B,1,L]

        Returns:
            question_global: [B,D]
            question_tokens: [B,L,D]
        """
        if question is None:
            return None, None

        question = question.to(
            device=device,
            non_blocking=True,
        )

        if question.ndim == 3 and question.size(1) == 1:
            question = question.squeeze(1)

        if question.ndim != 2:
            raise ValueError(
                "Question-only tokens must have shape [B,L] "
                "or [B,1,L], "
                f"got {tuple(question.shape)}."
            )

        if hasattr(self.lang_model, "encode_question"):
            question_global, question_tokens = (
                self.lang_model.encode_question(question)
            )
        else:
            question_global, question_tokens = self.lang_model(question)

        if question_global.ndim != 2:
            raise ValueError(
                "LanModel must return question features [B,D], "
                f"got {tuple(question_global.shape)}."
            )

        return question_global, question_tokens

    @staticmethod
    def _move_video_mask(
        video_mask,
        device: torch.device,
    ):
        if video_mask is None:
            return None

        return video_mask.to(
            device=device,
            non_blocking=True,
        )

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
        Refined CRA/Q-Bound V1.11 forward.

        The returned ``video_proj`` is the official posterior-stable,
        reliability-calibrated V1.11 MAIN representation. The raw BAPGR + PPBR
        Refiner and hard-top1 Composer representations are diagnostics only.

        ``qbound_mix`` is compatibility-only; V1.11 has no CRA/Q-Bound handover.
        """
        mode = mode.upper()

        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask,
            device,
        )

        answer_g, _ = self._encode_answers(
            answer,
            device,
        )

        batch_size = answer_g.size(0)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            batch_size,
            answer_g.device,
        )

        video_f = self.video_model(video)
        qsn_gt = None

        if mode == "VQ" and answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
            qsn_gt = qsn_g
        else:
            qsn_g = answer_g.max(dim=1).values
            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(
                    answer_g,
                    answer_indices,
                )

        use_qbound = (
            self.qbound_enabled
            and mode == "VQA"
        )

        question_g = None
        question_tokens = None

        if use_qbound:
            if question is None:
                if self.qbound_require_question_only:
                    raise ValueError(
                        "Q-Bound V1.11 requires question-only token ids. "
                        "Pass them through CRA.forward(..., question=...). "
                        "Do not reuse candidate-answer features."
                    )
            else:
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
            window_sizes=self.cfgs["model"].get(
                "window_sizes",
                [1, 3, 5],
            ),
            use_qbound=use_qbound,
            question_feature=question_g,
            qbound_video_tokens=(
                video if use_qbound else None
            ),
            qbound_mix=(
                qbound_mix
                if use_qbound
                else 0.0
            ),
        )

        # Keep CRA's answer candidate interface unchanged.
        answer_proj = answer_g

        qbound_aux = (
            time_param.get("qbound_aux")
            if isinstance(time_param, dict)
            else None
        )

        # --------------------------------------------------------------
        # Validate V1.11 decoder/core contract.
        # --------------------------------------------------------------
        if use_qbound:
            required_time_keys = {
                # V1.11 generic-input / counterfactual interface.
                "qbound_generic_input_adapter_enabled",
                "qbound_generic_input_position_enabled",
                "qbound_generic_input_context_reuse",
                "qbound_generic_input_raw_rms",
                "qbound_generic_input_adapted_rms",
                "qbound_generic_input_token_cosine",
                "qbound_generic_input_delta_ratio",
                "qbound_generic_context_raw_rms",
                "qbound_generic_context_adapted_rms",
                "qbound_counterfactual_evidence_interface_enabled",
                "qbound_counterfactual_keep_support",
                "qbound_counterfactual_remove_support",
                "qbound_counterfactual_keep_fraction",
                "qbound_counterfactual_remove_fraction",
                "qbound_input_is_generic",
                "qbound_input_uses_cra_embedding",

                # Official calibrated MAIN.
                "qbound_grounding_feature",
                "qbound_temporal_main_feature",
                "qbound_calibrated_grounding_feature",
                "qbound_calibrated_video_proj",
                "qbound_key",
                "qbound_calibrated_key",
                "refined_key",

                # Diagnostic answer/key branches.
                "qbound_raw_refined_grounding_feature",
                "qbound_raw_refined_video_proj",
                "qbound_raw_refined_key",
                "qbound_coarse_grounding_feature",
                "qbound_coarse_video_proj",
                "qbound_coarse_key",

                # Routing/calibration.
                "qbound_stable_refiner_enabled",
                "qbound_uncertainty_calibration_enabled",
                "qbound_minimal_sufficient_enabled",
                "qbound_refine_hard_coarse_route",
                "qbound_refine_coarse_detached",
                "qbound_all_span_refine_marginal_enabled",
                "qbound_calibration_uncertainty",
                "qbound_calibration_information",
                "qbound_calibration_confidence",
                "qbound_calibration_confidence_detached",
                "qbound_calibration_reliability",
                "qbound_refiner_information_gain",
                "qbound_refined_semantic_retention",
                "qbound_base_refiner_confidence",

                # V1.10.2 BAPGR + PPBR routing / posterior-stable reliability.
                "qbound_posterior_guided_refinement_enabled",
                "qbound_boundary_structured_refinement_enabled",
                "qbound_prior_preserving_bounded_refinement_enabled",
                "qbound_posterior_stable_reliability_enabled",
                "qbound_refine_boundary_anchor_power",
                "qbound_refine_correction_cap",
                "qbound_reliability_gain_factor",
                "qbound_composer_pair_prior",
                "qbound_composer_endpoint_pair_prior",
                "qbound_composer_pair_prior_valid_mask",
                "qbound_composer_pair_prior_entropy",
                "qbound_composer_pair_prior_entropy_ratio",
                "qbound_composer_pair_prior_effective_count",
                "qbound_composer_pair_prior_top1_posterior",
                "qbound_composer_pair_prior_candidate_count",
                "qbound_composer_uniform_pair_prior_entropy",
                "qbound_composer_uniform_pair_prior_entropy_ratio",
                "qbound_composer_uniform_pair_prior_effective_count",
                "qbound_composer_uniform_pair_prior_top1_posterior",
                "qbound_composer_pair_prior_expected_span_length",
                "qbound_composer_uniform_pair_prior_expected_span_length",
                "qbound_composer_pair_prior_length_gain",
                "qbound_composer_pair_soft_support",
                "qbound_composer_expected_start_frame",
                "qbound_composer_expected_end_frame",
                "qbound_composer_expected_span_length",
                "qbound_main_pair_posterior",
                "qbound_main_pair_entropy",
                "qbound_main_pair_effective_count",
                "qbound_main_pair_top1_posterior",

                # V1.10.2 PPBR diagnostics.
                "qbound_refiner_raw_correction_abs_mean",
                "qbound_refiner_raw_correction_abs_max",
                "qbound_refiner_bounded_correction_abs_mean",
                "qbound_refiner_bounded_correction_abs_max",
                "qbound_refiner_correction_saturation_ratio",
                "qbound_refiner_vs_prior_kl",
                "qbound_refiner_vs_prior_js",
                "qbound_raw_vs_prior_expected_length_ratio",

                # COMPOSE.
                "qbound_span_posterior",
                "qbound_span_entropy",
                "qbound_span_effective_count",
                "qbound_span_top1_posterior",
                "qbound_composed_temporal_prob",
                "qbound_selected_span_index",
                "qbound_selected_event_mask",
                "qbound_selected_reasoned_event_tokens",
                "qbound_selected_span_event_count",
                "qbound_selected_span_frame_length",
                "qbound_selected_span_fraction",

                # Raw fine proposal.
                "qbound_raw_refined_temporal_prob",
                "qbound_refine_selected_pair_entropy",
                "qbound_refine_selected_pair_effective_count",
                "qbound_refine_selected_pair_top1_posterior",
                "qbound_refine_selected_pair_candidate_count",
                "qbound_refine_selected_pair_entropy_ratio",

                # Calibrated core interval.
                "qbound_calibrated_temporal_prob",
                "qbound_calibrated_span_start_frame",
                "qbound_calibrated_span_end_frame",
                "qbound_calibrated_span_frame_length",
                "qbound_calibrated_span_fraction",
                "qbound_calibrated_retained_fraction",
                "qbound_calibrated_inside_coarse",

                # Raw refined core interval.
                "qbound_refined_span_start_frame",
                "qbound_refined_span_end_frame",
                "qbound_refined_span_frame_length",
                "qbound_refined_span_fraction",
                "qbound_refined_inside_coarse",

                # Adapter-level intervals.
                "qbound_calibrated_start",
                "qbound_calibrated_end",
                "qbound_calibrated_mask",
                "qbound_raw_refined_start",
                "qbound_raw_refined_end",
                "qbound_raw_refined_mask",
                "qbound_coarse_start",
                "qbound_coarse_end",
                "qbound_coarse_mask",

                # DECOMPOSE / RELATE.
                "qbound_event_assignment",
                "qbound_event_mass",
                "qbound_soft_effective_event_count",
                "qbound_decomp_loss",
                "qbound_event_visual_tokens",
                "qbound_evidence_visual_token",
                "qbound_event_valid_mask",
                "qbound_reasoned_event_tokens",
                "qbound_ordered_relation_gates",
                "qbound_ordered_update_gate",

                # Shortcut controls.
                "qbound_reasoning_residual",
                "qbound_reasoning_residual_used",
                "qbound_direct_reasoning_residual_enabled",

                # Historical diagnostics.
                "qbound_v15_diagnostic_key",
                "qbound_v15_diagnostic_start",
                "qbound_v15_diagnostic_end",
                "qbound_v14_diagnostic_key",
                "qbound_v14_diagnostic_start",
                "qbound_v14_diagnostic_end",
            }

            missing_time_keys = sorted(
                required_time_keys.difference(time_param)
            )
            if missing_time_keys:
                raise KeyError(
                    "RefineDecoder/Q-Bound V1.11 time_param output is missing: "
                    + ", ".join(missing_time_keys)
                )

            if qbound_aux is None:
                raise KeyError(
                    "RefineDecoder/Q-Bound V1.11 must expose "
                    "time_param['qbound_aux']."
                )

            required_aux_keys = {
                # V1.11 generic-input / counterfactual interface.
                "qbound_version",
                "generic_input_adapter_enabled",
                "generic_input_position_enabled",
                "generic_input_context_reuse",
                "generic_input_raw_rms",
                "generic_input_adapted_rms",
                "generic_input_token_cosine",
                "generic_input_delta_ratio",
                "generic_context_raw_rms",
                "generic_context_adapted_rms",
                "counterfactual_evidence_interface_enabled",
                "counterfactual_keep_support",
                "counterfactual_remove_support",
                "counterfactual_keep_fraction",
                "counterfactual_remove_fraction",
                "qbound_input_is_generic",
                "qbound_input_uses_cra_embedding",

                # Answer representations.
                "main_video_proj",
                "qbound_video_proj",
                "qbound_calibrated_video_proj",
                "qbound_refined_video_proj",
                "qbound_raw_refined_video_proj",
                "qbound_coarse_video_proj",

                # Temporal keys.
                "qbound_key",
                "qbound_calibrated_key",
                "qbound_raw_refined_key",
                "qbound_coarse_key",
                "final_key",

                # Temporal probabilities.
                "main_temporal_prob",
                "main_temporal_logits",
                "calibrated_temporal_prob",
                "calibrated_temporal_logits",
                "raw_refined_temporal_prob",
                "raw_refined_temporal_logits",
                "composed_temporal_prob",
                "composed_temporal_logits",

                # Routing/calibration.
                "stable_refiner_enabled",
                "uncertainty_calibration_enabled",
                "minimal_sufficient_enabled",
                "all_span_refine_marginal_enabled",
                "refine_hard_coarse_route",
                "refine_coarse_detached",
                "refine_selected_span_index",
                "calibration_uncertainty",
                "calibration_information",
                "calibration_confidence",
                "calibration_confidence_detached",
                "calibration_reliability",
                "refiner_information_gain",
                "refined_semantic_retention",
                "base_refiner_confidence",
                "reliability_gain_factor",
                "posterior_guided_refinement_enabled",
                "boundary_structured_refinement_enabled",
                "prior_preserving_bounded_refinement_enabled",
                "posterior_stable_reliability_enabled",
                "refine_boundary_anchor_power",
                "refine_correction_cap",

                # BAPGR Composer pair prior / V1.10.2 MAIN pair posterior.
                "composer_pair_prior",
                "composer_uniform_pair_prior",
                "composer_endpoint_pair_prior",
                "composer_pair_prior_valid_mask",
                "composer_pair_prior_entropy",
                "composer_pair_prior_entropy_ratio",
                "composer_pair_prior_effective_count",
                "composer_pair_prior_top1_posterior",
                "composer_pair_prior_candidate_count",
                "composer_uniform_pair_prior_entropy",
                "composer_uniform_pair_prior_entropy_ratio",
                "composer_uniform_pair_prior_effective_count",
                "composer_uniform_pair_prior_top1_posterior",
                "composer_pair_prior_expected_span_length",
                "composer_uniform_pair_prior_expected_span_length",
                "composer_pair_prior_length_gain",
                "composer_pair_soft_support",
                "composer_expected_start_frame",
                "composer_expected_end_frame",
                "composer_expected_span_length",
                "main_pair_posterior",
                "main_pair_entropy",
                "main_pair_effective_count",
                "main_pair_top1_posterior",

                # Pair posterior.
                "refine_selected_pair_posterior",
                "refine_selected_pair_entropy",
                "refine_selected_pair_effective_count",
                "refine_selected_pair_top1_posterior",
                "refine_selected_pair_candidate_count",
                "refine_selected_pair_entropy_ratio",

                # V1.10.2 PPBR diagnostics.
                "refiner_raw_correction",
                "refiner_bounded_correction",
                "refiner_raw_correction_abs_mean",
                "refiner_raw_correction_abs_max",
                "refiner_bounded_correction_abs_mean",
                "refiner_bounded_correction_abs_max",
                "refiner_correction_saturation_ratio",
                "refiner_vs_prior_kl",
                "refiner_vs_prior_js",
                "raw_vs_prior_expected_length_ratio",

                # Calibrated hard interval.
                "calibrated_span_start_frame",
                "calibrated_span_end_frame",
                "calibrated_span_frame_length",
                "calibrated_span_fraction",
                "calibrated_retained_fraction",
                "calibrated_inside_coarse",

                # Coarse/raw-refined intervals.
                "coarse_span_start_frame",
                "coarse_span_end_frame",
                "coarse_span_frame_length",
                "coarse_span_fraction",
                "refined_span_start_frame",
                "refined_span_end_frame",
                "refined_span_frame_length",
                "refined_span_fraction",
                "refined_inside_coarse",

                # DECOMPOSE/RELATE/COMPOSE.
                "event_count",
                "event_assignment",
                "event_mass",
                "soft_effective_event_count",
                "decomp_loss",
                "reasoned_event_tokens",
                "span_posterior",
                "selected_span_index",
                "selected_event_mask",
                "selected_reasoned_event_tokens",

                # Decoder controls.
                "qbound_handover_enabled",
                "qbound_uncertainty_calibration_enabled",
                "qbound_posterior_guided_refinement_enabled",
                "qbound_boundary_structured_refinement_enabled",
                "qbound_prior_preserving_bounded_refinement_enabled",
                "qbound_posterior_stable_reliability_enabled",
                "qbound_minimal_sufficient_enabled",
                "qbound_direct_reasoning_residual_enabled",
                "merge_counterfactual_available",
            }

            missing_aux_keys = sorted(
                required_aux_keys.difference(qbound_aux)
            )
            if missing_aux_keys:
                raise KeyError(
                    "Q-Bound V1.11 auxiliary output is missing: "
                    + ", ".join(missing_aux_keys)
                )

            def _scalar_flag(name):
                value = qbound_aux[name]
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"Q-Bound flag {name!r} must be a tensor."
                    )
                return float(value.detach().item())

            # ----------------------------------------------------------
            # V1.11 generic-input contract. Q-Bound must consume the raw /
            # generic feature stream passed beside CRA.VideoEmbedding output.
            # There is intentionally no fallback to the CRA-embedded stream.
            # ----------------------------------------------------------
            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "CRA/RefineDecoder/QBoundCore version mismatch: "
                    f"CRA={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )

            if _scalar_flag("qbound_input_is_generic") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 must consume the explicit generic visual "
                    "stream rather than CRA.VideoEmbedding output."
                )
            if _scalar_flag("qbound_input_uses_cra_embedding") != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.11 unexpectedly reports CRA-embedded input."
                )
            if _scalar_flag("generic_input_context_reuse") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 requires the adapted generic stream to be "
                    "reused as contextual input in the controlled first run."
                )

            expected_generic_adapter = bool(
                self.cfgs["model"].get("qbound", {}).get(
                    "use_generic_input_adapter",
                    True,
                )
            )
            actual_generic_adapter = bool(
                round(_scalar_flag("generic_input_adapter_enabled"))
            )
            if actual_generic_adapter != expected_generic_adapter:
                raise RuntimeError(
                    "Q-Bound V1.11 generic input adapter flag mismatch: "
                    f"config={expected_generic_adapter}, "
                    f"core={actual_generic_adapter}."
                )

            if _scalar_flag("uncertainty_calibration_enabled") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 retains calibrated coarse/fine fusion from V1.10.2."
                )

            if _scalar_flag("posterior_guided_refinement_enabled") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 retains detached posterior-guided REFINE from V1.10.2."
                )

            if _scalar_flag("boundary_structured_refinement_enabled") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 retains boundary-anchored "
                    "posterior-guided refinement."
                )

            if _scalar_flag("qbound_boundary_structured_refinement_enabled") != 1.0:
                raise RuntimeError(
                    "RefineDecoder must expose the active V1.11 "
                    "boundary-structured refinement contract."
                )

            if _scalar_flag("prior_preserving_bounded_refinement_enabled") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 retains prior-preserving "
                    "bounded refinement (PPBR)."
                )

            if _scalar_flag(
                "qbound_prior_preserving_bounded_refinement_enabled"
            ) != 1.0:
                raise RuntimeError(
                    "RefineDecoder must expose the active V1.11 "
                    "PPBR contract."
                )

            if _scalar_flag("posterior_stable_reliability_enabled") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 retains posterior-stable reliability calibration."
                )

            refinement_cfg = self.cfgs["model"]["qbound"].get(
                "refinement",
                {},
            )
            if not isinstance(refinement_cfg, dict):
                raise TypeError(
                    "model.qbound.refinement must be a mapping/dict."
                )
            expected_anchor_power = float(
                refinement_cfg.get(
                    "boundary_anchor_power",
                    self.cfgs["model"]["qbound"].get(
                        "refiner_boundary_anchor_power",
                        2.0,
                    ),
                )
            )
            actual_anchor_power = float(
                qbound_aux["refine_boundary_anchor_power"]
                .detach()
                .item()
            )
            if not math.isclose(
                actual_anchor_power,
                expected_anchor_power,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Q-Bound V1.11 boundary anchor power mismatch: "
                    f"config={expected_anchor_power}, "
                    f"core={actual_anchor_power}."
                )

            expected_correction_cap = float(
                refinement_cfg.get(
                    "correction_cap",
                    self.cfgs["model"]["qbound"].get(
                        "refiner_correction_cap",
                        2.0,
                    ),
                )
            )
            actual_correction_cap = float(
                qbound_aux["refine_correction_cap"]
                .detach()
                .item()
            )
            if not math.isclose(
                actual_correction_cap,
                expected_correction_cap,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Q-Bound V1.11 correction cap mismatch: "
                    f"config={expected_correction_cap}, "
                    f"core={actual_correction_cap}."
                )

            if _scalar_flag("minimal_sufficient_enabled") != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.11 requires V1.7 minimal-sufficient routing "
                    "to remain disabled."
                )

            if _scalar_flag("all_span_refine_marginal_enabled") != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.11 requires V1.6 differentiable all-span REFINE "
                    "marginalization to remain disabled."
                )

            if _scalar_flag("refine_coarse_detached") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 requires the Composer posterior routed into "
                    "REFINE to remain detached."
                )

            if _scalar_flag("refine_hard_coarse_route") != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.11 must not restrict REFINE to the V1.8 hard "
                    "top-1 Composer chain."
                )

            if _scalar_flag("calibration_confidence_detached") != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.11 requires detached reliability calibration."
                )

        # --------------------------------------------------------------
        # Collect shared outputs.
        # --------------------------------------------------------------
        def _tp(name):
            if not isinstance(time_param, dict):
                return None
            return time_param.get(name)

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
            "qbound_mix": (
                time_param.get("qbound_mix")
                if isinstance(time_param, dict)
                else None
            ),
            "qbound_handover_enabled": (
                time_param.get("qbound_handover_enabled")
                if isinstance(time_param, dict)
                else None
            ),
            "qbound_version": self.qbound_version,

            "gpeb_active": use_qbound,
        }

        common_output_names = [
            "qbound_grounding_feature",
            "qbound_temporal_main_feature",
            "qbound_reasoned_video_feature",

            "qbound_event_visual_tokens",
            "qbound_evidence_visual_token",
            "qbound_event_valid_mask",
            "qbound_frame_semantic_logits",
            "qbound_event_evidence_logits",
            "qbound_event_evidence_weights",
            "qbound_event_evidence_effective_count",
            "qbound_multi_event_evidence",
            "qbound_top1_event_evidence_share",
            "qbound_top2_event_evidence_share",

            "qbound_event_tokens",
            "qbound_reasoned_event_tokens",
            "qbound_reasoned_event_repr",
            "qbound_ordered_relation_gates",
            "qbound_ordered_relation_valid_mask",
            "qbound_ordered_update_gate",
            "qbound_reasoned_event_token_delta_norm",

            "qbound_event_assignment",
            "qbound_event_mass",
            "qbound_event_mass_max",
            "qbound_event_mass_min",
            "qbound_event_mass_mean",
            "qbound_effective_event_count",
            "qbound_soft_effective_event_count",

            "qbound_decomp_loss",
            "qbound_decomp_expected_energy",
            "qbound_decomp_selected_coherence",
            "qbound_decomp_selected_complexity",
            "qbound_decomp_selected_complexity_cost",
            "qbound_decomp_selected_boundary_reward",
            "qbound_decomp_selected_boundary_term",
            "qbound_decomp_selected_total_energy",
            "qbound_decomp_expected_event_count",
            "qbound_partition_entropy",
            "qbound_candidate_split_gain",
            "qbound_accepted_split_gain",
            "qbound_rejected_split_gain",

            "qbound_boundary_prob",
            "qbound_boundary_hard",
            "qbound_boundary_soft_marginal",
            "qbound_boundary_candidate_indices",
            "qbound_boundary_candidate_probs",
            "qbound_boundary_candidate_valid_mask",

            "qbound_reasoning_residual",
            "qbound_reasoning_residual_used",
            "qbound_reasoning_scale",
            "qbound_direct_reasoning_residual_enabled",

            "qbound_event_relevance_weights",
            "qbound_event_relevance_gates",
            "qbound_event_relevance_effective_count",
        ]

        for name in common_output_names:
            outs[name] = _tp(name)

        qbound_output_names = [
            # V1.11 generic-input / counterfactual interface.
            "qbound_generic_input_adapter_enabled",
            "qbound_generic_input_position_enabled",
            "qbound_generic_input_context_reuse",
            "qbound_generic_input_raw_rms",
            "qbound_generic_input_adapted_rms",
            "qbound_generic_input_token_cosine",
            "qbound_generic_input_delta_ratio",
            "qbound_generic_context_raw_rms",
            "qbound_generic_context_adapted_rms",
            "qbound_counterfactual_evidence_interface_enabled",
            "qbound_counterfactual_keep_support",
            "qbound_counterfactual_remove_support",
            "qbound_counterfactual_keep_fraction",
            "qbound_counterfactual_remove_fraction",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",

            # COMPOSE.
            "qbound_span_candidate_event_mask",
            "qbound_span_candidate_start_event",
            "qbound_span_candidate_end_event",
            "qbound_span_candidate_event_count",
            "qbound_span_candidate_valid_mask",
            "qbound_span_candidate_frame_mask",
            "qbound_span_candidate_frame_length",
            "qbound_span_candidate_center",
            "qbound_span_candidate_duration",
            "qbound_span_event_evidence_mean",
            "qbound_span_frame_evidence_mean",
            "qbound_span_learned_logits",
            "qbound_span_evidence_prior",
            "qbound_span_logits",
            "qbound_span_posterior",
            "qbound_span_entropy",
            "qbound_span_effective_count",
            "qbound_span_top1_posterior",
            "qbound_soft_span_support",
            "qbound_span_support_log_bias",
            "qbound_composed_temporal_logits",
            "qbound_composed_temporal_prob",

            # Selected event-chain / future LLM interface.
            "qbound_selected_span_index",
            "qbound_selected_event_mask",
            "qbound_selected_reasoned_event_tokens",
            "qbound_selected_frame_mask",
            "qbound_selected_span_start_event",
            "qbound_selected_span_end_event",
            "qbound_selected_span_event_count",
            "qbound_selected_span_start_frame",
            "qbound_selected_span_end_frame",
            "qbound_selected_span_frame_length",
            "qbound_selected_span_fraction",
            "qbound_selected_span_posterior",
            "qbound_selected_span_temporal_prob",
            "qbound_soft_selected_event_weights",
            "qbound_composed_reasoned_event_repr",

            # Routing/calibration flags.
            "qbound_refiner_enabled",
            "qbound_stable_refiner_enabled",
            "qbound_uncertainty_calibration_enabled",
            "qbound_minimal_sufficient_enabled",
            "qbound_all_span_refine_marginal_enabled",
            "qbound_refine_hard_coarse_route",
            "qbound_refine_coarse_detached",
            "qbound_refine_selected_span_index",

            # Calibration values.
            "qbound_calibration_uncertainty",
            "qbound_calibration_information",
            "qbound_calibration_confidence",
            "qbound_calibration_confidence_detached",
            "qbound_calibration_reliability",
            "qbound_refiner_information_gain",
            "qbound_refined_semantic_retention",
            "qbound_base_refiner_confidence",
            "qbound_reliability_gain_factor",

            # V1.10.2 BAPGR + PPBR prior / MAIN pair posterior.
            "qbound_posterior_guided_refinement_enabled",
            "qbound_boundary_structured_refinement_enabled",
            "qbound_prior_preserving_bounded_refinement_enabled",
            "qbound_posterior_stable_reliability_enabled",
            "qbound_refine_boundary_anchor_power",
            "qbound_refine_correction_cap",
            "qbound_composer_pair_prior",
            "qbound_composer_endpoint_pair_prior",
            "qbound_composer_pair_prior_valid_mask",
            "qbound_composer_pair_prior_entropy",
            "qbound_composer_pair_prior_entropy_ratio",
            "qbound_composer_pair_prior_effective_count",
            "qbound_composer_pair_prior_top1_posterior",
            "qbound_composer_pair_prior_candidate_count",
            "qbound_composer_uniform_pair_prior_entropy",
            "qbound_composer_uniform_pair_prior_entropy_ratio",
            "qbound_composer_uniform_pair_prior_effective_count",
            "qbound_composer_uniform_pair_prior_top1_posterior",
            "qbound_composer_pair_prior_expected_span_length",
            "qbound_composer_uniform_pair_prior_expected_span_length",
            "qbound_composer_pair_prior_length_gain",
            "qbound_composer_pair_soft_support",
            "qbound_composer_expected_start_frame",
            "qbound_composer_expected_end_frame",
            "qbound_composer_expected_span_length",
            "qbound_main_pair_posterior",
            "qbound_main_pair_entropy",
            "qbound_main_pair_effective_count",
            "qbound_main_pair_top1_posterior",

            # Pair posterior.
            "qbound_refine_start_logits",
            "qbound_refine_end_logits",
            "qbound_refine_start_candidate_mask",
            "qbound_refine_end_candidate_mask",
            "qbound_refine_selected_pair_valid_mask",
            "qbound_refine_selected_pair_posterior",
            "qbound_refine_selected_pair_entropy",
            "qbound_refine_selected_pair_effective_count",
            "qbound_refine_selected_pair_top1_posterior",
            "qbound_refine_selected_pair_candidate_count",
            "qbound_refine_selected_pair_entropy_ratio",
            "qbound_refine_selected_start_candidate_count",
            "qbound_refine_selected_end_candidate_count",

            # V1.10.2 PPBR diagnostics.
            "qbound_refiner_raw_correction",
            "qbound_refiner_bounded_correction",
            "qbound_refiner_raw_correction_abs_mean",
            "qbound_refiner_raw_correction_abs_max",
            "qbound_refiner_bounded_correction_abs_mean",
            "qbound_refiner_bounded_correction_abs_max",
            "qbound_refiner_correction_saturation_ratio",
            "qbound_refiner_vs_prior_kl",
            "qbound_refiner_vs_prior_js",

            # Raw fine proposal statistics.
            "qbound_expected_refined_span_length",
            "qbound_raw_vs_prior_expected_length_ratio",
            "qbound_expected_refined_length_ratio",
            "qbound_expected_refined_span_fraction",
            "qbound_expected_refine_trim_fraction",
            "qbound_expected_refined_start_frame",
            "qbound_expected_refined_end_frame",

            "qbound_refine_conditional_support",
            "qbound_refined_soft_support",
            "qbound_refined_support_log_bias",
            "qbound_refined_temporal_logits",
            "qbound_refined_temporal_prob",
            "qbound_raw_refined_temporal_logits",
            "qbound_raw_refined_temporal_prob",

            # Official calibrated temporal support.
            "qbound_calibrated_soft_support",
            "qbound_calibrated_temporal_logits",
            "qbound_calibrated_temporal_prob",
            "qbound_main_temporal_logits",
            "qbound_main_temporal_prob",

            # Coarse core interval.
            "qbound_coarse_span_start_frame",
            "qbound_coarse_span_end_frame",
            "qbound_coarse_span_frame_length",
            "qbound_coarse_span_fraction",
            "qbound_coarse_frame_mask",

            # Raw-refined core interval.
            "qbound_refined_span_start_frame",
            "qbound_refined_span_end_frame",
            "qbound_refined_frame_mask",
            "qbound_refined_span_frame_length",
            "qbound_refined_span_fraction",
            "qbound_refined_span_start",
            "qbound_refined_span_end",
            "qbound_refined_span_temporal_prob",
            "qbound_refine_trim_left_frames",
            "qbound_refine_trim_right_frames",
            "qbound_refine_trim_total_frames",
            "qbound_refine_retained_fraction",
            "qbound_refined_inside_coarse",

            # Calibrated core interval.
            "qbound_calibrated_start_frame_float",
            "qbound_calibrated_end_frame_float",
            "qbound_calibrated_expected_span_length",
            "qbound_calibrated_expected_length_ratio",
            "qbound_calibrated_span_start_frame",
            "qbound_calibrated_span_end_frame",
            "qbound_calibrated_frame_mask",
            "qbound_calibrated_span_frame_length",
            "qbound_calibrated_span_fraction",
            "qbound_calibrated_retained_fraction",
            "qbound_calibrated_trim_left_frames",
            "qbound_calibrated_trim_right_frames",
            "qbound_calibrated_trim_total_frames",
            "qbound_calibrated_span_start",
            "qbound_calibrated_span_end",
            "qbound_calibrated_span_temporal_prob",
            "qbound_calibrated_inside_coarse",
            "qbound_refiner_temperature",

            # Decoder-level calibrated branch.
            "qbound_calibrated_grounding_feature",
            "qbound_calibrated_video_proj",
            "qbound_calibrated_key",
            "qbound_calibrated_pre_smooth_key",
            "qbound_calibrated_post_smooth_key",
            "qbound_calibrated_start",
            "qbound_calibrated_end",
            "qbound_calibrated_mask",
            "qbound_calibrated_span_length",
            "qbound_calibrated_span_fraction",
            "qbound_calibrated_span_key_mass",

            # Decoder-level raw-refined branch.
            "qbound_raw_refined_grounding_feature",
            "qbound_raw_refined_video_proj",
            "qbound_raw_refined_key",
            "qbound_raw_refined_pre_smooth_key",
            "qbound_raw_refined_post_smooth_key",
            "qbound_raw_refined_start",
            "qbound_raw_refined_end",
            "qbound_raw_refined_mask",
            "qbound_raw_refined_span_length",
            "qbound_raw_refined_span_fraction",
            "qbound_raw_refined_span_key_mass",

            # Decoder-level coarse branch.
            "qbound_coarse_grounding_feature",
            "qbound_coarse_video_proj",
            "qbound_coarse_key",
            "qbound_coarse_pre_smooth_key",
            "qbound_coarse_post_smooth_key",
            "qbound_coarse_start",
            "qbound_coarse_end",
            "qbound_coarse_mask",
            "qbound_coarse_span_length",
            "qbound_coarse_span_fraction",
            "qbound_coarse_span_key_mass",

            # Compatibility: adaptive/refined == calibrated MAIN.
            "qbound_adaptive_span_enabled",
            "qbound_refined_span_enabled",
            "qbound_calibrated_span_enabled",
            "qbound_raw_refined_span_enabled",
            "qbound_adaptive_span_start_frame",
            "qbound_adaptive_span_end_frame",
            "qbound_adaptive_span_length",
            "qbound_adaptive_span_fraction",
            "qbound_adaptive_span_key_mass",
            "qbound_refined_span_length",
            "qbound_refined_span_key_mass",

            # Same calibrated key + alternate readouts.
            "qbound_coarse_v15_start",
            "qbound_coarse_v15_end",
            "qbound_coarse_v15_mask",
            "qbound_coarse_v15_span_length",
            "qbound_coarse_v15_span_fraction",
            "qbound_coarse_v15_span_key_mass",
            "qbound_legacy_fixed_window_start",
            "qbound_legacy_fixed_window_end",
            "qbound_legacy_fixed_window_mask",
            "qbound_legacy_fixed_window_span_length",
            "qbound_legacy_fixed_window_key_mass",
            "qbound_refined_vs_coarse_span_length_ratio",
            "qbound_refined_vs_legacy_span_length_ratio",
            "qbound_refined_minus_coarse_key_mass",
            "qbound_refined_minus_legacy_key_mass",
            "qbound_refined_trimmed_frame_count",

            # Historical full readouts.
            "qbound_v15_diagnostic_key",
            "qbound_v15_diagnostic_start",
            "qbound_v15_diagnostic_end",
            "qbound_v15_diagnostic_mask",
            "qbound_v15_diagnostic_span_length",
            "qbound_v14_diagnostic_key",
            "qbound_v14_diagnostic_start",
            "qbound_v14_diagnostic_end",
            "qbound_v14_diagnostic_mask",
            "qbound_v14_diagnostic_span_length",
        ]

        for name in qbound_output_names:
            outs[name] = _tp(name)

        # --------------------------------------------------------------
        # Explicit V1.11 answer/key interfaces.
        # --------------------------------------------------------------
        if isinstance(qbound_aux, dict):
            # V1.11 generic-input / counterfactual interface.
            outs["qbound_input_is_generic"] = qbound_aux.get(
                "qbound_input_is_generic"
            )
            outs["qbound_input_uses_cra_embedding"] = qbound_aux.get(
                "qbound_input_uses_cra_embedding"
            )
            outs["qbound_generic_input_adapter_enabled"] = qbound_aux.get(
                "generic_input_adapter_enabled"
            )
            outs["qbound_generic_input_position_enabled"] = qbound_aux.get(
                "generic_input_position_enabled"
            )
            outs["qbound_generic_input_context_reuse"] = qbound_aux.get(
                "generic_input_context_reuse"
            )
            outs["qbound_generic_input_raw_rms"] = qbound_aux.get(
                "generic_input_raw_rms"
            )
            outs["qbound_generic_input_adapted_rms"] = qbound_aux.get(
                "generic_input_adapted_rms"
            )
            outs["qbound_generic_input_token_cosine"] = qbound_aux.get(
                "generic_input_token_cosine"
            )
            outs["qbound_generic_input_delta_ratio"] = qbound_aux.get(
                "generic_input_delta_ratio"
            )
            outs["qbound_counterfactual_evidence_interface_enabled"] = (
                qbound_aux.get("counterfactual_evidence_interface_enabled")
            )
            outs["qbound_counterfactual_keep_support"] = qbound_aux.get(
                "counterfactual_keep_support"
            )
            outs["qbound_counterfactual_remove_support"] = qbound_aux.get(
                "counterfactual_remove_support"
            )
            outs["qbound_counterfactual_keep_fraction"] = qbound_aux.get(
                "counterfactual_keep_fraction"
            )
            outs["qbound_counterfactual_remove_fraction"] = qbound_aux.get(
                "counterfactual_remove_fraction"
            )

            # Official MAIN.
            outs["qbound_calibrated_video_proj"] = qbound_aux.get(
                "qbound_calibrated_video_proj"
            )
            outs["qbound_calibrated_key"] = qbound_aux.get(
                "qbound_calibrated_key"
            )

            # Compatibility: "refined" == calibrated MAIN.
            outs["qbound_refined_video_proj"] = qbound_aux.get(
                "qbound_refined_video_proj"
            )
            outs["qbound_refined_key"] = qbound_aux.get(
                "qbound_key"
            )

            # Diagnostics.
            outs["qbound_raw_refined_video_proj"] = qbound_aux.get(
                "qbound_raw_refined_video_proj"
            )
            outs["qbound_raw_refined_key"] = qbound_aux.get(
                "qbound_raw_refined_key"
            )
            outs["qbound_coarse_video_proj"] = qbound_aux.get(
                "qbound_coarse_video_proj"
            )
            outs["qbound_coarse_key"] = qbound_aux.get(
                "qbound_coarse_key"
            )

            # Calibration.
            outs["qbound_calibration_uncertainty"] = qbound_aux.get(
                "calibration_uncertainty"
            )
            outs["qbound_calibration_information"] = qbound_aux.get(
                "calibration_information"
            )
            outs["qbound_calibration_confidence"] = qbound_aux.get(
                "calibration_confidence"
            )
            # V1.10.2 reliability diagnostics. Legacy calibration_confidence is
            # retained as a compatibility alias of calibration_reliability.
            outs["qbound_calibration_reliability"] = qbound_aux.get(
                "calibration_reliability"
            )
            outs["qbound_refiner_information_gain"] = qbound_aux.get(
                "refiner_information_gain"
            )
            outs["qbound_refined_semantic_retention"] = qbound_aux.get(
                "refined_semantic_retention"
            )
            outs["qbound_base_refiner_confidence"] = qbound_aux.get(
                "base_refiner_confidence"
            )
            outs["qbound_reliability_gain_factor"] = qbound_aux.get(
                "reliability_gain_factor"
            )

            # Posterior-guided route diagnostics.
            outs["qbound_posterior_guided_refinement_enabled"] = qbound_aux.get(
                "posterior_guided_refinement_enabled"
            )
            outs["qbound_boundary_structured_refinement_enabled"] = qbound_aux.get(
                "boundary_structured_refinement_enabled"
            )
            outs[
                "qbound_prior_preserving_bounded_refinement_enabled"
            ] = qbound_aux.get(
                "prior_preserving_bounded_refinement_enabled"
            )
            outs["qbound_refine_boundary_anchor_power"] = qbound_aux.get(
                "refine_boundary_anchor_power"
            )
            outs["qbound_refine_correction_cap"] = qbound_aux.get(
                "refine_correction_cap"
            )
            outs["qbound_posterior_stable_reliability_enabled"] = qbound_aux.get(
                "posterior_stable_reliability_enabled"
            )
            outs["qbound_composer_pair_prior_entropy"] = qbound_aux.get(
                "composer_pair_prior_entropy"
            )
            outs["qbound_composer_pair_prior_entropy_ratio"] = qbound_aux.get(
                "composer_pair_prior_entropy_ratio"
            )
            outs["qbound_composer_pair_prior_effective_count"] = qbound_aux.get(
                "composer_pair_prior_effective_count"
            )
            outs["qbound_composer_pair_prior_top1_posterior"] = qbound_aux.get(
                "composer_pair_prior_top1_posterior"
            )
            outs["qbound_composer_uniform_pair_prior_entropy"] = qbound_aux.get(
                "composer_uniform_pair_prior_entropy"
            )
            outs["qbound_composer_uniform_pair_prior_entropy_ratio"] = qbound_aux.get(
                "composer_uniform_pair_prior_entropy_ratio"
            )
            outs["qbound_composer_uniform_pair_prior_effective_count"] = qbound_aux.get(
                "composer_uniform_pair_prior_effective_count"
            )
            outs["qbound_composer_uniform_pair_prior_top1_posterior"] = qbound_aux.get(
                "composer_uniform_pair_prior_top1_posterior"
            )
            outs["qbound_composer_pair_prior_expected_span_length"] = qbound_aux.get(
                "composer_pair_prior_expected_span_length"
            )
            outs[
                "qbound_composer_uniform_pair_prior_expected_span_length"
            ] = qbound_aux.get(
                "composer_uniform_pair_prior_expected_span_length"
            )
            outs["qbound_composer_pair_prior_length_gain"] = qbound_aux.get(
                "composer_pair_prior_length_gain"
            )
            outs["qbound_composer_expected_span_length"] = qbound_aux.get(
                "composer_expected_span_length"
            )
            outs["qbound_main_pair_entropy"] = qbound_aux.get(
                "main_pair_entropy"
            )
            outs["qbound_main_pair_effective_count"] = qbound_aux.get(
                "main_pair_effective_count"
            )
            outs["qbound_main_pair_top1_posterior"] = qbound_aux.get(
                "main_pair_top1_posterior"
            )

            # V1.10.2 PPBR diagnostics.
            outs["qbound_refiner_raw_correction_abs_mean"] = qbound_aux.get(
                "refiner_raw_correction_abs_mean"
            )
            outs["qbound_refiner_raw_correction_abs_max"] = qbound_aux.get(
                "refiner_raw_correction_abs_max"
            )
            outs["qbound_refiner_bounded_correction_abs_mean"] = qbound_aux.get(
                "refiner_bounded_correction_abs_mean"
            )
            outs["qbound_refiner_bounded_correction_abs_max"] = qbound_aux.get(
                "refiner_bounded_correction_abs_max"
            )
            outs["qbound_refiner_correction_saturation_ratio"] = qbound_aux.get(
                "refiner_correction_saturation_ratio"
            )
            outs["qbound_refiner_vs_prior_kl"] = qbound_aux.get(
                "refiner_vs_prior_kl"
            )
            outs["qbound_refiner_vs_prior_js"] = qbound_aux.get(
                "refiner_vs_prior_js"
            )
            outs["qbound_raw_vs_prior_expected_length_ratio"] = qbound_aux.get(
                "raw_vs_prior_expected_length_ratio"
            )

            # Raw fine proposal statistics.
            outs["qbound_expected_refined_span_length"] = qbound_aux.get(
                "expected_refined_span_length"
            )
            outs["qbound_expected_refined_length_ratio"] = qbound_aux.get(
                "expected_refined_length_ratio"
            )
            outs["qbound_expected_refined_span_fraction"] = qbound_aux.get(
                "expected_refined_span_fraction"
            )
            outs["qbound_expected_refine_trim_fraction"] = qbound_aux.get(
                "expected_refine_trim_fraction"
            )

            # Official V1.11 calibrated interval.
            outs["qbound_calibrated_start"] = qbound_aux.get(
                "qbound_calibrated_start"
            )
            outs["qbound_calibrated_end"] = qbound_aux.get(
                "qbound_calibrated_end"
            )
            outs["qbound_calibrated_mask"] = qbound_aux.get(
                "qbound_calibrated_mask"
            )

            # Raw-refined interval.
            outs["qbound_raw_refined_start"] = qbound_aux.get(
                "qbound_raw_refined_start"
            )
            outs["qbound_raw_refined_end"] = qbound_aux.get(
                "qbound_raw_refined_end"
            )
            outs["qbound_raw_refined_mask"] = qbound_aux.get(
                "qbound_raw_refined_mask"
            )

            # Compatibility aliases point to calibrated MAIN.
            outs["qbound_refined_start"] = qbound_aux.get(
                "qbound_refined_start"
            )
            outs["qbound_refined_end"] = qbound_aux.get(
                "qbound_refined_end"
            )
            outs["qbound_refined_mask"] = qbound_aux.get(
                "qbound_refined_mask"
            )
            outs["qbound_adaptive_start"] = qbound_aux.get(
                "qbound_adaptive_start"
            )
            outs["qbound_adaptive_end"] = qbound_aux.get(
                "qbound_adaptive_end"
            )
            outs["qbound_adaptive_mask"] = qbound_aux.get(
                "qbound_adaptive_mask"
            )

        else:
            for name in (
                "qbound_input_is_generic",
                "qbound_input_uses_cra_embedding",
                "qbound_generic_input_adapter_enabled",
                "qbound_generic_input_position_enabled",
                "qbound_generic_input_context_reuse",
                "qbound_generic_input_raw_rms",
                "qbound_generic_input_adapted_rms",
                "qbound_generic_input_token_cosine",
                "qbound_generic_input_delta_ratio",
                "qbound_counterfactual_evidence_interface_enabled",
                "qbound_counterfactual_keep_support",
                "qbound_counterfactual_remove_support",
                "qbound_counterfactual_keep_fraction",
                "qbound_counterfactual_remove_fraction",
                "qbound_calibrated_video_proj",
                "qbound_calibrated_key",
                "qbound_refined_video_proj",
                "qbound_refined_key",
                "qbound_raw_refined_video_proj",
                "qbound_raw_refined_key",
                "qbound_coarse_video_proj",
                "qbound_coarse_key",
                "qbound_calibration_uncertainty",
                "qbound_calibration_information",
                "qbound_calibration_confidence",
                "qbound_calibration_reliability",
                "qbound_refiner_information_gain",
                "qbound_refined_semantic_retention",
                "qbound_base_refiner_confidence",
                "qbound_reliability_gain_factor",
                "qbound_posterior_guided_refinement_enabled",
                "qbound_boundary_structured_refinement_enabled",
                "qbound_prior_preserving_bounded_refinement_enabled",
                "qbound_refine_boundary_anchor_power",
                "qbound_refine_correction_cap",
                "qbound_posterior_stable_reliability_enabled",
                "qbound_composer_pair_prior_entropy",
                "qbound_composer_pair_prior_entropy_ratio",
                "qbound_composer_pair_prior_effective_count",
                "qbound_composer_pair_prior_top1_posterior",
                "qbound_composer_uniform_pair_prior_entropy",
                "qbound_composer_uniform_pair_prior_entropy_ratio",
                "qbound_composer_uniform_pair_prior_effective_count",
                "qbound_composer_uniform_pair_prior_top1_posterior",
                "qbound_composer_pair_prior_expected_span_length",
                "qbound_composer_uniform_pair_prior_expected_span_length",
                "qbound_composer_pair_prior_length_gain",
                "qbound_composer_expected_span_length",
                "qbound_main_pair_entropy",
                "qbound_main_pair_effective_count",
                "qbound_main_pair_top1_posterior",
                "qbound_refiner_raw_correction_abs_mean",
                "qbound_refiner_raw_correction_abs_max",
                "qbound_refiner_bounded_correction_abs_mean",
                "qbound_refiner_bounded_correction_abs_max",
                "qbound_refiner_correction_saturation_ratio",
                "qbound_refiner_vs_prior_kl",
                "qbound_refiner_vs_prior_js",
                "qbound_raw_vs_prior_expected_length_ratio",
                "qbound_expected_refined_span_length",
                "qbound_expected_refined_length_ratio",
                "qbound_expected_refined_span_fraction",
                "qbound_expected_refine_trim_fraction",
                "qbound_calibrated_start",
                "qbound_calibrated_end",
                "qbound_calibrated_mask",
                "qbound_raw_refined_start",
                "qbound_raw_refined_end",
                "qbound_raw_refined_mask",
                "qbound_refined_start",
                "qbound_refined_end",
                "qbound_refined_mask",
                "qbound_adaptive_start",
                "qbound_adaptive_end",
                "qbound_adaptive_mask",
            ):
                outs[name] = None

        return video_proj, answer_proj, outs

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
        # Common call signature with refine_forward.
        del gsub, question, qbound_mix

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask,
            device,
        )

        answer_g, answer_tokens = self._encode_answers(
            answer,
            device,
        )

        answer_indices = self._prepare_answer_indices(
            answer_id,
            answer_g.size(0),
            answer_g.device,
        )

        video_f = self.video_model(video)

        if answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
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
                        torch.arange(
                            0,
                            index,
                            device=x_pos.device,
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
        """Compute InfoNCE with one positive and K negatives."""
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