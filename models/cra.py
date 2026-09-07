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
    CRA wrapper with Q-Bound Version 1.6.

    Q-Bound V1.6 follows:

        DECOMPOSE -> RELATE -> COMPOSE -> REFINE -> GROUND

    The wrapper deliberately keeps CRA as a downstream testbed while Q-Bound
    owns the temporal reasoning process.

    Main responsibilities
    ---------------------
    * preserve CRA's original candidate-conditioned VQ/VQA language path;
    * encode a separate question-only representation for Q-Bound;
    * keep answer-choice semantics out of Q-Bound topology/composition;
    * pass raw video features + question-only semantics to RefineDecoder;
    * preserve CRA's original candidate dot-product answer interface;
    * expose V1.6 decomposition, Q-OEI, latent event-span composition,
      frame-level boundary refinement, grounding, and selected event tokens
      to the pipeline.

    V1.6 controls
    -------------
    * no CRA -> Q-Bound temporal handover;
    * no direct reasoned-event residual into answer space;
    * DECOMPOSE and Q-OEI remain the V1.4 mechanisms;
    * COMPOSE scores contiguous chains of reasoned events;
    * REFINE learns frame-level start/end inside the first/last event of the
      selected coarse chain without modifying event topology;
    * main start/end/mask come from the refined frame interval, not directly
      from coarse event boundaries and not from CRA's fixed [1,3,5] search;
    * the complete V1.5 coarse readout and V1.4 fixed-window readout are
      retained only as detached diagnostics/ablations;
    * selected ordered event tokens are surfaced for a future frozen
      LLM/MLLM adapter;
    * temporal GT is never consumed here for training.
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
                    model_cfg.get("gpeb_version", "1.6"),
                ),
            )
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
        Refined CRA/Q-Bound V1.6 forward.

        ``question`` is required by default when Q-Bound is active because
        decomposition and ordered reasoning must remain question-only.

        ``qbound_mix`` is accepted only for compatibility with older pipeline
        callers. RefineDecoder V1.6 ignores the old handover schedule and uses
        the pure Q-Bound temporal path whenever Q-Bound is active.
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
            # Preserve original VQ behavior.
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
            qsn_gt = qsn_g
        else:
            # Preserve the original CRA candidate-conditioned context.
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
                        "Q-Bound V1.6 requires question-only token ids. "
                        "Pass them through CRA.forward(..., question=...). "
                        "Do not reuse candidate answer features because that "
                        "would leak answer-choice semantics into decomposition."
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
            # Compatibility only. RefineDecoder V1.6 ignores handover mixing.
            qbound_mix=(
                qbound_mix
                if use_qbound
                else 0.0
            ),
        )

        # Keep the repository's original answer-scoring interface.
        answer_proj = answer_g

        qbound_aux = (
            time_param.get("qbound_aux")
            if isinstance(time_param, dict)
            else None
        )

        # --------------------------------------------------------------
        # Validate V1.6 decoder/core/refiner interface early.
        # --------------------------------------------------------------
        if use_qbound:
            required_time_keys = {
                # Main / pure-QBound grounded features.
                "qbound_grounding_feature",
                "qbound_temporal_main_feature",
                "qbound_reasoned_video_feature",

                # Generic visual/event evidence.
                "qbound_event_visual_tokens",
                "qbound_evidence_visual_token",
                "qbound_event_valid_mask",
                "qbound_frame_semantic_logits",

                # Event evidence retained from V1.4.
                "qbound_event_evidence_logits",
                "qbound_event_evidence_weights",
                "qbound_event_evidence_effective_count",
                "qbound_multi_event_evidence",
                "qbound_top1_event_evidence_share",
                "qbound_top2_event_evidence_share",

                # Ordered reasoning.
                "qbound_event_tokens",
                "qbound_reasoned_event_tokens",
                "qbound_reasoned_event_repr",
                "qbound_ordered_relation_gates",
                "qbound_ordered_relation_valid_mask",
                "qbound_ordered_update_gate",
                "qbound_reasoned_event_token_delta_norm",

                # Latent decomposition.
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

                # Proposal diagnostics.
                "qbound_boundary_prob",
                "qbound_boundary_hard",
                "qbound_boundary_soft_marginal",
                "qbound_boundary_candidate_indices",
                "qbound_boundary_candidate_probs",
                "qbound_boundary_candidate_valid_mask",

                # Explicit V1.4 shortcut removal.
                "qbound_reasoning_residual",
                "qbound_reasoning_residual_used",
                "qbound_reasoning_scale",
                "qbound_direct_reasoning_residual_enabled",
            }

            required_time_keys.update({
                # V1.5 COMPOSE retained inside V1.6.
                "qbound_adaptive_span_enabled",
                "qbound_span_posterior",
                "qbound_span_entropy",
                "qbound_span_effective_count",
                "qbound_span_top1_posterior",
                "qbound_selected_event_mask",
                "qbound_selected_reasoned_event_tokens",
                "qbound_selected_frame_mask",
                "qbound_selected_span_start_frame",
                "qbound_selected_span_end_frame",
                "qbound_selected_span_event_count",
                "qbound_selected_span_frame_length",
                "qbound_selected_span_fraction",
                "qbound_selected_span_posterior",
                "qbound_soft_selected_event_weights",
                "qbound_composed_reasoned_event_repr",
                "qbound_composed_temporal_prob",

                # V1.6 REFINE main path.
                "qbound_refined_span_enabled",
                "qbound_refiner_enabled",
                "qbound_refine_start_logits",
                "qbound_refine_end_logits",
                "qbound_refine_start_candidate_mask",
                "qbound_refine_end_candidate_mask",
                "qbound_refined_temporal_prob",
                "qbound_refine_selected_pair_entropy",
                "qbound_refine_selected_pair_effective_count",
                "qbound_refine_selected_pair_top1_posterior",
                "qbound_refined_span_start_frame",
                "qbound_refined_span_end_frame",
                "qbound_refined_span_length",
                "qbound_refined_span_fraction",
                "qbound_refined_span_key_mass",
                "qbound_refined_span_mask_consistency",
                "qbound_refine_trim_left_frames",
                "qbound_refine_trim_right_frames",
                "qbound_refine_trim_total_frames",
                "qbound_refine_retained_fraction",

                # Coarse V1.5 and fixed-window diagnostics on V1.6 key.
                "qbound_coarse_v15_span_length",
                "qbound_coarse_v15_span_fraction",
                "qbound_coarse_v15_span_key_mass",
                "qbound_legacy_fixed_window_span_length",
                "qbound_legacy_fixed_window_key_mass",
                "qbound_refined_vs_coarse_span_length_ratio",
                "qbound_refined_vs_legacy_span_length_ratio",
                "qbound_refined_minus_coarse_key_mass",
                "qbound_refined_minus_legacy_key_mass",

                # Full detached historical readouts.
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
            })

            missing_time_keys = sorted(
                required_time_keys.difference(time_param)
            )
            if missing_time_keys:
                raise KeyError(
                    "RefineDecoder/Q-Bound V1.6 time_param/refiner output is missing: "
                    + ", ".join(missing_time_keys)
                )

            if qbound_aux is None:
                raise KeyError(
                    "RefineDecoder/Q-Bound V1.6 must expose "
                    "time_param['qbound_aux']."
                )

            required_aux_keys = {
                # Main temporal output.
                "event_temporal_prob",
                "event_temporal_logits",

                # Dynamic decomposition.
                "boundary_hard",
                "boundary_count",
                "event_count",
                "event_assignment",
                "event_mass",
                "event_mass_max",
                "event_mass_min",
                "soft_effective_event_count",
                "decomp_loss",
                "decomp_selected_coherence",
                "decomp_selected_complexity",
                "decomp_selected_total_energy",
                "decomp_expected_event_count",
                "partition_entropy",
                "accepted_split_gain",
                "rejected_split_gain",

                # Event evidence.
                "event_visual_tokens",
                "evidence_visual_token",
                "event_evidence_logits",
                "event_evidence_weights",
                "event_evidence_effective_count",
                "multi_event_evidence",

                # Native ordered reasoning.
                "event_tokens",
                "reasoned_event_tokens",
                "reasoned_event_repr",
                "ordered_relation_gates",
                "ordered_relation_valid_mask",
                "ordered_update_gate",
                "reasoned_event_token_delta_norm",

                # Decoder marks the absence of V1.3 main merge diagnostic.
                "merge_counterfactual_available",
            }

            required_aux_keys.update({
                # V1.6 main temporal outputs.
                "main_temporal_prob",
                "main_temporal_logits",
                "refined_temporal_prob",
                "refined_temporal_logits",

                # V1.5/V1.4 controlled temporal outputs.
                "legacy_v15_composed_temporal_prob",
                "composed_temporal_prob",
                "composed_temporal_logits",
                "legacy_v14_event_temporal_prob",

                # V1.5 COMPOSE.
                "span_candidate_event_mask",
                "span_candidate_valid_mask",
                "span_candidate_frame_mask",
                "span_logits",
                "span_posterior",
                "span_entropy",
                "span_effective_count",
                "span_top1_posterior",
                "soft_span_support",
                "selected_span_index",
                "selected_event_mask",
                "selected_reasoned_event_tokens",
                "selected_frame_mask",
                "selected_span_start_event",
                "selected_span_end_event",
                "selected_span_event_count",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_length",
                "selected_span_fraction",
                "selected_span_posterior",
                "soft_selected_event_weights",
                "composed_reasoned_event_repr",

                # V1.6 REFINE core.
                "refiner_enabled",
                "refine_start_logits",
                "refine_end_logits",
                "refine_start_candidate_mask",
                "refine_end_candidate_mask",
                "refined_soft_support",
                "refine_selected_pair_entropy",
                "refine_selected_pair_effective_count",
                "refine_selected_pair_top1_posterior",
                "coarse_span_start_frame",
                "coarse_span_end_frame",
                "coarse_span_frame_length",
                "coarse_span_fraction",
                "coarse_frame_mask",
                "refined_span_start_frame",
                "refined_span_end_frame",
                "refined_frame_mask",
                "refined_span_frame_length",
                "refined_span_fraction",
                "refine_trim_left_frames",
                "refine_trim_right_frames",
                "refine_trim_total_frames",
                "refine_retained_fraction",
                "refined_inside_coarse",

                # Refine/Grounding V1.6 adapter outputs.
                "qbound_adaptive_span_enabled",
                "qbound_refined_span_enabled",
                "qbound_adaptive_start",
                "qbound_adaptive_end",
                "qbound_adaptive_mask",
                "qbound_refined_start",
                "qbound_refined_end",
                "qbound_refined_mask",
                "qbound_refined_span_length",
                "qbound_refined_span_fraction",
                "qbound_refined_span_key_mass",
                "qbound_coarse_v15_span_length",
                "qbound_coarse_v15_span_fraction",
                "qbound_coarse_v15_span_key_mass",
                "qbound_legacy_fixed_window_span_length",
                "qbound_refined_vs_coarse_span_length_ratio",
                "qbound_refined_vs_legacy_span_length_ratio",
                "v15_diagnostic_key",
                "v15_diagnostic_start",
                "v15_diagnostic_end",
                "v15_diagnostic_mask",
                "v14_diagnostic_key",
                "v14_diagnostic_start",
                "v14_diagnostic_end",
                "v14_diagnostic_mask",
            })

            missing_aux_keys = sorted(
                required_aux_keys.difference(qbound_aux)
            )
            if missing_aux_keys:
                raise KeyError(
                    "Q-Bound V1.6 auxiliary/refiner output is missing: "
                    + ", ".join(missing_aux_keys)
                )

        # --------------------------------------------------------------
        # Collect pure-QBound / event evidence outputs.
        # --------------------------------------------------------------
        def _tp(name):
            if not isinstance(time_param, dict):
                return None
            return time_param.get(name)

        qbound_grounding_feature = _tp(
            "qbound_grounding_feature"
        )
        qbound_event_visual_tokens = _tp(
            "qbound_event_visual_tokens"
        )
        qbound_evidence_visual_token = _tp(
            "qbound_evidence_visual_token"
        )
        qbound_event_valid_mask = _tp(
            "qbound_event_valid_mask"
        )
        qbound_frame_semantic_logits = _tp(
            "qbound_frame_semantic_logits"
        )

        qbound_event_evidence_logits = _tp(
            "qbound_event_evidence_logits"
        )
        qbound_event_evidence_weights = _tp(
            "qbound_event_evidence_weights"
        )
        qbound_event_evidence_effective_count = _tp(
            "qbound_event_evidence_effective_count"
        )
        qbound_multi_event_evidence = _tp(
            "qbound_multi_event_evidence"
        )
        qbound_top1_event_evidence_share = _tp(
            "qbound_top1_event_evidence_share"
        )
        qbound_top2_event_evidence_share = _tp(
            "qbound_top2_event_evidence_share"
        )

        # --------------------------------------------------------------
        # Ordered event reasoning outputs.
        # --------------------------------------------------------------
        qbound_event_tokens = _tp("qbound_event_tokens")
        qbound_reasoned_event_tokens = _tp(
            "qbound_reasoned_event_tokens"
        )
        qbound_reasoned_event_repr = _tp(
            "qbound_reasoned_event_repr"
        )
        qbound_ordered_relation_gates = _tp(
            "qbound_ordered_relation_gates"
        )
        qbound_ordered_relation_valid_mask = _tp(
            "qbound_ordered_relation_valid_mask"
        )
        qbound_ordered_update_gate = _tp(
            "qbound_ordered_update_gate"
        )
        qbound_reasoned_event_token_delta_norm = _tp(
            "qbound_reasoned_event_token_delta_norm"
        )

        # --------------------------------------------------------------
        # Latent decomposition outputs.
        # --------------------------------------------------------------
        qbound_event_assignment = _tp(
            "qbound_event_assignment"
        )
        qbound_event_mass = _tp(
            "qbound_event_mass"
        )
        qbound_event_mass_max = _tp(
            "qbound_event_mass_max"
        )
        qbound_event_mass_min = _tp(
            "qbound_event_mass_min"
        )
        qbound_event_mass_mean = _tp(
            "qbound_event_mass_mean"
        )
        qbound_effective_event_count = _tp(
            "qbound_effective_event_count"
        )
        qbound_soft_effective_event_count = _tp(
            "qbound_soft_effective_event_count"
        )

        qbound_decomp_loss = _tp(
            "qbound_decomp_loss"
        )
        qbound_decomp_expected_energy = _tp(
            "qbound_decomp_expected_energy"
        )
        qbound_decomp_selected_coherence = _tp(
            "qbound_decomp_selected_coherence"
        )
        qbound_decomp_selected_complexity = _tp(
            "qbound_decomp_selected_complexity"
        )
        qbound_decomp_selected_complexity_cost = _tp(
            "qbound_decomp_selected_complexity_cost"
        )
        qbound_decomp_selected_boundary_reward = _tp(
            "qbound_decomp_selected_boundary_reward"
        )
        qbound_decomp_selected_boundary_term = _tp(
            "qbound_decomp_selected_boundary_term"
        )
        qbound_decomp_selected_total_energy = _tp(
            "qbound_decomp_selected_total_energy"
        )
        qbound_decomp_expected_event_count = _tp(
            "qbound_decomp_expected_event_count"
        )
        qbound_partition_entropy = _tp(
            "qbound_partition_entropy"
        )
        qbound_candidate_split_gain = _tp(
            "qbound_candidate_split_gain"
        )
        qbound_accepted_split_gain = _tp(
            "qbound_accepted_split_gain"
        )
        qbound_rejected_split_gain = _tp(
            "qbound_rejected_split_gain"
        )

        # Boundary proposal diagnostics.
        qbound_boundary_prob = _tp(
            "qbound_boundary_prob"
        )
        qbound_boundary_hard = _tp(
            "qbound_boundary_hard"
        )
        qbound_boundary_soft_marginal = _tp(
            "qbound_boundary_soft_marginal"
        )
        qbound_boundary_candidate_indices = _tp(
            "qbound_boundary_candidate_indices"
        )
        qbound_boundary_candidate_probs = _tp(
            "qbound_boundary_candidate_probs"
        )
        qbound_boundary_candidate_valid_mask = _tp(
            "qbound_boundary_candidate_valid_mask"
        )

        # Explicit shortcut-removal diagnostics.
        qbound_reasoning_residual = _tp(
            "qbound_reasoning_residual"
        )
        qbound_reasoning_residual_used = _tp(
            "qbound_reasoning_residual_used"
        )
        qbound_reasoning_scale = _tp(
            "qbound_reasoning_scale"
        )
        qbound_direct_reasoning_residual_enabled = _tp(
            "qbound_direct_reasoning_residual_enabled"
        )

        qbound_reasoned_video_feature = _tp(
            "qbound_reasoned_video_feature"
        )
        qbound_temporal_main_feature = _tp(
            "qbound_temporal_main_feature"
        )

        # Compatibility aliases for old loggers. These no longer define the
        # V1.4 main grounding mechanism.
        qbound_event_relevance_weights = _tp(
            "qbound_event_relevance_weights"
        )
        qbound_event_relevance_gates = _tp(
            "qbound_event_relevance_gates"
        )
        qbound_event_relevance_effective_count = _tp(
            "qbound_event_relevance_effective_count"
        )

        outs = {
            "fatt": video_attn,
            "time_param": time_param,

            # Original CRA alignment target (GT QA semantic feature when
            # available). It is never fed into Q-Bound decomposition.
            "align_gt": qsn_gt,
            "qa_condition": qsn_g,

            # Strict question-only Q-Bound condition.
            "question_condition": question_g,
            "question_tokens": question_tokens,
            "question_only_conditioning": torch.as_tensor(
                question_g is not None,
                device=video_proj.device,
                dtype=video_proj.dtype,
            ),

            # Weak-alignment targets/candidates.
            "qbound_align_question": question_g,
            "qbound_align_qa": qsn_gt,

            # Pure Q-Bound temporal / visual evidence.
            "qbound_grounding_feature": qbound_grounding_feature,
            "qbound_event_visual_tokens": qbound_event_visual_tokens,
            "qbound_evidence_visual_token": qbound_evidence_visual_token,
            "qbound_event_valid_mask": qbound_event_valid_mask,
            "qbound_frame_semantic_logits": qbound_frame_semantic_logits,

            # V1.4 event evidence.
            "qbound_event_evidence_logits": qbound_event_evidence_logits,
            "qbound_event_evidence_weights": qbound_event_evidence_weights,
            "qbound_event_evidence_effective_count": (
                qbound_event_evidence_effective_count
            ),
            "qbound_multi_event_evidence": qbound_multi_event_evidence,
            "qbound_top1_event_evidence_share": (
                qbound_top1_event_evidence_share
            ),
            "qbound_top2_event_evidence_share": (
                qbound_top2_event_evidence_share
            ),

            # Ordered event reasoning retained from V1.3.
            "qbound_event_tokens": qbound_event_tokens,
            "qbound_reasoned_event_tokens": qbound_reasoned_event_tokens,
            "qbound_reasoned_event_repr": qbound_reasoned_event_repr,
            "qbound_ordered_relation_gates": qbound_ordered_relation_gates,
            "qbound_ordered_relation_valid_mask": (
                qbound_ordered_relation_valid_mask
            ),
            "qbound_ordered_update_gate": qbound_ordered_update_gate,
            "qbound_reasoned_event_token_delta_norm": (
                qbound_reasoned_event_token_delta_norm
            ),

            # Latent decomposition.
            "qbound_event_assignment": qbound_event_assignment,
            "qbound_event_mass": qbound_event_mass,
            "qbound_event_mass_max": qbound_event_mass_max,
            "qbound_event_mass_min": qbound_event_mass_min,
            "qbound_event_mass_mean": qbound_event_mass_mean,
            "qbound_effective_event_count": qbound_effective_event_count,
            "qbound_soft_effective_event_count": (
                qbound_soft_effective_event_count
            ),
            "qbound_decomp_loss": qbound_decomp_loss,
            "qbound_decomp_expected_energy": qbound_decomp_expected_energy,
            "qbound_decomp_selected_coherence": (
                qbound_decomp_selected_coherence
            ),
            "qbound_decomp_selected_complexity": (
                qbound_decomp_selected_complexity
            ),
            "qbound_decomp_selected_complexity_cost": (
                qbound_decomp_selected_complexity_cost
            ),
            "qbound_decomp_selected_boundary_reward": (
                qbound_decomp_selected_boundary_reward
            ),
            "qbound_decomp_selected_boundary_term": (
                qbound_decomp_selected_boundary_term
            ),
            "qbound_decomp_selected_total_energy": (
                qbound_decomp_selected_total_energy
            ),
            "qbound_decomp_expected_event_count": (
                qbound_decomp_expected_event_count
            ),
            "qbound_partition_entropy": qbound_partition_entropy,
            "qbound_candidate_split_gain": qbound_candidate_split_gain,
            "qbound_accepted_split_gain": qbound_accepted_split_gain,
            "qbound_rejected_split_gain": qbound_rejected_split_gain,

            # Proposal diagnostics.
            "qbound_boundary_prob": qbound_boundary_prob,
            "qbound_boundary_hard": qbound_boundary_hard,
            "qbound_boundary_soft_marginal": qbound_boundary_soft_marginal,
            "qbound_boundary_candidate_indices": (
                qbound_boundary_candidate_indices
            ),
            "qbound_boundary_candidate_probs": (
                qbound_boundary_candidate_probs
            ),
            "qbound_boundary_candidate_valid_mask": (
                qbound_boundary_candidate_valid_mask
            ),

            # Grounding-mediated main-path semantics.
            "qbound_reasoned_video_feature": qbound_reasoned_video_feature,
            "qbound_temporal_main_feature": qbound_temporal_main_feature,

            # These must remain zero/disabled in V1.6. They remain only to keep
            # old diagnostics from crashing during the pipeline transition.
            "qbound_reasoning_residual": qbound_reasoning_residual,
            "qbound_reasoning_residual_used": qbound_reasoning_residual_used,
            "qbound_reasoning_scale": qbound_reasoning_scale,
            "qbound_direct_reasoning_residual_enabled": (
                qbound_direct_reasoning_residual_enabled
            ),

            # Old V1.3 relevance aliases, diagnostic only.
            "qbound_event_relevance_weights": (
                qbound_event_relevance_weights
            ),
            "qbound_event_relevance_gates": (
                qbound_event_relevance_gates
            ),
            "qbound_event_relevance_effective_count": (
                qbound_event_relevance_effective_count
            ),

            # Complete generic diagnostic package.
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

            # Temporary compatibility alias.
            "gpeb_active": use_qbound,
        }

        # --------------------------------------------------------------
        # Q-Bound V1.6 temporal hierarchy outputs.
        #
        # CRA remains a thin wrapper. All DECOMPOSE / RELATE / COMPOSE /
        # REFINE logic lives in QBoundCore / RefineDecoder.
        # --------------------------------------------------------------
        qbound_output_names = [
            # V1.5 Composer candidates/posterior retained in V1.6.
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

            # Hard coarse event-chain / future LLM interface.
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

            # V1.6 frame-level REFINE core outputs.
            "qbound_refiner_enabled",
            "qbound_refine_start_logits",
            "qbound_refine_end_logits",
            "qbound_refine_start_candidate_mask",
            "qbound_refine_end_candidate_mask",
            "qbound_refine_conditional_support",
            "qbound_refined_soft_support",
            "qbound_refined_support_log_bias",
            "qbound_refined_temporal_logits",
            "qbound_refined_temporal_prob",
            "qbound_main_temporal_logits",
            "qbound_main_temporal_prob",
            "qbound_refine_expected_pair_entropy",
            "qbound_refine_selected_pair_posterior",
            "qbound_refine_selected_pair_entropy",
            "qbound_refine_selected_pair_effective_count",
            "qbound_refine_selected_pair_top1_posterior",
            "qbound_refine_selected_start_candidate_count",
            "qbound_refine_selected_end_candidate_count",

            # Core coarse/fine interval comparison.
            "qbound_coarse_span_start_frame",
            "qbound_coarse_span_end_frame",
            "qbound_coarse_span_frame_length",
            "qbound_coarse_span_fraction",
            "qbound_coarse_frame_mask",
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
            "qbound_refiner_temperature",

            # Main grounding adapter: compatibility adaptive aliases now
            # correspond to the refined V1.6 interval.
            "qbound_adaptive_span_enabled",
            "qbound_refined_span_enabled",
            "qbound_adaptive_span_start_frame",
            "qbound_adaptive_span_end_frame",
            "qbound_adaptive_span_length",
            "qbound_adaptive_span_fraction",
            "qbound_adaptive_span_key_mass",
            "qbound_adaptive_span_peak_in_span",
            "qbound_adaptive_span_mask_consistency",

            # Explicit refined interval diagnostics.
            "qbound_refined_span_length",
            "qbound_refined_span_key_mass",
            "qbound_refined_span_peak_in_span",
            "qbound_refined_span_mask_consistency",

            # Coarse V1.5 span evaluated on the SAME V1.6 key.
            "qbound_coarse_v15_start",
            "qbound_coarse_v15_end",
            "qbound_coarse_v15_mask",
            "qbound_coarse_v15_max_indices",
            "qbound_coarse_v15_span_start_frame",
            "qbound_coarse_v15_span_end_frame",
            "qbound_coarse_v15_span_length",
            "qbound_coarse_v15_span_fraction",
            "qbound_coarse_v15_span_key_mass",
            "qbound_coarse_v15_span_peak_in_span",
            "qbound_coarse_v15_span_mask_consistency",

            # Same V1.6 key with legacy CRA fixed-window extraction.
            "qbound_legacy_fixed_window_start",
            "qbound_legacy_fixed_window_end",
            "qbound_legacy_fixed_window_mask",
            "qbound_legacy_fixed_window_span_length",
            "qbound_legacy_fixed_window_key_mass",

            # Direct V1.6 coarse-to-fine diagnostics.
            "qbound_refined_vs_coarse_span_length_ratio",
            "qbound_refined_vs_legacy_span_length_ratio",
            "qbound_coarse_vs_legacy_span_length_ratio",
            "qbound_refined_minus_coarse_key_mass",
            "qbound_refined_minus_legacy_key_mass",
            "qbound_refined_retained_fraction_of_coarse",
            "qbound_refined_trimmed_frame_count",

            # Full detached V1.5 readout.
            "qbound_v15_diagnostic_key",
            "qbound_v15_diagnostic_start",
            "qbound_v15_diagnostic_end",
            "qbound_v15_diagnostic_mask",
            "qbound_v15_diagnostic_span_length",

            # Full detached V1.4 readout.
            "qbound_v14_diagnostic_key",
            "qbound_v14_diagnostic_start",
            "qbound_v14_diagnostic_end",
            "qbound_v14_diagnostic_mask",
            "qbound_v14_diagnostic_span_length",
        ]

        for name in qbound_output_names:
            outs[name] = _tp(name)

        # Explicit final V1.6 interval used by evaluation.
        outs["qbound_refined_start"] = (
            qbound_aux.get("qbound_refined_start")
            if isinstance(qbound_aux, dict)
            else None
        )
        outs["qbound_refined_end"] = (
            qbound_aux.get("qbound_refined_end")
            if isinstance(qbound_aux, dict)
            else None
        )
        outs["qbound_refined_mask"] = (
            qbound_aux.get("qbound_refined_mask")
            if isinstance(qbound_aux, dict)
            else None
        )

        # Compatibility aliases. In V1.6 these point to the refined interval,
        # not to the coarse V1.5 Composer boundaries.
        outs["qbound_adaptive_start"] = (
            qbound_aux.get("qbound_adaptive_start")
            if isinstance(qbound_aux, dict)
            else None
        )
        outs["qbound_adaptive_end"] = (
            qbound_aux.get("qbound_adaptive_end")
            if isinstance(qbound_aux, dict)
            else None
        )
        outs["qbound_adaptive_mask"] = (
            qbound_aux.get("qbound_adaptive_mask")
            if isinstance(qbound_aux, dict)
            else None
        )

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
