from itertools import combinations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound Version 1.12 core.

    V1.12 is an architecture-level reset rather than another incremental
    Composer/Refiner patch.  It keeps the parts that remained stable across the
    V1.4--V1.11 experiments:

        generic visual input -> dynamic DECOMPOSE -> Q-OEI RELATE

    and replaces the historical

        COMPOSE -> BAPGR -> PPBR -> PSRC -> counterfactual donor

    decision chain with:

        exact temporal hypothesis bank
            -> question-only intra-video hypothesis scorer
            -> hard winning hypothesis
            -> temporal grounding

    Core scientific hypothesis
    --------------------------
    The persistent Q-Bound failure is not simply insufficient sharpening or
    span-length calibration.  The model lacks a strong mechanism for deciding
    which of several plausible temporal explanations *inside the same video*
    best explains the current question.  V1.12 therefore turns grounding into
    temporal-hypothesis discrimination instead of probability sharpening.

    Dynamic event contract
    ----------------------
    ``max_events`` is only an event-capacity ceiling K_max.  The selected
    partition may contain any K_i in [1, K_max] for each video/question pair.
    There is no event-count target and no requirement to fill the capacity.
    Events guide semantic reasoning, but final grounding boundaries are not
    constrained to event boundaries.

    Exact frame-level hypothesis bank
    ---------------------------------
    For the short sampled VideoQA sequence (e.g. T=32 in NExT-GQA), V1.12
    enumerates every valid contiguous frame interval [s,e].  Hence event
    boundary errors cannot impose an oracle ceiling on final temporal
    precision.  Event semantics still guide hypothesis scoring through the
    reasoned event field pooled inside each interval.

    For every hypothesis H_k the core exposes:
      * exact start/end frames and frame mask;
      * normalized selected-evidence and complement weights;
      * one-frame trim/expand neighbour indices;
      * a question-only hypothesis logit p_theta(H_k | V,Q).

    This interface is intentionally designed for a training-only privileged
    answer-aware critic in the downstream pipeline.  The core itself NEVER
    receives answer candidates or the ground-truth answer.  At inference,
    grounding depends only on V and Q.

    Answer-aware learning, answer-free grounding
    --------------------------------------------
    A later pipeline component may use frozen answer semantics during training
    to evaluate competing hypotheses (sufficiency, trim, expand, complement)
    and distil a soft/listwise ranking into ``hypothesis_logits``.  Teacher
    utilities must be detached.  This preserves a generic Q-Bound inference
    interface while exploiting answer supervision as privileged information.

    Important removals from the V1.12 main path
    --------------------------------------------
      * no latent Composer posterior as final span selector;
      * no BAPGR pair prior;
      * no PPBR correction cap/temperature;
      * no PSRC reliability fusion;
      * no donor counterfactual support;
      * no minimality/sparsity/duration target;
      * no temporal GT, IoU, or IoP training loss;
      * no answer features inside Q-Bound core.

    The full hypothesis bank is also exposed for an oracle-coverage diagnostic
    on validation data.  Ground-truth temporal spans remain evaluation-only.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        max_events: int = 6,
        use_temporal_prior: bool = False,
        dropout: float = 0.1,
        boundary_temperature: float = 0.7,
        boundary_threshold: float = 0.5,
        boundary_radius: int = 2,
        boundary_bias_init: float = -0.5,
        proposal_candidates: int = 8,
        decomp_complexity_weight: float = 1.25,
        decomp_boundary_weight: float = 0.50,
        decomp_temperature: float = 0.20,
        question_modulation_scale: float = 0.10,
        min_event_mass: float = 0.25,
        use_generic_input_adapter: bool = True,
        input_adapter_dropout: Optional[float] = None,
        input_adapter_use_position: bool = True,
    ) -> None:
        super().__init__()

        if max_events < 2:
            raise ValueError(
                "Version 1.12 requires max_events >= 2 as a capacity ceiling."
            )
        if proposal_candidates < 1:
            raise ValueError("proposal_candidates must be >= 1.")
        if proposal_candidates < max_events - 1:
            raise ValueError(
                "proposal_candidates must be >= max_events - 1 so all "
                "event counts up to the capacity can be represented."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if boundary_temperature <= 0.0:
            raise ValueError("boundary_temperature must be positive.")
        if not 0.0 < boundary_threshold < 1.0:
            raise ValueError("boundary_threshold must be in (0, 1).")
        if boundary_radius < 0:
            raise ValueError("boundary_radius must be >= 0.")
        if decomp_complexity_weight < 0.0:
            raise ValueError("decomp_complexity_weight must be >= 0.")
        if decomp_boundary_weight < 0.0:
            raise ValueError("decomp_boundary_weight must be >= 0.")
        if decomp_temperature <= 0.0:
            raise ValueError("decomp_temperature must be positive.")
        if not 0.0 < question_modulation_scale < 1.0:
            raise ValueError("question_modulation_scale must be in (0, 1).")
        if min_event_mass <= 0.0:
            raise ValueError("min_event_mass must be positive.")

        if input_adapter_dropout is None:
            input_adapter_dropout = dropout
        if not 0.0 <= float(input_adapter_dropout) < 1.0:
            raise ValueError("input_adapter_dropout must be in [0, 1).")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        # IMPORTANT: this is a capacity ceiling, not a forced event count.
        self.max_events = int(max_events)
        self.max_transitions = self.max_events - 1
        self.proposal_candidates = int(proposal_candidates)
        self.qbound_version = "1.12"

        self.use_generic_input_adapter = bool(use_generic_input_adapter)
        self.input_adapter_use_position = bool(input_adapter_use_position)
        self.input_adapter_dropout_p = float(input_adapter_dropout)
        self.use_temporal_prior = bool(use_temporal_prior)
        self.boundary_temperature = float(boundary_temperature)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_radius = int(boundary_radius)
        self.decomp_complexity_weight = float(decomp_complexity_weight)
        self.decomp_boundary_weight = float(decomp_boundary_weight)
        self.decomp_temperature = float(decomp_temperature)
        self.question_modulation_scale = float(question_modulation_scale)
        self.min_event_mass = float(min_event_mass)

        # ------------------------------------------------------------------
        # 0. GENERIC VISUAL INPUT ADAPTER -- retained from V1.11.
        # ------------------------------------------------------------------
        self.generic_input_proj = nn.Linear(input_dim, input_dim)
        self.generic_input_gelu = nn.GELU()
        self.generic_input_norm = nn.LayerNorm(input_dim, eps=1e-12)
        self.generic_input_dropout = nn.Dropout(self.input_adapter_dropout_p)

        if self.generic_input_proj.weight.shape[0] == self.generic_input_proj.weight.shape[1]:
            nn.init.eye_(self.generic_input_proj.weight)
        else:
            nn.init.xavier_uniform_(self.generic_input_proj.weight)
        nn.init.zeros_(self.generic_input_proj.bias)

        # ------------------------------------------------------------------
        # A. DYNAMIC TOPOLOGY / DECOMPOSITION.
        # ------------------------------------------------------------------
        self.decomp_question_gate = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.decomp_question_gate.weight)
        nn.init.zeros_(self.decomp_question_gate.bias)

        self.topology_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.topology_qa_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        boundary_input_dim = hidden_dim * 6 + 4
        self.boundary_feature_encoder = nn.Sequential(
            nn.Linear(boundary_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.boundary_classifier = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.boundary_classifier.weight, mean=0.0, std=2e-2)
        nn.init.constant_(self.boundary_classifier.bias, boundary_bias_init)

        # Enumerate every subset of proposal slots that yields at most K_max
        # events.  The selected split count is free to vary per sample.
        patterns = []
        for n_split in range(self.max_transitions + 1):
            for selected in combinations(range(self.proposal_candidates), n_split):
                row = torch.zeros(self.proposal_candidates, dtype=torch.bool)
                if selected:
                    row[list(selected)] = True
                patterns.append(row)
        partition_patterns = torch.stack(patterns, dim=0)
        self.register_buffer(
            "partition_patterns",
            partition_patterns,
            persistent=False,
        )
        self.register_buffer(
            "partition_split_count",
            partition_patterns.sum(dim=-1).to(torch.float32),
            persistent=False,
        )

        empty_idx = torch.nonzero(
            partition_patterns.sum(dim=-1) == 0,
            as_tuple=False,
        ).squeeze(-1)
        if empty_idx.numel() != 1:
            raise RuntimeError("Expected exactly one no-split partition pattern.")
        self.empty_partition_index = int(empty_idx.item())

        singleton_indices = []
        for slot in range(self.proposal_candidates):
            target = torch.zeros(self.proposal_candidates, dtype=torch.bool)
            target[slot] = True
            idx = torch.nonzero(
                torch.all(partition_patterns == target.unsqueeze(0), dim=-1),
                as_tuple=False,
            ).squeeze(-1)
            if idx.numel() != 1:
                raise RuntimeError("Could not locate singleton partition pattern.")
            singleton_indices.append(int(idx.item()))
        self.register_buffer(
            "singleton_partition_indices",
            torch.tensor(singleton_indices, dtype=torch.long),
            persistent=False,
        )

        # ------------------------------------------------------------------
        # B. SEMANTIC EVENT / LOCAL EVIDENCE BRANCH -- retained.
        # ------------------------------------------------------------------
        self.raw_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.qa_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.qa_film = nn.Linear(hidden_dim, hidden_dim * 2)

        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ------------------------------------------------------------------
        # C. RELATE: Q-OEI retained.
        # ------------------------------------------------------------------
        ordered_pair_dim = hidden_dim * 5 + 4
        self.ordered_relation_score = nn.Sequential(
            nn.Linear(ordered_pair_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(
            self.ordered_relation_score[-1].weight,
            mean=0.0,
            std=1e-2,
        )
        nn.init.zeros_(self.ordered_relation_score[-1].bias)

        self.ordered_value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.ordered_event_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ordered_update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(
            self.ordered_update_gate[-1].weight,
            mean=0.0,
            std=1e-2,
        )
        nn.init.constant_(self.ordered_update_gate[-1].bias, -1.5)
        self.ordered_event_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # D. LOCAL QUESTION-CONDITIONED EVIDENCE -- retained as features, not
        #    as the final selector.
        # ------------------------------------------------------------------
        self.event_evidence_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(
            self.event_evidence_score[-1].weight,
            mean=0.0,
            std=1e-2,
        )
        nn.init.zeros_(self.event_evidence_score[-1].bias)

        self.frame_evidence_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.frame_evidence_score = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.frame_evidence_score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.frame_evidence_score.bias)

        # ------------------------------------------------------------------
        # E. V1.12 QUESTION-ONLY TEMPORAL HYPOTHESIS SCORER.
        # ------------------------------------------------------------------
        # Hypothesis feature =
        #   start frame state
        #   end frame state
        #   pooled contextual state
        #   pooled reasoned-event field
        #   question
        #   pooled reasoned-event x question
        #   + center, duration, local evidence mean, frame evidence mean,
        #     left boundary evidence, right boundary evidence,
        #     touched-event ratio.
        hypothesis_input_dim = hidden_dim * 6 + 7
        self.hypothesis_feature_encoder = nn.Sequential(
            nn.Linear(hypothesis_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.hypothesis_score = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.hypothesis_score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.hypothesis_score.bias)

    @staticmethod
    def _sinusoidal_position_encoding(
        time_steps: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return deterministic [1,T,D] sinusoidal temporal positions."""
        position = torch.arange(
            time_steps,
            device=device,
            dtype=torch.float32,
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0,
                feature_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            * (-torch.log(torch.tensor(10000.0, device=device)) / feature_dim)
        )

        pe = torch.zeros(
            time_steps,
            feature_dim,
            device=device,
            dtype=torch.float32,
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if feature_dim > 1:
            pe[:, 1::2] = torch.cos(
                position * div_term[: pe[:, 1::2].shape[1]]
            )
        return pe.unsqueeze(0).to(dtype=dtype)

    def _adapt_generic_video_tokens(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Adapt generic frozen visual tokens into Q-Bound's private input space.

        This adapter is deliberately frame-wise.  Temporal reasoning remains
        the responsibility of DECOMPOSE/RELATE rather than a hidden CRA-style
        encoder in front of Q-Bound.
        """
        mask_f = video_mask.to(video_tokens.dtype).unsqueeze(-1)
        raw_masked = video_tokens * mask_f

        if self.use_generic_input_adapter:
            adapted = self.generic_input_proj(video_tokens)
            adapted = self.generic_input_gelu(adapted)
            adapted = self.generic_input_norm(adapted)

            if self.input_adapter_use_position:
                position = self._sinusoidal_position_encoding(
                    time_steps=video_tokens.size(1),
                    feature_dim=video_tokens.size(2),
                    device=video_tokens.device,
                    dtype=video_tokens.dtype,
                )
                adapted = adapted + position

            adapted = self.generic_input_dropout(adapted)
        else:
            adapted = video_tokens

        adapted = adapted * mask_f

        valid_count = video_mask.sum(dim=-1).to(video_tokens.dtype).clamp_min(1.0)
        raw_rms = torch.sqrt(
            raw_masked.pow(2).sum(dim=(1, 2))
            / (valid_count * float(video_tokens.size(-1))).clamp_min(1.0)
            + 1e-8
        )
        adapted_rms = torch.sqrt(
            adapted.pow(2).sum(dim=(1, 2))
            / (valid_count * float(adapted.size(-1))).clamp_min(1.0)
            + 1e-8
        )

        raw_unit = F.normalize(raw_masked, p=2, dim=-1)
        adapted_unit = F.normalize(adapted, p=2, dim=-1)
        token_cos = (raw_unit * adapted_unit).sum(dim=-1)
        token_cos = (
            token_cos * video_mask.to(token_cos.dtype)
        ).sum(dim=-1) / valid_count

        delta = (adapted - raw_masked) * mask_f
        delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        raw_norm = torch.linalg.vector_norm(raw_masked, dim=-1)
        delta_ratio = (
            delta_norm * video_mask.to(delta_norm.dtype)
        ).sum(dim=-1) / (
            (raw_norm * video_mask.to(raw_norm.dtype)).sum(dim=-1).clamp_min(1e-6)
        )

        diagnostics = {
            "generic_input_raw_rms": raw_rms,
            "generic_input_adapted_rms": adapted_rms,
            "generic_input_token_cosine": token_cos,
            "generic_input_delta_ratio": delta_ratio,
        }
        return adapted, diagnostics

    def _summarize_qa(self, qa_features: torch.Tensor) -> torch.Tensor:
        """Compatibility helper. Q-Bound receives question-only [B,D]."""
        if qa_features.ndim < 2:
            raise ValueError("qa_features must have at least two dimensions.")
        if qa_features.size(-1) != self.input_dim:
            raise ValueError(
                "The last QA dimension must equal input_dim. "
                f"Expected {self.input_dim}, got {qa_features.size(-1)}."
            )
        if qa_features.ndim == 2:
            return qa_features

        batch_size = qa_features.size(0)
        qa_features = qa_features.reshape(batch_size, -1, self.input_dim)
        return qa_features.mean(dim=1)

    @staticmethod
    def _normalize_mask(
        mask: Optional[torch.Tensor],
        batch_size: int,
        time_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(
                batch_size,
                time_steps,
                dtype=torch.bool,
                device=device,
            )

        while mask.ndim > 2 and mask.size(1) == 1:
            mask = mask.squeeze(1)

        if mask.ndim != 2:
            raise ValueError(
                "video_mask must reduce to [B,T], "
                f"got {tuple(mask.shape)}."
            )
        if mask.shape != (batch_size, time_steps):
            raise ValueError(
                "video_mask shape mismatch: expected "
                f"{(batch_size, time_steps)}, got {tuple(mask.shape)}."
            )
        return mask.bool()

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        mask = mask.bool()
        masked_logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * mask.to(weights.dtype)
        denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)
        return weights / denominator

    @staticmethod
    def _normalize_distribution(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        values = values * mask.to(values.dtype)
        denominator = values.sum(dim=dim, keepdim=True).clamp_min(1e-6)
        return values / denominator

    @staticmethod
    def _weighted_pool(
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """features [B,T,D], weights [B,T,K] -> [B,K,D]."""
        numerator = torch.einsum("btk,btd->bkd", weights, features)
        denominator = weights.sum(dim=1).unsqueeze(-1).clamp_min(1e-6)
        return numerator / denominator

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        numerator = (values * mask_f).sum(dim=dim)
        denominator = mask_f.sum(dim=dim).clamp_min(1.0)
        return numerator / denominator

    def _prepare_temporal_prior(
        self,
        temporal_prior: Optional[torch.Tensor],
        video_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return [B,T] detached prior features.

        Main V1.4 should use use_temporal_prior=False. This compatibility path
        never directly fixes the final partition.
        """
        batch_size, time_steps = video_mask.shape
        if (not self.use_temporal_prior) or temporal_prior is None:
            return torch.zeros(
                batch_size,
                time_steps,
                dtype=dtype,
                device=video_mask.device,
            )

        if temporal_prior.ndim == 3 and temporal_prior.size(-1) == 1:
            temporal_prior = temporal_prior.squeeze(-1)
        if temporal_prior.shape != (batch_size, time_steps):
            raise ValueError(
                "temporal_prior must have shape "
                f"[B,T], got {tuple(temporal_prior.shape)}."
            )

        prior = temporal_prior.detach().to(dtype=dtype)
        prior = torch.clamp(prior, min=0.0)
        prior = self._normalize_distribution(prior, video_mask, dim=-1)
        return prior

    # ----------------------------------------------------------------------
    # DECOMPOSE
    # ----------------------------------------------------------------------
    def _build_decomposition_features(
        self,
        video_tokens: torch.Tensor,
        qa_summary: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a bounded question-conditioned geometry for coherence measurement.

        The backbone token itself is detached: decomposition may select / reason
        over the pretrained geometry but cannot collapse the upstream backbone by
        minimizing within-event variance. The learnable question modulation is
        bounded to +/- question_modulation_scale per channel.
        """
        base = F.normalize(video_tokens.detach(), p=2, dim=-1)

        # Detach the external question embedding as well; this topology-specific
        # gate learns how to read question semantics without pushing the language
        # encoder toward a trivial decomposition solution.
        gate_raw = self.decomp_question_gate(qa_summary.detach())
        gate = 1.0 + self.question_modulation_scale * torch.tanh(gate_raw)
        decomp_features = F.normalize(base * gate.unsqueeze(1), p=2, dim=-1)
        decomp_features = (
            decomp_features * video_mask.to(decomp_features.dtype).unsqueeze(-1)
        )
        return decomp_features

    def _extract_boundary_proposals(
        self,
        boundary_prob: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Select up to proposal_candidates local maxima as *candidate* change points.

        Important V1.4 distinction:
        boundary_threshold is retained only as a diagnostic threshold. Candidate
        proposal generation does NOT require p >= threshold; otherwise the model
        can start with no candidates and the decomposition objective cannot decide
        whether a low-confidence split is structurally useful.
        """
        if boundary_prob.shape != boundary_mask.shape:
            raise ValueError("boundary_prob and boundary_mask must match.")

        boundary_mask = boundary_mask.bool()
        masked_prob = boundary_prob.masked_fill(~boundary_mask, -1e4)

        if self.boundary_radius > 0:
            kernel = 2 * self.boundary_radius + 1
            local_peak_value = F.max_pool1d(
                masked_prob.unsqueeze(1),
                kernel_size=kernel,
                stride=1,
                padding=self.boundary_radius,
            ).squeeze(1)
        else:
            local_peak_value = masked_prob

        local_peak_mask = (
            boundary_prob >= (local_peak_value - 1e-6)
        ) & boundary_mask
        threshold_mask = boundary_prob >= self.boundary_threshold
        threshold_candidate_mask = local_peak_mask & threshold_mask

        candidate_scores_full = boundary_prob.masked_fill(~local_peak_mask, -1e4)
        k = min(self.proposal_candidates, boundary_prob.size(-1))
        top_values, top_indices = torch.topk(candidate_scores_full, k=k, dim=-1)
        top_valid = top_values > -1e3

        # Sort candidates by temporal position. Q-OEI event slot order must follow
        # time, not proposal confidence rank.
        sorted_indices, sort_order = torch.sort(top_indices, dim=-1)
        sorted_values = torch.gather(top_values, 1, sort_order)
        sorted_valid = torch.gather(top_valid, 1, sort_order)

        batch_size = boundary_prob.size(0)
        if k < self.proposal_candidates:
            pad = self.proposal_candidates - k
            sorted_indices = F.pad(sorted_indices, (0, pad), value=0)
            sorted_values = F.pad(sorted_values, (0, pad), value=0.0)
            sorted_valid = F.pad(sorted_valid, (0, pad), value=False)

        candidate_probs = torch.gather(
            boundary_prob,
            1,
            sorted_indices.clamp(max=boundary_prob.size(-1) - 1),
        )
        candidate_probs = candidate_probs * sorted_valid.to(candidate_probs.dtype)

        proposal_indicator = F.one_hot(
            sorted_indices.clamp(max=boundary_prob.size(-1) - 1),
            num_classes=boundary_prob.size(-1),
        ).to(boundary_prob.dtype)
        proposal_indicator = (
            proposal_indicator * sorted_valid.to(boundary_prob.dtype).unsqueeze(-1)
        )

        return {
            "candidate_indices": sorted_indices,
            "candidate_probs": candidate_probs,
            "candidate_valid_mask": sorted_valid,
            "candidate_indicator": proposal_indicator,
            "local_peak_mask": local_peak_mask.detach(),
            "threshold_candidate_mask": threshold_candidate_mask.detach(),
            "local_peak_value": local_peak_value.detach(),
        }

    def _enumerate_partitions(
        self,
        candidate_indicator: torch.Tensor,
        candidate_probs: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
        decomp_features: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Enumerate latent contiguous partitions and compute their unified energy.

        Energy(P) = coherence(P)
                  + lambda_c * number_of_splits(P)
                  - lambda_b * selected_boundary_probability_sum(P)

        This single objective simultaneously discourages heterogeneous giant
        events and unnecessary over-segmentation while using the learned boundary
        scorer as change-point evidence rather than a hard topology oracle.
        """
        batch_size, time_steps, _ = decomp_features.shape
        patterns = self.partition_patterns.to(device=decomp_features.device)
        patterns_f = patterns.to(decomp_features.dtype)
        num_partitions = patterns.size(0)

        # A pattern is valid only when every selected proposal slot is valid.
        invalid_selected = patterns.unsqueeze(0) & (~candidate_valid_mask.unsqueeze(1))
        partition_valid_mask = ~invalid_selected.any(dim=-1)

        # [B,P,T-1] exact binary boundary map for every enumerated partition.
        partition_boundary = torch.einsum(
            "pc,bct->bpt",
            patterns_f,
            candidate_indicator,
        ).clamp(min=0.0, max=1.0)

        # Boundary at edge k separates frame k and k+1. Event index at frame t
        # equals number of selected boundaries before t.
        partition_event_coordinate = F.pad(
            torch.cumsum(partition_boundary, dim=-1),
            pad=(1, 0),
            value=0.0,
        )
        partition_event_index = (
            torch.round(partition_event_coordinate)
            .long()
            .clamp(min=0, max=self.max_events - 1)
        )
        partition_assignment = F.one_hot(
            partition_event_index,
            num_classes=self.max_events,
        ).to(decomp_features.dtype)
        partition_assignment = (
            partition_assignment
            * video_mask[:, None, :, None].to(decomp_features.dtype)
        )

        # Within-event SSE computed without materializing [B,P,T,K,D].
        event_mass = partition_assignment.sum(dim=2)  # [B,P,K]
        event_sum = torch.einsum(
            "bptk,btd->bpkd",
            partition_assignment,
            decomp_features,
        )
        event_mean = event_sum / event_mass.unsqueeze(-1).clamp_min(1e-6)

        token_sq_norm = decomp_features.pow(2).sum(dim=-1)
        event_sum_sq = torch.einsum(
            "bptk,bt->bpk",
            partition_assignment,
            token_sq_norm,
        )
        event_mean_sq = event_mean.pow(2).sum(dim=-1)
        event_sse = (
            event_sum_sq - event_mass * event_mean_sq
        ).clamp_min(0.0)

        valid_length = video_mask.sum(dim=-1).to(decomp_features.dtype).clamp_min(1.0)
        coherence = event_sse.sum(dim=-1) / valid_length.unsqueeze(1)

        split_count = self.partition_split_count.to(
            device=decomp_features.device,
            dtype=decomp_features.dtype,
        ).view(1, num_partitions).expand(batch_size, -1)

        selected_candidate_prob = (
            patterns_f.unsqueeze(0)
            * candidate_probs.unsqueeze(1)
            * candidate_valid_mask.unsqueeze(1).to(decomp_features.dtype)
        )
        boundary_reward = selected_candidate_prob.sum(dim=-1)

        # Normalize split costs by valid temporal length so their scale remains
        # comparable across videos. For unit-normalized features, an extra segment
        # can reduce empirical SSE by roughly O(1/T) even on unstructured data;
        # this length-normalized complexity term explicitly corrects that bias.
        length_scale = valid_length.unsqueeze(1)
        complexity_cost = (
            self.decomp_complexity_weight * split_count / length_scale
        )
        boundary_evidence_term = (
            self.decomp_boundary_weight * boundary_reward / length_scale
        )

        partition_energy = coherence + complexity_cost - boundary_evidence_term
        partition_energy_masked = partition_energy.masked_fill(
            ~partition_valid_mask,
            1e4,
        )

        posterior = self._masked_softmax(
            -partition_energy_masked / self.decomp_temperature,
            partition_valid_mask,
            dim=-1,
        )

        # EM-like expected energy: posterior is used as a soft assignment but is
        # detached in this loss term. This keeps optimization stable while still
        # providing gradients to coherence modulation and boundary probabilities.
        decomp_expected_energy = (
            posterior.detach() * partition_energy
        ).sum(dim=-1)

        best_partition_index = torch.argmin(
            partition_energy_masked.detach(),
            dim=-1,
        )

        batch_index = torch.arange(batch_size, device=decomp_features.device)
        hard_pattern = patterns[best_partition_index]
        hard_boundary = partition_boundary[batch_index, best_partition_index]
        hard_event_index = partition_event_index[batch_index, best_partition_index]
        hard_assignment = partition_assignment[batch_index, best_partition_index]

        # Crucial V1.4 control: downstream reasoning/grounding sees a hard,
        # contiguous topology but does not differentiate through its membership.
        hard_boundary = hard_boundary.detach()
        hard_event_index = hard_event_index.detach()
        hard_assignment = hard_assignment.detach()
        hard_pattern = hard_pattern.detach()

        selected_coherence = coherence[batch_index, best_partition_index]
        selected_split_count = split_count[batch_index, best_partition_index]
        selected_boundary_reward = boundary_reward[batch_index, best_partition_index]
        selected_complexity_cost = complexity_cost[batch_index, best_partition_index]
        selected_boundary_term = boundary_evidence_term[
            batch_index, best_partition_index
        ]
        selected_total_energy = partition_energy[
            batch_index, best_partition_index
        ]

        partition_entropy = -torch.sum(
            posterior * torch.log(posterior.clamp_min(1e-8)),
            dim=-1,
        )
        expected_event_count = (
            posterior * (split_count + 1.0)
        ).sum(dim=-1)

        # Soft boundary marginal is useful for diagnostics and decomposition
        # learning, but it is NOT used as downstream topology.
        boundary_soft_marginal = torch.einsum(
            "bp,bpt->bt",
            posterior,
            partition_boundary,
        )

        # Structural split-gain diagnostic: compare no split with each singleton
        # candidate split. This is intentionally diagnostic-only; the actual hard
        # partition may contain multiple interacting split decisions.
        no_split_coherence = coherence[:, self.empty_partition_index]
        singleton_coherence = coherence[:, self.singleton_partition_indices]
        candidate_split_gain = (
            no_split_coherence.unsqueeze(-1) - singleton_coherence
        )
        candidate_split_gain = (
            candidate_split_gain
            * candidate_valid_mask.to(candidate_split_gain.dtype)
        )

        accepted_mask = hard_pattern & candidate_valid_mask
        rejected_mask = (~hard_pattern) & candidate_valid_mask
        accepted_split_gain = self._masked_mean(
            candidate_split_gain,
            accepted_mask,
            dim=-1,
        )
        rejected_split_gain = self._masked_mean(
            candidate_split_gain,
            rejected_mask,
            dim=-1,
        )

        return {
            "partition_energy": partition_energy,
            "partition_valid_mask": partition_valid_mask,
            "partition_posterior": posterior,
            "partition_entropy": partition_entropy,
            "partition_boundary": partition_boundary,
            "partition_event_index": partition_event_index,
            "partition_assignment": partition_assignment,
            "best_partition_index": best_partition_index,
            "hard_pattern": hard_pattern,
            "hard_boundary": hard_boundary,
            "hard_event_index": hard_event_index,
            "hard_assignment": hard_assignment,
            "boundary_soft_marginal": boundary_soft_marginal,
            "decomp_loss": decomp_expected_energy,
            "decomp_expected_energy": decomp_expected_energy,
            "decomp_selected_coherence": selected_coherence,
            "decomp_selected_complexity": selected_split_count,
            "decomp_selected_complexity_cost": selected_complexity_cost,
            "decomp_selected_boundary_reward": selected_boundary_reward,
            "decomp_selected_boundary_term": selected_boundary_term,
            "decomp_selected_total_energy": selected_total_energy,
            "decomp_expected_event_count": expected_event_count,
            "candidate_split_gain": candidate_split_gain,
            "accepted_split_gain": accepted_split_gain,
            "rejected_split_gain": rejected_split_gain,
        }

    # ----------------------------------------------------------------------
    # RELATE
    # ----------------------------------------------------------------------
    def _ordered_event_interaction(
        self,
        event_tokens: torch.Tensor,
        event_valid_mask: torch.Tensor,
        qa_state: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Question-conditioned directed pairwise reasoning over ordered events."""
        batch_size, event_slots, hidden_dim = event_tokens.shape
        if event_slots != self.max_events or hidden_dim != self.hidden_dim:
            raise ValueError(
                "event_tokens must be [B,max_events,hidden_dim], got "
                f"{tuple(event_tokens.shape)}."
            )

        event_valid_mask = event_valid_mask.bool()

        receiver = event_tokens.unsqueeze(2).expand(-1, -1, event_slots, -1)
        source = event_tokens.unsqueeze(1).expand(-1, event_slots, -1, -1)
        qa_pair = qa_state[:, None, None, :].expand(
            -1, event_slots, event_slots, -1
        )

        slot = torch.arange(
            event_slots,
            dtype=event_tokens.dtype,
            device=event_tokens.device,
        )
        denom = float(max(event_slots - 1, 1))
        relative_order = (
            slot.view(1, event_slots) - slot.view(event_slots, 1)
        ) / denom
        abs_order = relative_order.abs()
        source_is_past = (relative_order < 0).to(event_tokens.dtype)
        source_is_future = (relative_order > 0).to(event_tokens.dtype)
        order_features = torch.stack(
            [
                relative_order,
                abs_order,
                source_is_past,
                source_is_future,
            ],
            dim=-1,
        ).unsqueeze(0).expand(batch_size, -1, -1, -1)

        pair_input = torch.cat(
            [
                receiver,
                source,
                source - receiver,
                receiver * source,
                qa_pair,
                order_features,
            ],
            dim=-1,
        )
        relation_logits_raw = self.ordered_relation_score(pair_input).squeeze(-1)

        identity = torch.eye(
            event_slots,
            dtype=torch.bool,
            device=event_tokens.device,
        ).unsqueeze(0)
        pair_valid_mask = (
            event_valid_mask.unsqueeze(2)
            & event_valid_mask.unsqueeze(1)
            & (~identity)
        )

        relation_gates = torch.sigmoid(relation_logits_raw)
        relation_gates = relation_gates * pair_valid_mask.to(relation_gates.dtype)

        source_values = self.ordered_value_proj(event_tokens)
        relation_context = torch.einsum(
            "bij,bjh->bih",
            relation_gates,
            source_values,
        )
        relation_denominator = relation_gates.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        relation_context = relation_context / relation_denominator

        qa_events = qa_state.unsqueeze(1).expand(-1, event_slots, -1)
        event_update = self.ordered_event_update(
            torch.cat(
                [
                    event_tokens,
                    relation_context,
                    event_tokens * relation_context,
                    qa_events,
                ],
                dim=-1,
            )
        )
        update_gate = torch.sigmoid(
            self.ordered_update_gate(
                torch.cat(
                    [event_tokens, relation_context, qa_events],
                    dim=-1,
                )
            ).squeeze(-1)
        )

        has_neighbor = pair_valid_mask.any(dim=-1)
        update_gate = (
            update_gate
            * has_neighbor.to(update_gate.dtype)
            * event_valid_mask.to(update_gate.dtype)
        )

        updated_event_tokens = self.ordered_event_norm(
            event_tokens + update_gate.unsqueeze(-1) * event_update
        )
        reasoned_event_tokens = torch.where(
            has_neighbor.unsqueeze(-1),
            updated_event_tokens,
            event_tokens,
        )
        reasoned_event_tokens = (
            reasoned_event_tokens
            * event_valid_mask.to(reasoned_event_tokens.dtype).unsqueeze(-1)
        )

        pair_count = pair_valid_mask.sum(dim=(1, 2)).clamp_min(1).to(
            relation_gates.dtype
        )
        relation_gate_mean = relation_gates.sum(dim=(1, 2)) / pair_count

        valid_event_count = event_valid_mask.sum(dim=-1).clamp_min(1).to(
            update_gate.dtype
        )
        update_gate_mean = update_gate.sum(dim=-1) / valid_event_count

        past_mask = (relative_order < 0).unsqueeze(0) & pair_valid_mask
        future_mask = (relative_order > 0).unsqueeze(0) & pair_valid_mask
        total_relation_mass = relation_gates.sum(dim=(1, 2)).clamp_min(1e-6)
        relation_past_mass = (
            relation_gates * past_mask.to(relation_gates.dtype)
        ).sum(dim=(1, 2)) / total_relation_mass
        relation_future_mass = (
            relation_gates * future_mask.to(relation_gates.dtype)
        ).sum(dim=(1, 2)) / total_relation_mass

        reasoned_delta = (
            reasoned_event_tokens - event_tokens
        ) * event_valid_mask.to(event_tokens.dtype).unsqueeze(-1)
        reasoned_delta_norm = torch.linalg.vector_norm(
            reasoned_delta,
            dim=-1,
        )
        reasoned_delta_norm = self._masked_mean(
            reasoned_delta_norm,
            event_valid_mask,
            dim=-1,
        )

        return {
            "reasoned_event_tokens": reasoned_event_tokens,
            "ordered_relation_logits": torch.where(
                pair_valid_mask,
                relation_logits_raw,
                torch.zeros_like(relation_logits_raw),
            ),
            "ordered_relation_gates": relation_gates,
            "ordered_relation_valid_mask": pair_valid_mask,
            "ordered_relation_context": relation_context,
            "ordered_update_gate": update_gate,
            "ordered_relation_gate_mean": relation_gate_mean,
            "ordered_update_gate_mean": update_gate_mean,
            "ordered_relation_past_mass": relation_past_mass,
            "ordered_relation_future_mass": relation_future_mass,
            "reasoned_event_token_delta_norm": reasoned_delta_norm,
        }

    # ----------------------------------------------------------------------
    # Build hard events + GROUND
    # ----------------------------------------------------------------------
    def _encode_events_and_ground(
        self,
        raw_states: torch.Tensor,
        contextual_states: torch.Tensor,
        raw_visual_states: torch.Tensor,
        contextual_visual_states: torch.Tensor,
        qa_state: torch.Tensor,
        video_mask: torch.Tensor,
        temporal_prior: torch.Tensor,
        event_assignment: torch.Tensor,
        hard_event_index: torch.Tensor,
        frame_semantic_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = contextual_states.shape

        # Topology is hard and detached before entering this method.
        event_assignment = event_assignment.detach()
        hard_event_index = hard_event_index.detach()

        event_mass = event_assignment.sum(dim=1)
        event_valid_mask = event_mass > self.min_event_mass
        event_valid_mask[:, 0] = video_mask.any(dim=1)
        event_activity = event_valid_mask.to(event_mass.dtype)
        event_count = event_valid_mask.sum(dim=-1).to(event_mass.dtype)

        local_delta = torch.zeros_like(raw_states)
        local_delta[:, 1:] = raw_states[:, 1:] - raw_states[:, :-1]

        context_pool = self._weighted_pool(contextual_states, event_assignment)
        raw_pool = self._weighted_pool(raw_states, event_assignment)
        delta_pool = self._weighted_pool(local_delta, event_assignment)

        # Weak-alignment visual content stays pre-FiLM so question modulation
        # cannot trivially copy q into the visual evidence token.
        visual_context_pool = self._weighted_pool(
            contextual_visual_states,
            event_assignment,
        )
        visual_raw_pool = self._weighted_pool(
            raw_visual_states,
            event_assignment,
        )
        event_visual_tokens = 0.5 * (visual_context_pool + visual_raw_pool)

        positions = torch.linspace(
            0.0,
            1.0,
            steps=time_steps,
            dtype=contextual_states.dtype,
            device=contextual_states.device,
        ).view(1, time_steps, 1)

        event_center = (
            (event_assignment * positions).sum(dim=1)
            / event_mass.clamp_min(1e-6)
        )
        valid_length = (
            video_mask.sum(dim=1, keepdim=True)
            .to(contextual_states.dtype)
            .clamp_min(1.0)
        )
        event_duration = event_mass / valid_length
        event_prior_mass = torch.einsum(
            "btk,bt->bk",
            event_assignment,
            temporal_prior,
        )

        qa_events = qa_state.unsqueeze(1).expand(-1, self.max_events, -1)
        event_tokens = self.event_encoder(
            torch.cat(
                [
                    context_pool,
                    raw_pool,
                    delta_pool,
                    qa_events,
                    event_center.unsqueeze(-1),
                    event_duration.unsqueeze(-1),
                    event_prior_mass.unsqueeze(-1),
                ],
                dim=-1,
            )
        )
        event_tokens = (
            event_tokens
            * event_valid_mask.to(event_tokens.dtype).unsqueeze(-1)
        )

        ordered_reasoning = self._ordered_event_interaction(
            event_tokens=event_tokens,
            event_valid_mask=event_valid_mask,
            qa_state=qa_state,
        )
        reasoned_event_tokens = ordered_reasoning["reasoned_event_tokens"]

        # ------------------------------------------------------------------
        # Main V1.4 event evidence: LOGITS, not independent gates and not an
        # event-level softmax. Multiple events can simultaneously contribute
        # positive evidence before the single final temporal normalization.
        # ------------------------------------------------------------------
        event_evidence_logits = self.event_evidence_score(
            torch.cat(
                [
                    reasoned_event_tokens,
                    qa_events,
                    reasoned_event_tokens * qa_events,
                ],
                dim=-1,
            )
        ).squeeze(-1)
        event_evidence_logits = (
            event_evidence_logits
            * event_valid_mask.to(event_evidence_logits.dtype)
        )

        frame_event_logits = torch.einsum(
            "btk,bk->bt",
            event_assignment,
            event_evidence_logits,
        )
        frame_event_logits = (
            frame_event_logits * video_mask.to(frame_event_logits.dtype)
        )

        event_temporal_logits = frame_semantic_logits + frame_event_logits
        event_temporal_logits = (
            event_temporal_logits
            * video_mask.to(event_temporal_logits.dtype)
        )

        # Exactly one temporal softmax in the main Q-Bound grounding path.
        frame_event_relevance = self._masked_softmax(
            event_temporal_logits,
            video_mask,
            dim=-1,
        )

        # Event-normalized weights are used ONLY where a distribution is
        # mathematically required (weak-alignment pooling / summary diagnostics).
        # They do not define main-path event relevance.
        event_evidence_weights = self._masked_softmax(
            event_evidence_logits,
            event_valid_mask,
            dim=-1,
        )
        evidence_visual_token = torch.einsum(
            "bk,bkd->bd",
            event_evidence_weights,
            event_visual_tokens,
        )
        reasoned_event_repr = torch.einsum(
            "bk,bkh->bh",
            event_evidence_weights,
            reasoned_event_tokens,
        )

        # Diagnostic-only independent gates retained as compatibility aliases.
        # They are NOT used by the V1.4 grounding computation above.
        event_evidence_gates = torch.sigmoid(event_evidence_logits)
        event_evidence_gates = (
            event_evidence_gates
            * event_valid_mask.to(event_evidence_gates.dtype)
        )

        valid_event_count_float = event_valid_mask.sum(dim=-1).clamp_min(1).to(
            event_evidence_logits.dtype
        )
        event_evidence_gate_mean = (
            event_evidence_gates.sum(dim=-1) / valid_event_count_float
        )
        event_evidence_active_count = (
            (event_evidence_gates >= 0.5) & event_valid_mask
        ).sum(dim=-1).to(event_evidence_logits.dtype)

        event_evidence_entropy = -torch.sum(
            event_evidence_weights
            * torch.log(event_evidence_weights.clamp_min(1e-8)),
            dim=-1,
        )
        event_evidence_effective_count = torch.exp(event_evidence_entropy)

        sorted_evidence_weights, _ = torch.sort(
            event_evidence_weights,
            dim=-1,
            descending=True,
        )
        top1_event_evidence_share = sorted_evidence_weights[:, 0]
        if self.max_events >= 2:
            top2_event_evidence_share = sorted_evidence_weights[:, :2].sum(dim=-1)
        else:
            top2_event_evidence_share = top1_event_evidence_share

        # A diagnostic "multi-event" condition: among samples with >=2 valid
        # events, at least two events jointly carry >=50% probability mass and
        # the second-largest event has >=20%. Thresholds are diagnostics only.
        second_share = (
            sorted_evidence_weights[:, 1]
            if self.max_events >= 2
            else torch.zeros_like(top1_event_evidence_share)
        )
        multi_event_evidence = (
            (event_count >= 2.0)
            & (top2_event_evidence_share >= 0.50)
            & (second_share >= 0.20)
        ).to(event_evidence_logits.dtype)

        assignment_entropy_per_frame = -torch.sum(
            event_assignment
            * torch.log(event_assignment.clamp_min(1e-8)),
            dim=-1,
        )
        assignment_entropy = (
            assignment_entropy_per_frame
            * video_mask.to(assignment_entropy_per_frame.dtype)
        ).sum(dim=-1) / video_mask.sum(dim=-1).clamp_min(1).to(
            assignment_entropy_per_frame.dtype
        )

        event_mass_distribution = event_mass / event_mass.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        event_mass_entropy = -torch.sum(
            event_mass_distribution
            * torch.log(event_mass_distribution.clamp_min(1e-8)),
            dim=-1,
        )
        soft_effective_event_count = torch.exp(event_mass_entropy)

        valid_mass_mask = event_valid_mask
        event_mass_max = event_mass.masked_fill(~valid_mass_mask, -1e4).max(dim=-1).values
        event_mass_min = event_mass.masked_fill(~valid_mass_mask, 1e4).min(dim=-1).values
        event_mass_mean = self._masked_mean(
            event_mass,
            valid_mass_mask,
            dim=-1,
        )

        return {
            "hard_event_index": hard_event_index,
            "event_assignment": event_assignment,
            "event_tokens": event_tokens,
            "reasoned_event_tokens": reasoned_event_tokens,
            "reasoned_event_repr": reasoned_event_repr,
            "event_visual_tokens": event_visual_tokens,
            "evidence_visual_token": evidence_visual_token,
            "event_mass": event_mass,
            "event_mass_distribution": event_mass_distribution,
            "event_mass_max": event_mass_max,
            "event_mass_min": event_mass_min,
            "event_mass_mean": event_mass_mean,
            "event_valid_mask": event_valid_mask,
            "event_activity": event_activity,
            "event_count": event_count,
            "event_prior_mass": event_prior_mass,
            "event_evidence_logits": event_evidence_logits,
            "event_evidence_weights": event_evidence_weights,
            "event_evidence_gates": event_evidence_gates,
            "event_evidence_gate_mean": event_evidence_gate_mean,
            "event_evidence_active_count": event_evidence_active_count,
            "event_evidence_entropy": event_evidence_entropy,
            "event_evidence_effective_count": event_evidence_effective_count,
            "top1_event_evidence_share": top1_event_evidence_share,
            "top2_event_evidence_share": top2_event_evidence_share,
            "multi_event_evidence": multi_event_evidence,
            "frame_event_logits": frame_event_logits,
            "frame_semantic_logits": frame_semantic_logits,
            "event_temporal_logits": event_temporal_logits,
            "frame_event_relevance": frame_event_relevance,
            "assignment_entropy": assignment_entropy,
            "effective_event_count": event_count,
            "soft_effective_event_count": soft_effective_event_count,
            **ordered_reasoning,
        }


    # ----------------------------------------------------------------------
    # COMPOSE: latent contiguous event-span composition
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    # V1.12 TEMPORAL HYPOTHESIS BANK
    # ----------------------------------------------------------------------
    def _enumerate_temporal_hypotheses(
        self,
        video_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """
        Enumerate every valid contiguous frame interval [s,e].

        At T=32 this gives 528 hypotheses.  This deliberate frame-level bank
        removes event-boundary oracle limitations while event semantics remain
        available to the scorer.  The bank also exposes one-frame neighbour
        indices so a downstream answer-aware critic can compare H with
        trim-left/right and expand-left/right alternatives without temporal GT.
        """
        if video_mask.ndim != 2:
            raise ValueError("video_mask must be [B,T].")

        batch_size, time_steps = video_mask.shape
        device = video_mask.device

        pair = torch.triu_indices(
            time_steps,
            time_steps,
            offset=0,
            device=device,
        )
        start = pair[0].long()
        end = pair[1].long()
        num_hypotheses = int(start.numel())

        frame_index = torch.arange(
            time_steps,
            device=device,
            dtype=torch.long,
        ).view(1, 1, time_steps)
        interval_mask = (
            (frame_index >= start.view(1, num_hypotheses, 1))
            & (frame_index <= end.view(1, num_hypotheses, 1))
        )
        hypothesis_frame_mask = interval_mask & video_mask[:, None, :]

        nominal_length = (end - start + 1).to(dtype)
        valid_frame_count = hypothesis_frame_mask.sum(dim=-1).to(dtype)
        hypothesis_valid_mask = (
            valid_frame_count >= nominal_length.view(1, -1) - 1e-6
        ) & (valid_frame_count > 0)

        frame_weights = (
            hypothesis_frame_mask.to(dtype)
            / valid_frame_count.clamp_min(1.0).unsqueeze(-1)
        )

        complement_frame_mask = (
            video_mask[:, None, :] & (~hypothesis_frame_mask)
        )
        complement_frame_count = complement_frame_mask.sum(dim=-1).to(dtype)
        complement_valid_mask = complement_frame_count > 0
        complement_frame_weights = (
            complement_frame_mask.to(dtype)
            / complement_frame_count.clamp_min(1.0).unsqueeze(-1)
        )

        # Pair -> candidate index lookup for local temporal alternatives.
        pair_to_index = torch.full(
            (time_steps, time_steps),
            -1,
            dtype=torch.long,
            device=device,
        )
        pair_to_index[start, end] = torch.arange(
            num_hypotheses,
            device=device,
            dtype=torch.long,
        )

        neg_one = torch.full_like(start, -1)
        trim_left_index = torch.where(
            start < end,
            pair_to_index[(start + 1).clamp(max=time_steps - 1), end],
            neg_one,
        )
        trim_right_index = torch.where(
            start < end,
            pair_to_index[start, (end - 1).clamp(min=0)],
            neg_one,
        )
        expand_left_index = torch.where(
            start > 0,
            pair_to_index[(start - 1).clamp(min=0), end],
            neg_one,
        )
        expand_right_index = torch.where(
            end < (time_steps - 1),
            pair_to_index[start, (end + 1).clamp(max=time_steps - 1)],
            neg_one,
        )

        def _neighbor_valid(index: torch.Tensor) -> torch.Tensor:
            has_neighbor = index >= 0
            safe_index = index.clamp(min=0)
            gathered = hypothesis_valid_mask[:, safe_index]
            return gathered & has_neighbor.unsqueeze(0)

        return {
            "hypothesis_start_frame": start,
            "hypothesis_end_frame": end,
            "hypothesis_nominal_length": nominal_length,
            "hypothesis_valid_mask": hypothesis_valid_mask,
            "hypothesis_frame_mask": hypothesis_frame_mask,
            "hypothesis_frame_weights": frame_weights,
            "hypothesis_complement_frame_mask": complement_frame_mask,
            "hypothesis_complement_frame_weights": complement_frame_weights,
            "hypothesis_complement_valid_mask": complement_valid_mask,
            "hypothesis_trim_left_index": trim_left_index,
            "hypothesis_trim_right_index": trim_right_index,
            "hypothesis_expand_left_index": expand_left_index,
            "hypothesis_expand_right_index": expand_right_index,
            "hypothesis_trim_left_valid_mask": _neighbor_valid(trim_left_index),
            "hypothesis_trim_right_valid_mask": _neighbor_valid(trim_right_index),
            "hypothesis_expand_left_valid_mask": _neighbor_valid(expand_left_index),
            "hypothesis_expand_right_valid_mask": _neighbor_valid(expand_right_index),
        }

    def _score_temporal_hypotheses(
        self,
        raw_states: torch.Tensor,
        contextual_states: torch.Tensor,
        qa_state: torch.Tensor,
        boundary_prob: torch.Tensor,
        grounding: Dict[str, torch.Tensor],
        bank: Dict[str, torch.Tensor],
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Score frame-exact temporal hypotheses with V+Q only.

        Event decomposition is used as semantic guidance, never as a hard
        boundary constraint.  A training-only answer-aware critic will later
        supervise these logits outside the core.
        """
        batch_size, time_steps, hidden_dim = contextual_states.shape
        if raw_states.shape != contextual_states.shape:
            raise ValueError("raw_states/contextual_states must match.")
        if hidden_dim != self.hidden_dim:
            raise ValueError("Unexpected hidden dimension.")

        start = bank["hypothesis_start_frame"]
        end = bank["hypothesis_end_frame"]
        valid = bank["hypothesis_valid_mask"].bool()
        weights = bank["hypothesis_frame_weights"]
        frame_mask = bank["hypothesis_frame_mask"].bool()
        num_hypotheses = int(start.numel())

        # Frame-level reasoned event field.  Hard event membership is detached
        # by the decomposition branch; reasoned event semantics remain trainable.
        frame_reasoned_event = torch.einsum(
            "btk,bkh->bth",
            grounding["event_assignment"],
            grounding["reasoned_event_tokens"],
        )

        contextual_pool = torch.einsum(
            "bst,bth->bsh",
            weights,
            contextual_states,
        )
        reasoned_pool = torch.einsum(
            "bst,bth->bsh",
            weights,
            frame_reasoned_event,
        )

        start_state = contextual_states[:, start, :]
        end_state = contextual_states[:, end, :]
        qa_h = qa_state.unsqueeze(1).expand(-1, num_hypotheses, -1)

        local_temporal_logits = grounding["event_temporal_logits"]
        frame_semantic_logits = grounding["frame_semantic_logits"]
        local_mean = torch.einsum(
            "bst,bt->bs",
            weights,
            local_temporal_logits,
        )
        frame_semantic_mean = torch.einsum(
            "bst,bt->bs",
            weights,
            frame_semantic_logits,
        )

        valid_length = video_mask.sum(dim=-1).to(
            contextual_states.dtype
        ).clamp_min(1.0)
        denom = (valid_length - 1.0).clamp_min(1.0)
        center = (
            (start.to(contextual_states.dtype).view(1, -1)
             + end.to(contextual_states.dtype).view(1, -1))
            * 0.5
        ) / denom.unsqueeze(-1)
        duration = (
            bank["hypothesis_nominal_length"].view(1, -1)
            / valid_length.unsqueeze(-1)
        )

        # Video edges are valid natural boundaries and receive confidence 1.
        if time_steps > 1:
            start_boundary_index = (start - 1).clamp(min=0, max=time_steps - 2)
            end_boundary_index = end.clamp(min=0, max=time_steps - 2)
            left_boundary = boundary_prob[:, start_boundary_index]
            right_boundary = boundary_prob[:, end_boundary_index]
            left_boundary = torch.where(
                (start == 0).unsqueeze(0),
                torch.ones_like(left_boundary),
                left_boundary,
            )
            right_boundary = torch.where(
                (end == time_steps - 1).unsqueeze(0),
                torch.ones_like(right_boundary),
                right_boundary,
            )
        else:
            left_boundary = torch.ones(
                batch_size,
                num_hypotheses,
                dtype=contextual_states.dtype,
                device=contextual_states.device,
            )
            right_boundary = left_boundary

        # How many of the dynamically discovered events does H touch?  This is
        # descriptive guidance only; there is no penalty for using many/few.
        event_overlap = torch.einsum(
            "bst,btk->bsk",
            frame_mask.to(contextual_states.dtype),
            grounding["event_assignment"],
        )
        touched_events = (event_overlap > 0.0) & grounding[
            "event_valid_mask"
        ].unsqueeze(1)
        touched_event_count = touched_events.sum(dim=-1).to(
            contextual_states.dtype
        )
        event_count = grounding["event_count"].clamp_min(1.0).unsqueeze(-1)
        touched_event_ratio = touched_event_count / event_count

        scalar_features = torch.stack(
            [
                center,
                duration,
                torch.tanh(local_mean),
                torch.tanh(frame_semantic_mean),
                left_boundary,
                right_boundary,
                touched_event_ratio,
            ],
            dim=-1,
        )

        hypothesis_input = torch.cat(
            [
                start_state,
                end_state,
                contextual_pool,
                reasoned_pool,
                qa_h,
                reasoned_pool * qa_h,
                scalar_features,
            ],
            dim=-1,
        )
        hypothesis_hidden = self.hypothesis_feature_encoder(hypothesis_input)
        hypothesis_logits_raw = self.hypothesis_score(
            hypothesis_hidden
        ).squeeze(-1)
        hypothesis_posterior = self._masked_softmax(
            hypothesis_logits_raw,
            valid,
            dim=-1,
        )

        posterior_entropy = -torch.sum(
            hypothesis_posterior
            * torch.log(hypothesis_posterior.clamp_min(1e-8)),
            dim=-1,
        )
        posterior_effective_count = torch.exp(posterior_entropy)
        posterior_top1 = hypothesis_posterior.max(dim=-1).values

        # Soft support is diagnostic only.  The V1.12 MAIN output is the hard
        # winning hypothesis to avoid recreating V1.6 all-span diffusion.
        soft_hypothesis_support = torch.einsum(
            "bs,bst->bt",
            hypothesis_posterior,
            frame_mask.to(hypothesis_posterior.dtype),
        )
        soft_hypothesis_support = (
            soft_hypothesis_support
            * video_mask.to(soft_hypothesis_support.dtype)
        )

        masked_logits = hypothesis_logits_raw.masked_fill(~valid, -1e4)
        selected_index = masked_logits.detach().argmax(dim=-1)
        batch_index = torch.arange(
            batch_size,
            device=contextual_states.device,
        )
        selected_start = start[selected_index]
        selected_end = end[selected_index]
        selected_frame_mask = frame_mask[
            batch_index,
            selected_index,
        ].detach()
        selected_frame_weights = weights[
            batch_index,
            selected_index,
        ]
        selected_hypothesis_logit = hypothesis_logits_raw[
            batch_index,
            selected_index,
        ]
        selected_hypothesis_posterior = hypothesis_posterior[
            batch_index,
            selected_index,
        ]
        selected_hypothesis_repr = hypothesis_hidden[
            batch_index,
            selected_index,
        ]
        selected_reasoned_event_repr = reasoned_pool[
            batch_index,
            selected_index,
        ]

        main_temporal_prob = self._masked_softmax(
            local_temporal_logits,
            selected_frame_mask,
            dim=-1,
        )
        main_temporal_logits = torch.log(
            main_temporal_prob.clamp_min(1e-8)
        ) * video_mask.to(main_temporal_prob.dtype)
        main_soft_support = selected_frame_mask.to(main_temporal_prob.dtype)

        selected_length = selected_frame_mask.sum(dim=-1).to(
            contextual_states.dtype
        )
        selected_fraction = selected_length / valid_length
        expected_length = torch.sum(
            hypothesis_posterior
            * bank["hypothesis_nominal_length"].view(1, -1),
            dim=-1,
        )

        # Exact one-hot endpoint posterior retained as a generic boundary
        # interface for downstream wrappers/metrics.
        main_pair_posterior = torch.zeros(
            batch_size,
            time_steps,
            time_steps,
            dtype=main_temporal_prob.dtype,
            device=main_temporal_prob.device,
        )
        main_pair_posterior[
            batch_index,
            selected_start,
            selected_end,
        ] = 1.0

        frame_denominator = (valid_length - 1.0).clamp_min(1.0)
        selected_start_norm = (
            selected_start.to(contextual_states.dtype) / frame_denominator
        ).clamp(0.0, 1.0)
        selected_end_norm = (
            selected_end.to(contextual_states.dtype) / frame_denominator
        ).clamp(0.0, 1.0)

        return {
            "hypothesis_logits": torch.where(
                valid,
                hypothesis_logits_raw,
                torch.zeros_like(hypothesis_logits_raw),
            ),
            "hypothesis_posterior": hypothesis_posterior,
            "hypothesis_entropy": posterior_entropy,
            "hypothesis_effective_count": posterior_effective_count,
            "hypothesis_top1_posterior": posterior_top1,
            "hypothesis_soft_support": soft_hypothesis_support,
            "hypothesis_expected_span_length": expected_length,
            "hypothesis_contextual_pool": contextual_pool,
            "hypothesis_reasoned_event_pool": reasoned_pool,
            "hypothesis_touched_event_count": touched_event_count,
            "hypothesis_touched_event_ratio": touched_event_ratio,
            "hypothesis_local_evidence_mean": local_mean,
            "hypothesis_frame_evidence_mean": frame_semantic_mean,
            "hypothesis_left_boundary_evidence": left_boundary,
            "hypothesis_right_boundary_evidence": right_boundary,
            "selected_hypothesis_index": selected_index,
            "selected_hypothesis_start_frame": selected_start.detach(),
            "selected_hypothesis_end_frame": selected_end.detach(),
            "selected_hypothesis_frame_mask": selected_frame_mask,
            "selected_hypothesis_frame_weights": selected_frame_weights,
            "selected_hypothesis_logit": selected_hypothesis_logit,
            "selected_hypothesis_posterior": selected_hypothesis_posterior,
            "selected_hypothesis_repr": selected_hypothesis_repr,
            "selected_reasoned_event_repr": selected_reasoned_event_repr,
            "selected_hypothesis_span_length": selected_length,
            "selected_hypothesis_span_fraction": selected_fraction,
            "selected_hypothesis_start": selected_start_norm,
            "selected_hypothesis_end": selected_end_norm,
            "main_temporal_prob": main_temporal_prob,
            "main_temporal_logits": main_temporal_logits,
            "main_soft_support": main_soft_support,
            "main_pair_posterior": main_pair_posterior,
        }

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(
        self,
        video_tokens: torch.Tensor,
        question_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        contextual_tokens: Optional[torch.Tensor] = None,
        temporal_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,T,D].")

        external_contextual_tokens = contextual_tokens
        if external_contextual_tokens is not None:
            if external_contextual_tokens.shape != video_tokens.shape:
                raise ValueError(
                    "video_tokens and contextual_tokens must have the same shape."
                )

        batch_size, time_steps, feature_dim = video_tokens.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {feature_dim}."
            )
        if time_steps < 2:
            raise ValueError("Version 1.12 requires at least 2 timesteps.")

        video_mask = self._normalize_mask(
            video_mask,
            batch_size,
            time_steps,
            video_tokens.device,
        )

        # Generic Q-Bound input.  The external visual encoder may be detached by
        # the caller; this private adapter remains trainable.
        raw_external_video_tokens = video_tokens
        video_tokens, input_adapter_diag = self._adapt_generic_video_tokens(
            raw_external_video_tokens,
            video_mask,
        )

        context_reuses_raw = (
            external_contextual_tokens is None
            or external_contextual_tokens is raw_external_video_tokens
            or (
                torch.is_tensor(external_contextual_tokens)
                and external_contextual_tokens.data_ptr()
                == raw_external_video_tokens.data_ptr()
            )
        )
        if context_reuses_raw:
            contextual_tokens = video_tokens
            contextual_adapter_diag = input_adapter_diag
        else:
            contextual_tokens, contextual_adapter_diag = (
                self._adapt_generic_video_tokens(
                    external_contextual_tokens,
                    video_mask,
                )
            )

        boundary_mask = video_mask[:, :-1] & video_mask[:, 1:]
        temporal_prior_used = self._prepare_temporal_prior(
            temporal_prior,
            video_mask,
            video_tokens.dtype,
        )
        qa_summary = self._summarize_qa(question_features)

        # ------------------------------------------------------------------
        # A. DYNAMIC DECOMPOSE -- K_i is selected per sample; K_max is capacity.
        # ------------------------------------------------------------------
        decomp_features = self._build_decomposition_features(
            video_tokens=video_tokens,
            qa_summary=qa_summary,
            video_mask=video_mask,
        )
        topology_states = self.topology_proj(decomp_features)
        topology_qa = self.topology_qa_proj(qa_summary.detach())

        left_topology = topology_states[:, :-1]
        right_topology = topology_states[:, 1:]
        qa_boundaries = topology_qa.unsqueeze(1).expand(
            -1,
            time_steps - 1,
            -1,
        )

        prior_left = temporal_prior_used[:, :-1]
        prior_right = temporal_prior_used[:, 1:]
        prior_change = prior_right - prior_left
        prior_features = torch.stack(
            [prior_left, prior_right, prior_change],
            dim=-1,
        )
        boundary_position = torch.linspace(
            0.0,
            1.0,
            steps=time_steps - 1,
            dtype=video_tokens.dtype,
            device=video_tokens.device,
        ).view(1, time_steps - 1, 1).expand(batch_size, -1, -1)

        boundary_input = torch.cat(
            [
                left_topology,
                right_topology,
                torch.abs(right_topology - left_topology),
                right_topology - left_topology,
                left_topology * right_topology,
                qa_boundaries,
                prior_features,
                boundary_position,
            ],
            dim=-1,
        )
        boundary_hidden = self.boundary_feature_encoder(boundary_input)
        boundary_logits = self.boundary_classifier(boundary_hidden).squeeze(-1)
        boundary_prob = torch.sigmoid(
            boundary_logits / self.boundary_temperature
        )
        boundary_prob = boundary_prob * boundary_mask.to(boundary_prob.dtype)

        proposals = self._extract_boundary_proposals(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
        )
        decomposition = self._enumerate_partitions(
            candidate_indicator=proposals["candidate_indicator"],
            candidate_probs=proposals["candidate_probs"],
            candidate_valid_mask=proposals["candidate_valid_mask"],
            decomp_features=decomp_features,
            video_mask=video_mask,
        )

        # ------------------------------------------------------------------
        # B. SEMANTIC STATES + RELATE.
        # ------------------------------------------------------------------
        raw_visual_states = self.raw_proj(video_tokens)
        contextual_visual_states = self.context_proj(contextual_tokens)
        qa_state = self.qa_proj(qa_summary)

        gamma, beta = self.qa_film(qa_state).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        raw_states = (
            raw_visual_states * (1.0 + gamma.unsqueeze(1))
            + beta.unsqueeze(1)
        )
        contextual_states = (
            contextual_visual_states * (1.0 + gamma.unsqueeze(1))
            + beta.unsqueeze(1)
        )
        raw_states = raw_states * video_mask.to(raw_states.dtype).unsqueeze(-1)
        contextual_states = (
            contextual_states
            * video_mask.to(contextual_states.dtype).unsqueeze(-1)
        )

        qa_frames = qa_state.unsqueeze(1).expand(-1, time_steps, -1)
        frame_hidden = self.frame_evidence_encoder(
            torch.cat(
                [
                    contextual_states,
                    raw_states,
                    contextual_states * qa_frames,
                    qa_frames,
                ],
                dim=-1,
            )
        )
        frame_semantic_logits = self.frame_evidence_score(
            frame_hidden
        ).squeeze(-1)
        frame_semantic_logits = (
            frame_semantic_logits * video_mask.to(frame_semantic_logits.dtype)
        )

        grounding = self._encode_events_and_ground(
            raw_states=raw_states,
            contextual_states=contextual_states,
            raw_visual_states=raw_visual_states,
            contextual_visual_states=contextual_visual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            temporal_prior=temporal_prior_used,
            event_assignment=decomposition["hard_assignment"],
            hard_event_index=decomposition["hard_event_index"],
            frame_semantic_logits=frame_semantic_logits,
        )

        # ------------------------------------------------------------------
        # C. V1.12 EXACT TEMPORAL HYPOTHESIS BANK + QUESTION-ONLY COMPETITION.
        # ------------------------------------------------------------------
        bank = self._enumerate_temporal_hypotheses(
            video_mask=video_mask,
            dtype=video_tokens.dtype,
        )
        hypotheses = self._score_temporal_hypotheses(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            boundary_prob=boundary_prob,
            grounding=grounding,
            bank=bank,
            video_mask=video_mask,
        )

        boundary_hard = decomposition["hard_boundary"]
        boundary_count = boundary_hard.sum(dim=-1)
        event_count = boundary_count + 1.0
        selected_boundary_prob_sum = (
            boundary_prob * boundary_hard
        ).sum(dim=-1)
        selected_boundary_count = boundary_count.clamp_min(1.0)
        selected_boundary_prob_mean = (
            selected_boundary_prob_sum / selected_boundary_count
        )
        selected_boundary_prob_mean = torch.where(
            boundary_count > 0,
            selected_boundary_prob_mean,
            torch.zeros_like(selected_boundary_prob_mean),
        )
        boundary_prob_mean = (
            boundary_prob.sum(dim=-1)
            / boundary_mask.sum(dim=-1).clamp_min(1).to(boundary_prob.dtype)
        )

        event_evidence_logits = grounding["event_evidence_logits"]
        event_evidence_weights = grounding["event_evidence_weights"]
        event_evidence_gates = grounding["event_evidence_gates"]

        return {
            # ----------------------------------------------------------
            # V1.12 architecture contract.
            # ----------------------------------------------------------
            "qbound_version": self.qbound_version,
            "dynamic_event_count_enabled": torch.ones(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "event_capacity": torch.as_tensor(
                float(self.max_events),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "hypothesis_competition_enabled": torch.ones(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "answer_features_used_in_core": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "legacy_composer_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "legacy_refiner_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "legacy_counterfactual_donor_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),

            # Generic input diagnostics.
            "generic_input_adapter_enabled": torch.as_tensor(
                float(self.use_generic_input_adapter),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "generic_input_position_enabled": torch.as_tensor(
                float(self.input_adapter_use_position),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "generic_input_context_reuse": torch.as_tensor(
                float(context_reuses_raw),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "generic_input_raw_rms": input_adapter_diag[
                "generic_input_raw_rms"
            ],
            "generic_input_adapted_rms": input_adapter_diag[
                "generic_input_adapted_rms"
            ],
            "generic_input_token_cosine": input_adapter_diag[
                "generic_input_token_cosine"
            ],
            "generic_input_delta_ratio": input_adapter_diag[
                "generic_input_delta_ratio"
            ],
            "generic_context_raw_rms": contextual_adapter_diag[
                "generic_input_raw_rms"
            ],
            "generic_context_adapted_rms": contextual_adapter_diag[
                "generic_input_adapted_rms"
            ],

            # Core states.
            "qa_state": qa_state,
            "topology_qa_state": topology_qa,
            "decomp_features": decomp_features,
            "raw_states": raw_states,
            "contextual_states": contextual_states,

            # Boundary proposal diagnostics.
            "boundary_logits": boundary_logits,
            "boundary_prob": boundary_prob,
            "boundary_mask": boundary_mask,
            "boundary_local_peak": proposals["local_peak_value"],
            "boundary_local_peak_mask": proposals["local_peak_mask"],
            "boundary_candidate_mask": proposals["threshold_candidate_mask"],
            "boundary_candidate_indices": proposals["candidate_indices"],
            "boundary_candidate_probs": proposals["candidate_probs"],
            "boundary_candidate_valid_mask": proposals[
                "candidate_valid_mask"
            ],
            "boundary_hard": boundary_hard,
            "boundary_mass": boundary_hard,
            "boundary_soft_marginal": decomposition[
                "boundary_soft_marginal"
            ],
            "boundary_count": boundary_count,
            "event_count": event_count,
            "boundary_prob_mean": boundary_prob_mean,
            "selected_boundary_prob_mean": selected_boundary_prob_mean,

            # Dynamic decomposition outputs.
            "partition_energy": decomposition["partition_energy"],
            "partition_valid_mask": decomposition["partition_valid_mask"],
            "partition_posterior": decomposition["partition_posterior"],
            "partition_entropy": decomposition["partition_entropy"],
            "best_partition_index": decomposition["best_partition_index"],
            "best_partition_pattern": decomposition["hard_pattern"],
            "decomp_loss": decomposition["decomp_loss"],
            "decomp_expected_energy": decomposition[
                "decomp_expected_energy"
            ],
            "decomp_selected_coherence": decomposition[
                "decomp_selected_coherence"
            ],
            "decomp_selected_complexity": decomposition[
                "decomp_selected_complexity"
            ],
            "decomp_selected_complexity_cost": decomposition[
                "decomp_selected_complexity_cost"
            ],
            "decomp_selected_boundary_reward": decomposition[
                "decomp_selected_boundary_reward"
            ],
            "decomp_selected_boundary_term": decomposition[
                "decomp_selected_boundary_term"
            ],
            "decomp_selected_total_energy": decomposition[
                "decomp_selected_total_energy"
            ],
            "decomp_expected_event_count": decomposition[
                "decomp_expected_event_count"
            ],
            "candidate_split_gain": decomposition["candidate_split_gain"],
            "accepted_split_gain": decomposition["accepted_split_gain"],
            "rejected_split_gain": decomposition["rejected_split_gain"],

            # Dynamic hard events + RELATE.
            "hard_event_index": grounding["hard_event_index"],
            "event_assignment": grounding["event_assignment"],
            "active_assignment": grounding["event_assignment"],
            "event_tokens": grounding["event_tokens"],
            "reasoned_event_tokens": grounding["reasoned_event_tokens"],
            "reasoned_event_repr": grounding["reasoned_event_repr"],
            "event_visual_tokens": grounding["event_visual_tokens"],
            "evidence_visual_token": grounding["evidence_visual_token"],
            "event_mass": grounding["event_mass"],
            "event_mass_distribution": grounding["event_mass_distribution"],
            "event_mass_max": grounding["event_mass_max"],
            "event_mass_min": grounding["event_mass_min"],
            "event_mass_mean": grounding["event_mass_mean"],
            "event_valid_mask": grounding["event_valid_mask"],
            "event_activity": grounding["event_activity"],
            "event_prior_mass": grounding["event_prior_mass"],
            "effective_event_count": grounding["effective_event_count"],
            "soft_effective_event_count": grounding[
                "soft_effective_event_count"
            ],
            "assignment_entropy": grounding["assignment_entropy"],
            "ordered_relation_logits": grounding["ordered_relation_logits"],
            "ordered_relation_gates": grounding["ordered_relation_gates"],
            "ordered_relation_valid_mask": grounding[
                "ordered_relation_valid_mask"
            ],
            "ordered_relation_context": grounding[
                "ordered_relation_context"
            ],
            "ordered_update_gate": grounding["ordered_update_gate"],
            "ordered_relation_gate_mean": grounding[
                "ordered_relation_gate_mean"
            ],
            "ordered_update_gate_mean": grounding[
                "ordered_update_gate_mean"
            ],
            "ordered_relation_past_mass": grounding[
                "ordered_relation_past_mass"
            ],
            "ordered_relation_future_mass": grounding[
                "ordered_relation_future_mass"
            ],
            "reasoned_event_token_delta_norm": grounding[
                "reasoned_event_token_delta_norm"
            ],

            # Local evidence features retained for the hypothesis scorer and
            # weak-alignment branch.
            "event_evidence_logits": event_evidence_logits,
            "event_evidence_weights": event_evidence_weights,
            "event_evidence_gates": event_evidence_gates,
            "event_evidence_gate_mean": grounding[
                "event_evidence_gate_mean"
            ],
            "event_evidence_active_count": grounding[
                "event_evidence_active_count"
            ],
            "event_evidence_entropy": grounding["event_evidence_entropy"],
            "event_evidence_effective_count": grounding[
                "event_evidence_effective_count"
            ],
            "top1_event_evidence_share": grounding[
                "top1_event_evidence_share"
            ],
            "top2_event_evidence_share": grounding[
                "top2_event_evidence_share"
            ],
            "multi_event_evidence": grounding["multi_event_evidence"],
            "frame_event_logits": grounding["frame_event_logits"],
            "frame_semantic_logits": grounding["frame_semantic_logits"],
            "event_temporal_logits": grounding["event_temporal_logits"],
            "frame_event_relevance": grounding["frame_event_relevance"],

            # ----------------------------------------------------------
            # V1.12 hypothesis bank -- training-time critic interface.
            # ----------------------------------------------------------
            **bank,
            "hypothesis_candidate_count": torch.as_tensor(
                float(bank["hypothesis_start_frame"].numel()),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            **hypotheses,

            # MAIN V1.12 grounding aliases.
            "event_temporal_prob": hypotheses["main_temporal_prob"],
            "calibrated_temporal_prob": hypotheses["main_temporal_prob"],
            "calibrated_temporal_logits": hypotheses["main_temporal_logits"],
            "calibrated_soft_support": hypotheses["main_soft_support"],
            "calibrated_span_start_frame": hypotheses[
                "selected_hypothesis_start_frame"
            ],
            "calibrated_span_end_frame": hypotheses[
                "selected_hypothesis_end_frame"
            ],
            "calibrated_frame_mask": hypotheses[
                "selected_hypothesis_frame_mask"
            ],
            "calibrated_span_frame_length": hypotheses[
                "selected_hypothesis_span_length"
            ],
            "calibrated_span_fraction": hypotheses[
                "selected_hypothesis_span_fraction"
            ],
            "calibrated_span_start": hypotheses[
                "selected_hypothesis_start"
            ],
            "calibrated_span_end": hypotheses[
                "selected_hypothesis_end"
            ],

            # Old names retained only where semantically safe for logger
            # migration.  They do not re-enable historical V1.5--V1.11 logic.
            "event_relevance_logits": event_evidence_logits,
            "event_relevance_weights": event_evidence_weights,
            "event_relevance_gates": event_evidence_gates,
            "event_relevance_gate_mean": grounding[
                "event_evidence_gate_mean"
            ],
            "event_relevance_active_count": grounding[
                "event_evidence_active_count"
            ],
            "event_relevance_effective_count": grounding[
                "event_evidence_effective_count"
            ],
            "event_relevance_entropy": grounding["event_evidence_entropy"],

            # Optional prior diagnostics.
            "temporal_prior_used": temporal_prior_used,
            "temporal_prior_enabled": torch.as_tensor(
                float(self.use_temporal_prior),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
        }

