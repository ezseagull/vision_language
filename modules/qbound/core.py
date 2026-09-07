from itertools import combinations
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound Version 1.6 core:
        DECOMPOSE -> RELATE -> COMPOSE -> REFINE -> GROUND

    V1.6 keeps the healthy DECOMPOSE / RELATE / COMPOSE mechanisms from
    V1.5 and addresses the failure revealed by that version: coarse event-chain
    boundaries gave excellent temporal coverage but were too coarse to serve as
    final grounding boundaries, producing very long spans and low IoP.

    The new REFINE stage performs hierarchical coarse-to-fine localization:
    the event-span composer proposes relevant contiguous event chains, then a
    question-conditioned frame-level boundary refiner trims the first/last
    boundary events without changing the latent event topology.

    ---------------------------------------------------------------------------
    Core method
    ---------------------------------------------------------------------------

      Video tokens X + question-only q
                  |
                  v
              DECOMPOSE
      question-conditioned latent partition
                  |
                  v
          E1 | E2 | ... | EM
                  |
                  v
               RELATE
      Question-Conditioned Ordered Event Interaction (Q-OEI)
                  |
                  v
          reasoned event tokens
                  |
                  v
              COMPOSE
      enumerate contiguous event chains:
          [E1], [E2], [E1,E2], [E2,E3], ...
                  |
                  v
      question-conditioned span posterior
                  |
             +----+------------------+
             |                       |
             v                       v
      soft adaptive support     hard selected chain
             |                       |
             v                       v
      frame-level refinement    selected event tokens
             |                       |
             v                       v
      composed temporal p(t)    future LLM adapter

    Design principles
    -----------------
    1) The V1.4 DECOMPOSE branch is preserved. max_events is still only an
       upper bound and hard event membership remains detached before downstream
       reasoning.

    2) Q-OEI is preserved. No direct reasoned-event residual is added to answer
       space inside this core.

    3) COMPOSE operates only on question-only semantics, reasoned event tokens,
       event/frame evidence, and the latent event topology. It never consumes
       answer-choice features, CRA temporal keys, or temporal GT.

    4) Every candidate span is a contiguous chain of latent events. With
       max_events=4 there are at most 10 candidates, so exhaustive latent span
       composition is cheap.

    5) The soft span posterior is differentiable and learns through downstream
       weak task supervision. The hard selected span is detached and exposed for
       adaptive start/end extraction and future selected-event-token interfaces.

    6) The main V1.5 temporal distribution is produced by a single frame-level
       softmax after span support has been composed with local frame/event
       evidence. The old V1.4 temporal probability is retained only as a
       diagnostic output.

    Backbone independence
    ---------------------
    The core receives generic video tokens plus question-only features.
    contextual_tokens and temporal_prior remain optional compatibility inputs;
    the main V1.5 configuration should keep use_temporal_prior=False.

    Weak supervision
    ----------------
    No temporal ground-truth interval is consumed by this module.

    Inputs
    ------
    video_tokens: [B,T,D]
    question_features: [B,D] preferred, question-only
    video_mask: [B,T], optional
    contextual_tokens: [B,T,D], optional
    temporal_prior: [B,T], optional compatibility prior
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        max_events: int = 4,
        use_temporal_prior: bool = False,
        dropout: float = 0.1,
        boundary_temperature: float = 0.7,
        boundary_threshold: float = 0.5,
        boundary_radius: int = 2,
        boundary_bias_init: float = -0.5,
        proposal_candidates: int = 6,
        decomp_complexity_weight: float = 1.25,
        decomp_boundary_weight: float = 0.50,
        decomp_temperature: float = 0.20,
        question_modulation_scale: float = 0.10,
        min_event_mass: float = 0.25,
        composer_temperature: float = 0.35,
        refiner_temperature: float = 0.30,
    ) -> None:
        super().__init__()

        if max_events < 2:
            raise ValueError("Version 1.6 requires max_events >= 2.")
        if proposal_candidates < 1:
            raise ValueError("proposal_candidates must be >= 1.")
        if proposal_candidates < max_events - 1:
            raise ValueError(
                "proposal_candidates should be >= max_events - 1 so every "
                "allowed event count can be represented."
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
        if composer_temperature <= 0.0:
            raise ValueError("composer_temperature must be positive.")
        if refiner_temperature <= 0.0:
            raise ValueError("refiner_temperature must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_events = int(max_events)
        self.max_transitions = self.max_events - 1
        self.proposal_candidates = int(proposal_candidates)

        self.use_temporal_prior = bool(use_temporal_prior)
        self.boundary_temperature = float(boundary_temperature)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_radius = int(boundary_radius)
        self.decomp_complexity_weight = float(decomp_complexity_weight)
        self.decomp_boundary_weight = float(decomp_boundary_weight)
        self.decomp_temperature = float(decomp_temperature)
        self.question_modulation_scale = float(question_modulation_scale)
        self.min_event_mass = float(min_event_mass)
        self.composer_temperature = float(composer_temperature)
        self.refiner_temperature = float(refiner_temperature)

        # ------------------------------------------------------------------
        # A. TOPOLOGY / DECOMPOSITION BRANCH
        # ------------------------------------------------------------------
        # The coherence geometry is measured directly on detached backbone
        # features. The question can only modulate each channel within a narrow
        # range [1-rho, 1+rho], preventing the decomposition objective from
        # learning a trivial constant representation.
        self.decomp_question_gate = nn.Linear(input_dim, input_dim)
        nn.init.zeros_(self.decomp_question_gate.weight)
        nn.init.zeros_(self.decomp_question_gate.bias)

        # Dedicated topology projections. These are not used by downstream
        # event semantics / VQA, which limits the V1.3 shortcut in which answer
        # gradients could reshape the boundary representation itself.
        self.topology_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.topology_qa_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # left/right topology state, abs change, signed change, interaction,
        # question, optional prior(left/right/diff), normalized position.
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

        # Enumerate all subsets of proposal slots with at most max_events-1
        # selected boundaries. With proposal_candidates=6 and max_events=4 this
        # is only 42 partitions per sample.
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
        # Latent contiguous event-span candidates used by V1.5 COMPOSE.
        #
        # For K=max_events, enumerate:
        #   [E0], [E1], ..., [E0,E1], [E1,E2], ..., [E0,...,E(K-1)]
        #
        # Dynamic samples simply invalidate candidates that touch padded event
        # slots. With max_events=4 this creates only 10 candidates.
        # ------------------------------------------------------------------
        span_patterns = []
        span_starts = []
        span_ends = []
        span_lengths = []

        for start_event in range(self.max_events):
            for end_event in range(start_event, self.max_events):
                pattern = torch.zeros(self.max_events, dtype=torch.bool)
                pattern[start_event : end_event + 1] = True
                span_patterns.append(pattern)
                span_starts.append(start_event)
                span_ends.append(end_event)
                span_lengths.append(end_event - start_event + 1)

        self.register_buffer(
            "composer_span_patterns",
            torch.stack(span_patterns, dim=0),
            persistent=False,
        )
        self.register_buffer(
            "composer_span_start_events",
            torch.tensor(span_starts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "composer_span_end_events",
            torch.tensor(span_ends, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "composer_span_event_counts",
            torch.tensor(span_lengths, dtype=torch.float32),
            persistent=False,
        )

        # ------------------------------------------------------------------
        # B. SEMANTIC EVENT / GROUNDING BRANCH
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

        # contextual, raw, dynamics, question, center, duration, prior mass.
        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ------------------------------------------------------------------
        # C. RELATE: Question-Conditioned Ordered Event Interaction (Q-OEI)
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
        # D. GROUND: evidence composition, no event-level winner selection.
        # ------------------------------------------------------------------
        # Main path uses raw event evidence logits, not sigmoid/softmax gates.
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
        # E. COMPOSE: Latent Event-Span Composer.
        # ------------------------------------------------------------------
        # Candidate representation:
        #   start event
        #   end event
        #   pooled reasoned span
        #   question
        #   pooled span x question
        #   + center, frame duration, event-count ratio,
        #     bounded event-evidence mean, bounded frame-evidence mean.
        #
        # A bounded mean event-evidence term is also added as a direct prior to
        # the learned span score. This makes the new composer start from the
        # useful V1.4 event evidence instead of an arbitrary uniform readout.
        composer_input_dim = hidden_dim * 5 + 5
        self.span_composer_score = nn.Sequential(
            nn.Linear(composer_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(
            self.span_composer_score[-1].weight,
            mean=0.0,
            std=1e-2,
        )
        nn.init.zeros_(self.span_composer_score[-1].bias)

        # ------------------------------------------------------------------
        # F. REFINE: Question-Conditioned Frame Boundary Refiner (V1.6)
        # ------------------------------------------------------------------
        # REFINE never modifies latent event topology. It predicts fine
        # start/end distributions inside the first/last event of each coarse
        # Composer candidate and is trained only through downstream weak task
        # supervision.
        refiner_input_dim = hidden_dim * 6 + 2
        self.boundary_refine_encoder = nn.Sequential(
            nn.Linear(refiner_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.boundary_start_score = nn.Linear(hidden_dim, 1)
        self.boundary_end_score = nn.Linear(hidden_dim, 1)

        nn.init.normal_(
            self.boundary_start_score.weight,
            mean=0.0,
            std=1e-2,
        )
        nn.init.zeros_(self.boundary_start_score.bias)
        nn.init.normal_(
            self.boundary_end_score.weight,
            mean=0.0,
            std=1e-2,
        )
        nn.init.zeros_(self.boundary_end_score.bias)

    # ----------------------------------------------------------------------
    # Generic helpers
    # ----------------------------------------------------------------------
    def _summarize_qa(self, qa_features: torch.Tensor) -> torch.Tensor:
        """Compatibility helper. V1.6 should receive question-only [B,D]."""
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
    def _compose_event_spans(
        self,
        reasoned_event_tokens: torch.Tensor,
        event_valid_mask: torch.Tensor,
        event_assignment: torch.Tensor,
        event_evidence_logits: torch.Tensor,
        local_temporal_logits: torch.Tensor,
        qa_state: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compose adaptive temporal support from contiguous chains of latent events.

        The V1.4 local temporal evidence remains useful, but it no longer has to
        be converted into a fixed 3/5-frame CRA window. Instead, this method:

          1) enumerates all valid contiguous event chains;
          2) scores them with question-conditioned reasoned event semantics;
          3) forms a differentiable posterior over candidate chains;
          4) converts that posterior into a soft frame-support prior;
          5) combines the support prior with local frame/event logits;
          6) applies ONE final frame-level temporal softmax.

        The hard selected chain is detached and used only as an adaptive support
        interface / diagnostic. Gradients from the main composed probability flow
        through the soft span posterior and the span scorer.
        """
        if reasoned_event_tokens.ndim != 3:
            raise ValueError(
                "reasoned_event_tokens must be [B,K,H], got "
                f"{tuple(reasoned_event_tokens.shape)}."
            )

        batch_size, event_slots, hidden_dim = reasoned_event_tokens.shape
        if event_slots != self.max_events or hidden_dim != self.hidden_dim:
            raise ValueError(
                "reasoned_event_tokens shape mismatch: expected "
                f"[B,{self.max_events},{self.hidden_dim}], got "
                f"{tuple(reasoned_event_tokens.shape)}."
            )

        if event_assignment.shape[:2] != video_mask.shape:
            raise ValueError(
                "event_assignment/video_mask temporal shape mismatch."
            )
        if event_assignment.size(-1) != self.max_events:
            raise ValueError(
                "event_assignment must end with max_events slots."
            )
        if event_evidence_logits.shape != (batch_size, self.max_events):
            raise ValueError(
                "event_evidence_logits must be [B,max_events]."
            )
        if local_temporal_logits.shape != video_mask.shape:
            raise ValueError(
                "local_temporal_logits and video_mask must both be [B,T]."
            )

        event_valid_mask = event_valid_mask.bool()
        video_mask = video_mask.bool()

        span_patterns = self.composer_span_patterns.to(
            device=reasoned_event_tokens.device
        )
        span_patterns_f = span_patterns.to(reasoned_event_tokens.dtype)
        num_spans = span_patterns.size(0)

        span_event_count = self.composer_span_event_counts.to(
            device=reasoned_event_tokens.device,
            dtype=reasoned_event_tokens.dtype,
        )

        # Candidate valid iff all event slots in the chain exist.
        valid_selected_events = torch.einsum(
            "bk,sk->bs",
            event_valid_mask.to(reasoned_event_tokens.dtype),
            span_patterns_f,
        )
        span_valid_mask = (
            valid_selected_events
            >= (span_event_count.view(1, -1) - 1e-6)
        )

        # [B,S,T]: exact hard frame support for every contiguous event chain.
        span_frame_membership = torch.einsum(
            "btk,sk->bst",
            event_assignment.detach(),
            span_patterns_f,
        )
        span_frame_mask = (
            span_frame_membership > 0.5
        ) & video_mask.unsqueeze(1)

        span_frame_length = span_frame_mask.sum(dim=-1).to(
            reasoned_event_tokens.dtype
        )
        span_valid_mask = span_valid_mask & (span_frame_length > 0)

        # Pooled reasoned event semantics at the event level.
        span_event_pool = torch.einsum(
            "sk,bkh->bsh",
            span_patterns_f,
            reasoned_event_tokens,
        )
        span_event_pool = (
            span_event_pool
            / span_event_count.view(1, num_spans, 1).clamp_min(1.0)
        )

        start_event_tokens = reasoned_event_tokens[
            :,
            self.composer_span_start_events,
            :,
        ]
        end_event_tokens = reasoned_event_tokens[
            :,
            self.composer_span_end_events,
            :,
        ]
        qa_spans = qa_state.unsqueeze(1).expand(-1, num_spans, -1)

        span_event_evidence_mean = torch.einsum(
            "sk,bk->bs",
            span_patterns_f,
            event_evidence_logits,
        ) / span_event_count.view(1, num_spans).clamp_min(1.0)

        frame_mask_f = span_frame_mask.to(local_temporal_logits.dtype)
        span_frame_evidence_mean = torch.einsum(
            "bst,bt->bs",
            frame_mask_f,
            local_temporal_logits,
        ) / span_frame_length.clamp_min(1.0)

        time_steps = video_mask.size(1)
        valid_length = video_mask.sum(dim=-1).to(
            reasoned_event_tokens.dtype
        ).clamp_min(1.0)

        positions = torch.linspace(
            0.0,
            1.0,
            steps=time_steps,
            dtype=reasoned_event_tokens.dtype,
            device=reasoned_event_tokens.device,
        ).view(1, 1, time_steps)

        span_center = (
            frame_mask_f * positions
        ).sum(dim=-1) / span_frame_length.clamp_min(1.0)

        span_duration = (
            span_frame_length
            / valid_length.unsqueeze(1)
        )
        span_event_count_ratio = (
            span_event_count.view(1, -1)
            / float(max(self.max_events, 1))
        ).expand(batch_size, -1)

        scalar_features = torch.stack(
            [
                span_center,
                span_duration,
                span_event_count_ratio,
                torch.tanh(span_event_evidence_mean),
                torch.tanh(span_frame_evidence_mean),
            ],
            dim=-1,
        )

        composer_input = torch.cat(
            [
                start_event_tokens,
                end_event_tokens,
                span_event_pool,
                qa_spans,
                span_event_pool * qa_spans,
                scalar_features,
            ],
            dim=-1,
        )

        learned_span_logits = self.span_composer_score(
            composer_input
        ).squeeze(-1)

        # Existing V1.4 event evidence is a bounded prior, not a teacher.
        # The learned composer can override it through task gradients.
        span_evidence_prior = torch.tanh(span_event_evidence_mean)
        span_logits_raw = learned_span_logits + span_evidence_prior

        span_posterior = self._masked_softmax(
            span_logits_raw / self.composer_temperature,
            span_valid_mask,
            dim=-1,
        )

        span_entropy = -torch.sum(
            span_posterior
            * torch.log(span_posterior.clamp_min(1e-8)),
            dim=-1,
        )
        span_effective_count = torch.exp(span_entropy)
        span_top1_posterior = span_posterior.max(dim=-1).values

        # Differentiable event-membership probabilities: useful both for
        # diagnostics and a future frozen-LLM event-token adapter.
        soft_selected_event_weights = torch.einsum(
            "bs,sk->bk",
            span_posterior,
            span_patterns_f,
        )
        soft_selected_event_weights = (
            soft_selected_event_weights
            * event_valid_mask.to(soft_selected_event_weights.dtype)
        )

        soft_event_denominator = soft_selected_event_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        composed_reasoned_event_repr = torch.einsum(
            "bk,bkh->bh",
            soft_selected_event_weights / soft_event_denominator,
            reasoned_event_tokens,
        )

        # Span membership probability at every frame. It is intentionally not
        # length-normalized: it represents P(frame is inside relevant support).
        soft_span_support = torch.einsum(
            "bs,bst->bt",
            span_posterior,
            frame_mask_f,
        )
        soft_span_support = (
            soft_span_support * video_mask.to(soft_span_support.dtype)
        )

        # Coarse-to-fine composition:
        #   coarse = latent event-span support
        #   fine   = V1.4 local event/frame evidence
        #
        # Adding log(support) to local logits is a probabilistically natural
        # product-of-evidence composition. A per-sample normalization constant
        # would cancel in the final softmax, so support need not sum to one.
        span_support_log_bias = torch.log(
            soft_span_support.clamp_min(1e-6)
        )
        composed_temporal_logits = (
            local_temporal_logits + span_support_log_bias
        )
        composed_temporal_logits = (
            composed_temporal_logits
            * video_mask.to(composed_temporal_logits.dtype)
        )

        composed_temporal_prob = self._masked_softmax(
            composed_temporal_logits,
            video_mask,
            dim=-1,
        )

        # ------------------------------------------------------------------
        # Hard adaptive span: detached interface for start/end extraction and
        # future selected-event-token routing.
        # ------------------------------------------------------------------
        selected_span_index = span_posterior.detach().argmax(dim=-1)
        batch_index = torch.arange(
            batch_size,
            device=reasoned_event_tokens.device,
        )

        selected_event_mask = span_patterns[
            selected_span_index
        ].detach()
        selected_event_mask = (
            selected_event_mask
            & event_valid_mask
        )

        selected_frame_mask = span_frame_mask[
            batch_index,
            selected_span_index,
        ].detach()

        frame_indices = torch.arange(
            time_steps,
            device=reasoned_event_tokens.device,
            dtype=torch.long,
        ).view(1, time_steps).expand(batch_size, -1)

        selected_span_start_frame = torch.where(
            selected_frame_mask,
            frame_indices,
            torch.full_like(frame_indices, time_steps),
        ).min(dim=-1).values
        selected_span_end_frame = torch.where(
            selected_frame_mask,
            frame_indices,
            torch.full_like(frame_indices, -1),
        ).max(dim=-1).values

        # Defensive fallback; a valid E0 span should make this unnecessary.
        fallback_start = torch.zeros_like(selected_span_start_frame)
        fallback_end = (
            video_mask.sum(dim=-1).clamp_min(1).long() - 1
        )
        invalid_selected_span = (
            selected_span_start_frame >= time_steps
        ) | (selected_span_end_frame < 0)

        selected_span_start_frame = torch.where(
            invalid_selected_span,
            fallback_start,
            selected_span_start_frame,
        )
        selected_span_end_frame = torch.where(
            invalid_selected_span,
            fallback_end,
            selected_span_end_frame,
        )

        selected_span_frame_length = selected_frame_mask.sum(
            dim=-1
        ).to(reasoned_event_tokens.dtype)
        selected_span_fraction = (
            selected_span_frame_length
            / valid_length.clamp_min(1.0)
        )

        selected_span_event_count = selected_event_mask.sum(
            dim=-1
        ).to(reasoned_event_tokens.dtype)

        selected_span_start_event = (
            self.composer_span_start_events[
                selected_span_index
            ].detach()
        )
        selected_span_end_event = (
            self.composer_span_end_events[
                selected_span_index
            ].detach()
        )
        selected_span_posterior = span_posterior[
            batch_index,
            selected_span_index,
        ]

        selected_reasoned_event_tokens = (
            reasoned_event_tokens
            * selected_event_mask.to(reasoned_event_tokens.dtype).unsqueeze(-1)
        )

        # Fine probability restricted to the selected hard support. Diagnostic
        # / optional adapter output; the main differentiable path stays soft.
        selected_span_temporal_prob = self._masked_softmax(
            local_temporal_logits,
            selected_frame_mask,
            dim=-1,
        )

        # Human-readable normalized boundaries. Grounding adapters may use the
        # exact integer frame indices instead.
        frame_denominator = (
            valid_length - 1.0
        ).clamp_min(1.0)
        selected_span_start = (
            selected_span_start_frame.to(reasoned_event_tokens.dtype)
            / frame_denominator
        )
        selected_span_end = (
            selected_span_end_frame.to(reasoned_event_tokens.dtype)
            / frame_denominator
        )
        selected_span_start = selected_span_start.clamp(0.0, 1.0)
        selected_span_end = selected_span_end.clamp(0.0, 1.0)

        return {
            "span_candidate_event_mask": span_patterns,
            "span_candidate_start_event": self.composer_span_start_events,
            "span_candidate_end_event": self.composer_span_end_events,
            "span_candidate_event_count": span_event_count,
            "span_candidate_valid_mask": span_valid_mask,
            "span_candidate_frame_mask": span_frame_mask,
            "span_candidate_frame_length": span_frame_length,
            "span_candidate_center": span_center,
            "span_candidate_duration": span_duration,
            "span_event_evidence_mean": span_event_evidence_mean,
            "span_frame_evidence_mean": span_frame_evidence_mean,
            "span_learned_logits": learned_span_logits,
            "span_evidence_prior": span_evidence_prior,
            "span_logits": torch.where(
                span_valid_mask,
                span_logits_raw,
                torch.zeros_like(span_logits_raw),
            ),
            "span_posterior": span_posterior,
            "span_entropy": span_entropy,
            "span_effective_count": span_effective_count,
            "span_top1_posterior": span_top1_posterior,
            "soft_selected_event_weights": soft_selected_event_weights,
            "composed_reasoned_event_repr": composed_reasoned_event_repr,
            "soft_span_support": soft_span_support,
            "span_support_log_bias": span_support_log_bias,
            "composed_temporal_logits": composed_temporal_logits,
            "composed_temporal_prob": composed_temporal_prob,
            "selected_span_index": selected_span_index,
            "selected_event_mask": selected_event_mask,
            "selected_reasoned_event_tokens": selected_reasoned_event_tokens,
            "selected_frame_mask": selected_frame_mask,
            "selected_span_start_event": selected_span_start_event,
            "selected_span_end_event": selected_span_end_event,
            "selected_span_event_count": selected_span_event_count,
            "selected_span_start_frame": selected_span_start_frame,
            "selected_span_end_frame": selected_span_end_frame,
            "selected_span_frame_length": selected_span_frame_length,
            "selected_span_fraction": selected_span_fraction,
            "selected_span_posterior": selected_span_posterior,
            "selected_span_temporal_prob": selected_span_temporal_prob,
            "selected_span_start": selected_span_start,
            "selected_span_end": selected_span_end,
        }

    # ----------------------------------------------------------------------
    # REFINE: hierarchical frame-level boundary localization
    # ----------------------------------------------------------------------
    def _refine_composed_spans(
        self,
        raw_states: torch.Tensor,
        contextual_states: torch.Tensor,
        qa_state: torch.Tensor,
        reasoned_event_tokens: torch.Tensor,
        event_assignment: torch.Tensor,
        local_temporal_logits: torch.Tensor,
        composition: Dict[str, torch.Tensor],
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Refine every valid coarse event-chain candidate to frame-level boundaries.

        Hierarchical latent model:
            P(span,start,end|X,q)
              = P(span|X,q) P(start,end|span,X,q)

        Start is restricted to the first event of a coarse candidate; end is
        restricted to its last event. Interior selected events are therefore kept
        intact. No temporal GT or explicit span-length penalty is used.

        The main V1.6 temporal path marginalizes all valid coarse spans and all
        valid start/end pairs, then applies one final temporal softmax.
        """
        batch_size, time_steps, hidden_dim = raw_states.shape

        if contextual_states.shape != raw_states.shape:
            raise ValueError("raw_states and contextual_states must match.")
        if hidden_dim != self.hidden_dim:
            raise ValueError("Unexpected hidden dimension in REFINE.")
        if qa_state.shape != (batch_size, self.hidden_dim):
            raise ValueError("qa_state must be [B,H].")
        if reasoned_event_tokens.shape != (
            batch_size,
            self.max_events,
            self.hidden_dim,
        ):
            raise ValueError("reasoned_event_tokens must be [B,K,H].")
        if event_assignment.shape != (
            batch_size,
            time_steps,
            self.max_events,
        ):
            raise ValueError("event_assignment must be [B,T,K].")
        if local_temporal_logits.shape != (batch_size, time_steps):
            raise ValueError("local_temporal_logits must be [B,T].")
        if video_mask.shape != (batch_size, time_steps):
            raise ValueError("video_mask must be [B,T].")

        video_mask = video_mask.bool()
        event_assignment = event_assignment.detach()

        span_posterior = composition["span_posterior"]
        span_valid_mask = composition["span_candidate_valid_mask"].bool()
        num_spans = span_posterior.size(1)

        # --------------------------------------------------------------
        # A. Frame-level question/event-aware boundary representation.
        # --------------------------------------------------------------
        qa_frames = qa_state.unsqueeze(1).expand(-1, time_steps, -1)

        frame_reasoned_event = torch.einsum(
            "btk,bkh->bth",
            event_assignment,
            reasoned_event_tokens,
        )

        mask_f = video_mask.to(local_temporal_logits.dtype)
        valid_length = mask_f.sum(dim=-1).clamp_min(1.0)

        local_mean = (
            local_temporal_logits * mask_f
        ).sum(dim=-1) / valid_length
        centered_local = (
            local_temporal_logits - local_mean.unsqueeze(-1)
        )
        local_var = (
            centered_local.pow(2) * mask_f
        ).sum(dim=-1) / valid_length
        local_std = torch.sqrt(local_var + 1e-6)
        local_z = centered_local / local_std.unsqueeze(-1).clamp_min(1e-4)
        local_z = local_z * mask_f

        frame_index_float = torch.arange(
            time_steps,
            dtype=raw_states.dtype,
            device=raw_states.device,
        ).view(1, time_steps).expand(batch_size, -1)

        global_position = (
            frame_index_float
            / (valid_length.unsqueeze(-1) - 1.0).clamp_min(1.0)
        ).clamp(0.0, 1.0)
        global_position = global_position * mask_f

        refine_frame_input = torch.cat(
            [
                contextual_states,
                raw_states,
                qa_frames,
                contextual_states * qa_frames,
                frame_reasoned_event,
                frame_reasoned_event * qa_frames,
                local_z.unsqueeze(-1),
                global_position.unsqueeze(-1),
            ],
            dim=-1,
        )

        refine_hidden = self.boundary_refine_encoder(
            refine_frame_input
        )
        start_logits_raw = self.boundary_start_score(
            refine_hidden
        ).squeeze(-1)
        end_logits_raw = self.boundary_end_score(
            refine_hidden
        ).squeeze(-1)

        start_logits_raw = start_logits_raw * mask_f
        end_logits_raw = end_logits_raw * mask_f

        # --------------------------------------------------------------
        # B. Candidate-specific boundary masks.
        # --------------------------------------------------------------
        start_events = self.composer_span_start_events.to(
            device=raw_states.device
        )
        end_events = self.composer_span_end_events.to(
            device=raw_states.device
        )

        start_selector = F.one_hot(
            start_events,
            num_classes=self.max_events,
        ).to(event_assignment.dtype)
        end_selector = F.one_hot(
            end_events,
            num_classes=self.max_events,
        ).to(event_assignment.dtype)

        start_candidate_mask = (
            torch.einsum(
                "btk,sk->bst",
                event_assignment,
                start_selector,
            ) > 0.5
        )
        end_candidate_mask = (
            torch.einsum(
                "btk,sk->bst",
                event_assignment,
                end_selector,
            ) > 0.5
        )

        start_candidate_mask = (
            start_candidate_mask
            & span_valid_mask.unsqueeze(-1)
            & video_mask.unsqueeze(1)
        )
        end_candidate_mask = (
            end_candidate_mask
            & span_valid_mask.unsqueeze(-1)
            & video_mask.unsqueeze(1)
        )

        # --------------------------------------------------------------
        # C. Conditional P(start,end | coarse span).
        # --------------------------------------------------------------
        start_grid = torch.arange(
            time_steps,
            device=raw_states.device,
        ).view(1, 1, time_steps, 1)
        end_grid = torch.arange(
            time_steps,
            device=raw_states.device,
        ).view(1, 1, 1, time_steps)

        pair_valid_mask = (
            start_candidate_mask.unsqueeze(-1)
            & end_candidate_mask.unsqueeze(-2)
            & (start_grid <= end_grid)
        )

        pair_logits_raw = (
            start_logits_raw[:, None, :, None]
            + end_logits_raw[:, None, None, :]
        ).expand(
            -1,
            num_spans,
            -1,
            -1,
        )

        flat_pair_logits = pair_logits_raw.reshape(
            batch_size,
            num_spans,
            time_steps * time_steps,
        )
        flat_pair_mask = pair_valid_mask.reshape(
            batch_size,
            num_spans,
            time_steps * time_steps,
        )

        pair_posterior = self._masked_softmax(
            flat_pair_logits / self.refiner_temperature,
            flat_pair_mask,
            dim=-1,
        ).reshape(
            batch_size,
            num_spans,
            time_steps,
            time_steps,
        )

        pair_entropy_per_span = -torch.sum(
            pair_posterior
            * torch.log(pair_posterior.clamp_min(1e-8)),
            dim=(-2, -1),
        )
        pair_effective_count_per_span = torch.exp(
            pair_entropy_per_span
        )
        pair_top1_per_span = pair_posterior.flatten(
            start_dim=-2
        ).max(dim=-1).values

        # --------------------------------------------------------------
        # D. Convert each pair posterior to P(frame inside | span).
        # --------------------------------------------------------------
        start_prefix = torch.cumsum(
            pair_posterior,
            dim=2,
        )
        start_prefix_end_suffix = torch.flip(
            torch.cumsum(
                torch.flip(
                    start_prefix,
                    dims=[3],
                ),
                dim=3,
            ),
            dims=[3],
        )
        conditional_refined_support = torch.diagonal(
            start_prefix_end_suffix,
            dim1=2,
            dim2=3,
        )

        conditional_refined_support = (
            conditional_refined_support
            * span_valid_mask.to(
                conditional_refined_support.dtype
            ).unsqueeze(-1)
            * video_mask.to(
                conditional_refined_support.dtype
            ).unsqueeze(1)
        )

        # Hierarchical marginal over coarse spans.
        refined_soft_support = torch.einsum(
            "bs,bst->bt",
            span_posterior,
            conditional_refined_support,
        )
        refined_soft_support = (
            refined_soft_support
            * video_mask.to(refined_soft_support.dtype)
        )

        refined_support_log_bias = torch.log(
            refined_soft_support.clamp_min(1e-6)
        )

        refined_temporal_logits = (
            local_temporal_logits
            + refined_support_log_bias
        )
        refined_temporal_logits = (
            refined_temporal_logits
            * video_mask.to(refined_temporal_logits.dtype)
        )

        refined_temporal_prob = self._masked_softmax(
            refined_temporal_logits,
            video_mask,
            dim=-1,
        )

        # --------------------------------------------------------------
        # E. Hard top-1 refined interval for evaluation / diagnostics.
        # --------------------------------------------------------------
        selected_span_index = composition[
            "selected_span_index"
        ].detach().long()

        batch_index = torch.arange(
            batch_size,
            device=raw_states.device,
        )

        selected_pair_posterior = pair_posterior[
            batch_index,
            selected_span_index,
        ]

        selected_pair_flat = selected_pair_posterior.detach().reshape(
            batch_size,
            time_steps * time_steps,
        )
        selected_pair_flat_index = selected_pair_flat.argmax(
            dim=-1
        )

        refined_start_frame = (
            selected_pair_flat_index // time_steps
        ).long()
        refined_end_frame = (
            selected_pair_flat_index % time_steps
        ).long()

        coarse_start_frame = composition[
            "selected_span_start_frame"
        ].detach().long()
        coarse_end_frame = composition[
            "selected_span_end_frame"
        ].detach().long()

        selected_pair_valid = pair_valid_mask[
            batch_index,
            selected_span_index,
        ].reshape(
            batch_size,
            -1,
        ).any(dim=-1)

        refined_start_frame = torch.where(
            selected_pair_valid,
            refined_start_frame,
            coarse_start_frame,
        )
        refined_end_frame = torch.where(
            selected_pair_valid,
            refined_end_frame,
            coarse_end_frame,
        )

        frame_indices = torch.arange(
            time_steps,
            device=raw_states.device,
        ).view(1, time_steps).expand(batch_size, -1)

        refined_frame_mask = (
            (frame_indices >= refined_start_frame.unsqueeze(-1))
            & (frame_indices <= refined_end_frame.unsqueeze(-1))
            & video_mask
        ).detach()

        refined_span_frame_length = refined_frame_mask.sum(
            dim=-1
        ).to(raw_states.dtype)
        refined_span_fraction = (
            refined_span_frame_length
            / valid_length.clamp_min(1.0)
        )

        coarse_span_frame_length = composition[
            "selected_span_frame_length"
        ].to(raw_states.dtype)
        coarse_span_fraction = composition[
            "selected_span_fraction"
        ].to(raw_states.dtype)

        trim_left_frames = (
            refined_start_frame - coarse_start_frame
        ).clamp_min(0).to(raw_states.dtype)
        trim_right_frames = (
            coarse_end_frame - refined_end_frame
        ).clamp_min(0).to(raw_states.dtype)
        trim_total_frames = (
            trim_left_frames + trim_right_frames
        )

        refined_retained_fraction = (
            refined_span_frame_length
            / coarse_span_frame_length.clamp_min(1.0)
        )

        frame_denominator = (
            valid_length - 1.0
        ).clamp_min(1.0)

        refined_span_start = (
            refined_start_frame.to(raw_states.dtype)
            / frame_denominator
        ).clamp(0.0, 1.0)
        refined_span_end = (
            refined_end_frame.to(raw_states.dtype)
            / frame_denominator
        ).clamp(0.0, 1.0)

        refined_span_temporal_prob = self._masked_softmax(
            local_temporal_logits,
            refined_frame_mask,
            dim=-1,
        )

        selected_pair_entropy = pair_entropy_per_span[
            batch_index,
            selected_span_index,
        ]
        selected_pair_effective_count = pair_effective_count_per_span[
            batch_index,
            selected_span_index,
        ]
        selected_pair_top1_posterior = pair_top1_per_span[
            batch_index,
            selected_span_index,
        ]

        expected_pair_entropy = torch.sum(
            span_posterior
            * pair_entropy_per_span,
            dim=-1,
        )

        selected_start_candidate_count = start_candidate_mask[
            batch_index,
            selected_span_index,
        ].sum(dim=-1).to(raw_states.dtype)
        selected_end_candidate_count = end_candidate_mask[
            batch_index,
            selected_span_index,
        ].sum(dim=-1).to(raw_states.dtype)

        refined_inside_coarse = (
            (refined_start_frame >= coarse_start_frame)
            & (refined_end_frame <= coarse_end_frame)
            & (refined_start_frame <= refined_end_frame)
        ).to(raw_states.dtype)

        return {
            "refine_start_logits": start_logits_raw,
            "refine_end_logits": end_logits_raw,
            "refine_start_candidate_mask": start_candidate_mask,
            "refine_end_candidate_mask": end_candidate_mask,
            "refine_conditional_support": conditional_refined_support,
            "refined_soft_support": refined_soft_support,
            "refined_support_log_bias": refined_support_log_bias,
            "refined_temporal_logits": refined_temporal_logits,
            "refined_temporal_prob": refined_temporal_prob,
            "refine_expected_pair_entropy": expected_pair_entropy,
            "refine_selected_pair_posterior": selected_pair_posterior,
            "refine_selected_pair_entropy": selected_pair_entropy,
            "refine_selected_pair_effective_count": (
                selected_pair_effective_count
            ),
            "refine_selected_pair_top1_posterior": (
                selected_pair_top1_posterior
            ),
            "refine_selected_start_candidate_count": (
                selected_start_candidate_count
            ),
            "refine_selected_end_candidate_count": (
                selected_end_candidate_count
            ),
            "coarse_span_start_frame": coarse_start_frame,
            "coarse_span_end_frame": coarse_end_frame,
            "coarse_span_frame_length": coarse_span_frame_length,
            "coarse_span_fraction": coarse_span_fraction,
            "coarse_frame_mask": composition["selected_frame_mask"],
            "refined_span_start_frame": refined_start_frame.detach(),
            "refined_span_end_frame": refined_end_frame.detach(),
            "refined_frame_mask": refined_frame_mask,
            "refined_span_frame_length": refined_span_frame_length,
            "refined_span_fraction": refined_span_fraction,
            "refined_span_start": refined_span_start,
            "refined_span_end": refined_span_end,
            "refined_span_temporal_prob": refined_span_temporal_prob,
            "refine_trim_left_frames": trim_left_frames,
            "refine_trim_right_frames": trim_right_frames,
            "refine_trim_total_frames": trim_total_frames,
            "refine_retained_fraction": refined_retained_fraction,
            "refined_inside_coarse": refined_inside_coarse,
            "refiner_temperature": torch.as_tensor(
                self.refiner_temperature,
                dtype=raw_states.dtype,
                device=raw_states.device,
            ),
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

        if contextual_tokens is None:
            contextual_tokens = video_tokens
        if contextual_tokens.shape != video_tokens.shape:
            raise ValueError(
                "video_tokens and contextual_tokens must have the same shape."
            )

        batch_size, time_steps, feature_dim = video_tokens.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {feature_dim}."
            )
        if time_steps < 2:
            raise ValueError("Version 1.6 requires at least 2 timesteps.")

        video_mask = self._normalize_mask(
            video_mask,
            batch_size,
            time_steps,
            video_tokens.device,
        )
        boundary_mask = video_mask[:, :-1] & video_mask[:, 1:]

        temporal_prior_used = self._prepare_temporal_prior(
            temporal_prior,
            video_mask,
            video_tokens.dtype,
        )

        qa_summary = self._summarize_qa(question_features)

        # ------------------------------------------------------------------
        # A. DECOMPOSE
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
        # B. SEMANTIC STATES used by RELATE/GROUND. These are separate from the
        # topology branch so downstream task gradients do not directly optimize
        # the boundary representation.
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
            contextual_states * video_mask.to(contextual_states.dtype).unsqueeze(-1)
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
        frame_semantic_logits = self.frame_evidence_score(frame_hidden).squeeze(-1)
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
        # C. COMPOSE (V1.5)
        # ------------------------------------------------------------------
        composition = self._compose_event_spans(
            reasoned_event_tokens=grounding["reasoned_event_tokens"],
            event_valid_mask=grounding["event_valid_mask"],
            event_assignment=grounding["event_assignment"],
            event_evidence_logits=grounding["event_evidence_logits"],
            local_temporal_logits=grounding["event_temporal_logits"],
            qa_state=qa_state,
            video_mask=video_mask,
        )

        # --------------------------------------------------------------
        # D. REFINE (V1.6)
        # --------------------------------------------------------------
        refinement = self._refine_composed_spans(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            reasoned_event_tokens=grounding["reasoned_event_tokens"],
            event_assignment=grounding["event_assignment"],
            local_temporal_logits=grounding["event_temporal_logits"],
            composition=composition,
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

        # Compatibility aliases from V1.3 are intentionally retained where they
        # do not alter the V1.4 method. In particular, event_relevance_* below
        # are diagnostic aliases for event evidence; the main path does not use
        # event-level gates.
        event_evidence_logits = grounding["event_evidence_logits"]
        event_evidence_weights = grounding["event_evidence_weights"]
        event_evidence_gates = grounding["event_evidence_gates"]

        return {
            # Core states.
            "qa_state": qa_state,
            "topology_qa_state": topology_qa,
            "decomp_features": decomp_features,

            # Boundary proposal diagnostics.
            "boundary_logits": boundary_logits,
            "boundary_prob": boundary_prob,
            "boundary_mask": boundary_mask,
            "boundary_local_peak": proposals["local_peak_value"],
            "boundary_local_peak_mask": proposals["local_peak_mask"],
            "boundary_candidate_mask": proposals["threshold_candidate_mask"],
            "boundary_candidate_indices": proposals["candidate_indices"],
            "boundary_candidate_probs": proposals["candidate_probs"],
            "boundary_candidate_valid_mask": proposals["candidate_valid_mask"],
            "boundary_hard": boundary_hard,
            "boundary_mass": boundary_hard,  # compatibility: final selected topology
            "boundary_soft_marginal": decomposition["boundary_soft_marginal"],
            "boundary_count": boundary_count,
            "event_count": event_count,
            "boundary_prob_mean": boundary_prob_mean,
            "selected_boundary_prob_mean": selected_boundary_prob_mean,

            # Latent decomposition outputs.
            "partition_energy": decomposition["partition_energy"],
            "partition_valid_mask": decomposition["partition_valid_mask"],
            "partition_posterior": decomposition["partition_posterior"],
            "partition_entropy": decomposition["partition_entropy"],
            "best_partition_index": decomposition["best_partition_index"],
            "best_partition_pattern": decomposition["hard_pattern"],
            "decomp_loss": decomposition["decomp_loss"],
            "decomp_expected_energy": decomposition["decomp_expected_energy"],
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

            # Hard dynamic events.
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

            # RELATE / Q-OEI.
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

            # Local V1.4 event/frame evidence retained inside V1.5.
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

            # V1.5 COMPOSE: latent contiguous event-chain readout.
            "span_candidate_event_mask": composition[
                "span_candidate_event_mask"
            ],
            "span_candidate_start_event": composition[
                "span_candidate_start_event"
            ],
            "span_candidate_end_event": composition[
                "span_candidate_end_event"
            ],
            "span_candidate_event_count": composition[
                "span_candidate_event_count"
            ],
            "span_candidate_valid_mask": composition[
                "span_candidate_valid_mask"
            ],
            "span_candidate_frame_mask": composition[
                "span_candidate_frame_mask"
            ],
            "span_candidate_frame_length": composition[
                "span_candidate_frame_length"
            ],
            "span_candidate_center": composition[
                "span_candidate_center"
            ],
            "span_candidate_duration": composition[
                "span_candidate_duration"
            ],
            "span_event_evidence_mean": composition[
                "span_event_evidence_mean"
            ],
            "span_frame_evidence_mean": composition[
                "span_frame_evidence_mean"
            ],
            "span_learned_logits": composition["span_learned_logits"],
            "span_evidence_prior": composition["span_evidence_prior"],
            "span_logits": composition["span_logits"],
            "span_posterior": composition["span_posterior"],
            "span_entropy": composition["span_entropy"],
            "span_effective_count": composition["span_effective_count"],
            "span_top1_posterior": composition["span_top1_posterior"],
            "soft_selected_event_weights": composition[
                "soft_selected_event_weights"
            ],
            "composed_reasoned_event_repr": composition[
                "composed_reasoned_event_repr"
            ],
            "soft_span_support": composition["soft_span_support"],
            "span_support_log_bias": composition["span_support_log_bias"],
            "composed_temporal_logits": composition[
                "composed_temporal_logits"
            ],
            "composed_temporal_prob": composition[
                "composed_temporal_prob"
            ],
            "legacy_v15_composed_temporal_logits": composition[
                "composed_temporal_logits"
            ],
            "composer_enabled": torch.as_tensor(
                1.0,
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "span_candidate_count": torch.as_tensor(
                float(self.composer_span_patterns.size(0)),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "selected_span_index": composition["selected_span_index"],
            "selected_event_mask": composition["selected_event_mask"],
            "selected_reasoned_event_tokens": composition[
                "selected_reasoned_event_tokens"
            ],
            "selected_frame_mask": composition["selected_frame_mask"],
            "selected_span_start_event": composition[
                "selected_span_start_event"
            ],
            "selected_span_end_event": composition[
                "selected_span_end_event"
            ],
            "selected_span_event_count": composition[
                "selected_span_event_count"
            ],
            "selected_span_start_frame": composition[
                "selected_span_start_frame"
            ],
            "selected_span_end_frame": composition[
                "selected_span_end_frame"
            ],
            "selected_span_frame_length": composition[
                "selected_span_frame_length"
            ],
            "selected_span_fraction": composition[
                "selected_span_fraction"
            ],
            "selected_span_posterior": composition[
                "selected_span_posterior"
            ],
            "selected_span_temporal_prob": composition[
                "selected_span_temporal_prob"
            ],
            "selected_span_start": composition["selected_span_start"],
            "selected_span_end": composition["selected_span_end"],

            # V1.6 REFINE outputs.
            "refiner_enabled": torch.as_tensor(
                1.0,
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "refine_start_logits": refinement["refine_start_logits"],
            "refine_end_logits": refinement["refine_end_logits"],
            "refine_start_candidate_mask": refinement[
                "refine_start_candidate_mask"
            ],
            "refine_end_candidate_mask": refinement[
                "refine_end_candidate_mask"
            ],
            "refine_conditional_support": refinement[
                "refine_conditional_support"
            ],
            "refined_soft_support": refinement["refined_soft_support"],
            "refined_support_log_bias": refinement[
                "refined_support_log_bias"
            ],
            "refined_temporal_logits": refinement[
                "refined_temporal_logits"
            ],
            "refined_temporal_prob": refinement[
                "refined_temporal_prob"
            ],
            "refine_expected_pair_entropy": refinement[
                "refine_expected_pair_entropy"
            ],
            "refine_selected_pair_posterior": refinement[
                "refine_selected_pair_posterior"
            ],
            "refine_selected_pair_entropy": refinement[
                "refine_selected_pair_entropy"
            ],
            "refine_selected_pair_effective_count": refinement[
                "refine_selected_pair_effective_count"
            ],
            "refine_selected_pair_top1_posterior": refinement[
                "refine_selected_pair_top1_posterior"
            ],
            "refine_selected_start_candidate_count": refinement[
                "refine_selected_start_candidate_count"
            ],
            "refine_selected_end_candidate_count": refinement[
                "refine_selected_end_candidate_count"
            ],
            "coarse_span_start_frame": refinement[
                "coarse_span_start_frame"
            ],
            "coarse_span_end_frame": refinement[
                "coarse_span_end_frame"
            ],
            "coarse_span_frame_length": refinement[
                "coarse_span_frame_length"
            ],
            "coarse_span_fraction": refinement[
                "coarse_span_fraction"
            ],
            "coarse_frame_mask": refinement["coarse_frame_mask"],
            "refined_span_start_frame": refinement[
                "refined_span_start_frame"
            ],
            "refined_span_end_frame": refinement[
                "refined_span_end_frame"
            ],
            "refined_frame_mask": refinement["refined_frame_mask"],
            "refined_span_frame_length": refinement[
                "refined_span_frame_length"
            ],
            "refined_span_fraction": refinement[
                "refined_span_fraction"
            ],
            "refined_span_start": refinement["refined_span_start"],
            "refined_span_end": refinement["refined_span_end"],
            "refined_span_temporal_prob": refinement[
                "refined_span_temporal_prob"
            ],
            "refine_trim_left_frames": refinement[
                "refine_trim_left_frames"
            ],
            "refine_trim_right_frames": refinement[
                "refine_trim_right_frames"
            ],
            "refine_trim_total_frames": refinement[
                "refine_trim_total_frames"
            ],
            "refine_retained_fraction": refinement[
                "refine_retained_fraction"
            ],
            "refined_inside_coarse": refinement[
                "refined_inside_coarse"
            ],
            "refiner_temperature": refinement["refiner_temperature"],

            # Main V1.6 temporal output. V1.5 and V1.4 are diagnostics.
            "event_temporal_prob": refinement["refined_temporal_prob"],
            "main_temporal_prob": refinement["refined_temporal_prob"],
            "main_temporal_logits": refinement["refined_temporal_logits"],
            "legacy_v15_composed_temporal_prob": composition[
                "composed_temporal_prob"
            ],
            "legacy_v14_event_temporal_prob": grounding[
                "frame_event_relevance"
            ],
            "frame_event_relevance": grounding["frame_event_relevance"],

            # Compatibility aliases for old V1.3 loggers. These must NOT be used
            # to reintroduce the direct reasoning residual in refine.py.
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
