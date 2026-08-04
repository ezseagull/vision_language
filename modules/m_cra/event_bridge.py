from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroundedPredictiveEventBridge(nn.Module):
    """
    Q-Bound Event Bridge - Version 1.0.

    Core idea:
        Learn a minimum sufficient, question-conditioned event partition.
        Event boundaries are latent and differentiable. A primary monotonic
        chain defines temporal structure, while a tightly bounded,
        non-exclusive concurrent association preserves evidence that may
        overlap across neighboring or simultaneous events. The concurrent
        association refines the same event tokens; it is not a separate
        reasoning branch and it cannot alter the structural boundaries.

        The module also returns original, merge-counterfactual, and
        split-counterfactual representations so the training pipeline can
        learn boundary necessity and within-event split invariance.

    Inputs:
        raw_video:
            Video features before CRA temporal refinement.
            Shape: [B, T, input_dim]

        contextual_video:
            Video features after CRA refinement.
            Shape: [B, T, input_dim]

        qa_features:
            Linguistic conditioning. Prefer a question-only feature [B, D]
            in Version 1.0. Candidate-aggregated features are still accepted
            temporarily for caller compatibility.

        video_mask:
            Valid timestep mask. Shape: [B, T]

        grounding_weights:
            Detached CRA frame-level grounding distribution. Shape: [B, T]

    Main outputs:
        bridge_repr: [B, input_dim]
        fusion_delta: [B, input_dim]
        boundary_prob: [B, T - 1]
        event_assignment: [B, T, K]
        event_tokens: [B, K, hidden_dim]
        transition_tokens: [B, K - 1, hidden_dim]
        frame_event_relevance: [B, T]
        concurrent_assignment: [B, T, K]
        merge_bridge_repr: [B, 1, input_dim]
        split_bridge_repr: [B, 1, input_dim]

    Notes:
        * ``num_bins`` is retained as an argument name for compatibility
          with the Version 0.5 caller, but in Version 1.0 it means the
          maximum number of latent events.
        * Counterfactual selection is soft and differentiable. The reported
          merge/split positions are argmax diagnostics only.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_bins: int = 4,
        use_grounding_prior: bool = True,
        dropout: float = 0.1,
        grounding_prior_clip: float = 4.0,
        grounding_scale_max: float = 5.0,
        boundary_temperature: float = 0.7,
        assignment_temperature: float = 0.35,
        boundary_bias_init: float = -2.0,
        counterfactual_temperature: float = 0.15,
        merge_strength: float = 1.0,
        split_strength: float = 0.9,
        min_event_mass: float = 0.25,
        concurrent_temperature: float = 0.5,
        concurrent_bias_init: float = -2.0,
        concurrent_context_scale: float = 0.10,
        concurrent_degree_max: float = 1.0,
    ) -> None:
        super().__init__()

        if num_bins < 2:
            raise ValueError(
                "Version 1.0 requires num_bins/max_events >= 2."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if grounding_prior_clip <= 0.0:
            raise ValueError(
                "grounding_prior_clip must be positive."
            )
        if grounding_scale_max <= 0.0:
            raise ValueError(
                "grounding_scale_max must be positive."
            )
        if boundary_temperature <= 0.0:
            raise ValueError(
                "boundary_temperature must be positive."
            )
        if assignment_temperature <= 0.0:
            raise ValueError(
                "assignment_temperature must be positive."
            )
        if counterfactual_temperature <= 0.0:
            raise ValueError(
                "counterfactual_temperature must be positive."
            )
        if not 0.0 <= merge_strength <= 1.0:
            raise ValueError("merge_strength must be in [0, 1].")
        if not 0.0 < split_strength <= 1.0:
            raise ValueError("split_strength must be in (0, 1].")
        if min_event_mass <= 0.0:
            raise ValueError("min_event_mass must be positive.")
        if concurrent_temperature <= 0.0:
            raise ValueError(
                "concurrent_temperature must be positive."
            )
        if not 0.0 <= concurrent_context_scale <= 1.0:
            raise ValueError(
                "concurrent_context_scale must be in [0, 1]."
            )
        if concurrent_degree_max <= 0.0:
            raise ValueError(
                "concurrent_degree_max must be positive."
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.max_events = num_bins
        self.use_grounding_prior = use_grounding_prior
        self.grounding_prior_clip = grounding_prior_clip
        self.grounding_scale_max = grounding_scale_max
        self.boundary_temperature = boundary_temperature
        self.assignment_temperature = assignment_temperature
        self.counterfactual_temperature = (
            counterfactual_temperature
        )
        self.merge_strength = merge_strength
        self.split_strength = split_strength
        self.min_event_mass = min_event_mass
        self.concurrent_temperature = concurrent_temperature
        self.concurrent_context_scale = concurrent_context_scale
        self.concurrent_degree_max = concurrent_degree_max

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

        # Small question-conditioned modulation is kept from Version 0.5.
        self.qa_film = nn.Linear(
            hidden_dim,
            hidden_dim * 2,
        )

        # Boundary features:
        # left context, right context, absolute context change,
        # raw local change, left-right interaction, question,
        # grounding(left/right/difference), temporal position.
        boundary_input_dim = hidden_dim * 6 + 4

        self.boundary_feature_encoder = nn.Sequential(
            nn.Linear(boundary_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.boundary_classifier = nn.Linear(
            hidden_dim,
            1,
        )
        # Start from a controlled, nearly uniform boundary prior instead
        # of allowing random classifier weights to saturate the total
        # boundary mass at K - 1 in the first iteration.
        nn.init.normal_(
            self.boundary_classifier.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.constant_(
            self.boundary_classifier.bias,
            boundary_bias_init,
        )

        # Event state:
        # contextual state, raw state, local dynamics,
        # question, event center, duration, grounding mass.
        self.event_encoder = nn.Sequential(
            nn.Linear(
                hidden_dim * 4 + 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Concurrent evidence is not another event decomposition. The
        # monotonic assignment remains the sole structural partition.
        # These projections only test whether a frame also supports a
        # non-primary event prototype. Because the association is sigmoid
        # based, one frame may support more than one event.
        self.concurrent_frame_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.concurrent_event_proj = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=False,
        )
        self.concurrent_frame_norm = nn.LayerNorm(
            hidden_dim
        )
        self.concurrent_event_norm = nn.LayerNorm(
            hidden_dim
        )
        self.concurrent_logit_scale = nn.Parameter(
            torch.tensor(1.38629436112)  # log(4)
        )
        self.concurrent_bias = nn.Parameter(
            torch.tensor(float(concurrent_bias_init))
        )

        # The adapter refines the primary event token using only the
        # context not already represented by its monotonic segment. Small
        # initialization and an explicit scale keep this correction
        # secondary to the primary chain.
        self.concurrent_adapter = nn.Linear(
            hidden_dim,
            hidden_dim,
        )
        nn.init.normal_(
            self.concurrent_adapter.weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.concurrent_adapter.bias)
        self.event_overlap_norm = nn.LayerNorm(
            hidden_dim
        )

        # Transition:
        # left event, right event, event difference,
        # interaction, local boundary state, question,
        # boundary strength.
        self.transition_encoder = nn.Sequential(
            nn.Linear(
                hidden_dim * 6 + 1,
                hidden_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Select evidence at the interpretable event/transition level
        # instead of using a 768-dimensional post-hoc vector gate.
        self.token_score = nn.Sequential(
            nn.Linear(
                hidden_dim * 3,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.bridge_hidden_norm = nn.LayerNorm(
            hidden_dim
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        # Kept from Version 0.5. The caller will cap this residual norm.
        self.fusion_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )

        last_fusion_layer = self.fusion_adapter[-1]
        nn.init.zeros_(last_fusion_layer.weight)
        nn.init.zeros_(last_fusion_layer.bias)

        # Grounding is a prior, not the sole event-selection mechanism.
        self.grounding_scale = nn.Parameter(
            torch.tensor(1.0)
        )

    def _summarize_qa(
        self,
        qa_features: torch.Tensor,
    ) -> torch.Tensor:
        if qa_features.ndim < 2:
            raise ValueError(
                "qa_features must have at least two dimensions."
            )
        if qa_features.size(-1) != self.input_dim:
            raise ValueError(
                "The last QA dimension must equal input_dim. "
                f"Expected {self.input_dim}, "
                f"got {qa_features.size(-1)}."
            )
        if qa_features.ndim == 2:
            return qa_features

        batch_size = qa_features.size(0)
        qa_features = qa_features.reshape(
            batch_size,
            -1,
            self.input_dim,
        )
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
                "video_mask must reduce to [B, T], "
                f"got {tuple(mask.shape)}."
            )
        if mask.size(0) != batch_size:
            raise ValueError("video_mask batch size mismatch.")
        if mask.size(1) != time_steps:
            raise ValueError(
                "video_mask time dimension mismatch."
            )

        return mask.bool()

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        mask = mask.bool()
        masked_logits = logits.masked_fill(
            ~mask,
            -1e4,
        )
        weights = torch.softmax(
            masked_logits,
            dim=dim,
        )
        weights = (
            weights * mask.to(weights.dtype)
        )
        denominator = weights.sum(
            dim=dim,
            keepdim=True,
        ).clamp_min(1e-6)
        return weights / denominator

    @staticmethod
    def _normalize_distribution(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        values = (
            values
            * mask.to(values.dtype)
        )
        denominator = values.sum(
            dim=dim,
            keepdim=True,
        ).clamp_min(1e-6)
        return values / denominator

    @staticmethod
    def _weighted_pool(
        features: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        features: [B, T, D]
        weights: [B, T, K]
        returns: [B, K, D]
        """
        numerator = torch.einsum(
            "btk,btd->bkd",
            weights,
            features,
        )
        denominator = weights.sum(
            dim=1,
        ).unsqueeze(-1).clamp_min(1e-6)
        return numerator / denominator

    def _cap_boundary_mass(
        self,
        boundary_mass: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Keep the cumulative event coordinate within [0, K - 1].
        The number of events remains adaptive when the total boundary
        mass is below K - 1.
        """
        boundary_mass = (
            boundary_mass
            * boundary_mask.to(boundary_mass.dtype)
        )

        max_total = float(self.max_events - 1)
        total_mass = boundary_mass.sum(
            dim=-1,
            keepdim=True,
        )

        scale = torch.clamp(
            max_total
            / total_mass.clamp_min(1e-6),
            max=1.0,
        )

        return boundary_mass * scale

    def _build_assignment(
        self,
        boundary_mass: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert non-negative boundary mass into a monotonic soft event
        assignment.

        boundary_mass: [B, T - 1]
        event_coordinate: [B, T]
        event_assignment: [B, T, K]
        """
        batch_size = boundary_mass.size(0)

        event_coordinate = F.pad(
            torch.cumsum(
                boundary_mass,
                dim=-1,
            ),
            pad=(1, 0),
            value=0.0,
        )

        event_centers = torch.arange(
            self.max_events,
            dtype=boundary_mass.dtype,
            device=boundary_mass.device,
        ).view(1, 1, self.max_events)

        distance = (
            event_coordinate.unsqueeze(-1)
            - event_centers
        )

        assignment_logits = -(
            distance.pow(2)
            / (
                2.0
                * self.assignment_temperature ** 2
            )
        )

        event_assignment = torch.softmax(
            assignment_logits,
            dim=-1,
        )

        event_assignment = (
            event_assignment
            * video_mask.to(
                event_assignment.dtype
            ).unsqueeze(-1)
        )

        # Defensive normalization for valid frames.
        denominator = event_assignment.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        event_assignment = torch.where(
            video_mask.unsqueeze(-1),
            event_assignment / denominator,
            torch.zeros_like(event_assignment),
        )

        if event_assignment.size(0) != batch_size:
            raise RuntimeError(
                "Unexpected event assignment batch size."
            )

        return event_assignment, event_coordinate

    def _build_concurrent_context(
        self,
        contextual_states: torch.Tensor,
        primary_event_tokens: torch.Tensor,
        primary_context_pool: torch.Tensor,
        event_assignment: torch.Tensor,
        event_valid_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Add bounded non-exclusive context to the primary event chain.

        The primary monotonic assignment is detached here on purpose:
        concurrent evidence may enrich event semantics, but it must not
        move or erase the structural boundaries learned by Q-Bound. Each
        frame can support multiple non-primary events through sigmoid
        associations, and the total extra membership per frame is capped.
        """
        frame_latent = self.concurrent_frame_proj(
            contextual_states
        )
        frame_latent = self.concurrent_frame_norm(
            frame_latent
        )
        frame_latent = F.normalize(
            frame_latent,
            p=2,
            dim=-1,
            eps=1e-6,
        )

        # Detached anchors keep this mechanism a semantic correction, not
        # a second path for changing the event partition.
        event_anchors = primary_event_tokens.detach()
        event_latent = self.concurrent_event_proj(
            event_anchors
        )
        event_latent = self.concurrent_event_norm(
            event_latent
        )
        event_latent = F.normalize(
            event_latent,
            p=2,
            dim=-1,
            eps=1e-6,
        )

        logit_scale = torch.clamp(
            self.concurrent_logit_scale.exp(),
            min=1.0,
            max=20.0,
        )

        concurrent_logits = (
            logit_scale
            * torch.einsum(
                "btd,bkd->btk",
                frame_latent,
                event_latent,
            )
            + self.concurrent_bias
        )

        concurrent_assignment = torch.sigmoid(
            concurrent_logits
            / self.concurrent_temperature
        )

        # Concurrent support should add evidence outside the primary
        # membership, not duplicate the dominant monotonic assignment.
        non_primary_capacity = (
            1.0 - event_assignment.detach()
        ).clamp(min=0.0, max=1.0)

        concurrent_assignment = (
            concurrent_assignment
            * non_primary_capacity
            * video_mask.to(
                concurrent_assignment.dtype
            ).unsqueeze(-1)
            * event_valid_mask.to(
                concurrent_assignment.dtype
            ).unsqueeze(1)
        )

        # At most one extra event-equivalent of support is allowed per
        # frame by default. Two events may therefore share a frame, while
        # unrestricted all-to-all smearing is prevented.
        concurrent_degree = concurrent_assignment.sum(
            dim=-1,
            keepdim=True,
        )
        degree_scale = torch.clamp(
            self.concurrent_degree_max
            / concurrent_degree.clamp_min(1e-6),
            max=1.0,
        )
        concurrent_assignment = (
            concurrent_assignment * degree_scale
        )

        concurrent_degree = concurrent_assignment.sum(
            dim=-1
        )
        concurrent_mass = concurrent_assignment.sum(
            dim=1
        )

        concurrent_context = self._weighted_pool(
            contextual_states,
            concurrent_assignment,
        )

        # Use only the residual context not already summarized by the
        # primary monotonic segment.
        concurrent_residual = (
            concurrent_context
            - primary_context_pool.detach()
        )
        concurrent_delta = self.concurrent_adapter(
            concurrent_residual
        )

        event_tokens = self.event_overlap_norm(
            primary_event_tokens
            + self.concurrent_context_scale
            * concurrent_delta
        )

        valid_count = video_mask.sum(
            dim=-1
        ).clamp_min(1).to(
            concurrent_degree.dtype
        )
        concurrent_degree_mean = (
            concurrent_degree
            * video_mask.to(
                concurrent_degree.dtype
            )
        ).sum(dim=-1) / valid_count

        # Diagnostics only. A frame is considered concurrently supported
        # when the extra membership exceeds half an event-equivalent.
        concurrent_frame_fraction = (
            (concurrent_degree > 0.5)
            & video_mask
        ).sum(dim=-1).to(
            concurrent_degree.dtype
        ) / valid_count

        return {
            "event_tokens": event_tokens,
            "concurrent_assignment": (
                concurrent_assignment
            ),
            "concurrent_logits": concurrent_logits,
            "concurrent_mass": concurrent_mass,
            "concurrent_degree": concurrent_degree,
            "concurrent_degree_mean": (
                concurrent_degree_mean
            ),
            "concurrent_frame_fraction": (
                concurrent_frame_fraction
            ),
            "concurrent_logit_scale_used": (
                logit_scale
            ),
            "concurrent_bias_used": self.concurrent_bias,
        }

    def _build_token_grounding_prior(
        self,
        token_grounding: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        log_prior = torch.log(
            token_grounding.clamp_min(1e-6)
        )

        valid = token_mask.to(log_prior.dtype)
        mean = (
            (log_prior * valid).sum(
                dim=-1,
                keepdim=True,
            )
            / valid.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
        )

        log_prior = torch.clamp(
            log_prior - mean,
            min=-self.grounding_prior_clip,
            max=self.grounding_prior_clip,
        )

        return log_prior * valid

    def _encode_partition(
        self,
        raw_states: torch.Tensor,
        contextual_states: torch.Tensor,
        qa_state: torch.Tensor,
        video_mask: torch.Tensor,
        grounding_distribution: torch.Tensor,
        boundary_mass: torch.Tensor,
        boundary_hidden: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = (
            contextual_states.shape
        )

        boundary_mass = self._cap_boundary_mass(
            boundary_mass,
            boundary_mask,
        )

        event_assignment, event_coordinate = (
            self._build_assignment(
                boundary_mass,
                video_mask,
            )
        )

        event_mass = event_assignment.sum(
            dim=1,
        )

        event_valid_mask = (
            event_mass >= self.min_event_mass
        )

        # Event 0 must remain valid whenever a sample has valid frames.
        event_valid_mask[:, 0] = (
            video_mask.any(dim=1)
        )

        local_delta = torch.zeros_like(
            raw_states
        )
        local_delta[:, 1:] = (
            raw_states[:, 1:]
            - raw_states[:, :-1]
        )

        grounding_boost = (
            1.0
            + torch.clamp(
                self.grounding_scale,
                min=0.0,
                max=self.grounding_scale_max,
            )
            * grounding_distribution
        )

        pool_weights = (
            event_assignment
            * grounding_boost.unsqueeze(-1)
        )

        context_pool = self._weighted_pool(
            contextual_states,
            pool_weights,
        )
        raw_pool = self._weighted_pool(
            raw_states,
            pool_weights,
        )
        delta_pool = self._weighted_pool(
            local_delta,
            pool_weights,
        )

        positions = torch.linspace(
            0.0,
            1.0,
            steps=time_steps,
            dtype=contextual_states.dtype,
            device=contextual_states.device,
        ).view(1, time_steps, 1)

        event_center = (
            event_assignment
            * positions
        ).sum(dim=1) / event_mass.clamp_min(1e-6)

        valid_length = video_mask.sum(
            dim=1,
            keepdim=True,
        ).to(contextual_states.dtype).clamp_min(1.0)

        event_duration = (
            event_mass / valid_length
        )

        event_grounding = torch.einsum(
            "btk,bt->bk",
            event_assignment,
            grounding_distribution,
        )

        qa_events = qa_state.unsqueeze(1).expand(
            -1,
            self.max_events,
            -1,
        )

        primary_event_tokens = self.event_encoder(
            torch.cat(
                [
                    context_pool,
                    raw_pool,
                    delta_pool,
                    qa_events,
                    event_center.unsqueeze(-1),
                    event_duration.unsqueeze(-1),
                    event_grounding.unsqueeze(-1),
                ],
                dim=-1,
            )
        )

        concurrent = self._build_concurrent_context(
            contextual_states=contextual_states,
            primary_event_tokens=primary_event_tokens,
            primary_context_pool=context_pool,
            event_assignment=event_assignment,
            event_valid_mask=event_valid_mask,
            video_mask=video_mask,
        )
        event_tokens = concurrent["event_tokens"]

        # Build soft local evidence for each ordered transition slot.
        boundary_mid_coordinate = (
            event_coordinate[:, :-1]
            + 0.5 * boundary_mass
        )

        transition_centers = (
            torch.arange(
                self.max_events - 1,
                dtype=contextual_states.dtype,
                device=contextual_states.device,
            )
            + 0.5
        ).view(1, 1, self.max_events - 1)

        transition_distance = (
            boundary_mid_coordinate.unsqueeze(-1)
            - transition_centers
        )

        transition_slot_logits = -(
            transition_distance.pow(2)
            / (
                2.0
                * self.assignment_temperature ** 2
            )
        )

        transition_slot_affinity = torch.softmax(
            transition_slot_logits,
            dim=-1,
        )

        transition_slot_affinity = (
            transition_slot_affinity
            * boundary_mask.to(
                transition_slot_affinity.dtype
            ).unsqueeze(-1)
        )

        transition_pool_weights = (
            transition_slot_affinity
            * boundary_mass.unsqueeze(-1)
        )

        transition_denominator = (
            transition_pool_weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        normalized_transition_weights = (
            transition_pool_weights
            / transition_denominator
        )

        local_boundary_state = torch.einsum(
            "btk,btd->bkd",
            normalized_transition_weights,
            boundary_hidden,
        )

        transition_strength = torch.sum(
            normalized_transition_weights
            * boundary_mass.unsqueeze(-1),
            dim=1,
        )

        left_event = event_tokens[:, :-1]
        right_event = event_tokens[:, 1:]
        qa_transitions = qa_state.unsqueeze(1).expand(
            -1,
            self.max_events - 1,
            -1,
        )

        transition_tokens = self.transition_encoder(
            torch.cat(
                [
                    left_event,
                    right_event,
                    right_event - left_event,
                    left_event * right_event,
                    local_boundary_state,
                    qa_transitions,
                    transition_strength.unsqueeze(-1),
                ],
                dim=-1,
            )
        )

        transition_valid_mask = (
            event_valid_mask[:, :-1]
            & event_valid_mask[:, 1:]
        )

        transition_grounding = 0.5 * (
            event_grounding[:, :-1]
            + event_grounding[:, 1:]
        )

        token_list = []
        token_mask_list = []
        token_grounding_list = []

        for event_index in range(
            self.max_events
        ):
            token_list.append(
                event_tokens[:, event_index]
            )
            token_mask_list.append(
                event_valid_mask[:, event_index]
            )
            token_grounding_list.append(
                event_grounding[:, event_index]
            )

            if event_index < self.max_events - 1:
                token_list.append(
                    transition_tokens[:, event_index]
                )
                token_mask_list.append(
                    transition_valid_mask[:, event_index]
                )
                token_grounding_list.append(
                    transition_grounding[:, event_index]
                )

        ordered_tokens = torch.stack(
            token_list,
            dim=1,
        )
        token_mask = torch.stack(
            token_mask_list,
            dim=1,
        )
        token_grounding = torch.stack(
            token_grounding_list,
            dim=1,
        )

        qa_tokens = qa_state.unsqueeze(1).expand(
            -1,
            ordered_tokens.size(1),
            -1,
        )

        token_learned_logits = self.token_score(
            torch.cat(
                [
                    ordered_tokens,
                    qa_tokens,
                    ordered_tokens * qa_tokens,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        token_grounding_prior = (
            self._build_token_grounding_prior(
                token_grounding,
                token_mask,
            )
        )

        token_logits = token_learned_logits

        if self.use_grounding_prior:
            token_logits = token_logits + (
                torch.clamp(
                    self.grounding_scale,
                    min=0.0,
                    max=self.grounding_scale_max,
                )
                * token_grounding_prior
            )

        token_weights = self._masked_softmax(
            token_logits,
            token_mask,
            dim=-1,
        )

        bridge_hidden = torch.sum(
            token_weights.unsqueeze(-1)
            * ordered_tokens,
            dim=1,
        )

        bridge_hidden = self.bridge_hidden_norm(
            bridge_hidden
        )

        bridge_repr = self.output_proj(
            bridge_hidden
        )

        event_token_weights = token_weights[:, 0::2]
        transition_token_weights = token_weights[:, 1::2]

        # Structural evidence comes from the monotonic partition. The
        # bounded concurrent association adds support for frames that also
        # explain another event, without creating a second event chain.
        event_support = (
            event_assignment
            + self.concurrent_context_scale
            * concurrent["concurrent_assignment"]
        )

        event_frame_relevance = torch.sum(
            event_support
            * event_token_weights.unsqueeze(1),
            dim=-1,
        )

        transition_frame_relevance = torch.sum(
            transition_slot_affinity
            * transition_token_weights.unsqueeze(1),
            dim=-1,
        )

        transition_frame_relevance = (
            transition_frame_relevance
            * boundary_mass
        )

        frame_transition_map = torch.zeros(
            batch_size,
            time_steps,
            dtype=contextual_states.dtype,
            device=contextual_states.device,
        )

        frame_transition_map[:, :-1] += (
            0.5 * transition_frame_relevance
        )
        frame_transition_map[:, 1:] += (
            0.5 * transition_frame_relevance
        )

        frame_event_relevance = (
            event_frame_relevance
            + frame_transition_map
        )

        frame_event_relevance = (
            frame_event_relevance
            * video_mask.to(
                frame_event_relevance.dtype
            )
        )

        frame_event_relevance = (
            frame_event_relevance
            / frame_event_relevance.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        assignment_entropy_per_frame = -torch.sum(
            event_assignment
            * torch.log(
                event_assignment.clamp_min(1e-8)
            ),
            dim=-1,
        )

        assignment_entropy = (
            assignment_entropy_per_frame
            * video_mask.to(
                assignment_entropy_per_frame.dtype
            )
        ).sum(dim=-1) / video_mask.sum(
            dim=-1
        ).clamp_min(1).to(
            assignment_entropy_per_frame.dtype
        )

        event_mass_distribution = (
            event_mass
            / event_mass.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        event_mass_entropy = -torch.sum(
            event_mass_distribution
            * torch.log(
                event_mass_distribution.clamp_min(1e-8)
            ),
            dim=-1,
        )

        effective_event_count = torch.exp(
            event_mass_entropy
        )

        token_entropy = -torch.sum(
            token_weights
            * torch.log(
                token_weights.clamp_min(1e-8)
            ),
            dim=-1,
        )

        return {
            "boundary_mass": boundary_mass,
            "event_coordinate": event_coordinate,
            "event_assignment": event_assignment,
            "primary_event_tokens": primary_event_tokens,
            "event_tokens": event_tokens,
            "transition_tokens": transition_tokens,
            "concurrent_assignment": (
                concurrent["concurrent_assignment"]
            ),
            "concurrent_logits": (
                concurrent["concurrent_logits"]
            ),
            "concurrent_mass": concurrent["concurrent_mass"],
            "concurrent_degree": (
                concurrent["concurrent_degree"]
            ),
            "concurrent_degree_mean": (
                concurrent["concurrent_degree_mean"]
            ),
            "concurrent_frame_fraction": (
                concurrent["concurrent_frame_fraction"]
            ),
            "concurrent_logit_scale_used": (
                concurrent["concurrent_logit_scale_used"]
            ),
            "concurrent_bias_used": (
                concurrent["concurrent_bias_used"]
            ),
            "concurrent_context_scale_used": (
                torch.as_tensor(
                    self.concurrent_context_scale,
                    dtype=contextual_states.dtype,
                    device=contextual_states.device,
                )
            ),
            "ordered_tokens": ordered_tokens,
            "event_mass": event_mass,
            "event_mass_distribution": (
                event_mass_distribution
            ),
            "event_valid_mask": event_valid_mask,
            "transition_valid_mask": (
                transition_valid_mask
            ),
            "event_grounding": event_grounding,
            "transition_grounding": (
                transition_grounding
            ),
            "transition_slot_affinity": (
                transition_slot_affinity
            ),
            "transition_strength": transition_strength,
            "token_mask": token_mask,
            "token_logits": token_logits,
            "token_learned_logits": (
                token_learned_logits
            ),
            "token_grounding_prior": (
                token_grounding_prior
            ),
            "token_weights": token_weights,
            "bridge_hidden": bridge_hidden,
            "bridge_repr": bridge_repr,
            "frame_event_relevance": (
                frame_event_relevance
            ),
            "assignment_entropy": assignment_entropy,
            "effective_event_count": (
                effective_event_count
            ),
            "token_entropy": token_entropy,
        }

    def _build_counterfactual_selector(
        self,
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Select exactly one valid temporal position while retaining a
        straight-through soft gradient.

        A plain softmax intervention changes every boundary when scores are
        similar, so it is not a true merge/split counterfactual. Here the
        forward pass uses a hard top-1 selector, while backward uses the
        temperature-controlled softmax surrogate.
        """
        mask = mask.bool()
        masked_scores = scores.masked_fill(~mask, -1e4)

        soft_selector = self._masked_softmax(
            masked_scores / self.counterfactual_temperature,
            mask,
            dim=-1,
        )

        hard_index = torch.argmax(
            masked_scores,
            dim=-1,
            keepdim=True,
        )
        hard_selector = torch.zeros_like(
            soft_selector
        ).scatter_(
            dim=-1,
            index=hard_index,
            value=1.0,
        )
        hard_selector = (
            hard_selector
            * mask.to(hard_selector.dtype)
        )

        # Forward: hard top-1. Backward: soft selector.
        return (
            hard_selector
            - soft_selector.detach()
            + soft_selector
        )

    def _build_split_mask(
        self,
        boundary_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Prefer interior positions for the split counterfactual. For very
        short clips, fall back to all valid transition positions.
        """
        time_steps_minus_one = boundary_mask.size(1)

        position = torch.arange(
            time_steps_minus_one,
            device=boundary_mask.device,
        ).view(1, -1)

        valid_length = video_mask.sum(
            dim=-1,
            keepdim=True,
        )

        interior_mask = (
            boundary_mask
            & (position >= 1)
            & (
                position
                < (valid_length - 2)
            )
        )

        has_interior = interior_mask.any(
            dim=-1,
            keepdim=True,
        )

        return torch.where(
            has_interior,
            interior_mask,
            boundary_mask,
        )

    def forward(
        self,
        raw_video: torch.Tensor,
        contextual_video: torch.Tensor,
        qa_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        grounding_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if raw_video.ndim != 3:
            raise ValueError(
                "raw_video must be [B, T, D]."
            )
        if contextual_video.shape != raw_video.shape:
            raise ValueError(
                "raw_video and contextual_video "
                "must have the same shape."
            )

        batch_size, time_steps, feature_dim = (
            raw_video.shape
        )

        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"got {feature_dim}."
            )
        if time_steps < 2:
            raise ValueError(
                "Version 1.0 requires at least 2 timesteps."
            )

        video_mask = self._normalize_mask(
            video_mask,
            batch_size,
            time_steps,
            raw_video.device,
        )

        boundary_mask = (
            video_mask[:, :-1]
            & video_mask[:, 1:]
        )

        if grounding_weights is None:
            grounding_distribution = (
                video_mask.to(raw_video.dtype)
            )
        else:
            if (
                grounding_weights.ndim == 3
                and grounding_weights.size(-1) == 1
            ):
                grounding_weights = (
                    grounding_weights.squeeze(-1)
                )
            if grounding_weights.shape != (
                batch_size,
                time_steps,
            ):
                raise ValueError(
                    "grounding_weights must have shape "
                    f"[B, T], got "
                    f"{tuple(grounding_weights.shape)}."
                )
            grounding_distribution = (
                grounding_weights
                * video_mask.to(
                    grounding_weights.dtype
                )
            )

        grounding_distribution = (
            self._normalize_distribution(
                grounding_distribution,
                video_mask,
                dim=-1,
            )
        )

        raw_states = self.raw_proj(
            raw_video
        )
        contextual_states = self.context_proj(
            contextual_video
        )

        qa_summary = self._summarize_qa(
            qa_features
        )
        qa_state = self.qa_proj(
            qa_summary
        )

        gamma, beta = self.qa_film(
            qa_state
        ).chunk(2, dim=-1)

        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)

        raw_states = (
            raw_states
            * (1.0 + gamma.unsqueeze(1))
            + beta.unsqueeze(1)
        )
        contextual_states = (
            contextual_states
            * (1.0 + gamma.unsqueeze(1))
            + beta.unsqueeze(1)
        )

        left_context = contextual_states[:, :-1]
        right_context = contextual_states[:, 1:]
        raw_change = (
            raw_states[:, 1:]
            - raw_states[:, :-1]
        )

        qa_boundaries = qa_state.unsqueeze(1).expand(
            -1,
            time_steps - 1,
            -1,
        )

        grounding_left = grounding_distribution[:, :-1]
        grounding_right = grounding_distribution[:, 1:]
        grounding_change = (
            grounding_right - grounding_left
        )

        grounding_features = torch.stack(
            [
                grounding_left,
                grounding_right,
                grounding_change,
            ],
            dim=-1,
        )

        boundary_position = torch.linspace(
            0.0,
            1.0,
            steps=time_steps - 1,
            dtype=raw_video.dtype,
            device=raw_video.device,
        ).view(1, time_steps - 1, 1).expand(
            batch_size,
            -1,
            -1,
        )

        boundary_input = torch.cat(
            [
                left_context,
                right_context,
                torch.abs(
                    right_context - left_context
                ),
                raw_change,
                left_context * right_context,
                qa_boundaries,
                grounding_features,
                boundary_position,
            ],
            dim=-1,
        )

        boundary_hidden = (
            self.boundary_feature_encoder(
                boundary_input
            )
        )

        boundary_logits = (
            self.boundary_classifier(
                boundary_hidden
            ).squeeze(-1)
        )

        boundary_prob = torch.sigmoid(
            boundary_logits
            / self.boundary_temperature
        )

        boundary_prob = (
            boundary_prob
            * boundary_mask.to(
                boundary_prob.dtype
            )
        )

        boundary_mass = self._cap_boundary_mass(
            boundary_prob,
            boundary_mask,
        )

        original = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            grounding_distribution=(
                grounding_distribution
            ),
            boundary_mass=boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
        )

        # Differentiable merge counterfactual:
        # softly remove the strongest predicted boundary.
        merge_selector = self._build_counterfactual_selector(
            scores=boundary_mass,
            mask=boundary_mask,
        )

        merge_boundary_mass = boundary_mass * (
            1.0
            - self.merge_strength
            * merge_selector
        )

        merge_output = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            grounding_distribution=(
                grounding_distribution
            ),
            boundary_mass=merge_boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
        )

        # Differentiable split counterfactual:
        # softly add a boundary where the model currently sees the least
        # boundary evidence inside a valid temporal region.
        split_mask = self._build_split_mask(
            boundary_mask,
            video_mask,
        )

        split_selector = self._build_counterfactual_selector(
            scores=-boundary_mass,
            mask=split_mask,
        )

        split_boundary_mass = (
            boundary_mass
            + self.split_strength
            * split_selector
            * (1.0 - boundary_mass)
        )

        split_boundary_mass = self._cap_boundary_mass(
            split_boundary_mass,
            boundary_mask,
        )

        split_output = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            grounding_distribution=(
                grounding_distribution
            ),
            boundary_mass=split_boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
        )

        fusion_delta = self.fusion_adapter(
            original["bridge_repr"]
        )

        boundary_count = boundary_mass.sum(
            dim=-1,
        )

        merge_positions = torch.argmax(
            merge_selector,
            dim=-1,
        ).unsqueeze(-1)

        split_positions = torch.argmax(
            split_selector,
            dim=-1,
        ).unsqueeze(-1)

        return {
            "bridge_repr": original["bridge_repr"],
            "fusion_delta": fusion_delta,
            "bridge_hidden": original["bridge_hidden"],
            "qa_state": qa_state,

            "boundary_logits": boundary_logits,
            "boundary_prob": boundary_prob,
            "boundary_mass": boundary_mass,
            "boundary_mask": boundary_mask,
            "boundary_count": boundary_count,
            "boundary_prob_mean": (
                boundary_prob.sum(dim=-1)
                / boundary_mask.sum(
                    dim=-1
                ).clamp_min(1).to(
                    boundary_prob.dtype
                )
            ),

            "event_coordinate": (
                original["event_coordinate"]
            ),
            "event_assignment": (
                original["event_assignment"]
            ),
            "primary_event_tokens": (
                original["primary_event_tokens"]
            ),
            "event_tokens": original["event_tokens"],
            "concurrent_assignment": (
                original["concurrent_assignment"]
            ),
            "concurrent_logits": (
                original["concurrent_logits"]
            ),
            "concurrent_mass": (
                original["concurrent_mass"]
            ),
            "concurrent_degree": (
                original["concurrent_degree"]
            ),
            "concurrent_degree_mean": (
                original["concurrent_degree_mean"]
            ),
            "concurrent_frame_fraction": (
                original["concurrent_frame_fraction"]
            ),
            "concurrent_logit_scale_used": (
                original["concurrent_logit_scale_used"]
            ),
            "concurrent_bias_used": (
                original["concurrent_bias_used"]
            ),
            "concurrent_context_scale_used": (
                original["concurrent_context_scale_used"]
            ),
            "transition_tokens": (
                original["transition_tokens"]
            ),
            "ordered_tokens": (
                original["ordered_tokens"]
            ),
            "event_mass": original["event_mass"],
            "event_mass_distribution": (
                original["event_mass_distribution"]
            ),
            "event_valid_mask": (
                original["event_valid_mask"]
            ),
            "transition_valid_mask": (
                original["transition_valid_mask"]
            ),
            "event_grounding": (
                original["event_grounding"]
            ),
            "transition_grounding": (
                original["transition_grounding"]
            ),
            "transition_strength": (
                original["transition_strength"]
            ),
            "token_mask": original["token_mask"],
            "token_logits": original["token_logits"],
            "token_learned_logits": (
                original["token_learned_logits"]
            ),
            "token_grounding_prior": (
                original["token_grounding_prior"]
            ),
            "token_weights": original["token_weights"],
            "frame_event_relevance": (
                original["frame_event_relevance"]
            ),

            "merge_bridge_repr": (
                merge_output["bridge_repr"].unsqueeze(1)
            ),
            "merge_frame_relevance": (
                merge_output[
                    "frame_event_relevance"
                ].unsqueeze(1)
            ),
            "merge_boundary_mass": (
                merge_output["boundary_mass"].unsqueeze(1)
            ),
            "merge_selector": (
                merge_selector.unsqueeze(1)
            ),
            "merge_positions": merge_positions,

            "split_bridge_repr": (
                split_output["bridge_repr"].unsqueeze(1)
            ),
            "split_frame_relevance": (
                split_output[
                    "frame_event_relevance"
                ].unsqueeze(1)
            ),
            "split_boundary_mass": (
                split_output["boundary_mass"].unsqueeze(1)
            ),
            "split_selector": (
                split_selector.unsqueeze(1)
            ),
            "split_positions": split_positions,

            "assignment_entropy": (
                original["assignment_entropy"]
            ),
            "effective_event_count": (
                original["effective_event_count"]
            ),
            "token_entropy": original["token_entropy"],
            "grounding_scale_used": torch.clamp(
                self.grounding_scale,
                min=0.0,
                max=self.grounding_scale_max,
            ),
        }
