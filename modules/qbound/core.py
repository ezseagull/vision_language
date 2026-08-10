from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound Version 1.1.1 core.

    Backbone-agnostic module for learning a question-conditioned, monotonic
    event decomposition and event-level temporal evidence.

    Inputs:
        video_tokens: [B, T, D]
            Backbone video features before optional contextual refinement.
        question_features: [B, D] (preferred)
            Question-only representation.
        video_mask: [B, T], optional
            Valid-frame mask.
        contextual_tokens: [B, T, D], optional
            Contextualized video features. Falls back to ``video_tokens``.
        temporal_prior: [B, T], optional
            Detached temporal prior from the backbone. When disabled or absent,
            it contributes exactly zero features.

    Core outputs:
        bridge_repr: [B, D]
        event_temporal_logits: [B, T]
        boundary_prob / boundary_mass: [B, T-1]
        event_assignment: [B, T, K]
        event_tokens: [B, K, H]
        transition_tokens: [B, K-1, H]
        merge_* transition-level counterfactual outputs

    Version 1.1.1 removes the weak frame-edge merge and the split branch.
    Counterfactual removal now acts on a whole learned event transition:
    frame-level boundary mass is first grouped into K-1 ordered transition
    slots, then the strongest transition slot is removed as one intervention.

    The core intentionally contains no CRA-specific fusion, no answer head,
    no grounding-consistency loss, no residual cap, and no concurrent branch.
    """


    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        max_events: int = 4,
        use_temporal_prior: bool = True,
        dropout: float = 0.1,
        boundary_temperature: float = 0.7,
        assignment_temperature: float = 0.35,
        boundary_bias_init: float = -2.0,
        min_event_mass: float = 0.25,
    ) -> None:
        super().__init__()

        if max_events < 2:
            raise ValueError("Version 1.1.1 requires max_events >= 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if boundary_temperature <= 0.0:
            raise ValueError("boundary_temperature must be positive.")
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive.")
        if min_event_mass <= 0.0:
            raise ValueError("min_event_mass must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_events = max_events

        self.use_temporal_prior = bool(use_temporal_prior)

        self.boundary_temperature = boundary_temperature
        self.assignment_temperature = assignment_temperature
        self.min_event_mass = min_event_mass

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

        # Retained from V1.0: question-conditioned modulation of visual states.
        self.qa_film = nn.Linear(hidden_dim, hidden_dim * 2)

        # left/right contextual state, abs change, raw change, interaction,
        # question, optional prior(left/right/diff), normalized position.
        boundary_input_dim = hidden_dim * 6 + 4
        self.boundary_feature_encoder = nn.Sequential(
            nn.Linear(boundary_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.boundary_classifier = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.boundary_classifier.weight, mean=0.0, std=1e-3)
        nn.init.constant_(self.boundary_classifier.bias, boundary_bias_init)

        # contextual, raw, dynamics, question, center, duration, prior mass.
        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4 + 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # left event, right event, difference, interaction,
        # local boundary state, question, transition strength.
        self.transition_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 6 + 1, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Question-conditioned selection of ordered event/transition evidence.
        self.token_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        # V1.1.1 removes the external beta gate. Keep the initial Q-Bound
        # temporal perturbation naturally small instead of hiding it behind a
        # learnable scalar that can collapse to zero.
        nn.init.normal_(self.token_score[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.token_score[-1].bias)

        self.bridge_hidden_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def _summarize_qa(self, qa_features: torch.Tensor) -> torch.Tensor:
        """Compatibility helper. V1.1 should receive question-only [B, D]."""
        if qa_features.ndim < 2:
            raise ValueError("qa_features must have at least two dimensions.")
        if qa_features.size(-1) != self.input_dim:
            raise ValueError(
                "The last QA dimension must equal input_dim. "
                f"Expected {self.input_dim}, got {qa_features.size(-1)}."
            )
        if qa_features.ndim == 2:
            return qa_features

        # Kept only so the old caller does not crash before refine.py is
        # refactored. We will verify there that this path is not used.
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
                batch_size, time_steps, dtype=torch.bool, device=device
            )

        while mask.ndim > 2 and mask.size(1) == 1:
            mask = mask.squeeze(1)

        if mask.ndim != 2:
            raise ValueError(
                "video_mask must reduce to [B, T], "
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

    def _prepare_temporal_prior(
        self,
        temporal_prior: Optional[torch.Tensor],
        video_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Return [B,T] prior features.

        Crucial V1.1 behavior:
            * disabled/absent prior -> all zeros, not a uniform distribution;
            * prior is detached so Q-Bound cannot modify the CRA prior source;
            * no direct prior addition to token logits or event pooling.
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

    def _cap_boundary_mass(
        self,
        boundary_mass: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        boundary_mass = boundary_mass * boundary_mask.to(boundary_mass.dtype)
        max_total = float(self.max_events - 1)
        total_mass = boundary_mass.sum(dim=-1, keepdim=True)
        scale = torch.clamp(
            max_total / total_mass.clamp_min(1e-6),
            max=1.0,
        )
        return boundary_mass * scale

    def _build_assignment(
        self,
        boundary_mass: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Boundary mass [B,T-1] -> monotonic assignment [B,T,K]."""
        event_coordinate = F.pad(
            torch.cumsum(boundary_mass, dim=-1),
            pad=(1, 0),
            value=0.0,
        )

        event_centers = torch.arange(
            self.max_events,
            dtype=boundary_mass.dtype,
            device=boundary_mass.device,
        ).view(1, 1, self.max_events)

        distance = event_coordinate.unsqueeze(-1) - event_centers
        assignment_logits = -(
            distance.pow(2) / (2.0 * self.assignment_temperature ** 2)
        )
        event_assignment = torch.softmax(assignment_logits, dim=-1)
        event_assignment = (
            event_assignment * video_mask.to(event_assignment.dtype).unsqueeze(-1)
        )
        denominator = event_assignment.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        event_assignment = torch.where(
            video_mask.unsqueeze(-1),
            event_assignment / denominator,
            torch.zeros_like(event_assignment),
        )
        return event_assignment, event_coordinate

    def _encode_partition(
        self,
        raw_states: torch.Tensor,
        contextual_states: torch.Tensor,
        qa_state: torch.Tensor,
        video_mask: torch.Tensor,
        temporal_prior: torch.Tensor,
        boundary_mass: torch.Tensor,
        boundary_hidden: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = contextual_states.shape

        boundary_mass = self._cap_boundary_mass(boundary_mass, boundary_mask)
        event_assignment, event_coordinate = self._build_assignment(
            boundary_mass, video_mask
        )

        event_mass = event_assignment.sum(dim=1)
        event_valid_mask = event_mass >= self.min_event_mass
        event_valid_mask[:, 0] = video_mask.any(dim=1)

        local_delta = torch.zeros_like(raw_states)
        local_delta[:, 1:] = raw_states[:, 1:] - raw_states[:, :-1]

        # V1.1: event semantics are pooled ONLY by the learned partition.
        # The optional CRA prior is a feature, not a pooling weight.
        context_pool = self._weighted_pool(contextual_states, event_assignment)
        raw_pool = self._weighted_pool(raw_states, event_assignment)
        delta_pool = self._weighted_pool(local_delta, event_assignment)

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

        # Optional prior enters as ONE event-level feature only.
        event_prior_mass = torch.einsum(
            "btk,bt->bk", event_assignment, temporal_prior
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

        # Ordered transition slots are preserved from V1.0.
        boundary_mid_coordinate = event_coordinate[:, :-1] + 0.5 * boundary_mass
        transition_centers = (
            torch.arange(
                self.max_events - 1,
                dtype=contextual_states.dtype,
                device=contextual_states.device,
            )
            + 0.5
        ).view(1, 1, self.max_events - 1)

        transition_distance = (
            boundary_mid_coordinate.unsqueeze(-1) - transition_centers
        )
        transition_slot_logits = -(
            transition_distance.pow(2)
            / (2.0 * self.assignment_temperature ** 2)
        )
        transition_slot_affinity = torch.softmax(
            transition_slot_logits, dim=-1
        )
        transition_slot_affinity = (
            transition_slot_affinity
            * boundary_mask.to(transition_slot_affinity.dtype).unsqueeze(-1)
        )

        transition_pool_weights = (
            transition_slot_affinity * boundary_mass.unsqueeze(-1)
        )

        # Total frame-level boundary mass assigned to each ordered event
        # transition.  This is the correct structural quantity for a
        # transition-level counterfactual; the previous V1.1 code used a
        # weighted average of frame-edge probabilities, which stayed near
        # ~0.05 and made a merge almost ineffectual.
        transition_mass = transition_pool_weights.sum(dim=1)

        transition_denominator = transition_mass.unsqueeze(1).clamp_min(1e-6)
        normalized_transition_weights = (
            transition_pool_weights / transition_denominator
        )

        local_boundary_state = torch.einsum(
            "btk,btd->bkd",
            normalized_transition_weights,
            boundary_hidden,
        )

        # Smooth probability-like existence strength for each transition.
        # If soft boundary masses in one transition accumulate to ~1, the
        # strength is already substantial (~0.63) without any new hyperparameter.
        transition_strength = 1.0 - torch.exp(-transition_mass)

        left_event = event_tokens[:, :-1]
        right_event = event_tokens[:, 1:]
        qa_transitions = qa_state.unsqueeze(1).expand(
            -1, self.max_events - 1, -1
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
            event_valid_mask[:, :-1] & event_valid_mask[:, 1:]
        )

        # Counterfactual selection must not disappear merely because one soft
        # event fell below min_event_mass.  A transition is structurally
        # selectable whenever it still owns non-zero boundary mass.
        transition_counterfactual_mask = transition_mass > 1e-6

        # Interleave ordered event and transition tokens:
        # E0, R01, E1, R12, E2, ...
        token_list = []
        token_mask_list = []
        for event_index in range(self.max_events):
            token_list.append(event_tokens[:, event_index])
            token_mask_list.append(event_valid_mask[:, event_index])
            if event_index < self.max_events - 1:
                token_list.append(transition_tokens[:, event_index])
                token_mask_list.append(transition_valid_mask[:, event_index])

        ordered_tokens = torch.stack(token_list, dim=1)
        token_mask = torch.stack(token_mask_list, dim=1)

        qa_tokens = qa_state.unsqueeze(1).expand(
            -1, ordered_tokens.size(1), -1
        )
        token_logits = self.token_score(
            torch.cat(
                [ordered_tokens, qa_tokens, ordered_tokens * qa_tokens],
                dim=-1,
            )
        ).squeeze(-1)

        # V1.1: NO direct CRA-prior addition here.
        token_weights = self._masked_softmax(token_logits, token_mask, dim=-1)
        bridge_hidden = torch.sum(
            token_weights.unsqueeze(-1) * ordered_tokens,
            dim=1,
        )
        bridge_hidden = self.bridge_hidden_norm(bridge_hidden)
        bridge_repr = self.output_proj(bridge_hidden)

        # ------------------------------------------------------------------
        # Event temporal logits for direct grounding integration in refine.py.
        # These are generated from Q-Bound's learned event/transition scores,
        # not copied from CRA grounding. V1.1.1 intentionally uses no external
        # alpha/beta bypass gate.
        # ------------------------------------------------------------------
        event_token_logits = token_logits[:, 0::2]
        transition_token_logits = token_logits[:, 1::2]

        event_token_logits = event_token_logits * event_valid_mask.to(
            event_token_logits.dtype
        )
        transition_token_logits = transition_token_logits * (
            transition_valid_mask.to(transition_token_logits.dtype)
        )

        event_frame_logits = torch.einsum(
            "btk,bk->bt", event_assignment, event_token_logits
        )

        transition_edge_logits = torch.sum(
            transition_slot_affinity
            * transition_token_logits.unsqueeze(1),
            dim=-1,
        )
        transition_edge_logits = transition_edge_logits * boundary_mass

        frame_transition_logits = torch.zeros(
            batch_size,
            time_steps,
            dtype=contextual_states.dtype,
            device=contextual_states.device,
        )
        frame_transition_logits[:, :-1] += 0.5 * transition_edge_logits
        frame_transition_logits[:, 1:] += 0.5 * transition_edge_logits

        event_temporal_logits = (
            event_frame_logits + frame_transition_logits
        ) * video_mask.to(event_frame_logits.dtype)

        # No external beta gate and no per-sample standardization in V1.1.1.
        # The token scorer is initialized near zero, so Q-Bound starts as a
        # small direct temporal correction and must earn larger influence from
        # the main task itself.
        frame_event_relevance = self._masked_softmax(
            event_temporal_logits,
            video_mask,
            dim=-1,
        )

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
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        event_mass_entropy = -torch.sum(
            event_mass_distribution
            * torch.log(event_mass_distribution.clamp_min(1e-8)),
            dim=-1,
        )
        effective_event_count = torch.exp(event_mass_entropy)

        token_entropy = -torch.sum(
            token_weights * torch.log(token_weights.clamp_min(1e-8)),
            dim=-1,
        )

        return {
            "boundary_mass": boundary_mass,
            "event_coordinate": event_coordinate,
            "event_assignment": event_assignment,
            "event_tokens": event_tokens,
            "transition_tokens": transition_tokens,
            "ordered_tokens": ordered_tokens,
            "event_mass": event_mass,
            "event_mass_distribution": event_mass_distribution,
            "event_valid_mask": event_valid_mask,
            "transition_valid_mask": transition_valid_mask,
            "event_prior_mass": event_prior_mass,
            "transition_slot_affinity": transition_slot_affinity,
            "transition_mass": transition_mass,
            "transition_strength": transition_strength,
            "transition_counterfactual_mask": transition_counterfactual_mask,
            "token_mask": token_mask,
            "token_logits": token_logits,
            "token_weights": token_weights,
            "bridge_hidden": bridge_hidden,
            "bridge_repr": bridge_repr,
            "event_temporal_logits": event_temporal_logits,
            "frame_event_relevance": frame_event_relevance,
            "assignment_entropy": assignment_entropy,
            "effective_event_count": effective_event_count,
            "token_entropy": token_entropy,
        }

    @staticmethod
    def _build_hard_selector(
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic top-1 intervention selector.

        The selector itself is not a learned routing mechanism.  It only
        chooses which existing transition to ablate, so V1.1.1 removes the
        straight-through soft selector and its temperature hyperparameter.
        """
        if scores.shape != mask.shape:
            raise ValueError(
                "scores and mask must have the same shape, got "
                f"{tuple(scores.shape)} and {tuple(mask.shape)}."
            )

        mask = mask.bool()
        masked_scores = scores.masked_fill(~mask, -1e4)
        hard_index = torch.argmax(masked_scores, dim=-1, keepdim=True)

        selector = torch.zeros_like(scores).scatter_(
            dim=-1,
            index=hard_index,
            value=1.0,
        )

        has_valid = mask.any(dim=-1, keepdim=True)
        selector = selector * mask.to(selector.dtype)
        selector = selector * has_valid.to(selector.dtype)
        return selector

    def forward(
        self,
        video_tokens: torch.Tensor,
        question_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        contextual_tokens: Optional[torch.Tensor] = None,
        temporal_prior: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B, T, D].")

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
            raise ValueError("Version 1.1.1 requires at least 2 timesteps.")

        video_mask = self._normalize_mask(
            video_mask,
            batch_size,
            time_steps,
            video_tokens.device,
        )
        boundary_mask = video_mask[:, :-1] & video_mask[:, 1:]

        # The optional prior is detached inside _prepare_temporal_prior().
        temporal_prior_used = self._prepare_temporal_prior(
            temporal_prior,
            video_mask,
            video_tokens.dtype,
        )

        raw_states = self.raw_proj(video_tokens)
        contextual_states = self.context_proj(contextual_tokens)

        qa_summary = self._summarize_qa(question_features)
        qa_state = self.qa_proj(qa_summary)

        gamma, beta = self.qa_film(qa_state).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma)
        beta = 0.1 * torch.tanh(beta)
        raw_states = (
            raw_states * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        )
        contextual_states = (
            contextual_states * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        )

        left_context = contextual_states[:, :-1]
        right_context = contextual_states[:, 1:]
        raw_change = raw_states[:, 1:] - raw_states[:, :-1]
        qa_boundaries = qa_state.unsqueeze(1).expand(
            -1, time_steps - 1, -1
        )

        # Optional prior features. When no prior is used they are all zero.
        prior_left = temporal_prior_used[:, :-1]
        prior_right = temporal_prior_used[:, 1:]
        prior_change = prior_right - prior_left
        prior_features = torch.stack(
            [prior_left, prior_right, prior_change], dim=-1
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
                left_context,
                right_context,
                torch.abs(right_context - left_context),
                raw_change,
                left_context * right_context,
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
        boundary_mass = self._cap_boundary_mass(boundary_prob, boundary_mask)

        original = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            temporal_prior=temporal_prior_used,
            boundary_mass=boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
        )

        # --------------------------------------------------------------
        # Transition-level counterfactual merge.
        #
        # V1.1 removed only one frame edge (~0.05 mass), while the event
        # coordinate was created by the cumulative mass of many such edges.
        # That intervention barely changed the partition.  V1.1.1 instead:
        #   1) groups frame-edge mass into K-1 ordered transition slots,
        #   2) selects the strongest transition,
        #   3) removes all boundary mass attributed to that transition.
        # --------------------------------------------------------------
        merge_transition_selector = self._build_hard_selector(
            scores=original["transition_strength"],
            mask=original["transition_counterfactual_mask"],
        )

        selected_transition_profile = torch.einsum(
            "bk,btk->bt",
            merge_transition_selector,
            original["transition_slot_affinity"],
        )
        selected_transition_profile = (
            selected_transition_profile
            * boundary_mask.to(selected_transition_profile.dtype)
        )

        # Because transition_slot_affinity sums to one over transition slots,
        # this removes exactly the component of boundary_mass attributed to
        # the selected transition, rather than deleting one arbitrary frame edge.
        merge_boundary_mass = boundary_mass * (
            1.0 - selected_transition_profile
        )

        merge_output = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            temporal_prior=temporal_prior_used,
            boundary_mass=merge_boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
        )

        boundary_count = boundary_mass.sum(dim=-1)

        selected_transition_mass = torch.sum(
            merge_transition_selector * original["transition_mass"],
            dim=-1,
        )
        selected_transition_strength = torch.sum(
            merge_transition_selector * original["transition_strength"],
            dim=-1,
        )
        removed_boundary_mass = (
            boundary_mass - merge_boundary_mass
        ).sum(dim=-1)

        merge_transition_index = torch.argmax(
            merge_transition_selector,
            dim=-1,
        )

        return {
            "bridge_repr": original["bridge_repr"],
            "bridge_hidden": original["bridge_hidden"],
            "qa_state": qa_state,

            "boundary_logits": boundary_logits,
            "boundary_prob": boundary_prob,
            "boundary_mass": boundary_mass,
            "boundary_mask": boundary_mask,
            "boundary_count": boundary_count,
            "boundary_prob_mean": (
                boundary_prob.sum(dim=-1)
                / boundary_mask.sum(dim=-1).clamp_min(1).to(boundary_prob.dtype)
            ),

            "event_coordinate": original["event_coordinate"],
            "event_assignment": original["event_assignment"],
            "event_tokens": original["event_tokens"],
            "transition_tokens": original["transition_tokens"],
            "ordered_tokens": original["ordered_tokens"],
            "event_mass": original["event_mass"],
            "event_mass_distribution": original["event_mass_distribution"],
            "event_valid_mask": original["event_valid_mask"],
            "transition_valid_mask": original["transition_valid_mask"],
            "event_prior_mass": original["event_prior_mass"],
            "transition_slot_affinity": original["transition_slot_affinity"],
            "transition_mass": original["transition_mass"],
            "transition_strength": original["transition_strength"],
            "transition_counterfactual_mask": (
                original["transition_counterfactual_mask"]
            ),
            "token_mask": original["token_mask"],
            "token_logits": original["token_logits"],
            "token_weights": original["token_weights"],

            "event_temporal_logits": original["event_temporal_logits"],
            "frame_event_relevance": original["frame_event_relevance"],

            # Transition-level merge counterfactual.
            "merge_event_temporal_logits": (
                merge_output["event_temporal_logits"].unsqueeze(1)
            ),
            "merge_frame_relevance": (
                merge_output["frame_event_relevance"].unsqueeze(1)
            ),
            "merge_boundary_mass": (
                merge_output["boundary_mass"].unsqueeze(1)
            ),
            "merge_transition_selector": (
                merge_transition_selector.unsqueeze(1)
            ),
            "merge_transition_index": (
                merge_transition_index.unsqueeze(-1)
            ),
            "selected_transition_profile": (
                selected_transition_profile.unsqueeze(1)
            ),
            "selected_transition_mass": selected_transition_mass,
            "selected_transition_strength": selected_transition_strength,
            "removed_boundary_mass": removed_boundary_mass,

            "assignment_entropy": original["assignment_entropy"],
            "effective_event_count": original["effective_event_count"],
            "token_entropy": original["token_entropy"],

            # Diagnostics for the CRA-prior ablation.
            "temporal_prior_used": temporal_prior_used,
            "temporal_prior_enabled": torch.as_tensor(
                float(self.use_temporal_prior),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
        }

