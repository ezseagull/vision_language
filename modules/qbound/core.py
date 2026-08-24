from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound Version 1.2 core.

    Main changes from V1.1.3
    ------------------------
    1) True dynamic topology.
       Boundary probabilities are no longer accumulated and rescaled to K-1.
       Instead, Q-Bound performs local-peak competition, thresholded dynamic
       activation, and a hard top-(K-1) safety cap. The forward topology is
       therefore allowed to contain 0..K-1 transitions for each sample.

    2) Actual contiguous events in the forward pass.
       Selected boundaries create integer event coordinates and hard contiguous
       event assignments. A straight-through soft surrogate is retained only
       for gradient propagation.

    3) No learned event-activity rescue gate.
       Event existence follows directly from the selected topology. Empty event
       slots are inactive; non-empty slots are active.

    4) No tanh-saturated semantic scoring.
       Event/transition token logits remain ordinary logits. Main temporal
       evidence is produced by event relevance together with a question-aware
       within-event frame scorer.

    5) Weak-alignment-ready outputs.
       The core exposes event_visual_tokens and evidence_visual_token so the
       training pipeline can add a weak cross-modal / QA alignment objective
       without putting dataset-specific supervision inside QBoundCore.

    Backbone independence
    ---------------------
    QBoundCore only consumes generic video/question representations. CRA-specific
    handover, losses, answer heads, and any optional temporal prior remain outside
    the architectural definition of Q-Bound.

    Inputs
    ------
    video_tokens: [B, T, D]
    question_features: [B, D] preferred
    video_mask: [B, T], optional
    contextual_tokens: [B, T, D], optional
    temporal_prior: [B, T], optional detached prior
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        max_events: int = 4,
        use_temporal_prior: bool = False,
        dropout: float = 0.1,
        boundary_temperature: float = 0.7,
        assignment_temperature: float = 0.5,
        boundary_threshold: float = 0.5,
        boundary_radius: int = 2,
        boundary_bias_init: float = -0.2,
        min_event_mass: float = 0.25,
    ) -> None:
        super().__init__()

        if max_events < 2:
            raise ValueError("Version 1.2 requires max_events >= 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if boundary_temperature <= 0.0:
            raise ValueError("boundary_temperature must be positive.")
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive.")
        if not 0.0 < boundary_threshold < 1.0:
            raise ValueError("boundary_threshold must be in (0, 1).")
        if boundary_radius < 0:
            raise ValueError("boundary_radius must be >= 0.")
        if min_event_mass <= 0.0:
            raise ValueError("min_event_mass must be positive.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.max_events = max_events
        self.max_transitions = max_events - 1

        self.use_temporal_prior = bool(use_temporal_prior)

        self.boundary_temperature = float(boundary_temperature)
        self.assignment_temperature = float(assignment_temperature)
        self.boundary_threshold = float(boundary_threshold)
        self.boundary_radius = int(boundary_radius)
        self.min_event_mass = float(min_event_mass)

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

        # Question-conditioned modulation remains intentionally weak.
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

        # V1.1.3 used an almost-zero classifier weight together with a strongly
        # negative bias. Under hard dynamic selection that would make the model
        # start with zero transitions almost everywhere. A small but non-trivial
        # random initialization lets different local peaks compete from the
        # beginning while the negative bias still discourages dense boundaries.
        nn.init.normal_(self.boundary_classifier.weight, mean=0.0, std=2e-2)
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

        # Ordered event / transition relevance. These are logits, not bounded
        # scores. They are normalized where a probability distribution is needed.
        self.token_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.token_score[-1].weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.token_score[-1].bias)

        # Question-aware within-event frame scorer. This is the main V1.2b
        # preparation: event relevance decides *which event* matters, while this
        # scorer decides *where inside that event* the temporal evidence lies.
        self.frame_evidence_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.frame_evidence_score = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.frame_evidence_score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.frame_evidence_score.bias)

    def _summarize_qa(self, qa_features: torch.Tensor) -> torch.Tensor:
        """Compatibility helper. V1.2 should receive question-only [B, D]."""
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
        Return [B,T] detached prior features.

        Disabled/absent prior -> all zeros. The prior never directly determines
        boundaries, event pooling, or temporal logits.
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

    def _select_dynamic_boundaries(
        self,
        boundary_prob: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert dense boundary probabilities [B,T-1] into a genuinely dynamic
        0/1 forward topology.

        Steps:
          1) local maximum competition within +/- boundary_radius;
          2) threshold activation;
          3) keep at most K-1 strongest candidates;
          4) straight-through surrogate for gradients.

        Returns
        -------
        boundary_mass_st:
            [B,T-1], forward values are exactly 0/1.
        boundary_hard:
            [B,T-1], detached binary selector used for diagnostics.
        boundary_candidate_mask:
            [B,T-1], local-peak + threshold candidates before the top-k cap.
        local_peak_value:
            [B,T-1], local max probability for diagnostics.
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

        # Ties are harmless because the global K-1 safety cap below remains.
        local_peak_mask = (
            boundary_prob >= (local_peak_value - 1e-6)
        ) & boundary_mask
        threshold_mask = boundary_prob >= self.boundary_threshold
        candidate_mask = local_peak_mask & threshold_mask

        candidate_scores = boundary_prob.masked_fill(~candidate_mask, -1e4)
        k = min(self.max_transitions, boundary_prob.size(-1))
        top_values, top_indices = torch.topk(candidate_scores, k=k, dim=-1)
        top_valid = top_values > -1e3

        boundary_hard = torch.zeros_like(boundary_prob)
        boundary_hard.scatter_(
            dim=-1,
            index=top_indices,
            src=top_valid.to(boundary_prob.dtype),
        )
        boundary_hard = boundary_hard * boundary_mask.to(boundary_prob.dtype)

        # Surrogate gradient is strongest at local peaks and attenuated away
        # from them. Selection/thresholding remain hard only in the forward pass.
        safe_local_peak = torch.where(
            boundary_mask,
            local_peak_value.clamp_min(1e-6),
            torch.ones_like(local_peak_value),
        )
        local_competition = (
            boundary_prob / safe_local_peak
        ).clamp(min=0.0, max=1.0)
        soft_surrogate = (
            boundary_prob
            * local_competition.detach()
            * boundary_mask.to(boundary_prob.dtype)
        )

        boundary_mass_st = (
            boundary_hard
            + soft_surrogate
            - soft_surrogate.detach()
        )

        return (
            boundary_mass_st,
            boundary_hard.detach(),
            candidate_mask.detach(),
            local_peak_value.detach(),
        )

    def _build_assignment(
        self,
        boundary_mass: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dynamic selected boundaries -> actual contiguous events.

        The forward assignment is hard one-hot and contiguous. Backward uses a
        Gaussian soft assignment around the same event coordinate.
        """
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
        soft_logits = -(
            distance.pow(2) / (2.0 * self.assignment_temperature ** 2)
        )
        soft_assignment = torch.softmax(soft_logits, dim=-1)

        hard_event_index = (
            torch.round(event_coordinate.detach())
            .long()
            .clamp(min=0, max=self.max_events - 1)
        )
        hard_assignment = F.one_hot(
            hard_event_index,
            num_classes=self.max_events,
        ).to(soft_assignment.dtype)

        event_assignment = (
            hard_assignment
            + soft_assignment
            - soft_assignment.detach()
        )
        event_assignment = (
            event_assignment
            * video_mask.to(event_assignment.dtype).unsqueeze(-1)
        )

        return event_assignment, event_coordinate, hard_event_index

    def _build_transition_affinity(
        self,
        event_coordinate: torch.Tensor,
        boundary_mass: torch.Tensor,
        boundary_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map selected frame edges to ordered transition slots with ST gradients."""
        boundary_mid_coordinate = (
            event_coordinate[:, :-1] + 0.5 * boundary_mass
        )
        transition_centers = (
            torch.arange(
                self.max_transitions,
                dtype=boundary_mass.dtype,
                device=boundary_mass.device,
            )
            + 0.5
        ).view(1, 1, self.max_transitions)

        distance = boundary_mid_coordinate.unsqueeze(-1) - transition_centers
        soft_logits = -(
            distance.pow(2) / (2.0 * self.assignment_temperature ** 2)
        )
        soft_affinity = torch.softmax(soft_logits, dim=-1)

        # A real selected boundary after j previous boundaries belongs to slot j.
        hard_slot_index = (
            torch.floor(event_coordinate[:, :-1].detach())
            .long()
            .clamp(min=0, max=self.max_transitions - 1)
        )
        hard_affinity = F.one_hot(
            hard_slot_index,
            num_classes=self.max_transitions,
        ).to(soft_affinity.dtype)

        affinity = hard_affinity + soft_affinity - soft_affinity.detach()
        affinity = (
            affinity
            * boundary_mask.to(affinity.dtype).unsqueeze(-1)
        )
        return affinity

    def _encode_partition(
        self,
        raw_states: torch.Tensor,
        contextual_states: torch.Tensor,
        raw_visual_states: torch.Tensor,
        contextual_visual_states: torch.Tensor,
        qa_state: torch.Tensor,
        video_mask: torch.Tensor,
        temporal_prior: torch.Tensor,
        boundary_mass: torch.Tensor,
        boundary_hidden: torch.Tensor,
        boundary_mask: torch.Tensor,
        frame_semantic_logits: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = contextual_states.shape

        event_assignment, event_coordinate, hard_event_index = self._build_assignment(
            boundary_mass,
            video_mask,
        )

        # Because the forward assignment is hard, these are actual frame counts.
        event_mass = event_assignment.sum(dim=1)
        event_valid_mask = event_mass > self.min_event_mass
        event_valid_mask[:, 0] = video_mask.any(dim=1)
        event_activity = event_valid_mask.to(event_mass.dtype)
        event_count = event_valid_mask.sum(dim=-1).to(event_mass.dtype)

        local_delta = torch.zeros_like(raw_states)
        local_delta[:, 1:] = raw_states[:, 1:] - raw_states[:, :-1]

        # Question-conditioned event semantics used by the downstream task.
        context_pool = self._weighted_pool(contextual_states, event_assignment)
        raw_pool = self._weighted_pool(raw_states, event_assignment)
        delta_pool = self._weighted_pool(local_delta, event_assignment)

        # Weak-alignment-ready visual content. These states are taken before the
        # FiLM modulation so the alignment signal cannot be solved merely by
        # copying the question modulation into the event representation.
        visual_context_pool = self._weighted_pool(
            contextual_visual_states, event_assignment
        )
        visual_raw_pool = self._weighted_pool(
            raw_visual_states, event_assignment
        )
        event_visual_tokens = 0.5 * (
            visual_context_pool + visual_raw_pool
        )

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

        transition_slot_affinity = self._build_transition_affinity(
            event_coordinate=event_coordinate,
            boundary_mass=boundary_mass,
            boundary_mask=boundary_mask,
        )

        transition_pool_weights = (
            transition_slot_affinity * boundary_mass.unsqueeze(-1)
        )
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

        # Forward transition mass is 0 or 1 for a valid dynamic topology.
        transition_strength = 1.0 - torch.exp(-transition_mass)

        left_event = event_tokens[:, :-1]
        right_event = event_tokens[:, 1:]
        qa_transitions = qa_state.unsqueeze(1).expand(
            -1, self.max_transitions, -1
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
            & (transition_mass > 0.5)
        )
        transition_counterfactual_mask = transition_mass > 0.5

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
        token_weights = self._masked_softmax(token_logits, token_mask, dim=-1)

        raw_event_token_logits = token_logits[:, 0::2]
        raw_transition_token_logits = token_logits[:, 1::2]

        # No tanh saturation. Invalid slots are removed by masks when a
        # probability distribution is required.
        event_relevance_weights = self._masked_softmax(
            raw_event_token_logits,
            event_valid_mask,
            dim=-1,
        )
        transition_relevance_weights = self._masked_softmax(
            raw_transition_token_logits,
            transition_valid_mask,
            dim=-1,
        )

        # Project event relevance back to frames. Since the forward assignment
        # is contiguous one-hot, each frame receives the relevance of its event.
        frame_event_gate = torch.einsum(
            "btk,bk->bt",
            event_assignment,
            event_relevance_weights,
        )
        frame_event_gate = (
            frame_event_gate * video_mask.to(frame_event_gate.dtype)
        )

        # Combine event-level selection and within-event semantic evidence in
        # logit space. This avoids double-softmax while preserving a direct
        # topology -> evidence dependency.
        event_frame_logits = torch.log(frame_event_gate.clamp_min(1e-6))

        # Transition evidence is retained for diagnostics/counterfactuals, but
        # it is no longer the dominant main grounding signal. Otherwise the
        # model can learn to peak specifically at boundaries rather than at the
        # evidence frames inside an event.
        transition_edge_logits = torch.sum(
            transition_slot_affinity
            * transition_relevance_weights.unsqueeze(1),
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

        # Main V1.2 grounding logits: event selection + within-event evidence.
        event_temporal_logits = (
            event_frame_logits + frame_semantic_logits
        )
        event_temporal_logits = (
            event_temporal_logits * video_mask.to(event_temporal_logits.dtype)
        )

        # Exactly one temporal softmax inside QBoundCore.
        frame_event_relevance = self._masked_softmax(
            event_temporal_logits,
            video_mask,
            dim=-1,
        )

        # Alignment-ready evidence token: a weighted sum of event visual content.
        # This remains generic; the actual weak alignment loss belongs outside.
        evidence_visual_token = torch.einsum(
            "bk,bkd->bd",
            event_relevance_weights,
            event_visual_tokens,
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
        soft_effective_event_count = torch.exp(event_mass_entropy)

        token_entropy = -torch.sum(
            token_weights * torch.log(token_weights.clamp_min(1e-8)),
            dim=-1,
        )
        event_relevance_entropy = -torch.sum(
            event_relevance_weights
            * torch.log(event_relevance_weights.clamp_min(1e-8)),
            dim=-1,
        )

        return {
            "boundary_mass": boundary_mass,
            "event_coordinate": event_coordinate,
            "hard_event_index": hard_event_index,
            "event_assignment": event_assignment,
            "event_tokens": event_tokens,
            "event_visual_tokens": event_visual_tokens,
            "evidence_visual_token": evidence_visual_token,
            "transition_tokens": transition_tokens,
            "ordered_tokens": ordered_tokens,
            "event_mass": event_mass,
            "event_mass_distribution": event_mass_distribution,
            "event_valid_mask": event_valid_mask,
            "event_activity": event_activity,
            "event_count": event_count,
            "event_prior_mass": event_prior_mass,
            "transition_slot_affinity": transition_slot_affinity,
            "transition_mass": transition_mass,
            "transition_strength": transition_strength,
            "transition_valid_mask": transition_valid_mask,
            "transition_counterfactual_mask": transition_counterfactual_mask,
            "token_mask": token_mask,
            "token_logits": token_logits,
            "token_weights": token_weights,
            "raw_event_token_logits": raw_event_token_logits,
            "raw_transition_token_logits": raw_transition_token_logits,
            "event_relevance_weights": event_relevance_weights,
            "transition_relevance_weights": transition_relevance_weights,
            "frame_event_gate": frame_event_gate,
            "frame_semantic_logits": frame_semantic_logits,
            "event_frame_logits": event_frame_logits,
            "frame_transition_logits": frame_transition_logits,
            "event_temporal_logits": event_temporal_logits,
            "frame_event_relevance": frame_event_relevance,
            "assignment_entropy": assignment_entropy,
            # Compatibility: in V1.2 this is the actual number of non-empty
            # events, not exp(entropy) of a soft fixed-bin mass distribution.
            "effective_event_count": event_count,
            "soft_effective_event_count": soft_effective_event_count,
            "token_entropy": token_entropy,
            "event_relevance_entropy": event_relevance_entropy,
            # Compatibility aliases. Semantics are intentionally updated.
            "active_assignment": event_assignment,
            "transition_activity": transition_valid_mask.to(event_mass.dtype),
            "event_token_scores": raw_event_token_logits,
            "transition_token_scores": raw_transition_token_logits,
        }

    @staticmethod
    def _build_hard_selector(
        scores: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic top-1 selector for transition counterfactuals."""
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
            raise ValueError("Version 1.2 requires at least 2 timesteps.")

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

        # Keep pre-FiLM states for weak alignment so the alignment target does
        # not trivially inherit the question modulation.
        raw_visual_states = self.raw_proj(video_tokens)
        contextual_visual_states = self.context_proj(contextual_tokens)

        qa_summary = self._summarize_qa(question_features)
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

        left_context = contextual_states[:, :-1]
        right_context = contextual_states[:, 1:]
        raw_change = raw_states[:, 1:] - raw_states[:, :-1]
        qa_boundaries = qa_state.unsqueeze(1).expand(
            -1, time_steps - 1, -1
        )

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

        (
            boundary_mass,
            boundary_hard,
            boundary_candidate_mask,
            boundary_local_peak,
        ) = self._select_dynamic_boundaries(
            boundary_prob=boundary_prob,
            boundary_mask=boundary_mask,
        )

        # Question-aware frame semantics, independent of the event slot score.
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
            frame_semantic_logits
            * video_mask.to(frame_semantic_logits.dtype)
        )

        original = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            raw_visual_states=raw_visual_states,
            contextual_visual_states=contextual_visual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            temporal_prior=temporal_prior_used,
            boundary_mass=boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
            frame_semantic_logits=frame_semantic_logits,
        )

        # Transition-level counterfactual merge. With dynamic hard boundaries,
        # removing one selected transition now really changes M -> M-1.
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

        merge_boundary_mass = boundary_mass * (
            1.0 - selected_transition_profile
        )

        merge_output = self._encode_partition(
            raw_states=raw_states,
            contextual_states=contextual_states,
            raw_visual_states=raw_visual_states,
            contextual_visual_states=contextual_visual_states,
            qa_state=qa_state,
            video_mask=video_mask,
            temporal_prior=temporal_prior_used,
            boundary_mass=merge_boundary_mass,
            boundary_hidden=boundary_hidden,
            boundary_mask=boundary_mask,
            frame_semantic_logits=frame_semantic_logits,
        )

        # Actual discrete transition count in the forward pass.
        boundary_count = boundary_hard.sum(dim=-1)
        event_count = boundary_count + 1.0

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

        selected_boundary_prob_sum = (
            boundary_prob * boundary_hard
        ).sum(dim=-1)
        selected_boundary_count = boundary_hard.sum(dim=-1).clamp_min(1.0)
        selected_boundary_prob_mean = (
            selected_boundary_prob_sum / selected_boundary_count
        )
        selected_boundary_prob_mean = torch.where(
            boundary_count > 0,
            selected_boundary_prob_mean,
            torch.zeros_like(selected_boundary_prob_mean),
        )

        return {
            "qa_state": qa_state,

            # Dynamic boundary diagnostics.
            "boundary_logits": boundary_logits,
            "boundary_prob": boundary_prob,
            "boundary_mass": boundary_mass,
            "boundary_hard": boundary_hard,
            "boundary_candidate_mask": boundary_candidate_mask,
            "boundary_local_peak": boundary_local_peak,
            "boundary_mask": boundary_mask,
            "boundary_count": boundary_count,
            "event_count": event_count,
            "boundary_prob_mean": (
                boundary_prob.sum(dim=-1)
                / boundary_mask.sum(dim=-1).clamp_min(1).to(boundary_prob.dtype)
            ),
            "selected_boundary_prob_mean": selected_boundary_prob_mean,

            "event_coordinate": original["event_coordinate"],
            "hard_event_index": original["hard_event_index"],
            "event_assignment": original["event_assignment"],
            "event_tokens": original["event_tokens"],
            "event_visual_tokens": original["event_visual_tokens"],
            "evidence_visual_token": original["evidence_visual_token"],
            "transition_tokens": original["transition_tokens"],
            "ordered_tokens": original["ordered_tokens"],
            "event_mass": original["event_mass"],
            "event_mass_distribution": original["event_mass_distribution"],
            "event_valid_mask": original["event_valid_mask"],
            "event_prior_mass": original["event_prior_mass"],
            "transition_slot_affinity": original["transition_slot_affinity"],
            "transition_mass": original["transition_mass"],
            "transition_strength": original["transition_strength"],
            "transition_counterfactual_mask": original[
                "transition_counterfactual_mask"
            ],
            "transition_valid_mask": original["transition_valid_mask"],
            "token_mask": original["token_mask"],
            "token_logits": original["token_logits"],
            "token_weights": original["token_weights"],

            # Compatibility keys from V1.1.3.
            "event_activity": original["event_activity"],
            "active_assignment": original["active_assignment"],
            "transition_activity": original["transition_activity"],
            "raw_event_token_logits": original["raw_event_token_logits"],
            "raw_transition_token_logits": original[
                "raw_transition_token_logits"
            ],
            "event_token_scores": original["event_token_scores"],
            "transition_token_scores": original["transition_token_scores"],

            # V1.2 evidence decomposition.
            "event_relevance_weights": original["event_relevance_weights"],
            "transition_relevance_weights": original[
                "transition_relevance_weights"
            ],
            "frame_event_gate": original["frame_event_gate"],
            "frame_semantic_logits": original["frame_semantic_logits"],
            "event_frame_logits": original["event_frame_logits"],
            "frame_transition_logits": original["frame_transition_logits"],

            "event_temporal_logits": original["event_temporal_logits"],
            "event_temporal_prob": original["frame_event_relevance"],
            "frame_event_relevance": original["frame_event_relevance"],

            # Transition-level merge counterfactual.
            "merge_event_temporal_logits": (
                merge_output["event_temporal_logits"].unsqueeze(1)
            ),
            "merge_event_temporal_prob": (
                merge_output["frame_event_relevance"].unsqueeze(1)
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
            "merge_transition_index": merge_transition_index.unsqueeze(-1),
            "selected_transition_profile": (
                selected_transition_profile.unsqueeze(1)
            ),
            "selected_transition_mass": selected_transition_mass,
            "selected_transition_strength": selected_transition_strength,
            "removed_boundary_mass": removed_boundary_mass,

            "assignment_entropy": original["assignment_entropy"],
            "effective_event_count": original["effective_event_count"],
            "soft_effective_event_count": original[
                "soft_effective_event_count"
            ],
            "token_entropy": original["token_entropy"],
            "event_relevance_entropy": original["event_relevance_entropy"],

            # Optional prior diagnostics.
            "temporal_prior_used": temporal_prior_used,
            "temporal_prior_enabled": torch.as_tensor(
                float(self.use_temporal_prior),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
        }
