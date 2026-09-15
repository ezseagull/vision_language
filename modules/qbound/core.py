from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound V2.0.0 -- Query-Adaptive Functional Temporal Composition.

    Scientific reset from V1.x
    -------------------------
    V1.x committed to a discrete event partition before grounding, then tried
    to repair errors with event reasoning, span composition/refinement, or
    hypothesis ranking.  V2.0.0 removes that assumption.

    The core hypothesis is:

        video tells us where temporal change may happen;
        the question tells us which changes matter.

    Therefore the core learns a *soft cut field* over adjacent temporal tokens
    instead of a fixed event list.  The number of query-specific latent events
    is emergent and is never capped by ``max_events``.

    Main path
    ---------
        generic video tokens
            -> temporal dynamics encoder
            -> video-only structural boundary prior
            -> bounded question-conditioned cut residual
            -> query-specific soft cut field
            -> boundary-gated temporal reasoning
            -> factorized start/end grounding
            -> exact hard span H* using an O(T) decoder

    The cut field is not an auxiliary branch.  It directly gates temporal
    information flow before the grounding head, so there is no raw-video bypass
    around composition.

    Weak supervision contract
    -------------------------
    This core never receives answer candidates, the ground-truth answer, or
    temporal ground truth.  The downstream training pipeline may use the GT
    answer to construct a stop-gradient functional KEEP-vs-MERGE target for
    each valid adjacent pair.  To support that training objective, the core
    exposes detached source visual states for left/right/merged adjacent units.

    Temporal GT remains evaluation-only.

    Anti-collapse design
    --------------------
    V2.0.0 intentionally has no event-count loss, no minimum event count, and
    no maximum event count.  Collapse is addressed mechanistically:

      1) a video-only dynamics prior anchors temporal structure;
      2) the question can only modify that prior through a bounded residual;
      3) the final cut field causally controls temporal information flow;
      4) a downstream functional composition loss supervises whether merging
         an adjacent distinction damages question-answer evidence.

    One event is allowed when it is truly sufficient.  Dataset-wide one-event
    collapse is therefore detectable but not forbidden by an arbitrary count
    target.

    Complexity
    ----------
    Composition decisions are only made between adjacent temporal tokens, so
    the cut field is O(T).  Long-range reasoning uses a small constant number
    of query anchors, giving O(T * R) rather than O(T^2) event interactions.
    The final factorized span decoder is O(T).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_generic_input_adapter: bool = True,
        input_adapter_dropout: Optional[float] = None,
        input_adapter_use_position: bool = True,
        cut_temperature: float = 1.0,
        cut_threshold: float = 0.5,
        question_residual_scale: float = 1.0,
        num_anchor_heads: int = 4,
        num_local_reasoning_layers: int = 2,
    ) -> None:
        super().__init__()

        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if input_adapter_dropout is None:
            input_adapter_dropout = dropout
        if not 0.0 <= float(input_adapter_dropout) < 1.0:
            raise ValueError("input_adapter_dropout must be in [0, 1).")
        if cut_temperature <= 0.0:
            raise ValueError("cut_temperature must be positive.")
        if not 0.0 < cut_threshold < 1.0:
            raise ValueError("cut_threshold must be in (0, 1).")
        if question_residual_scale <= 0.0:
            raise ValueError("question_residual_scale must be positive.")
        if num_anchor_heads < 1:
            raise ValueError("num_anchor_heads must be >= 1.")
        if num_local_reasoning_layers < 1:
            raise ValueError("num_local_reasoning_layers must be >= 1.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.qbound_version = "2.0.0"

        self.use_generic_input_adapter = bool(use_generic_input_adapter)
        self.input_adapter_use_position = bool(input_adapter_use_position)
        self.input_adapter_dropout_p = float(input_adapter_dropout)
        self.cut_temperature = float(cut_temperature)
        self.cut_threshold = float(cut_threshold)
        self.question_residual_scale = float(question_residual_scale)
        self.num_anchor_heads = int(num_anchor_heads)
        self.num_local_reasoning_layers = int(num_local_reasoning_layers)

        # ------------------------------------------------------------------
        # 0. Generic input adapter.
        # ------------------------------------------------------------------
        # ``content_tokens`` are kept position-free for temporal-change
        # measurement.  Positional encoding is added only to the reasoning
        # stream, avoiding a deterministic sinusoidal difference contaminating
        # the video-only dynamics prior.
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
        # 1. Temporal dynamics encoder.
        # ------------------------------------------------------------------
        self.state_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.accel_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.dynamics_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # 2. Video-only structural boundary prior.
        # ------------------------------------------------------------------
        # The prior has no question input.  It sees adjacent dynamics states and
        # explicit first-/second-order motion magnitudes.
        prior_input_dim = hidden_dim * 4 + 2
        self.video_boundary_encoder = nn.Sequential(
            nn.Linear(prior_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.video_boundary_score = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.video_boundary_score.weight, mean=0.0, std=2e-2)
        nn.init.zeros_(self.video_boundary_score.bias)

        # ------------------------------------------------------------------
        # 3. Question-conditioned functional composition residual.
        # ------------------------------------------------------------------
        self.question_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # left, right, |right-left|, left*right, q, left*q, right*q +
        # video-prior logit + local dynamics magnitude.
        composer_input_dim = hidden_dim * 7 + 2
        self.functional_composer = nn.Sequential(
            nn.Linear(composer_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.question_cut_residual = nn.Linear(hidden_dim, 1)
        # Start from the video-only prior.  Question-specific composition must
        # be learned rather than dominating from the first iteration.
        nn.init.zeros_(self.question_cut_residual.weight)
        nn.init.zeros_(self.question_cut_residual.bias)

        # ------------------------------------------------------------------
        # 4. Boundary-gated local temporal reasoning.
        # ------------------------------------------------------------------
        scan_input_dim = hidden_dim * 4
        self.forward_scan_updates = nn.ModuleList()
        self.forward_scan_gates = nn.ModuleList()
        self.forward_scan_norms = nn.ModuleList()
        self.backward_scan_updates = nn.ModuleList()
        self.backward_scan_gates = nn.ModuleList()
        self.backward_scan_norms = nn.ModuleList()

        for _ in range(self.num_local_reasoning_layers):
            self.forward_scan_updates.append(
                nn.Sequential(
                    nn.Linear(scan_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.forward_scan_gates.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
            )
            self.forward_scan_norms.append(nn.LayerNorm(hidden_dim))

            self.backward_scan_updates.append(
                nn.Sequential(
                    nn.Linear(scan_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            self.backward_scan_gates.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
            )
            self.backward_scan_norms.append(nn.LayerNorm(hidden_dim))

        self.local_reasoning_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # 5. Linear-cost sparse global reasoning through a constant number of
        #    query anchors.  R is a compute budget, NOT an event-count cap.
        # ------------------------------------------------------------------
        self.anchor_score = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_anchor_heads),
        )
        self.anchor_query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.anchor_key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.anchor_value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.anchor_update = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.anchor_update_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reasoning_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # 6. Direct frame-exact grounding.  No 528-hypothesis bank.
        # ------------------------------------------------------------------
        # reasoned, q, reasoned*q + left cut, right cut, local dynamics.
        endpoint_input_dim = hidden_dim * 3 + 3
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(endpoint_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.start_score = nn.Linear(hidden_dim, 1)
        self.end_score = nn.Linear(hidden_dim, 1)
        self.frame_evidence_score = nn.Linear(hidden_dim, 1)
        for head in (self.start_score, self.end_score, self.frame_evidence_score):
            nn.init.normal_(head.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(head.bias)

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sinusoidal_position_encoding(
        time_steps: int,
        feature_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.arange(
            time_steps, device=device, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(
                0, feature_dim, 2, device=device, dtype=torch.float32
            )
            * (-torch.log(torch.tensor(10000.0, device=device)) / feature_dim)
        )

        pe = torch.zeros(
            time_steps, feature_dim, device=device, dtype=torch.float32
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if feature_dim > 1:
            pe[:, 1::2] = torch.cos(
                position * div_term[: pe[:, 1::2].shape[1]]
            )
        return pe.unsqueeze(0).to(dtype=dtype)

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
        if mask.ndim != 2 or mask.shape != (batch_size, time_steps):
            raise ValueError(
                "video_mask must reduce to [B,T]; expected "
                f"{(batch_size, time_steps)}, got {tuple(mask.shape)}."
            )

        mask = mask.bool()
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("Every sample must contain at least one valid video token.")
        return mask

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        mask = mask.bool()
        masked_logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(masked_logits, dim=dim)
        weights = weights * mask.to(weights.dtype)
        return weights / weights.sum(dim=dim, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        mask_f = mask.to(values.dtype)
        while mask_f.ndim < values.ndim:
            mask_f = mask_f.unsqueeze(-1)
        numerator = (values * mask_f).sum(dim=dim)
        denominator = mask_f.sum(dim=dim).clamp_min(1.0)
        return numerator / denominator

    @staticmethod
    def _entropy_from_distribution(
        distribution: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        return -torch.sum(
            distribution * torch.log(distribution.clamp_min(1e-8)), dim=dim
        )

    def _summarize_question(self, question_features: torch.Tensor) -> torch.Tensor:
        """
        Accept [B,D] or [B,L,D].  V2.0.0 keeps the core interface generic.

        Token-level question reasoning can be introduced later at the wrapper
        level without changing the temporal-composition contract.  For the first
        V2 implementation we deliberately keep one question state so that gains
        can be attributed to temporal composition rather than a new language
        encoder.
        """
        if question_features.ndim < 2:
            raise ValueError("question_features must have at least 2 dimensions.")
        if question_features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected final question dimension {self.input_dim}, got "
                f"{question_features.size(-1)}."
            )
        if question_features.ndim == 2:
            return question_features

        batch_size = question_features.size(0)
        tokens = question_features.reshape(batch_size, -1, self.input_dim)
        return tokens.mean(dim=1)

    def _adapt_generic_video_tokens(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Return:
          reasoning_tokens: [B,T,D], with optional temporal position;
          content_tokens:   [B,T,D], position-free for dynamics measurement;
          diagnostics.
        """
        mask_f = video_mask.to(video_tokens.dtype).unsqueeze(-1)
        raw_masked = video_tokens * mask_f

        if self.use_generic_input_adapter:
            content = self.generic_input_proj(video_tokens)
            content = self.generic_input_gelu(content)
            content = self.generic_input_norm(content)
        else:
            content = video_tokens
        content = content * mask_f

        reasoning = content
        if self.input_adapter_use_position:
            position = self._sinusoidal_position_encoding(
                time_steps=video_tokens.size(1),
                feature_dim=video_tokens.size(2),
                device=video_tokens.device,
                dtype=video_tokens.dtype,
            )
            reasoning = reasoning + position * mask_f
        reasoning = self.generic_input_dropout(reasoning) * mask_f

        valid_count = video_mask.sum(dim=-1).to(video_tokens.dtype).clamp_min(1.0)
        feature_count = float(video_tokens.size(-1))
        raw_rms = torch.sqrt(
            raw_masked.pow(2).sum(dim=(1, 2))
            / (valid_count * feature_count).clamp_min(1.0)
            + 1e-8
        )
        adapted_rms = torch.sqrt(
            reasoning.pow(2).sum(dim=(1, 2))
            / (valid_count * feature_count).clamp_min(1.0)
            + 1e-8
        )

        raw_unit = F.normalize(raw_masked, p=2, dim=-1)
        content_unit = F.normalize(content, p=2, dim=-1)
        token_cos = (raw_unit * content_unit).sum(dim=-1)
        token_cos = (
            token_cos * video_mask.to(token_cos.dtype)
        ).sum(dim=-1) / valid_count

        delta = (content - raw_masked) * mask_f
        delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        raw_norm = torch.linalg.vector_norm(raw_masked, dim=-1)
        delta_ratio = (
            delta_norm * video_mask.to(delta_norm.dtype)
        ).sum(dim=-1) / (
            (raw_norm * video_mask.to(raw_norm.dtype))
            .sum(dim=-1)
            .clamp_min(1e-6)
        )

        return reasoning, content, {
            "generic_input_raw_rms": raw_rms,
            "generic_input_adapted_rms": adapted_rms,
            "generic_input_token_cosine": token_cos,
            "generic_input_delta_ratio": delta_ratio,
        }

    # ------------------------------------------------------------------
    # 1. Temporal dynamics
    # ------------------------------------------------------------------
    def _encode_temporal_dynamics(
        self,
        reasoning_tokens: torch.Tensor,
        content_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = content_tokens.shape
        mask_f = video_mask.to(content_tokens.dtype).unsqueeze(-1)

        first_delta = torch.zeros_like(content_tokens)
        if time_steps > 1:
            edge_valid = video_mask[:, 1:] & video_mask[:, :-1]
            first_delta[:, 1:] = content_tokens[:, 1:] - content_tokens[:, :-1]
            first_delta[:, 1:] = (
                first_delta[:, 1:]
                * edge_valid.to(content_tokens.dtype).unsqueeze(-1)
            )

        second_delta = torch.zeros_like(content_tokens)
        if time_steps > 2:
            second_valid = (
                video_mask[:, 2:]
                & video_mask[:, 1:-1]
                & video_mask[:, :-2]
            )
            second_delta[:, 2:] = first_delta[:, 2:] - first_delta[:, 1:-1]
            second_delta[:, 2:] = (
                second_delta[:, 2:]
                * second_valid.to(content_tokens.dtype).unsqueeze(-1)
            )

        state_h = self.state_proj(reasoning_tokens)
        delta_h = self.delta_proj(first_delta)
        accel_h = self.accel_proj(second_delta)
        dynamics_states = self.dynamics_fuse(
            torch.cat([state_h, delta_h, accel_h], dim=-1)
        )
        dynamics_states = dynamics_states * mask_f

        scale = float(max(self.input_dim, 1)) ** 0.5
        delta_magnitude = torch.linalg.vector_norm(first_delta, dim=-1) / scale
        accel_magnitude = torch.linalg.vector_norm(second_delta, dim=-1) / scale
        delta_magnitude = delta_magnitude * video_mask.to(delta_magnitude.dtype)
        accel_magnitude = accel_magnitude * video_mask.to(accel_magnitude.dtype)

        return {
            "dynamics_states": dynamics_states,
            "first_delta": first_delta,
            "second_delta": second_delta,
            "delta_magnitude": delta_magnitude,
            "accel_magnitude": accel_magnitude,
        }

    # ------------------------------------------------------------------
    # 2-3. Video prior + question-conditioned functional cut field
    # ------------------------------------------------------------------
    def _build_functional_cut_field(
        self,
        dynamics_states: torch.Tensor,
        delta_magnitude: torch.Tensor,
        accel_magnitude: torch.Tensor,
        question_state: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = dynamics_states.shape
        boundary_mask = video_mask[:, :-1] & video_mask[:, 1:]

        left = dynamics_states[:, :-1]
        right = dynamics_states[:, 1:]
        edge_delta = delta_magnitude[:, 1:]
        edge_accel = accel_magnitude[:, 1:]

        prior_features = torch.cat(
            [
                left,
                right,
                torch.abs(right - left),
                left * right,
                edge_delta.unsqueeze(-1),
                edge_accel.unsqueeze(-1),
            ],
            dim=-1,
        )
        prior_hidden = self.video_boundary_encoder(prior_features)
        prior_logits_raw = self.video_boundary_score(prior_hidden).squeeze(-1)
        prior_logits = prior_logits_raw.masked_fill(~boundary_mask, 0.0)
        prior_prob = torch.sigmoid(prior_logits / self.cut_temperature)
        prior_prob = prior_prob * boundary_mask.to(prior_prob.dtype)

        q_edge = question_state.unsqueeze(1).expand(-1, time_steps - 1, -1)
        local_variation = (edge_delta + 0.5 * edge_accel).unsqueeze(-1)
        composer_features = torch.cat(
            [
                left,
                right,
                torch.abs(right - left),
                left * right,
                q_edge,
                left * q_edge,
                right * q_edge,
                prior_logits.unsqueeze(-1),
                local_variation,
            ],
            dim=-1,
        )
        composer_hidden = self.functional_composer(composer_features)
        residual_raw = self.question_cut_residual(composer_hidden).squeeze(-1)
        residual_bounded = self.question_residual_scale * torch.tanh(residual_raw)
        residual_bounded = residual_bounded * boundary_mask.to(residual_bounded.dtype)

        # Video prior anchors the topology.  Question semantics can modify but
        # cannot arbitrarily replace it in one step because the residual is
        # bounded in logit space.
        cut_logits = prior_logits + residual_bounded
        cut_prob = torch.sigmoid(cut_logits / self.cut_temperature)
        cut_prob = cut_prob * boundary_mask.to(cut_prob.dtype)

        hard_cut = (cut_prob >= self.cut_threshold) & boundary_mask
        hard_event_index = F.pad(
            torch.cumsum(hard_cut.long(), dim=-1), (1, 0), value=0
        )
        hard_event_index = hard_event_index * video_mask.long()

        valid_edge_count = boundary_mask.sum(dim=-1).to(cut_prob.dtype).clamp_min(1.0)
        cut_mean = cut_prob.sum(dim=-1) / valid_edge_count
        prior_cut_mean = prior_prob.sum(dim=-1) / valid_edge_count
        residual_abs_mean = residual_bounded.abs().sum(dim=-1) / valid_edge_count

        centered = (cut_prob - cut_mean.unsqueeze(-1)) * boundary_mask.to(cut_prob.dtype)
        cut_std = torch.sqrt(
            centered.pow(2).sum(dim=-1) / valid_edge_count + 1e-8
        )

        hard_event_count = 1.0 + hard_cut.sum(dim=-1).to(cut_prob.dtype)
        soft_event_count = 1.0 + cut_prob.sum(dim=-1)
        valid_token_count = video_mask.sum(dim=-1).to(cut_prob.dtype)
        one_event_indicator = (hard_event_count <= 1.0 + 1e-6).to(cut_prob.dtype)
        all_split_indicator = (
            hard_event_count >= valid_token_count.clamp_min(1.0) - 1e-6
        ).to(cut_prob.dtype)

        return {
            "boundary_mask": boundary_mask,
            "video_boundary_prior_logits": prior_logits,
            "video_boundary_prior_prob": prior_prob,
            "question_cut_residual_raw": residual_raw,
            "question_cut_residual": residual_bounded,
            "functional_cut_logits": cut_logits,
            "functional_cut_prob": cut_prob,
            "functional_merge_gate": (1.0 - cut_prob)
            * boundary_mask.to(cut_prob.dtype),
            "hard_cut": hard_cut,
            "hard_event_index": hard_event_index,
            "hard_event_count": hard_event_count,
            "soft_effective_event_count": soft_event_count,
            "cut_mean": cut_mean,
            "cut_std": cut_std,
            "prior_cut_mean": prior_cut_mean,
            "question_residual_abs_mean": residual_abs_mean,
            "one_event_indicator": one_event_indicator,
            "all_split_indicator": all_split_indicator,
        }

    # ------------------------------------------------------------------
    # 4. Boundary-gated local scans
    # ------------------------------------------------------------------
    def _single_direction_scan(
        self,
        states: torch.Tensor,
        question_state: torch.Tensor,
        merge_gate: torch.Tensor,
        video_mask: torch.Tensor,
        update_module: nn.Module,
        gate_module: nn.Module,
        norm_module: nn.Module,
        reverse: bool,
    ) -> torch.Tensor:
        """
        O(T) recurrent information propagation.

        ``merge_gate[t] = 1 - cut_prob[t]`` controls whether information can
        cross the edge between t and t+1.  This is the causal point at which the
        composition field changes the representation used by grounding.
        """
        batch_size, time_steps, hidden_dim = states.shape
        if time_steps == 1:
            return states

        output = torch.zeros_like(states)
        q = question_state

        if not reverse:
            indices = range(time_steps)
            previous = None
            for t in indices:
                current = states[:, t]
                valid = video_mask[:, t]
                if previous is None:
                    new_state = current
                else:
                    structural_gate = merge_gate[:, t - 1]
                    local_input = torch.cat(
                        [current, previous, current * previous, q], dim=-1
                    )
                    message = update_module(local_input)
                    learned_gate = torch.sigmoid(
                        gate_module(
                            torch.cat([current, previous, q], dim=-1)
                        ).squeeze(-1)
                    )
                    gate = structural_gate * learned_gate
                    new_state = norm_module(current + gate.unsqueeze(-1) * message)
                new_state = new_state * valid.to(new_state.dtype).unsqueeze(-1)
                output[:, t] = new_state
                previous = new_state
        else:
            indices = range(time_steps - 1, -1, -1)
            previous = None
            for t in indices:
                current = states[:, t]
                valid = video_mask[:, t]
                if previous is None:
                    new_state = current
                else:
                    structural_gate = merge_gate[:, t]
                    local_input = torch.cat(
                        [current, previous, current * previous, q], dim=-1
                    )
                    message = update_module(local_input)
                    learned_gate = torch.sigmoid(
                        gate_module(
                            torch.cat([current, previous, q], dim=-1)
                        ).squeeze(-1)
                    )
                    gate = structural_gate * learned_gate
                    new_state = norm_module(current + gate.unsqueeze(-1) * message)
                new_state = new_state * valid.to(new_state.dtype).unsqueeze(-1)
                output[:, t] = new_state
                previous = new_state

        return output

    def _boundary_gated_reasoning(
        self,
        dynamics_states: torch.Tensor,
        question_state: torch.Tensor,
        merge_gate: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        forward_states = dynamics_states
        backward_states = dynamics_states

        for layer_idx in range(self.num_local_reasoning_layers):
            forward_states = self._single_direction_scan(
                states=forward_states,
                question_state=question_state,
                merge_gate=merge_gate,
                video_mask=video_mask,
                update_module=self.forward_scan_updates[layer_idx],
                gate_module=self.forward_scan_gates[layer_idx],
                norm_module=self.forward_scan_norms[layer_idx],
                reverse=False,
            )
            backward_states = self._single_direction_scan(
                states=backward_states,
                question_state=question_state,
                merge_gate=merge_gate,
                video_mask=video_mask,
                update_module=self.backward_scan_updates[layer_idx],
                gate_module=self.backward_scan_gates[layer_idx],
                norm_module=self.backward_scan_norms[layer_idx],
                reverse=True,
            )

        q_frames = question_state.unsqueeze(1).expand_as(dynamics_states)
        local_states = self.local_reasoning_fuse(
            torch.cat(
                [
                    dynamics_states,
                    forward_states,
                    backward_states,
                    q_frames,
                    dynamics_states * q_frames,
                ],
                dim=-1,
            )
        )
        local_states = local_states * video_mask.to(local_states.dtype).unsqueeze(-1)

        # Constant-size query anchors provide non-local interaction without an
        # all-pairs event graph.
        anchor_logits = self.anchor_score(
            torch.cat(
                [local_states, q_frames, local_states * q_frames], dim=-1
            )
        )
        anchor_mask = video_mask.unsqueeze(-1).expand_as(anchor_logits)
        anchor_weights = self._masked_softmax(
            anchor_logits, anchor_mask, dim=1
        )
        anchor_tokens = torch.einsum(
            "btr,bth->brh", anchor_weights, local_states
        )

        query = self.anchor_query_proj(local_states)
        key = self.anchor_key_proj(anchor_tokens)
        value = self.anchor_value_proj(anchor_tokens)
        anchor_attention_logits = torch.einsum(
            "bth,brh->btr", query, key
        ) / float(max(self.hidden_dim, 1)) ** 0.5
        anchor_attention = torch.softmax(anchor_attention_logits, dim=-1)
        anchor_context = torch.einsum(
            "btr,brh->bth", anchor_attention, value
        )

        update = self.anchor_update(
            torch.cat(
                [
                    local_states,
                    anchor_context,
                    local_states * anchor_context,
                    q_frames,
                ],
                dim=-1,
            )
        )
        update_gate = torch.sigmoid(
            self.anchor_update_gate(
                torch.cat([local_states, anchor_context, q_frames], dim=-1)
            ).squeeze(-1)
        )
        reasoned_states = self.reasoning_norm(
            local_states + update_gate.unsqueeze(-1) * update
        )
        reasoned_states = (
            reasoned_states * video_mask.to(reasoned_states.dtype).unsqueeze(-1)
        )

        anchor_entropy = self._entropy_from_distribution(
            anchor_weights.transpose(1, 2), dim=-1
        )
        anchor_effective_count = torch.exp(anchor_entropy)

        return {
            "forward_reasoned_states": forward_states,
            "backward_reasoned_states": backward_states,
            "local_reasoned_states": local_states,
            "anchor_logits": anchor_logits,
            "anchor_weights": anchor_weights,
            "anchor_tokens": anchor_tokens,
            "anchor_attention": anchor_attention,
            "anchor_update_gate": update_gate,
            "anchor_effective_token_count": anchor_effective_count,
            "reasoned_states": reasoned_states,
        }

    # ------------------------------------------------------------------
    # 6. Grounding and boundary-equilibrium interface
    # ------------------------------------------------------------------
    @staticmethod
    def _select_span_linear(
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Exact argmax of start[s] + end[e] subject to s <= e, in O(T).

        This replaces the V1.12 O(T^2) materialized hypothesis bank while still
        producing the exact best factorized contiguous span.
        """
        start_detached = start_logits.detach().masked_fill(~video_mask, -1e4)
        end_detached = end_logits.detach().masked_fill(~video_mask, -1e4)

        best_start_score, best_start_index = torch.cummax(start_detached, dim=1)
        end_candidate_score = best_start_score + end_detached
        end_candidate_score = end_candidate_score.masked_fill(~video_mask, -1e4)

        selected_end = end_candidate_score.argmax(dim=-1)
        selected_start = best_start_index.gather(
            1, selected_end.unsqueeze(-1)
        ).squeeze(-1)
        selected_pair_score = end_candidate_score.gather(
            1, selected_end.unsqueeze(-1)
        ).squeeze(-1)
        return selected_start, selected_end, selected_pair_score

    @staticmethod
    def _span_mask_from_bounds(
        start: torch.Tensor,
        end: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_steps = video_mask.size(1)
        index = torch.arange(
            time_steps, device=video_mask.device, dtype=torch.long
        ).view(1, time_steps)
        return (
            (index >= start.unsqueeze(-1))
            & (index <= end.unsqueeze(-1))
            & video_mask
        )

    @staticmethod
    def _mask_to_weights(
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = mask.to(dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def _build_equilibrium_views(
        self,
        selected_start: torch.Tensor,
        selected_end: torch.Tensor,
        video_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps = video_mask.shape
        device = video_mask.device

        first_valid = torch.argmax(video_mask.to(torch.int64), dim=-1)
        reverse_index = torch.argmax(
            torch.flip(video_mask, dims=[1]).to(torch.int64), dim=-1
        )
        last_valid = (time_steps - 1) - reverse_index

        trim_valid = selected_start < selected_end
        expand_left_valid = selected_start > first_valid
        expand_right_valid = selected_end < last_valid

        trim_left_start = torch.where(
            trim_valid, selected_start + 1, selected_start
        )
        trim_left_end = selected_end
        trim_right_start = selected_start
        trim_right_end = torch.where(
            trim_valid, selected_end - 1, selected_end
        )
        expand_left_start = torch.where(
            expand_left_valid, selected_start - 1, selected_start
        )
        expand_left_end = selected_end
        expand_right_start = selected_start
        expand_right_end = torch.where(
            expand_right_valid, selected_end + 1, selected_end
        )

        selected_mask = self._span_mask_from_bounds(
            selected_start, selected_end, video_mask
        )
        trim_left_mask = self._span_mask_from_bounds(
            trim_left_start, trim_left_end, video_mask
        )
        trim_right_mask = self._span_mask_from_bounds(
            trim_right_start, trim_right_end, video_mask
        )
        expand_left_mask = self._span_mask_from_bounds(
            expand_left_start, expand_left_end, video_mask
        )
        expand_right_mask = self._span_mask_from_bounds(
            expand_right_start, expand_right_end, video_mask
        )

        return {
            "selected_span_frame_mask": selected_mask.detach(),
            "selected_span_frame_weights": self._mask_to_weights(selected_mask, dtype),
            "trim_left_valid_mask": trim_valid.detach(),
            "trim_left_frame_mask": trim_left_mask.detach(),
            "trim_left_frame_weights": self._mask_to_weights(trim_left_mask, dtype),
            "trim_right_valid_mask": trim_valid.detach(),
            "trim_right_frame_mask": trim_right_mask.detach(),
            "trim_right_frame_weights": self._mask_to_weights(trim_right_mask, dtype),
            "expand_left_valid_mask": expand_left_valid.detach(),
            "expand_left_frame_mask": expand_left_mask.detach(),
            "expand_left_frame_weights": self._mask_to_weights(expand_left_mask, dtype),
            "expand_right_valid_mask": expand_right_valid.detach(),
            "expand_right_frame_mask": expand_right_mask.detach(),
            "expand_right_frame_weights": self._mask_to_weights(expand_right_mask, dtype),
            "trim_left_start_frame": trim_left_start.detach(),
            "trim_left_end_frame": trim_left_end.detach(),
            "trim_right_start_frame": trim_right_start.detach(),
            "trim_right_end_frame": trim_right_end.detach(),
            "expand_left_start_frame": expand_left_start.detach(),
            "expand_left_end_frame": expand_left_end.detach(),
            "expand_right_start_frame": expand_right_start.detach(),
            "expand_right_end_frame": expand_right_end.detach(),
        }

    def _ground(
        self,
        reasoned_states: torch.Tensor,
        question_state: torch.Tensor,
        cut_prob: torch.Tensor,
        delta_magnitude: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = reasoned_states.shape
        q_frames = question_state.unsqueeze(1).expand_as(reasoned_states)

        # Natural video edges are strong endpoint cues.  Interior endpoint cues
        # come from the learned query-specific cut field.
        left_cut = F.pad(cut_prob, (1, 0), value=1.0)
        right_cut = F.pad(cut_prob, (0, 1), value=1.0)
        left_cut = left_cut * video_mask.to(left_cut.dtype)
        right_cut = right_cut * video_mask.to(right_cut.dtype)

        endpoint_input = torch.cat(
            [
                reasoned_states,
                q_frames,
                reasoned_states * q_frames,
                left_cut.unsqueeze(-1),
                right_cut.unsqueeze(-1),
                delta_magnitude.unsqueeze(-1),
            ],
            dim=-1,
        )
        endpoint_hidden = self.endpoint_encoder(endpoint_input)
        endpoint_hidden = endpoint_hidden * video_mask.to(endpoint_hidden.dtype).unsqueeze(-1)

        start_logits_raw = self.start_score(endpoint_hidden).squeeze(-1)
        end_logits_raw = self.end_score(endpoint_hidden).squeeze(-1)
        frame_evidence_logits_raw = self.frame_evidence_score(endpoint_hidden).squeeze(-1)

        start_logits = start_logits_raw.masked_fill(~video_mask, -1e4)
        end_logits = end_logits_raw.masked_fill(~video_mask, -1e4)
        frame_evidence_logits = frame_evidence_logits_raw.masked_fill(~video_mask, -1e4)

        start_prob = self._masked_softmax(start_logits, video_mask, dim=-1)
        end_prob = self._masked_softmax(end_logits, video_mask, dim=-1)

        # Differentiable O(T) inclusion support used only as a backward bridge
        # for weak QA supervision.  Final inference remains the hard H* below.
        # P(t inside) ~= P(start <= t) * P(end >= t).
        start_cdf = torch.cumsum(start_prob, dim=-1)
        end_survival = torch.flip(
            torch.cumsum(torch.flip(end_prob, dims=[1]), dim=-1), dims=[1]
        )
        soft_span_support = start_cdf * end_survival
        soft_span_support = soft_span_support * video_mask.to(soft_span_support.dtype)
        soft_span_support = soft_span_support / soft_span_support.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        frame_evidence_prob = self._masked_softmax(
            frame_evidence_logits, video_mask, dim=-1
        )
        soft_temporal_prob = soft_span_support * frame_evidence_prob
        soft_temporal_prob = soft_temporal_prob / soft_temporal_prob.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        selected_start, selected_end, selected_pair_score = self._select_span_linear(
            start_logits=start_logits,
            end_logits=end_logits,
            video_mask=video_mask,
        )
        equilibrium = self._build_equilibrium_views(
            selected_start=selected_start,
            selected_end=selected_end,
            video_mask=video_mask,
            dtype=reasoned_states.dtype,
        )
        selected_mask = equilibrium["selected_span_frame_mask"]

        main_temporal_prob = self._masked_softmax(
            frame_evidence_logits, selected_mask, dim=-1
        )
        main_temporal_logits = torch.log(main_temporal_prob.clamp_min(1e-8))
        main_soft_support = selected_mask.to(main_temporal_prob.dtype)

        # Straight-through temporal distribution: forward value is exactly the
        # hard-span main distribution, while the backward pass also reaches
        # start/end probabilities through ``soft_temporal_prob``.  This avoids
        # reintroducing V1.6-style soft-span inference while still making the
        # weak QA loss capable of training the endpoint heads.
        training_temporal_prob = (
            main_temporal_prob
            + soft_temporal_prob
            - soft_temporal_prob.detach()
        )

        valid_length = video_mask.sum(dim=-1).to(reasoned_states.dtype).clamp_min(1.0)
        selected_length = selected_mask.sum(dim=-1).to(reasoned_states.dtype)
        selected_fraction = selected_length / valid_length
        denominator = (valid_length - 1.0).clamp_min(1.0)
        selected_start_norm = (
            selected_start.to(reasoned_states.dtype) / denominator
        ).clamp(0.0, 1.0)
        selected_end_norm = (
            selected_end.to(reasoned_states.dtype) / denominator
        ).clamp(0.0, 1.0)

        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "start_prob": start_prob,
            "end_prob": end_prob,
            "start_entropy": self._entropy_from_distribution(start_prob, dim=-1),
            "end_entropy": self._entropy_from_distribution(end_prob, dim=-1),
            "soft_span_support": soft_span_support,
            "soft_temporal_prob": soft_temporal_prob,
            "training_temporal_prob": training_temporal_prob,
            "frame_evidence_logits": frame_evidence_logits,
            "selected_span_start_frame": selected_start.detach(),
            "selected_span_end_frame": selected_end.detach(),
            "selected_span_start": selected_start_norm,
            "selected_span_end": selected_end_norm,
            "selected_span_length": selected_length,
            "selected_span_fraction": selected_fraction,
            "selected_pair_score": selected_pair_score,
            "main_temporal_prob": main_temporal_prob,
            "main_temporal_logits": main_temporal_logits,
            "main_soft_support": main_soft_support,
            **equilibrium,
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        video_tokens: torch.Tensor,
        question_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,T,D].")
        batch_size, time_steps, feature_dim = video_tokens.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {feature_dim}."
            )
        if time_steps < 2:
            raise ValueError("Q-Bound V2.0.0 requires at least 2 timesteps.")

        video_mask = self._normalize_mask(
            video_mask, batch_size, time_steps, video_tokens.device
        )
        question_summary = self._summarize_question(question_features)
        question_state = self.question_proj(question_summary)

        # Keep the external source tokens for a downstream stop-gradient
        # functional KEEP/MERGE target.  The core itself never sees answers.
        source_video_tokens = video_tokens.detach()

        reasoning_tokens, content_tokens, adapter_diag = (
            self._adapt_generic_video_tokens(video_tokens, video_mask)
        )
        dynamics = self._encode_temporal_dynamics(
            reasoning_tokens=reasoning_tokens,
            content_tokens=content_tokens,
            video_mask=video_mask,
        )
        composition = self._build_functional_cut_field(
            dynamics_states=dynamics["dynamics_states"],
            delta_magnitude=dynamics["delta_magnitude"],
            accel_magnitude=dynamics["accel_magnitude"],
            question_state=question_state,
            video_mask=video_mask,
        )
        reasoning = self._boundary_gated_reasoning(
            dynamics_states=dynamics["dynamics_states"],
            question_state=question_state,
            merge_gate=composition["functional_merge_gate"],
            video_mask=video_mask,
        )
        grounding = self._ground(
            reasoned_states=reasoning["reasoned_states"],
            question_state=question_state,
            cut_prob=composition["functional_cut_prob"],
            delta_magnitude=dynamics["delta_magnitude"],
            video_mask=video_mask,
        )

        # Adjacent source views for downstream functional supervision.  They are
        # detached by construction so a KEEP/MERGE target cannot be gamed by
        # modifying the target representation in the same backward pass.
        functional_left_source = source_video_tokens[:, :-1]
        functional_right_source = source_video_tokens[:, 1:]
        functional_merge_source = 0.5 * (
            functional_left_source + functional_right_source
        )

        return {
            # ----------------------------------------------------------
            # Architecture contract.
            # ----------------------------------------------------------
            "qbound_version": self.qbound_version,
            "functional_composition_enabled": torch.ones(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "dynamic_event_count_enabled": torch.ones(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "fixed_event_partition_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "event_count_cap_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "hypothesis_competition_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "teacher_distillation_enabled": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "answer_features_used_in_core": torch.zeros(
                (), dtype=video_tokens.dtype, device=video_tokens.device
            ),
            "boundary_equilibrium_interface_enabled": torch.ones(
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
            **adapter_diag,

            # Question + dynamics states.
            "question_state": question_state,
            "reasoning_tokens": reasoning_tokens,
            "content_tokens": content_tokens,
            **dynamics,

            # Query-adaptive functional composition.
            **composition,

            # O(T*R) composition-aware reasoning.
            **reasoning,

            # Functional KEEP/MERGE supervision interface.
            "functional_pair_valid_mask": composition["boundary_mask"],
            "functional_left_source": functional_left_source,
            "functional_right_source": functional_right_source,
            "functional_merge_source": functional_merge_source,

            # Direct frame-exact grounding + equilibrium views.
            **grounding,

            # Semantically safe compatibility aliases for downstream temporal
            # consumers.  They do not restore any V1.x hypothesis machinery.
            "event_temporal_prob": grounding["main_temporal_prob"],
            "calibrated_temporal_prob": grounding["main_temporal_prob"],
            "calibrated_temporal_logits": grounding["main_temporal_logits"],
            "calibrated_soft_support": grounding["main_soft_support"],
            "calibrated_span_start_frame": grounding[
                "selected_span_start_frame"
            ],
            "calibrated_span_end_frame": grounding[
                "selected_span_end_frame"
            ],
            "calibrated_frame_mask": grounding["selected_span_frame_mask"],
            "calibrated_span_frame_length": grounding[
                "selected_span_length"
            ],
            "calibrated_span_fraction": grounding["selected_span_fraction"],
            "calibrated_span_start": grounding["selected_span_start"],
            "calibrated_span_end": grounding["selected_span_end"],
        }
