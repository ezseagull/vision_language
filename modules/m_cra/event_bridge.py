from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class GroundedPredictiveEventBridge(nn.Module):
    """
    Grounded Predictive Event Bridge - Version 0.

    Inputs:
        raw_video:
            Video features before CRA temporal refinement.
            Shape: [B, T, input_dim]

        contextual_video:
            Video features after CRA refinement.
            Shape: [B, T, input_dim]

        qa_features:
            Linguistic features.
            Supported shapes:
                [B, input_dim]
                [B, N, input_dim]
                [B, ..., input_dim]

        video_mask:
            Valid timestep mask.
            Shape: [B, T]

        grounding_weights:
            CRA frame-level grounding distribution.
            Shape: [B, T]

    Outputs:
        bridge_repr:
            Aggregated GPEB representation.
            Shape: [B, input_dim]

        event_tokens:
            Ordered event bridge tokens.
            Shape: [B, K, hidden_dim]
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_bins: int = 4,
        use_grounding_prior: bool = True,
    ) -> None:
        super().__init__()

        if num_bins < 1:
            raise ValueError("num_bins must be at least 1.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.use_grounding_prior = use_grounding_prior

        # x before CRA refinement:
        # mainly used to obtain local temporal differences.
        self.raw_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # x_bar after CRA refinement:
        # used as semantic states.
        self.context_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.qa_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Residual FiLM conditioning.
        self.qa_film = nn.Linear(
            hidden_dim,
            hidden_dim * 2,
        )

        # Select relevant local differences using language.
        self.delta_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # delta summary
        # + end - start
        # + middle - start
        # + language
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Decode a semantic state at relative time rho.
        #
        # start + dynamics + language + rho
        self.probe_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # start
        # + dynamics
        # + middle
        # + end
        # + prediction residual
        # + normalized position
        # + duration
        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 5 + 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.event_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        # Learnable strength of CRA grounding prior.
        self.grounding_scale = nn.Parameter(
            torch.tensor(1.0)
        )

    def _summarize_qa(
        self,
        qa_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert QA features to one linguistic vector per sample.

        Current CRA may pass:
            [B, D]
        or:
            [B, number_of_candidates, D]

        Version 0 averages candidate/token dimensions.
        """
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

        # Common shapes:
        # [B, T]
        # [B, 1, T]
        # [B, 1, 1, T]
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
            raise ValueError("video_mask time dimension mismatch.")

        return mask.bool()

    @staticmethod
    def _masked_mean(
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: [B, L, D]
            mask: [B, L]
        """
        weights = mask.to(features.dtype).unsqueeze(-1)

        numerator = torch.sum(
            features * weights,
            dim=1,
        )

        denominator = torch.sum(
            weights,
            dim=1,
        ).clamp_min(1.0)

        return numerator / denominator

    @staticmethod
    def _masked_softmax(
        logits: torch.Tensor,
        mask: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        logits = logits.masked_fill(
            ~mask,
            -1e4,
        )

        weights = torch.softmax(
            logits,
            dim=dim,
        )

        weights = weights * mask.to(weights.dtype)

        denominator = weights.sum(
            dim=dim,
            keepdim=True,
        ).clamp_min(1e-6)

        return weights / denominator

    def _pool_start_middle_end(
        self,
        segment: torch.Tensor,
        segment_mask: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Divide one bin into start, middle and end regions.
        """
        length = segment.size(1)

        if length == 1:
            state = self._masked_mean(
                segment,
                segment_mask,
            )
            return state, state, state

        if length == 2:
            start = self._masked_mean(
                segment[:, :1],
                segment_mask[:, :1],
            )

            middle = self._masked_mean(
                segment,
                segment_mask,
            )

            end = self._masked_mean(
                segment[:, 1:],
                segment_mask[:, 1:],
            )

            return start, middle, end

        first_cut = max(1, length // 3)
        second_cut = max(
            first_cut + 1,
            (2 * length) // 3,
        )
        second_cut = min(
            second_cut,
            length - 1,
        )

        start = self._masked_mean(
            segment[:, :first_cut],
            segment_mask[:, :first_cut],
        )

        middle = self._masked_mean(
            segment[:, first_cut:second_cut],
            segment_mask[:, first_cut:second_cut],
        )

        end = self._masked_mean(
            segment[:, second_cut:],
            segment_mask[:, second_cut:],
        )

        return start, middle, end

    def _decode_probe(
        self,
        start_state: torch.Tensor,
        dynamics_code: torch.Tensor,
        qa_state: torch.Tensor,
        relative_time: float,
    ) -> torch.Tensor:
        rho = torch.full(
            size=(start_state.size(0), 1),
            fill_value=relative_time,
            dtype=start_state.dtype,
            device=start_state.device,
        )

        state_update = self.probe_decoder(
            torch.cat(
                [
                    start_state,
                    dynamics_code,
                    qa_state,
                    rho,
                ],
                dim=-1,
            )
        )

        return start_state + state_update

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

        batch_size, time_steps, feature_dim = raw_video.shape

        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"got {feature_dim}."
            )

        if time_steps < self.num_bins:
            raise ValueError(
                "The number of timesteps must be "
                "at least num_bins."
            )

        video_mask = self._normalize_mask(
            video_mask,
            batch_size,
            time_steps,
            raw_video.device,
        )

        if grounding_weights is not None:
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

        raw_states = self.raw_proj(raw_video)
        contextual_states = self.context_proj(
            contextual_video
        )

        qa_summary = self._summarize_qa(
            qa_features
        )
        qa_state = self.qa_proj(qa_summary)

        gamma, beta = self.qa_film(
            qa_state
        ).chunk(2, dim=-1)

        # Small residual modulation so that the initial
        # representation remains close to visual features.
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

        event_tokens = []
        predicted_middles = []
        target_middles = []
        predicted_ends = []
        target_ends = []
        event_valid_masks = []
        event_grounding_scores = []

        for bin_index in range(self.num_bins):
            start_index = round(
                bin_index
                * time_steps
                / self.num_bins
            )

            end_index = round(
                (bin_index + 1)
                * time_steps
                / self.num_bins
            )

            end_index = max(
                end_index,
                start_index + 1,
            )

            contextual_segment = contextual_states[
                :,
                start_index:end_index,
            ]

            raw_segment = raw_states[
                :,
                start_index:end_index,
            ]

            segment_mask = video_mask[
                :,
                start_index:end_index,
            ]

            start_state, middle_state, end_state = (
                self._pool_start_middle_end(
                    contextual_segment,
                    segment_mask,
                )
            )

            if raw_segment.size(1) > 1:
                local_deltas = (
                    raw_segment[:, 1:]
                    - raw_segment[:, :-1]
                )

                delta_mask = (
                    segment_mask[:, 1:]
                    & segment_mask[:, :-1]
                )

                qa_expanded = qa_state.unsqueeze(1).expand(
                    -1,
                    local_deltas.size(1),
                    -1,
                )

                delta_logits = self.delta_score(
                    torch.cat(
                        [
                            local_deltas,
                            qa_expanded,
                        ],
                        dim=-1,
                    )
                ).squeeze(-1)

                delta_weights = self._masked_softmax(
                    delta_logits,
                    delta_mask,
                    dim=-1,
                )

                delta_summary = torch.sum(
                    delta_weights.unsqueeze(-1)
                    * local_deltas,
                    dim=1,
                )
            else:
                delta_summary = torch.zeros_like(
                    start_state
                )

            dynamics_code = self.dynamics_encoder(
                torch.cat(
                    [
                        delta_summary,
                        end_state - start_state,
                        middle_state - start_state,
                        qa_state,
                    ],
                    dim=-1,
                )
            )

            predicted_middle = self._decode_probe(
                start_state=start_state,
                dynamics_code=dynamics_code,
                qa_state=qa_state,
                relative_time=0.5,
            )

            predicted_end = self._decode_probe(
                start_state=start_state,
                dynamics_code=dynamics_code,
                qa_state=qa_state,
                relative_time=1.0,
            )

            middle_residual = (
                middle_state
                - predicted_middle
            )

            end_residual = (
                end_state
                - predicted_end
            )

            residual_summary = 0.5 * (
                middle_residual
                + end_residual
            )

            normalized_position = torch.full(
                size=(batch_size, 1),
                fill_value=(
                    float(bin_index)
                    / max(self.num_bins - 1, 1)
                ),
                dtype=raw_video.dtype,
                device=raw_video.device,
            )

            normalized_duration = torch.full(
                size=(batch_size, 1),
                fill_value=(
                    float(end_index - start_index)
                    / float(time_steps)
                ),
                dtype=raw_video.dtype,
                device=raw_video.device,
            )

            event_token = self.event_encoder(
                torch.cat(
                    [
                        start_state,
                        dynamics_code,
                        middle_state,
                        end_state,
                        residual_summary,
                        normalized_position,
                        normalized_duration,
                    ],
                    dim=-1,
                )
            )

            event_tokens.append(event_token)
            predicted_middles.append(predicted_middle)
            target_middles.append(
                middle_state.detach()
            )
            predicted_ends.append(predicted_end)
            target_ends.append(
                end_state.detach()
            )

            event_valid = segment_mask.any(
                dim=1
            )
            event_valid_masks.append(event_valid)

            if grounding_weights is None:
                event_grounding = torch.ones(
                    batch_size,
                    dtype=raw_video.dtype,
                    device=raw_video.device,
                )
            else:
                event_grounding = (
                    grounding_weights[
                        :,
                        start_index:end_index,
                    ].sum(dim=1)
                )

            event_grounding_scores.append(
                event_grounding
            )

        event_tokens = torch.stack(
            event_tokens,
            dim=1,
        )

        predicted_middles = torch.stack(
            predicted_middles,
            dim=1,
        )

        target_middles = torch.stack(
            target_middles,
            dim=1,
        )

        predicted_ends = torch.stack(
            predicted_ends,
            dim=1,
        )

        target_ends = torch.stack(
            target_ends,
            dim=1,
        )

        event_valid_mask = torch.stack(
            event_valid_masks,
            dim=1,
        )

        event_grounding = torch.stack(
            event_grounding_scores,
            dim=1,
        )

        event_grounding = (
            event_grounding
            * event_valid_mask.to(
                event_grounding.dtype
            )
        )

        event_grounding = event_grounding / (
            event_grounding.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        qa_expanded = qa_state.unsqueeze(1).expand(
            -1,
            self.num_bins,
            -1,
        )

        event_logits = self.event_score(
            torch.cat(
                [
                    event_tokens,
                    qa_expanded,
                ],
                dim=-1,
            )
        ).squeeze(-1)

        if (
            self.use_grounding_prior
            and grounding_weights is not None
        ):
            event_logits = event_logits + (
                self.grounding_scale
                * torch.log(
                    event_grounding.clamp_min(1e-6)
                )
            )

        event_weights = self._masked_softmax(
            event_logits,
            event_valid_mask,
            dim=-1,
        )

        event_summary = torch.sum(
            event_weights.unsqueeze(-1)
            * event_tokens,
            dim=1,
        )

        bridge_repr = self.output_proj(
            event_summary
        )

        return {
            "bridge_repr": bridge_repr,
            "event_tokens": event_tokens,
            "event_weights": event_weights,
            "event_grounding": event_grounding,
            "predicted_middles": predicted_middles,
            "target_middles": target_middles,
            "predicted_ends": predicted_ends,
            "target_ends": target_ends,
            "event_valid_mask": event_valid_mask,
        }