from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound V2.4.0 -- Question-Conditioned Core-to-Closure Grounding.

    V2.4 factorizes temporal grounding into:
      1) Evidence Core a_t: discriminative question-specific evidence.
      2) Event Continuity j_t: whether the supporting event continues across
         adjacent frames.

    Final contiguous-span scores are induced from these two local structural
    variables rather than learned as unrelated span identities.

    Important contracts:
      - Frozen/local CLIP features remain the visual source.
      - Temporal enrichment is used for WHERE/localization only.
      - value_tokens preserve the original masked CLIP features for downstream
        evidence pooling.
      - Question is injected before temporal localization.
      - No answers, temporal GT, CRA temporal features, BoundaryRefiner,
        adaptive radius, endpoint heads, learned routing gates, length priors,
        Gaussian smoothing, or event-count logic are used here.
      - Hard MAP and soft frame membership come from the SAME P_Q(s,e).

    Core-to-Closure energy
    ----------------------
    Let a_t,j_t in [-1,1]. For H=[s,e]:

      C_core   = max_{t in H} a_t
      C_inside = mean_{t=s..e-1} j_t, or 0 when s=e
      C_cut    = mean(-j_{s-1}, -j_e) over real outside boundary edges only

      E_Q(s,e) = C_core + C_inside + C_cut

    This avoids three known shortcuts:
      - no raw support sum that rewards long spans;
      - no mean core score whose maximum is necessarily a singleton;
      - no product of continuation probabilities that penalizes long spans
        exponentially.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_question_heads: int = 4,
        question_ff_multiplier: int = 2,
        segment_temperature: float = 1.0,
    ) -> None:
        super().__init__()

        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1).")
        if num_question_heads < 1 or hidden_dim % num_question_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_question_heads."
            )
        if question_ff_multiplier < 1:
            raise ValueError("question_ff_multiplier must be >= 1.")
        if segment_temperature <= 0.0:
            raise ValueError("segment_temperature must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.num_question_heads = int(num_question_heads)
        self.question_ff_multiplier = int(question_ff_multiplier)
        self.segment_temperature = float(segment_temperature)
        self.qbound_version = "2.4.0"

        # WHERE input projection. VALUE remains the original CLIP feature.
        self.where_input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # One small temporal-context block, no learned gate.
        self.temporal_norm1 = nn.LayerNorm(self.hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_question_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.temporal_attn_delta = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )

        ff_dim = self.hidden_dim * self.question_ff_multiplier
        self.temporal_norm2 = nn.LayerNorm(self.hidden_dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(ff_dim, self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )

        # Question tokens + early Q -> V interaction.
        self.question_token_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.qv_norm = nn.LayerNorm(self.hidden_dim)
        self.qv_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_question_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.qv_attn_delta = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
        )
        self.qv_ff_norm = nn.LayerNorm(self.hidden_dim)
        self.qv_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(ff_dim, self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )

        # Evidence Core head.
        self.core_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )
        self.core_score = nn.Linear(self.hidden_dim, 1)

        # Event Continuity = visual structure + additive Q-specific correction.
        self.visual_continuity_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )
        self.visual_continuity_score = nn.Linear(self.hidden_dim, 1)

        self.question_continuity_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )
        self.question_continuity_score = nn.Linear(self.hidden_dim, 1)

        self._initialize_v240()

    def _initialize_v240(self) -> None:
        # Small residual changes preserve the projected visual state initially
        # without introducing any learned routing gate.
        for layer in (self.temporal_attn_delta, self.qv_attn_delta):
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(layer.bias)

        for ff in (self.temporal_ff, self.qv_ff):
            last_linear = None
            for module in ff.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is not None:
                nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
                nn.init.zeros_(last_linear.bias)

        for head in (
            self.core_score,
            self.visual_continuity_score,
            self.question_continuity_score,
        ):
            nn.init.normal_(head.weight, mean=0.0, std=1e-2)
            nn.init.zeros_(head.bias)

        # Q-specific continuity starts as a small correction.
        nn.init.normal_(
            self.question_continuity_score.weight,
            mean=0.0,
            std=1e-3,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_video_mask(
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

        if mask.ndim != 2 or mask.shape != (batch_size, time_steps):
            raise ValueError(
                "video_mask must reduce to [B,T]; expected "
                f"{(batch_size, time_steps)}, got {tuple(mask.shape)}."
            )

        mask = mask.bool()
        if not torch.all(mask.any(dim=-1)):
            raise ValueError(
                "Every sample must contain at least one valid video token."
            )
        return mask

    def _normalize_question_features(
        self,
        question_features: torch.Tensor,
        question_mask: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if question_features.ndim == 2:
            if question_features.shape != (batch_size, self.input_dim):
                raise ValueError(
                    "2-D question_features must be [B,D]; expected "
                    f"{(batch_size, self.input_dim)}, got "
                    f"{tuple(question_features.shape)}."
                )
            question_features = question_features.unsqueeze(1)

        if question_features.ndim != 3:
            raise ValueError(
                "question_features must be [B,D] or [B,L,D]."
            )
        if question_features.size(0) != batch_size:
            raise ValueError("Video/question batch sizes do not match.")
        if question_features.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected question feature dim {self.input_dim}, got "
                f"{question_features.size(-1)}."
            )

        q_len = question_features.size(1)
        if question_mask is None:
            q_mask = torch.ones(
                batch_size,
                q_len,
                dtype=torch.bool,
                device=device,
            )
        else:
            while question_mask.ndim > 2 and question_mask.size(1) == 1:
                question_mask = question_mask.squeeze(1)
            if question_mask.ndim != 2 or question_mask.shape != (
                batch_size,
                q_len,
            ):
                raise ValueError(
                    "question_mask must reduce to [B,L]; expected "
                    f"{(batch_size, q_len)}, got "
                    f"{tuple(question_mask.shape)}."
                )
            q_mask = question_mask.bool()
            if not torch.all(q_mask.any(dim=-1)):
                raise ValueError(
                    "Every sample must contain at least one valid question token."
                )

        return question_features, q_mask

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
    def _entropy_from_flat_distribution(
        flat_prob: torch.Tensor,
        flat_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = flat_prob * flat_valid.to(flat_prob.dtype)
        entropy = -(
            p * torch.log(p.clamp_min(1e-8))
        ).sum(dim=-1)
        valid_count = (
            flat_valid.sum(dim=-1)
            .to(flat_prob.dtype)
            .clamp_min(1.0)
        )
        max_entropy = torch.log(valid_count.clamp_min(2.0))
        ratio = torch.where(
            valid_count > 1.0,
            entropy / max_entropy.clamp_min(1e-6),
            torch.zeros_like(entropy),
        )
        return entropy, ratio

    @staticmethod
    def _span_mask_from_bounds(
        start: torch.Tensor,
        end: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> torch.Tensor:
        time_steps = video_mask.size(1)
        index = torch.arange(
            time_steps,
            device=video_mask.device,
            dtype=torch.long,
        ).view(1, time_steps)
        return (
            (index >= start.unsqueeze(-1))
            & (index <= end.unsqueeze(-1))
            & video_mask
        )

    @staticmethod
    def _uniform_weights_from_mask(
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = mask.to(dtype)
        return weights / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)

    # ------------------------------------------------------------------
    # WHERE representation
    # ------------------------------------------------------------------
    def _encode_temporal_context(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask_f = video_mask.to(video_tokens.dtype).unsqueeze(-1)

        where_input = self.where_input_proj(video_tokens) * mask_f
        normed = self.temporal_norm1(where_input)

        temporal_attn_out, temporal_attn = self.temporal_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=~video_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        temporal_delta = self.temporal_attn_delta(temporal_attn_out)
        context = where_input + temporal_delta
        context = context + self.temporal_ff(self.temporal_norm2(context))
        context = context * mask_f

        valid = video_mask.to(context.dtype)
        base_norm = (
            torch.linalg.vector_norm(where_input, dim=-1) * valid
        ).sum(dim=-1)
        delta_norm = (
            torch.linalg.vector_norm(context - where_input, dim=-1) * valid
        ).sum(dim=-1)
        delta_ratio = delta_norm / base_norm.clamp_min(1e-6)

        return {
            "where_input_states": where_input,
            "where_context_states": context,
            "temporal_self_attention": temporal_attn,
            "temporal_context_delta_ratio": delta_ratio,
        }

    def _condition_on_question(
        self,
        where_context: torch.Tensor,
        question_features: torch.Tensor,
        question_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        q_tokens = self.question_token_proj(question_features)
        q_tokens = q_tokens * question_mask.to(q_tokens.dtype).unsqueeze(-1)
        q_global = self._masked_mean(q_tokens, question_mask, dim=1)

        cross, cross_attn = self.qv_attn(
            query=self.qv_norm(where_context),
            key=q_tokens,
            value=q_tokens,
            key_padding_mask=~question_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        q_states = where_context + self.qv_attn_delta(cross)
        q_states = q_states + self.qv_ff(self.qv_ff_norm(q_states))
        q_states = q_states * video_mask.to(q_states.dtype).unsqueeze(-1)

        valid = video_mask.to(q_states.dtype)
        base_norm = (
            torch.linalg.vector_norm(where_context, dim=-1) * valid
        ).sum(dim=-1)
        delta_norm = (
            torch.linalg.vector_norm(q_states - where_context, dim=-1) * valid
        ).sum(dim=-1)
        interaction_ratio = delta_norm / base_norm.clamp_min(1e-6)

        return {
            "question_tokens_projected": q_tokens,
            "question_state": q_global,
            "question_conditioned_states": q_states,
            "visual_question_attention": cross_attn,
            "question_interaction_delta_ratio": interaction_ratio,
        }

    # ------------------------------------------------------------------
    # Evidence Core
    # ------------------------------------------------------------------
    def _score_evidence_core(
        self,
        q_states: torch.Tensor,
        question_state: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = q_states.shape
        q_frames = question_state.unsqueeze(1).expand(-1, time_steps, -1)

        features = torch.cat(
            [q_states, q_frames, q_states * q_frames],
            dim=-1,
        )
        hidden = self.core_encoder(features)
        core_raw = self.core_score(hidden).squeeze(-1)
        core_score = torch.tanh(core_raw)
        core_score = core_score * video_mask.to(core_score.dtype)

        masked_raw = core_raw.masked_fill(~video_mask, -1e4)
        core_prob = torch.softmax(masked_raw, dim=-1)
        core_prob = core_prob * video_mask.to(core_prob.dtype)
        core_prob = core_prob / core_prob.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        core_entropy = -(
            core_prob * torch.log(core_prob.clamp_min(1e-8))
        ).sum(dim=-1)
        valid_count = (
            video_mask.sum(dim=-1)
            .to(core_prob.dtype)
            .clamp_min(1.0)
        )
        core_entropy_ratio = torch.where(
            valid_count > 1.0,
            core_entropy
            / torch.log(valid_count.clamp_min(2.0)).clamp_min(1e-6),
            torch.zeros_like(core_entropy),
        )

        return {
            "evidence_core_raw": core_raw,
            "evidence_core": core_score,
            "evidence_core_prob": core_prob,
            "evidence_core_entropy": core_entropy,
            "evidence_core_entropy_ratio": core_entropy_ratio,
            "evidence_core_top1_prob": core_prob.max(dim=-1).values,
        }

    # ------------------------------------------------------------------
    # Event Continuity
    # ------------------------------------------------------------------
    def _score_event_continuity(
        self,
        where_context: torch.Tensor,
        q_states: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = where_context.shape

        if time_steps <= 1:
            empty = where_context.new_zeros(batch_size, 0)
            empty_mask = video_mask.new_zeros(batch_size, 0)
            return {
                "visual_continuity_raw": empty,
                "question_continuity_correction_raw": empty,
                "event_continuity_raw": empty,
                "event_continuity": empty,
                "continuity_mask": empty_mask,
            }

        continuity_mask = video_mask[:, :-1] & video_mask[:, 1:]

        left_v = where_context[:, :-1]
        right_v = where_context[:, 1:]
        visual_features = torch.cat(
            [
                left_v,
                right_v,
                torch.abs(right_v - left_v),
                left_v * right_v,
            ],
            dim=-1,
        )
        visual_hidden = self.visual_continuity_encoder(visual_features)
        visual_raw = self.visual_continuity_score(visual_hidden).squeeze(-1)

        left_q = q_states[:, :-1]
        right_q = q_states[:, 1:]
        question_features = torch.cat(
            [
                left_q,
                right_q,
                torch.abs(right_q - left_q),
                left_q * right_q,
            ],
            dim=-1,
        )
        question_hidden = self.question_continuity_encoder(question_features)
        question_correction_raw = self.question_continuity_score(
            question_hidden
        ).squeeze(-1)

        continuity_raw = visual_raw + question_correction_raw
        continuity = torch.tanh(continuity_raw)

        mask_f = continuity_mask.to(continuity.dtype)
        visual_raw = visual_raw * mask_f
        question_correction_raw = question_correction_raw * mask_f
        continuity_raw = continuity_raw * mask_f
        continuity = continuity * mask_f

        return {
            "visual_continuity_raw": visual_raw,
            "question_continuity_correction_raw": question_correction_raw,
            "event_continuity_raw": continuity_raw,
            "event_continuity": continuity,
            "continuity_mask": continuity_mask,
        }

    # ------------------------------------------------------------------
    # Core-to-Closure interval model
    # ------------------------------------------------------------------
    def _build_core_to_closure(
        self,
        evidence_core: torch.Tensor,
        event_continuity: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps = evidence_core.shape
        dtype = evidence_core.dtype
        device = evidence_core.device

        starts = torch.arange(time_steps, device=device).view(time_steps, 1)
        ends = torch.arange(time_steps, device=device).view(1, time_steps)
        span_len = ends - starts + 1
        upper = span_len >= 1

        valid_prefix = F.pad(
            torch.cumsum(video_mask.to(torch.long), dim=-1),
            (1, 0),
            value=0,
        )
        valid_count = (
            valid_prefix[:, 1:].unsqueeze(1)
            - valid_prefix[:, :-1].unsqueeze(2)
        )
        segment_valid = upper.unsqueeze(0) & (
            valid_count == span_len.unsqueeze(0)
        )

        # C_core: strongest contained evidence seed.
        index = torch.arange(time_steps, device=device)
        start_grid = index.view(time_steps, 1, 1)
        end_grid = index.view(1, time_steps, 1)
        frame_grid = index.view(1, 1, time_steps)
        inclusion = (start_grid <= frame_grid) & (frame_grid <= end_grid)

        included_core = evidence_core[:, None, None, :].expand(
            -1,
            time_steps,
            time_steps,
            -1,
        )
        core_mask = inclusion.unsqueeze(0) & video_mask[:, None, None, :]
        segment_core_support = included_core.masked_fill(
            ~core_mask,
            -1e4,
        ).max(dim=-1).values

        if time_steps > 1:
            edge_mask = video_mask[:, :-1] & video_mask[:, 1:]
            continuity = event_continuity * edge_mask.to(dtype)

            # C_inside = mean continuity inside [s,e].
            edge_prefix = F.pad(
                torch.cumsum(continuity, dim=-1),
                (1, 0),
                value=0.0,
            )
            inside_sum = edge_prefix.unsqueeze(1) - edge_prefix.unsqueeze(2)
            inside_count = (ends - starts).clamp_min(0).to(dtype)
            segment_inside_continuity = torch.where(
                inside_count.unsqueeze(0) > 0,
                inside_sum
                / inside_count.clamp_min(1.0).unsqueeze(0),
                torch.zeros_like(inside_sum),
            )

            # C_cut = low continuity immediately outside [s,e].
            # Video boundaries are neutral, not rewarded.
            left_cont = F.pad(continuity, (1, 0), value=0.0)
            right_cont = F.pad(continuity, (0, 1), value=0.0)
            left_valid = F.pad(edge_mask, (1, 0), value=False)
            right_valid = F.pad(edge_mask, (0, 1), value=False)

            left_2d = left_cont.unsqueeze(2)
            right_2d = right_cont.unsqueeze(1)
            left_valid_2d = left_valid.unsqueeze(2)
            right_valid_2d = right_valid.unsqueeze(1)

            boundary_count = (
                left_valid_2d.to(dtype) + right_valid_2d.to(dtype)
            )
            boundary_continuity_mean = (
                left_2d * left_valid_2d.to(dtype)
                + right_2d * right_valid_2d.to(dtype)
            ) / boundary_count.clamp_min(1.0)
            segment_boundary_closure = torch.where(
                boundary_count > 0,
                -boundary_continuity_mean,
                torch.zeros_like(boundary_continuity_mean),
            )
        else:
            edge_mask = video_mask.new_zeros(batch_size, 0)
            segment_inside_continuity = evidence_core.new_zeros(
                batch_size,
                1,
                1,
            )
            segment_boundary_closure = evidence_core.new_zeros(
                batch_size,
                1,
                1,
            )

        # No hand-tuned core/relation/boundary weights.
        segment_energy = (
            segment_core_support
            + segment_inside_continuity
            + segment_boundary_closure
        )

        segment_logits = (
            segment_energy / self.segment_temperature
        ).masked_fill(~segment_valid, -1e4)

        flat_logits = segment_logits.reshape(batch_size, -1)
        flat_valid = segment_valid.reshape(batch_size, -1)
        flat_prob = torch.softmax(
            flat_logits.masked_fill(~flat_valid, -1e4),
            dim=-1,
        )
        flat_prob = flat_prob * flat_valid.to(flat_prob.dtype)
        flat_prob = flat_prob / flat_prob.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)
        segment_prob = flat_prob.reshape(batch_size, time_steps, time_steps)

        # Hard MAP from the same interval model.
        selected_flat = flat_logits.masked_fill(
            ~flat_valid,
            -1e4,
        ).argmax(dim=-1)
        selected_start = torch.div(
            selected_flat,
            time_steps,
            rounding_mode="floor",
        )
        selected_end = selected_flat % time_steps
        selected_prob = flat_prob.gather(
            1,
            selected_flat.unsqueeze(-1),
        ).squeeze(-1)
        selected_energy = flat_logits.gather(
            1,
            selected_flat.unsqueeze(-1),
        ).squeeze(-1)

        selected_mask = self._span_mask_from_bounds(
            selected_start,
            selected_end,
            video_mask,
        )
        hard_temporal_prob = self._uniform_weights_from_mask(
            selected_mask,
            dtype=dtype,
        )

        # Soft frame membership from exactly the same P_Q(s,e).
        frame_membership_prob = torch.einsum(
            "bij,ijt->bt",
            segment_prob,
            inclusion.to(dtype),
        )
        frame_membership_prob = (
            frame_membership_prob * video_mask.to(dtype)
        )
        soft_temporal_prob = frame_membership_prob / frame_membership_prob.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

        # V2.4 uses soft temporal pooling for grounding supervision; no ST trick.
        training_temporal_prob = soft_temporal_prob

        token_count = video_mask.sum(dim=-1).to(dtype).clamp_min(1.0)
        selected_length = selected_mask.sum(dim=-1).to(dtype)
        selected_fraction = selected_length / token_count
        denominator = (token_count - 1.0).clamp_min(1.0)
        selected_start_norm = (
            selected_start.to(dtype) / denominator
        ).clamp(0.0, 1.0)
        selected_end_norm = (
            selected_end.to(dtype) / denominator
        ).clamp(0.0, 1.0)

        span_len_f = span_len.clamp_min(1).to(dtype)
        expected_length = (
            segment_prob * span_len_f.unsqueeze(0)
        ).sum(dim=(-2, -1))
        expected_fraction = expected_length / token_count

        entropy, entropy_ratio = self._entropy_from_flat_distribution(
            flat_prob,
            flat_valid,
        )

        diag_mask = torch.eye(
            time_steps,
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
        single_frame_prob_mass = (
            segment_prob * diag_mask.to(dtype)
        ).sum(dim=(-2, -1))
        short_mask = (
            upper & (span_len <= 2)
        ).unsqueeze(0) & segment_valid
        short_span_prob_mass = (
            segment_prob * short_mask.to(dtype)
        ).sum(dim=(-2, -1))

        return {
            "segment_core_support": segment_core_support,
            "segment_inside_continuity": segment_inside_continuity,
            "segment_boundary_closure": segment_boundary_closure,
            "segment_energy": segment_energy,
            "segment_logits": segment_logits,
            "segment_valid_mask": segment_valid,
            "segment_prob": segment_prob,
            "segment_entropy": entropy,
            "segment_entropy_ratio": entropy_ratio,
            "segment_top1_prob": selected_prob,
            "frame_membership_prob": frame_membership_prob,
            "soft_temporal_prob": soft_temporal_prob,
            "training_temporal_prob": training_temporal_prob,
            "main_temporal_prob": hard_temporal_prob,
            "selected_span_frame_mask": selected_mask.detach(),
            "selected_span_frame_weights": hard_temporal_prob,
            "selected_span_start_frame": selected_start.detach(),
            "selected_span_end_frame": selected_end.detach(),
            "selected_span_start": selected_start_norm,
            "selected_span_end": selected_end_norm,
            "selected_span_length": selected_length,
            "selected_span_fraction": selected_fraction,
            "selected_segment_energy": selected_energy,
            "soft_expected_span_length": expected_length,
            "soft_expected_span_fraction": expected_fraction,
            "single_frame_prob_mass": single_frame_prob_mass,
            "short_span_prob_mass": short_span_prob_mass,
            "selected_single_frame_indicator": (
                selected_length <= 1.0 + 1e-6
            ).to(dtype),
            "selected_full_video_indicator": (
                selected_length >= token_count - 1e-6
            ).to(dtype),
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        video_tokens: torch.Tensor,
        question_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,T,D].")

        batch_size, time_steps, feature_dim = video_tokens.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {feature_dim}."
            )
        if time_steps < 1:
            raise ValueError(
                "Q-Bound V2.4 requires at least one timestep."
            )

        video_mask = self._normalize_video_mask(
            video_mask,
            batch_size,
            time_steps,
            video_tokens.device,
        )
        question_features, question_mask = self._normalize_question_features(
            question_features,
            question_mask,
            batch_size,
            video_tokens.device,
        )

        # Preserve original CLIP semantic content for downstream VALUE pooling.
        value_tokens = video_tokens * video_mask.to(
            video_tokens.dtype
        ).unsqueeze(-1)

        temporal = self._encode_temporal_context(
            video_tokens,
            video_mask,
        )
        conditioned = self._condition_on_question(
            where_context=temporal["where_context_states"],
            question_features=question_features,
            question_mask=question_mask,
            video_mask=video_mask,
        )
        core = self._score_evidence_core(
            q_states=conditioned["question_conditioned_states"],
            question_state=conditioned["question_state"],
            video_mask=video_mask,
        )
        continuity = self._score_event_continuity(
            where_context=temporal["where_context_states"],
            q_states=conditioned["question_conditioned_states"],
            video_mask=video_mask,
        )
        segments = self._build_core_to_closure(
            evidence_core=core["evidence_core"],
            event_continuity=continuity["event_continuity"],
            video_mask=video_mask,
        )

        dtype = video_tokens.dtype
        device = video_tokens.device

        return {
            # Architecture contract.
            "qbound_version": self.qbound_version,
            "core_to_closure_enabled": torch.ones((), dtype=dtype, device=device),
            "temporal_context_enabled": torch.ones((), dtype=dtype, device=device),
            "early_question_conditioning_enabled": torch.ones((), dtype=dtype, device=device),
            "evidence_core_enabled": torch.ones((), dtype=dtype, device=device),
            "event_continuity_enabled": torch.ones((), dtype=dtype, device=device),
            "same_interval_model_for_hard_and_soft": torch.ones((), dtype=dtype, device=device),
            "value_where_separation_enabled": torch.ones((), dtype=dtype, device=device),

            # Explicitly absent V2.3 / historical shortcut machinery.
            "boundary_refiner_enabled": torch.zeros((), dtype=dtype, device=device),
            "local_candidate_refinement_enabled": torch.zeros((), dtype=dtype, device=device),
            "adaptive_radius_enabled": torch.zeros((), dtype=dtype, device=device),
            "learned_routing_gate_enabled": torch.zeros((), dtype=dtype, device=device),
            "independent_endpoint_heads_enabled": torch.zeros((), dtype=dtype, device=device),
            "answer_features_used_in_core": torch.zeros((), dtype=dtype, device=device),
            "temporal_gt_used_in_core": torch.zeros((), dtype=dtype, device=device),
            "answer_conditioned_temporal_critic_enabled": torch.zeros((), dtype=dtype, device=device),
            "length_prior_enabled": torch.zeros((), dtype=dtype, device=device),
            "gaussian_smoothing_enabled": torch.zeros((), dtype=dtype, device=device),
            "fixed_event_count_enabled": torch.zeros((), dtype=dtype, device=device),

            # Clean V2.4 interface.
            "video_mask": video_mask,
            "question_mask": question_mask,
            "value_tokens": value_tokens,
            **temporal,
            **conditioned,
            **core,
            **continuity,
            **segments,

            # Canonical downstream aliases.
            "qbound_value_tokens": value_tokens,
            "qbound_where_states": conditioned["question_conditioned_states"],
            "qbound_core_score": core["evidence_core"],
            "qbound_continuity_score": continuity["event_continuity"],
            "qbound_segment_energy": segments["segment_energy"],
            "qbound_segment_prob": segments["segment_prob"],
            "qbound_hard_temporal_prob": segments["main_temporal_prob"],
            "qbound_soft_temporal_prob": segments["soft_temporal_prob"],
            "qbound_training_temporal_prob": segments["training_temporal_prob"],
            "qbound_span_start_frame": segments["selected_span_start_frame"],
            "qbound_span_end_frame": segments["selected_span_end_frame"],
            "qbound_span_frame_mask": segments["selected_span_frame_mask"],
            "qbound_span_length": segments["selected_span_length"],
            "qbound_span_fraction": segments["selected_span_fraction"],
            "qbound_soft_expected_span_length": segments["soft_expected_span_length"],
            "qbound_soft_expected_span_fraction": segments["soft_expected_span_fraction"],
            "qbound_segment_entropy_ratio": segments["segment_entropy_ratio"],
            "qbound_segment_top1_prob": segments["segment_top1_prob"],
        }
