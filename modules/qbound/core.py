from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound V2.8.0 -- Decoupled Relevance-to-Extent Grounding.


    V3.8 compatibility contract (additive only)
    ------------------------------------------
    The V3.5/R2-C encoder, semantic seed, old support graph, and analytic
    interval decoder are intentionally NOT redesigned here.  New M0 event
    bank, limited context reader, and joint answer-event scorer are separate
    modules connected by exported feature tensors. M1 reads a separate,
    detached view of the original video features via export_v38_m1_inputs.
    The M1 text/visual projections, bank scoring and losses stay outside
    this core. This core remains answer-free and temporal-GT-free; no new
    trainable parameters are added.

    Research diagnosis
    ------------------
    Earlier Q-Bound versions repeatedly exposed the same failure mode under
    weak Video-Question supervision: the cross-modal signal can learn WHERE a
    discriminative clue is, but the same signal does not identify HOW FAR the
    temporal support should extend.  When Question semantics directly score
    temporal spans, optimization can prefer a minimum-sufficient snippet.

    V2.8 enforces a no-bypass factorization. R2-C specifically disables the
    predictive VIDEO-only branch and retains the same normalized fixed chain
    Laplacian as R2-B, with no predictive objective:

        Video + Question  -> Semantic Evidence Seed r_Q(t)        [RELEVANCE]
        Video only        -> Adjacent Support Graph w_t           [EXTENT STRUCTURE]
        r_Q + stopgrad(w) -> Structural Support p_Q(t)            [GROUNDING]

    The Question NEVER directly scores [s,e].  The support graph NEVER consumes
    Question/answer features and is detached from L_ground.  It is learned by a
    separate, video-only, leakage-free predictive objective in A/B; in R2-C
    the predictor is absent and adjacent graph weights are fixed.

    WHERE path
    ----------
      frozen/local CLIP
        -> Structured Visual Evidence (kept from V2.6)
        -> shallow Temporal Context (kept from V2.6)
        -> token-resolved visual-to-question interaction
        -> Semantic Evidence Seed r_Q(t)
        -> structural propagation through a detached video-only support graph
        -> Support Field p_Q(t)
        -> analytic start/end evidence and P_Q(s,e)
        -> posterior-barycentric endpoints (NO MAP temporal decision)

    Support path
    ------------
      detached raw CLIP
        -> independent support projection
        -> causal forward / reverse-causal recurrent contexts
        -> predict unseen adjacent raw CLIP targets
        -> symmetric predictive surprise per adjacent edge
        -> within-video standardized support affinity
        -> normalized 1-D graph Laplacian

    VALUE path
    ----------
      original masked CLIP is preserved as downstream VALUE.  Event/support
      representations are WHERE-only.

    Important contracts
    -------------------
      * Video + Question only at temporal inference.
      * Answers and temporal GT never enter QBoundCore.
      * Question semantics may discover seeds, but cannot directly score spans.
      * L_ground cannot update the video-only support predictor/graph.
      * L_pred cannot update the Question relevance branch.
      * Support prediction has no target leakage: x[t+1] is predicted only from
        left context <=t; x[t] is predicted only from right context >=t+1.
      * No neural span scorer, Event candidate selector, Core-to-Closure,
        Joint Continuity, BoundaryRefiner, learned start/end heads, length prior,
        Gaussian smoothing, event-count target, or minimum-duration constraint.
      * Training and inference use the SAME Support Field p_Q(t).
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        num_question_heads: int = 4,
        question_ff_multiplier: int = 2,
        support_graph_mode: str = "learned",
        support_predictor_enabled: bool = True,
    ) -> None:
        super().__init__()

        if input_dim < 1 or hidden_dim < 1:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1).")
        if num_question_heads < 1 or hidden_dim % num_question_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_question_heads.")
        if question_ff_multiplier < 1:
            raise ValueError("question_ff_multiplier must be >= 1.")
        if support_graph_mode not in ("learned", "fixed"):
            raise ValueError("support_graph_mode must be 'learned' or 'fixed'.")

        if not isinstance(support_predictor_enabled, bool):
            raise TypeError("support_predictor_enabled must be a boolean.")
        if not support_predictor_enabled and support_graph_mode != "fixed":
            raise ValueError("R2-C requires fixed graph when predictor is disabled.")
        self.support_graph_mode = support_graph_mode
        self.support_predictor_enabled = support_predictor_enabled
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout_p = float(dropout)
        self.num_question_heads = int(num_question_heads)
        self.question_ff_multiplier = int(question_ff_multiplier)
        self.qbound_version = "2.8.0"

        # ------------------------------------------------------------------
        # A. Structured Visual Evidence (kept from V2.6)
        # ------------------------------------------------------------------
        self.where_input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        self.visual_distinctiveness_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.visual_transition_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )
        self.visual_transition_frame_delta = nn.Linear(
            self.hidden_dim, self.hidden_dim
        )

        # Shallow temporal visual context, also kept from V2.6.
        self.temporal_norm1 = nn.LayerNorm(self.hidden_dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_question_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.temporal_attn_delta = nn.Linear(self.hidden_dim, self.hidden_dim)

        ff_dim = self.hidden_dim * self.question_ff_multiplier
        self.temporal_norm2 = nn.LayerNorm(self.hidden_dim)
        self.temporal_ff = nn.Sequential(
            nn.Linear(self.hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(ff_dim, self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )

        # ------------------------------------------------------------------
        # B. Relevance branch: rich token-level Q-V interaction.
        #
        # Question VALUE is used only to form a separate semantic context for
        # the relevance head.  It never overwrites the preserved visual stream.
        # ------------------------------------------------------------------
        self.question_token_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.relevance_visual_norm = nn.LayerNorm(self.hidden_dim)
        self.relevance_question_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_question_heads,
            dropout=self.dropout_p,
            batch_first=True,
        )
        self.relevance_question_context_norm = nn.LayerNorm(self.hidden_dim)

        self.semantic_seed_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Dropout(self.dropout_p),
        )
        self.semantic_seed_score = nn.Linear(self.hidden_dim, 1)

        # ------------------------------------------------------------------
        # C. Independent VIDEO-ONLY support branch.
        #
        # The target is the detached raw CLIP feature itself.  Therefore the
        # support representation cannot collapse together with a learned target
        # encoder, and the predictor never observes the frame being predicted.
        # ------------------------------------------------------------------
        self.support_input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.support_forward_cell = nn.GRUCell(
            self.hidden_dim, self.hidden_dim
        )
        self.support_backward_cell = nn.GRUCell(
            self.hidden_dim, self.hidden_dim
        )

        self.support_forward_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.input_dim),
        )
        self.support_backward_predictor = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.hidden_dim, self.input_dim),
        )

        self._initialize_v280()
        if not self.support_predictor_enabled:
            # Preserve EXACT initialization/RNG consumption of R2-B, then
            # remove the predictor completely before optimizer/model.to().
            # The live R2-C module has no predictor parameters or forward ops.
            for name in (
                "support_input_proj", "support_forward_cell",
                "support_backward_cell", "support_forward_predictor",
                "support_backward_predictor",
            ):
                delattr(self, name)

    # ==================================================================
    # Initialization
    # ==================================================================
    def _initialize_v280(self) -> None:
        """Identity-preserving initialization for the retained WHERE encoder."""
        small_residual_layers = [
            self.visual_transition_frame_delta,
            self.temporal_attn_delta,
        ]

        distinctive_last = None
        for module in self.visual_distinctiveness_encoder.modules():
            if isinstance(module, nn.Linear):
                distinctive_last = module
        if distinctive_last is not None:
            small_residual_layers.append(distinctive_last)

        temporal_ff_last = None
        for module in self.temporal_ff.modules():
            if isinstance(module, nn.Linear):
                temporal_ff_last = module
        if temporal_ff_last is not None:
            small_residual_layers.append(temporal_ff_last)

        for layer in small_residual_layers:
            nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # Start the semantic seed head close to non-committal, not strongly
        # concentrated on a random frame at initialization.
        nn.init.normal_(self.semantic_seed_score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.semantic_seed_score.bias)

    # ==================================================================
    # Generic helpers
    # ==================================================================
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
    def _masked_stats_1d(
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample masked mean/std for tensors shaped [B,N]."""
        mask_f = mask.to(values.dtype)
        count = mask_f.sum(dim=-1).clamp_min(1.0)
        mean = (values * mask_f).sum(dim=-1) / count
        centered = values - mean.unsqueeze(-1)
        var = (centered.pow(2) * mask_f).sum(dim=-1) / count
        std = torch.sqrt(var.clamp_min(0.0))
        return mean, std

    @staticmethod
    def _distribution_diagnostics(
        prob: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Diagnostics for a masked categorical distribution [B,T]."""
        p = prob * valid_mask.to(prob.dtype)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
        valid_count = valid_mask.sum(dim=-1).to(prob.dtype).clamp_min(1.0)
        max_entropy = torch.log(valid_count.clamp_min(2.0))
        entropy_ratio = torch.where(
            valid_count > 1.0,
            entropy / max_entropy.clamp_min(1e-6),
            torch.zeros_like(entropy),
        )
        top1 = p.max(dim=-1).values
        effective = torch.exp(entropy)

        return {
            "entropy": entropy,
            "entropy_ratio": entropy_ratio,
            "top1_prob": top1,
            "effective_count": effective,
        }

    @staticmethod
    def _entropy_from_flat_distribution(
        flat_prob: torch.Tensor,
        flat_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = flat_prob * flat_valid.to(flat_prob.dtype)
        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=-1)
        valid_count = (
            flat_valid.sum(dim=-1).to(flat_prob.dtype).clamp_min(1.0)
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

    # ==================================================================
    # A. Structured Visual Evidence (retained V2.6 representation)
    # ==================================================================
    def _encode_visual_evidence(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = video_tokens.shape
        mask_f = video_mask.to(video_tokens.dtype).unsqueeze(-1)

        # 1) State.
        state = self.where_input_proj(video_tokens) * mask_f

        # 2) Within-video distinctiveness.
        state_mean = self._masked_mean(state, video_mask, dim=1)
        distinctive_input = (state - state_mean.unsqueeze(1)) * mask_f
        distinctive_delta = (
            self.visual_distinctiveness_encoder(distinctive_input) * mask_f
        )

        # 3) Explicit adjacent visual transition.
        if time_steps > 1:
            transition_mask = video_mask[:, :-1] & video_mask[:, 1:]
            left = state[:, :-1]
            right = state[:, 1:]
            transition_features = torch.cat(
                [left, right, torch.abs(right - left), left * right],
                dim=-1,
            )
            transition_states = self.visual_transition_encoder(
                transition_features
            )
            transition_states = (
                transition_states
                * transition_mask.to(transition_states.dtype).unsqueeze(-1)
            )

            left_incident = F.pad(
                transition_states, (0, 0, 0, 1), value=0.0
            )
            right_incident = F.pad(
                transition_states, (0, 0, 1, 0), value=0.0
            )
            left_count = F.pad(
                transition_mask.to(state.dtype), (0, 1), value=0.0
            ).unsqueeze(-1)
            right_count = F.pad(
                transition_mask.to(state.dtype), (1, 0), value=0.0
            ).unsqueeze(-1)
            incident_count = left_count + right_count
            frame_transition = (
                (left_incident + right_incident)
                / incident_count.clamp_min(1.0)
            ) * mask_f
        else:
            transition_mask = video_mask.new_zeros(batch_size, 0)
            transition_states = state.new_zeros(
                batch_size, 0, self.hidden_dim
            )
            frame_transition = state.new_zeros(
                batch_size, 1, self.hidden_dim
            )

        transition_delta = (
            self.visual_transition_frame_delta(frame_transition) * mask_f
        )

        visual_evidence = (
            state + distinctive_delta + transition_delta
        ) * mask_f

        # Diagnostics.
        valid = video_mask.to(state.dtype)
        base_norm = (
            torch.linalg.vector_norm(state, dim=-1) * valid
        ).sum(dim=-1).clamp_min(1e-6)
        distinctive_input_norm = (
            torch.linalg.vector_norm(distinctive_input, dim=-1) * valid
        ).sum(dim=-1)
        distinctive_delta_norm = (
            torch.linalg.vector_norm(distinctive_delta, dim=-1) * valid
        ).sum(dim=-1)
        transition_delta_norm = (
            torch.linalg.vector_norm(transition_delta, dim=-1) * valid
        ).sum(dim=-1)
        enrichment_delta_norm = (
            torch.linalg.vector_norm(visual_evidence - state, dim=-1) * valid
        ).sum(dim=-1)

        return {
            "where_input_states": state,
            "visual_state_states": state,
            "visual_video_mean_state": state_mean,
            "visual_distinctiveness_states": distinctive_input,
            "visual_distinctiveness_delta": distinctive_delta,
            "visual_transition_states": transition_states,
            "visual_transition_frame_states": frame_transition,
            "visual_transition_delta": transition_delta,
            "visual_transition_mask": transition_mask,
            "visual_evidence_states": visual_evidence,
            "visual_distinctiveness_input_ratio": (
                distinctive_input_norm / base_norm
            ),
            "visual_distinctiveness_delta_ratio": (
                distinctive_delta_norm / base_norm
            ),
            "visual_transition_delta_ratio": (
                transition_delta_norm / base_norm
            ),
            "visual_enrichment_delta_ratio": (
                enrichment_delta_norm / base_norm
            ),
        }

    def _encode_temporal_context(
        self,
        visual_evidence: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        mask_f = video_mask.to(visual_evidence.dtype).unsqueeze(-1)

        normed = self.temporal_norm1(visual_evidence)
        temporal_attn_out, temporal_attn = self.temporal_attn(
            query=normed,
            key=normed,
            value=normed,
            key_padding_mask=~video_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        temporal_delta = self.temporal_attn_delta(temporal_attn_out)
        context = visual_evidence + temporal_delta
        context = context + self.temporal_ff(self.temporal_norm2(context))
        context = context * mask_f

        valid = video_mask.to(context.dtype)
        base_norm = (
            torch.linalg.vector_norm(visual_evidence, dim=-1) * valid
        ).sum(dim=-1)
        delta_norm = (
            torch.linalg.vector_norm(context - visual_evidence, dim=-1) * valid
        ).sum(dim=-1)
        delta_ratio = delta_norm / base_norm.clamp_min(1e-6)

        return {
            "where_context_states": context,
            "temporal_self_attention": temporal_attn,
            "temporal_context_delta_ratio": delta_ratio,
        }

    # ==================================================================
    # B. Question-conditioned RELEVANCE only
    # ==================================================================
    def _discover_semantic_seed(
        self,
        where_context: torch.Tensor,
        question_features: torch.Tensor,
        question_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        q_tokens = self.question_token_proj(question_features)
        q_tokens = (
            q_tokens
            * question_mask.to(q_tokens.dtype).unsqueeze(-1)
        )

        visual_query = self.relevance_visual_norm(where_context)
        q_context, q_attn = self.relevance_question_attn(
            query=visual_query,
            key=q_tokens,
            value=q_tokens,
            key_padding_mask=~question_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        q_context = self.relevance_question_context_norm(q_context)
        q_context = (
            q_context
            * video_mask.to(q_context.dtype).unsqueeze(-1)
        )

        features = torch.cat(
            [
                visual_query,
                q_context,
                visual_query * q_context,
                torch.abs(visual_query - q_context),
            ],
            dim=-1,
        )
        hidden = self.semantic_seed_encoder(features)
        seed_logits = self.semantic_seed_score(hidden).squeeze(-1)
        seed_logits = seed_logits.masked_fill(~video_mask, -1e4)

        seed_prob = torch.softmax(seed_logits, dim=-1)
        seed_prob = seed_prob * video_mask.to(seed_prob.dtype)
        seed_prob = seed_prob / seed_prob.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        seed_diag = self._distribution_diagnostics(
            seed_prob,
            video_mask,
        )

        # Per-frame entropy over Question tokens: a diagnostic for whether the
        # interaction is token-resolved or effectively global.
        q_attn_mean = q_attn.mean(dim=1)
        q_attn_mean = q_attn_mean * question_mask.to(q_attn_mean.dtype).unsqueeze(1)
        q_attn_mean = q_attn_mean / q_attn_mean.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        token_entropy = -(
            q_attn_mean * torch.log(q_attn_mean.clamp_min(1e-8))
        ).sum(dim=-1)
        q_count = question_mask.sum(dim=-1).to(token_entropy.dtype).clamp_min(1.0)
        token_entropy_ratio = torch.where(
            q_count.unsqueeze(-1) > 1.0,
            token_entropy
            / torch.log(q_count.clamp_min(2.0)).unsqueeze(-1).clamp_min(1e-6),
            torch.zeros_like(token_entropy),
        )
        token_entropy_ratio = (
            token_entropy_ratio * video_mask.to(token_entropy_ratio.dtype)
        )
        token_entropy_ratio_mean = token_entropy_ratio.sum(dim=-1) / (
            video_mask.sum(dim=-1).to(token_entropy_ratio.dtype).clamp_min(1.0)
        )

        return {
            "question_tokens_projected": q_tokens,
            "relevance_question_context": q_context,
            "relevance_question_attention": q_attn,
            "relevance_question_attention_entropy_ratio": token_entropy_ratio_mean,
            "semantic_seed_hidden": hidden,
            "semantic_seed_logits": seed_logits,
            "semantic_seed_prob": seed_prob,
            "semantic_seed_entropy": seed_diag["entropy"],
            "semantic_seed_entropy_ratio": seed_diag["entropy_ratio"],
            "semantic_seed_top1_prob": seed_diag["top1_prob"],
            "semantic_seed_effective_frames": seed_diag["effective_count"],
        }

    # ==================================================================
    # C. VIDEO-ONLY predictive support branch
    # ==================================================================
    def _masked_directional_gru(
        self,
        states: torch.Tensor,
        video_mask: torch.Tensor,
        cell: nn.GRUCell,
        reverse: bool,
    ) -> torch.Tensor:
        """
        Mask-aware causal/reverse-causal GRU contexts.

        Forward output at t only uses states <= t.
        Backward output at t only uses states >= t.
        """
        batch_size, time_steps, _ = states.shape
        hidden = states.new_zeros(batch_size, self.hidden_dim)
        outputs = [None] * time_steps
        time_range = range(time_steps - 1, -1, -1) if reverse else range(time_steps)

        for t in time_range:
            valid_t = video_mask[:, t].unsqueeze(-1)
            candidate = cell(states[:, t], hidden)
            hidden = torch.where(valid_t, candidate, hidden)
            outputs[t] = torch.where(
                valid_t,
                hidden,
                torch.zeros_like(hidden),
            )

        return torch.stack(outputs, dim=1)

    def _learn_video_support_structure(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = video_tokens.shape
        dtype = video_tokens.dtype

        # Explicitly detach the visual source so L_pred cannot tune the upstream
        # CLIP/CRA feature producer even if an outer adapter forgets to detach it.
        source = video_tokens.detach()
        mask_f = video_mask.to(dtype).unsqueeze(-1)
        support_states = self.support_input_proj(source) * mask_f

        forward_context = self._masked_directional_gru(
            support_states,
            video_mask,
            self.support_forward_cell,
            reverse=False,
        )
        backward_context = self._masked_directional_gru(
            support_states,
            video_mask,
            self.support_backward_cell,
            reverse=True,
        )

        forward_prediction = self.support_forward_predictor(forward_context)
        backward_prediction = self.support_backward_predictor(backward_context)

        if time_steps <= 1:
            empty = source.new_zeros(batch_size, 0)
            zero = source.new_zeros(batch_size)
            predictive_loss = source.new_zeros(())
            adjacency = source.new_zeros(batch_size, time_steps, time_steps)
            laplacian = source.new_zeros(batch_size, time_steps, time_steps)
            return {
                "support_states": support_states,
                "support_forward_context": forward_context,
                "support_backward_context": backward_context,
                "support_forward_prediction": forward_prediction,
                "support_backward_prediction": backward_prediction,
                "support_edge_mask": video_mask.new_zeros(batch_size, 0),
                "support_forward_prediction_cosine": empty,
                "support_backward_prediction_cosine": empty,
                "support_edge_forward_error": empty,
                "support_edge_backward_error": empty,
                "support_edge_surprise": empty,
                "support_edge_surprise_mean": zero,
                "support_edge_surprise_std": zero,
                "support_edge_affinity_raw": empty,
                "support_edge_affinity_learned": empty,
                "support_edge_affinity": empty,
                "support_edge_affinity_mean": zero,
                "support_edge_affinity_std": zero,
                "support_predictive_loss_per_sample": zero,
                "support_predictive_loss": predictive_loss,
                "support_graph_adjacency": adjacency,
                "support_graph_laplacian": laplacian,
            }

        edge_mask = video_mask[:, :-1] & video_mask[:, 1:]
        edge_mask_f = edge_mask.to(dtype)

        # NO target leakage:
        #   forward_context[t] uses <=t and predicts raw x[t+1]
        #   backward_context[t+1] uses >=t+1 and predicts raw x[t]
        pred_next = forward_prediction[:, :-1]
        target_next = source[:, 1:]
        pred_prev = backward_prediction[:, 1:]
        target_prev = source[:, :-1]

        forward_cos = F.cosine_similarity(
            pred_next,
            target_next,
            dim=-1,
            eps=1e-6,
        )
        backward_cos = F.cosine_similarity(
            pred_prev,
            target_prev,
            dim=-1,
            eps=1e-6,
        )

        forward_error = (1.0 - forward_cos) * edge_mask_f
        backward_error = (1.0 - backward_cos) * edge_mask_f
        edge_surprise = 0.5 * (forward_error + backward_error)

        edge_count = edge_mask_f.sum(dim=-1).clamp_min(1.0)
        predictive_loss_per_sample = edge_surprise.sum(dim=-1) / edge_count
        predictive_loss = predictive_loss_per_sample.mean()

        surprise_mean, surprise_std = self._masked_stats_1d(
            edge_surprise,
            edge_mask,
        )
        standardized_surprise = (
            edge_surprise - surprise_mean.unsqueeze(-1)
        ) / surprise_std.unsqueeze(-1).clamp_min(1e-6)
        standardized_surprise = standardized_surprise * edge_mask_f

        # Low predictive surprise -> strong same-support affinity.
        # This is a deterministic structural mapping, not a learned gate.
        edge_affinity_raw = torch.sigmoid(-standardized_surprise) * edge_mask_f

        # R2 controlled experiment: predictor and L_pred are computed for BOTH
        # modes; only the edge weights consumed by propagation are switched.
        # Detaching either edge source protects the same grounding firewall.
        learned_affinity = edge_affinity_raw.detach()
        edge_affinity = (
            learned_affinity
            if self.support_graph_mode == "learned"
            else edge_mask_f.detach()
        )

        # Used-edge statistics must reflect the actual graph in R2-B.
        # Retain separately the learned-affinity diagnostics for auditing.
        learned_affinity_mean, learned_affinity_std = self._masked_stats_1d(
            edge_affinity_raw,
            edge_mask,
        )
        affinity_mean, affinity_std = self._masked_stats_1d(
            edge_affinity,
            edge_mask,
        )

        adjacency, laplacian = self._build_chain_laplacian(
            edge_affinity=edge_affinity,
            video_mask=video_mask,
            dtype=dtype,
        )

        return {
            "support_states": support_states,
            "support_forward_context": forward_context,
            "support_backward_context": backward_context,
            "support_forward_prediction": forward_prediction,
            "support_backward_prediction": backward_prediction,
            "support_edge_mask": edge_mask,
            "support_forward_prediction_cosine": forward_cos * edge_mask_f,
            "support_backward_prediction_cosine": backward_cos * edge_mask_f,
            "support_edge_forward_error": forward_error,
            "support_edge_backward_error": backward_error,
            "support_edge_surprise": edge_surprise,
            "support_edge_surprise_mean": surprise_mean,
            "support_edge_surprise_std": surprise_std,
            "support_edge_affinity_raw": edge_affinity_raw,
            "support_edge_affinity_learned": learned_affinity,
            "support_edge_affinity": edge_affinity,
            "support_edge_affinity_mean": affinity_mean,
            "support_edge_affinity_std": affinity_std,
            "support_edge_affinity_learned_mean": learned_affinity_mean,
            "support_edge_affinity_learned_std": learned_affinity_std,
            "support_predictive_loss_per_sample": predictive_loss_per_sample,
            "support_predictive_loss": predictive_loss,
            "support_graph_adjacency": adjacency,
            "support_graph_laplacian": laplacian,
        }

    def _fixed_video_support_structure(self, video_tokens, video_mask):
        """R2-C: fixed chain graph, NO predictor computation or L_pred.

        Zero-valued predictive fields are compatibility-only placeholders for
        the original CRA/Refine interface. They are NOT trained, logged as
        learned edges or added to the objective. Their presence does not mean
        that a predictor was instantiated or executed.
        """
        batch_size, time_steps, input_dim = video_tokens.shape
        dtype, device = video_tokens.dtype, video_tokens.device
        edge_mask = video_mask[:, :-1] & video_mask[:, 1:]
        edge_affinity = edge_mask.to(dtype)
        adjacency, laplacian = self._build_chain_laplacian(
            edge_affinity=edge_affinity, video_mask=video_mask, dtype=dtype
        )
        empty_hidden = video_tokens.new_zeros(batch_size, time_steps, self.hidden_dim)
        empty_pred = video_tokens.new_zeros(batch_size, time_steps, input_dim)
        empty_edge = video_tokens.new_zeros(batch_size, max(time_steps-1, 0))
        zero_batch = video_tokens.new_zeros(batch_size)
        zero_scalar = video_tokens.new_zeros(())
        affinity_mean, affinity_std = self._masked_stats_1d(edge_affinity, edge_mask)
        return {
            "support_states": empty_hidden,
            "support_forward_context": empty_hidden,
            "support_backward_context": empty_hidden,
            "support_forward_prediction": empty_pred,
            "support_backward_prediction": empty_pred,
            "support_edge_mask": edge_mask,
            "support_forward_prediction_cosine": empty_edge,
            "support_backward_prediction_cosine": empty_edge,
            "support_edge_forward_error": empty_edge,
            "support_edge_backward_error": empty_edge,
            "support_edge_surprise": empty_edge,
            "support_edge_surprise_mean": zero_batch,
            "support_edge_surprise_std": zero_batch,
            "support_edge_affinity_raw": empty_edge,
            "support_edge_affinity_learned": empty_edge,
            "support_edge_affinity": edge_affinity,
            "support_edge_affinity_mean": affinity_mean,
            "support_edge_affinity_std": affinity_std,
            "support_edge_affinity_learned_mean": zero_batch,
            "support_edge_affinity_learned_std": zero_batch,
            "support_predictive_loss_per_sample": zero_batch,
            "support_predictive_loss": zero_scalar,
            "support_graph_adjacency": adjacency,
            "support_graph_laplacian": laplacian,
        }

    @staticmethod
    def _build_chain_laplacian(
        edge_affinity: torch.Tensor,
        video_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Symmetric normalized Laplacian for the adjacent 1-D support graph."""
        batch_size, time_steps = video_mask.shape
        device = video_mask.device

        adjacency = torch.zeros(
            batch_size,
            time_steps,
            time_steps,
            dtype=dtype,
            device=device,
        )

        if time_steps > 1:
            idx = torch.arange(time_steps - 1, device=device)
            adjacency[:, idx, idx + 1] = edge_affinity.to(dtype)
            adjacency[:, idx + 1, idx] = edge_affinity.to(dtype)

        degree = adjacency.sum(dim=-1)
        degree_positive = (degree > 0) & video_mask
        inv_sqrt_degree = torch.where(
            degree_positive,
            torch.rsqrt(degree.clamp_min(1e-8)),
            torch.zeros_like(degree),
        )
        normalized_adjacency = (
            adjacency
            * inv_sqrt_degree.unsqueeze(-1)
            * inv_sqrt_degree.unsqueeze(-2)
        )

        diag = torch.diag_embed(degree_positive.to(dtype))
        laplacian = diag - normalized_adjacency

        # Padded nodes are structurally inert.
        valid_pair = video_mask.unsqueeze(-1) & video_mask.unsqueeze(-2)
        adjacency = adjacency * valid_pair.to(dtype)
        laplacian = laplacian * valid_pair.to(dtype)
        return adjacency, laplacian

    # ==================================================================
    # D. No-bypass seed-to-support propagation
    # ==================================================================
    def _propagate_seed_to_support(
        self,
        semantic_seed_prob: torch.Tensor,
        support_laplacian: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps = semantic_seed_prob.shape
        dtype = semantic_seed_prob.dtype
        device = semantic_seed_prob.device

        if support_laplacian.shape != (batch_size, time_steps, time_steps):
            raise ValueError(
                "support_laplacian must be [B,T,T]; expected "
                f"{(batch_size, time_steps, time_steps)}, got "
                f"{tuple(support_laplacian.shape)}."
            )
        if support_laplacian.requires_grad:
            raise RuntimeError(
                "Grounding support_laplacian must be detached from the support "
                "predictor so L_ground cannot train temporal extent structure."
            )

        identity = torch.eye(
            time_steps,
            dtype=dtype,
            device=device,
        ).unsqueeze(0).expand(batch_size, -1, -1)
        system = identity + support_laplacian

        # Solve in fp32 for AMP/half stability, then cast back.  The system
        # matrix is detached, while gradients flow cleanly to semantic_seed_prob.
        solve_dtype = (
            torch.float32
            if dtype in (torch.float16, torch.bfloat16)
            else dtype
        )
        # Diagnostic CUDA/cuBLAS workaround for small T x T systems.
        # Intentionally DO NOT detach semantic_seed_prob: autograd must still
        # backpropagate through CPU solve and device transfers to the seed.
        # The graph Laplacian was already detached in the support branch.
        solved = torch.linalg.solve(
            system.to(device="cpu", dtype=solve_dtype),
            semantic_seed_prob.to(device="cpu", dtype=solve_dtype).unsqueeze(-1),
        ).squeeze(-1).to(device=device, dtype=dtype)

        support_prob = solved.clamp_min(0.0)
        support_prob = support_prob * video_mask.to(dtype)
        support_prob = support_prob / support_prob.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        seed_diag = self._distribution_diagnostics(
            semantic_seed_prob,
            video_mask,
        )
        support_diag = self._distribution_diagnostics(
            support_prob,
            video_mask,
        )

        l1_delta = (
            (support_prob - semantic_seed_prob).abs()
            * video_mask.to(dtype)
        ).sum(dim=-1)
        seed_support_cosine = F.cosine_similarity(
            semantic_seed_prob,
            support_prob,
            dim=-1,
            eps=1e-8,
        )

        return {
            "semantic_seed_prob_for_support": semantic_seed_prob,
            "support_temporal_prob": support_prob,
            "support_entropy": support_diag["entropy"],
            "support_entropy_ratio": support_diag["entropy_ratio"],
            "support_top1_prob": support_diag["top1_prob"],
            "support_effective_frames": support_diag["effective_count"],
            "support_propagation_l1_delta": l1_delta,
            "semantic_seed_support_cosine": seed_support_cosine,
            # Re-expose seed diagnostics next to final support for easy logging.
            "seed_entropy_ratio_for_support": seed_diag["entropy_ratio"],
            "seed_top1_prob_for_support": seed_diag["top1_prob"],
        }

    # ==================================================================
    # E. Analytic boundary / span posterior from the support field
    # ==================================================================
    def _support_to_interval_posterior(
        self,
        support_prob: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps = support_prob.shape
        dtype = support_prob.dtype
        device = support_prob.device

        # Rising/falling support mass.  The outside of the valid sequence is 0.
        prev_prob = F.pad(support_prob[:, :-1], (1, 0), value=0.0)
        next_prob = F.pad(support_prob[:, 1:], (0, 1), value=0.0)

        start_evidence = F.relu(support_prob - prev_prob)
        end_evidence = F.relu(support_prob - next_prob)
        mask_f = video_mask.to(dtype)
        start_evidence = start_evidence * mask_f
        end_evidence = end_evidence * mask_f

        start_prob = start_evidence / start_evidence.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        end_prob = end_evidence / end_evidence.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        starts = torch.arange(time_steps, device=device).view(time_steps, 1)
        ends = torch.arange(time_steps, device=device).view(1, time_steps)
        span_len = ends - starts + 1
        upper = span_len >= 1

        # Require every frame in [s,e] to be a valid video token.
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

        # Analytic, parameter-free interval posterior.
        segment_prob = start_prob.unsqueeze(-1) * end_prob.unsqueeze(-2)
        segment_prob = segment_prob * segment_valid.to(dtype)
        segment_prob = segment_prob / segment_prob.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(1e-8)

        index_f = torch.arange(
            time_steps,
            dtype=dtype,
            device=device,
        )
        start_grid = index_f.view(1, time_steps, 1)
        end_grid = index_f.view(1, 1, time_steps)

        expected_start = (segment_prob * start_grid).sum(dim=(-2, -1))
        expected_end = (segment_prob * end_grid).sum(dim=(-2, -1))
        expected_length = (
            segment_prob * span_len.clamp_min(1).to(dtype).unsqueeze(0)
        ).sum(dim=(-2, -1))

        # Integer endpoints are compatibility/evaluation views of the posterior
        # barycenter, NOT MAP decisions and NOT used as the training support.
        selected_start = torch.round(expected_start).long().clamp(
            min=0, max=max(time_steps - 1, 0)
        )
        selected_end = torch.round(expected_end).long().clamp(
            min=0, max=max(time_steps - 1, 0)
        )
        selected_end = torch.maximum(selected_end, selected_start)

        selected_mask = self._span_mask_from_bounds(
            selected_start,
            selected_end,
            video_mask,
        )
        selected_span_weights = self._uniform_weights_from_mask(
            selected_mask,
            dtype=dtype,
        )

        # IMPORTANT: the temporal representation used for training/inference is
        # the same Support Field, not the rounded interval mask above.
        training_temporal_prob = support_prob
        main_temporal_prob = support_prob
        soft_temporal_prob = support_prob

        token_count = video_mask.sum(dim=-1).to(dtype).clamp_min(1.0)
        selected_length = selected_mask.sum(dim=-1).to(dtype)
        selected_fraction = selected_length / token_count
        expected_fraction = expected_length / token_count
        denominator = (token_count - 1.0).clamp_min(1.0)
        selected_start_norm = (
            expected_start / denominator
        ).clamp(0.0, 1.0)
        selected_end_norm = (
            expected_end / denominator
        ).clamp(0.0, 1.0)

        flat_prob = segment_prob.reshape(batch_size, -1)
        flat_valid = segment_valid.reshape(batch_size, -1)
        segment_entropy, segment_entropy_ratio = (
            self._entropy_from_flat_distribution(flat_prob, flat_valid)
        )
        segment_top1_prob = flat_prob.max(dim=-1).values

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

        start_diag = self._distribution_diagnostics(start_prob, video_mask)
        end_diag = self._distribution_diagnostics(end_prob, video_mask)

        return {
            "start_boundary_evidence": start_evidence,
            "end_boundary_evidence": end_evidence,
            "start_boundary_prob": start_prob,
            "end_boundary_prob": end_prob,
            "start_boundary_entropy_ratio": start_diag["entropy_ratio"],
            "end_boundary_entropy_ratio": end_diag["entropy_ratio"],
            "segment_valid_mask": segment_valid,
            "segment_prob": segment_prob,
            "segment_entropy": segment_entropy,
            "segment_entropy_ratio": segment_entropy_ratio,
            "segment_top1_prob": segment_top1_prob,
            "training_temporal_prob": training_temporal_prob,
            "main_temporal_prob": main_temporal_prob,
            "soft_temporal_prob": soft_temporal_prob,
            "selected_span_frame_mask": selected_mask.detach(),
            "selected_span_frame_weights": selected_span_weights,
            "selected_span_start_frame": selected_start.detach(),
            "selected_span_end_frame": selected_end.detach(),
            "selected_span_start": selected_start_norm,
            "selected_span_end": selected_end_norm,
            "barycenter_start_frame": expected_start,
            "barycenter_end_frame": expected_end,
            "selected_span_length": selected_length,
            "selected_span_fraction": selected_fraction,
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

    # ==================================================================
    # V3.8-M1: raw visual interface, called explicitly by the M1 caller.
    # ==================================================================
    def export_v38_m1_inputs(
        self,
        video_tokens: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Export pre-Core video features for M1's independent frame scorer.

        Pass the same original visual tensor supplied to the legacy core,
        not ``where_context_states``, projected visual evidence, or CRA's
        contextual visual embeddings. This method does no temporal mixing,
        projection, question/answer encoding, or bank construction. The
        caller owns the text embeddings and the trainable M1 projectors.

        The detached [B,T,input_dim] feature tensor is zero on padding.
        The bank receives the matching left-aligned boolean mask and the
        per-row valid lengths. Calling this adapter never runs the legacy
        forward, and legacy/M0 callers need not call it.
        """
        if not torch.is_tensor(video_tokens):
            raise TypeError("M1 video_tokens must be a torch.Tensor.")
        if video_tokens.ndim != 3:
            raise ValueError("M1 video_tokens must be [B,T,input_dim].")
        if not torch.is_floating_point(video_tokens):
            raise TypeError("M1 video_tokens must be floating point.")
        batch_size, time_steps, feature_dim = video_tokens.shape
        if batch_size < 1 or time_steps < 1 or feature_dim != self.input_dim:
            raise ValueError("M1 expects nonempty [B,T,input_dim] video features.")
        if video_mask is not None:
            if not torch.is_tensor(video_mask):
                raise TypeError("M1 video_mask must be a torch.Tensor.")
            video_mask = video_mask.to(device=video_tokens.device)
        valid_mask = self._normalize_video_mask(
            video_mask, batch_size, time_steps, video_tokens.device,
        )
        if bool(((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any()):
            raise ValueError("M1 fixed windows require a left-aligned video mask.")

        frame_features = video_tokens.detach().masked_fill(
            ~valid_mask.unsqueeze(-1), 0.0,
        )
        if not bool(torch.isfinite(frame_features).all()):
            raise ValueError("M1 valid video features must be finite.")
        return {
            "qbound_v38_m1_frame_features": frame_features,
            "qbound_v38_m1_video_mask": valid_mask,
            "qbound_v38_m1_valid_lengths": valid_mask.long().sum(dim=-1),
        }

    # ==================================================================
    # V3.8-Minimal (M0): additive encoder/decoder adapters
    # ==================================================================
    def _v38_minimal_encoder_view(
        self,
        where_context_states: torch.Tensor,
        question_tokens_projected: torch.Tensor,
        video_mask: torch.Tensor,
        question_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Expose the unchanged encoder to the external joint-event head.

        Contract:
          * ``qbound_v38_event_frame_states``: H [B,T,hidden_dim]; visual
            temporal context BEFORE semantic-seed scoring/graph propagation.
            Do NOT use seed probability or old support as an M0 event prior.
          * ``qbound_v38_question_embedding``: q [B,hidden_dim], mean of
            the EXISTING projected, valid question tokens; answer-free.
          * The masks identify valid frames/tokens; downstream event windows
            must be generated per sample length, never on padded positions.
          * No new learnable projection, learned boundary, gate, or event graph.
          * Autograd is deliberately preserved. Freeze/detach controls for the
            P2 frozen-encoder probe belong to the caller, not this adapter.
        """
        if where_context_states.ndim != 3 or question_tokens_projected.ndim != 3:
            raise ValueError("V3.8 encoder view requires [B,T,d] and [B,L,d].")
        if where_context_states.shape[:2] != video_mask.shape:
            raise ValueError("V3.8 frame-state shape does not match video_mask.")
        if question_tokens_projected.shape[:2] != question_mask.shape:
            raise ValueError("V3.8 question-token shape does not match question_mask.")
        if where_context_states.size(0) != question_tokens_projected.size(0):
            raise ValueError("V3.8 video/question batch sizes differ.")
        if (where_context_states.size(-1) != self.hidden_dim or
                question_tokens_projected.size(-1) != self.hidden_dim):
            raise ValueError("V3.8 projected feature dimension mismatch.")

        question_embedding = self._masked_mean(
            question_tokens_projected,
            question_mask,
            dim=1,
        )
        return {
            "qbound_v38_event_frame_states": where_context_states,
            "qbound_v38_question_embedding": question_embedding,
            "qbound_v38_question_tokens": question_tokens_projected,
            "qbound_v38_event_video_mask": video_mask,
            "qbound_v38_event_question_mask": question_mask,
            "qbound_v38_event_valid_lengths": video_mask.long().sum(dim=-1),
        }

    def forward_v38_features(
        self,
        video_tokens: torch.Tensor,
        question_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """P2 probe path: encoder inputs only, without legacy Seed/Support/R1.

        Uses exactly the retained visual+temporal encoder weights. Call
        ``eval()`` for deterministic equivalence checks with full forward;
        freezing backbone parameters, if desired, is an external train policy.
        This method does NOT score answers or generate candidate windows.
        """
        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must be [B,T,D].")
        batch_size, time_steps, feature_dim = video_tokens.shape
        if time_steps < 1 or feature_dim != self.input_dim:
            raise ValueError("V3.8 expects nonempty video [B,T,input_dim].")
        video_mask = self._normalize_video_mask(
            video_mask, batch_size, time_steps, video_tokens.device,
        )
        question_features, question_mask = self._normalize_question_features(
            question_features, question_mask, batch_size, video_tokens.device,
        )
        visual = self._encode_visual_evidence(video_tokens, video_mask)
        temporal = self._encode_temporal_context(
            visual["visual_evidence_states"], video_mask,
        )
        projected_question_tokens = self.question_token_proj(question_features)
        projected_question_tokens = (
            projected_question_tokens
            * question_mask.to(projected_question_tokens.dtype).unsqueeze(-1)
        )
        return self._v38_minimal_encoder_view(
            where_context_states=temporal["where_context_states"],
            question_tokens_projected=projected_question_tokens,
            video_mask=video_mask,
            question_mask=question_mask,
        )

    def decode_v38_event_support(
        self,
        event_temporal_prob: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Apply the EXACT existing core analytic interval decoder to M0 support.

        The external joint scorer must first push forward its event posterior
        using target-window footprints into a normalized distribution [B,T].
        Context-window positions must NOT be included in this footprint.

        This is an adapter to ``_support_to_interval_posterior`` and does not
        invent a new R1 decoder. If ``refine.py`` wraps or further refines R1,
        its existing downstream path must be preserved during integration.
        """
        if event_temporal_prob.ndim != 2:
            raise ValueError("event_temporal_prob must be [B,T].")
        if video_mask.shape != event_temporal_prob.shape:
            raise ValueError("video_mask must match event_temporal_prob [B,T].")
        if video_mask.dtype != torch.bool:
            raise TypeError("video_mask must be boolean.")
        if event_temporal_prob.size(1) < 1 or not torch.all(video_mask.any(dim=-1)):
            raise ValueError("Every sample requires at least one valid frame.")
        if not bool(torch.isfinite(event_temporal_prob).all()):
            raise ValueError("Nonfinite V3.8 event temporal probabilities.")
        if bool((event_temporal_prob < -1e-7).any()):
            raise ValueError("V3.8 event temporal probabilities must be nonnegative.")
        if bool(event_temporal_prob.masked_select(~video_mask).abs().gt(1e-6).any()):
            raise ValueError("V3.8 event support must have zero mass on padding.")
        masses = (event_temporal_prob * video_mask.to(event_temporal_prob.dtype)).sum(dim=-1)
        if not bool(torch.allclose(masses, torch.ones_like(masses), atol=1e-4, rtol=1e-4)):
            raise ValueError("V3.8 event support must sum to 1 on valid frames.")
        # Existing decoder is called unchanged; no reweighting or learned
        # boundary manipulation is introduced by this method.
        return self._support_to_interval_posterior(event_temporal_prob, video_mask)

    # ==================================================================
    # Forward
    # ==================================================================
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
            raise ValueError("Q-Bound V2.8 requires at least one timestep.")

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

        # VALUE remains original masked CLIP.
        value_tokens = (
            video_tokens
            * video_mask.to(video_tokens.dtype).unsqueeze(-1)
        )

        # --------------------------------------------------------------
        # Relevance branch: Video + Question -> semantic seed.
        # --------------------------------------------------------------
        visual = self._encode_visual_evidence(
            video_tokens=video_tokens,
            video_mask=video_mask,
        )
        temporal = self._encode_temporal_context(
            visual_evidence=visual["visual_evidence_states"],
            video_mask=video_mask,
        )
        relevance = self._discover_semantic_seed(
            where_context=temporal["where_context_states"],
            question_features=question_features,
            question_mask=question_mask,
            video_mask=video_mask,
        )

        # V3.8-M0: export H and q; the future bank/scorer receives neither
        # legacy Seed/Support probabilities nor ground-truth temporal labels.
        v38_encoder = self._v38_minimal_encoder_view(
            where_context_states=temporal["where_context_states"],
            question_tokens_projected=relevance["question_tokens_projected"],
            video_mask=video_mask,
            question_mask=question_mask,
        )

        # --------------------------------------------------------------
        # Extent/support branch: VIDEO ONLY -> structural graph.
        # --------------------------------------------------------------
        support_structure = (
            self._learn_video_support_structure(
                video_tokens=video_tokens, video_mask=video_mask,
            ) if self.support_predictor_enabled else
            self._fixed_video_support_structure(
                video_tokens=video_tokens, video_mask=video_mask,
            )
        )

        # --------------------------------------------------------------
        # No-bypass composition.
        # Question-derived seed may pass through the detached structural graph,
        # but there is no direct Question -> span scorer.
        # --------------------------------------------------------------
        support = self._propagate_seed_to_support(
            semantic_seed_prob=relevance["semantic_seed_prob"],
            support_laplacian=support_structure["support_graph_laplacian"],
            video_mask=video_mask,
        )
        intervals = self._support_to_interval_posterior(
            support_prob=support["support_temporal_prob"],
            video_mask=video_mask,
        )

        dtype = video_tokens.dtype
        device = video_tokens.device
        one = torch.ones((), dtype=dtype, device=device)
        zero = torch.zeros((), dtype=dtype, device=device)

        return {
            # ----------------------------------------------------------
            # Architecture / causal contract
            # ----------------------------------------------------------
            "qbound_version": self.qbound_version,

            "structured_visual_evidence_enabled": one,
            "visual_distinctiveness_enabled": one,
            "visual_transition_enabled": one,
            "temporal_context_enabled": one,

            "relevance_extent_decoupling_enabled": one,
            "token_resolved_question_relevance_enabled": one,
            "semantic_evidence_seed_enabled": one,
            "question_direct_span_scoring_enabled": zero,
            "question_as_visual_value_enabled": zero,

            "video_only_support_branch_enabled": one if self.support_predictor_enabled else zero,
            "predictive_support_graph_enabled": one if self.support_predictor_enabled else zero,
            "support_predictor_enabled_flag": one if self.support_predictor_enabled else zero,
            "support_predictor_target_leakage_enabled": zero,
            "support_branch_uses_question_enabled": zero,
            "grounding_gradient_to_support_graph_enabled": zero,
            "support_graph_detached_for_grounding_enabled": one,

            "structural_seed_propagation_enabled": one,
            "analytic_boundary_posterior_enabled": one,
            "posterior_barycentric_grounding_enabled": one,
            "same_support_for_training_and_inference_enabled": one,
            "value_where_separation_enabled": one,
            # Encoder adapter is available; this flag does not claim the new
            # event bank or evidence head has already been constructed.
            "v38_minimal_encoder_adapter_available": one,

            # Explicitly removed shortcut mechanisms.
            "query_routed_visual_context_enabled": zero,
            "evidence_core_enabled": zero,
            "event_continuity_enabled": zero,
            "joint_transition_continuity_enabled": zero,
            "core_to_closure_enabled": zero,
            "additive_extent_energy_enabled": zero,
            "neural_span_scorer_enabled": zero,
            "event_candidate_selector_enabled": zero,
            "hard_map_temporal_decision_enabled": zero,
            "boundary_refiner_enabled": zero,
            "local_candidate_refinement_enabled": zero,
            "adaptive_radius_enabled": zero,
            "learned_routing_gate_enabled": zero,
            "independent_endpoint_heads_enabled": zero,
            "answer_features_used_in_core": zero,
            "temporal_gt_used_in_core": zero,
            "answer_conditioned_temporal_critic_enabled": zero,
            "length_prior_enabled": zero,
            "gaussian_smoothing_enabled": zero,
            "fixed_event_count_enabled": zero,
            "minimum_event_duration_enabled": zero,

            # ----------------------------------------------------------
            # Core tensors
            # ----------------------------------------------------------
            "video_mask": video_mask,
            "question_mask": question_mask,
            "value_tokens": value_tokens,
            **visual,
            **temporal,
            **relevance,
            **support_structure,
            **support,
            **intervals,
            **v38_encoder,

            # ----------------------------------------------------------
            # Canonical downstream aliases for V2.8 migration
            # ----------------------------------------------------------
            "qbound_value_tokens": value_tokens,
            "qbound_visual_evidence_states": visual["visual_evidence_states"],
            "qbound_visual_transition_states": visual["visual_transition_states"],
            "qbound_where_states": temporal["where_context_states"],

            "qbound_semantic_seed_logits": relevance["semantic_seed_logits"],
            "qbound_semantic_seed_prob": relevance["semantic_seed_prob"],
            "qbound_semantic_seed_entropy_ratio": relevance[
                "semantic_seed_entropy_ratio"
            ],
            "qbound_semantic_seed_top1_prob": relevance[
                "semantic_seed_top1_prob"
            ],

            "qbound_support_predictive_loss": support_structure[
                "support_predictive_loss"
            ],
            "qbound_support_edge_surprise": support_structure[
                "support_edge_surprise"
            ],
            "qbound_support_edge_affinity": support_structure[
                "support_edge_affinity"
            ],
            "qbound_support_graph_laplacian": support_structure[
                "support_graph_laplacian"
            ],

            "qbound_support_temporal_prob": support["support_temporal_prob"],
            "qbound_support_entropy_ratio": support["support_entropy_ratio"],
            "qbound_support_top1_prob": support["support_top1_prob"],
            "qbound_support_effective_frames": support[
                "support_effective_frames"
            ],

            "qbound_segment_prob": intervals["segment_prob"],
            "qbound_training_temporal_prob": intervals[
                "training_temporal_prob"
            ],
            "qbound_soft_temporal_prob": intervals["soft_temporal_prob"],
            "qbound_main_temporal_prob": intervals["main_temporal_prob"],
            # Legacy name retained only as an alias to the SAME continuous
            # support field; it is NOT a hard-MAP span in V2.8.
            "qbound_hard_temporal_prob": intervals["main_temporal_prob"],

            "qbound_span_start_frame": intervals["selected_span_start_frame"],
            "qbound_span_end_frame": intervals["selected_span_end_frame"],
            "qbound_span_frame_mask": intervals["selected_span_frame_mask"],
            "qbound_span_length": intervals["selected_span_length"],
            "qbound_span_fraction": intervals["selected_span_fraction"],
            "qbound_barycenter_start_frame": intervals[
                "barycenter_start_frame"
            ],
            "qbound_barycenter_end_frame": intervals[
                "barycenter_end_frame"
            ],
            "qbound_soft_expected_span_length": intervals[
                "soft_expected_span_length"
            ],
            "qbound_soft_expected_span_fraction": intervals[
                "soft_expected_span_fraction"
            ],
            "qbound_segment_entropy_ratio": intervals[
                "segment_entropy_ratio"
            ],
            "qbound_segment_top1_prob": intervals["segment_top1_prob"],
        }

    # ==================================================================
    # V3.5 / T2: optional same-video paired-question training interface.
    # ==================================================================
    def forward_paired(
        self,
        video_tokens: torch.Tensor,
        question_features: torch.Tensor,
        paired_question_features: torch.Tensor,
        video_mask: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        paired_question_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run two questions against ONE shared visual representation/graph.

        V3.5 / QDTG contract:
          * ``primary`` retains every canonical R2-C ``forward()`` field;
            V3.8 adds only independent encoder-view aliases.
          * Only the paired question's relevance and support are recomputed.
            Temporal encoding, masks, and the fixed chain graph are reused.
          * Both supports remain differentiable all the way to the SAME Seed
            scorer; neither support nor its logits are detached here.
          * No answers, ground-truth intervals, pseudo labels, learned graph,
            second selector, extra parameters, or modified decoder are used.
          * This method is opt-in: normal ``forward()``/inference is unchanged.

        The caller (RefineDecoder / pipeline) must form legitimate same-video
        question pairs, apply any QDTG loss, and implement pair filtering.
        No QDTG supervision, projection, or losses are computed in QBoundCore.

        Returns:
            (primary, paired)
            primary: complete legacy R2-C fields plus M0 encoder-view aliases.
            paired: paired question Seed/Support + M0 encoder-view aliases;
                    no redundant interval posterior.
        """
        primary = self.forward(
            video_tokens=video_tokens,
            question_features=question_features,
            video_mask=video_mask,
            question_mask=question_mask,
        )

        batch_size = primary["video_mask"].size(0)
        paired_features, paired_mask = self._normalize_question_features(
            paired_question_features,
            paired_question_mask,
            batch_size=batch_size,
            device=video_tokens.device,
        )
        if paired_features.device != video_tokens.device:
            raise ValueError(
                "paired_question_features must be on the same device as video_tokens."
            )

        # Reuse the SAME temporal states as primary; do not run temporal
        # self-attention twice with different dropout masks for one video.
        paired_relevance = self._discover_semantic_seed(
            where_context=primary["where_context_states"],
            question_features=paired_features,
            question_mask=paired_mask,
            video_mask=primary["video_mask"],
        )

        # R2-C propagation is parameter-free and uses the exact SAME fixed
        # Laplacian as primary. Autograd still flows to paired Seed logits.
        paired_support = self._propagate_seed_to_support(
            semantic_seed_prob=paired_relevance["semantic_seed_prob"],
            support_laplacian=primary["support_graph_laplacian"],
            video_mask=primary["video_mask"],
        )

        paired_v38_encoder = self._v38_minimal_encoder_view(
            where_context_states=primary["where_context_states"],
            question_tokens_projected=paired_relevance["question_tokens_projected"],
            video_mask=primary["video_mask"],
            question_mask=paired_mask,
        )

        paired = {
            **paired_v38_encoder,
            "video_mask": primary["video_mask"],
            "question_mask": paired_mask,
            "semantic_seed_logits": paired_relevance["semantic_seed_logits"],
            "semantic_seed_prob": paired_relevance["semantic_seed_prob"],
            "semantic_seed_entropy_ratio": paired_relevance[
                "semantic_seed_entropy_ratio"
            ],
            "support_temporal_prob": paired_support["support_temporal_prob"],
            "support_entropy_ratio": paired_support["support_entropy_ratio"],
            "support_effective_frames": paired_support["support_effective_frames"],
            # Canonical aliases consumed by the future V3.5 adapter.
            "qbound_semantic_seed_logits": paired_relevance["semantic_seed_logits"],
            "qbound_semantic_seed_prob": paired_relevance["semantic_seed_prob"],
            "qbound_support_temporal_prob": paired_support["support_temporal_prob"],
        }
        return primary, paired
