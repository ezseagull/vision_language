
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QBoundCore(nn.Module):
    """
    Q-Bound V2.1.0 -- Query-Conditioned Latent Segment Composition.

    Scientific motivation
    ---------------------
    V2.0 coupled predicted temporal cuts to information propagation.  The QA
    objective could therefore improve simply by closing cuts, producing the
    observed one-event collapse.  V2.0 also used independent start/end heads,
    which admitted a second shortcut: both endpoints could collapse onto one
    highly discriminative frame, producing micro-spans and a large soft/hard
    grounding gap.

    V2.1 removes both mechanisms.

    Core hypothesis
    ---------------
        Events are not pre-partitioned from the whole video.  A query-specific
        evidence event is composed from local temporal primitives according to

          (1) primitive relevance to the question,
          (2) coherence between adjacent primitives,
          (3) separation from neighboring primitives.

    V2.1 deliberately avoids both raw-sum and simple-mean span scoring:
        - a raw sum structurally favors long spans;
        - a simple mean has a dangerous opposite bias: the maximum mean over
          all contiguous spans is attained by a single maximum-scoring token.

    Instead, V2.1 centers node/edge evidence within each video and accumulates
    it with sqrt(length) normalization.  A coherent run of positive evidence
    can therefore beat a singleton, while adding irrelevant/negative evidence
    hurts the segment.  This is a scale-neutral structural choice rather than
    a target-duration prior.

    Importantly, temporal structure NEVER gates the primitive encoder.  The
    model first builds local frame/clip primitives, injects question semantics,
    then scores all variable-length contiguous segments using one shared energy
    function E(s,e).

    Input
    -----
        video_tokens:
            [B, T, D], expected to be temporally local frozen visual features
            (for the current CRA/NExT-GQA testbed: CLIP-L frame features).

        question_features:
            [B, D] or [B, L, D].
            V2.1 prefers token-level [B,L,D] features.

        video_mask:
            optional [B,T] boolean / 0-1 mask.

        question_mask:
            optional [B,L] boolean / 0-1 mask.  If omitted, every supplied
            question feature is treated as valid.

    Output
    ------
        - local primitive states and query-conditioned primitive states;
        - node relevance u_t;
        - adjacent relation/cohesion r_t;
        - segment_energy / segment_prob over all s <= e;
        - a hard segment H* = argmax E(s,e);
        - a soft differentiable inside distribution derived from the SAME
          segment energy;
        - a straight-through temporal distribution whose forward value is the
          hard segment and whose backward path follows the soft segment model.

    Weak-supervision contract
    -------------------------
    The core receives NO answer candidates, NO GT answer and NO temporal GT.
    Answer-only candidates may supervise the selected evidence downstream, but
    they must never enter this segment scorer.

    No V2.0 mechanisms are retained:
        - no cut field;
        - no cut-gated information propagation;
        - no event-count objective;
        - no independent start/end heads;
        - no functional KEEP/MERGE verifier;
        - no boundary-equilibrium objective;
        - no teacher distribution.

    Complexity
    ----------
    Primitive encoding and text interaction are O(TL).
    Segment energy materialization is O(T^2).  For NExT-GQA T=32, this is only
    528 valid contiguous segments and, unlike V1.12, they are NOT independent
    semantic hypothesis objects: every score is analytically composed from the
    same T node scores and T-1 adjacent-relation scores.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_generic_input_adapter: bool = True,
        input_adapter_dropout: Optional[float] = None,
        num_question_heads: int = 4,
        question_ff_multiplier: int = 2,
        segment_temperature: float = 1.0,
        relation_weight: float = 0.50,
        boundary_weight: float = 0.25,
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
        if num_question_heads < 1 or hidden_dim % num_question_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by num_question_heads."
            )
        if question_ff_multiplier < 1:
            raise ValueError("question_ff_multiplier must be >= 1.")
        if segment_temperature <= 0.0:
            raise ValueError("segment_temperature must be positive.")
        if relation_weight < 0.0:
            raise ValueError("relation_weight must be non-negative.")
        if boundary_weight < 0.0:
            raise ValueError("boundary_weight must be non-negative.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.qbound_version = "2.1.0"

        self.use_generic_input_adapter = bool(use_generic_input_adapter)
        self.input_adapter_dropout_p = float(input_adapter_dropout)
        self.num_question_heads = int(num_question_heads)
        self.segment_temperature = float(segment_temperature)
        self.relation_weight = float(relation_weight)
        self.boundary_weight = float(boundary_weight)

        # ------------------------------------------------------------------
        # 0. Generic visual input adapter.
        # ------------------------------------------------------------------
        # This is point-wise across time.  It cannot mix information between
        # temporal positions and therefore cannot create fake local grounding.
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
        # 1. LOCAL primitive encoder.
        # ------------------------------------------------------------------
        # No temporal self-attention and no recurrent global scan are allowed
        # before segment selection.  Each primitive sees only its own visual
        # feature and local first-/second-order temporal changes.
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
        # Relative time is an explicit local feature.  It gives the model an
        # order cue for words such as before/after without mixing video tokens.
        self.position_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
        )
        self.primitive_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # 2. Token-level question injection.
        # ------------------------------------------------------------------
        # Every visual primitive independently attends to the question tokens.
        # There is still no visual-to-visual communication here.
        self.question_token_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.visual_to_question_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=self.num_question_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.question_cross_norm = nn.LayerNorm(hidden_dim)
        ff_dim = hidden_dim * int(question_ff_multiplier)
        self.question_ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.question_ff_norm = nn.LayerNorm(hidden_dim)

        # ------------------------------------------------------------------
        # 3. Primitive relevance u_t.
        # ------------------------------------------------------------------
        # query-conditioned primitive, global q, interaction, dynamics, time.
        node_input_dim = hidden_dim * 3 + 3
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.node_score = nn.Linear(hidden_dim, 1)

        # ------------------------------------------------------------------
        # 4. Adjacent relation/cohesion r_t.
        # ------------------------------------------------------------------
        # A positive relation means "these adjacent primitives coherently belong
        # together for this question"; a negative relation is boundary evidence.
        edge_input_dim = hidden_dim * 5 + 4
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.edge_score = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.node_score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.node_score.bias)
        nn.init.normal_(self.edge_score.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.edge_score.bias)

    # ------------------------------------------------------------------
    # Generic helpers
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
                batch_size, q_len, dtype=torch.bool, device=device
            )
        else:
            while question_mask.ndim > 2 and question_mask.size(1) == 1:
                question_mask = question_mask.squeeze(1)
            if question_mask.ndim != 2 or question_mask.shape != (batch_size, q_len):
                raise ValueError(
                    "question_mask must reduce to [B,L]; expected "
                    f"{(batch_size, q_len)}, got {tuple(question_mask.shape)}."
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
    def _entropy_from_distribution(
        distribution: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        p = distribution * mask.to(distribution.dtype)
        entropy = -(p * torch.log(p.clamp_min(1e-8))).sum(dim=(-2, -1))
        valid_count = mask.sum(dim=(-2, -1)).to(distribution.dtype).clamp_min(1.0)
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
            time_steps, device=video_mask.device, dtype=torch.long
        ).view(1, time_steps)
        return (
            (index >= start.unsqueeze(-1))
            & (index <= end.unsqueeze(-1))
            & video_mask
        )

    @staticmethod
    def _mask_to_uniform_weights(
        mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weights = mask.to(dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    # ------------------------------------------------------------------
    # 0. Point-wise input adaptation
    # ------------------------------------------------------------------
    def _adapt_visual_tokens(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mask_f = video_mask.to(video_tokens.dtype).unsqueeze(-1)
        raw = video_tokens * mask_f

        if self.use_generic_input_adapter:
            adapted = self.generic_input_proj(video_tokens)
            adapted = self.generic_input_gelu(adapted)
            adapted = self.generic_input_norm(adapted)
            adapted = self.generic_input_dropout(adapted)
        else:
            adapted = video_tokens

        adapted = adapted * mask_f

        valid_count = video_mask.sum(dim=-1).to(video_tokens.dtype).clamp_min(1.0)
        raw_unit = F.normalize(raw, p=2, dim=-1)
        adapted_unit = F.normalize(adapted, p=2, dim=-1)
        cosine = (raw_unit * adapted_unit).sum(dim=-1)
        cosine = (
            cosine * video_mask.to(cosine.dtype)
        ).sum(dim=-1) / valid_count

        delta_ratio = torch.linalg.vector_norm(
            (adapted - raw) * mask_f, dim=-1
        )
        raw_norm = torch.linalg.vector_norm(raw, dim=-1)
        delta_ratio = (
            delta_ratio * video_mask.to(delta_ratio.dtype)
        ).sum(dim=-1) / (
            (raw_norm * video_mask.to(raw_norm.dtype))
            .sum(dim=-1)
            .clamp_min(1e-6)
        )

        return adapted, {
            "generic_input_adapter_enabled": torch.as_tensor(
                float(self.use_generic_input_adapter),
                dtype=video_tokens.dtype,
                device=video_tokens.device,
            ),
            "generic_input_token_cosine": cosine,
            "generic_input_delta_ratio": delta_ratio,
        }

    # ------------------------------------------------------------------
    # 1. Local temporal primitives
    # ------------------------------------------------------------------
    def _encode_local_primitives(
        self,
        content_tokens: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, _ = content_tokens.shape
        mask_f = video_mask.to(content_tokens.dtype).unsqueeze(-1)

        first_delta = torch.zeros_like(content_tokens)
        if time_steps > 1:
            edge_valid = video_mask[:, 1:] & video_mask[:, :-1]
            first_delta[:, 1:] = content_tokens[:, 1:] - content_tokens[:, :-1]
            first_delta[:, 1:] *= edge_valid.to(content_tokens.dtype).unsqueeze(-1)

        second_delta = torch.zeros_like(content_tokens)
        if time_steps > 2:
            second_valid = (
                video_mask[:, 2:]
                & video_mask[:, 1:-1]
                & video_mask[:, :-2]
            )
            second_delta[:, 2:] = first_delta[:, 2:] - first_delta[:, 1:-1]
            second_delta[:, 2:] *= second_valid.to(content_tokens.dtype).unsqueeze(-1)

        state_h = self.state_proj(content_tokens)
        delta_h = self.delta_proj(first_delta)
        accel_h = self.accel_proj(second_delta)

        denom = float(max(time_steps - 1, 1))
        position = torch.arange(
            time_steps,
            device=content_tokens.device,
            dtype=content_tokens.dtype,
        ).view(1, time_steps, 1) / denom
        position = position.expand(batch_size, -1, -1)
        position_h = self.position_proj(position)

        primitive_states = self.primitive_fuse(
            torch.cat([state_h, delta_h, accel_h, position_h], dim=-1)
        )
        primitive_states = primitive_states * mask_f

        scale = float(max(self.input_dim, 1)) ** 0.5
        delta_magnitude = torch.linalg.vector_norm(first_delta, dim=-1) / scale
        accel_magnitude = torch.linalg.vector_norm(second_delta, dim=-1) / scale
        delta_magnitude *= video_mask.to(delta_magnitude.dtype)
        accel_magnitude *= video_mask.to(accel_magnitude.dtype)

        return {
            "primitive_states": primitive_states,
            "first_delta": first_delta,
            "second_delta": second_delta,
            "delta_magnitude": delta_magnitude,
            "accel_magnitude": accel_magnitude,
            "relative_position": position.squeeze(-1),
        }

    # ------------------------------------------------------------------
    # 2. Question-conditioned primitive semantics
    # ------------------------------------------------------------------
    def _inject_question(
        self,
        primitive_states: torch.Tensor,
        question_features: torch.Tensor,
        question_mask: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        q_tokens = self.question_token_proj(question_features)
        q_tokens = q_tokens * question_mask.to(q_tokens.dtype).unsqueeze(-1)
        q_global = self._masked_mean(q_tokens, question_mask, dim=1)

        cross, attn = self.visual_to_question_attn(
            query=primitive_states,
            key=q_tokens,
            value=q_tokens,
            key_padding_mask=~question_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        q_states = self.question_cross_norm(primitive_states + cross)
        q_states = self.question_ff_norm(q_states + self.question_ff(q_states))
        q_states = q_states * video_mask.to(q_states.dtype).unsqueeze(-1)

        primitive_norm = torch.linalg.vector_norm(primitive_states, dim=-1)
        cross_delta = torch.linalg.vector_norm(
            (q_states - primitive_states)
            * video_mask.to(q_states.dtype).unsqueeze(-1),
            dim=-1,
        )
        valid = video_mask.to(cross_delta.dtype)
        interaction_ratio = (cross_delta * valid).sum(dim=-1) / (
            (primitive_norm * valid).sum(dim=-1).clamp_min(1e-6)
        )

        return {
            "question_tokens_projected": q_tokens,
            "question_state": q_global,
            "query_conditioned_states": q_states,
            "visual_question_attention": attn,
            "question_interaction_delta_ratio": interaction_ratio,
        }

    # ------------------------------------------------------------------
    # 3-4. Node relevance and adjacent relation
    # ------------------------------------------------------------------
    def _score_primitives_and_relations(
        self,
        query_states: torch.Tensor,
        question_state: torch.Tensor,
        delta_magnitude: torch.Tensor,
        accel_magnitude: torch.Tensor,
        relative_position: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, time_steps, hidden_dim = query_states.shape
        q_frames = question_state.unsqueeze(1).expand(-1, time_steps, -1)

        node_features = torch.cat(
            [
                query_states,
                q_frames,
                query_states * q_frames,
                delta_magnitude.unsqueeze(-1),
                accel_magnitude.unsqueeze(-1),
                relative_position.unsqueeze(-1),
            ],
            dim=-1,
        )
        node_hidden = self.node_encoder(node_features)
        node_raw = self.node_score(node_hidden).squeeze(-1)

        # Bound the primitive score.  This stabilizes length-normalized segment
        # energies and prevents a single frame from winning through arbitrarily
        # large logits.
        node_relevance = torch.tanh(node_raw)
        node_relevance = node_relevance * video_mask.to(node_relevance.dtype)

        if time_steps == 1:
            edge_mask = video_mask.new_zeros(batch_size, 0)
            edge_raw = query_states.new_zeros(batch_size, 0)
            edge_relation = edge_raw
        else:
            edge_mask = video_mask[:, :-1] & video_mask[:, 1:]
            left = query_states[:, :-1]
            right = query_states[:, 1:]
            q_edges = question_state.unsqueeze(1).expand(-1, time_steps - 1, -1)

            # Local dynamics attached to the right primitive correspond to the
            # change from t -> t+1.
            edge_delta = delta_magnitude[:, 1:]
            edge_accel = accel_magnitude[:, 1:]
            edge_pos = 0.5 * (
                relative_position[:, :-1] + relative_position[:, 1:]
            )

            edge_features = torch.cat(
                [
                    left,
                    right,
                    torch.abs(right - left),
                    left * right,
                    q_edges,
                    edge_delta.unsqueeze(-1),
                    edge_accel.unsqueeze(-1),
                    edge_pos.unsqueeze(-1),
                    (left * q_edges).mean(dim=-1, keepdim=True),
                ],
                dim=-1,
            )
            edge_hidden = self.edge_encoder(edge_features)
            edge_raw = self.edge_score(edge_hidden).squeeze(-1)
            edge_relation = torch.tanh(edge_raw)
            edge_relation = edge_relation * edge_mask.to(edge_relation.dtype)

        return {
            "node_relevance_raw": node_raw,
            "node_relevance": node_relevance,
            "adjacent_relation_raw": edge_raw,
            "adjacent_relation": edge_relation,
            "adjacent_relation_mask": edge_mask,
        }

    # ------------------------------------------------------------------
    # 5. Shared variable-scale segment energy
    # ------------------------------------------------------------------
    def _build_segment_energy(
        self,
        node_relevance: torch.Tensor,
        adjacent_relation: torch.Tensor,
        video_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build one shared variable-scale energy E(s,e).

        Critical anti-shortcut choice
        -----------------------------
        We do NOT use

            mean(node scores in span)

        as the main segment score, because the maximum average over all
        contiguous spans is mathematically achieved by a singleton containing
        the maximum-scoring token.  That would recreate V2.0's micro-span
        shortcut by construction.

        We also do NOT use an unnormalized raw sum, which would reward longer
        spans whenever scores are mostly positive.

        Instead, evidence is centered within each video and accumulated with
        sqrt(length) normalization:

            node_support(s,e)
                = sum_{t=s..e} (u_t - mean_valid(u)) / sqrt(L)

            relation_support(s,e)
                = sum_{t=s..e-1} (r_t - mean_valid(r)) / sqrt(L-1)

        A consistently relevant/coherent run becomes stronger as it grows,
        while irrelevant frames dilute or reverse the accumulated support.
        Full-video node support is approximately zero by construction, and
        single-frame spans receive no automatic advantage from the scoring
        formula.

        A boundary-separation term additionally penalizes cutting through an
        adjacent pair that the model itself considers strongly coherent.
        """
        batch_size, time_steps = node_relevance.shape
        dtype = node_relevance.dtype
        device = node_relevance.device

        starts = torch.arange(time_steps, device=device).view(time_steps, 1)
        ends = torch.arange(time_steps, device=device).view(1, time_steps)
        span_len = ends - starts + 1
        upper = span_len >= 1
        span_len_f = span_len.clamp_min(1).to(dtype)

        # A span is valid only when every token between s and e is valid.
        valid_prefix = F.pad(
            torch.cumsum(video_mask.to(torch.long), dim=-1),
            (1, 0),
            value=0,
        )
        valid_count = (
            valid_prefix[:, 1:].unsqueeze(1)
            - valid_prefix[:, :-1].unsqueeze(2)
        )
        segment_valid = (
            upper.unsqueeze(0)
            & (valid_count == span_len.unsqueeze(0))
        )

        # --------------------------------------------------------------
        # Center node evidence per video to remove a global positive offset.
        # This is not a duration target; it only makes evidence relative to
        # the current video's baseline relevance.
        # --------------------------------------------------------------
        node_baseline = self._masked_mean(
            node_relevance, video_mask, dim=1
        )
        node_centered = (
            node_relevance - node_baseline.unsqueeze(-1)
        ) * video_mask.to(dtype)

        node_prefix = F.pad(
            torch.cumsum(node_centered, dim=-1),
            (1, 0),
            value=0.0,
        )
        node_sum = (
            node_prefix[:, 1:].unsqueeze(1)
            - node_prefix[:, :-1].unsqueeze(2)
        )
        node_support = node_sum / torch.sqrt(
            span_len_f.unsqueeze(0)
        ).clamp_min(1.0)

        if time_steps > 1:
            edge_mask = video_mask[:, :-1] & video_mask[:, 1:]

            # Center relation evidence over valid adjacent pairs.
            edge_baseline = self._masked_mean(
                adjacent_relation, edge_mask, dim=1
            )
            relation_centered = (
                adjacent_relation - edge_baseline.unsqueeze(-1)
            ) * edge_mask.to(dtype)

            edge_prefix = F.pad(
                torch.cumsum(relation_centered, dim=-1),
                (1, 0),
                value=0.0,
            )  # [B,T], prefix[k] = sum of centered edges < k

            edge_sum = (
                edge_prefix.unsqueeze(1)
                - edge_prefix.unsqueeze(2)
            )
            edge_count = (ends - starts).clamp_min(0).to(dtype)

            relation_support = torch.where(
                edge_count.unsqueeze(0) > 0,
                edge_sum
                / torch.sqrt(
                    edge_count.clamp_min(1.0).unsqueeze(0)
                ),
                torch.zeros_like(edge_sum),
            )

            # Boundary separation uses the original bounded relation score:
            # positive = coherent pair, negative = likely boundary.
            left_rel = F.pad(adjacent_relation, (1, 0), value=0.0)
            right_rel = F.pad(adjacent_relation, (0, 1), value=0.0)
            left_valid = F.pad(edge_mask, (1, 0), value=False)
            right_valid = F.pad(edge_mask, (0, 1), value=False)

            left_rel_2d = left_rel.unsqueeze(2)
            right_rel_2d = right_rel.unsqueeze(1)
            left_valid_2d = left_valid.unsqueeze(2)
            right_valid_2d = right_valid.unsqueeze(1)

            boundary_count = (
                left_valid_2d.to(dtype)
                + right_valid_2d.to(dtype)
            )
            boundary_mean = (
                left_rel_2d * left_valid_2d.to(dtype)
                + right_rel_2d * right_valid_2d.to(dtype)
            ) / boundary_count.clamp_min(1.0)

            # Strong positive cohesion at a candidate boundary means the span
            # is cutting through a relation that should probably remain inside.
            boundary_separation = torch.where(
                boundary_count > 0,
                -boundary_mean,
                torch.zeros_like(boundary_mean),
            )
        else:
            edge_mask = video_mask.new_zeros(batch_size, 0)
            relation_centered = adjacent_relation
            relation_support = node_relevance.new_zeros(batch_size, 1, 1)
            boundary_separation = node_relevance.new_zeros(batch_size, 1, 1)

        segment_energy = (
            node_support
            + self.relation_weight * relation_support
            + self.boundary_weight * boundary_separation
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
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        segment_prob = flat_prob.reshape(
            batch_size, time_steps, time_steps
        )

        # Hard segment from exactly the same energy as the soft posterior.
        selected_flat = flat_logits.masked_fill(
            ~flat_valid, -1e4
        ).argmax(dim=-1)
        selected_start = torch.div(
            selected_flat, time_steps, rounding_mode="floor"
        )
        selected_end = selected_flat % time_steps

        selected_energy = flat_logits.gather(
            1, selected_flat.unsqueeze(-1)
        ).squeeze(-1)
        selected_prob = flat_prob.gather(
            1, selected_flat.unsqueeze(-1)
        ).squeeze(-1)

        selected_mask = self._span_mask_from_bounds(
            selected_start, selected_end, video_mask
        )
        hard_temporal_prob = self._mask_to_uniform_weights(
            selected_mask, dtype=dtype
        )

        # --------------------------------------------------------------
        # Soft frame membership from the SAME P(s,e).
        # --------------------------------------------------------------
        index = torch.arange(time_steps, device=device)
        start_grid = index.view(time_steps, 1, 1)
        end_grid = index.view(1, time_steps, 1)
        frame_grid = index.view(1, 1, time_steps)
        inclusion = (
            (start_grid <= frame_grid)
            & (frame_grid <= end_grid)
        ).to(dtype)  # [T,T,T]

        soft_inside_prob = torch.einsum(
            "bij,ijt->bt",
            segment_prob,
            inclusion,
        )
        soft_inside_prob = (
            soft_inside_prob * video_mask.to(dtype)
        )

        # Downstream pooling uses a normalized differentiable evidence weight.
        soft_temporal_prob = soft_inside_prob / soft_inside_prob.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)

        # Exact hard forward value, structured soft backward path.
        training_temporal_prob = (
            hard_temporal_prob
            + soft_temporal_prob
            - soft_temporal_prob.detach()
        )

        valid_token_count = (
            video_mask.sum(dim=-1).to(dtype).clamp_min(1.0)
        )
        selected_length = selected_mask.sum(dim=-1).to(dtype)
        selected_fraction = selected_length / valid_token_count

        denominator = (valid_token_count - 1.0).clamp_min(1.0)
        selected_start_norm = (
            selected_start.to(dtype) / denominator
        ).clamp(0.0, 1.0)
        selected_end_norm = (
            selected_end.to(dtype) / denominator
        ).clamp(0.0, 1.0)

        expected_length = (
            segment_prob * span_len_f.unsqueeze(0)
        ).sum(dim=(-2, -1))
        expected_fraction = expected_length / valid_token_count

        entropy, entropy_ratio = self._entropy_from_distribution(
            segment_prob, segment_valid
        )

        # Shortcut diagnostics.
        diag_mask = torch.eye(
            time_steps, dtype=torch.bool, device=device
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

        selected_single_frame_indicator = (
            selected_length <= 1.0 + 1e-6
        ).to(dtype)
        selected_full_video_indicator = (
            selected_length >= valid_token_count - 1e-6
        ).to(dtype)

        return {
            "node_relevance_centered": node_centered,
            "adjacent_relation_centered": relation_centered,
            "segment_node_support": node_support,
            "segment_relation_support": relation_support,
            "segment_boundary_separation": boundary_separation,
            "segment_energy": segment_energy,
            "segment_logits": segment_logits,
            "segment_valid_mask": segment_valid,
            "segment_prob": segment_prob,
            "segment_entropy": entropy,
            "segment_entropy_ratio": entropy_ratio,
            "segment_top1_prob": selected_prob,
            "soft_inside_prob": soft_inside_prob,
            "soft_temporal_prob": soft_temporal_prob,
            "training_temporal_prob": training_temporal_prob,
            "main_temporal_prob": hard_temporal_prob,
            "main_temporal_logits": torch.log(
                hard_temporal_prob.clamp_min(1e-8)
            ),
            "main_soft_support": selected_mask.to(dtype),
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
            "selected_single_frame_indicator": selected_single_frame_indicator,
            "selected_full_video_indicator": selected_full_video_indicator,
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
            raise ValueError("Q-Bound V2.1 requires at least one timestep.")

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

        content_tokens, adapter_diag = self._adapt_visual_tokens(
            video_tokens, video_mask
        )
        primitives = self._encode_local_primitives(
            content_tokens, video_mask
        )
        question_conditioned = self._inject_question(
            primitive_states=primitives["primitive_states"],
            question_features=question_features,
            question_mask=question_mask,
            video_mask=video_mask,
        )
        scoring = self._score_primitives_and_relations(
            query_states=question_conditioned["query_conditioned_states"],
            question_state=question_conditioned["question_state"],
            delta_magnitude=primitives["delta_magnitude"],
            accel_magnitude=primitives["accel_magnitude"],
            relative_position=primitives["relative_position"],
            video_mask=video_mask,
        )
        segments = self._build_segment_energy(
            node_relevance=scoring["node_relevance"],
            adjacent_relation=scoring["adjacent_relation"],
            video_mask=video_mask,
        )

        dtype = video_tokens.dtype
        device = video_tokens.device

        return {
            # ----------------------------------------------------------
            # Architecture contract.
            # ----------------------------------------------------------
            "qbound_version": self.qbound_version,
            "latent_segment_composition_enabled": torch.ones(
                (), dtype=dtype, device=device
            ),
            "question_token_interaction_enabled": torch.ones(
                (), dtype=dtype, device=device
            ),
            "segment_energy_enabled": torch.ones(
                (), dtype=dtype, device=device
            ),
            "fixed_event_partition_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "dynamic_event_count_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "event_count_cap_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "cut_field_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "boundary_gated_reasoning_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "independent_endpoint_heads_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "hypothesis_competition_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "teacher_distillation_enabled": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "answer_features_used_in_core": torch.zeros(
                (), dtype=dtype, device=device
            ),
            "temporal_gt_used_in_core": torch.zeros(
                (), dtype=dtype, device=device
            ),

            # Masks + generic input diagnostics.
            "video_mask": video_mask,
            "question_mask": question_mask,
            "content_tokens": content_tokens,
            **adapter_diag,

            # Local primitive representation.
            **primitives,

            # Token-level question interaction.
            **question_conditioned,

            # Local query-conditioned evidence.
            **scoring,

            # Shared soft/hard segment model.
            **segments,

            # ----------------------------------------------------------
            # Compatibility aliases used by existing downstream temporal
            # consumers.  These aliases are semantically valid and DO NOT
            # restore V1/V2.0 cut, endpoint, teacher or equilibrium machinery.
            # ----------------------------------------------------------
            "event_temporal_prob": segments["main_temporal_prob"],
            "calibrated_temporal_prob": segments["main_temporal_prob"],
            "calibrated_temporal_logits": segments["main_temporal_logits"],
            "calibrated_soft_support": segments["main_soft_support"],
            "calibrated_span_start_frame": segments[
                "selected_span_start_frame"
            ],
            "calibrated_span_end_frame": segments[
                "selected_span_end_frame"
            ],
            "calibrated_frame_mask": segments[
                "selected_span_frame_mask"
            ],
            "calibrated_span_frame_length": segments[
                "selected_span_length"
            ],
            "calibrated_span_fraction": segments[
                "selected_span_fraction"
            ],
            "calibrated_span_start": segments[
                "selected_span_start"
            ],
            "calibrated_span_end": segments[
                "selected_span_end"
            ],
        }
