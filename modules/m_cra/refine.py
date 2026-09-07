import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_cra.only_trans import (
    clones,
    PositionwiseFeedForward,
    LayerNorm,
    SublayerConnection,
    MultiHeadedAttention,
)
from modules.m_cra.grounding import GroundingModule
from modules.m_cra.causal_module import VisualIntervention
from modules.qbound import QBoundCore

class RefineMHA(MultiHeadedAttention):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__(h, d_model, dropout)
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = self.refined_attention

        self.attn = None
        self.refined_mask = None

    def ori_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def refined_attention(self, query, key, value, mask=None, dropout=None):
        bs, n_head, n_query, d_k = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)

        if mask is not None:
            mask = (
                mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, n_head, n_query, -1)
                .reshape(*attn.shape)
            )
            attn = attn * (mask + 1e-10)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        qkv = torch.matmul(attn, value)
        return qkv

    def causal_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        neg_scores = torch.matmul(query, -key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            neg_scores = neg_scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        neg_p_attn = F.softmax(neg_scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return (
            torch.matmul(p_attn, value) + torch.matmul(neg_p_attn, value),
            (p_attn, neg_p_attn),
        )

    def forward(self, query, key, value, mask=None, alpha=(0.125, 0.5)):
        nbatches = query.size(0)
        query, key, value = [
            linear(tensor)
            .view(nbatches, -1, self.h, self.d_k)
            .transpose(1, 2)
            for linear, tensor in zip(self.linears, (query, key, value))
        ]

        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = RefineMHA(num_heads, embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_self = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None, wm_mode=False):
        x = self.sublayer_self(x, lambda value: self.self_attn(value, value, value, mask))
        x = self.sublayer_ff(x, lambda value: self.feed_forward(value))
        return x




class Decoder(nn.Module):
    """
    CRA refinement decoder with Q-Bound Version 1.6.

    V1.6 follows:

        DECOMPOSE -> RELATE -> COMPOSE -> REFINE -> GROUND

    Relative to V1.5, DECOMPOSE, Q-OEI, and the latent event-span
    Composer are intentionally preserved. V1.6 adds a question-conditioned
    frame-level boundary refiner that trims the first/last selected event
    without changing latent topology.

    Main V1.6 path
    --------------
        raw video + question-only semantics
            -> Q-Bound DECOMPOSE
            -> ordered events
            -> Q-OEI RELATE
            -> latent event-span COMPOSE
            -> frame-level REFINE inside boundary events
            -> refined temporal probability
            -> probability-preserving smoothing
            -> refined start/end/mask (NO [1,3,5] re-selection)
            -> grounded CRA representation
            -> CRA answer scorer

    Important controls
    ------------------
    * CRA temporal grounding is diagnostic-only when Q-Bound is active.
    * No CRA -> Q-Bound handover.
    * No direct reasoned-event residual into answer space.
    * Q-Bound receives pre-CRA-refinement video tokens.
    * Hard topology remains controlled/detached inside QBoundCore.
    * The main V1.6 pre-hoc span comes from frame-level REFINE.
      The coarse V1.5 Composer span and CRA fixed-window extraction are
      retained only as controlled diagnostics.
    * No temporal GT is consumed here.

    Future LLM path
    ---------------
    ``selected_reasoned_event_tokens`` and ``selected_event_mask`` are
    propagated so a later adapter can project selected ordered event tokens
    directly into a frozen LLM/MLLM without redesigning Q-Bound.
    """

    def __init__(
        self,
        embed_dim=768,
        num_layer=2,
        num_heads=8,
        ff_dim=768,
        dropout=0.1,
        cfgs=None,
    ):
        super().__init__()

        if cfgs is None:
            raise ValueError("cfgs must not be None.")

        self.embed_dim = embed_dim
        self.cfgs = cfgs

        self.layers = clones(
            DecoderLayer(
                embed_dim,
                num_heads,
                ff_dim,
                dropout,
            ),
            num_layer,
        )
        self.norm = LayerNorm(embed_dim)

        # Kept for checkpoint compatibility with the original CRA code.
        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        # Preserve original parameter names:
        #   video_proj.0 = Dropout
        #   video_proj.1 = Linear
        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        model_cfg = self.cfgs["model"]
        qbound_cfg = model_cfg.get("qbound", {})

        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        composer_cfg = qbound_cfg.get("composer", {})
        if not isinstance(composer_cfg, dict):
            raise ValueError("model.qbound.composer must be a mapping/dict.")

        refinement_cfg = qbound_cfg.get("refinement", {})
        if not isinstance(refinement_cfg, dict):
            raise ValueError("model.qbound.refinement must be a mapping/dict.")

        self.grounding_module = GroundingModule(
            d_model=embed_dim,
            dropout=model_cfg["dropout"],
            initial_sigma=model_cfg["sigma"],
            window_sizes=model_cfg["window_sizes"],
        )

        self.do = VisualIntervention(self.cfgs)

        # --------------------------------------------------------------
        # Q-Bound V1.6
        # --------------------------------------------------------------
        self.qbound_enabled = bool(
            qbound_cfg.get(
                "enabled",
                model_cfg.get(
                    "qbound_enabled",
                    model_cfg.get("gpeb_enabled", False),
                ),
            )
        )

        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get(
                    "qbound_require_question_only",
                    model_cfg.get("gpeb_require_question_only", True),
                ),
            )
        )

        self.qbound_hidden_dim = None
        self.qbound_detach_backbone_inputs = True

        if self.qbound_enabled:
            self.qbound_hidden_dim = int(
                qbound_cfg.get(
                    "hidden_dim",
                    model_cfg.get("qbound_hidden_dim", 256),
                )
            )

            self.qbound = QBoundCore(
                input_dim=embed_dim,
                hidden_dim=self.qbound_hidden_dim,
                max_events=int(
                    qbound_cfg.get(
                        "max_events",
                        model_cfg.get("qbound_max_events", 4),
                    )
                ),
                use_temporal_prior=bool(
                    qbound_cfg.get("use_temporal_prior", False)
                ),
                dropout=float(
                    qbound_cfg.get(
                        "dropout",
                        model_cfg.get("qbound_dropout", dropout),
                    )
                ),
                boundary_temperature=float(
                    qbound_cfg.get(
                        "boundary_temperature",
                        model_cfg.get("qbound_boundary_temperature", 0.7),
                    )
                ),
                boundary_threshold=float(
                    qbound_cfg.get(
                        "boundary_threshold",
                        model_cfg.get("qbound_boundary_threshold", 0.5),
                    )
                ),
                boundary_radius=int(
                    qbound_cfg.get(
                        "boundary_radius",
                        model_cfg.get("qbound_boundary_radius", 2),
                    )
                ),
                boundary_bias_init=float(
                    qbound_cfg.get(
                        "boundary_bias_init",
                        model_cfg.get("qbound_boundary_bias_init", -0.5),
                    )
                ),
                proposal_candidates=int(
                    qbound_cfg.get("proposal_candidates", 6)
                ),
                decomp_complexity_weight=float(
                    qbound_cfg.get(
                        "decomp_complexity_weight",
                        qbound_cfg.get(
                            "decomposition",
                            {},
                        ).get("complexity_weight", 1.25),
                    )
                ),
                decomp_boundary_weight=float(
                    qbound_cfg.get(
                        "decomp_boundary_weight",
                        qbound_cfg.get(
                            "decomposition",
                            {},
                        ).get("boundary_weight", 0.50),
                    )
                ),
                decomp_temperature=float(
                    qbound_cfg.get(
                        "decomp_temperature",
                        qbound_cfg.get(
                            "decomposition",
                            {},
                        ).get("temperature", 0.20),
                    )
                ),
                question_modulation_scale=float(
                    qbound_cfg.get(
                        "question_modulation_scale",
                        0.10,
                    )
                ),
                min_event_mass=float(
                    qbound_cfg.get(
                        "min_event_mass",
                        model_cfg.get("qbound_min_event_mass", 0.25),
                    )
                ),
                composer_temperature=float(
                    qbound_cfg.get(
                        "composer_temperature",
                        composer_cfg.get("temperature", 0.35),
                    )
                ),
                refiner_temperature=float(
                    qbound_cfg.get(
                        "refiner_temperature",
                        refinement_cfg.get("temperature", 0.30),
                    )
                ),
            )

            # Controlled first V1.6 experiment: Q-Bound learns its own
            # parameters without changing the upstream video/language
            # representation through the Q-Bound input path.
            self.qbound_detach_backbone_inputs = bool(
                qbound_cfg.get(
                    "detach_backbone_inputs",
                    model_cfg.get(
                        "qbound_detach_backbone_inputs",
                        model_cfg.get(
                            "gpeb_detach_backbone_inputs",
                            True,
                        ),
                    ),
                )
            )

    # ------------------------------------------------------------------
    # Shape / normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_video_mask(
        mask,
        batch_size,
        time_steps,
        device,
    ):
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
                "self_mask must reduce to [B,T], got "
                f"{tuple(mask.shape)}."
            )

        if mask.shape != (batch_size, time_steps):
            raise ValueError(
                "self_mask shape mismatch. Expected "
                f"{(batch_size, time_steps)}, got {tuple(mask.shape)}."
            )

        return mask.to(device=device).bool()

    @staticmethod
    def _normalize_temporal_distribution(
        values,
        valid_mask,
        eps=1e-6,
    ):
        if values.ndim == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)

        if values.shape != valid_mask.shape:
            raise ValueError(
                "Temporal distribution and valid mask must have "
                "the same [B,T] shape. Got "
                f"{tuple(values.shape)} and {tuple(valid_mask.shape)}."
            )

        values = values.clamp_min(0.0)
        values = values * valid_mask.to(values.dtype)

        denominator = values.sum(dim=-1, keepdim=True)
        normalized = values / denominator.clamp_min(eps)

        fallback = valid_mask.to(values.dtype)
        fallback = (
            fallback
            / fallback.sum(dim=-1, keepdim=True).clamp_min(1.0)
        )

        return torch.where(
            denominator > eps,
            normalized,
            fallback,
        )

    @staticmethod
    def _masked_mean_video(
        features,
        valid_mask,
    ):
        weights = valid_mask.to(features.dtype).unsqueeze(-1)
        return (
            (features * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )

    # ------------------------------------------------------------------
    # Shared CRA projection helpers
    # ------------------------------------------------------------------
    def _sample_projection_dropout_mask(
        self,
        reference,
    ):
        """
        Sample one dropout realization and reuse it for CRA diagnostic and
        Q-Bound main paths. This keeps answer-space comparisons fair.
        """
        dropout_layer = self.video_proj[0]

        if not self.training or dropout_layer.p <= 0.0:
            return torch.ones_like(reference)

        keep_prob = 1.0 - float(dropout_layer.p)

        if keep_prob <= 0.0:
            raise ValueError("video projection dropout must be < 1.")

        mask = torch.empty_like(reference).bernoulli_(keep_prob)
        return mask / keep_prob

    def _apply_video_projection(
        self,
        pooled,
        dropout_mask,
    ):
        if pooled.shape != dropout_mask.shape:
            raise ValueError(
                "pooled representation and dropout mask must have "
                "the same shape. Got "
                f"{tuple(pooled.shape)} and {tuple(dropout_mask.shape)}."
            )

        return self.video_proj[1](pooled * dropout_mask)

    def _project_grounded_video(
        self,
        x_bar,
        temporal_weights,
        valid_mask,
        do,
        projection_dropout_mask,
        x_star_proj=None,
    ):
        del valid_mask

        pooled = torch.sum(
            x_bar * temporal_weights.unsqueeze(-1),
            dim=1,
        )

        projected = self._apply_video_projection(
            pooled,
            projection_dropout_mask,
        )

        if do:
            if x_star_proj is None:
                raise ValueError(
                    "x_star_proj is required when visual intervention is enabled."
                )

            projected = self.do(
                x_star_proj,
                projected,
            )

        return projected

    def _build_time_param(
        self,
        final_key,
        base_temporal,
        qbound_key,
        qbound_active,
        window_sizes,
        qbound_time=None,
    ):
        """
        Build the final temporal contract.

        V1.6 active:
            use Q-Bound REFINE frame-level interval directly.

        Q-Bound inactive:
            preserve the original CRA fixed-window interval extraction.
        """
        if qbound_active:
            if qbound_time is None:
                raise ValueError(
                    "qbound_time is required for active Q-Bound V1.6."
                )

            # Important: qbound_time["start"/"end"/"mask"] are the
            # frame-refined V1.6 interval, NOT the CRA [1,3,5] search and
            # NOT the coarse V1.5 event-chain boundaries.
            time_param = dict(qbound_time)
            time_param["key"] = final_key
        else:
            time_param = self.grounding_module.gen_grounding(
                final_key,
                window_sizes=window_sizes,
            )

        time_param["ori_key"] = base_temporal["ori_key"]
        time_param["cra_key"] = base_temporal["cra_key"]
        time_param["qbound_key"] = qbound_key
        time_param["refined_key"] = final_key

        # Compatibility field only: V1.6 has no temporal handover.
        time_param["qbound_mix"] = torch.full(
            (final_key.size(0),),
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )

        time_param["qbound_handover_enabled"] = torch.as_tensor(
            0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )

        # Compatibility alias: in V1.6 "adaptive" refers to the final
        # refined interval. Keep it so older pipeline code does not break.
        time_param["qbound_adaptive_span_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )

        time_param["qbound_refined_span_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )

        return time_param

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x,
        qa,
        self_mask=None,
        gsub=None,
        do=False,
        window_sizes=(1, 3, 5),
        use_qbound=False,
        question_feature=None,
        qbound_mix=1.0,
        # Temporary compatibility alias for older Decoder callers.
        use_gpeb=None,
    ):
        del gsub
        del qbound_mix  # V1.6 intentionally has no CRA->Q-Bound handover.

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # Raw pre-CRA-refinement video tokens are the generic Q-Bound input.
        x_raw = x

        # CRA refinement remains the downstream testbed representation.
        for layer in self.layers:
            x = layer(
                x,
                self_mask,
            )

        x_bar = self.norm(x)

        batch_size, time_steps, _ = x_bar.shape

        valid_mask = self._normalize_video_mask(
            self_mask,
            batch_size,
            time_steps,
            x_bar.device,
        )

        qbound_active = bool(
            self.qbound_enabled and use_qbound
        )

        # --------------------------------------------------------------
        # 1) CRA temporal grounding: detached diagnostic/reference only
        #    when Q-Bound is active.
        # --------------------------------------------------------------
        if qbound_active:
            with torch.no_grad():
                base_temporal = self.grounding_module.compute_base_temporal(
                    x_bar,
                    qa,
                    valid_mask=valid_mask,
                )
        else:
            base_temporal = self.grounding_module.compute_base_temporal(
                x_bar,
                qa,
                valid_mask=valid_mask,
            )

        cra_key = self._normalize_temporal_distribution(
            base_temporal["cra_key"],
            valid_mask,
        )

        if qbound_active:
            cra_key = cra_key.detach()

        # --------------------------------------------------------------
        # 2) Shared CRA answer-projection context.
        # --------------------------------------------------------------
        projection_reference = self._masked_mean_video(
            x_bar,
            valid_mask,
        )

        projection_dropout_mask = self._sample_projection_dropout_mask(
            projection_reference
        )

        x_star_proj = None

        if do:
            x_star_dropout_mask = self._sample_projection_dropout_mask(
                projection_reference
            )
            x_star_proj = self._apply_video_projection(
                projection_reference,
                x_star_dropout_mask,
            )

        cra_video_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=cra_key,
            valid_mask=valid_mask,
            do=do,
            projection_dropout_mask=projection_dropout_mask,
            x_star_proj=x_star_proj,
        )

        # --------------------------------------------------------------
        # 3) Q-Bound V1.6:
        #       DECOMPOSE -> RELATE -> COMPOSE -> REFINE -> GROUND
        # --------------------------------------------------------------
        qbound_aux = None
        qbound_time = None
        legacy_v15_time = None
        legacy_v14_time = None

        if qbound_active:
            if (
                question_feature is None
                and self.qbound_require_question_only
            ):
                raise ValueError(
                    "Q-Bound V1.6 requires question_feature=[B,D]. "
                    "Pass question-only features through CRA.forward(...)."
                )

            qbound_language = (
                question_feature
                if question_feature is not None
                else qa
            )

            if self.qbound_detach_backbone_inputs:
                video_for_qbound = x_raw.detach()
                language_for_qbound = qbound_language.detach()
            else:
                video_for_qbound = x_raw
                language_for_qbound = qbound_language

            qbound_aux = self.qbound(
                video_tokens=video_for_qbound,
                question_features=language_for_qbound,
                video_mask=valid_mask,
                contextual_tokens=video_for_qbound,
                temporal_prior=None,
            )

            required_keys = {
                # V1.6 main temporal output + controlled legacy outputs.
                "event_temporal_prob",
                "main_temporal_prob",
                "refined_temporal_prob",
                "refined_temporal_logits",
                "legacy_v15_composed_temporal_prob",
                "composed_temporal_prob",
                "composed_temporal_logits",
                "legacy_v14_event_temporal_prob",

                # Dynamic latent decomposition.
                "boundary_hard",
                "boundary_count",
                "event_count",
                "event_assignment",
                "event_mass",
                "event_mass_max",
                "event_mass_min",
                "soft_effective_event_count",
                "decomp_loss",
                "decomp_selected_coherence",
                "decomp_selected_complexity",
                "decomp_selected_total_energy",
                "decomp_expected_event_count",
                "partition_entropy",
                "accepted_split_gain",
                "rejected_split_gain",

                # Event/evidence outputs.
                "event_visual_tokens",
                "evidence_visual_token",
                "event_valid_mask",
                "event_evidence_logits",
                "event_evidence_weights",
                "event_evidence_effective_count",
                "top1_event_evidence_share",
                "top2_event_evidence_share",
                "multi_event_evidence",
                "frame_semantic_logits",

                # Ordered reasoning.
                "event_tokens",
                "reasoned_event_tokens",
                "reasoned_event_repr",
                "ordered_relation_gates",
                "ordered_relation_valid_mask",
                "ordered_update_gate",
                "reasoned_event_token_delta_norm",

                # V1.5 COMPOSE.
                "span_candidate_event_mask",
                "span_candidate_valid_mask",
                "span_candidate_frame_mask",
                "span_candidate_frame_length",
                "span_logits",
                "span_posterior",
                "span_entropy",
                "span_effective_count",
                "span_top1_posterior",
                "soft_span_support",
                "selected_span_index",
                "selected_event_mask",
                "selected_reasoned_event_tokens",
                "selected_frame_mask",
                "selected_span_start_event",
                "selected_span_end_event",
                "selected_span_event_count",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_length",
                "selected_span_fraction",
                "selected_span_posterior",
                "soft_selected_event_weights",
                "composed_reasoned_event_repr",

                # V1.6 REFINE.
                "refiner_enabled",
                "refine_start_logits",
                "refine_end_logits",
                "refine_start_candidate_mask",
                "refine_end_candidate_mask",
                "refine_conditional_support",
                "refined_soft_support",
                "refined_support_log_bias",
                "refine_expected_pair_entropy",
                "refine_selected_pair_posterior",
                "refine_selected_pair_entropy",
                "refine_selected_pair_effective_count",
                "refine_selected_pair_top1_posterior",
                "refine_selected_start_candidate_count",
                "refine_selected_end_candidate_count",
                "coarse_span_start_frame",
                "coarse_span_end_frame",
                "coarse_span_frame_length",
                "coarse_span_fraction",
                "coarse_frame_mask",
                "refined_span_start_frame",
                "refined_span_end_frame",
                "refined_frame_mask",
                "refined_span_frame_length",
                "refined_span_fraction",
                "refined_span_start",
                "refined_span_end",
                "refined_span_temporal_prob",
                "refine_trim_left_frames",
                "refine_trim_right_frames",
                "refine_trim_total_frames",
                "refine_retained_fraction",
                "refined_inside_coarse",
                "refiner_temperature",
            }

            missing = sorted(required_keys.difference(qbound_aux))

            if missing:
                raise KeyError(
                    "Q-Bound V1.6 output is missing: "
                    + ", ".join(missing)
                )

            # ----------------------------------------------------------
            # V1.6 MAIN grounding adapter.
            #
            # The temporal key comes from the differentiable refined support.
            # Final pre-hoc start/end/mask come from the frame-level boundary
            # refiner, while the coarse V1.5 span is passed only for controlled
            # diagnostics on the SAME V1.6 key.
            # ----------------------------------------------------------
            qbound_time = (
                self.grounding_module
                .finalize_refined_grounding(
                    temporal_probability=qbound_aux[
                        "refined_temporal_prob"
                    ],
                    refined_span_start_frame=qbound_aux[
                        "refined_span_start_frame"
                    ],
                    refined_span_end_frame=qbound_aux[
                        "refined_span_end_frame"
                    ],
                    valid_mask=valid_mask,
                    refined_frame_mask=qbound_aux[
                        "refined_frame_mask"
                    ],
                    coarse_span_start_frame=qbound_aux[
                        "coarse_span_start_frame"
                    ],
                    coarse_span_end_frame=qbound_aux[
                        "coarse_span_end_frame"
                    ],
                    coarse_frame_mask=qbound_aux[
                        "coarse_frame_mask"
                    ],
                    window_sizes=window_sizes,
                    detach_sigma=True,
                )
            )

            qbound_key = self._normalize_temporal_distribution(
                qbound_time["key"],
                valid_mask,
            )

            # Controlled full V1.5 diagnostic:
            # V1.5 composed probability + coarse event-chain interval.
            # Detached so it cannot influence V1.6 optimization.
            with torch.no_grad():
                legacy_v15_time = (
                    self.grounding_module
                    .finalize_composed_grounding(
                        temporal_probability=qbound_aux[
                            "legacy_v15_composed_temporal_prob"
                        ].detach(),
                        selected_span_start_frame=qbound_aux[
                            "selected_span_start_frame"
                        ],
                        selected_span_end_frame=qbound_aux[
                            "selected_span_end_frame"
                        ],
                        valid_mask=valid_mask,
                        selected_frame_mask=qbound_aux[
                            "selected_frame_mask"
                        ],
                        window_sizes=window_sizes,
                        detach_sigma=True,
                    )
                )

                # Controlled V1.4 diagnostic:
                # V1.4 local probability + old CRA fixed-window readout.
                legacy_v14_time = (
                    self.grounding_module
                    .finalize_probability_grounding(
                        qbound_aux[
                            "legacy_v14_event_temporal_prob"
                        ].detach(),
                        valid_mask=valid_mask,
                        window_sizes=window_sizes,
                        detach_sigma=True,
                    )
                )

            final_key = qbound_key

            qbound_video_proj = self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=qbound_key,
                valid_mask=valid_mask,
                do=do,
                projection_dropout_mask=projection_dropout_mask,
                x_star_proj=x_star_proj,
            )

            # No direct event-token residual into the answer representation.
            video_proj = qbound_video_proj

        else:
            qbound_key = cra_key
            final_key = cra_key
            qbound_video_proj = cra_video_proj
            video_proj = cra_video_proj

        # --------------------------------------------------------------
        # 4) Final temporal contract.
        #
        # Active V1.6 uses frame-refined start/end/mask.
        # Inactive Q-Bound preserves original CRA extraction.
        # --------------------------------------------------------------
        time_param = self._build_time_param(
            final_key=final_key,
            base_temporal=base_temporal,
            qbound_key=qbound_key,
            qbound_active=qbound_active,
            window_sizes=window_sizes,
            qbound_time=qbound_time,
        )

        # Main VQA feature follows the actually-used temporal key.
        time_param["grounding_feature"] = video_proj
        time_param["qbound_feature"] = video_proj
        time_param["qbound_grounding_feature"] = qbound_video_proj
        time_param["qbound_temporal_main_feature"] = qbound_video_proj
        time_param["cra_grounding_feature"] = cra_video_proj

        # --------------------------------------------------------------
        # Explicit V1.3 shortcut removal remains unchanged.
        # --------------------------------------------------------------
        zero_residual = torch.zeros_like(video_proj)
        time_param["qbound_reasoning_residual"] = zero_residual
        time_param["qbound_reasoning_residual_used"] = zero_residual
        time_param["qbound_reasoning_scale"] = torch.as_tensor(
            0.0,
            dtype=video_proj.dtype,
            device=video_proj.device,
        )
        time_param["qbound_direct_reasoning_residual_enabled"] = (
            torch.as_tensor(
                0.0,
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
        )

        # Compatibility alias: reasoned event tokens influence this feature
        # only through COMPOSE -> REFINE -> temporal evidence -> grounded pooling.
        time_param["qbound_reasoned_video_feature"] = qbound_video_proj

        if qbound_aux is not None:
            # ----------------------------------------------------------
            # 5) Generic event / weak-alignment outputs.
            # ----------------------------------------------------------
            time_param["qbound_event_visual_tokens"] = (
                qbound_aux["event_visual_tokens"]
            )
            time_param["qbound_evidence_visual_token"] = (
                qbound_aux["evidence_visual_token"]
            )
            time_param["qbound_event_valid_mask"] = (
                qbound_aux["event_valid_mask"]
            )
            time_param["qbound_frame_semantic_logits"] = (
                qbound_aux["frame_semantic_logits"]
            )

            time_param["qbound_event_evidence_logits"] = (
                qbound_aux["event_evidence_logits"]
            )
            time_param["qbound_event_evidence_weights"] = (
                qbound_aux["event_evidence_weights"]
            )
            time_param["qbound_event_evidence_effective_count"] = (
                qbound_aux["event_evidence_effective_count"]
            )
            time_param["qbound_multi_event_evidence"] = (
                qbound_aux["multi_event_evidence"]
            )
            time_param["qbound_top1_event_evidence_share"] = (
                qbound_aux["top1_event_evidence_share"]
            )
            time_param["qbound_top2_event_evidence_share"] = (
                qbound_aux["top2_event_evidence_share"]
            )

            # Compatibility aliases for old loggers; diagnostic only.
            time_param["qbound_event_relevance_weights"] = (
                qbound_aux["event_relevance_weights"]
            )
            time_param["qbound_event_relevance_gates"] = (
                qbound_aux["event_relevance_gates"]
            )
            time_param["qbound_event_relevance_effective_count"] = (
                qbound_aux["event_relevance_effective_count"]
            )

            # ----------------------------------------------------------
            # 6) RELATE / Q-OEI outputs.
            # ----------------------------------------------------------
            time_param["qbound_event_tokens"] = (
                qbound_aux["event_tokens"]
            )
            time_param["qbound_reasoned_event_tokens"] = (
                qbound_aux["reasoned_event_tokens"]
            )
            time_param["qbound_reasoned_event_repr"] = (
                qbound_aux["reasoned_event_repr"]
            )
            time_param["qbound_ordered_relation_gates"] = (
                qbound_aux["ordered_relation_gates"]
            )
            time_param["qbound_ordered_relation_valid_mask"] = (
                qbound_aux["ordered_relation_valid_mask"]
            )
            time_param["qbound_ordered_update_gate"] = (
                qbound_aux["ordered_update_gate"]
            )
            time_param["qbound_reasoned_event_token_delta_norm"] = (
                qbound_aux["reasoned_event_token_delta_norm"]
            )

            # ----------------------------------------------------------
            # 7) DECOMPOSE outputs.
            # ----------------------------------------------------------
            time_param["qbound_event_assignment"] = (
                qbound_aux["event_assignment"]
            )
            time_param["qbound_event_mass"] = qbound_aux["event_mass"]
            time_param["qbound_event_mass_max"] = (
                qbound_aux["event_mass_max"]
            )
            time_param["qbound_event_mass_min"] = (
                qbound_aux["event_mass_min"]
            )
            time_param["qbound_event_mass_mean"] = (
                qbound_aux["event_mass_mean"]
            )
            time_param["qbound_effective_event_count"] = (
                qbound_aux["effective_event_count"]
            )
            time_param["qbound_soft_effective_event_count"] = (
                qbound_aux["soft_effective_event_count"]
            )

            time_param["qbound_decomp_loss"] = qbound_aux["decomp_loss"]
            time_param["qbound_decomp_expected_energy"] = (
                qbound_aux["decomp_expected_energy"]
            )
            time_param["qbound_decomp_selected_coherence"] = (
                qbound_aux["decomp_selected_coherence"]
            )
            time_param["qbound_decomp_selected_complexity"] = (
                qbound_aux["decomp_selected_complexity"]
            )
            time_param["qbound_decomp_selected_complexity_cost"] = (
                qbound_aux["decomp_selected_complexity_cost"]
            )
            time_param["qbound_decomp_selected_boundary_reward"] = (
                qbound_aux["decomp_selected_boundary_reward"]
            )
            time_param["qbound_decomp_selected_boundary_term"] = (
                qbound_aux["decomp_selected_boundary_term"]
            )
            time_param["qbound_decomp_selected_total_energy"] = (
                qbound_aux["decomp_selected_total_energy"]
            )
            time_param["qbound_decomp_expected_event_count"] = (
                qbound_aux["decomp_expected_event_count"]
            )
            time_param["qbound_partition_entropy"] = (
                qbound_aux["partition_entropy"]
            )
            time_param["qbound_candidate_split_gain"] = (
                qbound_aux["candidate_split_gain"]
            )
            time_param["qbound_accepted_split_gain"] = (
                qbound_aux["accepted_split_gain"]
            )
            time_param["qbound_rejected_split_gain"] = (
                qbound_aux["rejected_split_gain"]
            )

            time_param["qbound_boundary_prob"] = (
                qbound_aux["boundary_prob"]
            )
            time_param["qbound_boundary_hard"] = (
                qbound_aux["boundary_hard"]
            )
            time_param["qbound_boundary_soft_marginal"] = (
                qbound_aux["boundary_soft_marginal"]
            )
            time_param["qbound_boundary_candidate_indices"] = (
                qbound_aux["boundary_candidate_indices"]
            )
            time_param["qbound_boundary_candidate_probs"] = (
                qbound_aux["boundary_candidate_probs"]
            )
            time_param["qbound_boundary_candidate_valid_mask"] = (
                qbound_aux["boundary_candidate_valid_mask"]
            )

            # ----------------------------------------------------------
            # 8) V1.5 COMPOSE outputs.
            # ----------------------------------------------------------
            composer_keys = [
                "span_candidate_event_mask",
                "span_candidate_start_event",
                "span_candidate_end_event",
                "span_candidate_event_count",
                "span_candidate_valid_mask",
                "span_candidate_frame_mask",
                "span_candidate_frame_length",
                "span_candidate_center",
                "span_candidate_duration",
                "span_event_evidence_mean",
                "span_frame_evidence_mean",
                "span_learned_logits",
                "span_evidence_prior",
                "span_logits",
                "span_posterior",
                "span_entropy",
                "span_effective_count",
                "span_top1_posterior",
                "soft_selected_event_weights",
                "composed_reasoned_event_repr",
                "soft_span_support",
                "span_support_log_bias",
                "composed_temporal_logits",
                "composed_temporal_prob",
                "main_temporal_logits",
                "composer_enabled",
                "span_candidate_count",
                "selected_span_index",
                "selected_event_mask",
                "selected_reasoned_event_tokens",
                "selected_frame_mask",
                "selected_span_start_event",
                "selected_span_end_event",
                "selected_span_event_count",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_length",
                "selected_span_fraction",
                "selected_span_posterior",
                "selected_span_temporal_prob",
                "selected_span_start",
                "selected_span_end",
                "legacy_v14_event_temporal_prob",
            ]

            for key in composer_keys:
                if key in qbound_aux:
                    time_param[f"qbound_{key}"] = qbound_aux[key]

            # ----------------------------------------------------------
            # 9) V1.6 REFINE outputs.
            # ----------------------------------------------------------
            refiner_keys = [
                "refiner_enabled",
                "refine_start_logits",
                "refine_end_logits",
                "refine_start_candidate_mask",
                "refine_end_candidate_mask",
                "refine_conditional_support",
                "refined_soft_support",
                "refined_support_log_bias",
                "refined_temporal_logits",
                "refined_temporal_prob",
                "main_temporal_logits",
                "main_temporal_prob",
                "refine_expected_pair_entropy",
                "refine_selected_pair_posterior",
                "refine_selected_pair_entropy",
                "refine_selected_pair_effective_count",
                "refine_selected_pair_top1_posterior",
                "refine_selected_start_candidate_count",
                "refine_selected_end_candidate_count",
                "coarse_span_start_frame",
                "coarse_span_end_frame",
                "coarse_span_frame_length",
                "coarse_span_fraction",
                "coarse_frame_mask",
                "refined_span_start_frame",
                "refined_span_end_frame",
                "refined_frame_mask",
                "refined_span_frame_length",
                "refined_span_fraction",
                "refined_span_start",
                "refined_span_end",
                "refined_span_temporal_prob",
                "refine_trim_left_frames",
                "refine_trim_right_frames",
                "refine_trim_total_frames",
                "refine_retained_fraction",
                "refined_inside_coarse",
                "refiner_temperature",
                "legacy_v15_composed_temporal_prob",
                "legacy_v15_composed_temporal_logits",
            ]

            for key in refiner_keys:
                if key in qbound_aux:
                    time_param[f"qbound_{key}"] = qbound_aux[key]

            # Grounding adapter diagnostics from the SAME V1.6 refined key.
            grounding_diag_keys = [
                # Compatibility aliases for the MAIN refined interval.
                "adaptive_span_used",
                "adaptive_span_start_frame",
                "adaptive_span_end_frame",
                "adaptive_span_length",
                "adaptive_span_fraction",
                "adaptive_span_key_mass",
                "adaptive_span_peak_in_span",
                "adaptive_span_composer_mask",
                "adaptive_span_mask_consistency",

                # Explicit V1.6 refined interval.
                "refined_span_used",
                "refined_span_start_frame",
                "refined_span_end_frame",
                "refined_span_length",
                "refined_span_fraction",
                "refined_span_key_mass",
                "refined_span_peak_in_span",
                "refined_span_core_mask",
                "refined_span_mask_consistency",

                # Coarse V1.5 span evaluated on the SAME V1.6 key.
                "coarse_v15_start",
                "coarse_v15_end",
                "coarse_v15_mask",
                "coarse_v15_max_indices",
                "coarse_v15_span_start_frame",
                "coarse_v15_span_end_frame",
                "coarse_v15_span_length",
                "coarse_v15_span_fraction",
                "coarse_v15_span_key_mass",
                "coarse_v15_span_peak_in_span",
                "coarse_v15_span_mask_consistency",

                # Fixed-window diagnostic on the SAME V1.6 key.
                "legacy_fixed_window_max_indices",
                "legacy_fixed_window_start",
                "legacy_fixed_window_end",
                "legacy_fixed_window_mask",
                "legacy_fixed_window_span_length",
                "legacy_fixed_window_key_mass",

                # Direct comparison diagnostics.
                "refined_vs_coarse_span_length_ratio",
                "refined_vs_legacy_span_length_ratio",
                "coarse_vs_legacy_span_length_ratio",
                "refined_minus_coarse_key_mass",
                "refined_minus_legacy_key_mass",
                "refined_retained_fraction_of_coarse",
                "refined_trimmed_frame_count",
            ]

            for key in grounding_diag_keys:
                if key in qbound_time:
                    time_param[f"qbound_{key}"] = qbound_time[key]

            # Separate full V1.5 readout diagnostic:
            # V1.5 composed distribution + V1.5 coarse Composer interval.
            if legacy_v15_time is not None:
                time_param["qbound_v15_diagnostic_key"] = (
                    legacy_v15_time["key"]
                )
                time_param["qbound_v15_diagnostic_start"] = (
                    legacy_v15_time["start"]
                )
                time_param["qbound_v15_diagnostic_end"] = (
                    legacy_v15_time["end"]
                )
                time_param["qbound_v15_diagnostic_mask"] = (
                    legacy_v15_time["mask"]
                )
                time_param["qbound_v15_diagnostic_span_length"] = (
                    legacy_v15_time["mask"]
                    .sum(dim=-1)
                    .to(qbound_key.dtype)
                )

            # Separate full V1.4 readout diagnostic:
            # V1.4 temporal distribution + V1.4 fixed-window extraction.
            if legacy_v14_time is not None:
                time_param["qbound_v14_diagnostic_key"] = (
                    legacy_v14_time["key"]
                )
                time_param["qbound_v14_diagnostic_start"] = (
                    legacy_v14_time["start"]
                )
                time_param["qbound_v14_diagnostic_end"] = (
                    legacy_v14_time["end"]
                )
                time_param["qbound_v14_diagnostic_mask"] = (
                    legacy_v14_time["mask"]
                )
                time_param["qbound_v14_diagnostic_span_length"] = (
                    legacy_v14_time["mask"]
                    .sum(dim=-1)
                    .to(qbound_key.dtype)
                )

            # ----------------------------------------------------------
            # 9) Populate qbound_aux with decoder/adaptor diagnostics.
            # ----------------------------------------------------------
            qbound_aux["cra_key"] = cra_key
            qbound_aux["event_key"] = qbound_key
            qbound_aux["qbound_key"] = qbound_key
            qbound_aux["final_key"] = final_key
            qbound_aux["refined_key"] = final_key
            qbound_aux["qbound_mix"] = torch.ones(
                batch_size,
                dtype=qbound_key.dtype,
                device=qbound_key.device,
            )
            qbound_aux["qbound_handover_enabled"] = torch.as_tensor(
                0.0,
                dtype=qbound_key.dtype,
                device=qbound_key.device,
            )
            # Compatibility + explicit V1.6 flags.
            qbound_aux["qbound_adaptive_span_enabled"] = torch.as_tensor(
                1.0,
                dtype=qbound_key.dtype,
                device=qbound_key.device,
            )
            qbound_aux["qbound_refined_span_enabled"] = torch.as_tensor(
                1.0,
                dtype=qbound_key.dtype,
                device=qbound_key.device,
            )

            qbound_aux["cra_video_proj"] = cra_video_proj
            qbound_aux["qbound_video_proj"] = qbound_video_proj
            qbound_aux["temporal_main_video_proj"] = qbound_video_proj
            qbound_aux["main_video_proj"] = video_proj

            qbound_aux["qbound_reasoning_residual"] = zero_residual
            qbound_aux["qbound_reasoning_residual_used"] = zero_residual
            qbound_aux["qbound_reasoning_scale"] = torch.as_tensor(
                0.0,
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
            qbound_aux["qbound_direct_reasoning_residual_enabled"] = (
                torch.as_tensor(
                    0.0,
                    dtype=video_proj.dtype,
                    device=video_proj.device,
                )
            )

            # Main V1.6 pre/post smoothing on refined support.
            qbound_aux["qbound_pre_smooth_key"] = (
                qbound_time["pre_smooth_key"]
            )
            qbound_aux["qbound_post_smooth_key"] = qbound_key

            # Main V1.6 refined frame-level interval (compatibility adaptive aliases).
            qbound_aux["qbound_adaptive_start"] = qbound_time["start"]
            qbound_aux["qbound_adaptive_end"] = qbound_time["end"]
            qbound_aux["qbound_adaptive_mask"] = qbound_time["mask"]
            qbound_aux["qbound_adaptive_span_length"] = qbound_time[
                "adaptive_span_length"
            ]
            qbound_aux["qbound_adaptive_span_fraction"] = qbound_time[
                "adaptive_span_fraction"
            ]
            qbound_aux["qbound_adaptive_span_key_mass"] = qbound_time[
                "adaptive_span_key_mass"
            ]

            # Explicit V1.6 refined interval.
            qbound_aux["qbound_refined_start"] = qbound_time["start"]
            qbound_aux["qbound_refined_end"] = qbound_time["end"]
            qbound_aux["qbound_refined_mask"] = qbound_time["mask"]
            qbound_aux["qbound_refined_span_length"] = qbound_time[
                "refined_span_length"
            ]
            qbound_aux["qbound_refined_span_fraction"] = qbound_time[
                "refined_span_fraction"
            ]
            qbound_aux["qbound_refined_span_key_mass"] = qbound_time[
                "refined_span_key_mass"
            ]

            # Coarse V1.5 interval evaluated on the SAME V1.6 temporal key.
            qbound_aux["qbound_coarse_v15_start"] = qbound_time[
                "coarse_v15_start"
            ]
            qbound_aux["qbound_coarse_v15_end"] = qbound_time[
                "coarse_v15_end"
            ]
            qbound_aux["qbound_coarse_v15_mask"] = qbound_time[
                "coarse_v15_mask"
            ]
            qbound_aux["qbound_coarse_v15_span_length"] = qbound_time[
                "coarse_v15_span_length"
            ]
            qbound_aux["qbound_coarse_v15_span_fraction"] = qbound_time[
                "coarse_v15_span_fraction"
            ]
            qbound_aux["qbound_coarse_v15_span_key_mass"] = qbound_time[
                "coarse_v15_span_key_mass"
            ]

            # Same V1.6 key, legacy short-window interval diagnostic.
            qbound_aux["qbound_legacy_fixed_window_start"] = qbound_time[
                "legacy_fixed_window_start"
            ]
            qbound_aux["qbound_legacy_fixed_window_end"] = qbound_time[
                "legacy_fixed_window_end"
            ]
            qbound_aux["qbound_legacy_fixed_window_mask"] = qbound_time[
                "legacy_fixed_window_mask"
            ]
            qbound_aux["qbound_legacy_fixed_window_span_length"] = qbound_time[
                "legacy_fixed_window_span_length"
            ]

            qbound_aux["qbound_refined_vs_coarse_span_length_ratio"] = (
                qbound_time["refined_vs_coarse_span_length_ratio"]
            )
            qbound_aux["qbound_refined_vs_legacy_span_length_ratio"] = (
                qbound_time["refined_vs_legacy_span_length_ratio"]
            )
            qbound_aux["qbound_refined_minus_coarse_key_mass"] = qbound_time[
                "refined_minus_coarse_key_mass"
            ]
            qbound_aux["qbound_refined_minus_legacy_key_mass"] = qbound_time[
                "refined_minus_legacy_key_mass"
            ]
            qbound_aux["qbound_refined_trimmed_frame_count"] = qbound_time[
                "refined_trimmed_frame_count"
            ]

            # Full V1.5 readout diagnostic, detached.
            if legacy_v15_time is not None:
                qbound_aux["v15_diagnostic_key"] = legacy_v15_time["key"]
                qbound_aux["v15_diagnostic_start"] = legacy_v15_time["start"]
                qbound_aux["v15_diagnostic_end"] = legacy_v15_time["end"]
                qbound_aux["v15_diagnostic_mask"] = legacy_v15_time["mask"]

            # Full V1.4 readout diagnostic, detached.
            if legacy_v14_time is not None:
                qbound_aux["v14_diagnostic_key"] = legacy_v14_time["key"]
                qbound_aux["v14_diagnostic_start"] = legacy_v14_time["start"]
                qbound_aux["v14_diagnostic_end"] = legacy_v14_time["end"]
                qbound_aux["v14_diagnostic_mask"] = legacy_v14_time["mask"]

            qbound_aux["cra_qbound_key_l1"] = (
                (qbound_key - cra_key).abs().mean(dim=-1)
            )

            qbound_aux["question_only_conditioning"] = torch.as_tensor(
                float(question_feature is not None),
                dtype=video_proj.dtype,
                device=video_proj.device,
            )

            # Keep old compatibility fields disabled.
            qbound_aux["merge_reasoning_counterfactual"] = torch.as_tensor(
                0.0,
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
            qbound_aux["merge_counterfactual_available"] = torch.as_tensor(
                0.0,
                dtype=video_proj.dtype,
                device=video_proj.device,
            )

        time_param["qbound_aux"] = qbound_aux

        return (
            video_proj,
            final_key,
            time_param,
        )
