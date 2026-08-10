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
from modules.m_cra.qbound_adapter import CRAQBoundAdapter

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
    CRA refinement decoder with Q-Bound Version 1.1.

    Responsibilities are deliberately separated:

      * ``QBoundCore`` learns question-conditioned event topology/evidence.
      * ``GroundingModule`` remains the CRA temporal grounding backbone.
      * ``CRAQBoundAdapter`` owns the two backbone-specific scalar fusions:

            z_final = z_grounded + alpha * A(z_event)
            l_final = l_CRA      + beta  * l_event

    Q-Bound temporal evidence is fused BEFORE the final key/start/end are
    generated.  Merge/split counterfactuals pass through the same grounding
    path and the same event adapter as the original path, so the training
    pipeline can evaluate all of them with the same main VQA scorer.

    At initialization alpha=beta=0 by default, therefore enabling Q-Bound
    preserves the CRA prediction path while still allowing the two scalar
    parameters to learn whether event evidence is useful.
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
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout),
            num_layer,
        )
        self.norm = LayerNorm(embed_dim)

        # Kept for compatibility with the original CRA implementation.
        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        model_cfg = self.cfgs["model"]
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        self.grounding_module = GroundingModule(
            d_model=embed_dim,
            dropout=model_cfg["dropout"],
            initial_sigma=model_cfg["sigma"],
            window_sizes=model_cfg["window_sizes"],
        )
        self.do = VisualIntervention(self.cfgs)

        # ------------------------------------------------------------------
        # Q-Bound V1.1
        #
        # New qbound_* names are preferred. gpeb_* fallbacks are retained so
        # the first V1.1 run does not require an immediate full YAML rewrite.
        # ------------------------------------------------------------------
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
                    model_cfg.get(
                        "gpeb_require_question_only",
                        True,
                    ),
                ),
            )
        )

        if self.qbound_enabled:
            self.qbound = QBoundCore(
                input_dim=embed_dim,
                hidden_dim=qbound_cfg.get(
                    "hidden_dim",
                    model_cfg.get(
                        "qbound_hidden_dim",
                        model_cfg.get("gpeb_hidden_dim", 256),
                    ),
                ),
                max_events=qbound_cfg.get(
                    "max_events",
                    model_cfg.get(
                        "qbound_max_events",
                        model_cfg.get(
                            "gpeb_max_events",
                            model_cfg.get("gpeb_num_bins", 4),
                        ),
                    ),
                ),
                use_temporal_prior=qbound_cfg.get(
                    "use_temporal_prior",
                    model_cfg.get(
                        "qbound_use_temporal_prior",
                        model_cfg.get(
                            "gpeb_use_grounding_prior",
                            False,
                        ),
                    ),
                ),
                dropout=qbound_cfg.get(
                    "dropout",
                    model_cfg.get(
                        "qbound_dropout",
                        model_cfg.get("gpeb_dropout", dropout),
                    ),
                ),
                boundary_temperature=qbound_cfg.get(
                    "boundary_temperature",
                    model_cfg.get(
                        "qbound_boundary_temperature",
                        model_cfg.get(
                            "gpeb_boundary_temperature",
                            0.7,
                        ),
                    ),
                ),
                assignment_temperature=qbound_cfg.get(
                    "assignment_temperature",
                    model_cfg.get(
                        "qbound_assignment_temperature",
                        model_cfg.get(
                            "gpeb_assignment_temperature",
                            0.35,
                        ),
                    ),
                ),
                boundary_bias_init=qbound_cfg.get(
                    "boundary_bias_init",
                    model_cfg.get(
                        "qbound_boundary_bias_init",
                        model_cfg.get(
                            "gpeb_boundary_bias_init",
                            -2.0,
                        ),
                    ),
                ),
                counterfactual_temperature=qbound_cfg.get(
                    "counterfactual_temperature",
                    model_cfg.get(
                        "qbound_counterfactual_temperature",
                        model_cfg.get(
                            "gpeb_counterfactual_temperature",
                            0.15,
                        ),
                    ),
                ),
                merge_strength=qbound_cfg.get(
                    "merge_strength",
                    model_cfg.get(
                        "qbound_merge_strength",
                        model_cfg.get("gpeb_merge_strength", 1.0),
                    ),
                ),
                split_strength=qbound_cfg.get(
                    "split_strength",
                    model_cfg.get(
                        "qbound_split_strength",
                        model_cfg.get("gpeb_split_strength", 0.9),
                    ),
                ),
                min_event_mass=qbound_cfg.get(
                    "min_event_mass",
                    model_cfg.get(
                        "qbound_min_event_mass",
                        model_cfg.get("gpeb_min_event_mass", 0.25),
                    ),
                ),
            )

            self.qbound_adapter = CRAQBoundAdapter(
                embed_dim=embed_dim,
                dropout=0.0,
                event_scale_init=qbound_cfg.get(
                    "event_scale_init",
                    model_cfg.get("qbound_event_scale_init", 0.0),
                ),
                temporal_scale_init=qbound_cfg.get(
                    "temporal_scale_init",
                    model_cfg.get("qbound_temporal_scale_init", 0.0),
                ),
            )

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
                "self_mask must reduce to [B, T], got "
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
                "Temporal distribution and valid mask must have the "
                f"same shape, got {tuple(values.shape)} and "
                f"{tuple(valid_mask.shape)}."
            )

        values = values.clamp_min(0.0)
        values = values * valid_mask.to(values.dtype)

        denominator = values.sum(
            dim=-1,
            keepdim=True,
        )
        normalized = values / denominator.clamp_min(eps)

        fallback = valid_mask.to(values.dtype)
        fallback = fallback / fallback.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)

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

    def _project_grounded_video(
        self,
        x_bar,
        temporal_weights,
        valid_mask,
        do,
    ):
        pooled = torch.sum(
            x_bar * temporal_weights.unsqueeze(-1),
            dim=1,
        )
        projected = self.video_proj(pooled)

        if do:
            x_star = self._masked_mean_video(
                x_bar,
                valid_mask,
            )
            x_star = self.video_proj(x_star)
            projected = self.do(x_star, projected)

        return projected

    def _build_temporal_bias(
        self,
        event_temporal_logits,
    ):
        """
        CRA-specific temporal contribution.

        QBoundCore produces scale-normalized event evidence.  The adapter
        contributes a single scalar beta; no per-frame gate or hard cap is
        introduced in V1.1.
        """
        return (
            self.qbound_adapter.beta
            * event_temporal_logits
        )

    def _finalize_counterfactual(
        self,
        base_temporal,
        x_bar,
        valid_mask,
        do,
        event_repr,
        event_temporal_logits,
        window_sizes,
    ):
        """
        Run a merge/split topology through the SAME grounding and fusion
        machinery used by the original topology.

        event_repr:
            [B, D]
        event_temporal_logits:
            [B, T]
        """
        temporal_bias = self._build_temporal_bias(
            event_temporal_logits
        )

        cf_time_param = (
            self.grounding_module.finalize_grounding(
                base_temporal,
                temporal_bias=temporal_bias,
                window_sizes=window_sizes,
            )
        )

        grounded_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=cf_time_param["key"],
            valid_mask=valid_mask,
            do=do,
        )

        fused_proj, _ = self.qbound_adapter.fuse_video(
            grounded_proj,
            event_repr,
        )

        return fused_proj, cf_time_param, temporal_bias

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
        # Compatibility alias for an older direct Decoder caller.
        use_gpeb=None,
    ):
        del gsub

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # Preserve pre-refinement video tokens for local-change cues.
        x_raw = x

        for layer in self.layers:
            x = layer(x, self_mask)
        x_bar = self.norm(x)

        batch_size, time_steps, _ = x_bar.shape
        valid_mask = self._normalize_video_mask(
            self_mask,
            batch_size,
            time_steps,
            x_bar.device,
        )

        # --------------------------------------------------------------
        # 1) Compute CRA temporal evidence ONCE, before final grounding.
        # --------------------------------------------------------------
        base_temporal = (
            self.grounding_module.compute_base_temporal(
                x_bar,
                qa,
                valid_mask=valid_mask,
            )
        )

        cra_key = self._normalize_temporal_distribution(
            base_temporal["cra_key"],
            valid_mask,
        )

        # CRA-only grounded representation is retained for diagnostics and
        # the original CRA alignment loss.
        cra_video_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=cra_key,
            valid_mask=valid_mask,
            do=do,
        )

        qbound_aux = None
        temporal_bias = torch.zeros_like(
            base_temporal["cra_temporal_logits"]
        )

        # --------------------------------------------------------------
        # 2) Q-Bound learns topology/evidence independently from CRA.
        # --------------------------------------------------------------
        if self.qbound_enabled and use_qbound:
            if (
                question_feature is None
                and self.qbound_require_question_only
            ):
                raise ValueError(
                    "Q-Bound V1.1 requires question_feature=[B,D]. "
                    "Pass question-only tokens through CRA.forward(..., "
                    "question=...)."
                )

            qbound_language = (
                question_feature
                if question_feature is not None
                else qa
            )

            if self.qbound_detach_backbone_inputs:
                video_for_qbound = x_raw.detach()
                contextual_for_qbound = x_bar.detach()
                language_for_qbound = qbound_language.detach()
            else:
                video_for_qbound = x_raw
                contextual_for_qbound = x_bar
                language_for_qbound = qbound_language

            qbound_aux = self.qbound(
                video_tokens=video_for_qbound,
                question_features=language_for_qbound,
                video_mask=valid_mask,
                contextual_tokens=contextual_for_qbound,
                # QBoundCore detaches the prior internally.  If prior use is
                # disabled in the core, it becomes an all-zero input feature.
                temporal_prior=cra_key,
            )

            required_keys = {
                "bridge_repr",
                "event_temporal_logits",
                "frame_event_relevance",
                "merge_bridge_repr",
                "merge_event_temporal_logits",
                "split_bridge_repr",
                "split_event_temporal_logits",
            }
            missing = sorted(
                required_keys.difference(qbound_aux)
            )
            if missing:
                raise KeyError(
                    "Q-Bound output is missing: "
                    + ", ".join(missing)
                )

            temporal_bias = self._build_temporal_bias(
                qbound_aux["event_temporal_logits"]
            )

        # --------------------------------------------------------------
        # 3) FINAL grounding.  Q-Bound acts BEFORE key/start/end.
        # --------------------------------------------------------------
        time_param = self.grounding_module.finalize_grounding(
            base_temporal,
            temporal_bias=temporal_bias,
            window_sizes=window_sizes,
        )

        final_key = self._normalize_temporal_distribution(
            time_param["key"],
            valid_mask,
        )
        time_param["key"] = final_key
        time_param["refined_key"] = final_key

        grounding_base_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=final_key,
            valid_mask=valid_mask,
            do=do,
        )

        # Preserve the original CRA alignment target as in V1.0.
        time_param["grounding_feature"] = cra_video_proj

        video_proj = grounding_base_proj

        # --------------------------------------------------------------
        # 4) Main representation fusion + full main-path counterfactuals.
        # --------------------------------------------------------------
        if qbound_aux is not None:
            video_proj, fusion_info = (
                self.qbound_adapter.fuse_video(
                    grounding_base_proj,
                    qbound_aux["bridge_repr"],
                )
            )

            # Event-only distribution is diagnostic only.  It is NOT forced
            # to agree with CRA and it is never used as a consistency target.
            event_key = self._normalize_temporal_distribution(
                qbound_aux["frame_event_relevance"],
                valid_mask,
            )

            merge_video_proj, merge_time_param, merge_bias = (
                self._finalize_counterfactual(
                    base_temporal=base_temporal,
                    x_bar=x_bar,
                    valid_mask=valid_mask,
                    do=do,
                    event_repr=(
                        qbound_aux["merge_bridge_repr"][:, 0]
                    ),
                    event_temporal_logits=(
                        qbound_aux[
                            "merge_event_temporal_logits"
                        ][:, 0]
                    ),
                    window_sizes=window_sizes,
                )
            )

            split_video_proj, split_time_param, split_bias = (
                self._finalize_counterfactual(
                    base_temporal=base_temporal,
                    x_bar=x_bar,
                    valid_mask=valid_mask,
                    do=do,
                    event_repr=(
                        qbound_aux["split_bridge_repr"][:, 0]
                    ),
                    event_temporal_logits=(
                        qbound_aux[
                            "split_event_temporal_logits"
                        ][:, 0]
                    ),
                    window_sizes=window_sizes,
                )
            )

            alpha = self.qbound_adapter.alpha
            beta = self.qbound_adapter.beta

            # One diagnostic for when topology-dependent structural training
            # should become meaningful.  The pipeline may detach/use this
            # rather than relying on an epoch schedule.
            qbound_use_strength = 0.5 * (
                alpha.abs() + beta.abs()
            )

            qbound_aux.update(
                {
                    # Main-path scalar integration.
                    "event_scale": (
                        self.qbound_adapter.event_scale
                    ),
                    "temporal_scale": (
                        self.qbound_adapter.temporal_scale
                    ),
                    "alpha": alpha,
                    "beta": beta,
                    "qbound_use_strength": (
                        qbound_use_strength
                    ),

                    # Original/main path.
                    "cra_key": cra_key,
                    "event_key": event_key,
                    "final_key": final_key,
                    "refined_key": final_key,
                    "temporal_bias": temporal_bias,
                    "cra_video_proj": cra_video_proj,
                    "grounding_base_proj": (
                        grounding_base_proj
                    ),
                    "event_delta": fusion_info[
                        "event_delta"
                    ],
                    "fused_video_proj": video_proj,

                    # Merge counterfactual through the same main path.
                    "merge_temporal_bias": merge_bias,
                    "merge_final_key": (
                        merge_time_param["key"].unsqueeze(1)
                    ),
                    "merge_start": (
                        merge_time_param["start"].unsqueeze(1)
                    ),
                    "merge_end": (
                        merge_time_param["end"].unsqueeze(1)
                    ),
                    "merge_fused_video_proj": (
                        merge_video_proj.unsqueeze(1)
                    ),

                    # Split counterfactual through the same main path.
                    "split_temporal_bias": split_bias,
                    "split_final_key": (
                        split_time_param["key"].unsqueeze(1)
                    ),
                    "split_start": (
                        split_time_param["start"].unsqueeze(1)
                    ),
                    "split_end": (
                        split_time_param["end"].unsqueeze(1)
                    ),
                    "split_fused_video_proj": (
                        split_video_proj.unsqueeze(1)
                    ),

                    "question_only_conditioning": (
                        torch.as_tensor(
                            float(question_feature is not None),
                            device=video_proj.device,
                            dtype=video_proj.dtype,
                        )
                    ),
                }
            )

        time_param["qbound_aux"] = qbound_aux
        time_param["qbound_feature"] = video_proj

        # Compatibility aliases for old logging/pipeline code.  These can be
        # removed after cra_pipeline.py is migrated to V1.1 names.
        time_param["gpeb_aux"] = qbound_aux
        time_param["gpeb_feature"] = video_proj

        return video_proj, final_key, time_param
