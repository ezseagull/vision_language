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
from modules.m_cra.event_bridge import GroundedPredictiveEventBridge


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
    CRA refinement decoder with Q-Bound GPEB Version 1.0.

    Version 1.0 keeps CRA as the grounding backbone and uses GPEB as a
    minimum-sufficient event reasoning branch. Compared with Version 0.5b:

      * the saturated D-dimensional fusion gate is removed;
      * event selection happens inside event_bridge.py at the interpretable
        event/transition level;
      * the GPEB residual is fused directly but smoothly norm-bounded;
      * optional temporal-grounding refinement uses a parameter-free,
        agreement-calibrated geometric interpolation;
      * original CRA grounding/alignment features are preserved separately;
      * a question-only feature can be passed explicitly for boundary learning.
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

        self.grounding_module = GroundingModule(
            d_model=embed_dim,
            dropout=model_cfg["dropout"],
            initial_sigma=model_cfg["sigma"],
            window_sizes=model_cfg["window_sizes"],
        )
        self.do = VisualIntervention(self.cfgs)

        # -------------------------------------------------------------
        # Q-Bound GPEB Version 1.0
        # -------------------------------------------------------------
        self.gpeb_enabled = bool(
            model_cfg.get("gpeb_enabled", False)
        )
        self.gpeb_require_question_only = bool(
            model_cfg.get(
                "gpeb_require_question_only",
                str(model_cfg.get("gpeb_version", "0.0")).startswith("1"),
            )
        )

        if self.gpeb_enabled:
            self.gpeb = GroundedPredictiveEventBridge(
                input_dim=embed_dim,
                hidden_dim=model_cfg.get(
                    "gpeb_hidden_dim",
                    256,
                ),
                # Retained constructor name. In Ver1.0 this is the maximum
                # number of latent events, not a fixed-bin count.
                num_bins=model_cfg.get(
                    "gpeb_num_bins",
                    model_cfg.get("gpeb_max_events", 4),
                ),
                use_grounding_prior=model_cfg.get(
                    "gpeb_use_grounding_prior",
                    True,
                ),
                dropout=model_cfg.get(
                    "gpeb_dropout",
                    dropout,
                ),
                grounding_prior_clip=model_cfg.get(
                    "gpeb_grounding_prior_clip",
                    4.0,
                ),
                grounding_scale_max=model_cfg.get(
                    "gpeb_grounding_scale_max",
                    5.0,
                ),
                boundary_temperature=model_cfg.get(
                    "gpeb_boundary_temperature",
                    0.7,
                ),
                assignment_temperature=model_cfg.get(
                    "gpeb_assignment_temperature",
                    0.35,
                ),
                boundary_bias_init=model_cfg.get(
                    "gpeb_boundary_bias_init",
                    -2.0,
                ),
                counterfactual_temperature=model_cfg.get(
                    "gpeb_counterfactual_temperature",
                    0.15,
                ),
                merge_strength=model_cfg.get(
                    "gpeb_merge_strength",
                    1.0,
                ),
                split_strength=model_cfg.get(
                    "gpeb_split_strength",
                    0.9,
                ),
                min_event_mass=model_cfg.get(
                    "gpeb_min_event_mass",
                    0.25,
                ),
                concurrent_temperature=model_cfg.get(
                    "gpeb_concurrent_temperature",
                    0.5,
                ),
                concurrent_bias_init=model_cfg.get(
                    "gpeb_concurrent_bias_init",
                    -2.0,
                ),
                concurrent_context_scale=model_cfg.get(
                    "gpeb_concurrent_context_scale",
                    0.10,
                ),
                concurrent_degree_max=model_cfg.get(
                    "gpeb_concurrent_degree_max",
                    1.0,
                ),
            )

            self.gpeb_detach_backbone_inputs = bool(
                model_cfg.get(
                    "gpeb_detach_backbone_inputs",
                    True,
                )
            )

            self.gpeb_residual_dropout = nn.Dropout(
                model_cfg.get(
                    "gpeb_residual_dropout",
                    dropout,
                )
            )

            # Ver1.0 removes the learned channel-wise gate. GPEB can only
            # change the CRA representation through a smoothly norm-bounded
            # residual. The default is deliberately below Ver0.5b's 0.15.
            self.gpeb_residual_max_ratio = float(
                model_cfg.get(
                    "gpeb_residual_max_ratio",
                    0.10,
                )
            )
            if self.gpeb_residual_max_ratio <= 0.0:
                raise ValueError(
                    "gpeb_residual_max_ratio must be positive."
                )

            # Grounding refinement is optional during the first smoke-test
            # stage. When enabled, no extra gate network is introduced.
            # The event evidence is mixed with CRA in log-probability space,
            # and its influence is automatically reduced when the two
            # distributions disagree.
            self.gpeb_enable_ground_refinement = bool(
                model_cfg.get(
                    "gpeb_enable_ground_refinement",
                    False,
                )
            )
            self.gpeb_ground_refine_max = float(
                model_cfg.get(
                    "gpeb_ground_refine_max",
                    0.05,
                )
            )
            if not 0.0 <= self.gpeb_ground_refine_max < 1.0:
                raise ValueError(
                    "gpeb_ground_refine_max must be in [0, 1)."
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

        # Avoid an all-zero distribution entering log-space refinement.
        # The fallback is uniform only over valid timesteps.
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
            # Masked mean avoids allowing padded frames to affect the
            # intervention representation while preserving CRA semantics.
            x_star = self._masked_mean_video(
                x_bar,
                valid_mask,
            )
            x_star = self.video_proj(x_star)
            projected = self.do(x_star, projected)

        return projected

    def _agreement_calibrated_grounding_refinement(
        self,
        cra_key,
        event_key,
        valid_mask,
    ):
        """
        Parameter-free bounded grounding refinement.

        CRA and event evidence are geometrically interpolated. The mixing
        coefficient is the configured upper bound multiplied by the
        detached Bhattacharyya agreement between the two distributions.
        Thus GPEB cannot strongly overwrite CRA when its evidence conflicts
        with the grounding backbone, and no new saturating gate is added.
        """
        eps = 1e-6
        cra_key = self._normalize_temporal_distribution(
            cra_key,
            valid_mask,
            eps=eps,
        )
        event_key = self._normalize_temporal_distribution(
            event_key,
            valid_mask,
            eps=eps,
        )

        agreement = torch.sum(
            torch.sqrt(
                cra_key.detach().clamp_min(eps)
                * event_key.detach().clamp_min(eps)
            ),
            dim=-1,
        ).clamp(min=0.0, max=1.0)

        ground_mix = (
            self.gpeb_ground_refine_max
            * agreement
        )

        mix = ground_mix.unsqueeze(-1)
        log_refined = (
            (1.0 - mix) * torch.log(cra_key.clamp_min(eps))
            + mix * torch.log(event_key.clamp_min(eps))
        )

        refined_key = torch.exp(log_refined)
        refined_key = self._normalize_temporal_distribution(
            refined_key,
            valid_mask,
            eps=eps,
        )

        grounding_shift = 0.5 * torch.sum(
            torch.abs(refined_key - cra_key),
            dim=-1,
        )

        return refined_key, agreement, ground_mix, grounding_shift

    def _smoothly_bound_residual(
        self,
        residual,
        base_feature,
    ):
        """
        Smoothly constrain ||residual|| / ||base|| below max_ratio.

        A tanh saturation is used instead of a hard clip. This preserves a
        magnitude gradient when the raw residual is too large. At exactly
        zero residual the scale is explicitly set to one, so the
        zero-initialized fusion adapter can start learning.
        """
        eps = 1e-6
        base_norm = base_feature.norm(
            p=2,
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
        residual_norm = residual.norm(
            p=2,
            dim=-1,
            keepdim=True,
        )

        raw_ratio = residual_norm / base_norm
        bounded_ratio = (
            self.gpeb_residual_max_ratio
            * torch.tanh(
                raw_ratio
                / self.gpeb_residual_max_ratio
            )
        )

        scale = torch.where(
            raw_ratio > eps,
            bounded_ratio / raw_ratio.clamp_min(eps),
            torch.ones_like(raw_ratio),
        )
        bounded_residual = residual * scale

        return (
            bounded_residual,
            raw_ratio.squeeze(-1),
            bounded_ratio.squeeze(-1),
            scale.squeeze(-1),
        )

    def forward(
        self,
        x,
        qa,
        self_mask=None,
        gsub=None,
        do=False,
        window_sizes=(1, 3, 5),
        use_gpeb=False,
        question_feature=None,
    ):
        # Preserve pre-refinement features for Q-Bound local change cues.
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

        # CRA remains the grounding backbone. Candidate-conditioned `qa`
        # is intentionally retained here for baseline compatibility.
        time_param = self.grounding_module(
            x_bar,
            qa,
            window_sizes=window_sizes,
        )

        cra_key = self._normalize_temporal_distribution(
            time_param["key"],
            valid_mask,
        )

        # Preserve every baseline grounding signal before any Ver1.0
        # refinement. Current evaluation code expects ori_key to exist.
        if "ori_key" not in time_param:
            time_param["ori_key"] = cra_key
        time_param["cra_key"] = cra_key

        cra_video_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=cra_key,
            valid_mask=valid_mask,
            do=do,
        )

        # Alignment keeps using the original CRA feature. This prevents
        # Q-Bound auxiliary/counterfactual objectives from silently changing
        # the old CRA alignment target.
        time_param["grounding_feature"] = cra_video_proj

        gpeb_aux = None
        final_key = cra_key
        fusion_base_proj = cra_video_proj
        video_proj = cra_video_proj

        if self.gpeb_enabled and use_gpeb:
            if (
                question_feature is None
                and self.gpeb_require_question_only
            ):
                raise ValueError(
                    "Q-Bound Ver1.0 requires question_feature=[B, D]. "
                    "Check models/cra.py and cra_pipeline.py."
                )

            # Compatibility fallback is allowed only when explicitly
            # configured for an older experiment.
            gpeb_language = (
                question_feature
                if question_feature is not None
                else qa
            )

            if self.gpeb_detach_backbone_inputs:
                raw_for_gpeb = x_raw.detach()
                contextual_for_gpeb = x_bar.detach()
                language_for_gpeb = gpeb_language.detach()
                grounding_for_gpeb = cra_key.detach()
            else:
                raw_for_gpeb = x_raw
                contextual_for_gpeb = x_bar
                language_for_gpeb = gpeb_language
                grounding_for_gpeb = cra_key

            gpeb_aux = self.gpeb(
                raw_video=raw_for_gpeb,
                contextual_video=contextual_for_gpeb,
                qa_features=language_for_gpeb,
                video_mask=valid_mask,
                grounding_weights=grounding_for_gpeb,
            )

            required_gpeb_keys = {
                "bridge_repr",
                "fusion_delta",
                "frame_event_relevance",
            }
            missing_gpeb_keys = sorted(
                required_gpeb_keys.difference(gpeb_aux)
            )
            if missing_gpeb_keys:
                raise KeyError(
                    "Event Bridge output is missing: "
                    + ", ".join(missing_gpeb_keys)
                )

            bridge_repr = gpeb_aux["bridge_repr"]
            raw_residual = self.gpeb_residual_dropout(
                gpeb_aux["fusion_delta"]
            )

            event_key = self._normalize_temporal_distribution(
                gpeb_aux["frame_event_relevance"],
                valid_mask,
            )
            time_param["event_key"] = event_key

            if self.gpeb_enable_ground_refinement:
                (
                    final_key,
                    ground_agreement,
                    ground_mix,
                    grounding_shift,
                ) = self._agreement_calibrated_grounding_refinement(
                    cra_key=cra_key,
                    event_key=event_key,
                    valid_mask=valid_mask,
                )

                fusion_base_proj = self._project_grounded_video(
                    x_bar=x_bar,
                    temporal_weights=final_key,
                    valid_mask=valid_mask,
                    do=do,
                )
            else:
                ground_agreement = torch.sum(
                    torch.sqrt(
                        cra_key.detach().clamp_min(1e-6)
                        * event_key.detach().clamp_min(1e-6)
                    ),
                    dim=-1,
                ).clamp(min=0.0, max=1.0)
                ground_mix = torch.zeros_like(
                    ground_agreement
                )
                grounding_shift = torch.zeros_like(
                    ground_agreement
                )

            (
                bounded_residual,
                residual_ratio_unclipped,
                residual_ratio,
                residual_scale,
            ) = self._smoothly_bound_residual(
                residual=raw_residual,
                base_feature=fusion_base_proj,
            )

            video_proj = fusion_base_proj + bounded_residual

            branch_cosine = F.cosine_similarity(
                fusion_base_proj,
                bridge_repr,
                dim=-1,
                eps=1e-6,
            )
            cra_branch_cosine = F.cosine_similarity(
                cra_video_proj,
                bridge_repr,
                dim=-1,
                eps=1e-6,
            )

            gpeb_aux.update(
                {
                    # Fusion diagnostics. `gated_residual` is retained only
                    # as a compatibility alias; no learned gate exists.
                    "bounded_residual": bounded_residual,
                    "gated_residual": bounded_residual,
                    "residual_ratio": residual_ratio,
                    "residual_ratio_mean": residual_ratio.mean(),
                    "residual_ratio_unclipped": (
                        residual_ratio_unclipped
                    ),
                    "residual_ratio_unclipped_mean": (
                        residual_ratio_unclipped.mean()
                    ),
                    "residual_scale": residual_scale,
                    "residual_scale_mean": residual_scale.mean(),
                    # Compatibility alias for the Ver0.5b logger.
                    "residual_clip_scale": residual_scale,
                    "residual_clip_scale_mean": (
                        residual_scale.mean()
                    ),
                    "residual_max_ratio": torch.as_tensor(
                        self.gpeb_residual_max_ratio,
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),
                    "branch_cosine_similarity": branch_cosine,
                    "branch_cosine_mean": branch_cosine.mean(),
                    "cra_branch_cosine_similarity": (
                        cra_branch_cosine
                    ),
                    "cra_branch_cosine_mean": (
                        cra_branch_cosine.mean()
                    ),

                    # Grounding diagnostics and live distributions used by
                    # the Ver1.0 consistency/counterfactual losses.
                    "cra_key": cra_key,
                    "event_key": event_key,
                    "refined_key": final_key,
                    "grounding_agreement": ground_agreement,
                    "grounding_agreement_mean": (
                        ground_agreement.mean()
                    ),
                    "ground_refine_mix": ground_mix,
                    "ground_refine_mix_mean": ground_mix.mean(),
                    "grounding_shift": grounding_shift,
                    "grounding_shift_mean": (
                        grounding_shift.mean()
                    ),
                    "ground_refinement_enabled": torch.as_tensor(
                        float(self.gpeb_enable_ground_refinement),
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),
                    "ground_refine_max": torch.as_tensor(
                        self.gpeb_ground_refine_max,
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),

                    # Features retained for auxiliary losses/diagnostics.
                    "cra_video_proj": cra_video_proj,
                    "fusion_base_proj": fusion_base_proj,
                    "fused_video_proj": video_proj,
                    "question_only_conditioning": torch.as_tensor(
                        float(question_feature is not None),
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),
                }
            )

        time_param["key"] = final_key
        time_param["refined_key"] = final_key
        time_param["gpeb_aux"] = gpeb_aux

        # Kept for compatibility with the current pipeline/model wrapper.
        time_param["gpeb_feature"] = video_proj

        return video_proj, final_key, time_param
