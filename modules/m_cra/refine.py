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
    CRA refinement decoder with Q-Bound Version 1.3.

    Design
    ------
    CRA is kept as a strong feature backbone and as a TEMPORARY temporal
    teacher, but CRA temporal grounding is not allowed to remain a permanent
    VQA bypass.

    VQ / CRA-only path:
        CRA temporal grounding -> CRA key -> grounded video

    VQA / Q-Bound path:
        CRA teacher key (stop-gradient)
                         \
                          -> deterministic probability handover -> final key
                         /
        Q-Bound topology -> Q-Bound temporal probability
                         -> probability-preserving Gaussian smoothing
                         -> temporal grounded feature

        Q-Bound dynamic events
                         -> ordered event interaction
                         -> reasoned_event_repr
                         -> small Q-Bound adapter
                         -> answer-side residual

        main_video_feature =
            temporal_grounded_feature
            + qbound_mix * reasoning_residual

    During training, ``qbound_mix`` is supplied externally by the pipeline.
    It must follow a fixed schedule and eventually reach 1.0.

        final_key =
            (1 - qbound_mix) * stopgrad(cra_key)
            + qbound_mix * qbound_key

    Validation/test should use qbound_mix=1.0 so checkpoint selection never
    benefits from a permanent CRA temporal shortcut.

    Important V1.3 properties
    -------------------------
    * Q-Bound core owns dynamic 0..K-1 boundary selection and contiguous events;
    * CRA never defines Q-Bound topology and remains outside the core;
    * no alpha/beta gate and no CRA + Q-Bound temporal-logit addition;
    * no second softmax on the Q-Bound Gaussian branch;
    * CRA sigma is detached on the Q-Bound smoothing branch;
    * main VQA still follows the deterministic CRA->Q-Bound temporal handover;
    * unlike V1.2, ordered-event reasoning has a direct residual path to the
      main answer representation, scaled by the same deterministic curriculum;
    * the residual adapter is small-initialized and has no learnable scalar
      bypass gate that can trivially disable ordered reasoning;
    * a PURE-QBound grounded feature is exposed for weak alignment so the
      alignment objective cannot be solved by the temporary CRA teacher;
    * event_visual_tokens/evidence_visual_token are propagated for future
      weak event alignment without putting dataset/loss logic inside the core;
    * transition-level merge remains a diagnostic counterfactual;
    * original, pure-QBound, CRA diagnostic, and merge paths share the same
      projection dropout realization whenever their dimensions match.
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

        # Kept only for checkpoint compatibility with the original CRA
        # implementation. The refine decoder returns its actual final temporal
        # key directly and does not use this attention head in V1.1.3.
        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        # Preserve original parameter names:
        #     video_proj.0 = Dropout
        #     video_proj.1 = Linear
        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        model_cfg = self.cfgs["model"]
        qbound_cfg = model_cfg.get("qbound", {})

        if not isinstance(qbound_cfg, dict):
            raise ValueError(
                "model.qbound must be a mapping/dict."
            )

        self.grounding_module = GroundingModule(
            d_model=embed_dim,
            dropout=model_cfg["dropout"],
            initial_sigma=model_cfg["sigma"],
            window_sizes=model_cfg["window_sizes"],
        )

        self.do = VisualIntervention(
            self.cfgs
        )

        # --------------------------------------------------------------
        # Q-Bound V1.3
        # --------------------------------------------------------------
        self.qbound_enabled = bool(
            qbound_cfg.get(
                "enabled",
                model_cfg.get(
                    "qbound_enabled",
                    model_cfg.get(
                        "gpeb_enabled",
                        False,
                    ),
                ),
            )
        )

        # Safe defaults keep the CRA-only / qbound-disabled path fully
        # functional. They are overwritten below when Q-Bound is enabled.
        self.qbound_hidden_dim = None
        self.qbound_reasoning_scale = 0.0

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
            # V1.2 intentionally stops inheriting the most topology-sensitive
            # defaults from old GPEB/V1.1 configs. In particular, the old
            # boundary_bias_init=-2.0 was appropriate for dense soft boundary
            # mass but can collapse hard dynamic selection to zero transitions.
            # Explicit values under model.qbound still override these defaults.
            qbound_hidden_dim = int(
                qbound_cfg.get(
                    "hidden_dim",
                    model_cfg.get("qbound_hidden_dim", 256),
                )
            )
            self.qbound_hidden_dim = qbound_hidden_dim

            self.qbound = QBoundCore(
                input_dim=embed_dim,
                hidden_dim=qbound_hidden_dim,
                max_events=qbound_cfg.get(
                    "max_events",
                    model_cfg.get("qbound_max_events", 4),
                ),
                # CRA temporal evidence is not part of V1.2 topology. This
                # option remains available for later generic-prior ablations,
                # but forward() below deliberately passes temporal_prior=None.
                use_temporal_prior=qbound_cfg.get(
                    "use_temporal_prior",
                    False,
                ),
                dropout=qbound_cfg.get(
                    "dropout",
                    model_cfg.get("qbound_dropout", dropout),
                ),
                boundary_temperature=qbound_cfg.get(
                    "boundary_temperature",
                    model_cfg.get("qbound_boundary_temperature", 0.7),
                ),
                # Soft assignment exists only as the ST gradient surrogate in
                # V1.2; the forward event partition itself is hard/contiguous.
                assignment_temperature=qbound_cfg.get(
                    "assignment_temperature",
                    model_cfg.get("qbound_assignment_temperature", 0.5),
                ),
                boundary_threshold=qbound_cfg.get(
                    "boundary_threshold",
                    model_cfg.get("qbound_boundary_threshold", 0.5),
                ),
                boundary_radius=qbound_cfg.get(
                    "boundary_radius",
                    model_cfg.get("qbound_boundary_radius", 2),
                ),
                boundary_bias_init=qbound_cfg.get(
                    "boundary_bias_init",
                    model_cfg.get("qbound_boundary_bias_init", -0.2),
                ),
                min_event_mass=qbound_cfg.get(
                    "min_event_mass",
                    model_cfg.get("qbound_min_event_mass", 0.25),
                ),
            )

            # ----------------------------------------------------------
            # V1.3 adapter: native Q-Bound ordered-event representation
            # [B,H] -> CRA answer space [B,D].
            #
            # The adapter is intentionally tiny and initialized with a small
            # weight.  Therefore V1.3 starts close to the already-stable V1.2
            # temporal path, while the main VQA loss can directly train the
            # ordered-event reasoner through this residual route.  Its module
            # name contains ``qbound`` so BasePipeline's existing Q-Bound
            # optimizer-group logic automatically includes these parameters.
            # ----------------------------------------------------------
            self.qbound_reasoning_adapter = nn.Sequential(
                nn.LayerNorm(qbound_hidden_dim),
                nn.Linear(qbound_hidden_dim, embed_dim, bias=False),
            )
            nn.init.normal_(
                self.qbound_reasoning_adapter[-1].weight,
                mean=0.0,
                std=1e-2,
            )

            # Fixed residual strength, not a learnable bypass gate. During the
            # CRA->Q-Bound curriculum it is additionally multiplied by
            # qbound_mix, so validation/test (mix=1) use the full event-
            # reasoning contribution while early training remains stable.
            self.qbound_reasoning_scale = float(
                qbound_cfg.get(
                    "reasoning_residual_scale",
                    model_cfg.get("qbound_reasoning_residual_scale", 1.0),
                )
            )
            if self.qbound_reasoning_scale < 0.0:
                raise ValueError(
                    "qbound.reasoning_residual_scale must be >= 0."
                )

            # Keep backbone detachment configurable. For the first controlled
            # V1.2 run, True isolates whether dynamic topology + weak alignment
            # can improve grounding without changing the CRA feature backbone.
            # This is NOT a dependency on CRA: a future generic backbone can be
            # detached or trained by the same adapter-level switch.
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

        while (
            mask.ndim > 2
            and mask.size(1) == 1
        ):
            mask = mask.squeeze(1)

        if mask.ndim != 2:
            raise ValueError(
                "self_mask must reduce to [B,T], got "
                f"{tuple(mask.shape)}."
            )

        if mask.shape != (
            batch_size,
            time_steps,
        ):
            raise ValueError(
                "self_mask shape mismatch. Expected "
                f"{(batch_size, time_steps)}, "
                f"got {tuple(mask.shape)}."
            )

        return mask.to(
            device=device,
        ).bool()

    @staticmethod
    def _normalize_temporal_distribution(
        values,
        valid_mask,
        eps=1e-6,
    ):
        if (
            values.ndim == 3
            and values.size(-1) == 1
        ):
            values = values.squeeze(-1)

        if values.shape != valid_mask.shape:
            raise ValueError(
                "Temporal distribution and valid mask must have "
                "the same [B,T] shape. Got "
                f"{tuple(values.shape)} and "
                f"{tuple(valid_mask.shape)}."
            )

        values = values.clamp_min(0.0)
        values = (
            values
            * valid_mask.to(values.dtype)
        )

        denominator = values.sum(
            dim=-1,
            keepdim=True,
        )

        normalized = (
            values
            / denominator.clamp_min(eps)
        )

        fallback = valid_mask.to(
            values.dtype
        )
        fallback = (
            fallback
            / fallback.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1.0)
        )

        return torch.where(
            denominator > eps,
            normalized,
            fallback,
        )

    @staticmethod
    def _normalize_qbound_mix(
        qbound_mix,
        batch_size,
        device,
        dtype,
    ):
        """
        Return a [B,1] deterministic CRA->Q-Bound handover weight.

        qbound_mix may be:
            float / int
            scalar tensor
            [B] tensor
            [B,1] tensor

        V1.2 intentionally has NO learnable mixing parameter.
        """
        if qbound_mix is None:
            qbound_mix = 1.0

        if torch.is_tensor(qbound_mix):
            mix = qbound_mix.to(
                device=device,
                dtype=dtype,
            )
        else:
            mix = torch.tensor(
                float(qbound_mix),
                device=device,
                dtype=dtype,
            )

        if mix.ndim == 0:
            mix = mix.expand(
                batch_size
            ).unsqueeze(-1)

        elif mix.ndim == 1:
            if mix.numel() == 1:
                mix = mix.expand(
                    batch_size
                ).unsqueeze(-1)
            elif mix.numel() == batch_size:
                mix = mix.unsqueeze(-1)
            else:
                raise ValueError(
                    "1-D qbound_mix must contain either one value "
                    f"or B={batch_size} values, got {mix.numel()}."
                )

        elif mix.ndim == 2:
            if mix.shape == (1, 1):
                mix = mix.expand(
                    batch_size,
                    1,
                )
            elif mix.shape != (
                batch_size,
                1,
            ):
                raise ValueError(
                    "2-D qbound_mix must be [B,1]. "
                    f"Expected {(batch_size, 1)}, "
                    f"got {tuple(mix.shape)}."
                )

        else:
            raise ValueError(
                "qbound_mix must be scalar, [B], or [B,1]."
            )

        if (
            (mix < 0.0).any()
            or (mix > 1.0).any()
        ):
            raise ValueError(
                "qbound_mix must lie in [0,1]."
            )

        # It is a schedule value, never a learnable model parameter.
        return mix.detach()

    @staticmethod
    def _masked_mean_video(
        features,
        valid_mask,
    ):
        weights = (
            valid_mask.to(features.dtype)
            .unsqueeze(-1)
        )

        return (
            (features * weights)
            .sum(dim=1)
            / weights.sum(
                dim=1
            ).clamp_min(1.0)
        )

    # ------------------------------------------------------------------
    # Shared CRA video projection helpers
    # ------------------------------------------------------------------
    def _sample_projection_dropout_mask(
        self,
        reference,
    ):
        """
        Sample one projection-dropout realization.

        The SAME realization is reused by CRA diagnostic, main/original,
        and merge counterfactual paths so task-harm measurements are not
        contaminated by different random dropout masks.
        """
        dropout_layer = self.video_proj[0]

        if (
            not self.training
            or dropout_layer.p <= 0.0
        ):
            return torch.ones_like(
                reference
            )

        keep_prob = (
            1.0
            - float(dropout_layer.p)
        )

        if keep_prob <= 0.0:
            raise ValueError(
                "video projection dropout must be < 1."
            )

        mask = (
            torch.empty_like(reference)
            .bernoulli_(keep_prob)
        )

        return mask / keep_prob

    def _apply_video_projection(
        self,
        pooled,
        dropout_mask,
    ):
        if (
            pooled.shape
            != dropout_mask.shape
        ):
            raise ValueError(
                "pooled representation and dropout mask must "
                "have the same shape. Got "
                f"{tuple(pooled.shape)} and "
                f"{tuple(dropout_mask.shape)}."
            )

        return self.video_proj[1](
            pooled * dropout_mask
        )

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
            x_bar
            * temporal_weights.unsqueeze(-1),
            dim=1,
        )

        projected = (
            self._apply_video_projection(
                pooled,
                projection_dropout_mask,
            )
        )

        if do:
            if x_star_proj is None:
                raise ValueError(
                    "x_star_proj is required when visual "
                    "intervention is enabled."
                )

            projected = self.do(
                x_star_proj,
                projected,
            )

        return projected

    def _project_qbound_reasoning(self, reasoned_event_repr):
        """Project Q-Bound's native [B,H] ordered-event state to [B,D]."""
        if reasoned_event_repr.ndim != 2:
            raise ValueError(
                "reasoned_event_repr must be [B,H], got "
                f"{tuple(reasoned_event_repr.shape)}."
            )
        if reasoned_event_repr.size(-1) != self.qbound_hidden_dim:
            raise ValueError(
                "reasoned_event_repr hidden size mismatch. Expected "
                f"{self.qbound_hidden_dim}, got "
                f"{reasoned_event_repr.size(-1)}."
            )
        return self.qbound_reasoning_adapter(reasoned_event_repr)

    def _inject_qbound_reasoning(
        self,
        temporal_video_proj,
        reasoning_residual,
        qbound_mix,
    ):
        """
        Add ordered-event reasoning to the main answer representation.

        This is deliberately a deterministic residual, not a learnable scalar
        gate that could trivially collapse to zero and recreate V1.2's bypass.
        The same CRA->Q-Bound curriculum value controls when the new path is
        introduced:

            main = temporal_feature
                   + mix * scale * ordered_event_residual

        At validation/test mix=1, CRA temporal grounding is already absent and
        the full ordered-event residual participates in answer prediction.
        """
        if temporal_video_proj.shape != reasoning_residual.shape:
            raise ValueError(
                "temporal_video_proj and reasoning_residual must both be "
                f"[B,D], got {tuple(temporal_video_proj.shape)} and "
                f"{tuple(reasoning_residual.shape)}."
            )
        if qbound_mix.ndim != 2 or qbound_mix.shape != (
            temporal_video_proj.size(0), 1
        ):
            raise ValueError(
                "qbound_mix must be normalized to [B,1] before reasoning "
                "injection."
            )

        scaled_residual = (
            self.qbound_reasoning_scale
            * qbound_mix
            * reasoning_residual
        )
        return temporal_video_proj + scaled_residual, scaled_residual

    # ------------------------------------------------------------------
    # V1.3 temporal handover helpers
    # ------------------------------------------------------------------
    def _mix_temporal_keys(
        self,
        cra_key,
        qbound_key,
        qbound_mix,
        valid_mask,
    ):
        """
        Deterministic probability-space handover.

        CRA temporal weights are detached here even if a caller accidentally
        supplies a tensor with gradients.

        final_key =
            (1-lambda) * stopgrad(cra_key)
            + lambda * qbound_key
        """
        if (
            cra_key.shape != qbound_key.shape
            or cra_key.shape != valid_mask.shape
        ):
            raise ValueError(
                "cra_key, qbound_key, and valid_mask must share "
                "shape [B,T]."
            )

        batch_size = cra_key.size(0)

        mix = self._normalize_qbound_mix(
            qbound_mix,
            batch_size,
            cra_key.device,
            cra_key.dtype,
        )

        mixed = (
            (1.0 - mix)
            * cra_key.detach()
            + mix
            * qbound_key
        )

        mixed = (
            mixed
            * valid_mask.to(mixed.dtype)
        )

        mixed = (
            mixed
            / mixed.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-6)
        )

        return mixed, mix

    def _build_time_param_from_key(
        self,
        final_key,
        base_temporal,
        qbound_key,
        qbound_mix,
        window_sizes,
    ):
        """
        Create start/end using the repository's unchanged interval extraction.
        """
        time_param = (
            self.grounding_module
            .gen_grounding(
                final_key,
                window_sizes=window_sizes,
            )
        )

        # CRA baseline/teacher diagnostics.
        time_param["ori_key"] = (
            base_temporal["ori_key"]
        )
        time_param["cra_key"] = (
            base_temporal["cra_key"]
        )

        # V1.3 branch diagnostics.
        time_param["qbound_key"] = (
            qbound_key
        )
        time_param["refined_key"] = (
            final_key
        )
        time_param["qbound_mix"] = (
            qbound_mix.squeeze(-1)
        )

        return time_param

    def _finalize_qbound_counterfactual(
        self,
        merge_temporal_prob,
        cra_key,
        qbound_mix,
        x_bar,
        valid_mask,
        do,
        window_sizes,
        projection_dropout_mask,
        x_star_proj,
    ):
        """
        Run transition-level MERGE through exactly the same Q-Bound grounding,
        handover, video projection, and visual-intervention path as original.
        """
        merge_qbound_time = (
            self.grounding_module
            .finalize_probability_grounding(
                merge_temporal_prob,
                valid_mask=valid_mask,
                window_sizes=window_sizes,
                detach_sigma=True,
            )
        )

        merge_qbound_key = (
            self._normalize_temporal_distribution(
                merge_qbound_time["key"],
                valid_mask,
            )
        )

        merge_final_key, _ = (
            self._mix_temporal_keys(
                cra_key=cra_key,
                qbound_key=merge_qbound_key,
                qbound_mix=qbound_mix,
                valid_mask=valid_mask,
            )
        )

        # Recompute interval from the actually-used mixed key.
        merge_time_param = (
            self.grounding_module
            .gen_grounding(
                merge_final_key,
                window_sizes=window_sizes,
            )
        )

        merge_time_param["qbound_key"] = (
            merge_qbound_key
        )
        merge_time_param["cra_key"] = (
            cra_key
        )
        merge_time_param["refined_key"] = (
            merge_final_key
        )

        merge_video_proj = (
            self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=merge_final_key,
                valid_mask=valid_mask,
                do=do,
                projection_dropout_mask=(
                    projection_dropout_mask
                ),
                x_star_proj=x_star_proj,
            )
        )

        return (
            merge_video_proj,
            merge_time_param,
            merge_qbound_key,
            merge_final_key,
        )

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
        # Temporary compatibility alias for an older direct Decoder caller.
        use_gpeb=None,
    ):
        del gsub

        if use_gpeb is not None:
            use_qbound = bool(
                use_qbound
                or use_gpeb
            )

        # Preserve pre-refinement tokens for Q-Bound local-change features.
        x_raw = x

        for layer in self.layers:
            x = layer(
                x,
                self_mask,
            )

        x_bar = self.norm(x)

        (
            batch_size,
            time_steps,
            _,
        ) = x_bar.shape

        valid_mask = (
            self._normalize_video_mask(
                self_mask,
                batch_size,
                time_steps,
                x_bar.device,
            )
        )

        qbound_active = bool(
            self.qbound_enabled
            and use_qbound
        )

        # --------------------------------------------------------------
        # 1) CRA temporal branch.
        #
        # For Q-Bound VQA this branch is a DETACHED teacher only.
        # For VQ / CRA-only mode it remains the original trainable CRA path.
        # --------------------------------------------------------------
        if qbound_active:
            with torch.no_grad():
                base_temporal = (
                    self.grounding_module
                    .compute_base_temporal(
                        x_bar,
                        qa,
                        valid_mask=valid_mask,
                    )
                )
        else:
            base_temporal = (
                self.grounding_module
                .compute_base_temporal(
                    x_bar,
                    qa,
                    valid_mask=valid_mask,
                )
            )

        cra_key = (
            self._normalize_temporal_distribution(
                base_temporal["cra_key"],
                valid_mask,
            )
        )

        if qbound_active:
            # Defensive detach in addition to the no-grad teacher block.
            cra_key = cra_key.detach()

        # --------------------------------------------------------------
        # 2) One shared stochastic projection context.
        # --------------------------------------------------------------
        projection_reference = (
            self._masked_mean_video(
                x_bar,
                valid_mask,
            )
        )

        projection_dropout_mask = (
            self._sample_projection_dropout_mask(
                projection_reference
            )
        )

        x_star_proj = None

        if do:
            x_star_dropout_mask = (
                self._sample_projection_dropout_mask(
                    projection_reference
                )
            )

            x_star_proj = (
                self._apply_video_projection(
                    projection_reference,
                    x_star_dropout_mask,
                )
            )

        # CRA representation is diagnostic-only in VQA/Q-Bound mode.
        # In VQ/CRA mode it is also the main representation because final_key
        # equals cra_key.
        cra_video_proj = (
            self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=cra_key,
                valid_mask=valid_mask,
                do=do,
                projection_dropout_mask=(
                    projection_dropout_mask
                ),
                x_star_proj=x_star_proj,
            )
        )

        # --------------------------------------------------------------
        # 3) Q-Bound branch.
        # --------------------------------------------------------------
        qbound_aux = None

        if qbound_active:
            if (
                question_feature is None
                and self.qbound_require_question_only
            ):
                raise ValueError(
                    "Q-Bound V1.3 requires "
                    "question_feature=[B,D]. Pass question-only "
                    "tokens through CRA.forward(..., question=...)."
                )

            qbound_language = (
                question_feature
                if question_feature is not None
                else qa
            )

            if self.qbound_detach_backbone_inputs:
                video_for_qbound = (
                    x_raw.detach()
                )
                contextual_for_qbound = (
                    x_bar.detach()
                )
                language_for_qbound = (
                    qbound_language.detach()
                )
            else:
                video_for_qbound = x_raw
                contextual_for_qbound = x_bar
                language_for_qbound = (
                    qbound_language
                )

            qbound_aux = self.qbound(
                video_tokens=video_for_qbound,
                question_features=(
                    language_for_qbound
                ),
                video_mask=valid_mask,
                contextual_tokens=(
                    contextual_for_qbound
                ),
                # V1.2 never needs CRA temporal evidence to define its
                # topology. Passing None guarantees zero prior features even
                # if an old config accidentally enables use_temporal_prior.
                temporal_prior=None,
            )

            required_keys = {
                # Main temporal path.
                "event_temporal_prob",
                "event_temporal_logits",

                # Dynamic-topology diagnostics: these keys distinguish the
                # V1.2 core from the old soft fixed-bin implementation.
                "boundary_hard",
                "boundary_count",
                "event_count",
                "selected_boundary_prob_mean",

                # Weak-alignment-ready visual/event evidence.
                "event_visual_tokens",
                "evidence_visual_token",
                "event_valid_mask",
                "event_relevance_weights",
                "frame_semantic_logits",

                # V1.3 ordered reasoning / multi-event utilization.
                "reasoned_event_tokens",
                "reasoned_event_repr",
                "ordered_relation_gates",
                "ordered_relation_valid_mask",
                "ordered_update_gate",
                "event_relevance_gates",
                "event_relevance_effective_count",

                # Counterfactual diagnostics.
                "merge_event_temporal_prob",
                "merge_transition_selector",
                "selected_transition_strength",
                "selected_transition_mass",
                "removed_boundary_mass",
            }

            missing = sorted(
                required_keys.difference(
                    qbound_aux
                )
            )

            if missing:
                raise KeyError(
                    "Q-Bound V1.3 output is missing: "
                    + ", ".join(missing)
                )

            qbound_time = (
                self.grounding_module
                .finalize_probability_grounding(
                    qbound_aux[
                        "event_temporal_prob"
                    ],
                    valid_mask=valid_mask,
                    window_sizes=window_sizes,
                    detach_sigma=True,
                )
            )

            qbound_key = (
                self._normalize_temporal_distribution(
                    qbound_time["key"],
                    valid_mask,
                )
            )

            # Pure-QBound grounded representation. This deliberately bypasses
            # the CRA->Q-Bound handover and is the representation that the new
            # weak alignment loss should supervise in cra_pipeline.py. Using
            # the mixed final representation here would let CRA solve part of
            # the alignment objective during the warm handover epochs.
            qbound_video_proj = (
                self._project_grounded_video(
                    x_bar=x_bar,
                    temporal_weights=qbound_key,
                    valid_mask=valid_mask,
                    do=do,
                    projection_dropout_mask=(
                        projection_dropout_mask
                    ),
                    x_star_proj=x_star_proj,
                )
            )

            final_key, mix = (
                self._mix_temporal_keys(
                    cra_key=cra_key,
                    qbound_key=qbound_key,
                    qbound_mix=qbound_mix,
                    valid_mask=valid_mask,
                )
            )

        else:
            # Original CRA/VQ behavior.
            qbound_key = cra_key
            final_key = cra_key
            qbound_video_proj = cra_video_proj

            mix = torch.zeros(
                batch_size,
                1,
                dtype=final_key.dtype,
                device=final_key.device,
            )

        # --------------------------------------------------------------
        # 4) Final temporal grounding and main video representation.
        # --------------------------------------------------------------
        time_param = (
            self._build_time_param_from_key(
                final_key=final_key,
                base_temporal=base_temporal,
                qbound_key=qbound_key,
                qbound_mix=mix,
                window_sizes=window_sizes,
            )
        )

        temporal_video_proj = (
            self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=final_key,
                valid_mask=valid_mask,
                do=do,
                projection_dropout_mask=(
                    projection_dropout_mask
                ),
                x_star_proj=x_star_proj,
            )
        )

        if qbound_aux is not None:
            qbound_reasoning_residual = (
                self._project_qbound_reasoning(
                    qbound_aux["reasoned_event_repr"]
                )
            )

            # Main VQA representation. The temporal branch still determines
            # grounded visual content, but V1.3 now gives the ordered event
            # state a direct path to the answer scorer. The residual is
            # introduced with the same deterministic curriculum as Q-Bound.
            video_proj, reasoning_residual_used = (
                self._inject_qbound_reasoning(
                    temporal_video_proj=temporal_video_proj,
                    reasoning_residual=qbound_reasoning_residual,
                    qbound_mix=mix,
                )
            )

            # Pure-QBound V1.3 answer feature, independent of the temporary CRA
            # teacher. This is useful for diagnostics/ablation and later LLM
            # adapters; it is not used by the current weak InfoNCE objective.
            qbound_reasoned_video_proj = (
                qbound_video_proj
                + self.qbound_reasoning_scale
                * qbound_reasoning_residual
            )
        else:
            qbound_reasoning_residual = torch.zeros_like(
                temporal_video_proj
            )
            reasoning_residual_used = torch.zeros_like(
                temporal_video_proj
            )
            qbound_reasoned_video_proj = qbound_video_proj
            video_proj = temporal_video_proj

        # Existing CRA/CMA-compatible feature: this follows the actually-used
        # main VQA path and therefore still follows the deterministic handover.
        # We keep it unchanged for compatibility until cra_pipeline.py is
        # explicitly refactored.
        time_param["grounding_feature"] = (
            video_proj
        )

        # V1.3 weak-alignment feature: PURE Q-Bound, independent of qbound_mix.
        # New Q-Bound alignment losses should use this field (or the event-level
        # features below), NOT grounding_feature, so early CRA teacher mixing
        # cannot satisfy the new objective.
        time_param["qbound_grounding_feature"] = (
            qbound_video_proj
        )

        # V1.3 answer-side features. ``qbound_reasoned_video_feature`` is the
        # pure-QBound endpoint (temporal evidence + ordered-event residual),
        # while ``qbound_temporal_main_feature`` records the actually-used
        # temporal feature before the residual is injected.
        time_param["qbound_reasoned_video_feature"] = (
            qbound_reasoned_video_proj
        )
        time_param["qbound_temporal_main_feature"] = (
            temporal_video_proj
        )
        time_param["qbound_reasoning_residual"] = (
            qbound_reasoning_residual
        )
        time_param["qbound_reasoning_residual_used"] = (
            reasoning_residual_used
        )
        time_param["qbound_reasoning_scale"] = torch.as_tensor(
            self.qbound_reasoning_scale,
            device=video_proj.device,
            dtype=video_proj.dtype,
        )

        # Useful CRA diagnostic only; no new Q-Bound loss should attach here.
        time_param["cra_grounding_feature"] = (
            cra_video_proj
        )

        if qbound_aux is not None:
            # These tensors are backbone-agnostic outputs from QBoundCore. They
            # make weak event/evidence alignment possible in the pipeline while
            # keeping all dataset- and loss-specific logic outside the core.
            time_param["qbound_event_visual_tokens"] = (
                qbound_aux["event_visual_tokens"]
            )
            time_param["qbound_evidence_visual_token"] = (
                qbound_aux["evidence_visual_token"]
            )
            time_param["qbound_event_valid_mask"] = (
                qbound_aux["event_valid_mask"]
            )
            time_param["qbound_event_relevance_weights"] = (
                qbound_aux["event_relevance_weights"]
            )
            time_param["qbound_frame_semantic_logits"] = (
                qbound_aux["frame_semantic_logits"]
            )

            # Native V1.3 reasoning outputs. These stay backbone-independent
            # and will be surfaced by CRA.forward in the next refactor step.
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
            time_param["qbound_event_relevance_gates"] = (
                qbound_aux["event_relevance_gates"]
            )
            time_param["qbound_event_relevance_effective_count"] = (
                qbound_aux["event_relevance_effective_count"]
            )

        # --------------------------------------------------------------
        # 5) Transition-level merge counterfactual.
        # --------------------------------------------------------------
        if qbound_aux is not None:
            (
                merge_video_proj,
                merge_time_param,
                merge_qbound_key,
                merge_final_key,
            ) = self._finalize_qbound_counterfactual(
                merge_temporal_prob=(
                    qbound_aux[
                        "merge_event_temporal_prob"
                    ][:, 0]
                ),
                cra_key=cra_key,
                qbound_mix=mix,
                x_bar=x_bar,
                valid_mask=valid_mask,
                do=do,
                window_sizes=window_sizes,
                projection_dropout_mask=(
                    projection_dropout_mask
                ),
                x_star_proj=x_star_proj,
            )

            # V1.3 note: QBoundCore currently exposes the MERGE temporal
            # counterfactual but not a separately re-reasoned merge event state.
            # To keep the existing merge-CE diagnostic fair, reuse the SAME
            # original ordered-event residual on both original and merge paths.
            # Therefore any current merge harm still measures temporal topology
            # only. A true reasoning counterfactual (merge/reverse/shuffle with
            # recomputed event reasoning) will be added in the pipeline/core
            # diagnostic step rather than silently conflating the two effects.
            merge_temporal_video_proj = merge_video_proj
            if qbound_aux is not None:
                merge_video_proj, _ = self._inject_qbound_reasoning(
                    temporal_video_proj=merge_temporal_video_proj,
                    reasoning_residual=qbound_reasoning_residual,
                    qbound_mix=mix,
                )

            qbound_aux.update(
                {
                    # CRA teacher / pure Q-Bound / actually-used main key.
                    "cra_key": cra_key,
                    "event_key": qbound_key,
                    "qbound_key": qbound_key,
                    "final_key": final_key,
                    "refined_key": final_key,
                    "qbound_mix": (
                        mix.squeeze(-1)
                    ),

                    # Video representations.
                    "cra_video_proj": (
                        cra_video_proj
                    ),
                    "qbound_video_proj": (
                        qbound_video_proj
                    ),
                    "qbound_reasoned_video_proj": (
                        qbound_reasoned_video_proj
                    ),
                    "temporal_main_video_proj": (
                        temporal_video_proj
                    ),
                    "qbound_reasoning_residual": (
                        qbound_reasoning_residual
                    ),
                    "qbound_reasoning_residual_used": (
                        reasoning_residual_used
                    ),
                    "main_video_proj": (
                        video_proj
                    ),

                    # Distinguish the core distribution from the repository
                    # Gaussian-smoothed key actually used for grounding. This
                    # is useful when checking whether smoothing, rather than the
                    # topology itself, moves the peak away from GT evidence.
                    "qbound_pre_smooth_key": (
                        qbound_aux["event_temporal_prob"]
                    ),
                    "qbound_post_smooth_key": (
                        qbound_key
                    ),

                    # Transition-level merge through the SAME handover.
                    "merge_qbound_key": (
                        merge_qbound_key.unsqueeze(1)
                    ),
                    "merge_final_key": (
                        merge_final_key.unsqueeze(1)
                    ),
                    "merge_start": (
                        merge_time_param[
                            "start"
                        ].unsqueeze(1)
                    ),
                    "merge_end": (
                        merge_time_param[
                            "end"
                        ].unsqueeze(1)
                    ),
                    "merge_temporal_video_proj": (
                        merge_temporal_video_proj.unsqueeze(1)
                    ),
                    "merge_video_proj": (
                        merge_video_proj.unsqueeze(1)
                    ),
                    "merge_reasoning_counterfactual": torch.as_tensor(
                        0.0,
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),

                    # Final-path intervention magnitude. During training this
                    # naturally scales with the deterministic handover lambda.
                    "merge_key_l1": (
                        (
                            merge_final_key
                            - final_key
                        )
                        .abs()
                        .mean(dim=-1)
                    ),

                    # Pure Q-Bound intervention magnitude, independent of
                    # handover. This is particularly useful during epochs 1-5.
                    "pure_qbound_merge_key_l1": (
                        (
                            merge_qbound_key
                            - qbound_key
                        )
                        .abs()
                        .mean(dim=-1)
                    ),

                    # Direct CRA-vs-Q-Bound disagreement before handover.
                    "cra_qbound_key_l1": (
                        (
                            qbound_key
                            - cra_key
                        )
                        .abs()
                        .mean(dim=-1)
                    ),

                    "question_only_conditioning": (
                        torch.as_tensor(
                            float(
                                question_feature
                                is not None
                            ),
                            device=video_proj.device,
                            dtype=video_proj.dtype,
                        )
                    ),
                }
            )

        time_param["qbound_aux"] = (
            qbound_aux
        )

        # Compatibility name used elsewhere in the current repository. Keep
        # this as the MAIN representation; the new pure-QBound feature has the
        # explicit qbound_grounding_feature name above.
        time_param["qbound_feature"] = (
            video_proj
        )

        return (
            video_proj,
            final_key,
            time_param,
        )
