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
    CRA refinement decoder with Q-Bound Version 1.1.1.

    V1.1.1 removes the two shortcut gates used in V1.1:

        z_final = z_grounded + alpha * A(z_event)   [removed]
        l_final = l_CRA + beta * l_event            [removed]

    Q-Bound now has exactly one route into the main task:

        question + video
            -> Q-Bound event topology
            -> event temporal evidence
            -> CRA temporal logits + Q-Bound temporal logits
            -> final grounding key
            -> grounded video representation
            -> main VQA scorer

    This prevents the model from improving VQA through a global event
    representation while ignoring the learned topology, and prevents beta
    from collapsing to zero.

    Counterfactual training uses only transition-level MERGE from QBoundCore
    V1.1.1.  Original and merge paths share the same CRA projection dropout
    realization and the same visual-intervention reference, so measured task
    harm reflects the topology intervention rather than stochastic dropout.
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

        # Keep the original parameter names (video_proj.1.weight/bias) so
        # existing CRA checkpoints remain load-compatible.
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
        # Q-Bound V1.1.1
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
            # Constructor matches QBoundCore V1.1.1 exactly.
            # V1.1 parameters counterfactual_temperature / merge_strength /
            # split_strength no longer exist.
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
                min_event_mass=qbound_cfg.get(
                    "min_event_mass",
                    model_cfg.get(
                        "qbound_min_event_mass",
                        model_cfg.get("gpeb_min_event_mass", 0.25),
                    ),
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

    def _sample_projection_dropout_mask(
        self,
        reference,
    ):
        """
        Sample ONE CRA video-projection dropout realization.

        The same mask is reused for original and merge representations.
        This preserves the marginal behavior of nn.Dropout while preventing
        counterfactual task harm from being caused by different random masks.
        """
        dropout_layer = self.video_proj[0]

        if (
            not self.training
            or dropout_layer.p <= 0.0
        ):
            return torch.ones_like(reference)

        keep_prob = 1.0 - float(dropout_layer.p)
        if keep_prob <= 0.0:
            raise ValueError(
                "video projection dropout must be < 1."
            )

        mask = torch.empty_like(reference).bernoulli_(keep_prob)
        return mask / keep_prob

    def _apply_video_projection(
        self,
        pooled,
        dropout_mask,
    ):
        """
        Apply the original Dropout -> Linear projection with an externally
        supplied dropout realization. Parameter names/checkpoint layout stay
        unchanged because the Linear remains self.video_proj[1].
        """
        if pooled.shape != dropout_mask.shape:
            raise ValueError(
                "pooled representation and dropout mask must have the "
                f"same shape, got {tuple(pooled.shape)} and "
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
                    "x_star_proj is required when visual intervention is on."
                )
            projected = self.do(
                x_star_proj,
                projected,
            )

        return projected

    @staticmethod
    def _build_temporal_bias(
        event_temporal_logits,
        valid_mask,
    ):
        """
        Direct Q-Bound temporal contribution in V1.1.1.

        There is deliberately NO beta gate. QBoundCore initializes its final
        temporal scorer near zero, so the correction starts small but receives
        main-task gradient directly.
        """
        if event_temporal_logits.shape != valid_mask.shape:
            raise ValueError(
                "event_temporal_logits must be [B,T] and match valid_mask. "
                f"Got {tuple(event_temporal_logits.shape)} and "
                f"{tuple(valid_mask.shape)}."
            )

        return (
            event_temporal_logits
            * valid_mask.to(event_temporal_logits.dtype)
        )

    def _finalize_counterfactual(
        self,
        base_temporal,
        x_bar,
        valid_mask,
        do,
        event_temporal_logits,
        window_sizes,
        projection_dropout_mask,
        x_star_proj,
    ):
        """
        Run the transition-level merge through exactly the same grounding and
        video-projection path as the original topology.

        No bridge representation and no alpha residual are used.
        """
        temporal_bias = self._build_temporal_bias(
            event_temporal_logits,
            valid_mask,
        )

        cf_time_param = self.grounding_module.finalize_grounding(
            base_temporal,
            temporal_bias=temporal_bias,
            window_sizes=window_sizes,
        )

        cf_key = self._normalize_temporal_distribution(
            cf_time_param["key"],
            valid_mask,
        )
        cf_time_param["key"] = cf_key
        cf_time_param["refined_key"] = cf_key

        projected = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=cf_key,
            valid_mask=valid_mask,
            do=do,
            projection_dropout_mask=projection_dropout_mask,
            x_star_proj=x_star_proj,
        )

        return projected, cf_time_param, temporal_bias

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
        # Temporary compatibility alias for an older direct Decoder caller.
        use_gpeb=None,
    ):
        del gsub

        if use_gpeb is not None:
            use_qbound = bool(
                use_qbound or use_gpeb
            )

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
        # 1) CRA temporal evidence, computed once.
        # --------------------------------------------------------------
        base_temporal = self.grounding_module.compute_base_temporal(
            x_bar,
            qa,
            valid_mask=valid_mask,
        )

        cra_key = self._normalize_temporal_distribution(
            base_temporal["cra_key"],
            valid_mask,
        )

        # One shared stochastic projection context for CRA / original / merge.
        projection_reference = self._masked_mean_video(
            x_bar,
            valid_mask,
        )
        projection_dropout_mask = (
            self._sample_projection_dropout_mask(
                projection_reference
            )
        )

        x_star_proj = None
        if do:
            # x_star is topology-independent, so compute it exactly once and
            # reuse it for CRA, original, and merge intervention paths.
            x_star_dropout_mask = (
                self._sample_projection_dropout_mask(
                    projection_reference
                )
            )
            x_star_proj = self._apply_video_projection(
                projection_reference,
                x_star_dropout_mask,
            )

        # CRA-only representation is retained only for the original CRA
        # alignment objective and diagnostics.
        cra_video_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=cra_key,
            valid_mask=valid_mask,
            do=do,
            projection_dropout_mask=projection_dropout_mask,
            x_star_proj=x_star_proj,
        )

        qbound_aux = None
        temporal_bias = torch.zeros_like(
            base_temporal["cra_temporal_logits"]
        )

        # --------------------------------------------------------------
        # 2) Question-conditioned Q-Bound topology.
        # --------------------------------------------------------------
        if self.qbound_enabled and use_qbound:
            if (
                question_feature is None
                and self.qbound_require_question_only
            ):
                raise ValueError(
                    "Q-Bound V1.1.1 requires question_feature=[B,D]. "
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
                # QBoundCore detaches this internally. With
                # use_temporal_prior:false it becomes exact zeros.
                temporal_prior=cra_key,
            )

            required_keys = {
                "event_temporal_logits",
                "frame_event_relevance",
                "merge_event_temporal_logits",
                "merge_transition_selector",
                "selected_transition_strength",
                "selected_transition_mass",
                "removed_boundary_mass",
            }
            missing = sorted(
                required_keys.difference(qbound_aux)
            )
            if missing:
                raise KeyError(
                    "Q-Bound V1.1.1 output is missing: "
                    + ", ".join(missing)
                )

            temporal_bias = self._build_temporal_bias(
                qbound_aux["event_temporal_logits"],
                valid_mask,
            )

        # --------------------------------------------------------------
        # 3) Final main grounding.
        #
        # Q-Bound has ONE route into the task:
        #   CRA logits + event temporal logits -> final key -> main video.
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

        video_proj = self._project_grounded_video(
            x_bar=x_bar,
            temporal_weights=final_key,
            valid_mask=valid_mask,
            do=do,
            projection_dropout_mask=projection_dropout_mask,
            x_star_proj=x_star_proj,
        )

        # Preserve the original CRA alignment target. Q-Bound does not alter
        # the target used by the existing CRA alignment objective.
        time_param["grounding_feature"] = cra_video_proj

        # --------------------------------------------------------------
        # 4) Transition-level merge counterfactual through the SAME path.
        # --------------------------------------------------------------
        if qbound_aux is not None:
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
                    event_temporal_logits=(
                        qbound_aux[
                            "merge_event_temporal_logits"
                        ][:, 0]
                    ),
                    window_sizes=window_sizes,
                    projection_dropout_mask=projection_dropout_mask,
                    x_star_proj=x_star_proj,
                )
            )

            qbound_aux.update(
                {
                    # Original/main path.
                    "cra_key": cra_key,
                    "event_key": event_key,
                    "final_key": final_key,
                    "refined_key": final_key,
                    "temporal_bias": temporal_bias,
                    "cra_video_proj": cra_video_proj,
                    "main_video_proj": video_proj,

                    # Transition-level merge through the same main path.
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
                    "merge_video_proj": (
                        merge_video_proj.unsqueeze(1)
                    ),

                    # Useful direct diagnostic of how much the topology
                    # intervention changed final temporal grounding.
                    "merge_key_l1": (
                        (
                            merge_time_param["key"]
                            - final_key
                        )
                        .abs()
                        .mean(dim=-1)
                    ),

                    "question_only_conditioning": torch.as_tensor(
                        float(question_feature is not None),
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),
                }
            )

        time_param["qbound_aux"] = qbound_aux
        time_param["qbound_feature"] = video_proj

        return video_proj, final_key, time_param
