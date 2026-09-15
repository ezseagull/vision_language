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
    CRA downstream decoder with Q-Bound V2.1.0.

    V2.1 is a clean architectural break from V2.0.

    The old V2.0 path:
        raw video -> learned cuts -> cut-gated temporal propagation
                  -> independent start/end heads

    is removed because the cuts created an information-flow shortcut and the
    independent endpoints admitted micro-span collapse.

    V2.1 instead uses:
        local CLIP primitives + question tokens
            -> query-conditioned primitive relevance / adjacent relation
            -> one shared variable-scale segment energy E(s,e)
            -> soft P(s,e) and hard H* from the SAME energy

    Scientific separation
    ---------------------
    * Q-Bound receives RAW / generic local visual features, never CRA x_bar.
    * CRA x_bar + CRA grounding are kept only as a detached reference/testbed.
    * The answer-space visual evidence for active Q-Bound is pooled from
      Q-Bound's temporally local ``content_tokens`` -- NOT from globally
      contextualized CRA x_bar.  This prevents a short selected span from
      secretly carrying whole-video context.
    * GT answers / answer candidates are never inputs to Q-BoundCore.
    * V2.0 functional KEEP/MERGE and boundary-equilibrium verifiers are removed.

    Training / inference alignment
    ------------------------------
    ``main_temporal_prob`` is the hard H* distribution used for inference.
    ``training_temporal_prob`` has exactly the same forward numeric value but
    uses the soft structured segment posterior in backward.  Consequently the
    downstream QA loss can train the shared segment energy without restoring an
    independent soft inference path.
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

        self.embed_dim = int(embed_dim)
        self.cfgs = cfgs
        self.qbound_version = "2.1.0"

        # ------------------------------------------------------------------
        # CRA downstream/reference branch.
        # ------------------------------------------------------------------
        self.layers = clones(
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout),
            num_layer,
        )
        self.norm = LayerNorm(embed_dim)

        # Kept for CRA checkpoint compatibility.
        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        # Keep original CRA parameter names for checkpoint compatibility.
        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        model_cfg = self.cfgs["model"]
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        configured_qbound_version = str(
            qbound_cfg.get("version", self.qbound_version)
        )
        if configured_qbound_version != self.qbound_version:
            raise ValueError(
                "Refine decoder/core version mismatch: expected "
                f"model.qbound.version={self.qbound_version!r}, got "
                f"{configured_qbound_version!r}."
            )

        self.grounding_module = GroundingModule(
            d_model=embed_dim,
            dropout=model_cfg["dropout"],
            initial_sigma=model_cfg["sigma"],
            window_sizes=model_cfg["window_sizes"],
        )
        self.do = VisualIntervention(self.cfgs)

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
        # V2.1 is explicitly designed around token-level question injection.
        self.qbound_require_token_question = bool(
            qbound_cfg.get("require_token_question", True)
        )
        self.qbound_detach_backbone_inputs = bool(
            qbound_cfg.get(
                "detach_backbone_inputs",
                model_cfg.get(
                    "qbound_detach_backbone_inputs",
                    model_cfg.get("gpeb_detach_backbone_inputs", True),
                ),
            )
        )

        self.qbound_hidden_dim = None
        if self.qbound_enabled:
            self.qbound_hidden_dim = int(
                qbound_cfg.get(
                    "hidden_dim",
                    model_cfg.get("qbound_hidden_dim", 256),
                )
            )

            # Fail fast if an old V2.0 configuration is accidentally reused.
            # These keys are not harmless hyperparameters: they belong to
            # mechanisms that V2.1 deliberately removes.
            forbidden_v20_keys = {
                "input_adapter_use_position",
                "cut_temperature",
                "cut_threshold",
                "question_residual_scale",
                "num_local_reasoning_layers",
            }
            stale = sorted(
                key for key in forbidden_v20_keys if key in qbound_cfg
            )
            if stale:
                raise ValueError(
                    "Q-Bound V2.1 config still contains removed V2.0 keys: "
                    + ", ".join(stale)
                    + ". Remove them rather than silently carrying V2.0 "
                      "mechanisms into V2.1."
                )

            self.qbound = QBoundCore(
                input_dim=embed_dim,
                hidden_dim=self.qbound_hidden_dim,
                dropout=float(
                    qbound_cfg.get(
                        "dropout",
                        model_cfg.get("qbound_dropout", dropout),
                    )
                ),
                use_generic_input_adapter=bool(
                    qbound_cfg.get("use_generic_input_adapter", True)
                ),
                input_adapter_dropout=float(
                    qbound_cfg.get(
                        "input_adapter_dropout",
                        qbound_cfg.get(
                            "dropout",
                            model_cfg.get("qbound_dropout", dropout),
                        ),
                    )
                ),
                num_question_heads=int(
                    qbound_cfg.get("num_question_heads", 4)
                ),
                question_ff_multiplier=int(
                    qbound_cfg.get("question_ff_multiplier", 2)
                ),
                segment_temperature=float(
                    qbound_cfg.get("segment_temperature", 1.0)
                ),
                relation_weight=float(
                    qbound_cfg.get("relation_weight", 0.50)
                ),
                boundary_weight=float(
                    qbound_cfg.get("boundary_weight", 0.25)
                ),
            )

            if str(getattr(self.qbound, "qbound_version", "")) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={getattr(self.qbound, 'qbound_version', None)!r}."
                )

    # ------------------------------------------------------------------
    # Shape / normalization helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_video_mask(mask, batch_size, time_steps, device):
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

        mask = mask.to(device=device).bool()
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("Every sample must contain at least one valid video token.")
        return mask

    @staticmethod
    def _normalize_question_mask(mask, question_feature, device):
        if mask is None:
            return None

        if question_feature.ndim != 3:
            raise ValueError(
                "question_mask is only meaningful with token-level "
                "question_feature=[B,L,D]."
            )

        batch_size, q_len, _ = question_feature.shape
        while mask.ndim > 2 and mask.size(1) == 1:
            mask = mask.squeeze(1)

        if mask.ndim != 2 or mask.shape != (batch_size, q_len):
            raise ValueError(
                "question_mask must be [B,L]; expected "
                f"{(batch_size, q_len)}, got {tuple(mask.shape)}."
            )

        mask = mask.to(device=device).bool()
        if not torch.all(mask.any(dim=-1)):
            raise ValueError(
                "Every sample must contain at least one valid question token."
            )
        return mask

    @staticmethod
    def _normalize_temporal_distribution(values, valid_mask, eps=1e-6):
        if values.ndim == 3 and values.size(-1) == 1:
            values = values.squeeze(-1)
        if values.shape != valid_mask.shape:
            raise ValueError(
                "Temporal distribution and valid mask must have the same "
                f"[B,T] shape; got {tuple(values.shape)} and "
                f"{tuple(valid_mask.shape)}."
            )

        values = values.clamp_min(0.0)
        values = values * valid_mask.to(values.dtype)
        denominator = values.sum(dim=-1, keepdim=True)
        normalized = values / denominator.clamp_min(eps)

        fallback = valid_mask.to(values.dtype)
        fallback = fallback / fallback.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        return torch.where(denominator > eps, normalized, fallback)

    @staticmethod
    def _masked_mean_video(features, valid_mask):
        weights = valid_mask.to(features.dtype).unsqueeze(-1)
        return (
            (features * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )

    # ------------------------------------------------------------------
    # Shared projection helpers
    # ------------------------------------------------------------------
    def _sample_projection_dropout_mask(self, reference):
        dropout_layer = self.video_proj[0]
        if not self.training or dropout_layer.p <= 0.0:
            return torch.ones_like(reference)

        keep_prob = 1.0 - float(dropout_layer.p)
        if keep_prob <= 0.0:
            raise ValueError("video projection dropout must be < 1.")
        mask = torch.empty_like(reference).bernoulli_(keep_prob)
        return mask / keep_prob

    def _apply_video_projection(self, pooled, dropout_mask):
        if pooled.shape != dropout_mask.shape:
            raise ValueError(
                "pooled representation and dropout mask must have the same "
                f"shape; got {tuple(pooled.shape)} and "
                f"{tuple(dropout_mask.shape)}."
            )
        return self.video_proj[1](pooled * dropout_mask)

    def _project_temporal_sequence(
        self,
        sequence,
        temporal_weights,
        projection_dropout_mask,
        do=False,
        x_star_proj=None,
    ):
        if sequence.ndim != 3:
            raise ValueError(
                "Temporal evidence sequence must be [B,T,D], got "
                f"{tuple(sequence.shape)}."
            )
        if temporal_weights.shape != sequence.shape[:2]:
            raise ValueError(
                "Temporal weights must match evidence [B,T]; got "
                f"{tuple(temporal_weights.shape)} for "
                f"{tuple(sequence.shape)}."
            )

        pooled = torch.sum(
            sequence * temporal_weights.unsqueeze(-1),
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
            projected = self.do(x_star_proj, projected)
        return projected

    # ------------------------------------------------------------------
    # V2.0 verifier methods are intentionally invalid in V2.1.
    # Leaving explicit fail-fast stubs catches stale pipeline code.
    # ------------------------------------------------------------------
    def score_functional_composition_pairs(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "score_functional_composition_pairs() was removed in Q-Bound V2.1. "
            "V2.1 has no functional KEEP/MERGE verifier. Update cra_pipeline.py."
        )

    def score_boundary_equilibrium_views(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "score_boundary_equilibrium_views() was removed in Q-Bound V2.1. "
            "V2.1 has no boundary-equilibrium verifier. Update cra_pipeline.py."
        )

    # ------------------------------------------------------------------
    # Temporal contract
    # ------------------------------------------------------------------
    def _build_time_param(
        self,
        final_key,
        training_key,
        base_temporal,
        qbound_active,
        qbound_time=None,
    ):
        if qbound_active:
            if qbound_time is None:
                raise ValueError(
                    "qbound_time is required for active Q-Bound V2.1."
                )
            time_param = dict(qbound_time)
            time_param["key"] = final_key
        else:
            time_param = self.grounding_module.gen_grounding(
                final_key,
                window_sizes=self.cfgs["model"].get(
                    "window_sizes", [1, 3, 5]
                ),
            )

        time_param["ori_key"] = base_temporal["ori_key"]
        time_param["cra_key"] = base_temporal["cra_key"]
        time_param["qbound_key"] = final_key
        time_param["qbound_training_key"] = training_key
        time_param["qbound_mix"] = torch.full(
            (final_key.size(0),),
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_handover_enabled"] = torch.zeros(
            (), dtype=final_key.dtype, device=final_key.device
        )
        time_param["qbound_smoothing_applied"] = torch.zeros(
            (), dtype=final_key.dtype, device=final_key.device
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
        qbound_video_tokens=None,
        qbound_mix=1.0,
        use_gpeb=None,
        question_mask=None,
    ):
        del gsub
        del qbound_mix
        del window_sizes

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # --------------------------------------------------------------
        # CRA branch is retained only as the downstream benchmark/reference.
        # --------------------------------------------------------------
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

        qbound_active = bool(self.qbound_enabled and use_qbound)
        if qbound_active and do:
            raise RuntimeError(
                "Controlled Q-Bound V2.1 requires model.do=false. "
                "VisualIntervention would confound the temporal-composition "
                "experiment."
            )

        # CRA temporal grounding remains a detached reference while Q-Bound
        # is active.  It cannot teach or shape Q-Bound's segment structure.
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

        projection_reference = self._masked_mean_video(
            x_bar, valid_mask
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

        cra_video_proj = self._project_temporal_sequence(
            sequence=x_bar,
            temporal_weights=cra_key,
            projection_dropout_mask=projection_dropout_mask,
            do=do,
            x_star_proj=x_star_proj,
        )

        qbound_aux = None
        qbound_time = None

        if qbound_active:
            # ----------------------------------------------------------
            # Q-Bound input contract.
            # ----------------------------------------------------------
            if question_feature is None and self.qbound_require_question_only:
                raise ValueError(
                    "Q-Bound V2.1 requires question_feature. "
                    "Do not reuse candidate-answer features as Q-Bound input."
                )

            qbound_language = (
                question_feature if question_feature is not None else qa
            )
            if self.qbound_require_token_question and qbound_language.ndim != 3:
                raise ValueError(
                    "Q-Bound V2.1 requires token-level question features "
                    "[B,L,D]. Pass LanModel.encode_question(...)[1], not only "
                    "the pooled question vector."
                )

            if qbound_video_tokens is None:
                raise ValueError(
                    "Q-Bound V2.1 requires qbound_video_tokens=[B,T,D] "
                    "containing the raw/generic local visual sequence."
                )
            if qbound_video_tokens.ndim != 3:
                raise ValueError(
                    "qbound_video_tokens must be [B,T,D], got "
                    f"{tuple(qbound_video_tokens.shape)}."
                )
            if qbound_video_tokens.shape[:2] != (batch_size, time_steps):
                raise ValueError(
                    "qbound_video_tokens batch/time mismatch: expected "
                    f"{(batch_size, time_steps)}, got "
                    f"{tuple(qbound_video_tokens.shape[:2])}."
                )
            if qbound_video_tokens.size(-1) != self.embed_dim:
                raise ValueError(
                    "Controlled V2.1 currently expects generic visual "
                    f"dimension {self.embed_dim}; got "
                    f"{qbound_video_tokens.size(-1)}."
                )

            qbound_video_tokens = qbound_video_tokens.to(
                device=x_bar.device,
                dtype=x_bar.dtype,
            )
            qbound_language = qbound_language.to(
                device=x_bar.device,
                dtype=x_bar.dtype,
            )

            normalized_question_mask = self._normalize_question_mask(
                question_mask,
                qbound_language,
                x_bar.device,
            )

            if self.qbound_detach_backbone_inputs:
                video_for_qbound = qbound_video_tokens.detach()
                language_for_qbound = qbound_language.detach()
            else:
                video_for_qbound = qbound_video_tokens
                language_for_qbound = qbound_language

            qbound_aux = self.qbound(
                video_tokens=video_for_qbound,
                question_features=language_for_qbound,
                video_mask=valid_mask,
                question_mask=normalized_question_mask,
            )

            required_core_keys = {
                "qbound_version",
                "latent_segment_composition_enabled",
                "question_token_interaction_enabled",
                "segment_energy_enabled",
                "fixed_event_partition_enabled",
                "dynamic_event_count_enabled",
                "event_count_cap_enabled",
                "cut_field_enabled",
                "boundary_gated_reasoning_enabled",
                "independent_endpoint_heads_enabled",
                "hypothesis_competition_enabled",
                "teacher_distillation_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",
                "content_tokens",
                "primitive_states",
                "query_conditioned_states",
                "node_relevance",
                "adjacent_relation",
                "segment_energy",
                "segment_prob",
                "segment_entropy_ratio",
                "segment_top1_prob",
                "soft_temporal_prob",
                "training_temporal_prob",
                "main_temporal_prob",
                "main_temporal_logits",
                "main_soft_support",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_mask",
                "selected_span_frame_weights",
                "selected_span_length",
                "selected_span_fraction",
                "soft_expected_span_length",
                "soft_expected_span_fraction",
                "single_frame_prob_mass",
                "short_span_prob_mass",
                "selected_single_frame_indicator",
                "selected_full_video_indicator",
            }
            missing = sorted(required_core_keys.difference(qbound_aux))
            if missing:
                raise KeyError(
                    "Q-Bound V2.1 core output is missing: "
                    + ", ".join(missing)
                )

            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore V2.1 runtime version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )

            def _flag(name):
                value = qbound_aux[name]
                if not torch.is_tensor(value):
                    raise TypeError(
                        f"Q-Bound flag {name!r} must be a tensor."
                    )
                return float(value.detach().item())

            # Positive V2.1 requirements.
            if _flag("latent_segment_composition_enabled") != 1.0:
                raise RuntimeError(
                    "V2.1 requires latent segment composition."
                )
            if _flag("question_token_interaction_enabled") != 1.0:
                raise RuntimeError(
                    "V2.1 requires token-level question interaction."
                )
            if _flag("segment_energy_enabled") != 1.0:
                raise RuntimeError(
                    "V2.1 requires one shared segment-energy model."
                )

            # Removed mechanisms must stay off.
            forbidden_active_flags = {
                "fixed_event_partition_enabled",
                "dynamic_event_count_enabled",
                "event_count_cap_enabled",
                "cut_field_enabled",
                "boundary_gated_reasoning_enabled",
                "independent_endpoint_heads_enabled",
                "hypothesis_competition_enabled",
                "teacher_distillation_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",
            }
            for name in forbidden_active_flags:
                if _flag(name) != 0.0:
                    raise RuntimeError(
                        f"Q-Bound V2.1 requires {name}=0."
                    )

            # ----------------------------------------------------------
            # One shared soft/hard segment model.
            # ----------------------------------------------------------
            final_key = self._normalize_temporal_distribution(
                qbound_aux["main_temporal_prob"],
                valid_mask,
            )
            qbound_key = final_key

            training_key = self._normalize_temporal_distribution(
                qbound_aux["training_temporal_prob"],
                valid_mask,
            )
            soft_key = self._normalize_temporal_distribution(
                qbound_aux["soft_temporal_prob"],
                valid_mask,
            )

            qbound_time = self.grounding_module.gen_adaptive_grounding(
                final_key,
                selected_span_start_frame=qbound_aux[
                    "selected_span_start_frame"
                ],
                selected_span_end_frame=qbound_aux[
                    "selected_span_end_frame"
                ],
                valid_mask=valid_mask,
                selected_frame_mask=qbound_aux[
                    "selected_span_frame_mask"
                ],
            )
            qbound_time["pre_smooth_key"] = final_key
            qbound_time["post_smooth_key"] = final_key
            qbound_time["qbound_key"] = final_key
            qbound_time["qbound_training_key"] = training_key
            qbound_time["qbound_soft_key"] = soft_key
            qbound_time["qbound_smoothing_applied"] = torch.zeros(
                (), dtype=final_key.dtype, device=final_key.device
            )

            # ----------------------------------------------------------
            # CRITICAL V2.1 anti-leak design:
            # Answer evidence is pooled from Q-Bound's temporally LOCAL
            # content tokens, not from globally contextualized CRA x_bar.
            # ----------------------------------------------------------
            local_evidence_sequence = qbound_aux["content_tokens"]
            if local_evidence_sequence.shape != qbound_video_tokens.shape:
                raise ValueError(
                    "Q-Bound content_tokens must preserve [B,T,D] for local "
                    "answer evidence; expected "
                    f"{tuple(qbound_video_tokens.shape)}, got "
                    f"{tuple(local_evidence_sequence.shape)}."
                )

            qbound_hard_video_proj = self._project_temporal_sequence(
                sequence=local_evidence_sequence,
                temporal_weights=final_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            # Forward value is numerically the same as hard H*, but gradients
            # pass through the soft structured posterior.
            qbound_training_video_proj = self._project_temporal_sequence(
                sequence=local_evidence_sequence,
                temporal_weights=training_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            # True soft evidence is exposed for diagnostics / optional
            # answer-only auxiliary supervision in the next integration step.
            qbound_soft_video_proj = self._project_temporal_sequence(
                sequence=local_evidence_sequence,
                temporal_weights=soft_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            # Official downstream answer feature.  Forward = hard H*;
            # backward = structured soft segment posterior.
            qbound_video_proj = qbound_training_video_proj
            video_proj = qbound_video_proj

            # The ST feature should match the hard feature numerically in
            # forward. Log any accidental divergence explicitly.
            st_hard_max_abs_diff = (
                qbound_training_video_proj
                - qbound_hard_video_proj
            ).abs().amax(dim=-1)

            qbound_aux["qbound_local_evidence_sequence"] = (
                local_evidence_sequence
            )
            qbound_aux["qbound_hard_video_proj"] = qbound_hard_video_proj
            qbound_aux["qbound_training_video_proj"] = (
                qbound_training_video_proj
            )
            qbound_aux["qbound_soft_video_proj"] = qbound_soft_video_proj
            qbound_aux["qbound_st_hard_feature_max_abs_diff"] = (
                st_hard_max_abs_diff
            )
            qbound_aux["qbound_question_mask_provided"] = torch.as_tensor(
                float(normalized_question_mask is not None),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )

            # Explicit anti-shortcut provenance flags.
            qbound_aux["qbound_input_is_generic"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_input_uses_cra_embedding"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_answer_evidence_uses_local_tokens"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_answer_evidence_uses_cra_context"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["functional_verifier_used"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["boundary_equilibrium_used"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["teacher_distribution_used"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )

            # Downstream/reference aliases only.
            qbound_aux["cra_key"] = cra_key
            qbound_aux["qbound_key"] = final_key
            qbound_aux["qbound_training_key"] = training_key
            qbound_aux["qbound_soft_key"] = soft_key
            qbound_aux["final_key"] = final_key
            qbound_aux["main_video_proj"] = qbound_video_proj
            qbound_aux["qbound_video_proj"] = qbound_video_proj
            qbound_aux["cra_video_proj"] = cra_video_proj
        else:
            final_key = cra_key
            training_key = cra_key
            qbound_key = cra_key
            video_proj = cra_video_proj

        time_param = self._build_time_param(
            final_key=final_key,
            training_key=training_key,
            base_temporal=base_temporal,
            qbound_active=qbound_active,
            qbound_time=qbound_time,
        )

        # Official downstream answer-space feature.
        time_param["grounding_feature"] = video_proj
        time_param["qbound_feature"] = video_proj
        time_param["qbound_grounding_feature"] = video_proj
        time_param["qbound_temporal_main_feature"] = video_proj
        time_param["cra_grounding_feature"] = cra_video_proj

        # No direct hidden-event residual into the answer representation.
        zero_residual = torch.zeros_like(video_proj)
        time_param["qbound_reasoning_residual"] = zero_residual
        time_param["qbound_reasoning_residual_used"] = zero_residual
        time_param["qbound_reasoning_scale"] = torch.zeros(
            (), dtype=video_proj.dtype, device=video_proj.device
        )
        time_param["qbound_direct_reasoning_residual_enabled"] = torch.zeros(
            (), dtype=video_proj.dtype, device=video_proj.device
        )

        if qbound_aux is not None:
            # Prefix the complete V2.1 interface into time_param.
            for key, value in qbound_aux.items():
                if key == "qbound_version":
                    continue
                time_param[f"qbound_{key}"] = value

            time_param["qbound_version"] = self.qbound_version

            # Main exact-span aliases used by metrics / wrappers.
            time_param["qbound_selected_span_start_frame"] = qbound_aux[
                "selected_span_start_frame"
            ]
            time_param["qbound_selected_span_end_frame"] = qbound_aux[
                "selected_span_end_frame"
            ]
            time_param["qbound_selected_span_mask"] = qbound_aux[
                "selected_span_frame_mask"
            ]
            time_param["qbound_selected_span_length"] = qbound_aux[
                "selected_span_length"
            ]
            time_param["qbound_selected_span_fraction"] = qbound_aux[
                "selected_span_fraction"
            ]
            time_param["qbound_soft_expected_span_length"] = qbound_aux[
                "soft_expected_span_length"
            ]
            time_param["qbound_soft_expected_span_fraction"] = qbound_aux[
                "soft_expected_span_fraction"
            ]

            # Generic evaluation aliases. These refer to the same V2.1 hard
            # segment; no V1.x refiner/calibration machinery is restored.
            time_param["qbound_calibrated_video_proj"] = video_proj
            time_param["qbound_calibrated_key"] = final_key
            time_param["qbound_calibrated_start"] = time_param["start"]
            time_param["qbound_calibrated_end"] = time_param["end"]
            time_param["qbound_calibrated_mask"] = time_param["mask"]
            time_param["qbound_calibrated_span_length"] = qbound_aux[
                "selected_span_length"
            ]
            time_param["qbound_calibrated_span_fraction"] = qbound_aux[
                "selected_span_fraction"
            ]

            # Architecture flags for fail-fast integration checks.
            architecture_flags = (
                "latent_segment_composition_enabled",
                "question_token_interaction_enabled",
                "segment_energy_enabled",
                "fixed_event_partition_enabled",
                "dynamic_event_count_enabled",
                "event_count_cap_enabled",
                "cut_field_enabled",
                "boundary_gated_reasoning_enabled",
                "independent_endpoint_heads_enabled",
                "hypothesis_competition_enabled",
                "teacher_distillation_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",
            )
            for name in architecture_flags:
                time_param[f"qbound_{name}"] = qbound_aux[name]

            qbound_aux["qbound_handover_enabled"] = torch.zeros(
                (), dtype=video_proj.dtype, device=video_proj.device
            )
            qbound_aux["qbound_direct_reasoning_residual_enabled"] = torch.zeros(
                (), dtype=video_proj.dtype, device=video_proj.device
            )
            qbound_aux["qbound_reasoning_residual"] = zero_residual
            qbound_aux["qbound_reasoning_residual_used"] = zero_residual

        time_param["qbound_aux"] = qbound_aux

        return (
            video_proj,
            final_key,
            time_param,
        )
