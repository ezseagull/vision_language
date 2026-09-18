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
    CRA downstream decoder with Q-Bound V2.4.0.

    V2.4 removes the V2.3 two-stage coarse->BoundaryRefiner design.

    Final Q-Bound path:
        raw/generic CLIP video + question tokens
            -> QBoundCore
            -> temporal context + early Q->V conditioning
            -> Evidence Core a_t + Event Continuity j_t
            -> Core-to-Closure P_Q(s,e)
            -> hard MAP span for official downstream QA
            -> soft frame marginal for Q-Bound grounding supervision.

    Separation contract
    -------------------
    * Q-Bound receives only video + question.
    * Q-Bound never receives answer candidates or GT answer.
    * Q-Bound never receives temporal GT / IoU / IoP targets.
    * CRA x_bar / CRA grounding remain a detached reference while Q-Bound is active.
    * The official CRA QA feature is pooled with the HARD Q-Bound MAP key.
      Therefore the downstream VQA loss does not backpropagate through Q-Bound
      temporal topology.
    * The differentiable SOFT Q-Bound evidence is exposed separately for the
      Video-Question grounding objective implemented downstream in cra_pipeline.py.
    * The evidence values are the original/light raw visual tokens returned by
      QBoundCore, not the Q-conditioned WHERE states.

    No BoundaryRefiner, local candidate bank, answer critic, adaptive radius,
    learned routing gate, independent endpoint heads, Gaussian smoothing, or
    length prior is used here.
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
        self.qbound_version = "2.4.0"

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

        # Preserve original CRA parameter names for checkpoint compatibility.
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

        # V2.4 is strictly question-only for temporal localization.
        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get(
                    "qbound_require_question_only",
                    True,
                ),
            )
        )
        if self.qbound_enabled and not self.qbound_require_question_only:
            raise ValueError(
                "Q-Bound V2.4 requires question-only temporal input. "
                "Set model.qbound.require_question_only=true."
            )

        self.qbound_require_token_question = bool(
            qbound_cfg.get("require_token_question", True)
        )
        if self.qbound_enabled and not self.qbound_require_token_question:
            raise ValueError(
                "Q-Bound V2.4 requires token-level question features for early "
                "Q->V cross-attention. Set require_token_question=true."
            )

        self.qbound_detach_backbone_inputs = bool(
            qbound_cfg.get(
                "detach_backbone_inputs",
                model_cfg.get(
                    "qbound_detach_backbone_inputs",
                    True,
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

            # V2.4 deliberately removes the following historical mechanisms.
            # Fail fast instead of silently carrying stale config into the new
            # experiment.
            forbidden_keys = {
                # V2.0 / V2.1 leftovers.
                "input_adapter_use_position",
                "cut_temperature",
                "cut_threshold",
                "question_residual_scale",
                "num_local_reasoning_layers",

                # V2.3 input-adapter / segment-composition knobs removed in V2.4.
                "use_generic_input_adapter",
                "input_adapter_dropout",
                "relation_weight",
                "boundary_weight",

                # V2.3 exact-boundary stage.
                "boundary_refiner",
                "adaptive_radius_ratio",
                "min_radius",
                "joint_scoring",

                # Removed regularizers / topology controls.
                "length_prior",
                "minimality_weight",
                "entropy_weight",
                "sparsity_weight",
                "gaussian_sigma",
                "event_count",
            }

            stale = sorted(
                key for key in forbidden_keys if key in qbound_cfg
            )
            if stale:
                raise ValueError(
                    "Q-Bound V2.4 config contains removed/stale keys: "
                    + ", ".join(stale)
                    + ". Remove them rather than silently reintroducing older "
                      "mechanisms."
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
                num_question_heads=int(
                    qbound_cfg.get("num_question_heads", 4)
                ),
                question_ff_multiplier=int(
                    qbound_cfg.get("question_ff_multiplier", 2)
                ),
                segment_temperature=float(
                    qbound_cfg.get("segment_temperature", 1.0)
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
            raise ValueError(
                "Every sample must contain at least one valid video token."
            )
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

        return torch.where(
            denominator > eps,
            normalized,
            fallback,
        )

    @staticmethod
    def _masked_mean_video(features, valid_mask):
        weights = valid_mask.to(features.dtype).unsqueeze(-1)
        return (
            (features * weights).sum(dim=1)
            / weights.sum(dim=1).clamp_min(1.0)
        )

    @staticmethod
    def _pool_temporal_sequence(sequence, temporal_weights):
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

        return torch.sum(
            sequence * temporal_weights.unsqueeze(-1),
            dim=1,
        )

    # ------------------------------------------------------------------
    # Shared CRA projection helpers
    # ------------------------------------------------------------------
    def _sample_projection_dropout_mask(self, reference):
        dropout_layer = self.video_proj[0]

        if not self.training or dropout_layer.p <= 0.0:
            return torch.ones_like(reference)

        keep_prob = 1.0 - float(dropout_layer.p)
        if keep_prob <= 0.0:
            raise ValueError(
                "video projection dropout must be < 1."
            )

        mask = torch.empty_like(reference).bernoulli_(keep_prob)
        return mask / keep_prob

    def _apply_video_projection(self, pooled, dropout_mask):
        if pooled.shape != dropout_mask.shape:
            raise ValueError(
                "pooled representation and dropout mask must have the same "
                f"shape; got {tuple(pooled.shape)} and "
                f"{tuple(dropout_mask.shape)}."
            )

        return self.video_proj[1](
            pooled * dropout_mask
        )

    def _project_temporal_sequence(
        self,
        sequence,
        temporal_weights,
        projection_dropout_mask,
        do=False,
        x_star_proj=None,
    ):
        pooled = self._pool_temporal_sequence(
            sequence,
            temporal_weights,
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

        return pooled, projected

    # ------------------------------------------------------------------
    # Explicit fail-fast stubs for mechanisms removed before V2.4.
    # ------------------------------------------------------------------
    def score_functional_composition_pairs(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "score_functional_composition_pairs() is not part of Q-Bound V2.4. "
            "V2.4 has no answer-based functional temporal critic."
        )

    def score_boundary_equilibrium_views(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "score_boundary_equilibrium_views() is not part of Q-Bound V2.4."
        )

    # ------------------------------------------------------------------
    # Temporal contract
    # ------------------------------------------------------------------
    def _build_time_param(
        self,
        final_key,
        training_key,
        soft_key,
        base_temporal,
        qbound_active,
        qbound_time=None,
    ):
        if qbound_active:
            if qbound_time is None:
                raise ValueError(
                    "qbound_time is required for active Q-Bound V2.4."
                )

            time_param = dict(qbound_time)
            time_param["key"] = final_key
        else:
            time_param = self.grounding_module.gen_grounding(
                final_key,
                window_sizes=self.cfgs["model"].get(
                    "window_sizes",
                    [1, 3, 5],
                ),
            )

        time_param["ori_key"] = base_temporal["ori_key"]
        time_param["cra_key"] = base_temporal["cra_key"]

        # Canonical Q-Bound temporal interfaces.
        time_param["qbound_key"] = final_key
        time_param["qbound_training_key"] = training_key
        time_param["qbound_soft_key"] = soft_key

        time_param["qbound_mix"] = torch.full(
            (final_key.size(0),),
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )

        time_param["qbound_handover_enabled"] = torch.zeros(
            (),
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_smoothing_applied"] = torch.zeros(
            (),
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
        # CRA branch: retained as benchmark/reference and QA host.
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

        qbound_active = bool(
            self.qbound_enabled and use_qbound
        )

        if qbound_active and do:
            raise RuntimeError(
                "Controlled Q-Bound V2.4 requires model.do=false. "
                "VisualIntervention would confound the grounding experiment."
            )

        # CRA temporal grounding remains a detached reference whenever Q-Bound
        # is active.  CRA must not teach Q-Bound's temporal topology.
        if qbound_active:
            with torch.no_grad():
                base_temporal = (
                    self.grounding_module.compute_base_temporal(
                        x_bar,
                        qa,
                        valid_mask=valid_mask,
                    )
                )
        else:
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

        if qbound_active:
            cra_key = cra_key.detach()

        # Shared projection dropout is sampled once so CRA/Q-Bound hard/soft
        # feature comparisons are not contaminated by different dropout masks.
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
            x_star_dropout_mask = (
                self._sample_projection_dropout_mask(
                    projection_reference
                )
            )
            x_star_proj = self._apply_video_projection(
                projection_reference,
                x_star_dropout_mask,
            )

        _, cra_video_proj = self._project_temporal_sequence(
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
            # Strict V2.4 question-only input contract.
            # ----------------------------------------------------------
            if question_feature is None:
                raise ValueError(
                    "Q-Bound V2.4 requires question_feature. "
                    "Answer/candidate-answer features may not be substituted."
                )

            if self.qbound_require_token_question and question_feature.ndim != 3:
                raise ValueError(
                    "Q-Bound V2.4 requires token-level question features "
                    "[B,L,D] for early Q->V cross-attention."
                )

            if qbound_video_tokens is None:
                raise ValueError(
                    "Q-Bound V2.4 requires qbound_video_tokens=[B,T,D] "
                    "containing the raw/generic CLIP sequence."
                )

            if qbound_video_tokens.ndim != 3:
                raise ValueError(
                    "qbound_video_tokens must be [B,T,D], got "
                    f"{tuple(qbound_video_tokens.shape)}."
                )

            if qbound_video_tokens.shape[:2] != (
                batch_size,
                time_steps,
            ):
                raise ValueError(
                    "qbound_video_tokens batch/time mismatch: expected "
                    f"{(batch_size, time_steps)}, got "
                    f"{tuple(qbound_video_tokens.shape[:2])}."
                )

            if qbound_video_tokens.size(-1) != self.embed_dim:
                raise ValueError(
                    "Q-Bound V2.4 currently expects visual dimension "
                    f"{self.embed_dim}; got "
                    f"{qbound_video_tokens.size(-1)}."
                )

            qbound_video_tokens = qbound_video_tokens.to(
                device=x_bar.device,
                dtype=x_bar.dtype,
            )

            qbound_language = question_feature.to(
                device=x_bar.device,
                dtype=x_bar.dtype,
            )

            normalized_question_mask = (
                self._normalize_question_mask(
                    question_mask,
                    qbound_language,
                    x_bar.device,
                )
            )

            # External feature sources are detached by default so the V2.4
            # experiment learns only Q-Bound's own WHERE representation.
            if self.qbound_detach_backbone_inputs:
                video_for_qbound = (
                    qbound_video_tokens.detach()
                )
                language_for_qbound = (
                    qbound_language.detach()
                )
            else:
                video_for_qbound = qbound_video_tokens
                language_for_qbound = qbound_language

            # ----------------------------------------------------------
            # One-stage V2.4 Core-to-Closure grounding.
            # ----------------------------------------------------------
            qbound_aux = self.qbound(
                video_tokens=video_for_qbound,
                question_features=language_for_qbound,
                video_mask=valid_mask,
                question_mask=normalized_question_mask,
            )

            required_core_keys = {
                "qbound_version",

                # Architecture flags.
                "core_to_closure_enabled",
                "temporal_context_enabled",
                "early_question_conditioning_enabled",
                "evidence_core_enabled",
                "event_continuity_enabled",
                "same_interval_model_for_hard_and_soft",
                "value_where_separation_enabled",

                "boundary_refiner_enabled",
                "local_candidate_refinement_enabled",
                "adaptive_radius_enabled",
                "learned_routing_gate_enabled",
                "independent_endpoint_heads_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",
                "answer_conditioned_temporal_critic_enabled",
                "length_prior_enabled",
                "gaussian_smoothing_enabled",
                "fixed_event_count_enabled",

                # Representation.
                "value_tokens",
                "where_context_states",
                "question_conditioned_states",
                "question_state",

                # Core / continuity.
                "evidence_core",
                "evidence_core_prob",
                "evidence_core_entropy_ratio",
                "evidence_core_top1_prob",
                "event_continuity",
                "continuity_mask",

                # Shared interval model.
                "segment_energy",
                "segment_prob",
                "segment_entropy_ratio",
                "segment_top1_prob",
                "main_temporal_prob",
                "soft_temporal_prob",
                "training_temporal_prob",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_mask",
                "selected_span_length",
                "selected_span_fraction",
                "soft_expected_span_length",
                "soft_expected_span_fraction",
                "single_frame_prob_mass",
                "short_span_prob_mass",
            }

            missing = sorted(
                required_core_keys.difference(qbound_aux)
            )
            if missing:
                raise KeyError(
                    "Q-Bound V2.4 core output is missing: "
                    + ", ".join(missing)
                )

            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore V2.4 runtime version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )

            def _flag(container, name):
                value = container[name]
                if not torch.is_tensor(value) or value.numel() != 1:
                    raise TypeError(
                        f"Q-Bound flag {name!r} must be a scalar tensor."
                    )
                return float(value.detach().item())

            required_core_on = {
                "core_to_closure_enabled",
                "temporal_context_enabled",
                "early_question_conditioning_enabled",
                "evidence_core_enabled",
                "event_continuity_enabled",
                "same_interval_model_for_hard_and_soft",
                "value_where_separation_enabled",
            }

            for name in required_core_on:
                if _flag(qbound_aux, name) != 1.0:
                    raise RuntimeError(
                        f"Q-Bound V2.4 requires core flag {name}=1."
                    )

            required_core_off = {
                "boundary_refiner_enabled",
                "local_candidate_refinement_enabled",
                "adaptive_radius_enabled",
                "learned_routing_gate_enabled",
                "independent_endpoint_heads_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",
                "answer_conditioned_temporal_critic_enabled",
                "length_prior_enabled",
                "gaussian_smoothing_enabled",
                "fixed_event_count_enabled",
            }

            for name in required_core_off:
                if _flag(qbound_aux, name) != 0.0:
                    raise RuntimeError(
                        f"Q-Bound V2.4 requires core flag {name}=0."
                    )

            # ----------------------------------------------------------
            # Final V2.4 temporal distributions.
            #
            # final_key:
            #     hard MAP support for official downstream CRA QA.
            #
            # training_key / soft_key:
            #     differentiable soft membership from the SAME P(s,e), exposed
            #     for the Q-Bound Video-Question grounding objective.
            #
            # The soft key is deliberately NOT substituted into the official
            # CRA answer feature; therefore VQA cannot shape Q-Bound topology.
            # ----------------------------------------------------------
            final_key = self._normalize_temporal_distribution(
                qbound_aux["main_temporal_prob"],
                valid_mask,
            ).detach()

            soft_key = self._normalize_temporal_distribution(
                qbound_aux["soft_temporal_prob"],
                valid_mask,
            )

            training_key = self._normalize_temporal_distribution(
                qbound_aux["training_temporal_prob"],
                valid_mask,
            )

            # Core V2.4 guarantees training_temporal_prob == soft_temporal_prob.
            # Check numerically without cutting the differentiable path.
            soft_training_max_abs_diff = (
                training_key.detach()
                - soft_key.detach()
            ).abs().amax(dim=-1)

            # ----------------------------------------------------------
            # Exact hard interval metadata.  No smoothing/refinement.
            # ----------------------------------------------------------
            qbound_time = (
                self.grounding_module.gen_adaptive_grounding(
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
            )

            qbound_time["pre_smooth_key"] = final_key
            qbound_time["post_smooth_key"] = final_key
            qbound_time["qbound_key"] = final_key
            qbound_time["qbound_training_key"] = training_key
            qbound_time["qbound_soft_key"] = soft_key
            qbound_time["qbound_smoothing_applied"] = torch.zeros(
                (),
                dtype=final_key.dtype,
                device=final_key.device,
            )

            # ----------------------------------------------------------
            # VALUE evidence.
            #
            # QBoundCore returns the original masked CLIP/value tokens.
            # The Q-conditioned WHERE states are NEVER pooled as downstream
            # visual evidence in V2.4.
            # ----------------------------------------------------------
            value_sequence = qbound_aux["value_tokens"]

            if value_sequence.shape != qbound_video_tokens.shape:
                raise ValueError(
                    "Q-Bound value_tokens must preserve [B,T,D]; expected "
                    f"{tuple(qbound_video_tokens.shape)}, got "
                    f"{tuple(value_sequence.shape)}."
                )

            # Hard evidence: official downstream QA feature.
            (
                qbound_hard_pooled_value,
                qbound_hard_video_proj,
            ) = self._project_temporal_sequence(
                sequence=value_sequence,
                temporal_weights=final_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            # Soft evidence: differentiable grounding supervision feature.
            (
                qbound_soft_pooled_value,
                qbound_soft_video_proj,
            ) = self._project_temporal_sequence(
                sequence=value_sequence,
                temporal_weights=soft_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            (
                qbound_training_pooled_value,
                qbound_training_video_proj,
            ) = self._project_temporal_sequence(
                sequence=value_sequence,
                temporal_weights=training_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            # Official answer-space feature is HARD and contains no gradient
            # through the discrete temporal topology.
            video_proj = qbound_hard_video_proj

            # ----------------------------------------------------------
            # Expose clean V2.4 contract for cra_pipeline.py / diagnostics.
            # ----------------------------------------------------------
            qbound_aux["qbound_key"] = final_key
            qbound_aux["qbound_training_key"] = training_key
            qbound_aux["qbound_soft_key"] = soft_key
            qbound_aux["final_key"] = final_key

            qbound_aux["qbound_value_sequence"] = value_sequence

            qbound_aux["qbound_hard_pooled_value"] = (
                qbound_hard_pooled_value
            )
            qbound_aux["qbound_soft_pooled_value"] = (
                qbound_soft_pooled_value
            )
            qbound_aux["qbound_training_pooled_value"] = (
                qbound_training_pooled_value
            )

            qbound_aux["qbound_hard_video_proj"] = (
                qbound_hard_video_proj
            )
            qbound_aux["qbound_soft_video_proj"] = (
                qbound_soft_video_proj
            )
            qbound_aux["qbound_training_video_proj"] = (
                qbound_training_video_proj
            )

            qbound_aux["qbound_soft_training_key_max_abs_diff"] = (
                soft_training_max_abs_diff
            )

            qbound_aux["qbound_question_mask_provided"] = torch.as_tensor(
                float(normalized_question_mask is not None),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )

            # Provenance / anti-shortcut flags.
            qbound_aux["qbound_input_is_generic"] = torch.ones(
                (),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            qbound_aux["qbound_input_uses_cra_embedding"] = torch.zeros(
                (),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            qbound_aux["qbound_value_uses_question_conditioned_states"] = (
                torch.zeros(
                    (),
                    dtype=x_bar.dtype,
                    device=x_bar.device,
                )
            )
            qbound_aux["qbound_value_uses_raw_visual_tokens"] = torch.ones(
                (),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            qbound_aux["qbound_vqa_gradient_to_topology_enabled"] = (
                torch.zeros(
                    (),
                    dtype=x_bar.dtype,
                    device=x_bar.device,
                )
            )
            qbound_aux["qbound_soft_grounding_feature_exposed"] = torch.ones(
                (),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            qbound_aux["functional_critic_used_in_decoder"] = torch.zeros(
                (),
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            qbound_aux["answer_features_used_in_temporal_inference"] = (
                torch.zeros(
                    (),
                    dtype=x_bar.dtype,
                    device=x_bar.device,
                )
            )

            # Reference branch.
            qbound_aux["cra_key"] = cra_key
            qbound_aux["cra_video_proj"] = cra_video_proj
            qbound_aux["main_video_proj"] = video_proj
            qbound_aux["qbound_video_proj"] = video_proj

        else:
            final_key = cra_key
            training_key = cra_key
            soft_key = cra_key
            video_proj = cra_video_proj

        time_param = self._build_time_param(
            final_key=final_key,
            training_key=training_key,
            soft_key=soft_key,
            base_temporal=base_temporal,
            qbound_active=qbound_active,
            qbound_time=qbound_time,
        )

        # --------------------------------------------------------------
        # Official downstream QA feature.
        # --------------------------------------------------------------
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
            (),
            dtype=video_proj.dtype,
            device=video_proj.device,
        )
        time_param["qbound_direct_reasoning_residual_enabled"] = torch.zeros(
            (),
            dtype=video_proj.dtype,
            device=video_proj.device,
        )

        if qbound_aux is not None:
            # Copy Q-Bound outputs into time_param without producing redundant
            # qbound_qbound_* keys.
            for key, value in qbound_aux.items():
                if key == "qbound_version":
                    continue

                if key.startswith("qbound_"):
                    time_param[key] = value
                else:
                    time_param[f"qbound_{key}"] = value

            time_param["qbound_version"] = self.qbound_version

            # Canonical final-span aliases expected by metrics/wrappers.
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

            # Compatibility aliases: "calibrated" now means the final V2.4
            # Core-to-Closure MAP output.  No calibration/smoothing mechanism is
            # actually restored.
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

            # Explicit architecture flags for fail-fast integration checks.
            architecture_flags = (
                "core_to_closure_enabled",
                "temporal_context_enabled",
                "early_question_conditioning_enabled",
                "evidence_core_enabled",
                "event_continuity_enabled",
                "same_interval_model_for_hard_and_soft",
                "value_where_separation_enabled",

                "boundary_refiner_enabled",
                "local_candidate_refinement_enabled",
                "adaptive_radius_enabled",
                "learned_routing_gate_enabled",
                "independent_endpoint_heads_enabled",
                "answer_features_used_in_core",
                "temporal_gt_used_in_core",
                "answer_conditioned_temporal_critic_enabled",
                "length_prior_enabled",
                "gaussian_smoothing_enabled",
                "fixed_event_count_enabled",
            )

            for name in architecture_flags:
                time_param[f"qbound_{name}"] = qbound_aux[name]

            time_param["qbound_handover_enabled"] = torch.zeros(
                (),
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
            time_param["qbound_direct_reasoning_residual_enabled"] = (
                torch.zeros(
                    (),
                    dtype=video_proj.dtype,
                    device=video_proj.device,
                )
            )

        time_param["qbound_aux"] = qbound_aux

        return (
            video_proj,
            final_key,
            time_param,
        )
