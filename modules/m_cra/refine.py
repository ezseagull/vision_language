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
    CRA downstream decoder with Q-Bound V2.0.0.

    V2.0.0 removes the V1.12 fixed-event / exhaustive-hypothesis / teacher-
    distillation path.  Q-Bound now consumes the raw/generic visual sequence
    and question-only features, learns a video-anchored query-specific soft cut
    field, performs boundary-gated temporal reasoning, and predicts exact
    start/end evidence directly.

    CRA remains only a downstream VideoQA consumer / benchmark testbed:
      * CRA ``x_bar`` is NOT used as Q-Bound input;
      * CRA temporal grounding is a detached diagnostic when Q-Bound is active;
      * CRA's answer space may be used at training time as a stop-gradient
        functional verifier for local KEEP-vs-MERGE and boundary-equilibrium
        consequences, but there is no teacher distribution or distillation;
      * Q-Bound inference itself remains V+Q only.

    The hard V2 span is used in the forward pass.  During training, the core's
    straight-through ``training_temporal_prob`` has the same forward value as
    the hard span but provides gradients to the start/end heads.
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
        self.qbound_version = "2.0.0"

        # CRA decoder is retained as downstream answer consumer / reference.
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

        # Preserve CRA parameter names:
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
                input_adapter_use_position=bool(
                    qbound_cfg.get("input_adapter_use_position", True)
                ),
                cut_temperature=float(
                    qbound_cfg.get("cut_temperature", 1.0)
                ),
                cut_threshold=float(
                    qbound_cfg.get("cut_threshold", 0.5)
                ),
                question_residual_scale=float(
                    qbound_cfg.get("question_residual_scale", 1.0)
                ),
                num_anchor_heads=int(
                    qbound_cfg.get("num_anchor_heads", 4)
                ),
                num_local_reasoning_layers=int(
                    qbound_cfg.get("num_local_reasoning_layers", 2)
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
        return mask.to(device=device).bool()

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
    # Shared CRA projection helpers
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
            projected = self.do(x_star_proj, projected)
        return projected

    # ------------------------------------------------------------------
    # V2.0.0 training-only functional verifier interfaces
    # ------------------------------------------------------------------
    def _detached_frame_answer_logits(self, answer_proj, qbound_aux):
        """
        Deterministic stop-gradient CRA answer evidence per temporal token.

        This is NOT a teacher distribution.  It is a downstream functional
        verifier used only to measure how local temporal perturbations change
        GT-answer evidence during training.
        """
        if answer_proj.ndim != 3:
            raise ValueError(
                "answer_proj must be [B,C,D], got "
                f"{tuple(answer_proj.shape)}."
            )
        if not isinstance(qbound_aux, dict):
            raise TypeError("qbound_aux must be a dict.")

        required = {
            "qbound_version",
            "functional_verifier_video_sequence",
            "functional_verifier_video_mask",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "V2.0.0 functional verifier interface is missing: "
                + ", ".join(missing)
            )
        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "Functional verifier received incompatible Q-Bound output: "
                f"expected {self.qbound_version!r}, got "
                f"{qbound_aux['qbound_version']!r}."
            )

        sequence = qbound_aux["functional_verifier_video_sequence"].detach()
        video_mask = qbound_aux["functional_verifier_video_mask"].detach().bool()
        answer_detached = answer_proj.detach()

        if sequence.ndim != 3:
            raise ValueError("functional_verifier_video_sequence must be [B,T,D].")
        batch_size, time_steps, dim = sequence.shape
        if answer_detached.shape[0] != batch_size or answer_detached.shape[-1] != dim:
            raise ValueError(
                "answer_proj must match functional verifier batch/feature dimensions."
            )
        if video_mask.shape != (batch_size, time_steps):
            raise ValueError("functional_verifier_video_mask must be [B,T].")

        linear = self.video_proj[1]
        weight = linear.weight.detach()
        bias = linear.bias.detach() if linear.bias is not None else None

        projected_sequence = F.linear(sequence, weight, bias)
        frame_answer_logits = torch.einsum(
            "btd,bcd->btc",
            projected_sequence,
            answer_detached,
        )
        frame_answer_logits = frame_answer_logits * video_mask.unsqueeze(-1).to(
            frame_answer_logits.dtype
        )
        return frame_answer_logits, video_mask

    def score_functional_composition_pairs(self, answer_proj, qbound_aux):
        """
        Local KEEP-vs-MERGE answer evidence for each adjacent temporal pair.

        KEEP preserves the two local states and aggregates their answer logits
        with log-mean-exp.  MERGE averages the two states before answer scoring.
        Since the downstream projection is linear, MERGE is the arithmetic mean
        of the two frame logits; KEEP remains sensitive to a strong answer cue
        existing on only one side of the boundary.

        All returned logits are stop-gradient.  The training pipeline should use
        the GT-answer margin difference to construct a detached cut target.
        """
        frame_answer_logits, video_mask = self._detached_frame_answer_logits(
            answer_proj, qbound_aux
        )
        pair_valid = qbound_aux["functional_pair_valid_mask"].detach().bool()
        expected = (frame_answer_logits.size(0), frame_answer_logits.size(1) - 1)
        if pair_valid.shape != expected:
            raise ValueError(
                "functional_pair_valid_mask must be [B,T-1], got "
                f"{tuple(pair_valid.shape)}."
            )

        left_logits = frame_answer_logits[:, :-1]
        right_logits = frame_answer_logits[:, 1:]
        stacked = torch.stack([left_logits, right_logits], dim=2)
        keep_logits = torch.logsumexp(stacked, dim=2) - math.log(2.0)
        merge_logits = 0.5 * (left_logits + right_logits)

        mask_f = pair_valid.unsqueeze(-1).to(keep_logits.dtype)
        return {
            "functional_keep_answer_logits": keep_logits * mask_f,
            "functional_merge_answer_logits": merge_logits * mask_f,
            "functional_left_answer_logits": left_logits * mask_f,
            "functional_right_answer_logits": right_logits * mask_f,
            "functional_pair_valid_mask": pair_valid,
            "functional_verifier_stop_gradient": torch.ones(
                (), dtype=keep_logits.dtype, device=keep_logits.device
            ),
            "teacher_distribution_used": torch.zeros(
                (), dtype=keep_logits.dtype, device=keep_logits.device
            ),
        }

    def _score_detached_temporal_view(self, frame_answer_logits, weights):
        if weights.ndim != 2:
            raise ValueError("temporal view weights must be [B,T].")
        if weights.shape[:2] != frame_answer_logits.shape[:2]:
            raise ValueError("temporal view weights must match [B,T].")
        return torch.einsum(
            "bt,btc->bc",
            weights.detach(),
            frame_answer_logits,
        )

    def score_boundary_equilibrium_views(self, answer_proj, qbound_aux):
        """
        Answer logits for H*, trim-left/right, and expand-left/right.

        This is a five-view local verifier, not an exhaustive hypothesis teacher.
        The downstream pipeline can use GT-answer margin changes to supervise the
        boundary-equilibrium objective without temporal ground truth.
        """
        frame_answer_logits, video_mask = self._detached_frame_answer_logits(
            answer_proj, qbound_aux
        )
        del video_mask

        required = {
            "selected_span_frame_weights",
            "trim_left_frame_weights",
            "trim_right_frame_weights",
            "expand_left_frame_weights",
            "expand_right_frame_weights",
            "trim_left_valid_mask",
            "trim_right_valid_mask",
            "expand_left_valid_mask",
            "expand_right_valid_mask",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "V2.0.0 boundary-equilibrium interface is missing: "
                + ", ".join(missing)
            )

        selected = self._score_detached_temporal_view(
            frame_answer_logits,
            qbound_aux["selected_span_frame_weights"],
        )
        trim_left = self._score_detached_temporal_view(
            frame_answer_logits,
            qbound_aux["trim_left_frame_weights"],
        )
        trim_right = self._score_detached_temporal_view(
            frame_answer_logits,
            qbound_aux["trim_right_frame_weights"],
        )
        expand_left = self._score_detached_temporal_view(
            frame_answer_logits,
            qbound_aux["expand_left_frame_weights"],
        )
        expand_right = self._score_detached_temporal_view(
            frame_answer_logits,
            qbound_aux["expand_right_frame_weights"],
        )

        return {
            "equilibrium_selected_answer_logits": selected,
            "equilibrium_trim_left_answer_logits": trim_left,
            "equilibrium_trim_right_answer_logits": trim_right,
            "equilibrium_expand_left_answer_logits": expand_left,
            "equilibrium_expand_right_answer_logits": expand_right,
            "equilibrium_trim_left_valid_mask": qbound_aux[
                "trim_left_valid_mask"
            ].detach().bool(),
            "equilibrium_trim_right_valid_mask": qbound_aux[
                "trim_right_valid_mask"
            ].detach().bool(),
            "equilibrium_expand_left_valid_mask": qbound_aux[
                "expand_left_valid_mask"
            ].detach().bool(),
            "equilibrium_expand_right_valid_mask": qbound_aux[
                "expand_right_valid_mask"
            ].detach().bool(),
            "equilibrium_verifier_stop_gradient": torch.ones(
                (), dtype=selected.dtype, device=selected.device
            ),
            "teacher_distribution_used": torch.zeros(
                (), dtype=selected.dtype, device=selected.device
            ),
        }

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
                    "qbound_time is required for active Q-Bound V2.0.0."
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
    ):
        del gsub
        del qbound_mix
        del window_sizes

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # CRA decoder remains the downstream answer consumer/testbed.
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
                "Controlled Q-Bound V2.0.0 requires model.do=false. "
                "VisualIntervention would confound the functional-composition "
                "experiment."
            )

        # CRA temporal grounding is a detached reference whenever Q-Bound is active.
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

        projection_reference = self._masked_mean_video(x_bar, valid_mask)
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

        qbound_aux = None
        qbound_time = None

        if qbound_active:
            if question_feature is None and self.qbound_require_question_only:
                raise ValueError(
                    "Q-Bound V2.0.0 requires question_feature=[B,D] or [B,L,D]. "
                    "Do not reuse candidate-answer features as Q-Bound input."
                )
            qbound_language = (
                question_feature if question_feature is not None else qa
            )

            if qbound_video_tokens is None:
                raise ValueError(
                    "Q-Bound V2.0.0 requires qbound_video_tokens=[B,T,D] "
                    "containing the raw/generic visual feature sequence."
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
                    "Controlled V2.0.0 currently expects generic visual "
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
            )

            required_core_keys = {
                "qbound_version",
                "functional_composition_enabled",
                "dynamic_event_count_enabled",
                "fixed_event_partition_enabled",
                "event_count_cap_enabled",
                "hypothesis_competition_enabled",
                "teacher_distillation_enabled",
                "answer_features_used_in_core",
                "boundary_equilibrium_interface_enabled",
                "functional_cut_prob",
                "functional_merge_gate",
                "hard_event_count",
                "soft_effective_event_count",
                "one_event_indicator",
                "all_split_indicator",
                "functional_pair_valid_mask",
                "start_prob",
                "end_prob",
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
                "trim_left_frame_weights",
                "trim_right_frame_weights",
                "expand_left_frame_weights",
                "expand_right_frame_weights",
                "trim_left_valid_mask",
                "trim_right_valid_mask",
                "expand_left_valid_mask",
                "expand_right_valid_mask",
            }
            missing = sorted(required_core_keys.difference(qbound_aux))
            if missing:
                raise KeyError(
                    "Q-Bound V2.0.0 core output is missing: "
                    + ", ".join(missing)
                )

            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore V2.0.0 runtime version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )

            def _flag(name):
                value = qbound_aux[name]
                if not torch.is_tensor(value):
                    raise TypeError(f"Q-Bound flag {name!r} must be a tensor.")
                return float(value.detach().item())

            if _flag("functional_composition_enabled") != 1.0:
                raise RuntimeError("V2.0.0 requires functional composition.")
            if _flag("dynamic_event_count_enabled") != 1.0:
                raise RuntimeError("V2.0.0 requires emergent dynamic event count.")
            if _flag("fixed_event_partition_enabled") != 0.0:
                raise RuntimeError("V2.0.0 forbids a fixed event partition.")
            if _flag("event_count_cap_enabled") != 0.0:
                raise RuntimeError("V2.0.0 forbids a semantic event-count cap.")
            if _flag("hypothesis_competition_enabled") != 0.0:
                raise RuntimeError("V2.0.0 removes exhaustive hypothesis competition.")
            if _flag("teacher_distillation_enabled") != 0.0:
                raise RuntimeError("V2.0.0 removes teacher-distribution distillation.")
            if _flag("answer_features_used_in_core") != 0.0:
                raise RuntimeError("Q-Bound V2.0.0 core must remain answer-free.")
            if _flag("boundary_equilibrium_interface_enabled") != 1.0:
                raise RuntimeError("V2.0.0 requires the boundary-equilibrium interface.")

            # Hard H* is the official inference/evaluation key.
            final_key = self._normalize_temporal_distribution(
                qbound_aux["main_temporal_prob"],
                valid_mask,
            )
            qbound_key = final_key

            # Same forward value as H*, but straight-through gradients also reach
            # start/end heads during weak QA training.
            training_key = self._normalize_temporal_distribution(
                qbound_aux["training_temporal_prob"],
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
            qbound_time["qbound_smoothing_applied"] = torch.zeros(
                (), dtype=final_key.dtype, device=final_key.device
            )

            # Forward numerics equal the hard H*, while backward also trains the
            # endpoint heads through the core's straight-through bridge.
            qbound_video_proj = self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=training_key,
                valid_mask=valid_mask,
                do=False,
                projection_dropout_mask=projection_dropout_mask,
                x_star_proj=None,
            )
            video_proj = qbound_video_proj

            # Training-only functional verifier source.  This does not enter the
            # Q-Bound forward path and is fully detached.
            qbound_aux["functional_verifier_video_sequence"] = x_bar.detach()
            qbound_aux["functional_verifier_video_mask"] = valid_mask.detach()
            qbound_aux["functional_answer_verifier_ready"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["boundary_equilibrium_verifier_ready"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["functional_verifier_consumer_stop_gradient"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["teacher_distribution_used"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_input_is_generic"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_input_uses_cra_embedding"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )

            # Downstream consumer aliases only.
            qbound_aux["cra_key"] = cra_key
            qbound_aux["qbound_key"] = final_key
            qbound_aux["qbound_training_key"] = training_key
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

        # V2 has no direct event-token residual into the answer representation.
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
            # Prefix the complete V2 core/verifier interface into time_param.
            # These are references to the same tensors, not tensor copies.
            for key, value in qbound_aux.items():
                if key == "qbound_version":
                    continue
                time_param[f"qbound_{key}"] = value

            time_param["qbound_version"] = self.qbound_version
            time_param["qbound_input_is_generic"] = qbound_aux[
                "qbound_input_is_generic"
            ]
            time_param["qbound_input_uses_cra_embedding"] = qbound_aux[
                "qbound_input_uses_cra_embedding"
            ]

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

            # Semantically safe generic-evaluation aliases.  They refer to the
            # V2 hard span; no V1.x calibration/refiner is being restored.
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
            time_param["qbound_functional_composition_enabled"] = qbound_aux[
                "functional_composition_enabled"
            ]
            time_param["qbound_dynamic_event_count_enabled"] = qbound_aux[
                "dynamic_event_count_enabled"
            ]
            time_param["qbound_fixed_event_partition_enabled"] = qbound_aux[
                "fixed_event_partition_enabled"
            ]
            time_param["qbound_event_count_cap_enabled"] = qbound_aux[
                "event_count_cap_enabled"
            ]
            time_param["qbound_hypothesis_competition_enabled"] = qbound_aux[
                "hypothesis_competition_enabled"
            ]
            time_param["qbound_teacher_distillation_enabled"] = qbound_aux[
                "teacher_distillation_enabled"
            ]
            time_param["qbound_answer_features_used_in_core"] = qbound_aux[
                "answer_features_used_in_core"
            ]
            time_param["qbound_boundary_equilibrium_interface_enabled"] = qbound_aux[
                "boundary_equilibrium_interface_enabled"
            ]

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
