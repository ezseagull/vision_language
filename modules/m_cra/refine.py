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
    CRA refinement decoder with Q-Bound Version 1.12.

    V1.12 is an architecture-level reset.  The historical

        COMPOSE -> BAPGR -> PPBR -> PSRC -> donor-CF

    route is removed from the active path.  Q-Bound now performs:

        generic visual input
            -> dynamic DECOMPOSE (1..K_max events)
            -> Q-OEI RELATE
            -> exact frame-level temporal hypothesis bank
            -> question-only intra-video hypothesis competition
            -> hard winning hypothesis
            -> exact temporal grounding

    Events are semantic reasoning structure, not hard final boundaries.  The
    final hypothesis bank contains every valid contiguous frame interval for
    the sampled sequence, so a slightly wrong event boundary cannot impose an
    oracle ceiling on frame-level grounding precision.

    Privileged answer-aware teacher interface
    -----------------------------------------
    QBoundCore remains answer-free.  This decoder exposes a stop-gradient,
    deterministic critic interface built from CRA's downstream ``x_bar`` and
    the exact hypothesis/complement weights.  A later CRA/pipeline component
    may call ``score_privileged_hypothesis_bank`` with candidate-answer
    embeddings to obtain answer logits for every hypothesis, its complement,
    and the full video.  Those logits are training-only teacher signals; the
    Q-Bound scorer itself still predicts p(H | V,Q) and needs no answer feature
    at inference.

    The critic is intentionally efficient.  It exploits linearity of CRA's
    ``video_proj.1`` so it never materializes [B,S,D] projected features for
    all S hypotheses.  It also disables projection dropout on the teacher
    branch, making within-video ranking deterministic.  All critic inputs and
    projection parameters are detached, so teacher losses cannot update CRA.

    Main-path contract
    ------------------
    * Q-Bound consumes explicit raw/generic visual tokens, not CRA VideoEmbedding.
    * Question-only features condition Q-Bound.
    * MAIN uses the exact hard winning hypothesis from QBoundCore.
    * No Gaussian smoothing is applied after hypothesis selection; smoothing
      would leak probability outside the selected exact interval and conflict
      with the V1.12 hypothesis semantics.
    * CRA temporal grounding remains a detached diagnostic/reference branch.
    * VisualIntervention is forbidden in the controlled V1.12 Q-Bound path.
    * No temporal GT / IoU / IoP supervision is introduced here.
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
        self.qbound_version = "1.12"

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

        # Kept for CRA checkpoint compatibility.
        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        # Preserve original CRA parameter names:
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
        self.qbound_event_capacity = None
        self.qbound_proposal_candidates = None

        if self.qbound_enabled:
            self.qbound_hidden_dim = int(
                qbound_cfg.get(
                    "hidden_dim",
                    model_cfg.get("qbound_hidden_dim", 256),
                )
            )
            # V1.12: K_max is a capacity ceiling, never a forced event count.
            self.qbound_event_capacity = int(
                qbound_cfg.get(
                    "max_events",
                    model_cfg.get("qbound_max_events", 6),
                )
            )
            self.qbound_proposal_candidates = int(
                qbound_cfg.get("proposal_candidates", 8)
            )

            decomposition_cfg = qbound_cfg.get("decomposition", {})
            if not isinstance(decomposition_cfg, dict):
                raise ValueError(
                    "model.qbound.decomposition must be a mapping/dict."
                )

            self.qbound = QBoundCore(
                input_dim=embed_dim,
                hidden_dim=self.qbound_hidden_dim,
                max_events=self.qbound_event_capacity,
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
                proposal_candidates=self.qbound_proposal_candidates,
                decomp_complexity_weight=float(
                    qbound_cfg.get(
                        "decomp_complexity_weight",
                        decomposition_cfg.get("complexity_weight", 1.25),
                    )
                ),
                decomp_boundary_weight=float(
                    qbound_cfg.get(
                        "decomp_boundary_weight",
                        decomposition_cfg.get("boundary_weight", 0.50),
                    )
                ),
                decomp_temperature=float(
                    qbound_cfg.get(
                        "decomp_temperature",
                        decomposition_cfg.get("temperature", 0.20),
                    )
                ),
                question_modulation_scale=float(
                    qbound_cfg.get("question_modulation_scale", 0.10)
                ),
                min_event_mass=float(
                    qbound_cfg.get(
                        "min_event_mass",
                        model_cfg.get("qbound_min_event_mass", 0.25),
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
    # V1.12 privileged answer-aware critic
    # ------------------------------------------------------------------
    def score_privileged_hypothesis_bank(
        self,
        answer_proj,
        qbound_aux,
    ):
        """
        Score every temporal hypothesis with a stop-gradient CRA answer critic.

        Parameters
        ----------
        answer_proj:
            Candidate answer embeddings [B,C,D].  They are detached here.
        qbound_aux:
            V1.12 auxiliary dict returned by ``forward``.

        Returns
        -------
        Dict with:
            hypothesis_answer_logits: [B,S,C]
            complement_answer_logits: [B,S,C]
            full_video_answer_logits: [B,C]
            hypothesis_valid_mask: [B,S]
            complement_valid_mask: [B,S]

        The computation is algebraically equivalent to:

            pooled_H = sum_t w_H(t) * x_bar(t)
            projected_H = Linear(pooled_H)
            score(H,a) = <projected_H, answer_a>

        but moves the frozen linear projection onto answer embeddings first.
        This avoids materializing/projecting [B,S,D] representations for all
        hypotheses.  Projection dropout is deliberately omitted on the critic
        branch so the ranking target is deterministic.
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
            "critic_video_sequence",
            "critic_full_video_weights",
            "hypothesis_frame_weights",
            "hypothesis_complement_frame_weights",
            "hypothesis_valid_mask",
            "hypothesis_complement_valid_mask",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "V1.12 privileged critic interface is missing: "
                + ", ".join(missing)
            )
        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "Privileged critic received an incompatible Q-Bound output: "
                f"expected {self.qbound_version!r}, got "
                f"{qbound_aux['qbound_version']!r}."
            )

        sequence = qbound_aux["critic_video_sequence"]
        hypothesis_weights = qbound_aux["hypothesis_frame_weights"]
        complement_weights = qbound_aux[
            "hypothesis_complement_frame_weights"
        ]
        full_video_weights = qbound_aux["critic_full_video_weights"]

        if sequence.ndim != 3:
            raise ValueError("critic_video_sequence must be [B,T,D].")
        batch_size, time_steps, dim = sequence.shape
        if answer_proj.shape[0] != batch_size or answer_proj.shape[-1] != dim:
            raise ValueError(
                "answer_proj must match critic batch/feature dimensions."
            )
        if hypothesis_weights.ndim != 3:
            raise ValueError("hypothesis_frame_weights must be [B,S,T].")
        if hypothesis_weights.shape[0] != batch_size:
            raise ValueError("hypothesis bank batch mismatch.")
        if hypothesis_weights.shape[-1] != time_steps:
            raise ValueError("hypothesis bank temporal length mismatch.")
        if complement_weights.shape != hypothesis_weights.shape:
            raise ValueError(
                "hypothesis complement weights must match hypothesis weights."
            )
        if full_video_weights.shape != (batch_size, time_steps):
            raise ValueError("critic_full_video_weights must be [B,T].")

        # Teacher branch: no gradient to CRA sequence, answer encoder, or
        # answer-space projection.  Utilities built from these logits must also
        # be detached by the downstream training pipeline.
        sequence = sequence.detach()
        answer_detached = answer_proj.detach()
        hypothesis_weights = hypothesis_weights.detach()
        complement_weights = complement_weights.detach()
        full_video_weights = full_video_weights.detach()

        linear = self.video_proj[1]
        weight = linear.weight.detach()
        bias = linear.bias.detach() if linear.bias is not None else None

        # For y = Linear(u) = u W^T + b and score=<a,y>:
        # score = <u, a W> + <a,b>.
        answer_in_input_space = torch.matmul(
            answer_detached,
            weight,
        )
        frame_answer_logits = torch.einsum(
            "btd,bcd->btc",
            sequence,
            answer_in_input_space,
        )

        if bias is None:
            answer_bias = torch.zeros(
                batch_size,
                answer_proj.size(1),
                dtype=sequence.dtype,
                device=sequence.device,
            )
        else:
            answer_bias = torch.einsum(
                "bcd,d->bc",
                answer_detached,
                bias,
            )

        hypothesis_answer_logits = torch.einsum(
            "bst,btc->bsc",
            hypothesis_weights,
            frame_answer_logits,
        ) + answer_bias.unsqueeze(1)

        complement_answer_logits = torch.einsum(
            "bst,btc->bsc",
            complement_weights,
            frame_answer_logits,
        ) + answer_bias.unsqueeze(1)

        full_video_answer_logits = torch.einsum(
            "bt,btc->bc",
            full_video_weights,
            frame_answer_logits,
        ) + answer_bias

        return {
            "hypothesis_answer_logits": hypothesis_answer_logits,
            "complement_answer_logits": complement_answer_logits,
            "full_video_answer_logits": full_video_answer_logits,
            "hypothesis_valid_mask": qbound_aux[
                "hypothesis_valid_mask"
            ].detach().bool(),
            "complement_valid_mask": qbound_aux[
                "hypothesis_complement_valid_mask"
            ].detach().bool(),
            "critic_stop_gradient": torch.ones(
                (),
                dtype=sequence.dtype,
                device=sequence.device,
            ),
            "critic_projection_dropout_enabled": torch.zeros(
                (),
                dtype=sequence.dtype,
                device=sequence.device,
            ),
        }

    # ------------------------------------------------------------------
    # Temporal contract
    # ------------------------------------------------------------------
    def _build_time_param(
        self,
        final_key,
        base_temporal,
        qbound_active,
        qbound_time=None,
    ):
        if qbound_active:
            if qbound_time is None:
                raise ValueError(
                    "qbound_time is required for active Q-Bound V1.12."
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
        time_param["qbound_calibrated_key"] = final_key
        time_param["refined_key"] = final_key
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

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # CRA decoder representation remains the downstream consumer/testbed.
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
                "Controlled Q-Bound V1.12 requires model.do=false. "
                "VisualIntervention would confound hypothesis competition and "
                "the privileged answer-aware critic."
            )

        # CRA temporal grounding is reference-only whenever Q-Bound is active.
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

        # One factual projection-dropout realization is shared by CRA and
        # Q-Bound main answer branches.  The privileged teacher branch is
        # deterministic and does not use this mask.
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
        legacy_fixed_time = None

        if qbound_active:
            if question_feature is None and self.qbound_require_question_only:
                raise ValueError(
                    "Q-Bound V1.12 requires question_feature=[B,D]. "
                    "Do not reuse candidate-answer features as Q-Bound input."
                )

            qbound_language = (
                question_feature if question_feature is not None else qa
            )

            if qbound_video_tokens is None:
                raise ValueError(
                    "Q-Bound V1.12 requires qbound_video_tokens=[B,T,D] "
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
                    "Controlled V1.12 currently expects generic visual "
                    f"dimension {self.embed_dim}; got "
                    f"{qbound_video_tokens.size(-1)}."
                )

            qbound_video_tokens = qbound_video_tokens.to(
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
                contextual_tokens=None,
                temporal_prior=None,
            )

            required_core_keys = {
                "qbound_version",
                "dynamic_event_count_enabled",
                "event_capacity",
                "hypothesis_competition_enabled",
                "answer_features_used_in_core",
                "legacy_composer_enabled",
                "legacy_refiner_enabled",
                "legacy_counterfactual_donor_enabled",
                "generic_input_adapter_enabled",
                "generic_input_context_reuse",
                "event_count",
                "event_assignment",
                "event_valid_mask",
                "event_visual_tokens",
                "evidence_visual_token",
                "reasoned_event_tokens",
                "decomp_loss",
                "hypothesis_start_frame",
                "hypothesis_end_frame",
                "hypothesis_valid_mask",
                "hypothesis_frame_mask",
                "hypothesis_frame_weights",
                "hypothesis_complement_frame_mask",
                "hypothesis_complement_frame_weights",
                "hypothesis_complement_valid_mask",
                "hypothesis_trim_left_index",
                "hypothesis_trim_right_index",
                "hypothesis_expand_left_index",
                "hypothesis_expand_right_index",
                "hypothesis_trim_left_valid_mask",
                "hypothesis_trim_right_valid_mask",
                "hypothesis_expand_left_valid_mask",
                "hypothesis_expand_right_valid_mask",
                "hypothesis_logits",
                "hypothesis_posterior",
                "hypothesis_entropy",
                "hypothesis_effective_count",
                "hypothesis_top1_posterior",
                "hypothesis_soft_support",
                "hypothesis_expected_span_length",
                "selected_hypothesis_index",
                "selected_hypothesis_start_frame",
                "selected_hypothesis_end_frame",
                "selected_hypothesis_frame_mask",
                "selected_hypothesis_frame_weights",
                "selected_hypothesis_span_length",
                "selected_hypothesis_span_fraction",
                "main_temporal_prob",
                "main_temporal_logits",
                "main_soft_support",
                "main_pair_posterior",
            }
            missing = sorted(required_core_keys.difference(qbound_aux))
            if missing:
                raise KeyError(
                    "Q-Bound V1.12 core output is missing: "
                    + ", ".join(missing)
                )

            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore V1.12 runtime version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )

            def _flag(name):
                value = qbound_aux[name]
                if not torch.is_tensor(value):
                    raise TypeError(f"Q-Bound flag {name!r} must be a tensor.")
                return float(value.detach().item())

            if _flag("dynamic_event_count_enabled") != 1.0:
                raise RuntimeError(
                    "V1.12 requires dynamic event-count decomposition."
                )
            if _flag("hypothesis_competition_enabled") != 1.0:
                raise RuntimeError(
                    "V1.12 requires intra-video hypothesis competition."
                )
            if _flag("answer_features_used_in_core") != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.12 core must remain answer-free."
                )
            for legacy_flag in (
                "legacy_composer_enabled",
                "legacy_refiner_enabled",
                "legacy_counterfactual_donor_enabled",
            ):
                if _flag(legacy_flag) != 0.0:
                    raise RuntimeError(
                        f"V1.12 must keep {legacy_flag} disabled."
                    )
            if _flag("generic_input_context_reuse") != 1.0:
                raise RuntimeError(
                    "Controlled V1.12 expects one adapted generic stream to "
                    "be reused for Q-Bound raw/context states."
                )

            # MAIN V1.12 key is already a probability restricted to the hard
            # winning hypothesis.  Do not Gaussian-smooth it: smoothing would
            # inject mass outside H* and undermine frame-exact competition.
            final_key = self._normalize_temporal_distribution(
                qbound_aux["main_temporal_prob"],
                valid_mask,
            )
            qbound_key = final_key

            qbound_time = self.grounding_module.gen_adaptive_grounding(
                final_key,
                selected_span_start_frame=qbound_aux[
                    "selected_hypothesis_start_frame"
                ],
                selected_span_end_frame=qbound_aux[
                    "selected_hypothesis_end_frame"
                ],
                valid_mask=valid_mask,
                selected_frame_mask=qbound_aux[
                    "selected_hypothesis_frame_mask"
                ],
            )
            qbound_time["pre_smooth_key"] = final_key
            qbound_time["post_smooth_key"] = final_key
            qbound_time["qbound_key"] = final_key
            qbound_time["qbound_smoothing_applied"] = torch.zeros(
                (), dtype=final_key.dtype, device=final_key.device
            )

            # CRA-style fixed-window readout on the SAME key is diagnostic only.
            with torch.no_grad():
                legacy_fixed_time = self.grounding_module.gen_grounding(
                    final_key.detach(),
                    window_sizes=window_sizes,
                )

            qbound_video_proj = self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=final_key,
                valid_mask=valid_mask,
                do=False,
                projection_dropout_mask=projection_dropout_mask,
                x_star_proj=None,
            )
            video_proj = qbound_video_proj

            # ----------------------------------------------------------
            # Training-only privileged critic interface.
            # ----------------------------------------------------------
            critic_full_video_weights = valid_mask.to(x_bar.dtype)
            critic_full_video_weights = (
                critic_full_video_weights
                / critic_full_video_weights.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1.0)
            )
            qbound_aux["critic_video_sequence"] = x_bar.detach()
            qbound_aux["critic_full_video_weights"] = (
                critic_full_video_weights.detach()
            )
            qbound_aux["critic_consumer_stop_gradient"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["critic_projection_dropout_enabled"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["privileged_answer_teacher_ready"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_input_is_generic"] = torch.ones(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
            qbound_aux["qbound_input_uses_cra_embedding"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )

            # Decoder / answer-space aliases.
            qbound_aux["cra_key"] = cra_key
            qbound_aux["qbound_key"] = final_key
            qbound_aux["final_key"] = final_key
            qbound_aux["main_video_proj"] = qbound_video_proj
            qbound_aux["qbound_video_proj"] = qbound_video_proj
            qbound_aux["qbound_selected_video_proj"] = qbound_video_proj
            qbound_aux["cra_video_proj"] = cra_video_proj
            qbound_aux["qbound_smoothing_applied"] = torch.zeros(
                (), dtype=x_bar.dtype, device=x_bar.device
            )
        else:
            final_key = cra_key
            qbound_key = cra_key
            video_proj = cra_video_proj

        time_param = self._build_time_param(
            final_key=final_key,
            base_temporal=base_temporal,
            qbound_active=qbound_active,
            qbound_time=qbound_time,
        )

        # Official answer-space feature.
        time_param["grounding_feature"] = video_proj
        time_param["qbound_feature"] = video_proj
        time_param["qbound_grounding_feature"] = video_proj
        time_param["qbound_temporal_main_feature"] = video_proj
        time_param["cra_grounding_feature"] = cra_video_proj

        # No direct event-token residual into answer representation.
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
            # Prefix the complete V1.12 core interface into time_param.  These
            # are references to the same tensors, not tensor copies.
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
            time_param["qbound_privileged_answer_teacher_ready"] = qbound_aux[
                "privileged_answer_teacher_ready"
            ]
            time_param["qbound_critic_consumer_stop_gradient"] = qbound_aux[
                "critic_consumer_stop_gradient"
            ]
            time_param["qbound_critic_projection_dropout_enabled"] = qbound_aux[
                "critic_projection_dropout_enabled"
            ]

            # Main V1.12 exact-hypothesis aliases.
            time_param["qbound_selected_video_proj"] = qbound_aux[
                "qbound_selected_video_proj"
            ]
            time_param["qbound_selected_hypothesis_index"] = qbound_aux[
                "selected_hypothesis_index"
            ]
            time_param["qbound_selected_hypothesis_start_frame"] = qbound_aux[
                "selected_hypothesis_start_frame"
            ]
            time_param["qbound_selected_hypothesis_end_frame"] = qbound_aux[
                "selected_hypothesis_end_frame"
            ]
            time_param["qbound_selected_hypothesis_mask"] = qbound_aux[
                "selected_hypothesis_frame_mask"
            ]
            time_param["qbound_selected_hypothesis_span_length"] = qbound_aux[
                "selected_hypothesis_span_length"
            ]
            time_param["qbound_selected_hypothesis_span_fraction"] = qbound_aux[
                "selected_hypothesis_span_fraction"
            ]

            # Compatibility aliases for generic evaluation code.  They refer to
            # the exact winning hypothesis; there is no V1.10 calibration.
            time_param["qbound_calibrated_video_proj"] = video_proj
            time_param["qbound_refined_video_proj"] = video_proj
            time_param["qbound_calibrated_key"] = final_key
            time_param["qbound_calibrated_start"] = time_param["start"]
            time_param["qbound_calibrated_end"] = time_param["end"]
            time_param["qbound_calibrated_mask"] = time_param["mask"]
            time_param["qbound_calibrated_span_length"] = qbound_aux[
                "selected_hypothesis_span_length"
            ]
            time_param["qbound_calibrated_span_fraction"] = qbound_aux[
                "selected_hypothesis_span_fraction"
            ]

            # Explicit architecture flags replacing legacy Refiner/CF flags.
            time_param["qbound_dynamic_event_count_enabled"] = qbound_aux[
                "dynamic_event_count_enabled"
            ]
            time_param["qbound_hypothesis_competition_enabled"] = qbound_aux[
                "hypothesis_competition_enabled"
            ]
            time_param["qbound_answer_features_used_in_core"] = qbound_aux[
                "answer_features_used_in_core"
            ]
            time_param["qbound_legacy_composer_enabled"] = qbound_aux[
                "legacy_composer_enabled"
            ]
            time_param["qbound_legacy_refiner_enabled"] = qbound_aux[
                "legacy_refiner_enabled"
            ]
            time_param["qbound_legacy_counterfactual_donor_enabled"] = qbound_aux[
                "legacy_counterfactual_donor_enabled"
            ]

            if legacy_fixed_time is not None:
                time_param["qbound_legacy_fixed_window_key"] = (
                    legacy_fixed_time["key"]
                )
                time_param["qbound_legacy_fixed_window_start"] = (
                    legacy_fixed_time["start"]
                )
                time_param["qbound_legacy_fixed_window_end"] = (
                    legacy_fixed_time["end"]
                )
                time_param["qbound_legacy_fixed_window_mask"] = (
                    legacy_fixed_time["mask"]
                )

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
