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
    CRA refinement decoder with Q-Bound Version 1.11.

    V1.11 preserves the V1.10.2 temporal mechanism and adds an independent
    generic visual-input contract:

        raw/frozen visual tokens -> Q-Bound-owned input adapter
                                 -> DECOMPOSE -> RELATE -> COMPOSE
                           -> BOUNDARY-ANCHORED POSTERIOR-GUIDED REFINE
                           -> PRIOR-PRESERVING BOUNDED REFINEMENT
                           -> POSTERIOR-STABLE RELIABILITY CALIBRATION
                           -> GROUND

    The Q-Bound core exposes three temporal branches:

        1) COARSE:
           differentiable COMPOSE probability. It remains part of the MAIN
           temporal probability so the main VQA task supervises Composer.

        2) RAW REFINE:
           fine boundary posterior conditioned on a DETACHED Composer posterior
           pair prior. V1.10.2 keeps the V1.10.1 BAPGR prior and applies a
           differentiable bounded correction before posterior normalization.
           The Refiner can re-rank legal pairs, but it cannot override the
           structural prior with an arbitrarily large learned correction.

        3) RELIABILITY-CALIBRATED MAIN:
           parameter-free fusion of coarse semantic coverage and raw fine
           localization. Reliability is derived from Refiner information gain
           beyond the Composer pair prior, semantic evidence retention, and
           Refiner concentration. The reliability scalar is detached before
           fusion.

    Important routing contract:

        Composer posterior
            |-- differentiable --> COARSE MAIN component --> task gradient
            |
            `-- stop-gradient --> boundary-anchored pair prior
                                  --> bounded Refiner correction --> REFINE

    Therefore V1.10.2 does NOT restore the V1.6 differentiable all-span
    Composer x Refiner marginalization. REFINE cannot train Composer through
    its search prior. PPBR is parameter-free apart from a fixed correction cap.

    V1.11 input contract:
      * CRA-side ``x`` remains the downstream testbed representation.
      * ``qbound_video_tokens`` must be passed explicitly and must contain the
        pre-CRA generic feature sequence (e.g. sampled CLIP-L [B,T,768]).
      * Q-Bound owns the trainable generic input adapter; no CRA temporal key
        or CRA VideoEmbedding output is used as its visual input.
      * the calibrated temporal output is still consumed by CRA ``x_bar`` in
        this version so input decoupling is isolated from output-adapter work.

    V1.11-B counterfactual consumer contract:
      * QBoundCore exposes the mass-preserving differentiable MAIN support:
            keep(t), remove(t)=1-keep(t)
        whose forward values are identical to V1.11-A support.
      * RefineDecoder creates a deterministic in-batch cyclic donor sequence.
      * The factual CRA sequence and donor sequence are detached for the
        counterfactual path, so the new objective cannot reshape CRA x_bar.
      * MAIN temporal pooling weights are detached for the counterfactual path.
      * CRA ``video_proj`` weights/bias are also detached for this path.
      * Therefore answer-level counterfactual gradients are intended to flow
        through keep/remove support back into Q-Bound, not into the CRA
        consumer. The ordinary factual MAIN branch remains fully unchanged.
      * No counterfactual answer loss is computed in this file; the pipeline
        owns sufficiency/necessity objectives.

    The official answer-space representations are:

        qbound_calibrated_video_proj     # V1.10.2 MAIN
        qbound_raw_refined_video_proj    # diagnostic
        qbound_coarse_video_proj         # diagnostic

    All branches use the same CRA x_bar representation, projection layer, and
    projection-dropout realization for controlled comparison.

    Compatibility:
      * qbound_video_proj aliases the V1.10.2 calibrated MAIN representation.
      * qbound_refined_video_proj also aliases calibrated MAIN.
      * legacy V1.8 calibration names are retained where the core maps them to
        V1.10.2 reliability semantics.
      * selected_span_* remains available as a diagnostic hard top-1 Composer
        readout, but it no longer defines the Refiner search region.

    No temporal GT, IoU/IoP loss, target span length, minimality loss,
    event-count forcing, CRA->Q-Bound handover, or direct reasoned-event
    answer residual is introduced here.

    selected_reasoned_event_tokens and selected_event_mask remain propagated
    for future frozen LLM/MLLM adapters.
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
        self.qbound_version = "1.11"

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

        # --------------------------------------------------------------
        # Q-Bound V1.11
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
        self.qbound_counterfactual_evidence_interface = bool(
            qbound_cfg.get(
                "counterfactual_evidence_interface",
                False,
            )
        )

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
                refiner_boundary_anchor_power=float(
                    qbound_cfg.get(
                        "refiner_boundary_anchor_power",
                        refinement_cfg.get("boundary_anchor_power", 2.0),
                    )
                ),
                refiner_correction_cap=float(
                    qbound_cfg.get(
                        "refiner_correction_cap",
                        refinement_cfg.get("correction_cap", 2.0),
                    )
                ),
                # V1.11: Q-Bound owns its generic visual input adapter.
                # The caller passes raw/frozen visual features directly; these
                # parameters are not shared with CRA.VideoEmbedding.
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
                # V1.11-A keeps this disabled. V1.11-B enables the
                # mass-preserving counterfactual evidence interface.
                counterfactual_evidence_interface=(
                    self.qbound_counterfactual_evidence_interface
                ),
            )

            if str(getattr(self.qbound, "qbound_version", "")) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={getattr(self.qbound, 'qbound_version', None)!r}."
                )

            # V1.11 keeps external visual/language encoders detached while Q-Bound learns its own
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

    def _apply_video_projection_frozen(
        self,
        pooled,
        dropout_mask,
    ):
        """
        Apply CRA's answer-space projection as a fixed consumer.

        V1.11-B counterfactual losses must supervise Q-Bound temporal evidence,
        not update CRA's projection layer.  The factual MAIN branch still uses
        the ordinary trainable projection helper above.
        """
        if pooled.shape != dropout_mask.shape:
            raise ValueError(
                "pooled representation and dropout mask must have "
                "the same shape. Got "
                f"{tuple(pooled.shape)} and {tuple(dropout_mask.shape)}."
            )

        linear = self.video_proj[1]
        weight = linear.weight.detach()
        bias = (
            linear.bias.detach()
            if linear.bias is not None
            else None
        )
        return F.linear(
            pooled * dropout_mask.detach(),
            weight,
            bias,
        )

    def _project_counterfactual_sequence(
        self,
        sequence,
        temporal_weights,
        valid_mask,
        projection_dropout_mask,
    ):
        """
        Pool a counterfactual sequence with a fixed MAIN temporal readout and
        project it through a frozen CRA consumer.

        Only ``sequence`` may carry counterfactual gradients.  In V1.11-B the
        sequence depends on Q-Bound keep/remove support while factual/donor
        features are detached.
        """
        if sequence.ndim != 3:
            raise ValueError(
                "counterfactual sequence must be [B,T,D]."
            )
        if temporal_weights.shape != sequence.shape[:2]:
            raise ValueError(
                "counterfactual temporal weights must be [B,T]."
            )
        if valid_mask.shape != sequence.shape[:2]:
            raise ValueError(
                "counterfactual valid mask must be [B,T]."
            )

        fixed_weights = self._normalize_temporal_distribution(
            temporal_weights.detach(),
            valid_mask,
        )
        pooled = torch.sum(
            sequence * fixed_weights.unsqueeze(-1),
            dim=1,
        )
        return self._apply_video_projection_frozen(
            pooled,
            projection_dropout_mask,
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
        coarse_qbound_key=None,
        raw_refined_qbound_key=None,
    ):
        """
        Build the final temporal contract.

        V1.10.2 active:
            official key/start/end/mask are RELIABILITY-CALIBRATED from the
            Composer endpoint posterior and posterior-guided Refiner.

        Explicit branches:
            qbound_key / qbound_calibrated_key
                -> official calibrated MAIN key

            qbound_raw_refined_key
                -> raw V1.9 posterior-guided fine-boundary diagnostic key

            qbound_coarse_key
                -> COMPOSE coarse diagnostic key

        Q-Bound inactive:
            preserve the original CRA fixed-window path.
        """
        if qbound_active:
            if qbound_time is None:
                raise ValueError(
                    "qbound_time is required for active Q-Bound V1.10.2."
                )

            time_param = dict(qbound_time)
            time_param["key"] = final_key
        else:
            time_param = self.grounding_module.gen_grounding(
                final_key,
                window_sizes=window_sizes,
            )

        if coarse_qbound_key is None:
            coarse_qbound_key = qbound_key
        if raw_refined_qbound_key is None:
            raw_refined_qbound_key = qbound_key

        time_param["ori_key"] = base_temporal["ori_key"]
        time_param["cra_key"] = base_temporal["cra_key"]

        # Official V1.10.2 MAIN aliases.
        time_param["qbound_key"] = qbound_key
        time_param["qbound_calibrated_key"] = final_key
        time_param["refined_key"] = final_key

        # Controlled diagnostic branches.
        time_param["qbound_raw_refined_key"] = raw_refined_qbound_key
        time_param["qbound_coarse_key"] = coarse_qbound_key

        # No CRA/Q-Bound temporal handover.
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

        # Compatibility aliases: adaptive/refined mean V1.10.2 calibrated MAIN.
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

        # Explicit V1.10.2 branch flags.
        time_param["qbound_calibrated_span_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_raw_refined_span_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_stable_refiner_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_uncertainty_calibration_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_posterior_guided_refinement_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_boundary_structured_refinement_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_prior_preserving_bounded_refinement_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_posterior_stable_reliability_enabled"] = torch.as_tensor(
            1.0 if qbound_active else 0.0,
            dtype=final_key.dtype,
            device=final_key.device,
        )
        time_param["qbound_minimal_sufficient_enabled"] = torch.as_tensor(
            0.0,
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
    ):
        del gsub
        del qbound_mix  # V1.10.2 intentionally has no CRA->Q-Bound handover.

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # V1.11 contract:
        #   x                   = CRA-side VideoEmbedding output used only by
        #                         the downstream CRA testbed/consumer.
        #   qbound_video_tokens = raw/generic visual features (e.g. CLIP-L)
        #                         routed independently into Q-Bound.
        # Q-Bound must no longer silently fall back to CRA-embedded tokens.

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
        # 1) CRA temporal grounding: detached reference when Q-Bound active.
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
        # 2) Shared answer-projection context.
        #
        # CRA / coarse / refined branches reuse the SAME dropout realization.
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
        # 3) Q-Bound V1.10.2:
        #
        #    DECOMPOSE -> RELATE -> COMPOSE
        #                         |\
        #                         | \ differentiable coarse MAIN component
        #                         |
        #                         \-> DETACHED Composer posterior
        #                              -> boundary-anchored pair prior
        #                              -> bounded Refiner correction (PPBR)
        #                              -> REFINE
        #                              -> reliability-calibrated MAIN
        #
        # The detached posterior prior preserves Composer uncertainty without
        # restoring the V1.6 Refiner -> Composer gradient feedback loop.
        # --------------------------------------------------------------
        qbound_aux = None
        qbound_time = None
        raw_refined_qbound_time = None
        coarse_qbound_time = None
        legacy_v15_time = None
        legacy_v14_time = None

        if qbound_active:
            if (
                question_feature is None
                and self.qbound_require_question_only
            ):
                raise ValueError(
                    "Q-Bound V1.11 requires question_feature=[B,D]. "
                    "Pass question-only features through CRA.forward(...)."
                )

            qbound_language = (
                question_feature
                if question_feature is not None
                else qa
            )

            if qbound_video_tokens is None:
                raise ValueError(
                    "Q-Bound V1.11 requires qbound_video_tokens=[B,T,D] "
                    "containing the raw/generic visual feature sequence. "
                    "Do not reuse CRA VideoEmbedding output as an implicit "
                    "fallback; pass the pre-CRA feature tensor explicitly."
                )
            if qbound_video_tokens.ndim != 3:
                raise ValueError(
                    "qbound_video_tokens must be [B,T,D], got "
                    f"{tuple(qbound_video_tokens.shape)}."
                )
            if qbound_video_tokens.size(0) != batch_size:
                raise ValueError(
                    "qbound_video_tokens batch mismatch: expected "
                    f"{batch_size}, got {qbound_video_tokens.size(0)}."
                )
            if qbound_video_tokens.size(1) != time_steps:
                raise ValueError(
                    "qbound_video_tokens temporal length must match CRA "
                    f"tokens/mask: expected {time_steps}, got "
                    f"{qbound_video_tokens.size(1)}."
                )
            if qbound_video_tokens.size(-1) != self.embed_dim:
                raise ValueError(
                    "Controlled V1.11 currently expects generic visual "
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
                # None tells V1.11 core to reuse the exact adapted generic
                # stream for context, avoiding a second independent dropout
                # realization and any hidden CRA representation dependency.
                contextual_tokens=None,
                temporal_prior=None,
            )

            required_keys = {
                # V1.11 generic-input / counterfactual interface.
                "qbound_version",
                "generic_input_adapter_enabled",
                "generic_input_position_enabled",
                "generic_input_context_reuse",
                "generic_input_raw_rms",
                "generic_input_adapted_rms",
                "generic_input_token_cosine",
                "generic_input_delta_ratio",
                "generic_context_raw_rms",
                "generic_context_adapted_rms",
                "counterfactual_evidence_interface_enabled",
                "counterfactual_mass_preserving_gradient_enabled",
                "counterfactual_raw_keep_support",
                "counterfactual_raw_remove_support",
                "counterfactual_raw_keep_fraction",
                "counterfactual_keep_support",
                "counterfactual_remove_support",
                "counterfactual_keep_fraction",
                "counterfactual_remove_fraction",
                "counterfactual_support_forward_error",
                "counterfactual_support_complement_error",

                # Official V1.11 MAIN (BAPGR/PPBR/PSRC retained).
                "event_temporal_prob",
                "main_temporal_prob",
                "main_temporal_logits",
                "calibrated_temporal_prob",
                "calibrated_temporal_logits",
                "calibrated_soft_support",

                # Raw fine and coarse branches.
                "raw_refined_temporal_prob",
                "raw_refined_temporal_logits",
                "refined_temporal_prob",
                "refined_temporal_logits",
                "composed_temporal_prob",
                "composed_temporal_logits",
                "legacy_v15_composed_temporal_prob",
                "legacy_v14_event_temporal_prob",

                # V1.10.2 routing/reliability calibration.
                "stable_refiner_enabled",
                "uncertainty_calibration_enabled",
                "minimal_sufficient_enabled",
                "all_span_refine_marginal_enabled",
                "refine_hard_coarse_route",
                "refine_coarse_detached",
                "refine_selected_span_index",
                "calibration_uncertainty",
                "calibration_information",
                "calibration_confidence",
                "calibration_confidence_detached",
                "calibration_reliability",
                "refiner_information_gain",
                "refined_semantic_retention",
                "base_refiner_confidence",
                "reliability_gain_factor",

                # V1.10.2 BAPGR + PPBR pair prior / MAIN pair posterior.
                "posterior_guided_refinement_enabled",
                "boundary_structured_refinement_enabled",
                "prior_preserving_bounded_refinement_enabled",
                "posterior_stable_reliability_enabled",
                "refine_boundary_anchor_power",
                "refine_correction_cap",
                "composer_pair_prior",
                "composer_uniform_pair_prior",
                "composer_endpoint_pair_prior",
                "composer_pair_prior_valid_mask",
                "composer_pair_prior_entropy",
                "composer_pair_prior_entropy_ratio",
                "composer_pair_prior_effective_count",
                "composer_pair_prior_top1_posterior",
                "composer_pair_prior_candidate_count",
                "composer_uniform_pair_prior_entropy",
                "composer_uniform_pair_prior_entropy_ratio",
                "composer_uniform_pair_prior_effective_count",
                "composer_uniform_pair_prior_top1_posterior",
                "composer_pair_prior_expected_span_length",
                "composer_uniform_pair_prior_expected_span_length",
                "composer_pair_prior_length_gain",
                "composer_pair_soft_support",
                "composer_expected_start_frame",
                "composer_expected_end_frame",
                "composer_expected_span_length",
                "main_pair_posterior",
                "main_pair_entropy",
                "main_pair_effective_count",
                "main_pair_top1_posterior",

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

                # COMPOSE.
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

                # Raw REFINE posterior.
                "refiner_enabled",
                "refine_start_logits",
                "refine_end_logits",
                "refine_start_candidate_mask",
                "refine_end_candidate_mask",
                "refine_selected_pair_valid_mask",
                "refined_soft_support",
                "refined_support_log_bias",
                "refine_selected_pair_posterior",
                "refine_selected_pair_entropy",
                "refine_selected_pair_effective_count",
                "refine_selected_pair_top1_posterior",
                "refine_selected_pair_candidate_count",
                "refine_selected_pair_entropy_ratio",
                "refine_selected_start_candidate_count",
                "refine_selected_end_candidate_count",

                # V1.10.2 PPBR diagnostics.
                "refiner_raw_correction",
                "refiner_bounded_correction",
                "refiner_raw_correction_abs_mean",
                "refiner_raw_correction_abs_max",
                "refiner_bounded_correction_abs_mean",
                "refiner_bounded_correction_abs_max",
                "refiner_correction_saturation_ratio",
                "refiner_vs_prior_kl",
                "refiner_vs_prior_js",
                "expected_refined_span_length",
                "raw_vs_prior_expected_length_ratio",
                "expected_refined_length_ratio",
                "expected_refined_span_fraction",
                "expected_refine_trim_fraction",
                "expected_refined_start_frame",
                "expected_refined_end_frame",

                # Coarse and raw-refined hard spans.
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

                # Official V1.10.2 calibrated hard span.
                "calibrated_start_frame_float",
                "calibrated_end_frame_float",
                "calibrated_expected_span_length",
                "calibrated_expected_length_ratio",
                "calibrated_span_start_frame",
                "calibrated_span_end_frame",
                "calibrated_frame_mask",
                "calibrated_span_frame_length",
                "calibrated_span_fraction",
                "calibrated_retained_fraction",
                "calibrated_trim_left_frames",
                "calibrated_trim_right_frames",
                "calibrated_trim_total_frames",
                "calibrated_span_start",
                "calibrated_span_end",
                "calibrated_span_temporal_prob",
                "calibrated_inside_coarse",
                "refiner_temperature",
            }

            missing = sorted(required_keys.difference(qbound_aux))
            if missing:
                raise KeyError(
                    "Q-Bound V1.11 output is missing: "
                    + ", ".join(missing)
                )

            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore V1.11 runtime version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )
            if float(
                qbound_aux["generic_input_adapter_enabled"].detach().item()
            ) != float(
                bool(
                    self.cfgs.get("model", {})
                    .get("qbound", {})
                    .get("use_generic_input_adapter", True)
                )
            ):
                raise RuntimeError(
                    "Q-Bound V1.11 generic input adapter flag does not "
                    "match model.qbound.use_generic_input_adapter."
                )
            if float(
                qbound_aux["generic_input_context_reuse"].detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Controlled V1.11 expects Q-Bound to reuse one adapted "
                    "generic visual stream for raw/context states."
                )

            configured_cf_interface = float(
                self.qbound_counterfactual_evidence_interface
            )
            actual_cf_interface = float(
                qbound_aux[
                    "counterfactual_evidence_interface_enabled"
                ].detach().item()
            )
            if actual_cf_interface != configured_cf_interface:
                raise RuntimeError(
                    "Q-Bound V1.11 counterfactual interface flag mismatch: "
                    f"decoder={configured_cf_interface}, "
                    f"core={actual_cf_interface}."
                )

            actual_mass_guard = float(
                qbound_aux[
                    "counterfactual_mass_preserving_gradient_enabled"
                ].detach().item()
            )
            if actual_mass_guard != configured_cf_interface:
                raise RuntimeError(
                    "Q-Bound V1.11-B mass-preserving counterfactual "
                    "gradient guard does not match the configured interface."
                )

            cf_forward_error = float(
                qbound_aux[
                    "counterfactual_support_forward_error"
                ].detach().max().item()
            )
            cf_complement_error = float(
                qbound_aux[
                    "counterfactual_support_complement_error"
                ].detach().max().item()
            )
            if cf_forward_error > 1e-6:
                raise RuntimeError(
                    "V1.11-B counterfactual gradient view changed the "
                    "forward MAIN support; max error="
                    f"{cf_forward_error}."
                )
            if cf_complement_error > 1e-6:
                raise RuntimeError(
                    "V1.11-B keep/remove supports are not complementary; "
                    f"max error={cf_complement_error}."
                )

            # Fail early if a V1.8/V1.6 routing contract is accidentally loaded.
            if float(
                qbound_aux["uncertainty_calibration_enabled"]
                .detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 requires calibrated coarse/fine fusion."
                )
            if float(
                qbound_aux["posterior_guided_refinement_enabled"]
                .detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 requires detached posterior-guided REFINE."
                )
            if float(
                qbound_aux["boundary_structured_refinement_enabled"]
                .detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 requires boundary-anchored "
                    "posterior-guided refinement."
                )
            if float(
                qbound_aux["prior_preserving_bounded_refinement_enabled"]
                .detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 requires prior-preserving "
                    "bounded refinement (PPBR)."
                )
            if float(
                qbound_aux["posterior_stable_reliability_enabled"]
                .detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 requires posterior-stable reliability calibration."
                )
            # NOTE:
            # qbound_cfg/refinement_cfg are local variables from __init__ and
            # are not in scope inside forward(). Re-read the immutable runtime
            # config from self.cfgs here to keep this fail-fast check valid.
            runtime_qbound_cfg = (
                self.cfgs.get("model", {}).get("qbound", {})
            )
            if not isinstance(runtime_qbound_cfg, dict):
                raise TypeError(
                    "model.qbound must be a mapping/dict."
                )

            runtime_refinement_cfg = runtime_qbound_cfg.get(
                "refinement",
                {},
            )
            if not isinstance(runtime_refinement_cfg, dict):
                raise TypeError(
                    "model.qbound.refinement must be a mapping/dict."
                )

            configured_anchor_power = float(
                runtime_refinement_cfg.get(
                    "boundary_anchor_power",
                    runtime_qbound_cfg.get(
                        "refiner_boundary_anchor_power",
                        2.0,
                    ),
                )
            )
            actual_anchor_power = float(
                qbound_aux["refine_boundary_anchor_power"].detach().item()
            )
            if not math.isclose(
                actual_anchor_power,
                configured_anchor_power,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Q-Bound V1.10.2 boundary anchor power mismatch: "
                    f"config={configured_anchor_power}, core={actual_anchor_power}."
                )

            configured_correction_cap = float(
                runtime_refinement_cfg.get(
                    "correction_cap",
                    runtime_qbound_cfg.get(
                        "refiner_correction_cap",
                        2.0,
                    ),
                )
            )
            actual_correction_cap = float(
                qbound_aux["refine_correction_cap"].detach().item()
            )
            if not math.isclose(
                actual_correction_cap,
                configured_correction_cap,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Q-Bound V1.10.2 correction cap mismatch: "
                    f"config={configured_correction_cap}, "
                    f"core={actual_correction_cap}."
                )
            if float(
                qbound_aux["minimal_sufficient_enabled"]
                .detach().item()
            ) != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 must not enable V1.7 minimal-sufficient routing."
                )
            if float(
                qbound_aux["all_span_refine_marginal_enabled"]
                .detach().item()
            ) != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 must keep V1.6 differentiable all-span "
                    "REFINE marginalization disabled."
                )
            if float(
                qbound_aux["refine_coarse_detached"]
                .detach().item()
            ) != 1.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 requires the Composer posterior routed into "
                    "REFINE to remain detached."
                )
            if float(
                qbound_aux["refine_hard_coarse_route"]
                .detach().item()
            ) != 0.0:
                raise RuntimeError(
                    "Q-Bound V1.10.2 must not restrict REFINE to the V1.8 hard "
                    "top-1 Composer chain."
                )

            # ----------------------------------------------------------
            # 3a) OFFICIAL V1.10.2 reliability-calibrated grounding branch.
            #
            # This branch stays differentiable. Its temporal probability
            # already contains both:
            #   coarse COMPOSE contribution -> Composer task gradient
            #   raw REFINE contribution     -> Refiner task gradient
            # ----------------------------------------------------------
            qbound_time = (
                self.grounding_module
                .finalize_refined_grounding(
                    temporal_probability=qbound_aux[
                        "calibrated_temporal_prob"
                    ],
                    refined_span_start_frame=qbound_aux[
                        "calibrated_span_start_frame"
                    ],
                    refined_span_end_frame=qbound_aux[
                        "calibrated_span_end_frame"
                    ],
                    valid_mask=valid_mask,
                    refined_frame_mask=qbound_aux[
                        "calibrated_frame_mask"
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

            calibrated_qbound_key = self._normalize_temporal_distribution(
                qbound_time["key"],
                valid_mask,
            )
            qbound_key = calibrated_qbound_key
            final_key = calibrated_qbound_key

            # ----------------------------------------------------------
            # 3b) RAW REFINE / legacy hard-top1 COARSE / historical diagnostics.
            #
            # These readouts are detached. The RAW branch now uses the V1.10.2
            # posterior-guided Refiner; the COARSE branch deliberately keeps
            # the Composer hard top-1 interval as a controlled diagnostic.
            # The official MAIN is the reliability-calibrated branch above.
            # ----------------------------------------------------------
            with torch.no_grad():
                raw_refined_qbound_time = (
                    self.grounding_module
                    .finalize_refined_grounding(
                        temporal_probability=qbound_aux[
                            "raw_refined_temporal_prob"
                        ].detach(),
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
                raw_refined_qbound_key = (
                    self._normalize_temporal_distribution(
                        raw_refined_qbound_time["key"],
                        valid_mask,
                    )
                )

                coarse_qbound_time = (
                    self.grounding_module
                    .finalize_composed_grounding(
                        temporal_probability=qbound_aux[
                            "composed_temporal_prob"
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
                coarse_qbound_key = (
                    self._normalize_temporal_distribution(
                        coarse_qbound_time["key"],
                        valid_mask,
                    )
                )

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

            # SAME x_bar, SAME projection layer, SAME dropout realization.
            calibrated_qbound_video_proj = self._project_grounded_video(
                x_bar=x_bar,
                temporal_weights=calibrated_qbound_key,
                valid_mask=valid_mask,
                do=do,
                projection_dropout_mask=projection_dropout_mask,
                x_star_proj=x_star_proj,
            )

            # ----------------------------------------------------------
            # 3c) V1.11-B COUNTERFACTUAL EVIDENCE CONSUMER.
            #
            # The core's mass-preserving keep/remove support is used to
            # construct two feature-level interventions:
            #
            #   X_suff = keep * X + remove * X_donor
            #   X_nec  = remove * X + keep * X_donor
            #
            # Controlled-gradient contract:
            #   * X / donor X are detached CRA representations;
            #   * donor assignment is a deterministic cyclic derangement;
            #   * MAIN temporal pooling weights are detached;
            #   * CRA video_proj weights are detached;
            #   * VisualIntervention must remain OFF;
            #   * gradient therefore flows from future CF answer losses mainly
            #     into Q-Bound keep/remove support.
            # ----------------------------------------------------------
            counterfactual_sufficient_video_proj = None
            counterfactual_necessary_video_proj = None
            counterfactual_donor_available = torch.as_tensor(
                0.0,
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            counterfactual_donor_shift = torch.as_tensor(
                0.0,
                dtype=x_bar.dtype,
                device=x_bar.device,
            )
            counterfactual_donor_valid_overlap = torch.as_tensor(
                1.0,
                dtype=x_bar.dtype,
                device=x_bar.device,
            )

            if self.qbound_counterfactual_evidence_interface:
                if do:
                    raise RuntimeError(
                        "Controlled Q-Bound V1.11-B counterfactual training "
                        "requires model.do=false. VisualIntervention must not "
                        "enter the counterfactual consumer."
                    )

                keep_support = qbound_aux[
                    "counterfactual_keep_support"
                ]
                remove_support = qbound_aux[
                    "counterfactual_remove_support"
                ]

                if keep_support.shape != valid_mask.shape:
                    raise ValueError(
                        "counterfactual_keep_support must match [B,T]."
                    )
                if remove_support.shape != valid_mask.shape:
                    raise ValueError(
                        "counterfactual_remove_support must match [B,T]."
                    )

                factual_cf_sequence = x_bar.detach()

                if batch_size > 1:
                    donor_cf_sequence = torch.roll(
                        factual_cf_sequence,
                        shifts=1,
                        dims=0,
                    )
                    donor_valid_mask = torch.roll(
                        valid_mask,
                        shifts=1,
                        dims=0,
                    )
                    counterfactual_donor_available = torch.as_tensor(
                        1.0,
                        dtype=x_bar.dtype,
                        device=x_bar.device,
                    )
                    counterfactual_donor_shift = torch.as_tensor(
                        1.0,
                        dtype=x_bar.dtype,
                        device=x_bar.device,
                    )
                else:
                    # A one-sample minibatch has no valid in-batch donor.
                    # Keep the forward finite and expose availability=0 so the
                    # pipeline can skip the counterfactual loss for this batch.
                    donor_cf_sequence = factual_cf_sequence
                    donor_valid_mask = valid_mask

                receiver_valid = valid_mask.to(x_bar.dtype)
                donor_valid = donor_valid_mask.to(x_bar.dtype)
                valid_intersection = receiver_valid * donor_valid
                counterfactual_donor_valid_overlap = (
                    valid_intersection.sum(dim=-1)
                    / receiver_valid.sum(dim=-1).clamp_min(1.0)
                ).mean()

                # NExT-GQA controlled V1.11-B uses fixed T=32 masks.  Fail
                # loudly rather than silently mixing padded donor positions.
                if batch_size > 1 and not torch.equal(
                    donor_valid_mask,
                    valid_mask,
                ):
                    raise RuntimeError(
                        "Controlled V1.11-B cyclic donor requires identical "
                        "valid temporal masks within the minibatch. Current "
                        "NExT-GQA loader should provide fixed T=32 masks."
                    )

                keep_3d = keep_support.unsqueeze(-1)
                remove_3d = remove_support.unsqueeze(-1)

                counterfactual_sufficient_sequence = (
                    keep_3d * factual_cf_sequence
                    + remove_3d * donor_cf_sequence
                )
                counterfactual_necessary_sequence = (
                    remove_3d * factual_cf_sequence
                    + keep_3d * donor_cf_sequence
                )

                counterfactual_sufficient_video_proj = (
                    self._project_counterfactual_sequence(
                        sequence=counterfactual_sufficient_sequence,
                        temporal_weights=calibrated_qbound_key,
                        valid_mask=valid_mask,
                        projection_dropout_mask=projection_dropout_mask,
                    )
                )
                counterfactual_necessary_video_proj = (
                    self._project_counterfactual_sequence(
                        sequence=counterfactual_necessary_sequence,
                        temporal_weights=calibrated_qbound_key,
                        valid_mask=valid_mask,
                        projection_dropout_mask=projection_dropout_mask,
                    )
                )

            # Diagnostic projections do not define training objectives.
            with torch.no_grad():
                raw_refined_qbound_video_proj = (
                    self._project_grounded_video(
                        x_bar=x_bar,
                        temporal_weights=raw_refined_qbound_key,
                        valid_mask=valid_mask,
                        do=do,
                        projection_dropout_mask=projection_dropout_mask,
                        x_star_proj=x_star_proj,
                    )
                )
                coarse_qbound_video_proj = (
                    self._project_grounded_video(
                        x_bar=x_bar,
                        temporal_weights=coarse_qbound_key,
                        valid_mask=valid_mask,
                        do=do,
                        projection_dropout_mask=projection_dropout_mask,
                        x_star_proj=x_star_proj,
                    )
                )

            # Compatibility: "refined" now aliases calibrated MAIN.
            refined_qbound_video_proj = calibrated_qbound_video_proj
            qbound_video_proj = calibrated_qbound_video_proj
            video_proj = calibrated_qbound_video_proj

        else:
            coarse_qbound_key = cra_key
            raw_refined_qbound_key = cra_key
            calibrated_qbound_key = cra_key
            qbound_key = cra_key
            final_key = cra_key

            coarse_qbound_video_proj = cra_video_proj
            raw_refined_qbound_video_proj = cra_video_proj
            calibrated_qbound_video_proj = cra_video_proj
            refined_qbound_video_proj = cra_video_proj
            qbound_video_proj = cra_video_proj
            video_proj = cra_video_proj

        # --------------------------------------------------------------
        # 4) Final temporal contract.
        # --------------------------------------------------------------
        time_param = self._build_time_param(
            final_key=final_key,
            base_temporal=base_temporal,
            qbound_key=qbound_key,
            coarse_qbound_key=coarse_qbound_key,
            raw_refined_qbound_key=raw_refined_qbound_key,
            qbound_active=qbound_active,
            window_sizes=window_sizes,
            qbound_time=qbound_time,
        )

        # Official answer feature is the V1.10.2 reliability-calibrated MAIN branch.
        time_param["grounding_feature"] = video_proj
        time_param["qbound_feature"] = video_proj
        time_param["qbound_grounding_feature"] = (
            calibrated_qbound_video_proj
        )
        time_param["qbound_temporal_main_feature"] = (
            calibrated_qbound_video_proj
        )

        # Explicit V1.10.2 answer-space branches.
        time_param["qbound_calibrated_grounding_feature"] = (
            calibrated_qbound_video_proj
        )
        time_param["qbound_calibrated_video_proj"] = (
            calibrated_qbound_video_proj
        )
        time_param["qbound_raw_refined_grounding_feature"] = (
            raw_refined_qbound_video_proj
        )
        time_param["qbound_raw_refined_video_proj"] = (
            raw_refined_qbound_video_proj
        )
        time_param["qbound_coarse_grounding_feature"] = (
            coarse_qbound_video_proj
        )
        time_param["qbound_coarse_video_proj"] = (
            coarse_qbound_video_proj
        )

        # Compatibility: refined == calibrated MAIN in V1.10.2.
        time_param["qbound_refined_video_proj"] = (
            calibrated_qbound_video_proj
        )
        time_param["cra_grounding_feature"] = cra_video_proj

        # No direct event-token residual into answer representation.
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

        time_param["qbound_reasoned_video_feature"] = (
            calibrated_qbound_video_proj
        )

        if qbound_aux is not None:
            # ----------------------------------------------------------
            # 5) V1.11 generic-input / counterfactual evidence interface.
            # These diagnostics establish that Q-Bound consumes the explicit
            # pre-CRA feature stream. Counterfactual supports are exposed only
            # as an interface here; answer-level intervention/loss belongs to
            # the downstream consumer, not to QBoundCore/RefineDecoder.
            # ----------------------------------------------------------
            v111_interface_keys = [
                "generic_input_adapter_enabled",
                "generic_input_position_enabled",
                "generic_input_context_reuse",
                "generic_input_raw_rms",
                "generic_input_adapted_rms",
                "generic_input_token_cosine",
                "generic_input_delta_ratio",
                "generic_context_raw_rms",
                "generic_context_adapted_rms",
                "counterfactual_evidence_interface_enabled",
                "counterfactual_mass_preserving_gradient_enabled",
                "counterfactual_raw_keep_support",
                "counterfactual_raw_remove_support",
                "counterfactual_raw_keep_fraction",
                "counterfactual_keep_support",
                "counterfactual_remove_support",
                "counterfactual_keep_fraction",
                "counterfactual_remove_fraction",
                "counterfactual_support_forward_error",
                "counterfactual_support_complement_error",
            ]
            for key in v111_interface_keys:
                time_param[f"qbound_{key}"] = qbound_aux[key]

            time_param["qbound_input_is_generic"] = torch.as_tensor(
                1.0,
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
            time_param["qbound_input_uses_cra_embedding"] = torch.as_tensor(
                0.0,
                dtype=video_proj.dtype,
                device=video_proj.device,
            )

            qbound_aux["qbound_input_is_generic"] = time_param[
                "qbound_input_is_generic"
            ]
            qbound_aux["qbound_input_uses_cra_embedding"] = time_param[
                "qbound_input_uses_cra_embedding"
            ]

            # ----------------------------------------------------------
            # 5b) V1.11-B counterfactual consumer outputs.
            # ----------------------------------------------------------
            time_param["qbound_counterfactual_donor_available"] = (
                counterfactual_donor_available
            )
            time_param["qbound_counterfactual_donor_shift"] = (
                counterfactual_donor_shift
            )
            time_param["qbound_counterfactual_donor_valid_overlap"] = (
                counterfactual_donor_valid_overlap
            )
            time_param["qbound_counterfactual_consumer_frozen"] = (
                torch.as_tensor(
                    float(
                        self.qbound_counterfactual_evidence_interface
                    ),
                    dtype=video_proj.dtype,
                    device=video_proj.device,
                )
            )

            qbound_aux["qbound_counterfactual_donor_available"] = (
                counterfactual_donor_available
            )
            qbound_aux["qbound_counterfactual_donor_shift"] = (
                counterfactual_donor_shift
            )
            qbound_aux["qbound_counterfactual_donor_valid_overlap"] = (
                counterfactual_donor_valid_overlap
            )
            qbound_aux["qbound_counterfactual_consumer_frozen"] = (
                time_param["qbound_counterfactual_consumer_frozen"]
            )

            if self.qbound_counterfactual_evidence_interface:
                if counterfactual_sufficient_video_proj is None:
                    raise RuntimeError(
                        "V1.11-B sufficiency representation was not built."
                    )
                if counterfactual_necessary_video_proj is None:
                    raise RuntimeError(
                        "V1.11-B necessity representation was not built."
                    )

                time_param[
                    "qbound_counterfactual_factual_video_proj"
                ] = calibrated_qbound_video_proj
                time_param[
                    "qbound_counterfactual_sufficient_video_proj"
                ] = counterfactual_sufficient_video_proj
                time_param[
                    "qbound_counterfactual_necessary_video_proj"
                ] = counterfactual_necessary_video_proj

                qbound_aux[
                    "qbound_counterfactual_factual_video_proj"
                ] = calibrated_qbound_video_proj
                qbound_aux[
                    "qbound_counterfactual_sufficient_video_proj"
                ] = counterfactual_sufficient_video_proj
                qbound_aux[
                    "qbound_counterfactual_necessary_video_proj"
                ] = counterfactual_necessary_video_proj

                factual_detached = calibrated_qbound_video_proj.detach()
                suff_detached = (
                    counterfactual_sufficient_video_proj.detach()
                )
                nec_detached = (
                    counterfactual_necessary_video_proj.detach()
                )

                suff_cosine = F.cosine_similarity(
                    factual_detached,
                    suff_detached,
                    dim=-1,
                )
                nec_cosine = F.cosine_similarity(
                    factual_detached,
                    nec_detached,
                    dim=-1,
                )
                suff_l2 = torch.linalg.vector_norm(
                    suff_detached - factual_detached,
                    dim=-1,
                )
                nec_l2 = torch.linalg.vector_norm(
                    nec_detached - factual_detached,
                    dim=-1,
                )

                time_param[
                    "qbound_counterfactual_sufficient_factual_cosine"
                ] = suff_cosine
                time_param[
                    "qbound_counterfactual_necessary_factual_cosine"
                ] = nec_cosine
                time_param[
                    "qbound_counterfactual_sufficient_factual_l2"
                ] = suff_l2
                time_param[
                    "qbound_counterfactual_necessary_factual_l2"
                ] = nec_l2

                qbound_aux[
                    "qbound_counterfactual_sufficient_factual_cosine"
                ] = suff_cosine
                qbound_aux[
                    "qbound_counterfactual_necessary_factual_cosine"
                ] = nec_cosine
                qbound_aux[
                    "qbound_counterfactual_sufficient_factual_l2"
                ] = suff_l2
                qbound_aux[
                    "qbound_counterfactual_necessary_factual_l2"
                ] = nec_l2

            # ----------------------------------------------------------
            # 6) Generic event / weak-alignment outputs.
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
            # 7) RELATE / Q-OEI outputs.
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
            # 8) DECOMPOSE outputs.
            # ----------------------------------------------------------
            decompose_direct = [
                "event_assignment",
                "event_mass",
                "event_mass_max",
                "event_mass_min",
                "event_mass_mean",
                "effective_event_count",
                "soft_effective_event_count",
                "decomp_loss",
                "decomp_expected_energy",
                "decomp_selected_coherence",
                "decomp_selected_complexity",
                "decomp_selected_complexity_cost",
                "decomp_selected_boundary_reward",
                "decomp_selected_boundary_term",
                "decomp_selected_total_energy",
                "decomp_expected_event_count",
                "partition_entropy",
                "candidate_split_gain",
                "accepted_split_gain",
                "rejected_split_gain",
                "boundary_prob",
                "boundary_hard",
                "boundary_soft_marginal",
                "boundary_candidate_indices",
                "boundary_candidate_probs",
                "boundary_candidate_valid_mask",
            ]
            for key in decompose_direct:
                if key in qbound_aux:
                    time_param[f"qbound_{key}"] = qbound_aux[key]

            # ----------------------------------------------------------
            # 9) COMPOSE outputs.
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
                "legacy_v15_composed_temporal_prob",
                "legacy_v15_composed_temporal_logits",
                "legacy_v14_event_temporal_prob",
            ]
            for key in composer_keys:
                if key in qbound_aux:
                    time_param[f"qbound_{key}"] = qbound_aux[key]

            # ----------------------------------------------------------
            # ----------------------------------------------------------
            # 10) V1.11 BAPGR + PPBR REFINE + POSTERIOR-STABLE RELIABILITY outputs.
            # ----------------------------------------------------------
            refiner_keys = [
                "refiner_enabled",
                "stable_refiner_enabled",
                "uncertainty_calibration_enabled",
                "minimal_sufficient_enabled",
                "all_span_refine_marginal_enabled",
                "refine_hard_coarse_route",
                "refine_coarse_detached",
                "refine_selected_span_index",

                "refine_start_logits",
                "refine_end_logits",
                "refine_start_candidate_mask",
                "refine_end_candidate_mask",
                "refine_selected_pair_valid_mask",
                "refine_selected_pair_posterior",
                "refine_selected_pair_entropy",
                "refine_selected_pair_effective_count",
                "refine_selected_pair_top1_posterior",
                "refine_selected_pair_candidate_count",
                "refine_selected_pair_entropy_ratio",
                "refine_selected_start_candidate_count",
                "refine_selected_end_candidate_count",

                # V1.10.2 PPBR diagnostics.
                "refiner_raw_correction",
                "refiner_bounded_correction",
                "refiner_raw_correction_abs_mean",
                "refiner_raw_correction_abs_max",
                "refiner_bounded_correction_abs_mean",
                "refiner_bounded_correction_abs_max",
                "refiner_correction_saturation_ratio",
                "refiner_vs_prior_kl",
                "refiner_vs_prior_js",

                "calibration_uncertainty",
                "calibration_information",
                "calibration_confidence",
                "calibration_confidence_detached",
                "calibration_reliability",
                "refiner_information_gain",
                "refined_semantic_retention",
                "base_refiner_confidence",
                "reliability_gain_factor",

                # V1.10.2 BAPGR + PPBR prior / posterior-stable route.
                "posterior_guided_refinement_enabled",
                "boundary_structured_refinement_enabled",
                "prior_preserving_bounded_refinement_enabled",
                "posterior_stable_reliability_enabled",
                "refine_boundary_anchor_power",
                "refine_correction_cap",
                "composer_pair_prior",
                "composer_uniform_pair_prior",
                "composer_endpoint_pair_prior",
                "composer_pair_prior_valid_mask",
                "composer_pair_prior_entropy",
                "composer_pair_prior_entropy_ratio",
                "composer_pair_prior_effective_count",
                "composer_pair_prior_top1_posterior",
                "composer_pair_prior_candidate_count",
                "composer_uniform_pair_prior_entropy",
                "composer_uniform_pair_prior_entropy_ratio",
                "composer_uniform_pair_prior_effective_count",
                "composer_uniform_pair_prior_top1_posterior",
                "composer_pair_prior_expected_span_length",
                "composer_uniform_pair_prior_expected_span_length",
                "composer_pair_prior_length_gain",
                "composer_pair_soft_support",
                "composer_expected_start_frame",
                "composer_expected_end_frame",
                "composer_expected_span_length",
                "main_pair_posterior",
                "main_pair_entropy",
                "main_pair_effective_count",
                "main_pair_top1_posterior",

                # Raw-refined differentiable statistics/diagnostics.
                "refine_conditional_support",
                "refined_soft_support",
                "refined_support_log_bias",
                "refined_temporal_logits",
                "refined_temporal_prob",
                "raw_refined_temporal_logits",
                "raw_refined_temporal_prob",
                "refine_expected_pair_entropy",
                "expected_refined_span_length",
                "raw_vs_prior_expected_length_ratio",
                "expected_refined_length_ratio",
                "expected_refined_span_fraction",
                "expected_refine_trim_fraction",
                "expected_refined_start_frame",
                "expected_refined_end_frame",

                # Official calibrated soft branch.
                "calibrated_soft_support",
                "calibrated_temporal_logits",
                "calibrated_temporal_prob",
                "main_temporal_logits",
                "main_temporal_prob",

                # Coarse interval.
                "coarse_span_start_frame",
                "coarse_span_end_frame",
                "coarse_span_frame_length",
                "coarse_span_fraction",
                "coarse_frame_mask",

                # Raw V1.10.2 BAPGR + PPBR hard-refined interval.
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

                # Official V1.10.2 calibrated hard interval.
                "calibrated_start_frame_float",
                "calibrated_end_frame_float",
                "calibrated_expected_span_length",
                "calibrated_expected_length_ratio",
                "calibrated_span_start_frame",
                "calibrated_span_end_frame",
                "calibrated_frame_mask",
                "calibrated_span_frame_length",
                "calibrated_span_fraction",
                "calibrated_retained_fraction",
                "calibrated_trim_left_frames",
                "calibrated_trim_right_frames",
                "calibrated_trim_total_frames",
                "calibrated_span_start",
                "calibrated_span_end",
                "calibrated_span_temporal_prob",
                "calibrated_inside_coarse",

                "refiner_temperature",
            ]
            for key in refiner_keys:
                if key in qbound_aux:
                    time_param[f"qbound_{key}"] = qbound_aux[key]

            # ----------------------------------------------------------
            # 10) Grounding-adapter diagnostics from calibrated MAIN key.
            # ----------------------------------------------------------
            grounding_diag_keys = [
                "adaptive_span_used",
                "adaptive_span_start_frame",
                "adaptive_span_end_frame",
                "adaptive_span_length",
                "adaptive_span_fraction",
                "adaptive_span_key_mass",
                "adaptive_span_peak_in_span",
                "adaptive_span_composer_mask",
                "adaptive_span_mask_consistency",
                "refined_span_used",
                "refined_span_start_frame",
                "refined_span_end_frame",
                "refined_span_length",
                "refined_span_fraction",
                "refined_span_key_mass",
                "refined_span_peak_in_span",
                "refined_span_core_mask",
                "refined_span_mask_consistency",
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
                "legacy_fixed_window_max_indices",
                "legacy_fixed_window_start",
                "legacy_fixed_window_end",
                "legacy_fixed_window_mask",
                "legacy_fixed_window_span_length",
                "legacy_fixed_window_key_mass",
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

            # ----------------------------------------------------------
            # 10b) Explicit V1.10.2 CALIBRATED MAIN adapter aliases.
            #
            # `time_param["start"/"end"/"mask"]` are already the official
            # calibrated MAIN interval. These explicit aliases are required
            # by CRA.forward's V1.9 contract and make the three branches
            # symmetric:
            #
            #   calibrated MAIN / raw-refined / coarse.
            # ----------------------------------------------------------
            time_param["qbound_calibrated_pre_smooth_key"] = (
                qbound_time["pre_smooth_key"]
            )
            time_param["qbound_calibrated_post_smooth_key"] = (
                calibrated_qbound_key
            )
            time_param["qbound_calibrated_start"] = (
                qbound_time["start"]
            )
            time_param["qbound_calibrated_end"] = (
                qbound_time["end"]
            )
            time_param["qbound_calibrated_mask"] = (
                qbound_time["mask"]
            )
            time_param["qbound_calibrated_span_length"] = (
                qbound_time["refined_span_length"]
            )
            time_param["qbound_calibrated_span_fraction"] = (
                qbound_time["refined_span_fraction"]
            )
            time_param["qbound_calibrated_span_key_mass"] = (
                qbound_time["refined_span_key_mass"]
            )
            # V1.10.2 reliability aliases from the core.
            time_param["qbound_calibration_reliability"] = (
                qbound_aux["calibration_reliability"]
            )
            time_param["qbound_refiner_information_gain"] = (
                qbound_aux["refiner_information_gain"]
            )
            time_param["qbound_refined_semantic_retention"] = (
                qbound_aux["refined_semantic_retention"]
            )
            time_param["qbound_base_refiner_confidence"] = (
                qbound_aux["base_refiner_confidence"]
            )
            time_param["qbound_boundary_structured_refinement_enabled"] = (
                qbound_aux["boundary_structured_refinement_enabled"]
            )
            time_param[
                "qbound_prior_preserving_bounded_refinement_enabled"
            ] = qbound_aux["prior_preserving_bounded_refinement_enabled"]
            time_param["qbound_refine_boundary_anchor_power"] = (
                qbound_aux["refine_boundary_anchor_power"]
            )
            time_param["qbound_refine_correction_cap"] = (
                qbound_aux["refine_correction_cap"]
            )
            time_param["qbound_refiner_raw_correction_abs_mean"] = (
                qbound_aux["refiner_raw_correction_abs_mean"]
            )
            time_param["qbound_refiner_raw_correction_abs_max"] = (
                qbound_aux["refiner_raw_correction_abs_max"]
            )
            time_param["qbound_refiner_bounded_correction_abs_mean"] = (
                qbound_aux["refiner_bounded_correction_abs_mean"]
            )
            time_param["qbound_refiner_bounded_correction_abs_max"] = (
                qbound_aux["refiner_bounded_correction_abs_max"]
            )
            time_param["qbound_refiner_correction_saturation_ratio"] = (
                qbound_aux["refiner_correction_saturation_ratio"]
            )
            time_param["qbound_refiner_vs_prior_kl"] = (
                qbound_aux["refiner_vs_prior_kl"]
            )
            time_param["qbound_refiner_vs_prior_js"] = (
                qbound_aux["refiner_vs_prior_js"]
            )
            time_param["qbound_raw_vs_prior_expected_length_ratio"] = (
                qbound_aux["raw_vs_prior_expected_length_ratio"]
            )
            time_param["qbound_composer_uniform_pair_prior_entropy"] = (
                qbound_aux["composer_uniform_pair_prior_entropy"]
            )
            time_param["qbound_composer_uniform_pair_prior_entropy_ratio"] = (
                qbound_aux["composer_uniform_pair_prior_entropy_ratio"]
            )
            time_param["qbound_composer_uniform_pair_prior_effective_count"] = (
                qbound_aux["composer_uniform_pair_prior_effective_count"]
            )
            time_param["qbound_composer_uniform_pair_prior_top1_posterior"] = (
                qbound_aux["composer_uniform_pair_prior_top1_posterior"]
            )
            time_param["qbound_composer_pair_prior_expected_span_length"] = (
                qbound_aux["composer_pair_prior_expected_span_length"]
            )
            time_param[
                "qbound_composer_uniform_pair_prior_expected_span_length"
            ] = qbound_aux[
                "composer_uniform_pair_prior_expected_span_length"
            ]
            time_param["qbound_composer_pair_prior_length_gain"] = (
                qbound_aux["composer_pair_prior_length_gain"]
            )

            # Strong contract: the generic official interval must be exactly
            # the same tensors as the explicit calibrated aliases.
            if time_param["qbound_calibrated_start"] is not time_param["start"]:
                raise RuntimeError(
                    "V1.10.2 calibrated start alias does not match MAIN start."
                )
            if time_param["qbound_calibrated_end"] is not time_param["end"]:
                raise RuntimeError(
                    "V1.10.2 calibrated end alias does not match MAIN end."
                )
            if time_param["qbound_calibrated_mask"] is not time_param["mask"]:
                raise RuntimeError(
                    "V1.10.2 calibrated mask alias does not match MAIN mask."
                )

            # ----------------------------------------------------------
            # 11) Explicit detached hard-top1 COARSE branch diagnostics.
            # ----------------------------------------------------------
            time_param["qbound_coarse_pre_smooth_key"] = (
                coarse_qbound_time["pre_smooth_key"]
            )
            time_param["qbound_coarse_post_smooth_key"] = (
                coarse_qbound_key
            )
            time_param["qbound_coarse_start"] = (
                coarse_qbound_time["start"]
            )
            time_param["qbound_coarse_end"] = (
                coarse_qbound_time["end"]
            )
            time_param["qbound_coarse_mask"] = (
                coarse_qbound_time["mask"]
            )
            time_param["qbound_coarse_span_length"] = (
                coarse_qbound_time["adaptive_span_length"]
            )
            time_param["qbound_coarse_span_fraction"] = (
                coarse_qbound_time["adaptive_span_fraction"]
            )
            time_param["qbound_coarse_span_key_mass"] = (
                coarse_qbound_time["adaptive_span_key_mass"]
            )

            # ----------------------------------------------------------
            # 11b) RAW V1.10.2 BAPGR + PPBR diagnostic branch.
            # ----------------------------------------------------------
            time_param["qbound_raw_refined_pre_smooth_key"] = (
                raw_refined_qbound_time["pre_smooth_key"]
            )
            time_param["qbound_raw_refined_post_smooth_key"] = (
                raw_refined_qbound_key
            )
            time_param["qbound_raw_refined_start"] = (
                raw_refined_qbound_time["start"]
            )
            time_param["qbound_raw_refined_end"] = (
                raw_refined_qbound_time["end"]
            )
            time_param["qbound_raw_refined_mask"] = (
                raw_refined_qbound_time["mask"]
            )
            time_param["qbound_raw_refined_span_length"] = (
                raw_refined_qbound_time["refined_span_length"]
            )
            time_param["qbound_raw_refined_span_fraction"] = (
                raw_refined_qbound_time["refined_span_fraction"]
            )
            time_param["qbound_raw_refined_span_key_mass"] = (
                raw_refined_qbound_time["refined_span_key_mass"]
            )

            # Historical detached readouts.
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
            # ----------------------------------------------------------
            # 12) Populate qbound_aux with V1.10.2 decoder/adaptor interfaces.
            # ----------------------------------------------------------
            qbound_aux["cra_key"] = cra_key

            # Official calibrated MAIN.
            qbound_aux["event_key"] = calibrated_qbound_key
            qbound_aux["qbound_key"] = calibrated_qbound_key
            qbound_aux["qbound_calibrated_key"] = calibrated_qbound_key
            qbound_aux["final_key"] = final_key

            # Compatibility: refined_key now means calibrated MAIN.
            qbound_aux["refined_key"] = final_key

            # Explicit diagnostic temporal branches.
            qbound_aux["qbound_raw_refined_key"] = (
                raw_refined_qbound_key
            )
            qbound_aux["qbound_coarse_key"] = coarse_qbound_key

            qbound_aux["qbound_mix"] = torch.ones(
                batch_size,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux["qbound_handover_enabled"] = torch.as_tensor(
                0.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )

            # Branch flags.
            qbound_aux["qbound_adaptive_span_enabled"] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux["qbound_refined_span_enabled"] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux["qbound_calibrated_span_enabled"] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux["qbound_raw_refined_span_enabled"] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux["qbound_stable_refiner_enabled"] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux[
                "qbound_uncertainty_calibration_enabled"
            ] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux[
                "qbound_posterior_guided_refinement_enabled"
            ] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux[
                "qbound_boundary_structured_refinement_enabled"
            ] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux[
                "qbound_prior_preserving_bounded_refinement_enabled"
            ] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux[
                "qbound_posterior_stable_reliability_enabled"
            ] = torch.as_tensor(
                1.0,
                dtype=calibrated_qbound_key.dtype,
                device=calibrated_qbound_key.device,
            )
            qbound_aux["qbound_minimal_sufficient_enabled"] = (
                torch.as_tensor(
                    0.0,
                    dtype=calibrated_qbound_key.dtype,
                    device=calibrated_qbound_key.device,
                )
            )

            # ----------------------------------------------------------
            # V1.10.2 answer-space representations.
            # ----------------------------------------------------------
            qbound_aux["cra_video_proj"] = cra_video_proj

            qbound_aux["qbound_video_proj"] = (
                calibrated_qbound_video_proj
            )
            qbound_aux["main_video_proj"] = (
                calibrated_qbound_video_proj
            )
            qbound_aux["temporal_main_video_proj"] = (
                calibrated_qbound_video_proj
            )
            qbound_aux["qbound_calibrated_video_proj"] = (
                calibrated_qbound_video_proj
            )

            # Compatibility: "refined" == calibrated MAIN.
            qbound_aux["qbound_refined_video_proj"] = (
                calibrated_qbound_video_proj
            )

            # Diagnostic-only branch representations.
            qbound_aux["qbound_raw_refined_video_proj"] = (
                raw_refined_qbound_video_proj
            )
            qbound_aux["qbound_coarse_video_proj"] = (
                coarse_qbound_video_proj
            )

            # No direct event-token residual into answer space.
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

            # ----------------------------------------------------------
            # MAIN calibrated key before/after grounding-module smoothing.
            # ----------------------------------------------------------
            qbound_aux["qbound_pre_smooth_key"] = (
                qbound_time["pre_smooth_key"]
            )
            qbound_aux["qbound_post_smooth_key"] = (
                calibrated_qbound_key
            )
            qbound_aux["qbound_calibrated_pre_smooth_key"] = (
                qbound_time["pre_smooth_key"]
            )
            qbound_aux["qbound_calibrated_post_smooth_key"] = (
                calibrated_qbound_key
            )

            # RAW REFINE diagnostic key.
            qbound_aux["qbound_raw_refined_pre_smooth_key"] = (
                raw_refined_qbound_time["pre_smooth_key"]
            )
            qbound_aux["qbound_raw_refined_post_smooth_key"] = (
                raw_refined_qbound_key
            )

            # COARSE diagnostic key.
            qbound_aux["qbound_coarse_pre_smooth_key"] = (
                coarse_qbound_time["pre_smooth_key"]
            )
            qbound_aux["qbound_coarse_post_smooth_key"] = (
                coarse_qbound_key
            )

            # ----------------------------------------------------------
            # Official V1.10.2 calibrated interval.
            # ----------------------------------------------------------
            qbound_aux["qbound_calibrated_start"] = qbound_time["start"]
            qbound_aux["qbound_calibrated_end"] = qbound_time["end"]
            qbound_aux["qbound_calibrated_mask"] = qbound_time["mask"]
            qbound_aux["qbound_calibrated_span_length"] = qbound_time[
                "refined_span_length"
            ]
            qbound_aux["qbound_calibrated_span_fraction"] = qbound_time[
                "refined_span_fraction"
            ]
            qbound_aux["qbound_calibrated_span_key_mass"] = qbound_time[
                "refined_span_key_mass"
            ]

            # Compatibility: adaptive/refined aliases mean calibrated MAIN.
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

            # ----------------------------------------------------------
            # RAW V1.10.2 BAPGR + PPBR refined interval.
            # ----------------------------------------------------------
            qbound_aux["qbound_raw_refined_start"] = (
                raw_refined_qbound_time["start"]
            )
            qbound_aux["qbound_raw_refined_end"] = (
                raw_refined_qbound_time["end"]
            )
            qbound_aux["qbound_raw_refined_mask"] = (
                raw_refined_qbound_time["mask"]
            )
            qbound_aux["qbound_raw_refined_span_length"] = (
                raw_refined_qbound_time["refined_span_length"]
            )
            qbound_aux["qbound_raw_refined_span_fraction"] = (
                raw_refined_qbound_time["refined_span_fraction"]
            )
            qbound_aux["qbound_raw_refined_span_key_mass"] = (
                raw_refined_qbound_time["refined_span_key_mass"]
            )

            # ----------------------------------------------------------
            # COARSE interval/key.
            # ----------------------------------------------------------
            qbound_aux["qbound_coarse_start"] = (
                coarse_qbound_time["start"]
            )
            qbound_aux["qbound_coarse_end"] = (
                coarse_qbound_time["end"]
            )
            qbound_aux["qbound_coarse_mask"] = (
                coarse_qbound_time["mask"]
            )
            qbound_aux["qbound_coarse_span_length"] = (
                coarse_qbound_time["adaptive_span_length"]
            )
            qbound_aux["qbound_coarse_span_fraction"] = (
                coarse_qbound_time["adaptive_span_fraction"]
            )
            qbound_aux["qbound_coarse_span_key_mass"] = (
                coarse_qbound_time["adaptive_span_key_mass"]
            )

            # Same calibrated MAIN key + alternate hard readouts.
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

            qbound_aux[
                "qbound_refined_vs_coarse_span_length_ratio"
            ] = qbound_time["refined_vs_coarse_span_length_ratio"]
            qbound_aux[
                "qbound_refined_vs_legacy_span_length_ratio"
            ] = qbound_time["refined_vs_legacy_span_length_ratio"]
            qbound_aux[
                "qbound_refined_minus_coarse_key_mass"
            ] = qbound_time["refined_minus_coarse_key_mass"]
            qbound_aux[
                "qbound_refined_minus_legacy_key_mass"
            ] = qbound_time["refined_minus_legacy_key_mass"]
            qbound_aux[
                "qbound_refined_trimmed_frame_count"
            ] = qbound_time["refined_trimmed_frame_count"]

            # Historical detached readouts.
            if legacy_v15_time is not None:
                qbound_aux["v15_diagnostic_key"] = legacy_v15_time["key"]
                qbound_aux["v15_diagnostic_start"] = (
                    legacy_v15_time["start"]
                )
                qbound_aux["v15_diagnostic_end"] = (
                    legacy_v15_time["end"]
                )
                qbound_aux["v15_diagnostic_mask"] = (
                    legacy_v15_time["mask"]
                )

            if legacy_v14_time is not None:
                qbound_aux["v14_diagnostic_key"] = legacy_v14_time["key"]
                qbound_aux["v14_diagnostic_start"] = (
                    legacy_v14_time["start"]
                )
                qbound_aux["v14_diagnostic_end"] = (
                    legacy_v14_time["end"]
                )
                qbound_aux["v14_diagnostic_mask"] = (
                    legacy_v14_time["mask"]
                )

            # Key-distance diagnostics.
            qbound_aux["cra_qbound_key_l1"] = (
                (calibrated_qbound_key - cra_key)
                .abs().mean(dim=-1)
            )
            qbound_aux["coarse_calibrated_key_l1"] = (
                (coarse_qbound_key - calibrated_qbound_key)
                .abs().mean(dim=-1)
            )
            qbound_aux["raw_refined_calibrated_key_l1"] = (
                (raw_refined_qbound_key - calibrated_qbound_key)
                .abs().mean(dim=-1)
            )
            qbound_aux["coarse_raw_refined_key_l1"] = (
                (coarse_qbound_key - raw_refined_qbound_key)
                .abs().mean(dim=-1)
            )

            # Compatibility alias used by old loggers.
            qbound_aux["coarse_refined_key_l1"] = (
                qbound_aux["coarse_calibrated_key_l1"]
            )

            qbound_aux["question_only_conditioning"] = torch.as_tensor(
                float(question_feature is not None),
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
            qbound_aux["qbound_version"] = self.qbound_version

            # Old counterfactual/merge interfaces remain disabled.
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