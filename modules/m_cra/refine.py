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
        _, n_head, n_query, _ = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1)
        )
        attn = F.softmax(scores, dim=-1)

        if mask is not None:
            mask = (
                mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, n_head, n_query, -1)
                .reshape_as(attn)
            )
            attn = attn * (mask + 1e-10)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        return torch.matmul(attn, value)

    def causal_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        neg_scores = torch.matmul(query, -key.transpose(-2, -1)) / math.sqrt(
            d_k
        )
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
        del alpha
        nbatches = query.size(0)
        query, key, value = [
            linear(tensor)
            .view(nbatches, -1, self.h, self.d_k)
            .transpose(1, 2)
            for linear, tensor in zip(self.linears, (query, key, value))
        ]

        x = self.attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
        )
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
        self.feed_forward = PositionwiseFeedForward(
            embed_dim,
            ff_dim,
            dropout,
        )
        self.sublayer_self = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None, wm_mode=False):
        del wm_mode
        x = self.sublayer_self(
            x,
            lambda value: self.self_attn(value, value, value, mask),
        )
        return self.sublayer_ff(x, lambda value: self.feed_forward(value))


class Decoder(nn.Module):
    """
    CRA downstream decoder with Q-Bound R2-C/V3.5 pairing and H1 EQ-PTC.

    V3.7 optionally adds a question-conditioned interval candidate comparator
    to the EQ-PTC Semantic Seed. Only the Seed is modified: the fixed R2-C
    support graph, CLIP VALUE, QA/grounding gradient separation, and analytic
    interval posterior are unchanged. Language masked pooling is configured
    in CRA/LanModel, not in this decoder.
    V3.5 train-only Q1/Q2 uses separate question-conditioned temporal states
    while sharing the same video evidence/global context and fixed graph.
    qbound_version remains "2.8.0" for the existing CRA ABI; architecture
    selection is controlled by model.qbound.temporal_context_mode.

    V2.8/R2 factorizes temporal grounding into two causal roles. In R2-C
    the video-only predictor is removed and fixed adjacent edges are used:

        raw/generic CLIP + Question -> semantic seed       (WHERE)
        detached raw/generic CLIP   -> predictive graph    (EXTENT)
        seed + stopgrad(graph)      -> continuous support field
        support field + raw VALUE   -> Grounding and QA evidence

    Separation contract
    -------------------
    * Q-Bound receives only video + question.
    * Q-Bound never receives answer candidates, GT answers, or temporal GT.
    * CRA grounding remains a detached reference while Q-Bound is active.
    * Grounding and QA pool the SAME continuous Support Field.
    * Grounding keeps the support gradient so L_ground can train relevance.
    * QA uses the numerically identical but detached support view, so L_VQA
      cannot train semantic relevance or temporal extent topology.
    * The video-only predictor is trained only by the separately exposed
      qbound_support_predictive_loss.
    * VALUE is the original masked CLIP sequence. WHERE/support states are
      never pooled as downstream answer evidence.
    * Posterior-barycentric endpoints are compatibility/evaluation metadata;
      neither training nor QA uses a hard-MAP span.

    In event_relative mode, the core scores question-conditioned candidate
    intervals BEFORE the Seed-to-Support propagation. It does not use direct
    MAP span decisions, learned boundary heads, temporal GT, or answers.
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
        self.qbound_version = "2.8.0"

        # CRA downstream/reference branch.
        self.layers = clones(
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout),
            num_layer,
        )
        self.norm = LayerNorm(embed_dim)

        # Retained for CRA checkpoint compatibility.
        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        # Preserve original CRA parameter names.
        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        model_cfg = self.cfgs["model"]
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        configured_version = str(
            qbound_cfg.get("version", self.qbound_version)
        )
        if configured_version != self.qbound_version:
            raise ValueError(
                "Refine decoder/core version mismatch: expected "
                f"model.qbound.version={self.qbound_version!r}, got "
                f"{configured_version!r}."
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

        # R2-A/B controlled ablation. Default reproduces original V2.8.
        # This switch changes ONLY the edge affinities used by the temporal
        # Laplacian in QBoundCore. The predictive branch and L_pred stay active
        # in both arms. No R1 decoder change is introduced here.
        graph_mode = qbound_cfg.get("support_graph_mode", "learned")
        if not isinstance(graph_mode, str) or graph_mode not in ("learned", "fixed"):
            raise ValueError(
                "model.qbound.support_graph_mode must be exactly "
                "'learned' (R2-A) or 'fixed' (R2-B)."
            )
        self.qbound_support_graph_mode = graph_mode
        predictor_enabled = qbound_cfg.get("support_predictor_enabled", True)
        if not isinstance(predictor_enabled, bool):
            raise TypeError("model.qbound.support_predictor_enabled must be boolean.")
        if not predictor_enabled and graph_mode != "fixed":
            raise ValueError("R2-C requires fixed graph when predictor is disabled.")
        self.qbound_support_predictor_enabled = predictor_enabled

        # H1 B+C experiment is an explicit architecture switch, separate
        # from the stable QBound V2.8.0 CRA/Refine interface version.
        # eq_ptc is the intended mode for this patched core; legacy is the
        # exact V3.5 B+C control. No new loss or temporal GT input is added.
        temporal_mode = qbound_cfg.get("temporal_context_mode", "eq_ptc")
        if not isinstance(temporal_mode, str) or temporal_mode not in ("legacy", "eq_ptc"):
            raise ValueError(
                "model.qbound.temporal_context_mode must be 'legacy' or 'eq_ptc'."
            )
        self.qbound_temporal_context_mode = temporal_mode
        self.qbound_eqptc_scales = {}
        for field in ("eqptc_early_scale", "eqptc_position_scale", "eqptc_temporal_scale"):
            value = qbound_cfg.get(field, 0.25)
            if isinstance(value, bool):
                raise TypeError(f"model.qbound.{field} must be a float in [0,1].")
            try:
                value = float(value)
            except (ValueError, TypeError) as exc:
                raise TypeError(
                    f"model.qbound.{field} must be a numeric value in [0,1]."
                ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"model.qbound.{field} must be finite and in [0,1].")
            self.qbound_eqptc_scales[field] = value

        # V3.7: explicit Event-Relative Grounding configuration. Defaults to
        # legacy to preserve old checkpoint/initialization and inference ABI.
        event_mode = qbound_cfg.get("event_grounding_mode", "legacy")
        if not isinstance(event_mode, str) or event_mode not in (
            "legacy", "event_relative",
        ):
            raise ValueError(
                "model.qbound.event_grounding_mode must be 'legacy' "
                "or 'event_relative'."
            )
        self.qbound_event_grounding_mode = event_mode
        self.qbound_event_kwargs = {"event_grounding_mode": event_mode}
        for field, default in (
            ("event_residual_scale", 0.10),
            ("event_relation_temperature", 0.20),
        ):
            value = qbound_cfg.get(field, default)
            if isinstance(value, bool):
                raise TypeError(f"model.qbound.{field} must be numeric.")
            try:
                value = float(value)
            except (ValueError, TypeError) as exc:
                raise TypeError(f"model.qbound.{field} must be numeric.") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"model.qbound.{field} must be finite and > 0.")
            if field == "event_residual_scale" and value > 1.0:
                raise ValueError("model.qbound.event_residual_scale must be <= 1.")
            self.qbound_event_kwargs[field] = value
        max_length = qbound_cfg.get("event_candidate_max_length", None)
        if max_length is not None and (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise TypeError(
                "model.qbound.event_candidate_max_length must be null "
                "or an integer >= 1."
            )
        self.qbound_event_kwargs["event_candidate_max_length"] = max_length
        if self.qbound_enabled and event_mode == "event_relative":
            if self.qbound_temporal_context_mode != "eq_ptc":
                raise ValueError("V3.7 event_relative requires EQ-PTC temporal context.")
            if self.qbound_support_graph_mode != "fixed" or self.qbound_support_predictor_enabled:
                raise ValueError("V3.7 event_relative requires R2-C: fixed graph, predictor disabled.")

        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get("qbound_require_question_only", True),
            )
        )
        if self.qbound_enabled and not self.qbound_require_question_only:
            raise ValueError(
                "Q-Bound V2.8 requires question-only temporal input. "
                "Set model.qbound.require_question_only=true."
            )

        self.qbound_require_token_question = bool(
            qbound_cfg.get("require_token_question", True)
        )
        if self.qbound_enabled and not self.qbound_require_token_question:
            raise ValueError(
                "Q-Bound V2.8 requires token-level question features for "
                "semantic relevance. Set require_token_question=true."
            )

        self.qbound_detach_backbone_inputs = bool(
            qbound_cfg.get(
                "detach_backbone_inputs",
                model_cfg.get("qbound_detach_backbone_inputs", True),
            )
        )
        if self.qbound_enabled and not self.qbound_detach_backbone_inputs:
            raise ValueError(
                "Controlled Q-Bound V2.8 requires detached/frozen backbone "
                "features. Set model.qbound.detach_backbone_inputs=true."
            )

        self.qbound_hidden_dim = None
        if self.qbound_enabled:
            self.qbound_hidden_dim = int(
                qbound_cfg.get(
                    "hidden_dim",
                    model_cfg.get("qbound_hidden_dim", 256),
                )
            )

            # Fail fast on knobs from older temporal decision mechanisms.
            forbidden_keys = {
                "input_adapter_use_position",
                "cut_temperature",
                "cut_threshold",
                "question_residual_scale",
                "num_local_reasoning_layers",
                "use_generic_input_adapter",
                "input_adapter_dropout",
                "relation_weight",
                "boundary_weight",
                "boundary_refiner",
                "adaptive_radius_ratio",
                "min_radius",
                "joint_scoring",
                "length_prior",
                "minimality_weight",
                "entropy_weight",
                "sparsity_weight",
                "gaussian_sigma",
                "event_count",
                "visual_continuity_weight",
                "question_continuity_weight",
                "continuity_mix",
                "continuity_gate",
                "closure_temperature",
                "closure_weight",
                "core_weight",
                "exact_bottleneck_closure",
                "visual_distinctiveness_weight",
                "visual_transition_weight",
                "query_routing_weight",
                "query_routing_temperature",
                "segment_temperature",
            }
            stale = sorted(key for key in forbidden_keys if key in qbound_cfg)
            if stale:
                raise ValueError(
                    "Q-Bound V2.8 config contains removed/stale keys: "
                    + ", ".join(stale)
                    + ". Remove them instead of silently restoring an older "
                    "temporal mechanism."
                )

            core_kwargs = {
                "input_dim": embed_dim,
                "hidden_dim": self.qbound_hidden_dim,
                "dropout": float(
                    qbound_cfg.get(
                        "dropout",
                        model_cfg.get("qbound_dropout", dropout),
                    )
                ),
                "num_question_heads": int(
                    qbound_cfg.get("num_question_heads", 4)
                ),
                "question_ff_multiplier": int(
                    qbound_cfg.get("question_ff_multiplier", 2)
                ),
            }
            # Learned mode keeps the exact original constructor call: it can
            # still load canonical V2.8 weights without any new parameters.
            # Fixed mode requires the matching graph-mode-aware core.py.
            if self.qbound_support_graph_mode == "fixed":
                import inspect

                if "support_graph_mode" not in inspect.signature(
                    QBoundCore.__init__
                ).parameters:
                    raise RuntimeError(
                        "R2-B fixed graph requires the updated "
                        "modules/qbound/core.py with a support_graph_mode "
                        "constructor argument. Do not run R2-B using the "
                        "original V2.8 core.py."
                    )
                core_kwargs["support_graph_mode"] = "fixed"

            if not self.qbound_support_predictor_enabled:
                import inspect
                if "support_predictor_enabled" not in inspect.signature(
                    QBoundCore.__init__
                ).parameters:
                    raise RuntimeError("R2-C requires the matching updated core.py")
                core_kwargs["support_predictor_enabled"] = False

            # Fail fast instead of silently selecting the core's default
            # mode when YAML and core.py do not match. Always pass the new
            # switches explicitly, INCLUDING the legacy control.
            import inspect
            signature = inspect.signature(QBoundCore.__init__).parameters
            eqptc_kwargs = {
                "temporal_context_mode": self.qbound_temporal_context_mode,
                **self.qbound_eqptc_scales,
            }
            missing_args = sorted(set(eqptc_kwargs).difference(signature))
            if missing_args:
                raise RuntimeError(
                    "EQ-PTC refine.py requires the matching updated "
                    "modules/qbound/core.py; constructor args missing: "
                    + ", ".join(missing_args)
                )
            core_kwargs.update(eqptc_kwargs)
            # Fail fast if core.py is older than the V3.7 event-aware interface.
            event_missing = sorted(set(self.qbound_event_kwargs).difference(signature))
            if event_missing:
                raise RuntimeError(
                    "V3.7 refine.py requires matching modules/qbound/core.py; "
                    "missing event constructor args: " + ", ".join(event_missing)
                )
            core_kwargs.update(self.qbound_event_kwargs)
            self.qbound = QBoundCore(**core_kwargs)
            if self.qbound.event_grounding_mode != self.qbound_event_grounding_mode:
                raise RuntimeError("RefineDecoder/Core event grounding mode mismatch.")
            for field, expected in self.qbound_event_kwargs.items():
                if field == "event_grounding_mode":
                    continue
                actual = getattr(self.qbound, field)
                if expected is None:
                    if actual is not None:
                        raise RuntimeError(f"RefineDecoder/Core {field} mismatch.")
                elif isinstance(expected, float):
                    if not math.isclose(float(actual), expected, abs_tol=1e-8, rel_tol=1e-7):
                        raise RuntimeError(f"RefineDecoder/Core {field} mismatch.")
                elif actual != expected:
                    raise RuntimeError(f"RefineDecoder/Core {field} mismatch.")
            if self.qbound.temporal_context_mode != self.qbound_temporal_context_mode:
                raise RuntimeError("RefineDecoder/Core temporal context mode mismatch.")
            for name, expected_scale in self.qbound_eqptc_scales.items():
                if float(getattr(self.qbound, name)) != expected_scale:
                    raise RuntimeError(f"RefineDecoder/Core EQ-PTC scale mismatch: {name}.")
            if bool(getattr(self.qbound, "support_predictor_enabled", True)) != self.qbound_support_predictor_enabled:
                raise RuntimeError("RefineDecoder/Core predictor-enabled mismatch.")
            actual_graph_mode = getattr(self.qbound, "support_graph_mode", "learned")
            if actual_graph_mode != self.qbound_support_graph_mode:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore support graph mode mismatch: "
                    f"config={self.qbound_support_graph_mode!r}, "
                    f"core={actual_graph_mode!r}."
                )

            core_version = str(
                getattr(self.qbound, "qbound_version", "")
            )
            if core_version != self.qbound_version:
                raise RuntimeError(
                    "RefineDecoder/QBoundCore version mismatch: "
                    f"decoder={self.qbound_version!r}, "
                    f"core={core_version!r}."
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
                "Temporal distribution and valid mask must share [B,T]; got "
                f"{tuple(values.shape)} and {tuple(valid_mask.shape)}."
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
            raise ValueError("video projection dropout must be < 1.")

        mask = torch.empty_like(reference).bernoulli_(keep_prob)
        return mask / keep_prob

    def _apply_video_projection(self, pooled, dropout_mask):
        if pooled.shape != dropout_mask.shape:
            raise ValueError(
                "pooled representation and dropout mask must share shape; got "
                f"{tuple(pooled.shape)} and {tuple(dropout_mask.shape)}."
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
            projected = self.do(x_star_proj, projected)
        return pooled, projected

    # ------------------------------------------------------------------
    # Explicit fail-fast stubs for mechanisms absent from V2.8.
    # ------------------------------------------------------------------
    def score_functional_composition_pairs(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "score_functional_composition_pairs() is not part of Q-Bound V2.8. "
            "V2.8 has no answer-based temporal critic."
        )

    def score_boundary_equilibrium_views(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(
            "score_boundary_equilibrium_views() is not part of Q-Bound V2.8."
        )

    # ------------------------------------------------------------------
    # V2.8 core validation
    # ------------------------------------------------------------------
    @staticmethod
    def _flag(container, name):
        value = container[name]
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(
                f"Q-Bound flag {name!r} must be a scalar tensor."
            )
        return float(value.detach().item())

    @staticmethod
    def _assert_finite(name, value):
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values.")

    @staticmethod
    def _assert_shape(name, value, expected):
        if tuple(value.shape) != tuple(expected):
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got "
                f"{tuple(value.shape)}."
            )

    @staticmethod
    def _assert_distribution(name, value, valid_mask, atol=1e-4):
        if value.shape != valid_mask.shape:
            raise ValueError(
                f"{name} must be [B,T]={tuple(valid_mask.shape)}, got "
                f"{tuple(value.shape)}."
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values.")
        if torch.any(value.detach() < -1e-6):
            raise RuntimeError(f"{name} contains negative probability mass.")

        masked = value * valid_mask.to(value.dtype)
        total = masked.sum(dim=-1)
        if not torch.allclose(
            total,
            torch.ones_like(total),
            atol=atol,
            rtol=atol,
        ):
            raise RuntimeError(f"{name} must sum to one over valid frames.")

        invalid_mass = (
            value * (~valid_mask).to(value.dtype)
        ).abs().amax()
        if float(invalid_mass.detach().item()) > atol:
            raise RuntimeError(f"{name} assigns mass to padded frames.")

    def _validate_eqptc_outputs(
        self, output, batch_size, time_steps, q_len, valid_mask
    ):
        """Validate the extra WHERE-only diagnostics of the EQ-PTC B+C branch."""
        if self.qbound_temporal_context_mode != "eq_ptc":
            return
        expected = {
            "eqptc_global_context_states": (batch_size, time_steps, self.qbound_hidden_dim),
            "eqptc_conditioned_visual_states": (batch_size, time_steps, self.qbound_hidden_dim),
            "eqptc_early_question_attention":
                (batch_size, self.qbound.num_question_heads, time_steps, q_len),
            "eqptc_temporal_attention":
                (batch_size, self.qbound.num_question_heads, time_steps, time_steps),
            "eqptc_early_delta_ratio": (batch_size,),
            "eqptc_temporal_delta_ratio": (batch_size,),
        }
        for name, shape in expected.items():
            if name not in output:
                raise KeyError(f"EQ-PTC is active but {name!r} is missing.")
            self._assert_shape(name, output[name], shape)
            self._assert_finite(name, output[name])
        for name in ("eqptc_global_context_states", "eqptc_conditioned_visual_states"):
            invalid_values = output[name] * (~valid_mask).to(output[name].dtype).unsqueeze(-1)
            if invalid_values.detach().abs().amax() > 1e-5:
                raise RuntimeError(f"EQ-PTC WHERE state {name} includes padded visual tokens.")
        for name in ("eqptc_early_delta_ratio", "eqptc_temporal_delta_ratio"):
            if (output[name].detach() < 0).any():
                raise RuntimeError(f"EQ-PTC delta ratio must be nonnegative: {name}.")

    def _validate_event_relative_outputs(
        self, output, batch_size, time_steps, valid_mask,
        geometry_source=None,
    ):
        """V3.7 interval/Seed residual contract; paired Q2 reuses Q1 geometry.

        Validation is tensor-based, never uses labels or answers, and does not
        detach anything in the main computational graph.
        """
        geometry = output if geometry_source is None else geometry_source
        geometry_keys = (
            "event_candidate_starts", "event_candidate_ends",
            "event_candidate_lengths", "event_candidate_valid",
            "event_candidate_frame_weights",
        )
        required = (
            "legacy_semantic_seed_logits", "legacy_semantic_seed_prob",
            "event_candidate_scores", "event_candidate_prob",
            "event_candidate_count", "event_candidate_top_index",
            "event_candidate_top_start_frame", "event_candidate_top_end_frame",
            "event_candidate_top1_prob", "event_frame_prob",
            "event_geometry_frame_prior", "event_seed_residual_logits",
            "event_residual_rms", "event_residual_to_legacy_std_ratio",
        )
        missing = [k for k in required if k not in output]
        missing += [k for k in geometry_keys if k not in geometry]
        if missing:
            raise KeyError("V3.7 event-relative output missing: " + ", ".join(missing))

        starts, ends, lengths = (geometry[k] for k in geometry_keys[:3])
        if starts.ndim != 1 or ends.ndim != 1 or lengths.ndim != 1:
            raise ValueError("V3.7 event interval geometry must be [K].")
        K = starts.numel()
        if K < 1:
            raise ValueError("V3.7 event candidate bank must not be empty.")
        expected_max = self.qbound_event_kwargs["event_candidate_max_length"]
        candidate_lengths = torch.arange(1, time_steps + 1, device=starts.device)
        if expected_max is not None:
            candidate_lengths = candidate_lengths[:expected_max]
        expected_K = int((time_steps - candidate_lengths + 1).sum().item())
        if K != expected_K:
            raise RuntimeError(f"V3.7 expected {expected_K} intervals, got {K}.")
        for name, tensor in (("starts", starts), ("ends", ends), ("lengths", lengths)):
            self._assert_shape("event_candidate_" + name, tensor, (K,))
            if tensor.dtype not in (torch.int32, torch.int64):
                raise TypeError("V3.7 interval indices must have integer dtype.")
        if (starts < 0).any() or (ends >= time_steps).any() or (ends < starts).any():
            raise RuntimeError("V3.7 candidate boundaries outside [0,T).")
        if not torch.equal(lengths, ends - starts + 1):
            raise RuntimeError("V3.7 candidate interval lengths are inconsistent.")

        candidate_valid = geometry["event_candidate_valid"]
        frame_weights = geometry["event_candidate_frame_weights"]
        self._assert_shape("event_candidate_valid", candidate_valid, (batch_size, K))
        self._assert_shape("event_candidate_frame_weights", frame_weights, (K, time_steps))
        if candidate_valid.dtype != torch.bool:
            raise TypeError("V3.7 event_candidate_valid must be bool.")
        self._assert_finite("event_candidate_frame_weights", frame_weights)
        if not torch.allclose(frame_weights.sum(dim=-1),
                              torch.ones_like(frame_weights.sum(dim=-1)), atol=1e-5):
            raise RuntimeError("V3.7 candidate frame weights must sum to 1.")
        idx = torch.arange(time_steps, device=valid_mask.device)
        membership = (idx.unsqueeze(0) >= starts.unsqueeze(1)) & (
            idx.unsqueeze(0) <= ends.unsqueeze(1)
        )
        expected_weights = membership.to(frame_weights.dtype) / lengths.to(
            frame_weights.dtype
        ).unsqueeze(1)
        if not torch.allclose(frame_weights, expected_weights, atol=1e-6, rtol=1e-6):
            raise RuntimeError("V3.7 candidate geometry weights do not match boundaries.")
        expected_valid = (
            (membership.unsqueeze(0) & ~valid_mask.unsqueeze(1)).sum(dim=-1) == 0
        )
        if not torch.equal(candidate_valid, expected_valid):
            raise RuntimeError("V3.7 candidate validity crosses a padded video frame.")
        if not torch.all(candidate_valid.any(dim=-1)):
            raise RuntimeError("V3.7 every sample must have >=1 valid candidate.")

        for name in ("event_candidate_scores", "event_candidate_prob"):
            self._assert_shape(name, output[name], (batch_size, K))
            self._assert_finite(name, output[name])
        probs = output["event_candidate_prob"]
        if (probs.detach() < -1e-6).any() or (
            (probs.detach() * ~candidate_valid).abs().amax() > 1e-5
        ):
            raise RuntimeError("V3.7 event candidates assign probability to invalid intervals.")
        if not torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)),
                              atol=1e-4, rtol=1e-4):
            raise RuntimeError("V3.7 event candidate probabilities must sum to 1.")

        for name in (
            "event_candidate_count", "event_candidate_top_index",
            "event_candidate_top_start_frame", "event_candidate_top_end_frame",
            "event_candidate_top1_prob", "event_residual_rms",
            "event_residual_to_legacy_std_ratio",
        ):
            self._assert_shape(name, output[name], (batch_size,))
            self._assert_finite(name, output[name])
        if not torch.equal(output["event_candidate_count"], candidate_valid.sum(dim=-1)):
            raise RuntimeError("V3.7 reported candidate count mismatch.")
        top_index = output["event_candidate_top_index"]
        if not torch.equal(top_index, probs.argmax(dim=-1)):
            raise RuntimeError("V3.7 reported top candidate index mismatch.")
        if not torch.equal(output["event_candidate_top_start_frame"], starts[top_index]):
            raise RuntimeError("V3.7 top candidate start mismatch.")
        if not torch.equal(output["event_candidate_top_end_frame"], ends[top_index]):
            raise RuntimeError("V3.7 top candidate end mismatch.")
        if not torch.allclose(output["event_candidate_top1_prob"],
                              probs.max(dim=-1).values, atol=1e-5, rtol=1e-5):
            raise RuntimeError("V3.7 event top1 probability mismatch.")

        self._assert_distribution("legacy_semantic_seed_prob",
                                  output["legacy_semantic_seed_prob"], valid_mask)
        self._assert_distribution("event_frame_prob", output["event_frame_prob"], valid_mask)
        self._assert_distribution("event_geometry_frame_prior",
                                  output["event_geometry_frame_prior"], valid_mask)
        for name in ("legacy_semantic_seed_logits", "event_seed_residual_logits"):
            self._assert_shape(name, output[name], (batch_size, time_steps))
            self._assert_finite(name, output[name])
        residual = output["event_seed_residual_logits"]
        if (residual.detach() * ~valid_mask).abs().amax() > 1e-5:
            raise RuntimeError("V3.7 event residual must be zero on padded frames.")
        expected_frame = torch.matmul(probs, frame_weights)
        if not torch.allclose(output["event_frame_prob"], expected_frame,
                              atol=1e-5, rtol=1e-5):
            raise RuntimeError("V3.7 event-to-frame mass is inconsistent.")
        if output["event_residual_rms"].detach().min().item() < -1e-8:
            raise RuntimeError("V3.7 event residual RMS must be nonnegative.")
        if output["event_residual_to_legacy_std_ratio"].detach().min().item() < -1e-8:
            raise RuntimeError("V3.7 residual/legacy std ratio must be nonnegative.")
        # The canonical Seed is the fused one; no alternate inference shortcut.
        if not torch.allclose(
            output["semantic_seed_logits"].masked_fill(~valid_mask, 0.0),
            (output["legacy_semantic_seed_logits"] + residual).masked_fill(~valid_mask, 0.0),
            atol=1e-5, rtol=1e-5,
        ):
            raise RuntimeError("V3.7 fused Seed logits must equal legacy + residual.")

    def _validate_qbound_core(
        self,
        qbound_aux,
        batch_size,
        time_steps,
        q_len,
        valid_mask,
    ):
        required = {
            "qbound_version",
            # Architecture contract.
            "structured_visual_evidence_enabled",
            "visual_distinctiveness_enabled",
            "visual_transition_enabled",
            "temporal_context_enabled",
            "relevance_extent_decoupling_enabled",
            "token_resolved_question_relevance_enabled",
            "semantic_evidence_seed_enabled",
            "question_direct_span_scoring_enabled",
            "question_as_visual_value_enabled",
            "video_only_support_branch_enabled",
            "predictive_support_graph_enabled",
            "support_predictor_target_leakage_enabled",
            "support_branch_uses_question_enabled",
            "grounding_gradient_to_support_graph_enabled",
            "support_graph_detached_for_grounding_enabled",
            "structural_seed_propagation_enabled",
            "analytic_boundary_posterior_enabled",
            "posterior_barycentric_grounding_enabled",
            "same_support_for_training_and_inference_enabled",
            "value_where_separation_enabled",
            # Removed shortcuts.
            "query_routed_visual_context_enabled",
            "evidence_core_enabled",
            "event_continuity_enabled",
            "joint_transition_continuity_enabled",
            "core_to_closure_enabled",
            "additive_extent_energy_enabled",
            "neural_span_scorer_enabled",
            "event_candidate_selector_enabled",
            "hard_map_temporal_decision_enabled",
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
            "minimum_event_duration_enabled",
            # VALUE / relevance / support.
            "value_tokens",
            "visual_state_states",
            "visual_video_mean_state",
            "visual_distinctiveness_states",
            "visual_distinctiveness_delta",
            "visual_transition_states",
            "visual_transition_frame_states",
            "visual_transition_delta",
            "visual_transition_mask",
            "visual_evidence_states",
            "where_context_states",
            "temporal_self_attention",
            "question_tokens_projected",
            "relevance_question_context",
            "relevance_question_attention",
            "semantic_seed_hidden",
            "semantic_seed_logits",
            "semantic_seed_prob",
            "support_states",
            "support_forward_context",
            "support_backward_context",
            "support_forward_prediction",
            "support_backward_prediction",
            "support_edge_mask",
            "support_edge_surprise",
            "support_edge_affinity_raw",
            "support_edge_affinity",
            "support_predictive_loss_per_sample",
            "support_predictive_loss",
            "support_graph_adjacency",
            "support_graph_laplacian",
            "support_temporal_prob",
            "support_entropy_ratio",
            "support_effective_frames",
            "support_propagation_l1_delta",
            "semantic_seed_support_cosine",
            # Analytic interval views.
            "start_boundary_prob",
            "end_boundary_prob",
            "segment_valid_mask",
            "segment_prob",
            "training_temporal_prob",
            "main_temporal_prob",
            "soft_temporal_prob",
            "selected_span_frame_mask",
            "selected_span_frame_weights",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "barycenter_start_frame",
            "barycenter_end_frame",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",
            "single_frame_prob_mass",
            "short_span_prob_mass",
            "selected_single_frame_indicator",
            "selected_full_video_indicator",
            # Canonical aliases needed by the next integration stage.
            "qbound_support_predictive_loss",
            "qbound_support_temporal_prob",
            "qbound_training_temporal_prob",
            "qbound_soft_temporal_prob",
            "qbound_main_temporal_prob",
            "qbound_hard_temporal_prob",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "Q-Bound V2.8 core output is missing: "
                + ", ".join(missing)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "RefineDecoder/QBoundCore V2.8 runtime version mismatch: "
                f"decoder={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        # V3.7 activation contract: assert actual Seed path, never just YAML.
        event_active = self.qbound_event_grounding_mode == "event_relative"
        if "event_relative_enabled_flag" not in qbound_aux:
            raise KeyError("V3.7 core output missing event_relative_enabled_flag.")
        if self._flag(qbound_aux, "event_relative_enabled_flag") != float(event_active):
            raise RuntimeError("Configured V3.7 event mode differs from executed QBoundCore.")
        for field in ("event_residual_scale", "event_relation_temperature"):
            reported_name = field + "_used"
            if reported_name not in qbound_aux:
                raise KeyError(f"V3.7 core output missing {reported_name!r}.")
            if not math.isclose(
                self._flag(qbound_aux, reported_name),
                self.qbound_event_kwargs[field], rel_tol=1e-5, abs_tol=1e-6,
            ):
                raise RuntimeError(f"V3.7 event parameter mismatch: {reported_name}.")
        if event_active:
            self._validate_event_relative_outputs(
                qbound_aux, batch_size, time_steps, valid_mask,
            )
        elif "event_candidate_prob" in qbound_aux:
            raise RuntimeError("Legacy core should not compute event candidates.")

        # H1 activation contract. Merely configuring eq_ptc is NOT enough:
        # enforce that the actual core forward ran the new B+C branch.
        if "eqptc_enabled_flag" not in qbound_aux:
            raise KeyError("Expected EQ-PTC-aware core.py output 'eqptc_enabled_flag'.")
        eqptc_active = self.qbound_temporal_context_mode == "eq_ptc"
        if self._flag(qbound_aux, "eqptc_enabled_flag") != float(eqptc_active):
            raise RuntimeError("Configured EQ-PTC mode differs from executed QBoundCore mode.")
        for name, scale in self.qbound_eqptc_scales.items():
            output_name = name + "_used"
            if output_name not in qbound_aux:
                raise KeyError(f"EQ-PTC core output missing {output_name!r}.")
            reported = self._flag(qbound_aux, output_name)
            if not math.isclose(reported, scale, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(f"EQ-PTC core scale mismatch: {output_name}.")
        if eqptc_active:
            self._validate_eqptc_outputs(
                qbound_aux,
                batch_size=batch_size,
                time_steps=time_steps,
                q_len=q_len,
                valid_mask=valid_mask,
            )

        required_on = {
            "structured_visual_evidence_enabled",
            "visual_distinctiveness_enabled",
            "visual_transition_enabled",
            "temporal_context_enabled",
            "relevance_extent_decoupling_enabled",
            "token_resolved_question_relevance_enabled",
            "semantic_evidence_seed_enabled",
            "video_only_support_branch_enabled",
            "predictive_support_graph_enabled",
            "support_graph_detached_for_grounding_enabled",
            "structural_seed_propagation_enabled",
            "analytic_boundary_posterior_enabled",
            "posterior_barycentric_grounding_enabled",
            "same_support_for_training_and_inference_enabled",
            "value_where_separation_enabled",
        }
        required_off = {
            "question_direct_span_scoring_enabled",
            "question_as_visual_value_enabled",
            "support_predictor_target_leakage_enabled",
            "support_branch_uses_question_enabled",
            "grounding_gradient_to_support_graph_enabled",
            "query_routed_visual_context_enabled",
            "evidence_core_enabled",
            "event_continuity_enabled",
            "joint_transition_continuity_enabled",
            "core_to_closure_enabled",
            "additive_extent_energy_enabled",
            "neural_span_scorer_enabled",
            "event_candidate_selector_enabled",
            "hard_map_temporal_decision_enabled",
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
            "minimum_event_duration_enabled",
        }
        # In V3.7 these three flags correctly identify interval scoring.
        # They MUST NOT be reported as zero merely to bypass this validator.
        event_flags = {
            "question_direct_span_scoring_enabled",
            "neural_span_scorer_enabled",
            "event_candidate_selector_enabled",
        }
        if event_active:
            required_off -= event_flags
            required_on |= event_flags
        if not self.qbound_support_predictor_enabled:
            required_on -= {"video_only_support_branch_enabled", "predictive_support_graph_enabled"}
            required_off |= {"video_only_support_branch_enabled", "predictive_support_graph_enabled"}
        for name in required_on:
            if self._flag(qbound_aux, name) != 1.0:
                raise RuntimeError(
                    f"Q-Bound V2.8 requires core flag {name}=1."
                )
        for name in required_off:
            if self._flag(qbound_aux, name) != 0.0:
                raise RuntimeError(
                    f"Q-Bound V2.8 requires core flag {name}=0."
                )

        hidden_dim = int(self.qbound_hidden_dim)
        edge_steps = max(time_steps - 1, 0)

        frame_hidden_names = (
            "visual_state_states",
            "visual_distinctiveness_states",
            "visual_distinctiveness_delta",
            "visual_transition_frame_states",
            "visual_transition_delta",
            "visual_evidence_states",
            "where_context_states",
            "relevance_question_context",
            "semantic_seed_hidden",
            "support_states",
            "support_forward_context",
            "support_backward_context",
        )
        for name in frame_hidden_names:
            value = qbound_aux[name]
            self._assert_shape(
                name,
                value,
                (batch_size, time_steps, hidden_dim),
            )
            self._assert_finite(name, value)

        self._assert_shape(
            "visual_video_mean_state",
            qbound_aux["visual_video_mean_state"],
            (batch_size, hidden_dim),
        )
        self._assert_shape(
            "visual_transition_states",
            qbound_aux["visual_transition_states"],
            (batch_size, edge_steps, hidden_dim),
        )
        self._assert_shape(
            "visual_transition_mask",
            qbound_aux["visual_transition_mask"],
            (batch_size, edge_steps),
        )
        self._assert_shape(
            "question_tokens_projected",
            qbound_aux["question_tokens_projected"],
            (batch_size, q_len, hidden_dim),
        )

        temporal_attn = qbound_aux["temporal_self_attention"]
        if (
            temporal_attn.ndim != 4
            or temporal_attn.size(0) != batch_size
            or temporal_attn.size(2) != time_steps
            or temporal_attn.size(3) != time_steps
        ):
            raise ValueError(
                "temporal_self_attention must be [B,H,T,T], got "
                f"{tuple(temporal_attn.shape)}."
            )

        relevance_attn = qbound_aux["relevance_question_attention"]
        if (
            relevance_attn.ndim != 4
            or relevance_attn.size(0) != batch_size
            or relevance_attn.size(2) != time_steps
            or relevance_attn.size(3) != q_len
        ):
            raise ValueError(
                "relevance_question_attention must be [B,H,T,L], got "
                f"{tuple(relevance_attn.shape)}."
            )

        for name in (
            "support_forward_prediction",
            "support_backward_prediction",
        ):
            value = qbound_aux[name]
            self._assert_shape(
                name,
                value,
                (batch_size, time_steps, self.embed_dim),
            )
            self._assert_finite(name, value)

        for name in (
            "support_edge_mask",
            "support_edge_surprise",
            "support_edge_affinity_raw",
            "support_edge_affinity",
        ):
            self._assert_shape(
                name,
                qbound_aux[name],
                (batch_size, edge_steps),
            )
            self._assert_finite(name, qbound_aux[name])

        for name in (
            "support_graph_adjacency",
            "support_graph_laplacian",
        ):
            self._assert_shape(
                name,
                qbound_aux[name],
                (batch_size, time_steps, time_steps),
            )
            self._assert_finite(name, qbound_aux[name])

        if qbound_aux["support_edge_affinity"].requires_grad:
            raise RuntimeError(
                "support_edge_affinity must be detached before grounding."
            )
        if qbound_aux["support_graph_laplacian"].requires_grad:
            raise RuntimeError(
                "support_graph_laplacian must be detached before propagation."
            )

        # Ensure the two R2 arms really consume different graph definitions.
        # A/B must not silently revert to learned affinities in fixed mode.
        used_edges = qbound_aux["support_edge_affinity"].detach()
        if self.qbound_support_graph_mode == "fixed":
            expected_edges = qbound_aux["support_edge_mask"].to(used_edges.dtype)
        else:
            expected_edges = qbound_aux["support_edge_affinity_raw"].detach()
        if not torch.allclose(
            used_edges, expected_edges, atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError(
                "R2 graph contract violated: propagation does not use "
                f"the expected {self.qbound_support_graph_mode!r} edges."
            )

        adjacency = qbound_aux["support_graph_adjacency"]
        laplacian = qbound_aux["support_graph_laplacian"]
        if not torch.allclose(
            adjacency.detach(),
            adjacency.detach().transpose(-2, -1),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("V2.8 support adjacency must be symmetric.")
        if not torch.allclose(
            laplacian.detach(),
            laplacian.detach().transpose(-2, -1),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError("V2.8 support Laplacian must be symmetric.")

        for name in (
            "semantic_seed_prob",
            "support_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
            "main_temporal_prob",
            "qbound_support_temporal_prob",
            "qbound_training_temporal_prob",
            "qbound_soft_temporal_prob",
            "qbound_main_temporal_prob",
            "qbound_hard_temporal_prob",
        ):
            self._assert_distribution(
                name,
                qbound_aux[name],
                valid_mask,
            )

        for name in (
            "semantic_seed_logits",
            "start_boundary_prob",
            "end_boundary_prob",
            "selected_span_frame_mask",
            "selected_span_frame_weights",
        ):
            self._assert_shape(
                name,
                qbound_aux[name],
                (batch_size, time_steps),
            )
            self._assert_finite(name, qbound_aux[name])

        for name in ("segment_valid_mask", "segment_prob"):
            self._assert_shape(
                name,
                qbound_aux[name],
                (batch_size, time_steps, time_steps),
            )
        self._assert_finite("segment_prob", qbound_aux["segment_prob"])
        segment_total = qbound_aux["segment_prob"].sum(dim=(-2, -1))
        if not torch.allclose(
            segment_total,
            torch.ones_like(segment_total),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RuntimeError(
                "V2.8 analytic segment_prob must sum to one."
            )

        predictive_loss = qbound_aux["support_predictive_loss"]
        if not torch.is_tensor(predictive_loss) or predictive_loss.numel() != 1:
            raise TypeError("support_predictive_loss compatibility slot must be scalar.")
        self._assert_finite("support_predictive_loss", predictive_loss)
        if not self.qbound_support_predictor_enabled:
            if predictive_loss.requires_grad or predictive_loss.detach().item() != 0.0:
                raise RuntimeError("R2-C must not compute L_pred.")
            for name in ("support_input_proj", "support_forward_cell",
                         "support_backward_cell", "support_forward_predictor",
                         "support_backward_predictor"):
                if hasattr(self.qbound, name):
                    raise RuntimeError(f"R2-C still contains predictor {name}.")

    # ------------------------------------------------------------------
    # V3.5 paired evidence (auxiliary TRAINING interface only)
    # ------------------------------------------------------------------
    def _validate_v35_pair(
        self, pair, pair_valid_mask, valid_mask, pair_length, primary=None,
    ):
        """Fail fast if paired grounding violates the R2-C firewall.

        No answer/temporal GT is accepted by this method. This method never
        changes the Q1 time distribution or produces a second QA prediction.
        """
        batch_size, time_steps = valid_mask.shape
        required = (
            "semantic_seed_logits",
            "semantic_seed_prob",
            "support_temporal_prob",
            "question_mask",
            "video_mask",
        )
        absent = [name for name in required if name not in pair]
        if absent:
            raise KeyError("V3.5 paired core output missing: " + ", ".join(absent))
        self._assert_shape(
            "pair_valid_mask", pair_valid_mask, (batch_size,)
        )
        if pair_valid_mask.dtype != torch.bool:
            raise TypeError("V3.5 pair_valid_mask must be boolean.")
        if not torch.equal(pair["video_mask"], valid_mask):
            raise RuntimeError("V3.5 Q1/Q2 must use the SAME video mask.")
        if self.qbound_temporal_context_mode == "eq_ptc":
            # IMPORTANT: Q2 must compute its OWN question-conditioned Bq
            # states. Reusing Q1's where_context_states would silently break
            # the very H1 question-disambiguation experiment.
            self._validate_eqptc_outputs(
                pair,
                batch_size=batch_size,
                time_steps=time_steps,
                q_len=pair_length,
                valid_mask=valid_mask,
            )
        if self.qbound_event_grounding_mode == "event_relative":
            if primary is None:
                raise RuntimeError("V3.7 paired validation needs Q1 candidate geometry.")
            self._validate_event_relative_outputs(
                pair, batch_size, time_steps, valid_mask, geometry_source=primary,
            )
        elif "event_candidate_prob" in pair:
            raise RuntimeError("Legacy Q2 must not compute event candidates.")
        self._assert_shape(
            "paired_question_mask", pair["question_mask"],
            (batch_size, pair_length),
        )
        self._assert_shape(
            "paired_semantic_seed_logits", pair["semantic_seed_logits"],
            (batch_size, time_steps),
        )
        self._assert_finite("paired_semantic_seed_logits", pair["semantic_seed_logits"])
        for name in ("semantic_seed_prob", "support_temporal_prob"):
            self._assert_distribution(
                "paired_" + name, pair[name], valid_mask
            )
        if not pair["support_temporal_prob"].requires_grad:
            raise RuntimeError(
                "V3.5 paired Support must retain its gradient to the "
                "shared Semantic Seed during training."
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
                    "qbound_time is required for active Q-Bound V2.8."
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
        # Q-Bound V3.5 / QDTG: all three inputs below are train-only.
        # Both questions refer to the SAME video; invalid/unpaired rows may
        # be padded and are excluded downstream with pair_valid_mask.
        paired_question_feature=None,
        paired_question_mask=None,
        pair_valid_mask=None,
    ):
        del gsub, qbound_mix, window_sizes

        paired_requested = paired_question_feature is not None
        if not paired_requested and (
            paired_question_mask is not None or pair_valid_mask is not None
        ):
            raise ValueError(
                "V3.5 paired_question_mask/pair_valid_mask requires "
                "paired_question_feature."
            )
        if paired_requested:
            if not self.training:
                raise RuntimeError(
                    "V3.5 paired-question computation is training-only; "
                    "do not pass paired_question_feature to eval/test."
                )
            if not bool(self.qbound_enabled and (use_qbound or use_gpeb)):
                raise RuntimeError(
                    "V3.5 paired-question computation requires active Q-Bound."
                )
            if not hasattr(self.qbound, "forward_paired"):
                raise RuntimeError(
                    "V3.5 requires modules/qbound/core.py with forward_paired()."
                )

        if use_gpeb is not None:
            use_qbound = bool(use_qbound or use_gpeb)

        # CRA branch: retained as downstream QA host and detached reference.
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
                "Controlled Q-Bound V2.8 requires model.do=false. "
                "VisualIntervention would confound the grounding experiment."
            )

        # CRA temporal grounding cannot supervise V2.8 topology.
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

        # One shared projection dropout sample keeps feature comparisons fair.
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
            if question_feature is None:
                raise ValueError(
                    "Q-Bound V2.8 requires question_feature. "
                    "Answer/candidate-answer features may not be substituted."
                )
            if (
                self.qbound_require_token_question
                and question_feature.ndim != 3
            ):
                raise ValueError(
                    "Q-Bound V2.8 requires token-level question features "
                    "[B,L,D] for semantic relevance."
                )
            if qbound_video_tokens is None:
                raise ValueError(
                    "Q-Bound V2.8 requires qbound_video_tokens=[B,T,D] "
                    "containing the raw/generic CLIP sequence."
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
                    "Q-Bound V2.8 expects visual dimension "
                    f"{self.embed_dim}; got {qbound_video_tokens.size(-1)}."
                )
            if question_feature.size(0) != batch_size:
                raise ValueError(
                    "question_feature batch size does not match video batch."
                )
            if question_feature.size(-1) != self.embed_dim:
                raise ValueError(
                    "Q-Bound V2.8 expects question feature dimension "
                    f"{self.embed_dim}; got {question_feature.size(-1)}."
                )

            qbound_video_tokens = qbound_video_tokens.to(
                device=x_bar.device,
                dtype=x_bar.dtype,
            )
            qbound_language = question_feature.to(
                device=x_bar.device,
                dtype=x_bar.dtype,
            )
            normalized_question_mask = self._normalize_question_mask(
                question_mask,
                qbound_language,
                x_bar.device,
            )
            question_mask_was_provided = normalized_question_mask is not None

            # Freeze external feature producers. QBoundCore also independently
            # detaches the video-only predictor target/source.
            video_for_qbound = qbound_video_tokens.detach()
            language_for_qbound = qbound_language.detach()

            # Both branches preserve the same external V3.5 ABI. In EQ-PTC,
            # Q1/Q2 share video evidence, B0 global temporal context and the
            # detached fixed graph, but each computes its own C-pre/Bq context.
            # In legacy mode, Q1/Q2 additionally share B0 as their WHERE.
            qbound_pair_aux = None
            normalized_pair_valid_mask = None
            if paired_requested:
                if not torch.is_tensor(paired_question_feature):
                    raise TypeError("paired_question_feature must be a tensor.")
                if paired_question_feature.ndim != 3:
                    raise ValueError(
                        "V3.5 paired_question_feature must be question-only "
                        "token embeddings [B,L2,D], not Q+A embeddings."
                    )
                if (
                    paired_question_feature.size(0) != batch_size
                    or paired_question_feature.size(-1) != self.embed_dim
                    or paired_question_feature.size(1) < 1
                ):
                    raise ValueError(
                        "V3.5 paired_question_feature must be [B,L2,D], "
                        "matching primary batch and feature dimension."
                    )
                pair_language = paired_question_feature.to(
                    device=x_bar.device, dtype=x_bar.dtype
                )
                pair_question_mask = self._normalize_question_mask(
                    paired_question_mask, pair_language, x_bar.device
                )
                if pair_valid_mask is None:
                    normalized_pair_valid_mask = torch.ones(
                        batch_size, dtype=torch.bool, device=x_bar.device
                    )
                else:
                    if not torch.is_tensor(pair_valid_mask):
                        raise TypeError("pair_valid_mask must be a boolean tensor [B].")
                    if pair_valid_mask.dtype != torch.bool:
                        raise TypeError("pair_valid_mask must have dtype torch.bool.")
                    if pair_valid_mask.shape != (batch_size,):
                        raise ValueError(
                            "pair_valid_mask must be [B]; got "
                            f"{tuple(pair_valid_mask.shape)}."
                        )
                    normalized_pair_valid_mask = pair_valid_mask.to(x_bar.device)
                qbound_aux, qbound_pair_aux = self.qbound.forward_paired(
                    video_tokens=video_for_qbound,
                    question_features=language_for_qbound,
                    paired_question_features=pair_language.detach(),
                    video_mask=valid_mask,
                    question_mask=normalized_question_mask,
                    paired_question_mask=pair_question_mask,
                )
            else:
                qbound_aux = self.qbound(
                    video_tokens=video_for_qbound,
                    question_features=language_for_qbound,
                    video_mask=valid_mask,
                    question_mask=normalized_question_mask,
                )

            self._validate_qbound_core(
                qbound_aux=qbound_aux,
                batch_size=batch_size,
                time_steps=time_steps,
                q_len=qbound_language.size(1),
                valid_mask=valid_mask,
            )
            if qbound_pair_aux is not None:
                self._validate_v35_pair(
                    qbound_pair_aux,
                    normalized_pair_valid_mask,
                    valid_mask,
                    pair_language.size(1),
                    primary=qbound_aux,
                )
                # Separate training-only companion output: never overwrites
                # Q1's support, QA projection, time_param or metric decoder.
                # The pipeline will implement L_pair (NOT this decoder).
                # Use explicit qbound_ prefix so CRA.forward()'s ordinary
                # key forwarding exposes them in model_outputs.
                qbound_aux["qbound_pair_aux"] = qbound_pair_aux
                qbound_aux["qbound_pair_valid_mask"] = normalized_pair_valid_mask
                qbound_aux["qbound_pair_question_mask"] = qbound_pair_aux[
                    "question_mask"
                ]
                qbound_aux["qbound_pair_semantic_seed_logits"] = qbound_pair_aux[
                    "semantic_seed_logits"
                ]
                qbound_aux["qbound_pair_semantic_seed_prob"] = qbound_pair_aux[
                    "semantic_seed_prob"
                ]
                qbound_aux["qbound_pair_support_temporal_prob"] = qbound_pair_aux[
                    "support_temporal_prob"
                ]
                qbound_aux["qbound_pair_support_entropy_ratio"] = qbound_pair_aux[
                    "support_entropy_ratio"
                ]
                qbound_aux["qbound_pair_support_effective_frames"] = qbound_pair_aux[
                    "support_effective_frames"
                ]
                # V3.7 paired diagnostic tensors are Q2-specific; never mix
                # Q1 candidate probabilities/residual with Q2's evidence.
                if self.qbound_event_grounding_mode == "event_relative":
                    for event_key in (
                        "event_candidate_scores", "event_candidate_prob",
                        "event_candidate_count", "event_candidate_top_index",
                        "event_candidate_top_start_frame", "event_candidate_top_end_frame",
                        "event_candidate_top1_prob", "event_frame_prob",
                        "event_seed_residual_logits", "event_residual_rms",
                        "event_residual_to_legacy_std_ratio",
                    ):
                        qbound_aux["qbound_pair_" + event_key] = qbound_pair_aux[event_key]

            # Diagnostics only. No tensor, loss or gradient is changed here.
            # Copied to time_param as "qbound_support_graph_mode" below.
            qbound_aux["support_graph_mode"] = self.qbound_support_graph_mode
            qbound_aux["support_predictor_enabled"] = self.qbound_support_predictor_enabled
            qbound_aux["temporal_context_mode"] = self.qbound_temporal_context_mode
            qbound_aux["event_grounding_mode"] = self.qbound_event_grounding_mode

            # ----------------------------------------------------------
            # One V2.8 Support Field, with two gradient views.
            # ----------------------------------------------------------
            support_key = self._normalize_temporal_distribution(
                qbound_aux["support_temporal_prob"],
                valid_mask,
            )

            same_support_names = (
                "main_temporal_prob",
                "soft_temporal_prob",
                "training_temporal_prob",
                "qbound_support_temporal_prob",
                "qbound_main_temporal_prob",
                "qbound_soft_temporal_prob",
                "qbound_training_temporal_prob",
                "qbound_hard_temporal_prob",
            )
            support_diffs = []
            for name in same_support_names:
                candidate = self._normalize_temporal_distribution(
                    qbound_aux[name],
                    valid_mask,
                )
                support_diffs.append(
                    (candidate.detach() - support_key.detach())
                    .abs()
                    .amax(dim=-1)
                )
            same_support_max_abs_diff = torch.stack(
                support_diffs,
                dim=0,
            ).amax(dim=0)
            if float(
                same_support_max_abs_diff.detach().max().item()
            ) > 1e-5:
                raise RuntimeError(
                    "Q-Bound V2.8 requires training, inference, grounding, "
                    "and QA to use the same continuous Support Field."
                )

            # Grounding preserves gradient to the semantic seed. QA receives
            # the exact same numerical support, detached from topology.
            training_key = support_key
            soft_key = support_key
            final_key = support_key.detach()

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
                (),
                dtype=final_key.dtype,
                device=final_key.device,
            )

            # VALUE is original masked CLIP, never a WHERE/support state.
            value_sequence = qbound_aux["value_tokens"]
            if value_sequence.shape != qbound_video_tokens.shape:
                raise ValueError(
                    "Q-Bound value_tokens must preserve [B,T,D]; expected "
                    f"{tuple(qbound_video_tokens.shape)}, got "
                    f"{tuple(value_sequence.shape)}."
                )
            expected_raw_value = (
                qbound_video_tokens
                * valid_mask.to(qbound_video_tokens.dtype).unsqueeze(-1)
            )
            if not torch.allclose(
                value_sequence.detach(),
                expected_raw_value.detach(),
                atol=1e-6,
                rtol=1e-6,
            ):
                raise RuntimeError(
                    "V2.8 VALUE must remain the original masked CLIP "
                    "sequence; WHERE/support states cannot replace it."
                )

            # Differentiable view for Video-Question grounding.
            (
                qbound_grounding_pooled_value,
                qbound_grounding_video_proj,
            ) = self._project_temporal_sequence(
                sequence=value_sequence,
                temporal_weights=training_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            # Same support values for QA, but no VQA -> topology gradient.
            (
                qbound_qa_pooled_value,
                qbound_qa_video_proj,
            ) = self._project_temporal_sequence(
                sequence=value_sequence,
                temporal_weights=final_key,
                projection_dropout_mask=projection_dropout_mask,
                do=False,
                x_star_proj=None,
            )

            qa_grounding_feature_max_abs_diff = (
                qbound_qa_video_proj.detach()
                - qbound_grounding_video_proj.detach()
            ).abs().amax(dim=-1)
            if float(
                qa_grounding_feature_max_abs_diff.max().item()
            ) > 1e-5:
                raise RuntimeError(
                    "QA and grounding must pool the same V2.8 Support Field."
                )

            video_proj = qbound_qa_video_proj
            grounding_video_proj = qbound_grounding_video_proj

            # ----------------------------------------------------------
            # Clean V2.8 integration contract.
            # ----------------------------------------------------------
            qbound_aux["qbound_key"] = final_key
            qbound_aux["qbound_qa_key"] = final_key
            qbound_aux["qbound_grounding_key"] = training_key
            qbound_aux["qbound_training_key"] = training_key
            qbound_aux["qbound_soft_key"] = soft_key
            qbound_aux["final_key"] = final_key
            qbound_aux["qbound_main_temporal_prob"] = final_key
            qbound_aux["qbound_hard_temporal_prob"] = final_key
            qbound_aux["qbound_value_sequence"] = value_sequence

            qbound_aux["qbound_qa_pooled_value"] = qbound_qa_pooled_value
            qbound_aux["qbound_grounding_pooled_value"] = (
                qbound_grounding_pooled_value
            )
            qbound_aux["qbound_hard_pooled_value"] = qbound_qa_pooled_value
            qbound_aux["qbound_soft_pooled_value"] = (
                qbound_grounding_pooled_value
            )
            qbound_aux["qbound_training_pooled_value"] = (
                qbound_grounding_pooled_value
            )

            qbound_aux["qbound_qa_video_proj"] = qbound_qa_video_proj
            qbound_aux["qbound_grounding_video_proj"] = (
                qbound_grounding_video_proj
            )
            qbound_aux["qbound_hard_video_proj"] = qbound_qa_video_proj
            qbound_aux["qbound_soft_video_proj"] = (
                qbound_grounding_video_proj
            )
            qbound_aux["qbound_training_video_proj"] = (
                qbound_grounding_video_proj
            )

            qbound_aux["qbound_same_support_max_abs_diff"] = (
                same_support_max_abs_diff
            )
            qbound_aux["qbound_qa_grounding_feature_max_abs_diff"] = (
                qa_grounding_feature_max_abs_diff
            )
            qbound_aux["qbound_soft_training_key_max_abs_diff"] = (
                torch.zeros_like(same_support_max_abs_diff)
            )

            dtype = x_bar.dtype
            device = x_bar.device
            one = torch.ones((), dtype=dtype, device=device)
            zero = torch.zeros((), dtype=dtype, device=device)

            qbound_aux["qbound_question_mask_provided"] = torch.as_tensor(
                float(question_mask_was_provided),
                dtype=dtype,
                device=device,
            )
            qbound_aux["qbound_input_is_generic"] = one
            qbound_aux["qbound_input_uses_cra_embedding"] = zero
            qbound_aux[
                "qbound_same_support_for_qa_and_grounding_enabled"
            ] = one
            qbound_aux["qbound_qa_support_detached_enabled"] = one
            qbound_aux[
                "qbound_grounding_gradient_to_semantic_seed_enabled"
            ] = one
            qbound_aux[
                "qbound_vqa_gradient_to_topology_enabled"
            ] = zero
            qbound_aux[
                "qbound_grounding_gradient_to_support_graph_enabled"
            ] = zero
            qbound_aux["qbound_predictive_loss_exposed"] = one if self.qbound_support_predictor_enabled else zero
            qbound_aux["qbound_soft_grounding_feature_exposed"] = one
            qbound_aux[
                "qbound_value_uses_question_conditioned_states"
            ] = zero
            qbound_aux["qbound_value_uses_raw_visual_tokens"] = one
            qbound_aux["functional_critic_used_in_decoder"] = zero
            qbound_aux[
                "answer_features_used_in_temporal_inference"
            ] = zero
            qbound_aux[
                "qbound_visual_enrichment_used_for_where"
            ] = one
            qbound_aux[
                "qbound_visual_enrichment_used_for_value"
            ] = zero
            qbound_aux[
                "qbound_question_conditions_semantic_relevance"
            ] = one
            # Historical routing flag refers to an older removed module;
            # do not reuse its name for EQ-PTC. New context diagnostic below
            # explicitly records early question-conditioned Bq interaction.
            qbound_aux[
                "qbound_question_routes_visual_context"
            ] = zero
            qbound_aux["qbound_eqptc_question_conditions_temporal_context"] = (
                one if self.qbound_temporal_context_mode == "eq_ptc" else zero
            )
            qbound_aux[
                "qbound_question_used_as_visual_value"
            ] = zero

            # CRA reference and official downstream views.
            qbound_aux["cra_key"] = cra_key
            qbound_aux["cra_video_proj"] = cra_video_proj
            qbound_aux["main_video_proj"] = video_proj
            qbound_aux["qbound_video_proj"] = video_proj
        else:
            final_key = cra_key
            training_key = cra_key
            soft_key = cra_key
            video_proj = cra_video_proj
            grounding_video_proj = cra_video_proj

        time_param = self._build_time_param(
            final_key=final_key,
            training_key=training_key,
            soft_key=soft_key,
            base_temporal=base_temporal,
            qbound_active=qbound_active,
            qbound_time=qbound_time,
        )

        # Explicitly separate the QA consumer from the grounding consumer.
        time_param["grounding_feature"] = grounding_video_proj
        time_param["qbound_grounding_feature"] = grounding_video_proj
        time_param["qbound_temporal_main_feature"] = grounding_video_proj
        time_param["qbound_feature"] = video_proj
        time_param["qbound_qa_feature"] = video_proj
        time_param["cra_grounding_feature"] = cra_video_proj

        # No direct hidden-event/support residual enters answer space.
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
            # Copy outputs without creating qbound_qbound_* duplicates.
            for key, value in qbound_aux.items():
                if key == "qbound_version":
                    continue
                if key.startswith("qbound_"):
                    time_param[key] = value
                else:
                    time_param[f"qbound_{key}"] = value

            time_param["qbound_version"] = self.qbound_version

            # Make the two optimization interfaces impossible to confuse.
            time_param["grounding_feature"] = grounding_video_proj
            time_param["qbound_grounding_feature"] = grounding_video_proj
            time_param["qbound_temporal_main_feature"] = (
                grounding_video_proj
            )
            time_param["qbound_feature"] = video_proj
            time_param["qbound_qa_feature"] = video_proj
            time_param["qbound_support_predictive_loss"] = qbound_aux[
                "support_predictive_loss"
            ]
            time_param["support_predictive_loss"] = qbound_aux[
                "support_predictive_loss"
            ]

            # Barycentric interval aliases are metadata/evaluation views only.
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
            time_param["qbound_barycenter_start_frame"] = qbound_aux[
                "barycenter_start_frame"
            ]
            time_param["qbound_barycenter_end_frame"] = qbound_aux[
                "barycenter_end_frame"
            ]
            time_param["qbound_soft_expected_span_length"] = qbound_aux[
                "soft_expected_span_length"
            ]
            time_param["qbound_soft_expected_span_fraction"] = qbound_aux[
                "soft_expected_span_fraction"
            ]

            # Legacy "calibrated" aliases now point to the continuous V2.8
            # Support Field and its barycentric interval view.
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

            architecture_flags = (
                "structured_visual_evidence_enabled",
                "visual_distinctiveness_enabled",
                "visual_transition_enabled",
                "temporal_context_enabled",
                "relevance_extent_decoupling_enabled",
                "token_resolved_question_relevance_enabled",
                "semantic_evidence_seed_enabled",
                "question_direct_span_scoring_enabled",
                "question_as_visual_value_enabled",
                "video_only_support_branch_enabled",
                "predictive_support_graph_enabled",
                "support_predictor_target_leakage_enabled",
                "support_branch_uses_question_enabled",
                "grounding_gradient_to_support_graph_enabled",
                "support_graph_detached_for_grounding_enabled",
                "structural_seed_propagation_enabled",
                "analytic_boundary_posterior_enabled",
                "posterior_barycentric_grounding_enabled",
                "same_support_for_training_and_inference_enabled",
                "value_where_separation_enabled",
                "query_routed_visual_context_enabled",
                "evidence_core_enabled",
                "event_continuity_enabled",
                "joint_transition_continuity_enabled",
                "core_to_closure_enabled",
                "additive_extent_energy_enabled",
                "neural_span_scorer_enabled",
                "event_candidate_selector_enabled",
                "hard_map_temporal_decision_enabled",
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
                "minimum_event_duration_enabled",
            )
            for name in architecture_flags:
                time_param[f"qbound_{name}"] = qbound_aux[name]

            time_param["qbound_handover_enabled"] = torch.zeros(
                (),
                dtype=video_proj.dtype,
                device=video_proj.device,
            )
            time_param[
                "qbound_direct_reasoning_residual_enabled"
            ] = torch.zeros(
                (),
                dtype=video_proj.dtype,
                device=video_proj.device,
            )

        time_param["qbound_aux"] = qbound_aux
        return video_proj, final_key, time_param
