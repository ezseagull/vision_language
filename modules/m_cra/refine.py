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
    CRA downstream decoder with Q-Bound R2-C and optional V3.5 pairing.

    V3.5 adds ONLY train-time paired-question evidence outputs for QDTG;
    ordinary forward/inference, state_dict, QA-gradient firewall and fixed
    seed-to-support propagation are unchanged. ``qbound_version`` remains
    "2.8.0" for exact R2-C compatibility until the CRA wrapper/YAML migration.

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

    V2.8 has no Question-to-span scorer, Core-to-Closure, Joint Continuity,
    event proposal bank, BoundaryRefiner, answer critic, learned duration
    prior, Gaussian smoothing, or event-count target.

    V3.8-Minimal (M0) integration is additive, not a replacement of R2-C:
      * The existing QBoundCore exports H and question features for the
        separate fixed-window event bank and joint answer-event scorer.
      * The new event head is NOT instantiated in RefineDecoder. It belongs
        to the CRA wrapper; it must receive question/answer candidates there.
      * ``decode_v38_event_support`` applies the SAME existing analytic R1
        and adaptive grounding wrapper to a scorer-produced [B,T] support.
      * The old Support Field, CRA-only QA, QDTG pair interface, and all old
        gradients remain unchanged by default. Event-only vs old-support
        selection is an experiment/pipeline decision, never silent fusion.
      * No answer or temporal ground truth is passed into QBoundCore.
      * The optional question-only selector uses the existing bank geometry
        and R1. Its differentiable scores are separate from CRA QA and M0.
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
        # ``qbound_version`` is the unchanged R2-C *core wire version*.
        # The V3.8-Minimal experiment opts in via a separate version label;
        # it does not rename core parameters, bypass legacy validation, or
        # silently replace the old QA / grounding outputs.
        if configured_version not in (self.qbound_version, "3.8-minimal"):
            raise ValueError(
                "Unsupported model.qbound.version: expected '2.8.0' "
                "(R2-C/V3.5 interface) or '3.8-minimal' (M0 adapter), got "
                f"{configured_version!r}."
            )
        self.qbound_experiment_version = configured_version
        self.qbound_v38_minimal_enabled = (configured_version == "3.8-minimal")
        self.qbound_window_selector_input = qbound_cfg.get("window_selector_input")
        if self.qbound_window_selector_input not in (None, "local", "contextual"):
            raise ValueError(
                "model.qbound.window_selector_input must be null, 'local', or 'contextual'."
            )
        if self.qbound_window_selector_input is not None and not self.qbound_v38_minimal_enabled:
            raise ValueError("The window selector requires model.qbound.version='3.8-minimal'.")

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

        if self.qbound_v38_minimal_enabled:
            if not self.qbound_enabled:
                raise ValueError("V3.8-Minimal requires model.qbound.enabled=true.")
            if graph_mode != "fixed" or predictor_enabled:
                raise ValueError(
                    "V3.8-Minimal M0 must retain V3.5 R2-C: "
                    "support_graph_mode='fixed' and support_predictor_enabled=false."
                )

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
        self.qbound_window_selector = None
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
            self.qbound = QBoundCore(**core_kwargs)
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

            if self.qbound_v38_minimal_enabled:
                # Fail fast if a previous core.py was accidentally installed.
                for method_name in (
                    "forward_v38_features", "decode_v38_event_support",
                ):
                    if not callable(getattr(self.qbound, method_name, None)):
                        raise RuntimeError(
                            "V3.8-Minimal requires the updated "
                            f"modules/qbound/core.py ({method_name} missing)."
                        )
            if self.qbound_window_selector_input is not None:
                from modules.qbound.core import QBoundWindowSelector

                self.qbound_window_selector = QBoundWindowSelector(
                    hidden_dim=self.qbound_hidden_dim,
                    num_question_heads=core_kwargs["num_question_heads"],
                    dropout=core_kwargs["dropout"],
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
    # V3.8-Minimal (M0): non-invasive feature and R1 interfaces
    # ------------------------------------------------------------------
    def _validate_v38_minimal_encoder_view(
        self, view, valid_mask, question_length
    ):
        """Validate the exact unchanged-encoder contract for event evidence.

        M0 assumes left-aligned valid frame positions: candidate windows must
        never include padding. This validation is only active for V3.8-M0,
        never a hidden new restriction on the historical R2-C baseline.
        """
        if not self.qbound_v38_minimal_enabled:
            return
        batch_size, time_steps = valid_mask.shape
        d = int(self.qbound_hidden_dim)
        expected_shapes = {
            "qbound_v38_event_frame_states": (batch_size, time_steps, d),
            "qbound_v38_question_embedding": (batch_size, d),
            "qbound_v38_question_tokens": (batch_size, question_length, d),
            "qbound_v38_event_video_mask": (batch_size, time_steps),
            "qbound_v38_event_question_mask": (batch_size, question_length),
            "qbound_v38_event_valid_lengths": (batch_size,),
        }
        if self.qbound_window_selector_input == "local":
            expected_shapes["qbound_v38_local_frame_states"] = (batch_size, time_steps, d)
        missing = sorted(set(expected_shapes) - set(view))
        if missing:
            raise KeyError("V3.8-Minimal core encoder view missing: " + ", ".join(missing))
        for name, shape in expected_shapes.items():
            self._assert_shape(name, view[name], shape)
            self._assert_finite(name, view[name])
        if not torch.equal(view["qbound_v38_event_video_mask"], valid_mask):
            raise RuntimeError("M0 frame mask differs from the legacy video mask.")
        q_mask = view["qbound_v38_event_question_mask"]
        if q_mask.dtype != torch.bool or not torch.all(q_mask.any(dim=-1)):
            raise ValueError("M0 question mask must be boolean with valid tokens.")
        lengths = valid_mask.long().sum(dim=-1)
        if not torch.equal(view["qbound_v38_event_valid_lengths"], lengths):
            raise RuntimeError("M0 valid temporal lengths mismatch.")
        if time_steps > 1 and bool(((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any()):
            raise ValueError(
                "M0 fixed-window bank requires left-aligned valid frames "
                "(no internal holes in temporal video_mask)."
            )
        H = view["qbound_v38_event_frame_states"]
        q_tokens = view["qbound_v38_question_tokens"]
        if bool(H.masked_select(~valid_mask.unsqueeze(-1)).abs().gt(1e-5).any()):
            raise RuntimeError("M0 encoder H has nonzero padded frame features.")
        if self.qbound_window_selector_input == "local":
            local_frames = view["qbound_v38_local_frame_states"]
            if bool(local_frames.masked_select(~valid_mask.unsqueeze(-1)).abs().gt(1e-5).any()):
                raise RuntimeError("Local encoder view has nonzero padded frame features.")
        if bool(q_tokens.masked_select(~q_mask.unsqueeze(-1)).abs().gt(1e-5).any()):
            raise RuntimeError("M0 question tokens have nonzero padding features.")
        q_mean = self._masked_mean_video(q_tokens, q_mask)
        if not torch.allclose(
            view["qbound_v38_question_embedding"], q_mean, atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError("M0 question embedding disagrees with masked token mean.")

    def forward_v38_encoder_features(
        self, qbound_video_tokens, question_feature,
        self_mask=None, question_mask=None,
    ):
        """Optional P2 frozen-feature probe: no CRA QA / old Seed / graph.

        Shares QBoundCore weights and source-input detach policy with the full
        ``forward`` path. No ``torch.no_grad()`` is imposed here: freezing the
        existing encoder is an explicit training-policy decision. This method
        does not compute answers, event windows, or temporal targets.
        """
        if not self.qbound_enabled:
            raise RuntimeError("V3.8 encoder probe requires enabled QBoundCore.")
        if not callable(getattr(self.qbound, "forward_v38_features", None)):
            raise RuntimeError("Updated QBoundCore.forward_v38_features is required.")
        if qbound_video_tokens.ndim != 3 or question_feature.ndim != 3:
            raise ValueError("M0 probe expects video [B,T,D] and question [B,L,D].")
        batch_size, time_steps, dim = qbound_video_tokens.shape
        if dim != self.embed_dim or question_feature.shape[0] != batch_size:
            raise ValueError("M0 probe video/question dimensions do not match decoder.")
        if question_feature.shape[-1] != self.embed_dim:
            raise ValueError("M0 probe question embedding dimension mismatch.")
        valid_mask = self._normalize_video_mask(
            self_mask, batch_size, time_steps, qbound_video_tokens.device
        )
        q_mask = self._normalize_question_mask(
            question_mask, question_feature, qbound_video_tokens.device
        )
        view = self.qbound.forward_v38_features(
            video_tokens=qbound_video_tokens.detach(),
            question_features=question_feature.detach(),
            video_mask=valid_mask,
            question_mask=q_mask,
        )
        self._validate_v38_minimal_encoder_view(
            view, valid_mask, question_feature.size(1)
        )
        return view

    def decode_v38_event_support(self, event_temporal_prob, video_mask):
        """Decode a *scorer-produced* target-only temporal distribution.

        This accepts target-only temporal mass from the question-window
        selector or from any of the M0 posteriors.

        Returns independent M0 outputs. Does NOT overwrite the legacy
        ``final_key``, CRA QA features, QDTG loss, or old-support grounding.
        The returned differentiable R1 interval can be used for diagnostics
        and carefully controlled loss experiments; the wrapper receives the
        detached inference view, exactly as in the old support path.
        """
        if not self.qbound_enabled:
            raise RuntimeError("M0 event-support decoder requires QBoundCore.")
        if not callable(getattr(self.qbound, "decode_v38_event_support", None)):
            raise RuntimeError("Updated QBoundCore.decode_v38_event_support is required.")
        if not torch.is_tensor(event_temporal_prob) or event_temporal_prob.ndim != 2:
            raise ValueError("M0 event_temporal_prob must be tensor [B,T].")
        if not torch.is_floating_point(event_temporal_prob):
            raise TypeError("M0 event_temporal_prob must have floating-point dtype.")
        batch_size, time_steps = event_temporal_prob.shape
        valid_mask = self._normalize_video_mask(
            video_mask, batch_size, time_steps, event_temporal_prob.device
        )
        # Reject invalid mass rather than quietly fixing scorer/padding bugs.
        self._assert_distribution("M0 event_temporal_prob", event_temporal_prob, valid_mask)
        if bool(event_temporal_prob.masked_select(~valid_mask).abs().gt(1e-6).any()):
            raise ValueError("M0 event_temporal_prob assigns mass to padded frames.")
        if time_steps > 1 and bool(((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any()):
            raise ValueError("M0 event support requires left-aligned temporal masks.")

        interval = self.qbound.decode_v38_event_support(
            event_temporal_prob=event_temporal_prob,
            video_mask=valid_mask,
        )
        # Same implementation and endpoint convention as the old path:
        # old: qbound_aux[selected_*] -> gen_adaptive_grounding(final_key).
        # M0:  interval[selected_*]   -> gen_adaptive_grounding(event_key).
        inference_key = event_temporal_prob.detach()
        adaptive = self.grounding_module.gen_adaptive_grounding(
            inference_key,
            selected_span_start_frame=interval["selected_span_start_frame"],
            selected_span_end_frame=interval["selected_span_end_frame"],
            valid_mask=valid_mask,
            selected_frame_mask=interval["selected_span_frame_mask"],
        )
        # Keep this standalone: _build_time_param needs a CRA reference and
        # would silently make it look like we replaced the old training path.
        adaptive["pre_smooth_key"] = inference_key
        adaptive["post_smooth_key"] = inference_key
        adaptive["qbound_key"] = inference_key
        adaptive["qbound_training_key"] = event_temporal_prob
        adaptive["qbound_soft_key"] = event_temporal_prob
        adaptive["qbound_smoothing_applied"] = inference_key.new_zeros(())
        return {
            "event_temporal_prob": event_temporal_prob,
            "event_inference_temporal_prob": inference_key,
            "event_r1_interval": interval,
            "event_adaptive_time_param": adaptive,
        }

    def score_v38_windows(self, encoder_view, event_bank):
        """Score the existing bank; caller may decode event_temporal_question.

        The CRA wrapper supplies its existing FixedTemporalEventBank and the
        encoder view in time_param["qbound_aux"] (or the encoder probe view).
        Encoder features are held fixed for the local/contextual comparison.
        Keep event_question_prior_logits attached for the train-time rank loss.
        """
        if self.qbound_window_selector is None:
            raise RuntimeError("Set model.qbound.window_selector_input to enable window scoring.")
        from modules.qbound.event_evidence import score_question_windows

        frame_key = (
            "qbound_v38_local_frame_states"
            if self.qbound_window_selector_input == "local"
            else "qbound_v38_event_frame_states"
        )
        video_mask = encoder_view["qbound_v38_event_video_mask"]
        return score_question_windows(
            self.qbound_window_selector,
            event_bank.geometry(video_mask),
            frame_states=encoder_view[frame_key].detach(),
            question_tokens=encoder_view["qbound_v38_question_tokens"].detach(),
            video_mask=video_mask,
            question_mask=encoder_view["qbound_v38_event_question_mask"],
        )

    def decode_v38_grounding_views(self, posteriors, video_mask):
        """Independent R1 decoding for P_Q, marginal, conditional, old-support.

        ``posteriors`` maps diagnostic names to normalized [B,T] support
        distributions. This does no learned fusion and uses exactly the same
        decoder for every entry. The caller decides how to report/select them.
        """
        if not isinstance(posteriors, dict) or not posteriors:
            raise ValueError("posteriors must be a nonempty name -> [B,T] dict.")
        return {
            name: self.decode_v38_event_support(distribution, video_mask)
            for name, distribution in posteriors.items()
        }

    # ------------------------------------------------------------------
    # V3.5 paired evidence (auxiliary TRAINING interface only)
    # ------------------------------------------------------------------
    def _validate_v35_pair(self, pair, pair_valid_mask, valid_mask, pair_length):
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

            # Main R2-C forward remains bit-for-bit unchanged when no paired
            # question is supplied. In pair mode, Q1/Q2 share one temporal
            # encoding and one fixed, detached support graph.
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
            self._validate_v38_minimal_encoder_view(
                qbound_aux, valid_mask, qbound_language.size(1)
            )
            if qbound_pair_aux is not None:
                self._validate_v35_pair(
                    qbound_pair_aux,
                    normalized_pair_valid_mask,
                    valid_mask,
                    pair_language.size(1),
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

            # Diagnostics only. No tensor, loss or gradient is changed here.
            # Copied to time_param as "qbound_support_graph_mode" below.
            qbound_aux["support_graph_mode"] = self.qbound_support_graph_mode
            qbound_aux["support_predictor_enabled"] = self.qbound_support_predictor_enabled
            qbound_aux["qbound_v38_experiment_version"] = self.qbound_experiment_version
            qbound_aux["qbound_v38_minimal_enabled"] = torch.as_tensor(
                float(self.qbound_v38_minimal_enabled),
                dtype=x_bar.dtype, device=x_bar.device,
            )

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
            qbound_aux[
                "qbound_question_routes_visual_context"
            ] = zero
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
