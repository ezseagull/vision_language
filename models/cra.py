import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_cra.only_trans import LanModel
from modules.m_tempclip.transformer import VideoEmbedding
from modules.m_cra.only_trans import Decoder as BaselineDecoder
from modules.m_cra.refine import Decoder as RefineDecoder
from modules.m_cra.causal_module import LinguisticInterventionGraph


class CRA(nn.Module):
    """
    CRA answer host and controlled downstream wrapper for R2-C / V3.5.

    V3.5 QDTG adds a TRAIN-ONLY second question for the SAME video.
    It reuses the same LanModel question-only encoder and the shared
    QBoundCore. The second question never enters CRA QA, original InfoNCE,
    temporal inference, or checkpoint parameters. Loss is implemented later
    in cra_pipeline.py, not in this wrapper.

    V2.8 factorizes temporal grounding as:

        raw/local CLIP + question tokens -> semantic evidence seed
        detached raw/local CLIP          -> predictive support graph
        seed + stopgrad(graph)           -> continuous Support Field
        Support Field + raw CLIP VALUE   -> Grounding and QA evidence

    Wrapper contract
    ----------------
    1. Q-Bound receives raw/generic video features, never CRA x_bar.
    2. Q-Bound receives question-only token features, never
       [Question SEP Answer] candidate features.
    3. CRA grounding remains a detached reference while Q-Bound is active.
    4. Grounding and QA use the same continuous Support Field.
    5. The QA view is detached from temporal topology; L_VQA can still train
       the ordinary answer-space projection/classifier.
    6. The grounding view remains differentiable to semantic relevance.
    7. The support graph is video-only and detached from L_ground.
    8. A/B predictor is trained by L_pred; R2-C has no predictor or L_pred.
    9. Raw masked CLIP remains VALUE; WHERE/support states are never pooled.
   10. Barycentric span endpoints are metadata/evaluation views only.
   11. Temporal GT, IoU, IoP, and answers never enter Q-Bound.
   12. V3.5's Q2 uses question-only tokens, is training-only, and its
        question-global embedding is detached as a contrastive target.

    CRA therefore remains the multiple-choice answer host without becoming
    part of Q-Bound's relevance or extent decision.

    V3.7 (opt-in): explicit masked/legacy language pooling selection and
    question-conditioned event-relative Seed refinement inside QBoundCore.
    CRA preserves its original answer head, GT-free temporal inference,
    detached QA support, and training-only Q2 interface. Event candidates,
    residual fusion, and their consistency are validated rather than
    silently suppressed by the historical V2.8 flag contract.
    """

    def __init__(self, cfgs, tokenizer) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.tokenizer = tokenizer

        model_cfg = cfgs["model"]
        dataset_cfg = cfgs["dataset"]

        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        self.baseline = model_cfg["baseline"]
        self.qmax_words = dataset_cfg["qmax_words"]
        self.max_feats = dataset_cfg["max_feats"]
        self.n_negs = model_cfg["n_negs"]

        self.qbound_enabled = bool(
            qbound_cfg.get(
                "enabled",
                model_cfg.get(
                    "qbound_enabled",
                    model_cfg.get("gpeb_enabled", False),
                ),
            )
        )
        self.qbound_version = str(
            qbound_cfg.get(
                "version",
                model_cfg.get(
                    "qbound_version",
                    model_cfg.get("gpeb_version", "2.8.0"),
                ),
            )
        )
        if self.qbound_enabled and self.qbound_version != "2.8.0":
            raise ValueError(
                "Q-Bound is enabled, but this CRA wrapper expects "
                "model.qbound.version='2.8.0'; got "
                f"{self.qbound_version!r}."
            )

        # H1: experiment identity is separate from the stable R2-C interface
        # version. The previous CRA code silently ignored EQ-PTC config, which
        # could make an apparent H1 experiment run the old temporal module.
        # RefineDecoder and QBoundCore must agree with this wrapper exactly.
        self.qbound_temporal_context_mode = None
        self.qbound_eqptc_scales = {}
        # V3.7 switches are distinct from the stable V2.8.0 interface ABI.
        self.qbound_event_grounding_mode = "legacy"
        self.qbound_language_pooling_mode = "legacy"
        self.qbound_event_kwargs = {}
        if self.qbound_enabled:
            mode = qbound_cfg.get("temporal_context_mode", "eq_ptc")
            if not isinstance(mode, str) or mode not in ("legacy", "eq_ptc"):
                raise ValueError(
                    "model.qbound.temporal_context_mode must be 'legacy' or 'eq_ptc'."
                )
            self.qbound_temporal_context_mode = mode
            for key in (
                "eqptc_early_scale", "eqptc_position_scale", "eqptc_temporal_scale"
            ):
                raw = qbound_cfg.get(key, 0.25)
                if isinstance(raw, bool):
                    raise TypeError(f"model.qbound.{key} cannot be boolean.")
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"model.qbound.{key} must be a float in [0,1]."
                    ) from exc
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"model.qbound.{key} must be finite and in [0,1]."
                    )
                self.qbound_eqptc_scales[key] = value

            # V3.7: keep legacy configs behavior unchanged unless the new
            # event_relative mode (or an explicit pooling-only ablation) is
            # selected. The final V3.7 YAML must set both modes explicitly.
            event_mode = qbound_cfg.get("event_grounding_mode", "legacy")
            if not isinstance(event_mode, str) or event_mode not in (
                "legacy", "event_relative",
            ):
                raise ValueError(
                    "model.qbound.event_grounding_mode must be 'legacy' "
                    "or 'event_relative'."
                )
            self.qbound_event_grounding_mode = event_mode
            pooling_default = "masked" if event_mode == "event_relative" else "legacy"
            pooling_mode = qbound_cfg.get("language_pooling_mode", pooling_default)
            if not isinstance(pooling_mode, str) or pooling_mode not in (
                "legacy", "masked",
            ):
                raise ValueError(
                    "model.qbound.language_pooling_mode must be 'legacy' "
                    "or 'masked'."
                )
            self.qbound_language_pooling_mode = pooling_mode

            self.qbound_event_kwargs = {
                "event_grounding_mode": event_mode,
            }
            for field, default in (
                ("event_residual_scale", 0.10),
                ("event_relation_temperature", 0.20),
            ):
                raw = qbound_cfg.get(field, default)
                if isinstance(raw, bool):
                    raise TypeError(f"model.qbound.{field} cannot be boolean.")
                try:
                    value = float(raw)
                except (TypeError, ValueError) as exc:
                    raise TypeError(
                        f"model.qbound.{field} must be a numeric value."
                    ) from exc
                if not math.isfinite(value) or value <= 0.0:
                    raise ValueError(
                        f"model.qbound.{field} must be finite and > 0."
                    )
                if field == "event_residual_scale" and value > 1.0:
                    raise ValueError(
                        "model.qbound.event_residual_scale must be <= 1."
                    )
                self.qbound_event_kwargs[field] = value
            max_length = qbound_cfg.get("event_candidate_max_length", None)
            if max_length is not None and (
                isinstance(max_length, bool)
                or not isinstance(max_length, int)
                or max_length < 1
            ):
                raise ValueError(
                    "model.qbound.event_candidate_max_length must be "
                    "null or an integer >= 1."
                )
            self.qbound_event_kwargs["event_candidate_max_length"] = max_length
            if event_mode == "event_relative" and mode != "eq_ptc":
                raise ValueError(
                    "V3.7 event_relative requires temporal_context_mode='eq_ptc'."
                )

            # EQ-PTC H1 must compare with V3.5 R2-C, not accidentally change
            # support extent or introduce the unrelated predictive objective.
            if mode == "eq_ptc" and (
                qbound_cfg.get("support_graph_mode", "learned") != "fixed"
                or qbound_cfg.get("support_predictor_enabled", True) is not False
            ):
                raise ValueError(
                    "EQ-PTC H1 requires R2-C: support_graph_mode='fixed' "
                    "and support_predictor_enabled=false."
                )

        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get("qbound_require_question_only", True),
            )
        )
        if self.qbound_enabled and not self.qbound_require_question_only:
            raise ValueError(
                "Controlled Q-Bound V2.8 requires "
                "model.qbound.require_question_only=true."
            )

        self.qbound_require_token_question = bool(
            qbound_cfg.get("require_token_question", True)
        )
        if self.qbound_enabled and not self.qbound_require_token_question:
            raise ValueError(
                "Controlled Q-Bound V2.8 requires token-level question "
                "features. Set model.qbound.require_token_question=true."
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

        if self.qbound_enabled and bool(model_cfg.get("do", False)):
            raise ValueError(
                "Controlled Q-Bound V2.8 requires model.do=false. "
                "VisualIntervention would confound temporal grounding."
            )

        # Compatibility aliases retained for repository call sites.
        self.gpeb_enabled = self.qbound_enabled
        self.gpeb_version = self.qbound_version
        self.gpeb_require_question_only = (
            self.qbound_require_question_only
        )

        d_model = model_cfg["d_model"]
        # Fail fast if CRA is paired with a pre-V3.7 only_trans.py. When
        # Q-Bound is disabled, retain exact legacy CRA pooling behavior.
        import inspect
        if "pooling_mode" not in inspect.signature(LanModel.__init__).parameters:
            raise RuntimeError(
                "V3.7 CRA requires matching only_trans.py with pooling_mode."
            )
        self.lang_model = LanModel(
            tokenizer,
            lan=model_cfg["lan"],
            out_dim=d_model,
            pooling_mode=self.qbound_language_pooling_mode,
        )
        if getattr(self.lang_model, "pooling_mode", None) != self.qbound_language_pooling_mode:
            raise RuntimeError("CRA/LanModel language pooling mode mismatch.")
        self.video_model = VideoEmbedding(cfgs)

        # Kept only for repository/checkpoint compatibility.
        self.do = LinguisticInterventionGraph(cfgs)

        if self.qbound_enabled and self.baseline != "refine":
            raise ValueError(
                "Q-Bound V2.8 requires model.baseline='refine'; otherwise "
                "the enabled Q-Bound branch would be silently bypassed."
            )

        if self.baseline == "baseline":
            self.decoder = BaselineDecoder()
            self.forward = self.baseline_forward
        elif self.baseline == "refine":
            self.decoder = RefineDecoder(
                num_layer=model_cfg["num_layers"],
                cfgs=cfgs,
            )
            if self.qbound_enabled:
                decoder_version = str(
                    getattr(self.decoder, "qbound_version", "")
                )
                if decoder_version != self.qbound_version:
                    raise RuntimeError(
                        "CRA/RefineDecoder Q-Bound version mismatch: "
                        f"CRA={self.qbound_version!r}, "
                        f"decoder={decoder_version!r}."
                    )

                # Fail early when a patched CRA is accidentally paired with
                # the old refine.py or core.py. This is not a version bump.
                actual_mode = getattr(
                    self.decoder, "qbound_temporal_context_mode", None
                )
                if actual_mode != self.qbound_temporal_context_mode:
                    raise RuntimeError(
                        "CRA/RefineDecoder EQ-PTC mode mismatch: "
                        f"configured={self.qbound_temporal_context_mode!r}, "
                        f"decoder={actual_mode!r}."
                    )
                core = getattr(self.decoder, "qbound", None)
                if core is None or getattr(core, "temporal_context_mode", None) != actual_mode:
                    raise RuntimeError(
                        "CRA expects a matching EQ-PTC-enabled QBoundCore "
                        "inside RefineDecoder."
                    )
                for key, expected in self.qbound_eqptc_scales.items():
                    actual = getattr(core, key, None)
                    if actual is None or not math.isclose(
                        float(actual), expected, rel_tol=0.0, abs_tol=1e-8
                    ):
                        raise RuntimeError(
                            f"CRA/RefineDecoder/Core EQ-PTC scale mismatch: {key}."
                        )

                # V3.7 must be active in the actual decoder/core, not just YAML.
                decoder_event = getattr(
                    self.decoder, "qbound_event_grounding_mode", None,
                )
                core_event = getattr(core, "event_grounding_mode", None)
                if (decoder_event != self.qbound_event_grounding_mode
                        or core_event != self.qbound_event_grounding_mode):
                    raise RuntimeError(
                        "CRA/RefineDecoder/Core event grounding mode mismatch."
                    )
                for key, expected in self.qbound_event_kwargs.items():
                    if key == "event_grounding_mode":
                        continue
                    actual = getattr(core, key, None)
                    if key == "event_candidate_max_length":
                        if actual != expected:
                            raise RuntimeError(
                                "CRA/Core event_candidate_max_length mismatch."
                            )
                    elif actual is None or not math.isclose(
                        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-8,
                    ):
                        raise RuntimeError(f"CRA/Core V3.7 scale mismatch: {key}.")
            self.forward = self.refine_forward
        else:
            raise ValueError(
                "Unknown model.baseline value: "
                f"{self.baseline!r}. Expected 'baseline' or 'refine'."
            )

    # ------------------------------------------------------------------
    # Language / input helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare_answer_indices(
        answer_id,
        batch_size: int,
        device: torch.device,
    ):
        if answer_id is None:
            return None

        answer_indices = answer_id.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).reshape(-1)
        if answer_indices.numel() != batch_size:
            raise ValueError(
                "answer_id must contain one candidate index per sample. "
                f"Expected {batch_size} values, got "
                f"{answer_indices.numel()}."
            )
        return answer_indices

    @staticmethod
    def _select_candidate_features(
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must be [B,C,D], got "
                f"{tuple(candidate_features.shape)}."
            )
        if candidate_indices.ndim != 1:
            raise ValueError(
                "candidate_indices must be [B], got "
                f"{tuple(candidate_indices.shape)}."
            )

        batch_indices = torch.arange(
            candidate_features.size(0),
            device=candidate_features.device,
        )
        return candidate_features[batch_indices, candidate_indices]

    def _encode_candidates(
        self,
        tokens,
        device: torch.device,
        name: str,
    ):
        if tokens is None:
            return None, None

        tokens = tokens.to(device=device, non_blocking=True)
        if tokens.ndim != 3:
            raise ValueError(
                f"{name} tokens must have shape [B,C,L], got "
                f"{tuple(tokens.shape)}."
            )

        if hasattr(self.lang_model, "encode_candidates"):
            candidate_global, candidate_tokens = (
                self.lang_model.encode_candidates(tokens)
            )
        else:
            candidate_global, candidate_tokens = self.lang_model(tokens)

        if candidate_global.ndim != 3:
            raise ValueError(
                "LanModel must return candidate features [B,C,D], got "
                f"{tuple(candidate_global.shape)} for {name}."
            )
        return candidate_global, candidate_tokens

    def _encode_answers(
        self,
        answer,
        device: torch.device,
    ):
        """
        Encode existing CRA [Question SEP Answer_i] candidates.

        These candidates remain downstream answer-space representations only.
        """
        if answer is None:
            raise ValueError("answer tokens must not be None.")
        return self._encode_candidates(
            answer,
            device=device,
            name="answer",
        )

    def _question_mask_from_padded_ids(
        self,
        question_ids: torch.Tensor,
    ):
        if question_ids.ndim != 2:
            raise ValueError("question_ids must be [B,L].")

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id for Q-Bound V2.8."
            )

        mask = question_ids.ne(int(pad_token_id))

        # Preserve a leading special token for tokenizer families that reuse
        # id=0 for the supplied leading token and right padding.
        if int(pad_token_id) == 0 and mask.size(1) > 0:
            mask[:, 0] = True

        if not torch.all(mask.any(dim=-1)):
            raise ValueError(
                "Every question must contain at least one valid token."
            )
        return mask

    def _encode_question(
        self,
        question,
        device: torch.device,
    ):
        """
        Encode QUESTION ONLY.

        Returns
        -------
        question_global : [B,D]
        question_tokens : [B,L,D]
        question_mask   : [B,L] bool
        """
        if question is None:
            return None, None, None

        question = question.to(device=device, non_blocking=True)
        if question.ndim == 3 and question.size(1) == 1:
            question = question.squeeze(1)
        if question.ndim != 2:
            raise ValueError(
                "Question-only tokens must be [B,L] or [B,1,L], got "
                f"{tuple(question.shape)}."
            )

        question_mask = self._question_mask_from_padded_ids(question)
        if hasattr(self.lang_model, "encode_question"):
            question_global, question_tokens = (
                self.lang_model.encode_question(question)
            )
        else:
            question_global, question_tokens = self.lang_model(question)

        if question_global.ndim != 2:
            raise ValueError(
                "LanModel must return question features [B,D], got "
                f"{tuple(question_global.shape)}."
            )
        if question_tokens.ndim != 3:
            raise ValueError(
                "LanModel must return token-level question features [B,L,D], "
                f"got {tuple(question_tokens.shape)}."
            )
        if question_tokens.shape[:2] != question_mask.shape:
            raise ValueError(
                "Question token features and question mask do not align."
            )
        return question_global, question_tokens, question_mask

    @staticmethod
    def _move_video_mask(video_mask, device: torch.device):
        if video_mask is None:
            return None
        return video_mask.to(device=device, non_blocking=True)

    @staticmethod
    def _scalar_flag(mapping, name):
        if name not in mapping:
            raise KeyError(f"Missing Q-Bound flag {name!r}.")
        value = mapping[name]
        if not torch.is_tensor(value):
            raise TypeError(
                f"Q-Bound flag {name!r} must be a tensor."
            )
        if value.numel() != 1:
            raise ValueError(
                f"Q-Bound flag {name!r} must be scalar."
            )
        return float(value.detach().item())

    @staticmethod
    def _assert_shape(name, value, expected):
        if tuple(value.shape) != tuple(expected):
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got "
                f"{tuple(value.shape)}."
            )

    @staticmethod
    def _assert_finite(name, value):
        if not torch.isfinite(value).all():
            raise RuntimeError(f"{name} contains non-finite values.")

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
            raise RuntimeError(
                f"{name} must sum to one over valid frames."
            )

        invalid_mass = (
            value * (~valid_mask).to(value.dtype)
        ).abs().amax()
        if float(invalid_mass.detach().item()) > atol:
            raise RuntimeError(f"{name} assigns mass to padded frames.")

    # ------------------------------------------------------------------
    # H1 EQ-PTC integration contract (wrapper checks, no new trainable paths)
    # ------------------------------------------------------------------
    def _validate_eqptc_contract(
        self, time_param, qbound_aux, batch_size, time_steps
    ):
        """Assert that the configured B+C experiment actually ran.

        This check has no backward side effects and does not introduce QA
        gradients, answer tokens or GT into QBoundCore.
        """
        if not self.qbound_enabled:
            return

        mode = self.qbound_temporal_context_mode
        if qbound_aux.get("temporal_context_mode") != mode:
            raise RuntimeError("CRA/Core temporal_context_mode mismatch in forward.")
        if time_param.get("qbound_temporal_context_mode") != mode:
            raise RuntimeError("CRA/Refine time_param temporal_context_mode mismatch.")

        expected_active = float(mode == "eq_ptc")
        for mapping, key in (
            (qbound_aux, "eqptc_enabled_flag"),
            (time_param, "qbound_eqptc_enabled_flag"),
            (qbound_aux, "qbound_eqptc_question_conditions_temporal_context"),
        ):
            if self._scalar_flag(mapping, key) != expected_active:
                raise RuntimeError(f"EQ-PTC activation mismatch: {key}.")

        for key, value in self.qbound_eqptc_scales.items():
            actual = self._scalar_flag(qbound_aux, key + "_used")
            if not math.isclose(actual, value, rel_tol=1e-5, abs_tol=1e-6):
                raise RuntimeError(f"EQ-PTC scale mismatch at CRA boundary: {key}.")

        # Existing V3.5 and R2-C contracts are unchanged. In EQ mode only,
        # verify new B+C tensors actually flow through the CRA wrapper.
        if mode == "eq_ptc":
            q_len = qbound_aux["question_mask"].size(1)
            head_count = self.decoder.qbound.num_question_heads
            hidden = self.decoder.qbound.hidden_dim
            expected = {
                "eqptc_global_context_states": (batch_size, time_steps, hidden),
                "eqptc_conditioned_visual_states": (batch_size, time_steps, hidden),
                "eqptc_early_question_attention":
                    (batch_size, head_count, time_steps, q_len),
                "eqptc_temporal_attention":
                    (batch_size, head_count, time_steps, time_steps),
            }
            for name, shape in expected.items():
                if name not in qbound_aux:
                    raise KeyError(f"EQ-PTC core output {name!r} is missing.")
                self._assert_shape(name, qbound_aux[name], shape)
                self._assert_finite(name, qbound_aux[name])
            for name in (
                "qbound_eqptc_early_delta_ratio",
                "qbound_eqptc_temporal_delta_ratio",
            ):
                if name not in time_param:
                    raise KeyError(f"EQ-PTC refine diagnostic {name!r} is missing.")
                self._assert_shape(name, time_param[name], (batch_size,))
                self._assert_finite(name, time_param[name])

    # ------------------------------------------------------------------
    # V3.7 question-conditioned event-relative integration contract.
    # ------------------------------------------------------------------
    def _validate_event_relative_contract(
        self, time_param, qbound_aux, batch_size, time_steps,
    ):
        """Validate event mode, Q1 candidates and CRA forwarding (no GT)."""
        mode = self.qbound_event_grounding_mode
        if qbound_aux.get("event_grounding_mode") != mode:
            raise RuntimeError("CRA/Core event_grounding_mode mismatch in forward.")
        if time_param.get("qbound_event_grounding_mode") != mode:
            raise RuntimeError("CRA/Refine event_grounding_mode not forwarded.")
        enabled = mode == "event_relative"
        if self._scalar_flag(qbound_aux, "event_relative_enabled_flag") != float(enabled):
            raise RuntimeError("CRA/Core V3.7 event activation flag mismatch.")
        if self._scalar_flag(time_param, "qbound_event_relative_enabled_flag") != float(enabled):
            raise RuntimeError("CRA/Refine V3.7 event activation flag mismatch.")
        for key in ("event_residual_scale", "event_relation_temperature"):
            actual = self._scalar_flag(qbound_aux, key + "_used")
            if not math.isclose(
                actual, float(self.qbound_event_kwargs[key]),
                rel_tol=1e-5, abs_tol=1e-6,
            ):
                raise RuntimeError(f"CRA/Core V3.7 diagnostic mismatch: {key}.")
        if not enabled:
            if "event_candidate_prob" in qbound_aux:
                raise RuntimeError("V3.7 legacy mode unexpectedly computed event candidates.")
            return

        video_mask = qbound_aux["video_mask"].bool()
        K = qbound_aux["event_candidate_starts"].numel()
        if K < 1:
            raise RuntimeError("V3.7 candidate bank cannot be empty.")
        max_length = self.qbound_event_kwargs["event_candidate_max_length"]
        valid_lengths = torch.arange(1, time_steps + 1, device=video_mask.device)
        if max_length is not None:
            valid_lengths = valid_lengths[:max_length]
        expected_K = int((time_steps - valid_lengths + 1).sum().item())
        if K != expected_K:
            raise RuntimeError(
                f"CRA expected {expected_K} candidate intervals, got {K}."
            )
        required = {
            "legacy_semantic_seed_logits": (batch_size, time_steps),
            "legacy_semantic_seed_prob": (batch_size, time_steps),
            "event_candidate_scores": (batch_size, K),
            "event_candidate_prob": (batch_size, K),
            "event_candidate_valid": (batch_size, K),
            "event_candidate_frame_weights": (K, time_steps),
            "event_frame_prob": (batch_size, time_steps),
            "event_geometry_frame_prior": (batch_size, time_steps),
            "event_seed_residual_logits": (batch_size, time_steps),
            "event_residual_rms": (batch_size,),
            "event_residual_to_legacy_std_ratio": (batch_size,),
        }
        for key, shape in required.items():
            if key not in qbound_aux:
                raise KeyError(f"CRA V3.7 core output missing {key!r}.")
            self._assert_shape(key, qbound_aux[key], shape)
            if key != "event_candidate_valid":
                self._assert_finite(key, qbound_aux[key])
            if key != "event_candidate_valid":
                prefixed = "qbound_" + key
                if prefixed not in time_param:
                    raise KeyError(f"CRA V3.7 time_param missing {prefixed!r}.")
                if not torch.equal(time_param[prefixed], qbound_aux[key]):
                    raise RuntimeError(f"V3.7 output forwarding mismatch: {prefixed}.")
        valid = qbound_aux["event_candidate_valid"]
        if valid.dtype != torch.bool:
            raise TypeError("V3.7 event_candidate_valid must be bool.")
        prob = qbound_aux["event_candidate_prob"]
        if (prob.detach() * (~valid)).abs().amax() > 1e-5:
            raise RuntimeError("V3.7 event scores assigned mass to invalid spans.")
        if not torch.allclose(
            prob.sum(dim=-1), torch.ones(batch_size, device=prob.device, dtype=prob.dtype),
            atol=1e-4, rtol=1e-4,
        ):
            raise RuntimeError("V3.7 event_candidate_prob must sum to one.")
        for key in (
            "legacy_semantic_seed_prob", "event_frame_prob",
            "event_geometry_frame_prior",
        ):
            self._assert_distribution(key, qbound_aux[key], video_mask)
        reconstructed = (
            qbound_aux["legacy_semantic_seed_logits"]
            + qbound_aux["event_seed_residual_logits"]
        )
        if not torch.allclose(
            reconstructed.masked_fill(~video_mask, 0.0),
            qbound_aux["semantic_seed_logits"].masked_fill(~video_mask, 0.0),
            atol=1e-5, rtol=1e-5,
        ):
            raise RuntimeError("CRA V3.7 Seed fusion differs from legacy + event residual.")
        # No gradient detachment across the new WHERE branch before support.
        if self.training and torch.is_grad_enabled() and not (
            qbound_aux["event_candidate_prob"].requires_grad
            and qbound_aux["support_temporal_prob"].requires_grad
        ):
            raise RuntimeError("V3.7 event-to-Support gradient path was detached.")

    # ------------------------------------------------------------------
    # V3.5 / QDTG optional same-video question-pair interface
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_paired_input(
        paired_question,
        pair_valid_mask,
        primary_question,
        batch_size,
        device,
    ):
        """Validate the interface, not the semantic validity of a pair.

        The sampler (NOT this wrapper) must verify that Q1 and Q2 belong to
        the same video and do not require the same temporal evidence.
        """
        if paired_question is None:
            if pair_valid_mask is not None:
                raise ValueError(
                    "pair_valid_mask requires paired_question token ids."
                )
            return None

        if not torch.is_tensor(paired_question):
            raise TypeError("paired_question must be a token id tensor.")
        if paired_question.ndim == 3 and paired_question.size(1) == 1:
            paired_question = paired_question.squeeze(1)
        if paired_question.ndim != 2 or paired_question.size(0) != batch_size:
            raise ValueError(
                "paired_question must have shape [B,L2] or [B,1,L2], "
                "using the same batch/video order as Q1."
            )
        if paired_question.size(1) < 1:
            raise ValueError("paired_question must contain tokens.")
        if paired_question.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64
        ):
            raise TypeError("paired_question must contain integer token ids.")
        if not torch.is_tensor(pair_valid_mask):
            raise TypeError(
                "V3.5 requires pair_valid_mask: bool [B] alongside Q2 "
                "so unmatched rows cannot silently enter the loss."
            )
        if pair_valid_mask.dtype != torch.bool or pair_valid_mask.shape != (batch_size,):
            raise ValueError("pair_valid_mask must have dtype bool and shape [B].")
        pair_valid_mask = pair_valid_mask.to(device=device, non_blocking=True)

        # One cheap fail-fast check. It does NOT certify GT disjointness.
        q1 = primary_question
        if q1 is not None:
            if q1.ndim == 3 and q1.size(1) == 1:
                q1 = q1.squeeze(1)
            if q1.ndim == 2 and q1.shape == paired_question.shape:
                matches = q1.to(device=device).eq(paired_question.to(device=device)).all(dim=-1)
                if (matches & pair_valid_mask).any():
                    raise ValueError(
                        "A V3.5 valid pair cannot use identical Q1/Q2 token ids. "
                        "Mark placeholder rows invalid with pair_valid_mask=False."
                    )
        return pair_valid_mask

    # ------------------------------------------------------------------
    # Q-Bound V2.8 wrapper/integration validation
    # ------------------------------------------------------------------
    def _validate_v280_contract(
        self,
        time_param,
        qbound_aux,
        video_proj,
        batch_size,
        time_steps,
    ):
        if not isinstance(time_param, dict):
            raise TypeError(
                "RefineDecoder must return time_param as a dict."
            )
        if not isinstance(qbound_aux, dict):
            raise TypeError(
                "Q-Bound V2.8 must expose "
                "time_param['qbound_aux'] as a dict."
            )

        required_aux = {
            "qbound_version",
            # Core causal architecture.
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
            # Inputs / representations.
            "video_mask",
            "question_mask",
            "value_tokens",
            "visual_evidence_states",
            "where_context_states",
            "question_tokens_projected",
            "relevance_question_context",
            "relevance_question_attention",
            "semantic_seed_logits",
            "semantic_seed_prob",
            # Video-only extent branch.
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
            # Propagation / interval views.
            "support_temporal_prob",
            "support_entropy_ratio",
            "support_effective_frames",
            "support_propagation_l1_delta",
            "semantic_seed_support_cosine",
            "start_boundary_prob",
            "end_boundary_prob",
            "segment_valid_mask",
            "segment_prob",
            "main_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "selected_span_frame_mask",
            "selected_span_frame_weights",
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
            # Canonical core aliases.
            "qbound_support_predictive_loss",
            "qbound_support_edge_affinity",
            "qbound_support_graph_laplacian",
            "qbound_support_temporal_prob",
            "qbound_main_temporal_prob",
            "qbound_soft_temporal_prob",
            "qbound_training_temporal_prob",
            "qbound_hard_temporal_prob",
            "qbound_segment_prob",
            # RefineDecoder gradient views.
            "qbound_key",
            "qbound_qa_key",
            "qbound_grounding_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
            "qbound_value_sequence",
            "qbound_qa_pooled_value",
            "qbound_grounding_pooled_value",
            "qbound_qa_video_proj",
            "qbound_grounding_video_proj",
            "qbound_hard_video_proj",
            "qbound_soft_video_proj",
            "qbound_training_video_proj",
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
            # RefineDecoder provenance.
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_same_support_for_qa_and_grounding_enabled",
            "qbound_qa_support_detached_enabled",
            "qbound_grounding_gradient_to_semantic_seed_enabled",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_grounding_gradient_to_support_graph_enabled",
            "qbound_predictive_loss_exposed",
            "qbound_soft_grounding_feature_exposed",
            "qbound_value_uses_question_conditioned_states",
            "qbound_value_uses_raw_visual_tokens",
            "functional_critic_used_in_decoder",
            "answer_features_used_in_temporal_inference",
            "qbound_visual_enrichment_used_for_where",
            "qbound_visual_enrichment_used_for_value",
            "qbound_question_conditions_semantic_relevance",
            "qbound_question_routes_visual_context",
            "qbound_question_used_as_visual_value",
            # CRA reference / downstream views.
            "cra_key",
            "cra_video_proj",
            "main_video_proj",
            "qbound_video_proj",
        }
        missing_aux = sorted(required_aux.difference(qbound_aux))
        if missing_aux:
            raise KeyError(
                "Q-Bound V2.8 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_time = {
            "grounding_feature",
            "qbound_grounding_feature",
            "qbound_temporal_main_feature",
            "qbound_feature",
            "qbound_qa_feature",
            "cra_grounding_feature",
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
            "qbound_support_predictive_loss",
            "support_predictive_loss",
            "qbound_selected_span_start_frame",
            "qbound_selected_span_end_frame",
            "qbound_selected_span_mask",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",
            "qbound_barycenter_start_frame",
            "qbound_barycenter_end_frame",
            "qbound_soft_expected_span_length",
            "qbound_soft_expected_span_fraction",
            "qbound_smoothing_applied",
            "qbound_structured_visual_evidence_enabled",
            "qbound_relevance_extent_decoupling_enabled",
            "qbound_video_only_support_branch_enabled",
            "qbound_predictive_support_graph_enabled",
            "qbound_structural_seed_propagation_enabled",
            "qbound_same_support_for_training_and_inference_enabled",
            "qbound_hard_map_temporal_decision_enabled",
        }
        missing_time = sorted(required_time.difference(time_param))
        if missing_time:
            raise KeyError(
                "RefineDecoder/Q-Bound V2.8 time_param output is missing: "
                + ", ".join(missing_time)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "CRA/RefineDecoder/Q-Bound V2.8 version mismatch: "
                f"CRA={self.qbound_version!r}, "
                f"decoder_aux={qbound_aux['qbound_version']!r}."
            )

        self._validate_eqptc_contract(
            time_param, qbound_aux, batch_size, time_steps
        )
        self._validate_event_relative_contract(
            time_param, qbound_aux, batch_size, time_steps
        )

        predictor_enabled = bool(getattr(self.decoder, "qbound_support_predictor_enabled", True))
        expected_on = {
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
            "qbound_input_is_generic",
            "qbound_same_support_for_qa_and_grounding_enabled",
            "qbound_qa_support_detached_enabled",
            "qbound_grounding_gradient_to_semantic_seed_enabled",
            "qbound_predictive_loss_exposed",
            "qbound_soft_grounding_feature_exposed",
            "qbound_value_uses_raw_visual_tokens",
            "qbound_visual_enrichment_used_for_where",
            "qbound_question_conditions_semantic_relevance",
        }
        expected_off = {
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
            "qbound_input_uses_cra_embedding",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_grounding_gradient_to_support_graph_enabled",
            "qbound_value_uses_question_conditioned_states",
            "functional_critic_used_in_decoder",
            "answer_features_used_in_temporal_inference",
            "qbound_visual_enrichment_used_for_value",
            "qbound_question_routes_visual_context",
            "qbound_question_used_as_visual_value",
        }
        if not predictor_enabled:
            expected_on -= {"video_only_support_branch_enabled", "predictive_support_graph_enabled",
                            "qbound_predictive_loss_exposed"}
            expected_off |= {"video_only_support_branch_enabled", "predictive_support_graph_enabled",
                             "qbound_predictive_loss_exposed"}
        if self.qbound_event_grounding_mode == "event_relative":
            # V3.7 intentionally activates these event scoring components.
            event_flags = {
                "question_direct_span_scoring_enabled",
                "neural_span_scorer_enabled",
                "event_candidate_selector_enabled",
            }
            expected_off -= event_flags
            expected_on |= event_flags
        for name in expected_on:
            actual = self._scalar_flag(qbound_aux, name)
            if actual != 1.0:
                raise RuntimeError(
                    f"V2.8 contract violation: {name}={actual}, expected 1."
                )
        for name in expected_off:
            actual = self._scalar_flag(qbound_aux, name)
            if actual != 0.0:
                raise RuntimeError(
                    f"V2.8 contract violation: {name}={actual}, expected 0."
                )

        video_mask = qbound_aux["video_mask"].bool()
        self._assert_shape(
            "video_mask",
            video_mask,
            (batch_size, time_steps),
        )
        if not torch.all(video_mask.any(dim=-1)):
            raise RuntimeError(
                "Every V2.8 sample must contain a valid video frame."
            )

        distribution_names = (
            "semantic_seed_prob",
            "support_temporal_prob",
            "main_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
            "qbound_support_temporal_prob",
            "qbound_main_temporal_prob",
            "qbound_soft_temporal_prob",
            "qbound_training_temporal_prob",
            "qbound_hard_temporal_prob",
            "qbound_key",
            "qbound_qa_key",
            "qbound_grounding_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
        )
        for name in distribution_names:
            self._assert_distribution(
                name,
                qbound_aux[name],
                video_mask,
            )

        support_reference = qbound_aux[
            "support_temporal_prob"
        ].detach()
        for name in distribution_names[2:]:
            if not torch.allclose(
                qbound_aux[name].detach(),
                support_reference,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError(
                    f"{name} diverges from the V2.8 Support Field."
                )

        for name in (
            "qbound_key",
            "qbound_qa_key",
            "qbound_main_temporal_prob",
            "qbound_hard_temporal_prob",
            "final_key",
        ):
            if qbound_aux[name].requires_grad:
                raise RuntimeError(
                    f"{name} must be detached from downstream VQA."
                )

        for name in (
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
        ):
            value = qbound_aux[name]
            self._assert_finite(name, value)
            if float(value.detach().max().item()) > 1e-5:
                raise RuntimeError(
                    f"{name} shows a train/QA/grounding support mismatch."
                )

        edge_steps = max(time_steps - 1, 0)
        for name in (
            "support_edge_mask",
            "support_edge_surprise",
            "support_edge_affinity_raw",
            "support_edge_affinity",
            "qbound_support_edge_affinity",
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
            "qbound_support_graph_laplacian",
        ):
            self._assert_shape(
                name,
                qbound_aux[name],
                (batch_size, time_steps, time_steps),
            )
            self._assert_finite(name, qbound_aux[name])

        if qbound_aux["support_edge_affinity"].requires_grad:
            raise RuntimeError(
                "Grounding support_edge_affinity must be detached."
            )
        if qbound_aux["support_graph_laplacian"].requires_grad:
            raise RuntimeError(
                "Grounding support_graph_laplacian must be detached."
            )

        if not predictor_enabled and (
            qbound_aux["support_predictive_loss"].requires_grad
            or float(qbound_aux["support_predictive_loss"].detach().item()) != 0.0
        ):
            raise RuntimeError("R2-C must not compute a predictive objective.")
        for name in (
            "support_predictive_loss",
            "qbound_support_predictive_loss",
        ):
            value = qbound_aux[name]
            if not torch.is_tensor(value) or value.numel() != 1:
                raise TypeError(f"{name} must be a scalar tensor.")
            self._assert_finite(name, value)
        if not torch.allclose(
            qbound_aux["support_predictive_loss"].detach(),
            qbound_aux["qbound_support_predictive_loss"].detach(),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise RuntimeError(
                "Predictive-loss aliases do not reference the same objective."
            )
        if not torch.allclose(
            time_param["qbound_support_predictive_loss"].detach(),
            qbound_aux["support_predictive_loss"].detach(),
            atol=1e-7,
            rtol=1e-7,
        ):
            raise RuntimeError(
                "time_param does not expose support_predictive_loss correctly."
            )

        self._assert_shape(
            "segment_prob",
            qbound_aux["segment_prob"],
            (batch_size, time_steps, time_steps),
        )
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

        start = qbound_aux["selected_span_start_frame"]
        end = qbound_aux["selected_span_end_frame"]
        self._assert_shape(
            "selected_span_start_frame",
            start,
            (batch_size,),
        )
        self._assert_shape(
            "selected_span_end_frame",
            end,
            (batch_size,),
        )
        if torch.any(start > end):
            raise RuntimeError(
                "V2.8 barycentric span view has start > end."
            )

        question_mask = qbound_aux["question_mask"]
        if question_mask.ndim != 2 or question_mask.size(0) != batch_size:
            raise ValueError("question_mask must be [B,L].")
        q_len = question_mask.size(1)
        question_attention = qbound_aux["relevance_question_attention"]
        if (
            question_attention.ndim != 4
            or question_attention.size(0) != batch_size
            or question_attention.size(2) != time_steps
            or question_attention.size(3) != q_len
        ):
            raise ValueError(
                "relevance_question_attention must be [B,H,T,L], got "
                f"{tuple(question_attention.shape)}."
            )

        value_sequence = qbound_aux["qbound_value_sequence"]
        if (
            value_sequence.ndim != 3
            or value_sequence.shape[:2] != (batch_size, time_steps)
        ):
            raise ValueError(
                "qbound_value_sequence must be [B,T,D], got "
                f"{tuple(value_sequence.shape)}."
            )

        feature_names = (
            "qbound_qa_video_proj",
            "qbound_grounding_video_proj",
            "qbound_hard_video_proj",
            "qbound_soft_video_proj",
            "qbound_training_video_proj",
            "main_video_proj",
            "qbound_video_proj",
            "cra_video_proj",
        )
        for name in feature_names:
            value = qbound_aux[name]
            if value.shape != video_proj.shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match "
                    f"returned QA feature {tuple(video_proj.shape)}."
                )
            self._assert_finite(name, value)

        if not torch.allclose(
            qbound_aux["qbound_qa_video_proj"].detach(),
            video_proj.detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "Returned CRA video feature must be the detached-support "
                "V2.8 QA view."
            )
        if not torch.allclose(
            qbound_aux["qbound_qa_video_proj"].detach(),
            qbound_aux["qbound_grounding_video_proj"].detach(),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise RuntimeError(
                "QA and grounding features must pool the same Support Field."
            )
        if not torch.allclose(
            time_param["qbound_qa_feature"].detach(),
            video_proj.detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "time_param.qbound_qa_feature does not match returned QA view."
            )
        if not torch.allclose(
            time_param["grounding_feature"].detach(),
            qbound_aux["qbound_grounding_video_proj"].detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "time_param.grounding_feature must be the differentiable "
                "V2.8 grounding view."
            )

    # ------------------------------------------------------------------
    # Q-Bound V2.8 refine forward
    # ------------------------------------------------------------------
    def refine_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
        question=None,
        qbound_mix=1.0,
        answer_only=None,
        paired_question=None,
        pair_valid_mask=None,
    ):
        """
        CRA/Q-Bound R2-C forward with optional TRAIN-ONLY V3.5 question pair.

        answer
            Existing CRA [Question SEP Answer_i] candidates. They are used only
            by the downstream multiple-choice answer branch and detached CRA
            reference grounding.

        question
            Question-only token ids. These are the only language features sent
            to Q-Bound semantic relevance.

        answer_only
            Accepted for repository signature compatibility and never consumed
            by V2.8.

        qbound_mix
            Accepted for compatibility. R2-C never mixes CRA and Q-Bound keys.

        paired_question
            Optional Q2 QUESTION-ONLY token ids [B,L2] for V3.5 training.
            Q2 must refer to the same video as the corresponding Q1 row.
            Never provide Q2 during evaluation/inference.

        pair_valid_mask
            Required if Q2 is provided; boolean [B]. Rows marked False
            are excluded by the future pair loss in cra_pipeline.py.
            False rows must still have a nonempty placeholder token sequence.
        """
        del qbound_mix, answer_only

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(video_mask, device)

        paired_requested = paired_question is not None
        if paired_requested:
            if not self.training:
                raise RuntimeError(
                    "V3.5 paired_question is TRAIN-ONLY; do not provide Q2 "
                    "during validation or inference."
                )
            if mode != "VQA" or not self.qbound_enabled:
                raise RuntimeError(
                    "V3.5 Q2 is only supported with active Q-Bound and mode='VQA'."
                )
            if question is None:
                raise ValueError("V3.5 Q2 requires primary question=Q1.")
        pair_valid_mask = self._validate_paired_input(
            paired_question,
            pair_valid_mask,
            question,
            batch_size=video.size(0),
            device=device,
        )

        # Ordinary CRA candidate-answer representation.
        answer_g, _ = self._encode_answers(answer, device)
        batch_size = answer_g.size(0)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            batch_size,
            answer_g.device,
        )

        # CRA visual tokens remain the downstream/reference branch.
        video_f = self.video_model(video)

        qsn_gt = None
        if mode == "VQ" and answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
            qsn_gt = qsn_g
        else:
            # Candidate aggregate is used only by the detached CRA reference.
            qsn_g = answer_g.max(dim=1).values
            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(
                    answer_g,
                    answer_indices,
                )

        use_qbound = bool(self.qbound_enabled and mode == "VQA")

        question_g = None
        question_tokens = None
        question_mask = None
        if use_qbound:
            if question is None:
                raise ValueError(
                    "Q-Bound V2.8 requires question-only token ids. "
                    "Pass question=... through CRA.forward()."
                )

            (
                question_g,
                question_tokens,
                question_mask,
            ) = self._encode_question(question, device)

            if question_g.size(0) != batch_size:
                raise ValueError(
                    "Question and answer batch sizes do not match: "
                    f"{question_g.size(0)} vs {batch_size}."
                )
            if (
                self.qbound_require_token_question
                and (
                    question_tokens is None
                    or question_tokens.ndim != 3
                )
            ):
                raise RuntimeError(
                    "V2.8 requires token-level question features."
                )

        # V3.5: encode Q2 with the SAME question-only LanModel but stop its
        # gradient at the external encoder firewall. Isolate RNG so the extra
        # language dropout does not perturb Q1's canonical decoder/core path.
        pair_question_g = None
        pair_question_tokens = None
        pair_question_mask = None
        if paired_requested:
            rng_devices = (
                [device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda" else []
            )
            with torch.random.fork_rng(devices=rng_devices, enabled=True):
                with torch.no_grad():
                    pair_question_g, pair_question_tokens, pair_question_mask = (
                        self._encode_question(paired_question, device)
                    )
            if pair_question_g.size(0) != batch_size:
                raise ValueError("Paired question and video batch sizes differ.")
            if pair_question_g.size(-1) != question_g.size(-1):
                raise ValueError("Q1/Q2 question-global embedding dimensions differ.")
            if pair_question_tokens.size(-1) != question_tokens.size(-1):
                raise ValueError("Q1/Q2 question-token embedding dimensions differ.")

        # R2-C Q1 inference path stays unchanged. Optional Q2 path computes only
        # shared-encoder Seed and Support inside the patched RefineDecoder.
        pair_decoder_kwargs = (
            {
                "paired_question_feature": pair_question_tokens,
                "paired_question_mask": pair_question_mask,
                "pair_valid_mask": pair_valid_mask,
            }
            if paired_requested else {}
        )
        (
            video_proj,
            video_attn,
            time_param,
        ) = self.decoder(
            video_f,
            qsn_g,
            video_mask,
            gsub=gsub,
            do=self.cfgs["model"].get("do", False),
            window_sizes=self.cfgs["model"].get(
                "window_sizes",
                [1, 3, 5],
            ),
            use_qbound=use_qbound,
            question_feature=(
                question_tokens if use_qbound else None
            ),
            qbound_video_tokens=(
                video if use_qbound else None
            ),
            qbound_mix=0.0,
            question_mask=(
                question_mask if use_qbound else None
            ),
            **pair_decoder_kwargs,
        )

        answer_proj = answer_g
        qbound_aux = (
            time_param.get("qbound_aux")
            if isinstance(time_param, dict)
            else None
        )

        if use_qbound:
            self._validate_v280_contract(
                time_param=time_param,
                qbound_aux=qbound_aux,
                video_proj=video_proj,
                batch_size=batch_size,
                time_steps=video.shape[1],
            )

            dtype = video_proj.dtype
            device = video_proj.device
            one = torch.ones((), dtype=dtype, device=device)
            zero = torch.zeros((), dtype=dtype, device=device)

            wrapper_flags = {
                "grounding_requires_answer_at_inference": zero,
                "temporal_gt_used_in_training": zero,
                "answer_candidates_used_in_qbound_temporal_path": zero,
                "answer_only_used_in_v28": zero,
                "visual_enrichment_only_affects_where": one,
                "question_conditions_relevance_only": one,
                "video_only_extent_structure": one,
                "shared_support_for_qa_and_grounding": one,
                "qa_support_detached": one,
                "posterior_barycentric_span_is_metadata_only": one,
                "functional_critic_enabled": zero,
                "vqa_gradient_to_qbound_topology_enabled": zero,
                "question_grounding_objective_expected": one,
                "support_predictive_objective_expected": (
                    one if getattr(self.decoder, "qbound_support_predictor_enabled", True)
                    else zero
                ),
                # Stop-gradient target prevents the alignment objective from
                # being solved by moving the question encoder.
                "grounding_question_stop_gradient": one,
            }
            qbound_aux.update(wrapper_flags)
            for key, value in wrapper_flags.items():
                time_param[f"qbound_{key}"] = value

        # Shared outputs for cra_pipeline.py and evaluation.
        outs = {
            "fatt": video_attn,
            "time_param": time_param,
            # Existing CRA answer/alignment state.
            "align_gt": qsn_gt,
            "qa_condition": qsn_g,
            "answer_proj": answer_proj,
            # Question-only state.
            "question_condition": question_g,
            "question_tokens": question_tokens,
            "question_mask": question_mask,
            "question_only_conditioning": torch.as_tensor(
                question_g is not None,
                device=video_proj.device,
                dtype=video_proj.dtype,
            ),
            "qbound_align_question": question_g,
            "qbound_align_qa": qsn_gt,
            "qbound_aux": qbound_aux,
            "qbound_active": use_qbound,
            "qbound_version": self.qbound_version,
            "qbound_language_pooling_mode": self.qbound_language_pooling_mode,
            "gpeb_active": use_qbound,
            # Historical answer-only temporal branch remains absent.
            "answer_only_proj": None,
        }

        if paired_requested:
            if not isinstance(qbound_aux, dict):
                raise RuntimeError("V3.5 decoder did not return Q-Bound outputs.")
            for name in (
                "qbound_pair_valid_mask",
                "qbound_pair_support_temporal_prob",
                "qbound_pair_semantic_seed_prob",
                "qbound_pair_semantic_seed_logits",
            ):
                if name not in qbound_aux:
                    raise KeyError(f"V3.5 RefineDecoder missing {name!r}.")
            returned_valid = qbound_aux["qbound_pair_valid_mask"]
            if not torch.equal(returned_valid, pair_valid_mask):
                raise RuntimeError("V3.5 pair_valid_mask changed in decoder.")
            if self.qbound_event_grounding_mode == "event_relative":
                pair_core = qbound_aux.get("qbound_pair_aux")
                if not isinstance(pair_core, dict):
                    raise RuntimeError("V3.7 Q2 event diagnostics are missing.")
                k_candidates = qbound_aux["event_candidate_starts"].numel()
                for name, shape in (
                    ("event_candidate_scores", (batch_size, k_candidates)),
                    ("event_candidate_prob", (batch_size, k_candidates)),
                    ("event_frame_prob", (batch_size, video.size(1))),
                    ("event_seed_residual_logits", (batch_size, video.size(1))),
                ):
                    if name not in pair_core:
                        raise KeyError(f"V3.7 Q2 output missing {name!r}.")
                    self._assert_shape("Q2 " + name, pair_core[name], shape)
                    self._assert_finite("Q2 " + name, pair_core[name])
                    alias = "qbound_pair_" + name
                    if alias not in qbound_aux or not torch.equal(
                        qbound_aux[alias], pair_core[name]
                    ):
                        raise RuntimeError(f"V3.7 Q2 alias mismatch: {alias}.")
                prob2 = pair_core["event_candidate_prob"]
                if not torch.allclose(
                    prob2.sum(dim=-1), torch.ones_like(prob2.sum(dim=-1)),
                    atol=1e-4, rtol=1e-4,
                ):
                    raise RuntimeError("V3.7 Q2 candidate probabilities must sum to 1.")
                if (prob2.detach() * ~qbound_aux["event_candidate_valid"]).abs().amax() > 1e-5:
                    raise RuntimeError("V3.7 Q2 selected invalid candidate spans.")
            # Question-global targets are detached like Q1 L_ground target.
            # Neither Q2's QA labels nor its answer candidates are consumed.
            outs["qbound_pair_question"] = pair_question_g.detach()
            outs["qbound_pair_grounding_question"] = pair_question_g.detach()
            outs["qbound_pair_valid_mask"] = returned_valid
            outs["qbound_pair_question_mask"] = pair_question_mask
            outs["qbound_pair_support_temporal_prob"] = qbound_aux[
                "qbound_pair_support_temporal_prob"
            ]
            outs["qbound_pair_semantic_seed_prob"] = qbound_aux[
                "qbound_pair_semantic_seed_prob"
            ]
            outs["qbound_pair_semantic_seed_logits"] = qbound_aux[
                "qbound_pair_semantic_seed_logits"
            ]
            if outs["qbound_pair_grounding_question"].requires_grad:
                raise RuntimeError("V3.5 Q2 contrastive target must be detached.")
            if not outs["qbound_pair_support_temporal_prob"].requires_grad:
                raise RuntimeError("V3.5 Q2 Support must retain Seed gradients.")

            if self.qbound_temporal_context_mode == "eq_ptc":
                # Paired Q2 needs a SEPARATE C-pre/Bq pass. Q1 and Q2
                # share video-only visual evidence/graph, not Q1 context.
                pair_aux = qbound_aux.get("qbound_pair_aux")
                if not isinstance(pair_aux, dict):
                    raise RuntimeError("EQ-PTC Q2 must expose paired core diagnostics.")
                for name, size in (
                    ("eqptc_early_question_attention", (
                        batch_size, self.decoder.qbound.num_question_heads,
                        video.size(1), pair_question_mask.size(1),
                    )),
                    ("eqptc_temporal_attention", (
                        batch_size, self.decoder.qbound.num_question_heads,
                        video.size(1), video.size(1),
                    )),
                    ("where_context_states", (
                        batch_size, video.size(1), self.decoder.qbound.hidden_dim,
                    )),
                ):
                    if name not in pair_aux:
                        raise KeyError(f"EQ-PTC Q2 context missing {name!r}.")
                    self._assert_shape("Q2 " + name, pair_aux[name], size)
                    self._assert_finite("Q2 " + name, pair_aux[name])
                if not torch.equal(pair_aux["video_mask"], qbound_aux["video_mask"]):
                    raise RuntimeError("EQ-PTC Q1/Q2 must share the same video mask.")

        if use_qbound:
            # Grounding target is stopped; video evidence remains
            # differentiable to semantic relevance, not to the support graph.
            grounding_question = question_g.detach()
            outs.update(
                {
                    "qbound_grounding_question": grounding_question,
                    "qbound_grounding_video_proj": qbound_aux[
                        "qbound_grounding_video_proj"
                    ],
                    "qbound_grounding_feature": qbound_aux[
                        "qbound_grounding_video_proj"
                    ],
                    "qbound_grounding_pooled_value": qbound_aux[
                        "qbound_grounding_pooled_value"
                    ],
                    "qbound_grounding_key": qbound_aux[
                        "qbound_grounding_key"
                    ],
                    "qbound_vqa_video_proj": qbound_aux[
                        "qbound_qa_video_proj"
                    ],
                    "qbound_vqa_key": qbound_aux["qbound_qa_key"],
                    "qbound_support_predictive_loss": qbound_aux[
                        "support_predictive_loss"
                    ],
                    "support_predictive_loss": qbound_aux[
                        "support_predictive_loss"
                    ],
                }
            )

        if isinstance(time_param, dict):
            for key, value in time_param.items():
                if key.startswith("qbound_"):
                    outs[key] = value

        if isinstance(qbound_aux, dict):
            # Expose every diagnostic as a reference, not a tensor copy. This
            # keeps logging/ablation code independent of RefineDecoder prefixing.
            for key, value in qbound_aux.items():
                outs.setdefault(key, value)

            # Canonical downstream aliases.
            outs["qbound_video_proj"] = qbound_aux[
                "qbound_video_proj"
            ]
            outs["qbound_qa_video_proj"] = qbound_aux[
                "qbound_qa_video_proj"
            ]
            outs["qbound_hard_video_proj"] = qbound_aux[
                "qbound_hard_video_proj"
            ]
            outs["qbound_soft_video_proj"] = qbound_aux[
                "qbound_soft_video_proj"
            ]
            outs["qbound_training_video_proj"] = qbound_aux[
                "qbound_training_video_proj"
            ]
            outs["qbound_key"] = qbound_aux["qbound_key"]
            outs["qbound_qa_key"] = qbound_aux["qbound_qa_key"]
            outs["qbound_training_key"] = qbound_aux[
                "qbound_training_key"
            ]
            outs["qbound_soft_key"] = qbound_aux["qbound_soft_key"]
            outs["qbound_support_temporal_prob"] = qbound_aux[
                "support_temporal_prob"
            ]
            outs["qbound_semantic_seed_prob"] = qbound_aux[
                "semantic_seed_prob"
            ]
            outs["qbound_support_predictive_loss"] = qbound_aux[
                "support_predictive_loss"
            ]

            # Compatibility aliases now refer to the continuous Support Field
            # and its posterior-barycentric interval view, never a MAP span.
            outs["qbound_calibrated_video_proj"] = qbound_aux[
                "qbound_qa_video_proj"
            ]
            outs["qbound_calibrated_key"] = qbound_aux[
                "qbound_key"
            ]
            outs["cra_video_proj"] = qbound_aux["cra_video_proj"]
            outs["cra_key"] = qbound_aux["cra_key"]

        return video_proj, answer_proj, outs

    # ------------------------------------------------------------------
    # Original CRA baseline path
    # ------------------------------------------------------------------
    def baseline_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
        question=None,
        qbound_mix=1.0,
        answer_only=None,
    ):
        # V2.8 extras are accepted so call sites can keep one signature.
        del gsub, question, qbound_mix, answer_only

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(video_mask, device)
        answer_g, answer_tokens = self._encode_answers(answer, device)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            answer_g.size(0),
            answer_g.device,
        )

        video_f = self.video_model(video)
        if answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
        else:
            qsn_g = answer_g.max(dim=1).values

        video_proj, video_attn = self.decoder(
            video_f,
            answer_tokens,
            video_mask,
        )
        outs = {
            "fatt": video_attn,
            "qa_condition": qsn_g,
        }
        return video_proj, answer_g, outs

    # ------------------------------------------------------------------
    # Existing weak alignment objective
    # ------------------------------------------------------------------
    def align_loss(
        self,
        qsn,
        grounding_feature,
        neg_sample=None,
    ):
        def _sample_negatives(x_pos, k):
            batch_size, feature_dim = x_pos.size()
            if batch_size < 2:
                raise ValueError(
                    "InfoNCE negative sampling requires batch_size >= 2."
                )

            x_neg = torch.zeros(
                batch_size,
                k,
                feature_dim,
                device=x_pos.device,
                dtype=x_pos.dtype,
            )
            for index in range(batch_size):
                candidate_indices = torch.cat(
                    [
                        torch.arange(
                            0,
                            index,
                            device=x_pos.device,
                        ),
                        torch.arange(
                            index + 1,
                            batch_size,
                            device=x_pos.device,
                        ),
                    ]
                )
                sampled_positions = torch.randint(
                    low=0,
                    high=candidate_indices.numel(),
                    size=(k,),
                    device=x_pos.device,
                )
                x_neg[index] = x_pos[
                    candidate_indices[sampled_positions]
                ]
            return x_neg

        if neg_sample is None:
            negative_count = max(
                1,
                self.cfgs["dataset"]["batch_size"] // 2,
            )
            x_neg = _sample_negatives(
                grounding_feature,
                negative_count,
            )
        else:
            x_neg = neg_sample

        return self._info_nce_loss(
            qsn,
            grounding_feature,
            x_neg,
        )

    @staticmethod
    def _info_nce_loss(
        x,
        x_pos,
        x_neg,
        temperature=0.07,
    ):
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_neg = F.normalize(x_neg, dim=-1)

        pos_sim = torch.sum(
            x * x_pos,
            dim=-1,
            keepdim=True,
        ) / temperature
        neg_sim = torch.bmm(
            x.unsqueeze(1),
            x_neg.transpose(1, 2),
        ).squeeze(1) / temperature

        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        labels = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device,
        )
        return F.cross_entropy(logits, labels)
