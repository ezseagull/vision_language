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
    CRA wrapper used as the downstream VideoQA host/reference for
    Q-Bound V2.4.0.

    -------------------------------------------------------------------------
    V2.4 scientific contract
    -------------------------------------------------------------------------
    Temporal grounding is one-stage and strictly Video + Question:

        raw/local CLIP features + question-only token features
            -> QBoundCore V2.4
            -> Temporal Context + early Q->V interaction
            -> Evidence Core a_t + Event Continuity j_t
            -> Core-to-Closure P_Q(s,e)
            -> hard MAP temporal support for ordinary CRA VideoQA
            -> soft temporal membership for the Q-Bound grounding objective.

    CRA is deliberately NOT part of the Q-Bound temporal decision:

        1) Q-Bound receives raw/generic CLIP features, never CRA x_bar.
        2) Q-Bound receives question-only token features, never
           [Question SEP Answer] candidate features.
        3) CRA temporal grounding remains a detached reference whenever
           Q-Bound is active.
        4) The official downstream VQA feature uses the HARD Q-Bound MAP key.
           The key is detached inside RefineDecoder, so VQA cannot shape
           Q-Bound temporal topology.
        5) The differentiable SOFT Q-Bound evidence is exposed separately for
           Video-Question grounding supervision.
        6) Answer-only candidates and the V2.3 functional critic are removed.
        7) Temporal GT / IoU / IoP are never consumed here.

    This wrapper therefore preserves CRA as the controlled answer host while
    making Q-Bound independently replaceable in later versions.
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
                    model_cfg.get("gpeb_version", "2.4.0"),
                ),
            )
        )

        if self.qbound_enabled and self.qbound_version != "2.4.0":
            raise ValueError(
                "Q-Bound is enabled, but this CRA wrapper expects "
                f"model.qbound.version='2.4.0'; got {self.qbound_version!r}."
            )

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
                "Controlled Q-Bound V2.4 requires "
                "model.qbound.require_question_only=true."
            )

        self.qbound_require_token_question = bool(
            qbound_cfg.get(
                "require_token_question",
                True,
            )
        )
        if self.qbound_enabled and not self.qbound_require_token_question:
            raise ValueError(
                "Controlled Q-Bound V2.4 requires token-level question "
                "conditioning. Set model.qbound.require_token_question=true."
            )

        if self.qbound_enabled and bool(model_cfg.get("do", False)):
            raise ValueError(
                "Controlled Q-Bound V2.4 requires model.do=false. "
                "VisualIntervention would confound the temporal-grounding "
                "experiment."
            )

        # Compatibility aliases retained for the repository.
        self.gpeb_enabled = self.qbound_enabled
        self.gpeb_version = self.qbound_version
        self.gpeb_require_question_only = (
            self.qbound_require_question_only
        )

        d_model = model_cfg["d_model"]

        self.lang_model = LanModel(
            tokenizer,
            lan=model_cfg["lan"],
            out_dim=d_model,
        )
        self.video_model = VideoEmbedding(cfgs)

        # Kept only for repository/checkpoint compatibility.
        self.do = LinguisticInterventionGraph(cfgs)

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
                    getattr(
                        self.decoder,
                        "qbound_version",
                        "",
                    )
                )
                if decoder_version != self.qbound_version:
                    raise RuntimeError(
                        "CRA/RefineDecoder Q-Bound version mismatch: "
                        f"CRA={self.qbound_version!r}, "
                        f"decoder={decoder_version!r}."
                    )

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

        return candidate_features[
            batch_indices,
            candidate_indices,
        ]

    def _encode_candidates(
        self,
        tokens,
        device: torch.device,
        name: str,
    ):
        if tokens is None:
            return None, None

        tokens = tokens.to(
            device=device,
            non_blocking=True,
        )

        if tokens.ndim != 3:
            raise ValueError(
                f"{name} tokens must have shape [B,C,L], got "
                f"{tuple(tokens.shape)}."
            )

        if hasattr(
            self.lang_model,
            "encode_candidates",
        ):
            candidate_global, candidate_tokens = (
                self.lang_model.encode_candidates(
                    tokens
                )
            )
        else:
            candidate_global, candidate_tokens = (
                self.lang_model(tokens)
            )

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
        Existing CRA candidates:
            [Question SEP Answer_i]

        They remain part of the ordinary downstream CRA VideoQA host only.
        """
        if answer is None:
            raise ValueError(
                "answer tokens must not be None."
            )

        return self._encode_candidates(
            answer,
            device=device,
            name="answer",
        )

    def _question_mask_from_padded_ids(
        self,
        question_ids: torch.Tensor,
    ):
        """
        Build a question mask from the tokenizer's actual pad id.
        """
        if question_ids.ndim != 2:
            raise ValueError(
                "question_ids must be [B,L]."
            )

        pad_token_id = getattr(
            self.tokenizer,
            "pad_token_id",
            None,
        )
        if pad_token_id is None:
            raise ValueError(
                "Tokenizer must define pad_token_id for Q-Bound V2.4."
            )

        mask = question_ids.ne(
            int(pad_token_id)
        )

        # Preserve a leading special token if a tokenizer family uses id=0
        # both as a supplied leading special token and as padding.
        if int(pad_token_id) == 0 and mask.size(1) > 0:
            mask[:, 0] = True

        if not torch.all(
            mask.any(dim=-1)
        ):
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

        question = question.to(
            device=device,
            non_blocking=True,
        )

        if (
            question.ndim == 3
            and question.size(1) == 1
        ):
            question = question.squeeze(1)

        if question.ndim != 2:
            raise ValueError(
                "Question-only tokens must have shape [B,L] or [B,1,L], got "
                f"{tuple(question.shape)}."
            )

        question_mask = (
            self._question_mask_from_padded_ids(
                question
            )
        )

        if hasattr(
            self.lang_model,
            "encode_question",
        ):
            question_global, question_tokens = (
                self.lang_model.encode_question(
                    question
                )
            )
        else:
            question_global, question_tokens = (
                self.lang_model(question)
            )

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

        if (
            question_tokens.shape[:2]
            != question_mask.shape
        ):
            raise ValueError(
                "Question token features and question mask do not align."
            )

        return (
            question_global,
            question_tokens,
            question_mask,
        )

    @staticmethod
    def _move_video_mask(
        video_mask,
        device: torch.device,
    ):
        if video_mask is None:
            return None

        return video_mask.to(
            device=device,
            non_blocking=True,
        )

    @staticmethod
    def _scalar_flag(
        mapping,
        name,
    ):
        if name not in mapping:
            raise KeyError(
                f"Missing Q-Bound flag {name!r}."
            )

        value = mapping[name]

        if not torch.is_tensor(value):
            raise TypeError(
                f"Q-Bound flag {name!r} must be a tensor."
            )

        if value.numel() != 1:
            raise ValueError(
                f"Q-Bound flag {name!r} must be scalar."
            )

        return float(
            value.detach().item()
        )

    # ------------------------------------------------------------------
    # Q-Bound V2.4 contract validation
    # ------------------------------------------------------------------
    def _validate_v240_contract(
        self,
        time_param,
        qbound_aux,
        video_proj,
        batch_size,
        time_steps,
    ):
        if not isinstance(
            time_param,
            dict,
        ):
            raise TypeError(
                "RefineDecoder must return time_param as a dict."
            )

        if not isinstance(
            qbound_aux,
            dict,
        ):
            raise TypeError(
                "Q-Bound V2.4 must expose "
                "time_param['qbound_aux'] as a dict."
            )

        required_aux = {
            "qbound_version",

            # Architecture.
            "core_to_closure_enabled",
            "temporal_context_enabled",
            "early_question_conditioning_enabled",
            "evidence_core_enabled",
            "event_continuity_enabled",
            "same_interval_model_for_hard_and_soft",
            "value_where_separation_enabled",

            # Explicitly removed mechanisms.
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

            # Core / continuity diagnostics.
            "evidence_core",
            "evidence_core_prob",
            "evidence_core_entropy_ratio",
            "evidence_core_top1_prob",
            "event_continuity",
            "continuity_mask",
            "segment_energy",
            "segment_prob",
            "segment_entropy_ratio",
            "segment_top1_prob",
            "single_frame_prob_mass",
            "short_span_prob_mass",

            # Final shared interval outputs.
            "main_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "selected_span_frame_mask",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",

            # Final keys / visual evidence.
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
            "qbound_value_sequence",
            "qbound_hard_pooled_value",
            "qbound_soft_pooled_value",
            "qbound_training_pooled_value",
            "qbound_hard_video_proj",
            "qbound_soft_video_proj",
            "qbound_training_video_proj",
            "qbound_soft_training_key_max_abs_diff",

            # Provenance / anti-shortcut.
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_value_uses_question_conditioned_states",
            "qbound_value_uses_raw_visual_tokens",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_soft_grounding_feature_exposed",
            "functional_critic_used_in_decoder",
            "answer_features_used_in_temporal_inference",

            # CRA reference.
            "cra_key",
            "cra_video_proj",
            "main_video_proj",
            "qbound_video_proj",
        }

        missing_aux = sorted(
            required_aux.difference(
                qbound_aux
            )
        )
        if missing_aux:
            raise KeyError(
                "Q-Bound V2.4 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_time = {
            "grounding_feature",
            "qbound_grounding_feature",
            "qbound_temporal_main_feature",

            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",

            "qbound_selected_span_start_frame",
            "qbound_selected_span_end_frame",
            "qbound_selected_span_mask",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",

            "qbound_soft_expected_span_length",
            "qbound_soft_expected_span_fraction",
            "qbound_smoothing_applied",
        }

        missing_time = sorted(
            required_time.difference(
                time_param
            )
        )
        if missing_time:
            raise KeyError(
                "RefineDecoder/Q-Bound V2.4 time_param output is missing: "
                + ", ".join(missing_time)
            )

        if str(
            qbound_aux["qbound_version"]
        ) != self.qbound_version:
            raise RuntimeError(
                "CRA/RefineDecoder/Q-Bound V2.4 version mismatch: "
                f"CRA={self.qbound_version!r}, "
                f"decoder_aux={qbound_aux['qbound_version']!r}."
            )

        expected_on = {
            "core_to_closure_enabled",
            "temporal_context_enabled",
            "early_question_conditioning_enabled",
            "evidence_core_enabled",
            "event_continuity_enabled",
            "same_interval_model_for_hard_and_soft",
            "value_where_separation_enabled",

            "qbound_input_is_generic",
            "qbound_value_uses_raw_visual_tokens",
            "qbound_soft_grounding_feature_exposed",
        }

        for name in expected_on:
            actual = self._scalar_flag(
                qbound_aux,
                name,
            )
            if actual != 1.0:
                raise RuntimeError(
                    f"V2.4 contract violation: {name}={actual}, "
                    "expected 1."
                )

        expected_off = {
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

            "qbound_input_uses_cra_embedding",
            "qbound_value_uses_question_conditioned_states",
            "qbound_vqa_gradient_to_topology_enabled",
            "functional_critic_used_in_decoder",
            "answer_features_used_in_temporal_inference",
        }

        for name in expected_off:
            actual = self._scalar_flag(
                qbound_aux,
                name,
            )
            if actual != 0.0:
                raise RuntimeError(
                    f"V2.4 contract violation: {name}={actual}, "
                    "expected 0."
                )

        # --------------------------------------------------------------
        # Temporal shapes and normalization.
        # --------------------------------------------------------------
        temporal_names = (
            "main_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
            "selected_span_frame_mask",
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
        )

        for name in temporal_names:
            value = qbound_aux[name]
            if value.shape != (
                batch_size,
                time_steps,
            ):
                raise ValueError(
                    f"{name} must be [B,T]={batch_size,time_steps}, got "
                    f"{tuple(value.shape)}."
                )

        for name in (
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
        ):
            value = qbound_aux[name]
            if not torch.isfinite(
                value
            ).all():
                raise RuntimeError(
                    f"{name} contains non-finite values."
                )

            sums = value.sum(
                dim=-1
            )
            if not torch.allclose(
                sums,
                torch.ones_like(sums),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise RuntimeError(
                    f"{name} must sum to one over valid frames."
                )

        # Hard MAP key is intentionally detached from Q-Bound topology.
        if qbound_aux[
            "qbound_key"
        ].requires_grad:
            raise RuntimeError(
                "V2.4 qbound_key must be detached so downstream VQA cannot "
                "backpropagate through temporal topology."
            )

        # Training and soft keys are the same differentiable distribution.
        if not torch.allclose(
            qbound_aux[
                "qbound_training_key"
            ].detach(),
            qbound_aux[
                "qbound_soft_key"
            ].detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "V2.4 training key must equal soft key numerically."
            )

        soft_training_diff = qbound_aux[
            "qbound_soft_training_key_max_abs_diff"
        ]
        if not torch.isfinite(
            soft_training_diff
        ).all():
            raise RuntimeError(
                "qbound_soft_training_key_max_abs_diff is non-finite."
            )

        if float(
            soft_training_diff.detach().max().item()
        ) > 1e-6:
            raise RuntimeError(
                "V2.4 soft and training temporal keys diverged."
            )

        # --------------------------------------------------------------
        # Interval validity.
        # --------------------------------------------------------------
        start = qbound_aux[
            "selected_span_start_frame"
        ]
        end = qbound_aux[
            "selected_span_end_frame"
        ]

        if start.shape != (batch_size,):
            raise ValueError(
                "selected_span_start_frame must be [B]."
            )
        if end.shape != (batch_size,):
            raise ValueError(
                "selected_span_end_frame must be [B]."
            )
        if torch.any(
            start > end
        ):
            raise RuntimeError(
                "V2.4 selected span has start > end."
            )

        # --------------------------------------------------------------
        # Core / continuity dimensions.
        # --------------------------------------------------------------
        if qbound_aux[
            "evidence_core"
        ].shape != (
            batch_size,
            time_steps,
        ):
            raise ValueError(
                "evidence_core must be [B,T]."
            )

        expected_edge_steps = max(
            time_steps - 1,
            0,
        )
        if qbound_aux[
            "event_continuity"
        ].shape != (
            batch_size,
            expected_edge_steps,
        ):
            raise ValueError(
                "event_continuity must be [B,T-1]."
            )

        if qbound_aux[
            "segment_prob"
        ].shape != (
            batch_size,
            time_steps,
            time_steps,
        ):
            raise ValueError(
                "segment_prob must be [B,T,T]."
            )

        # --------------------------------------------------------------
        # Visual evidence shapes.
        # --------------------------------------------------------------
        for name in (
            "qbound_hard_video_proj",
            "qbound_soft_video_proj",
            "qbound_training_video_proj",
            "main_video_proj",
            "qbound_video_proj",
            "cra_video_proj",
        ):
            value = qbound_aux[name]
            if value.shape != video_proj.shape:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match "
                    f"returned video feature {tuple(video_proj.shape)}."
                )

        if not torch.allclose(
            qbound_aux[
                "main_video_proj"
            ].detach(),
            video_proj.detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "V2.4 main_video_proj must be the hard MAP QA feature."
            )

    # ------------------------------------------------------------------
    # Q-Bound V2.4 refine forward
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
    ):
        """
        CRA/Q-Bound V2.4 forward.

        ``answer``
            Existing CRA [Question SEP Answer_i] candidates used only by the
            ordinary downstream multiple-choice QA branch.

        ``question``
            Question-only token ids.  These are the ONLY language features
            supplied to Q-Bound temporal grounding.

        ``answer_only``
            Accepted for repository call-signature compatibility only.
            V2.4 does not encode or use answer-only candidates.

        ``qbound_mix``
            Accepted for compatibility only.  V2.4 never mixes CRA and Q-Bound
            temporal keys.
        """
        del qbound_mix
        del answer_only

        mode = mode.upper()
        if mode not in {
            "VQ",
            "VQA",
        }:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask,
            device,
        )

        # --------------------------------------------------------------
        # Ordinary CRA QA-candidate representation.
        # --------------------------------------------------------------
        answer_g, _ = self._encode_answers(
            answer,
            device,
        )

        batch_size = answer_g.size(0)

        answer_indices = (
            self._prepare_answer_indices(
                answer_id,
                batch_size,
                answer_g.device,
            )
        )

        # CRA visual branch remains the downstream/reference branch.
        video_f = self.video_model(
            video
        )

        qsn_gt = None

        if (
            mode == "VQ"
            and answer_indices is not None
        ):
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
            qsn_gt = qsn_g
        else:
            # Preserve CRA's historical candidate-QA aggregate for the CRA
            # reference grounding branch.  Q-Bound NEVER receives qsn_g.
            qsn_g = answer_g.max(
                dim=1
            ).values

            if answer_indices is not None:
                qsn_gt = (
                    self._select_candidate_features(
                        answer_g,
                        answer_indices,
                    )
                )

        use_qbound = bool(
            self.qbound_enabled
            and mode == "VQA"
        )

        question_g = None
        question_tokens = None
        question_mask = None

        if use_qbound:
            if question is None:
                raise ValueError(
                    "Q-Bound V2.4 requires question-only token ids. "
                    "Pass question=... through CRA.forward()."
                )

            (
                question_g,
                question_tokens,
                question_mask,
            ) = self._encode_question(
                question,
                device,
            )

            if (
                question_g.size(0)
                != batch_size
            ):
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
                    "V2.4 requires token-level question features."
                )

        question_for_qbound = (
            question_tokens
            if use_qbound
            else None
        )

        # --------------------------------------------------------------
        # RefineDecoder V2.4 performs ONE Q-Bound temporal decision:
        #
        #   temporal context -> early Q->V -> core + continuity
        #   -> Core-to-Closure P(s,e).
        # --------------------------------------------------------------
        (
            video_proj,
            video_attn,
            time_param,
        ) = self.decoder(
            video_f,
            qsn_g,
            video_mask,
            gsub=gsub,
            do=self.cfgs["model"].get(
                "do",
                False,
            ),
            window_sizes=self.cfgs["model"].get(
                "window_sizes",
                [1, 3, 5],
            ),
            use_qbound=use_qbound,
            question_feature=question_for_qbound,
            qbound_video_tokens=(
                video
                if use_qbound
                else None
            ),
            qbound_mix=0.0,
            question_mask=(
                question_mask
                if use_qbound
                else None
            ),
        )

        answer_proj = answer_g

        qbound_aux = (
            time_param.get(
                "qbound_aux"
            )
            if isinstance(
                time_param,
                dict,
            )
            else None
        )

        if use_qbound:
            time_steps = video.shape[1]

            self._validate_v240_contract(
                time_param=time_param,
                qbound_aux=qbound_aux,
                video_proj=video_proj,
                batch_size=batch_size,
                time_steps=time_steps,
            )

            # ----------------------------------------------------------
            # Wrapper-level anti-leak / provenance contract.
            # ----------------------------------------------------------
            wrapper_flags = {
                "grounding_requires_answer_at_inference":
                    torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                "temporal_gt_used_in_training":
                    torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                "answer_candidates_used_in_qbound_temporal_path":
                    torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                "answer_only_used_in_v24":
                    torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                "functional_critic_enabled":
                    torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                "vqa_gradient_to_qbound_topology_enabled":
                    torch.zeros(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                "question_grounding_objective_expected":
                    torch.ones(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),

                # The canonical grounding target below is detached so that
                # the new temporal objective cannot be solved simply by
                # moving the question encoder toward the selected video.
                "qbound_grounding_question_stop_gradient":
                    torch.ones(
                        (),
                        dtype=video_proj.dtype,
                        device=video_proj.device,
                    ),
            }

            qbound_aux.update(
                wrapper_flags
            )

            for key, value in (
                wrapper_flags.items()
            ):
                time_param[
                    f"qbound_{key}"
                ] = value

        # --------------------------------------------------------------
        # Shared outputs for cra_pipeline.py / evaluation.
        # --------------------------------------------------------------
        outs = {
            "fatt": video_attn,
            "time_param": time_param,

            # Existing CRA alignment / answer-side state.
            "align_gt": qsn_gt,
            "qa_condition": qsn_g,
            "answer_proj": answer_proj,

            # Question-only state.
            "question_condition": question_g,
            "question_tokens": question_tokens,
            "question_mask": question_mask,
            "question_only_conditioning":
                torch.as_tensor(
                    question_g is not None,
                    device=video_proj.device,
                    dtype=video_proj.dtype,
                ),

            "qbound_align_question":
                question_g,
            "qbound_align_qa":
                qsn_gt,

            "qbound_aux":
                qbound_aux,
            "qbound_active":
                use_qbound,
            "qbound_version":
                self.qbound_version,
            "gpeb_active":
                use_qbound,

            # V2.3 answer-only branch is deliberately absent.
            "answer_only_proj":
                None,
        }

        if use_qbound:
            # ----------------------------------------------------------
            # Canonical V2.4 grounding-supervision interface.
            #
            # Question target is STOP-GRADIENT.  The soft video evidence remains
            # differentiable through P_Q(s,e), allowing the temporal objective to
            # train Core-to-Closure without answer semantics.
            # ----------------------------------------------------------
            grounding_question = (
                question_g.detach()
            )

            outs[
                "qbound_grounding_question"
            ] = grounding_question

            outs[
                "qbound_grounding_video_proj"
            ] = qbound_aux[
                "qbound_soft_video_proj"
            ]

            outs[
                "qbound_grounding_pooled_value"
            ] = qbound_aux[
                "qbound_soft_pooled_value"
            ]

            outs[
                "qbound_grounding_key"
            ] = qbound_aux[
                "qbound_soft_key"
            ]

            # Official VQA path: hard MAP feature.
            outs[
                "qbound_vqa_video_proj"
            ] = video_proj

        if isinstance(
            time_param,
            dict,
        ):
            for key, value in (
                time_param.items()
            ):
                if key.startswith(
                    "qbound_"
                ):
                    outs[key] = value

        if isinstance(
            qbound_aux,
            dict,
        ):
            # High-value V2.4 diagnostics exposed without requiring callers to
            # know RefineDecoder's internal prefixing convention.
            direct_aux_names = (
                # WHERE representation.
                "where_context_states",
                "question_conditioned_states",
                "question_state",
                "temporal_context_delta_ratio",
                "question_interaction_delta_ratio",

                # Evidence Core.
                "evidence_core",
                "evidence_core_raw",
                "evidence_core_prob",
                "evidence_core_entropy",
                "evidence_core_entropy_ratio",
                "evidence_core_top1_prob",

                # Event Continuity.
                "visual_continuity_raw",
                "question_continuity_correction_raw",
                "event_continuity_raw",
                "event_continuity",
                "continuity_mask",

                # Core-to-Closure decomposition.
                "segment_core_support",
                "segment_inside_continuity",
                "segment_boundary_closure",
                "segment_energy",
                "segment_prob",
                "segment_entropy",
                "segment_entropy_ratio",
                "segment_top1_prob",

                # Final interval diagnostics.
                "main_temporal_prob",
                "training_temporal_prob",
                "soft_temporal_prob",
                "frame_membership_prob",
                "selected_span_start_frame",
                "selected_span_end_frame",
                "selected_span_frame_mask",
                "selected_span_length",
                "selected_span_fraction",
                "soft_expected_span_length",
                "soft_expected_span_fraction",
                "single_frame_prob_mass",
                "short_span_prob_mass",
                "selected_single_frame_indicator",
                "selected_full_video_indicator",

                # VALUE / hard-soft evidence.
                "qbound_value_sequence",
                "qbound_hard_pooled_value",
                "qbound_soft_pooled_value",
                "qbound_training_pooled_value",
                "qbound_hard_video_proj",
                "qbound_soft_video_proj",
                "qbound_training_video_proj",
                "qbound_soft_training_key_max_abs_diff",

                # Keys / references.
                "qbound_key",
                "qbound_training_key",
                "qbound_soft_key",
                "cra_key",
                "cra_video_proj",
                "main_video_proj",
                "qbound_video_proj",

                # Architecture.
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

                # Provenance / anti-shortcut.
                "qbound_input_is_generic",
                "qbound_input_uses_cra_embedding",
                "qbound_value_uses_question_conditioned_states",
                "qbound_value_uses_raw_visual_tokens",
                "qbound_vqa_gradient_to_topology_enabled",
                "qbound_soft_grounding_feature_exposed",
                "functional_critic_used_in_decoder",
                "answer_features_used_in_temporal_inference",

                # Wrapper-level invariants.
                "grounding_requires_answer_at_inference",
                "temporal_gt_used_in_training",
                "answer_candidates_used_in_qbound_temporal_path",
                "answer_only_used_in_v24",
                "functional_critic_enabled",
                "vqa_gradient_to_qbound_topology_enabled",
                "question_grounding_objective_expected",
                "qbound_grounding_question_stop_gradient",
            )

            for name in direct_aux_names:
                if name in qbound_aux:
                    outs[name] = qbound_aux[
                        name
                    ]

            # Canonical final aliases.
            outs["qbound_video_proj"] = (
                qbound_aux[
                    "qbound_video_proj"
                ]
            )
            outs["qbound_hard_video_proj"] = (
                qbound_aux[
                    "qbound_hard_video_proj"
                ]
            )
            outs["qbound_soft_video_proj"] = (
                qbound_aux[
                    "qbound_soft_video_proj"
                ]
            )
            outs["qbound_training_video_proj"] = (
                qbound_aux[
                    "qbound_training_video_proj"
                ]
            )

            outs["qbound_key"] = (
                qbound_aux[
                    "qbound_key"
                ]
            )
            outs["qbound_training_key"] = (
                qbound_aux[
                    "qbound_training_key"
                ]
            )
            outs["qbound_soft_key"] = (
                qbound_aux[
                    "qbound_soft_key"
                ]
            )

            # Compatibility aliases: calibrated now means the final V2.4 MAP
            # result; no calibration/smoothing is applied.
            outs["qbound_calibrated_video_proj"] = (
                qbound_aux[
                    "qbound_video_proj"
                ]
            )
            outs["qbound_calibrated_key"] = (
                qbound_aux[
                    "qbound_key"
                ]
            )

            outs["cra_video_proj"] = (
                qbound_aux[
                    "cra_video_proj"
                ]
            )
            outs["cra_key"] = (
                qbound_aux[
                    "cra_key"
                ]
            )

        return (
            video_proj,
            answer_proj,
            outs,
        )

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
        # V2.4-only extras are accepted so repository call sites can keep one
        # common signature.  Baseline CRA does not use them.
        del gsub
        del question
        del qbound_mix
        del answer_only

        mode = mode.upper()
        if mode not in {
            "VQ",
            "VQA",
        }:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask,
            device,
        )

        (
            answer_g,
            answer_tokens,
        ) = self._encode_answers(
            answer,
            device,
        )

        answer_indices = (
            self._prepare_answer_indices(
                answer_id,
                answer_g.size(0),
                answer_g.device,
            )
        )

        video_f = self.video_model(
            video
        )

        if answer_indices is not None:
            qsn_g = (
                self._select_candidate_features(
                    answer_g,
                    answer_indices,
                )
            )
        else:
            qsn_g = answer_g.max(
                dim=1
            ).values

        video_proj, video_attn = self.decoder(
            video_f,
            answer_tokens,
            video_mask,
        )

        outs = {
            "fatt": video_attn,
            "qa_condition": qsn_g,
        }

        return (
            video_proj,
            answer_g,
            outs,
        )

    # ------------------------------------------------------------------
    # Existing weak alignment objective
    # ------------------------------------------------------------------
    def align_loss(
        self,
        qsn,
        grounding_feature,
        neg_sample=None,
    ):
        def _sample_negatives(
            x_pos,
            k,
        ):
            batch_size, feature_dim = (
                x_pos.size()
            )

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

            for index in range(
                batch_size
            ):
                candidate_indices = (
                    torch.cat(
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
                )

                sampled_positions = (
                    torch.randint(
                        low=0,
                        high=candidate_indices.numel(),
                        size=(k,),
                        device=x_pos.device,
                    )
                )

                x_neg[index] = x_pos[
                    candidate_indices[
                        sampled_positions
                    ]
                ]

            return x_neg

        if neg_sample is None:
            negative_count = max(
                1,
                self.cfgs[
                    "dataset"
                ]["batch_size"] // 2,
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
        x = F.normalize(
            x,
            dim=-1,
        )
        x_pos = F.normalize(
            x_pos,
            dim=-1,
        )
        x_neg = F.normalize(
            x_neg,
            dim=-1,
        )

        pos_sim = torch.sum(
            x * x_pos,
            dim=-1,
            keepdim=True,
        ) / temperature

        neg_sim = torch.bmm(
            x.unsqueeze(1),
            x_neg.transpose(1, 2),
        ).squeeze(1) / temperature

        logits = torch.cat(
            [
                pos_sim,
                neg_sim,
            ],
            dim=-1,
        )

        labels = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device,
        )

        return F.cross_entropy(
            logits,
            labels,
        )
