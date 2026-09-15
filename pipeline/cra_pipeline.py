import collections
import os
import time
from random import shuffle

import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate
from torch import nn
import torch.nn.functional as F
import wandb

from .base import BasePipeline
from utils.misc.misc import (
    get_mask,
    get_remain_time,
    load_file,
)



class CRAPipeline(BasePipeline):
    """
    CRA training/evaluation pipeline for Q-Bound V2.1.0.

    V2.1 is intentionally smaller than V2.0.

    Main temporal path
    ------------------
        local/generic CLIP features + question TOKEN features
            -> local temporal primitives
            -> query-conditioned node relevance u_t
            -> adjacent relation/cohesion r_t
            -> one shared variable-scale segment energy E(s,e)
            -> soft P(s,e) + hard H* from the SAME energy

    Q-BoundCore is answer-free.  Candidate answers and GT answer ids never enter
    the temporal segment scorer.

    Training objective
    ------------------
    The repository's ordinary CRA factual objective is retained as the main
    VideoQA testbed objective.  V2.1 adds exactly ONE new Q-Bound-specific loss:

        L = L_CRA/main
            + lambda_A * L_answer_only

    where L_answer_only is cross entropy over answer-only candidates using
    Q-Bound's straight-through evidence representation.

    The forward value of that representation is the hard selected segment H*,
    while the backward path follows the soft structured segment posterior
    P(s,e).  Answer-only candidate embeddings are detached in CRA.py, so this
    auxiliary loss can improve temporal/visual evidence but cannot solve itself
    by moving the language targets.

    Deliberately absent
    -------------------
      * no V2.0 functional KEEP/MERGE verifier;
      * no boundary-equilibrium pseudo target;
      * no cut loss;
      * no event-count loss;
      * no start/end pseudo supervision;
      * no target duration / minimum span;
      * no temporal GT / IoU / IoP training;
      * no teacher distribution / distillation.

    The hard segment remains the official inference/evaluation grounding.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        self.qbound_version = "2.1.0"
        if self._qbound_enabled():
            model_cfg = self.cfgs.get("model", {})
            qbound_cfg = model_cfg.get("qbound", {})
            if not isinstance(qbound_cfg, dict):
                raise TypeError("model.qbound must be a mapping/dict.")

            configured_version = str(
                qbound_cfg.get(
                    "version",
                    model_cfg.get("qbound_version", ""),
                )
            )
            if configured_version != self.qbound_version:
                raise ValueError(
                    "CRAPipeline/Q-Bound version mismatch: expected "
                    f"model.qbound.version={self.qbound_version!r}, got "
                    f"{configured_version!r}."
                )

            # Do not silently carry V2.0 training mechanisms into V2.1.
            stale_v20 = [
                key
                for key in (
                    "functional_composition",
                    "boundary_equilibrium",
                )
                if key in qbound_cfg
            ]
            if stale_v20:
                raise ValueError(
                    "Q-Bound V2.1 config still contains removed V2.0 "
                    "objective blocks: "
                    + ", ".join(stale_v20)
                    + ". Remove them before running V2.1."
                )

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _model_device(self):
        return next(self.model.parameters()).device

    def _qbound_enabled(self):
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})

        if isinstance(qbound_cfg, dict) and "enabled" in qbound_cfg:
            return bool(qbound_cfg["enabled"])

        return bool(
            model_cfg.get(
                "qbound_enabled",
                model_cfg.get("gpeb_enabled", False),
            )
        )

    def _qbound_handover_settings(self):
        """Compatibility helper: V2.1 has no CRA -> Q-Bound handover."""
        return 1.0, 0

    def _train_qbound_mix(
        self,
        epoch,
        batch_index,
        batches_per_epoch,
    ):
        del epoch, batch_index, batches_per_epoch
        return 1.0 if self._qbound_enabled() else 0.0

    def _get_qbound_aux(self, gqa_output):
        if not self._qbound_enabled():
            return None
        if gqa_output is None or len(gqa_output) < 3:
            return None

        model_outputs = gqa_output[2]
        if not isinstance(model_outputs, dict):
            return None

        time_param = model_outputs.get("time_param")
        if not isinstance(time_param, dict):
            return None

        return time_param.get(
            "qbound_aux",
            time_param.get("gpeb_aux"),
        )

    @staticmethod
    def _prepare_labels(answer_id, device):
        return answer_id.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).reshape(-1)

    @staticmethod
    def _scalar_mean(values):
        if values is None:
            return None
        if not torch.is_tensor(values):
            return float(values)
        if values.numel() == 0:
            return 0.0
        return values.detach().float().mean().item()

    @staticmethod
    def _normalize_distribution(values, mask=None, eps=1e-8):
        values = values.clamp_min(0.0)
        if mask is not None:
            while mask.ndim < values.ndim:
                mask = mask.unsqueeze(1)
            values = values * mask.to(values.dtype)
        return values / values.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)

    @classmethod
    def _js_divergence(cls, first, second, mask=None, eps=1e-8):
        first = cls._normalize_distribution(
            first, mask=mask, eps=eps
        )
        second = cls._normalize_distribution(
            second, mask=mask, eps=eps
        )
        middle = 0.5 * (first + second)

        first_kl = torch.sum(
            first
            * (
                torch.log(first.clamp_min(eps))
                - torch.log(middle.clamp_min(eps))
            ),
            dim=-1,
        )
        second_kl = torch.sum(
            second
            * (
                torch.log(second.clamp_min(eps))
                - torch.log(middle.clamp_min(eps))
            ),
            dim=-1,
        )
        return 0.5 * (first_kl + second_kl)

    @staticmethod
    def _answer_decision_margin(logits, labels):
        """
        z_y - logsumexp_{a != y}(z_a), for diagnostics only.
        """
        if logits.ndim != 2:
            raise ValueError(
                "Answer logits must be [B,C], got "
                f"{tuple(logits.shape)}."
            )
        if logits.size(-1) < 2:
            raise ValueError(
                "Answer discrimination requires >=2 candidates."
            )

        labels = labels.reshape(-1)
        if labels.numel() != logits.size(0):
            raise ValueError(
                "labels do not match answer-logit batch size."
            )

        gather_index = labels[:, None]
        correct = logits.gather(
            -1, gather_index
        ).squeeze(-1)

        wrong = logits.clone()
        wrong.scatter_(
            -1,
            gather_index,
            float("-inf"),
        )
        wrong_lse = torch.logsumexp(
            wrong,
            dim=-1,
        )
        return correct - wrong_lse

    # ------------------------------------------------------------------
    # V2.1 objective configuration
    # ------------------------------------------------------------------
    def _qbound_v210_loss_settings(self):
        """
        Minimal V2.1 YAML:

            model:
              qbound:
                answer_only_supervision:
                  loss_weight: 0.10

        V2.1 deliberately introduces only one new loss weight.
        """
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        answer_cfg = qbound_cfg.get(
            "answer_only_supervision",
            {},
        )
        if answer_cfg is None:
            answer_cfg = {}
        if not isinstance(answer_cfg, dict):
            raise TypeError(
                "model.qbound.answer_only_supervision must be "
                "a mapping/dict."
            )

        loss_weight = float(
            answer_cfg.get("loss_weight", 0.10)
        )
        if loss_weight < 0.0:
            raise ValueError(
                "qbound answer_only_supervision.loss_weight "
                "must be >= 0."
            )

        return {
            "answer_only_weight": loss_weight,
        }

    # ------------------------------------------------------------------
    # Repository answer scoring
    # ------------------------------------------------------------------
    def _score_answer_candidates(
        self,
        answer_proj,
        representation,
        detach_answer=True,
        normalize=False,
        logit_scale=1.0,
    ):
        if answer_proj.ndim != 3:
            raise ValueError(
                "answer_proj must be [B,C,D], got "
                f"{tuple(answer_proj.shape)}."
            )

        squeeze_dim = False
        if representation.ndim == 2:
            representation = representation.unsqueeze(1)
            squeeze_dim = True
        elif representation.ndim != 3:
            raise ValueError(
                "representation must be [B,D] or [B,M,D], got "
                f"{tuple(representation.shape)}."
            )

        if answer_proj.size(0) != representation.size(0):
            raise ValueError(
                "answer_proj and representation batch sizes differ."
            )
        if answer_proj.size(-1) != representation.size(-1):
            raise ValueError(
                "answer_proj and representation dimensions differ: "
                f"{answer_proj.size(-1)} vs "
                f"{representation.size(-1)}."
            )

        candidate_features = (
            answer_proj.detach()
            if detach_answer
            else answer_proj
        )
        if normalize:
            candidate_features = F.normalize(
                candidate_features,
                p=2,
                dim=-1,
                eps=1e-6,
            )
            representation = F.normalize(
                representation,
                p=2,
                dim=-1,
                eps=1e-6,
            )

        if logit_scale <= 0.0:
            raise ValueError(
                "logit_scale must be positive."
            )

        logits = float(logit_scale) * torch.einsum(
            "bcd,bmd->bmc",
            candidate_features,
            representation,
        )

        prop_num = int(
            self.cfgs["model"].get("prop_num", 1)
        )
        if prop_num < 1:
            raise ValueError(
                "model.prop_num must be positive."
            )

        total_candidates = logits.size(-1)
        if total_candidates % prop_num != 0:
            raise ValueError(
                "Candidate dimension is not divisible by prop_num: "
                f"{total_candidates} vs {prop_num}."
            )

        logits = logits.reshape(
            logits.size(0),
            logits.size(1),
            prop_num,
            total_candidates // prop_num,
        ).mean(dim=2)

        if squeeze_dim:
            logits = logits.squeeze(1)
        return logits

    # ------------------------------------------------------------------
    # V2.1 answer-only objective
    # ------------------------------------------------------------------
    def _qbound_answer_only_objective(
        self,
        qbound_aux,
        labels,
    ):
        """
        One clean answer-discriminative supervision signal.

        Uses:
            qbound_answer_only_training_logits

        Their forward value is identical to hard H* logits, but gradient reaches
        the shared segment-energy model through V2.1's straight-through soft
        posterior.  Answer-only embeddings are already detached in CRA.py.
        """
        required = {
            "qbound_answer_only_training_logits",
            "qbound_answer_only_hard_logits",
            "qbound_answer_only_soft_logits",
            "qbound_answer_only_train_hard_logit_max_abs_diff",
            "qbound_answer_only_supervision_available",
            "qbound_answer_only_embeddings_stop_gradient",
            "qbound_answer_only_affects_segment_forward",
            "grounding_requires_answer_at_inference",
            "answer_only_used_inside_qbound_core",
            "temporal_gt_used_in_training",
        }
        missing = sorted(
            required.difference(qbound_aux)
        )
        if missing:
            raise KeyError(
                "Q-Bound V2.1 answer-only interface is missing: "
                + ", ".join(missing)
            )

        def _require_flag(name, expected):
            value = qbound_aux[name]
            if (
                not torch.is_tensor(value)
                or value.numel() != 1
            ):
                raise TypeError(
                    f"Q-Bound flag {name!r} must be scalar tensor."
                )
            actual = float(
                value.detach().item()
            )
            if actual != float(expected):
                raise RuntimeError(
                    f"V2.1 answer-only contract violation: "
                    f"{name}={actual}, expected {expected}."
                )

        _require_flag(
            "qbound_answer_only_supervision_available",
            1.0,
        )
        _require_flag(
            "qbound_answer_only_embeddings_stop_gradient",
            1.0,
        )
        _require_flag(
            "qbound_answer_only_affects_segment_forward",
            0.0,
        )
        _require_flag(
            "grounding_requires_answer_at_inference",
            0.0,
        )
        _require_flag(
            "answer_only_used_inside_qbound_core",
            0.0,
        )
        _require_flag(
            "temporal_gt_used_in_training",
            0.0,
        )

        training_logits = qbound_aux[
            "qbound_answer_only_training_logits"
        ]
        hard_logits = qbound_aux[
            "qbound_answer_only_hard_logits"
        ]
        soft_logits = qbound_aux[
            "qbound_answer_only_soft_logits"
        ]

        if training_logits.ndim != 2:
            raise ValueError(
                "qbound_answer_only_training_logits must be [B,C]."
            )
        if hard_logits.shape != training_logits.shape:
            raise ValueError(
                "Hard/training answer-only logits must have "
                "the same shape."
            )
        if soft_logits.shape != training_logits.shape:
            raise ValueError(
                "Soft/training answer-only logits must have "
                "the same shape."
            )
        if labels.numel() != training_logits.size(0):
            raise ValueError(
                "Answer labels do not match answer-only logits."
            )
        if (
            labels.min().item() < 0
            or labels.max().item() >= training_logits.size(-1)
        ):
            raise ValueError(
                "answer_id is outside the answer-only candidate range."
            )

        # This is the ONLY new V2.1 auxiliary training loss.
        answer_only_loss = F.cross_entropy(
            training_logits,
            labels,
        )

        with torch.no_grad():
            hard_ce = F.cross_entropy(
                hard_logits,
                labels,
            ).item()
            soft_ce = F.cross_entropy(
                soft_logits,
                labels,
            ).item()

            hard_acc = (
                hard_logits.argmax(dim=-1) == labels
            ).float().mean().item()
            soft_acc = (
                soft_logits.argmax(dim=-1) == labels
            ).float().mean().item()

            hard_margin = self._answer_decision_margin(
                hard_logits,
                labels,
            )
            soft_margin = self._answer_decision_margin(
                soft_logits,
                labels,
            )

            st_logit_diff = qbound_aux[
                "qbound_answer_only_train_hard_logit_max_abs_diff"
            ].detach().float()

            diagnostics = {
                "answer_only_hard_ce": hard_ce,
                "answer_only_soft_ce": soft_ce,
                "answer_only_hard_acc": hard_acc,
                "answer_only_soft_acc": soft_acc,
                "answer_only_hard_margin": (
                    hard_margin.mean().item()
                ),
                "answer_only_soft_margin": (
                    soft_margin.mean().item()
                ),
                "answer_only_soft_minus_hard_margin": (
                    (soft_margin - hard_margin)
                    .mean()
                    .item()
                ),
                "answer_only_train_hard_logit_max_abs_diff": (
                    st_logit_diff.max().item()
                ),
            }

        return answer_only_loss, diagnostics

    # ------------------------------------------------------------------
    # V2.1 contract validation
    # ------------------------------------------------------------------
    def _validate_v210_contract(self, qbound_aux):
        if qbound_aux is None:
            raise KeyError(
                "Q-Bound V2.1 must expose qbound_aux."
            )

        required = {
            "qbound_version",

            # Provenance / anti-leak.
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_answer_evidence_uses_local_tokens",
            "qbound_answer_evidence_uses_cra_context",
            "functional_verifier_used",
            "boundary_equilibrium_used",
            "teacher_distribution_used",
            "grounding_requires_answer_at_inference",
            "temporal_gt_used_in_training",
            "answer_only_used_inside_qbound_core",

            # Architecture.
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

            # Segment model.
            "node_relevance",
            "adjacent_relation",
            "segment_energy",
            "segment_prob",
            "segment_entropy_ratio",
            "segment_top1_prob",
            "soft_temporal_prob",
            "training_temporal_prob",
            "main_temporal_prob",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",
            "single_frame_prob_mass",
            "short_span_prob_mass",
            "selected_single_frame_indicator",
            "selected_full_video_indicator",

            # Soft/hard feature contract.
            "qbound_hard_video_proj",
            "qbound_training_video_proj",
            "qbound_soft_video_proj",
            "qbound_st_hard_feature_max_abs_diff",

            # Answer-only auxiliary.
            "qbound_answer_only_hard_logits",
            "qbound_answer_only_training_logits",
            "qbound_answer_only_soft_logits",
            "qbound_answer_only_train_hard_logit_max_abs_diff",
            "qbound_answer_only_supervision_available",
            "qbound_answer_only_embeddings_stop_gradient",
            "qbound_answer_only_affects_segment_forward",

            # Comparison keys.
            "cra_key",
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
        }

        missing = sorted(
            required.difference(qbound_aux)
        )
        if missing:
            raise KeyError(
                "Missing Q-Bound V2.1 outputs: "
                + ", ".join(missing)
            )

        if str(
            qbound_aux["qbound_version"]
        ) != self.qbound_version:
            raise RuntimeError(
                "Pipeline/Core Q-Bound version mismatch: "
                f"pipeline={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        def _flag(name, expected):
            value = qbound_aux[name]
            if (
                not torch.is_tensor(value)
                or value.numel() != 1
            ):
                raise TypeError(
                    f"Q-Bound flag {name!r} must be a scalar tensor."
                )
            actual = float(
                value.detach().item()
            )
            if actual != float(expected):
                raise RuntimeError(
                    f"Q-Bound V2.1 flag {name!r} "
                    f"must be {expected}; got {actual}."
                )

        expected_flags = {
            "qbound_input_is_generic": 1.0,
            "qbound_input_uses_cra_embedding": 0.0,
            "qbound_answer_evidence_uses_local_tokens": 1.0,
            "qbound_answer_evidence_uses_cra_context": 0.0,
            "functional_verifier_used": 0.0,
            "boundary_equilibrium_used": 0.0,
            "teacher_distribution_used": 0.0,

            "latent_segment_composition_enabled": 1.0,
            "question_token_interaction_enabled": 1.0,
            "segment_energy_enabled": 1.0,

            "fixed_event_partition_enabled": 0.0,
            "dynamic_event_count_enabled": 0.0,
            "event_count_cap_enabled": 0.0,
            "cut_field_enabled": 0.0,
            "boundary_gated_reasoning_enabled": 0.0,
            "independent_endpoint_heads_enabled": 0.0,
            "hypothesis_competition_enabled": 0.0,
            "teacher_distillation_enabled": 0.0,
            "answer_features_used_in_core": 0.0,
            "temporal_gt_used_in_core": 0.0,

            "qbound_answer_only_supervision_available": 1.0,
            "qbound_answer_only_embeddings_stop_gradient": 1.0,
            "qbound_answer_only_affects_segment_forward": 0.0,
            "grounding_requires_answer_at_inference": 0.0,
            "temporal_gt_used_in_training": 0.0,
            "answer_only_used_inside_qbound_core": 0.0,
        }
        for name, expected in expected_flags.items():
            _flag(name, expected)

        segment_prob = qbound_aux["segment_prob"]
        if segment_prob.ndim != 3:
            raise ValueError(
                "segment_prob must be [B,T,T]."
            )
        batch_size, time_steps, time_steps_2 = (
            segment_prob.shape
        )
        if time_steps != time_steps_2:
            raise ValueError(
                "segment_prob must be square over time."
            )

        for name in (
            "main_temporal_prob",
            "training_temporal_prob",
            "soft_temporal_prob",
        ):
            value = qbound_aux[name]
            if value.shape != (
                batch_size,
                time_steps,
            ):
                raise ValueError(
                    f"{name} must be [B,T], got "
                    f"{tuple(value.shape)}."
                )

        if qbound_aux["node_relevance"].shape != (
            batch_size,
            time_steps,
        ):
            raise ValueError(
                "node_relevance must be [B,T]."
            )

        if qbound_aux["adjacent_relation"].shape != (
            batch_size,
            max(time_steps - 1, 0),
        ):
            raise ValueError(
                "adjacent_relation must be [B,T-1]."
            )

        start = qbound_aux[
            "selected_span_start_frame"
        ]
        end = qbound_aux[
            "selected_span_end_frame"
        ]
        if (
            start.shape != (batch_size,)
            or end.shape != (batch_size,)
        ):
            raise ValueError(
                "selected span endpoints must be [B]."
            )
        if torch.any(start > end):
            raise RuntimeError(
                "Q-Bound V2.1 produced start > end."
            )

        # Hard/ST forward equality.
        feature_diff = qbound_aux[
            "qbound_st_hard_feature_max_abs_diff"
        ].detach()
        logit_diff = qbound_aux[
            "qbound_answer_only_train_hard_logit_max_abs_diff"
        ].detach()

        if (
            not torch.isfinite(feature_diff).all()
            or not torch.isfinite(logit_diff).all()
        ):
            raise RuntimeError(
                "Q-Bound hard/ST diagnostics are non-finite."
            )

        if float(feature_diff.max().item()) > 1e-5:
            raise RuntimeError(
                "V2.1 training feature differs from hard H* "
                "in the forward pass."
            )
        if float(logit_diff.max().item()) > 1e-5:
            raise RuntimeError(
                "V2.1 training answer logits differ from hard H* "
                "in the forward pass."
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _collect_qbound_stats(
        self,
        gqa_output,
        extra_logs=None,
    ):
        stats = {}
        if extra_logs:
            stats.update(extra_logs)

        qbound_aux = self._get_qbound_aux(
            gqa_output
        )
        if qbound_aux is None:
            return stats

        scalar_keys = (
            # Input / question interaction.
            "generic_input_adapter_enabled",
            "generic_input_token_cosine",
            "generic_input_delta_ratio",
            "delta_magnitude",
            "accel_magnitude",
            "question_interaction_delta_ratio",

            # Architecture flags / provenance.
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_answer_evidence_uses_local_tokens",
            "qbound_answer_evidence_uses_cra_context",
            "latent_segment_composition_enabled",
            "question_token_interaction_enabled",
            "segment_energy_enabled",
            "cut_field_enabled",
            "independent_endpoint_heads_enabled",
            "answer_features_used_in_core",
            "teacher_distribution_used",
            "temporal_gt_used_in_training",

            # Segment health.
            "node_relevance",
            "adjacent_relation",
            "segment_entropy_ratio",
            "segment_top1_prob",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",
            "single_frame_prob_mass",
            "short_span_prob_mass",
            "selected_single_frame_indicator",
            "selected_full_video_indicator",

            # ST contract.
            "qbound_st_hard_feature_max_abs_diff",
            "qbound_answer_only_train_hard_logit_max_abs_diff",
        )
        for key in scalar_keys:
            scalar = self._scalar_mean(
                qbound_aux.get(key)
            )
            if scalar is not None:
                stats[f"qbound_{key}"] = scalar

        # Node/edge spread matters more than only their mean.
        for name in (
            "node_relevance",
            "adjacent_relation",
        ):
            values = qbound_aux.get(name)
            if (
                torch.is_tensor(values)
                and values.numel() > 0
            ):
                values = values.detach().float()
                stats[f"qbound_{name}_std"] = (
                    values.std(unbiased=False).item()
                )
                stats[f"qbound_{name}_min"] = (
                    values.min().item()
                )
                stats[f"qbound_{name}_max"] = (
                    values.max().item()
                )

        # Hard vs soft structured grounding.
        hard_key = qbound_aux.get("qbound_key")
        soft_key = qbound_aux.get("qbound_soft_key")
        training_key = qbound_aux.get(
            "qbound_training_key"
        )
        cra_key = qbound_aux.get("cra_key")

        if (
            torch.is_tensor(hard_key)
            and torch.is_tensor(soft_key)
        ):
            hard = self._normalize_distribution(
                hard_key.detach()
            )
            soft = self._normalize_distribution(
                soft_key.detach()
            )
            stats["qbound_hard_soft_js"] = (
                self._js_divergence(
                    hard, soft
                ).mean().item()
            )
            stats["qbound_hard_soft_l1"] = (
                (hard - soft)
                .abs()
                .mean()
                .item()
            )

        if (
            torch.is_tensor(hard_key)
            and torch.is_tensor(cra_key)
        ):
            hard = self._normalize_distribution(
                hard_key.detach()
            )
            cra = self._normalize_distribution(
                cra_key.detach()
            )
            stats["qbound_cra_qbound_js"] = (
                self._js_divergence(
                    cra, hard
                ).mean().item()
            )
            stats["qbound_cra_qbound_key_l1"] = (
                (cra - hard)
                .abs()
                .mean()
                .item()
            )

        if (
            torch.is_tensor(training_key)
            and torch.is_tensor(hard_key)
        ):
            stats[
                "qbound_training_hard_forward_l1"
            ] = (
                training_key.detach()
                - hard_key.detach()
            ).abs().mean().item()

        return stats

    # ------------------------------------------------------------------
    # Gradient diagnostics
    # ------------------------------------------------------------------
    def _qbound_grad_norm(self):
        if not self._qbound_enabled():
            return None

        total_sq = 0.0
        found = False
        for name, parameter in (
            self.model.named_parameters()
        ):
            if (
                "qbound" not in name.lower()
                or parameter.grad is None
            ):
                continue

            grad = (
                parameter.grad
                .detach()
                .float()
            )
            total_sq += (
                grad.pow(2).sum().item()
            )
            found = True

        return (
            float(total_sq ** 0.5)
            if found
            else 0.0
        )

    def _qbound_component_grad_norms(self):
        if not self._qbound_enabled():
            return {
                "qbound_input_adapter_grad_norm": 0.0,
                "qbound_primitive_grad_norm": 0.0,
                "qbound_question_interaction_grad_norm": 0.0,
                "qbound_node_grad_norm": 0.0,
                "qbound_edge_grad_norm": 0.0,
            }

        totals = {
            "input": 0.0,
            "primitive": 0.0,
            "question": 0.0,
            "node": 0.0,
            "edge": 0.0,
        }

        for name, parameter in (
            self.model.named_parameters()
        ):
            if (
                parameter.grad is None
                or "qbound" not in name.lower()
            ):
                continue

            lower = name.lower()
            grad_sq = (
                parameter.grad
                .detach()
                .float()
                .pow(2)
                .sum()
                .item()
            )

            if "generic_input_" in lower:
                totals["input"] += grad_sq

            if any(
                token in lower
                for token in (
                    "state_proj",
                    "delta_proj",
                    "accel_proj",
                    "position_proj",
                    "primitive_fuse",
                )
            ):
                totals["primitive"] += grad_sq

            if any(
                token in lower
                for token in (
                    "question_token_proj",
                    "visual_to_question_attn",
                    "question_cross_norm",
                    "question_ff",
                    "question_ff_norm",
                )
            ):
                totals["question"] += grad_sq

            if (
                "node_encoder" in lower
                or "node_score" in lower
            ):
                totals["node"] += grad_sq

            if (
                "edge_encoder" in lower
                or "edge_score" in lower
            ):
                totals["edge"] += grad_sq

        return {
            "qbound_input_adapter_grad_norm": float(
                totals["input"] ** 0.5
            ),
            "qbound_primitive_grad_norm": float(
                totals["primitive"] ** 0.5
            ),
            "qbound_question_interaction_grad_norm": float(
                totals["question"] ** 0.5
            ),
            "qbound_node_grad_norm": float(
                totals["node"] ** 0.5
            ),
            "qbound_edge_grad_norm": float(
                totals["edge"] ** 0.5
            ),
        }

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_grad_norm",
            "qbound_input_adapter_grad_norm",
            "qbound_primitive_grad_norm",
            "qbound_question_interaction_grad_norm",
            "qbound_node_grad_norm",
            "qbound_edge_grad_norm",

            "qbound_answer_only_weight",
            "qbound_answer_only_discrimination_loss",
            "qbound_answer_only_weighted_loss",
            "qbound_answer_only_hard_ce",
            "qbound_answer_only_soft_ce",
            "qbound_answer_only_hard_acc",
            "qbound_answer_only_soft_acc",
            "qbound_answer_only_hard_margin",
            "qbound_answer_only_soft_margin",
            "qbound_answer_only_soft_minus_hard_margin",
            "qbound_answer_only_train_hard_logit_max_abs_diff",

            "qbound_generic_input_token_cosine",
            "qbound_generic_input_delta_ratio",
            "qbound_question_interaction_delta_ratio",

            "qbound_node_relevance",
            "qbound_node_relevance_std",
            "qbound_node_relevance_min",
            "qbound_node_relevance_max",
            "qbound_adjacent_relation",
            "qbound_adjacent_relation_std",
            "qbound_adjacent_relation_min",
            "qbound_adjacent_relation_max",

            "qbound_segment_entropy_ratio",
            "qbound_segment_top1_prob",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",
            "qbound_soft_expected_span_length",
            "qbound_soft_expected_span_fraction",
            "qbound_single_frame_prob_mass",
            "qbound_short_span_prob_mass",
            "qbound_selected_single_frame_indicator",
            "qbound_selected_full_video_indicator",

            "qbound_hard_soft_js",
            "qbound_hard_soft_l1",
            "qbound_cra_qbound_js",
            "qbound_cra_qbound_key_l1",
            "qbound_training_hard_forward_l1",
            "qbound_qbound_st_hard_feature_max_abs_diff",
            "qbound_qbound_answer_only_train_hard_logit_max_abs_diff",
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def get_loss(self, batch, qbound_mix=1.0):
        del qbound_mix
        device = self._model_device()

        answer_id = batch["answer_id"]
        answer = batch["answer"].to(
            device,
            non_blocking=True,
        )
        video_frames = batch[
            "video_frames"
        ].to(
            device,
            non_blocking=True,
        )
        question = batch[
            "question"
        ].to(
            device,
            non_blocking=True,
        )
        qsns_id = batch["qsns_id"]
        qsns_token_ids = batch[
            "qsns_token_ids"
        ]
        video_len = batch["video_len"]

        answer_only = batch.get("answer_only")
        if self._qbound_enabled():
            if answer_only is None:
                raise KeyError(
                    "Q-Bound V2.1 requires batch['answer_only']. "
                    "Use dataset_qbound_v2_1.py and "
                    "dataloaders_qbound_v2_1.py."
                )
            answer_only = answer_only.to(
                device,
                non_blocking=True,
            )

        video_mask = (
            get_mask(
                video_len,
                self.cfgs["dataset"]["max_feats"],
            ).to(
                device,
                non_blocking=True,
            )
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_device = self._prepare_labels(
            answer_id,
            device,
        )

        prediction_dict = {}
        target_dict = {}

        # --------------------------------------------------------------
        # Original CRA VQ objective remains unchanged and Q-Bound-free.
        # --------------------------------------------------------------
        if self.cfgs["model"]["vg_loss"]:
            vq_output = self.model(
                video_frames,
                video_mask,
                answer=qsns_token_ids,
                answer_id=qsns_id,
                mode="VQ",
            )

            vq_video_proj = (
                vq_output[0].unsqueeze(-1)
            )
            vq_predicts = torch.bmm(
                vq_output[1],
                vq_video_proj,
            ).squeeze(-1)

            vq_predicts = vq_predicts.reshape(
                batch_size,
                self.cfgs["model"]["prop_num"],
                -1,
            ).mean(dim=1)

            prediction_dict[
                "vq_pred"
            ] = vq_predicts
            target_dict[
                "vq_target"
            ] = qsns_id.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ).reshape(-1)

            prediction_dict[
                "vq_align_pred"
            ] = (
                vq_output[2]["time_param"][
                    "grounding_feature"
                ]
            )
            target_dict[
                "vq_align_gt"
            ] = vq_output[2]["align_gt"]

        # --------------------------------------------------------------
        # Main VQA.
        # Forward numerics use hard H*.  Backward reaches the shared
        # segment-energy model through the ST temporal distribution.
        # --------------------------------------------------------------
        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
            qbound_mix=(
                1.0
                if self._qbound_enabled()
                else 0.0
            ),
            answer_only=answer_only,
        )

        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]

        main_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=main_video_proj,
            detach_answer=False,
        )

        prediction_dict[
            "vqa_pred"
        ] = main_logits
        target_dict[
            "vqa_target"
        ] = answer_id_device

        # Preserve repository main objective exactly as a CRA testbed.
        prediction_dict[
            "vqa_align_pred"
        ] = (
            gqa_output[2]["time_param"][
                "grounding_feature"
            ]
        )
        target_dict[
            "vqa_align_gt"
        ] = gqa_output[2]["align_gt"]

        loss = self.criterion(
            prediction_dict,
            target_dict,
        )

        settings = (
            self._qbound_v210_loss_settings()
        )
        qbound_aux = self._get_qbound_aux(
            gqa_output
        )
        qbound_logs = {}

        if self._qbound_enabled():
            self._validate_v210_contract(
                qbound_aux
            )

            # Exactly ONE new V2.1 auxiliary objective.
            if (
                settings["answer_only_weight"]
                > 0.0
            ):
                (
                    answer_only_loss,
                    answer_diag,
                ) = (
                    self._qbound_answer_only_objective(
                        qbound_aux=qbound_aux,
                        labels=answer_id_device,
                    )
                )

                weighted_answer_only = (
                    settings["answer_only_weight"]
                    * answer_only_loss
                )

                loss[
                    "qbound_answer_only_discrimination_loss"
                ] = answer_only_loss
                loss[
                    "qbound_answer_only_weighted_loss"
                ] = weighted_answer_only
                loss[
                    "total_loss"
                ] = (
                    loss["total_loss"]
                    + weighted_answer_only
                )

                qbound_logs.update(
                    {
                        f"qbound_{name}": value
                        for name, value
                        in answer_diag.items()
                    }
                )

            qbound_logs[
                "qbound_answer_only_weight"
            ] = settings[
                "answer_only_weight"
            ]

        predicted = (
            main_logits.argmax(
                dim=-1
            ).detach().cpu()
        )
        running_accuracy = (
            predicted
            == answer_id.reshape(-1).cpu()
        ).float().mean().item()

        qbound_stats = (
            self._collect_qbound_stats(
                gqa_output,
                extra_logs=qbound_logs,
            )
        )

        return (
            loss,
            running_accuracy,
            qbound_stats,
        )

    def _train_epoch(self, epoch):
        self.monitor.log_info(
            f"\n{'#' * 20} Epoch "
            f"{epoch}/{self.cfgs['optim']['epochs']} "
            f"... {'#' * 20}\n"
        )
        print("Training Stage")
        self.model.train()

        self.monitor.reset_kv(
            f"{self.cfgs['stat']['monitor']['metric']}"
        )
        self.monitor.reset_kv(
            "total_loss"
        )

        for key in (
            self.cfgs["optim"]["loss"].keys()
        ):
            if key in {
                "ce",
                "align",
            }:
                continue
            self.monitor.reset_kv(
                f"{key}_loss"
            )

        if self._qbound_enabled():
            self.monitor.reset_kv(
                "qbound_answer_only_discrimination_loss"
            )
            self.monitor.reset_kv(
                "qbound_answer_only_weighted_loss"
            )
            for key in (
                self._qbound_monitor_keys()
            ):
                self.monitor.reset_kv(key)

        for index, batch in enumerate(
            self.train_dataloader
        ):
            start_time = time.time()

            (
                loss,
                monitor_metric,
                qbound_stats,
            ) = self.get_loss(
                batch,
                qbound_mix=(
                    1.0
                    if self._qbound_enabled()
                    else 0.0
                ),
            )

            self.monitor.logkv_mean(
                "acc",
                monitor_metric,
            )
            for name, value in (
                qbound_stats.items()
            ):
                self.monitor.logkv_mean(
                    name,
                    value,
                )

            self.optimizer.zero_grad()
            loss[
                "total_loss"
            ].backward()

            if self._qbound_enabled():
                self.monitor.logkv_mean(
                    "qbound_grad_norm",
                    self._qbound_grad_norm(),
                )
                for (
                    grad_name,
                    grad_value,
                ) in (
                    self._qbound_component_grad_norms()
                    .items()
                ):
                    self.monitor.logkv_mean(
                        grad_name,
                        grad_value,
                    )

            clip_value = (
                self.cfgs["optim"].get(
                    "clip",
                    0,
                )
            )
            if clip_value:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip_value,
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            progress = float(
                index + 1
            ) / len(
                self.train_dataloader
            )
            end_time = time.time()

            (
                _,
                remaining_batch_time,
                remaining_epoch_time,
            ) = get_remain_time(
                start_time,
                end_time,
                index,
                len(self.train_dataloader),
                epoch,
                self.cfgs["optim"]["epochs"],
            )

            print_info = (
                f"Progress: {progress:3.2%} "
                f"(rbt-ret: "
                f"{remaining_batch_time}-"
                f"{remaining_epoch_time}) | "
                f"Train "
                f"{self.cfgs['stat']['monitor']['metric']}: "
                f"{monitor_metric:3.2%}"
            )

            for (
                loss_name,
                loss_value,
            ) in loss.items():
                running_loss = (
                    loss_value
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
                self.monitor.logkv_mean(
                    loss_name,
                    running_loss,
                )
                print_info += (
                    f" {loss_name}: "
                    f"{running_loss:.4f}"
                )

            print(
                f"\r{print_info}",
                end="",
                flush=True,
            )

        table = [[], []]
        for (
            name,
            value,
        ) in self.monitor.name2val.items():
            table[0].append(
                f"mean {name}"
            )
            table[1].append(
                f"{value:.4f}"
            )

        self.monitor.log_info(
            "\n" + tabulate(
                table,
                headers="firstrow",
                tablefmt="grid",
            ),
            end="\n",
        )

        print("Eval Stage")
        val_score, val_result = (
            self.eval(
                self.val_dataloader
            )
        )

        if (
            self.cfgs["dataset"]["name"]
            == "star"
        ):
            test_score = val_score
            test_result = val_result
        else:
            (
                test_score,
                test_result,
            ) = self.eval(
                self.test_dataloader
            )

        score = {
            **{
                f"val_{key}": value
                for key, value
                in val_score.items()
            },
            **{
                f"test_{key}": value
                for key, value
                in test_score.items()
            },
        }
        self.print_score(score)

        wandb_data = {
            "epoch": epoch
        }
        for (
            name,
            value,
        ) in self.monitor.name2val.items():
            wandb_data[
                f"train/{name}"
            ] = value

        for (
            split_name,
            split_dict,
        ) in (
            ("val", val_score),
            ("test", test_score),
        ):
            for (
                key,
                value,
            ) in split_dict.items():
                if isinstance(
                    value,
                    dict,
                ):
                    for (
                        sub_key,
                        sub_value,
                    ) in value.items():
                        wandb_data[
                            f"{split_name}/"
                            f"{key}_{sub_key}"
                        ] = sub_value
                else:
                    wandb_data[
                        f"{split_name}/{key}"
                    ] = value

        wandb.log(
            wandb_data
        )
        return (
            score,
            test_result,
        )

    # ------------------------------------------------------------------
    # Reporting / visualization
    # ------------------------------------------------------------------
    def print_score(self, score):
        def _print_score(
            score_dict
        ):
            self.monitor.log_info("\n")
            for key in (
                score_dict.keys()
            ):
                self.monitor.log_info(
                    f"{key}  "
                )
            self.monitor.log_info("\n")
            for key in (
                score_dict.keys()
            ):
                self.monitor.log_info(
                    f"{score_dict[key]:.1f} \t "
                )

        for split in (
            "val",
            "test",
        ):
            acc_key = f"{split}_acc"
            if acc_key in score:
                self.monitor.log_info(
                    f"\n{split.title()} Set\t "
                    f"Q-Bound V2.1 latent-segment "
                    f"acc: {score[acc_key]:.2%}"
                )

            prehoc_key = (
                f"{split}_pre-hoc"
            )
            if prehoc_key in score:
                self.monitor.log_info(
                    f"\n========== "
                    f"{split} V2.1 hard H* "
                    f"=========="
                )
                _print_score(
                    score[prehoc_key]
                )

            posthoc_key = (
                f"{split}_post-hoc"
            )
            if posthoc_key in score:
                self.monitor.log_info(
                    f"\n========== "
                    f"{split} post-hoc "
                    f"=========="
                )
                _print_score(
                    score[posthoc_key]
                )

            merge_key = (
                f"{split}_merge"
            )
            if merge_key in score:
                self.monitor.log_info(
                    f"\n========== "
                    f"{split} merge "
                    f"=========="
                )
                _print_score(
                    score[merge_key]
                )

        qbound_keys = [
            key
            for key in score
            if (
                key.startswith(
                    "val_qbound_"
                )
                or key.startswith(
                    "test_qbound_"
                )
            )
        ]
        if qbound_keys:
            self.monitor.log_info(
                "\n================ "
                "Q-Bound V2.1 diagnostics "
                "================"
            )
            for key in sorted(
                qbound_keys
            ):
                self.monitor.log_info(
                    f"{key}: "
                    f"{score[key]:.6f}"
                )

        temporal_prefixes = (
            "val_QBound-",
            "test_QBound-",
            "val_QBoundSoft-",
            "test_QBoundSoft-",
            "val_CRA-",
            "test_CRA-",
        )
        temporal_keys = [
            key
            for key in score
            if key.startswith(
                temporal_prefixes
            )
        ]
        if temporal_keys:
            self.monitor.log_info(
                "\n============ "
                "Temporal-key faithfulness "
                "============"
            )
            for key in sorted(
                temporal_keys
            ):
                self.monitor.log_info(
                    f"{key}: "
                    f"{score[key]:.6f}"
                )

        print("\n")

    def draw_grounding(
        self,
        gt,
        posthoc,
        attn,
        merge,
        epoch,
        probs=None,
    ):
        image_path = os.path.join(
            self.save_dir,
            "record",
            "grounding",
        )
        os.makedirs(
            image_path,
            exist_ok=True,
        )

        video_question_ids = list(
            posthoc.keys()
        )
        sample_count = min(
            32,
            len(video_question_ids),
        )
        shuffle(video_question_ids)

        for index in range(
            sample_count
        ):
            video_question_id = (
                video_question_ids[index]
            )
            (
                video_id,
                question_id,
            ) = video_question_id.split(
                "_"
            )

            duration = gt[
                video_id
            ]["duration"]
            ground_truth = gt[
                video_id
            ]["location"][question_id]

            posthoc_ground = posthoc[
                video_question_id
            ]
            attention_ground = np.round(
                np.asarray(
                    attn[
                        video_question_id
                    ]
                )
                * duration,
                1,
            )
            merged_ground = merge[
                video_question_id
            ]

            figure, axis = plt.subplots()
            axis.plot(
                attention_ground,
                [3, 3],
                linewidth=6,
            )
            axis.plot(
                posthoc_ground,
                [2, 2],
                linewidth=6,
            )
            axis.plot(
                merged_ground,
                [1, 1],
                linewidth=6,
            )
            axis.plot(
                ground_truth[0],
                [0, 0],
                linewidth=6,
            )

            if probs is not None:
                probability = np.asarray(
                    probs[
                        video_question_id
                    ]
                ) * 3
                axis.plot(
                    np.linspace(
                        0,
                        duration,
                        len(probability),
                    ),
                    probability,
                    linewidth=3,
                    label="Temporal relevance",
                )

            plt.tight_layout()
            figure.savefig(
                os.path.join(
                    image_path,
                    f"{epoch}_"
                    f"{video_id}_"
                    f"{question_id}.png",
                ),
                dpi=300,
            )
            plt.close(figure)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval(
        self,
        data_loader,
        epoch=None,
        save=False,
    ):
        del epoch, save

        self.model.eval()
        device = self._model_device()

        count = 0
        score = {
            "acc": 0.0
        }
        results = {}
        ground_res = {}
        ground_pre = {}
        keyframe = {}
        ori_keyframe = {}
        qbound_keyframe = {}
        qbound_soft_keyframe = {}
        cra_keyframe = {}

        qbound_stat_sums = (
            collections.defaultdict(float)
        )
        qbound_stat_count = 0
        saw_time_param = False

        with torch.no_grad():
            if not self.cfgs[
                "dataset"
            ]["mc"]:
                model_core = (
                    self.model.module
                    if hasattr(
                        self.model,
                        "module",
                    )
                    else self.model
                )
                if not hasattr(
                    model_core,
                    "_compute_answer_embedding",
                ):
                    raise AttributeError(
                        "Non-multiple-choice evaluation "
                        "requires "
                        "_compute_answer_embedding "
                        "on the model."
                    )
                model_core._compute_answer_embedding(
                    self.a2v
                )

            for (
                index,
                batch,
            ) in enumerate(
                data_loader
            ):
                answer_id = batch[
                    "answer_id"
                ]
                answer = batch[
                    "answer"
                ].to(
                    device,
                    non_blocking=True,
                )
                video_frames = batch[
                    "video_frames"
                ].to(
                    device,
                    non_blocking=True,
                )
                question = batch[
                    "question"
                ].to(
                    device,
                    non_blocking=True,
                )

                answer_only = batch.get(
                    "answer_only"
                )
                if self._qbound_enabled():
                    if answer_only is None:
                        raise KeyError(
                            "Q-Bound V2.1 evaluation "
                            "requires batch['answer_only']."
                        )
                    answer_only = (
                        answer_only.to(
                            device,
                            non_blocking=True,
                        )
                    )

                question_id = batch[
                    "question_id"
                ]
                video_len = batch[
                    "video_len"
                ]
                video_mask = get_mask(
                    video_len,
                    video_frames.size(1),
                ).to(
                    device,
                    non_blocking=True,
                )

                batch_size = answer_id.size(0)
                count += batch_size

                labels_device = (
                    self._prepare_labels(
                        answer_id,
                        device,
                    )
                )
                labels_cpu = (
                    answer_id
                    .reshape(-1)
                    .cpu()
                )

                (
                    fusion_proj,
                    answer_proj,
                    model_outputs,
                ) = self.model(
                    video_frames,
                    video_mask,
                    answer=answer,
                    answer_id=answer_id,
                    mode="VQA",
                    question=question,
                    qbound_mix=(
                        1.0
                        if self._qbound_enabled()
                        else 0.0
                    ),
                    answer_only=answer_only,
                )

                main_logits = (
                    self._score_answer_candidates(
                        answer_proj=answer_proj,
                        representation=fusion_proj,
                        detach_answer=False,
                    )
                )
                predicted = (
                    main_logits.argmax(
                        dim=-1
                    ).cpu()
                )
                score["acc"] += (
                    predicted
                    == labels_cpu
                ).sum().item()

                time_param = (
                    model_outputs.get(
                        "time_param"
                    )
                )
                qbound_aux = None
                eval_wrapper = None

                if isinstance(
                    time_param,
                    dict,
                ):
                    saw_time_param = True
                    eval_wrapper = (
                        None,
                        None,
                        {
                            "time_param":
                                time_param
                        },
                    )
                    qbound_aux = (
                        time_param.get(
                            "qbound_aux",
                            time_param.get(
                                "gpeb_aux"
                            ),
                        )
                    )

                extra_logs = {}
                if (
                    self._qbound_enabled()
                    and qbound_aux is not None
                ):
                    self._validate_v210_contract(
                        qbound_aux
                    )

                    # Evaluation-only diagnostics of the SAME answer-only
                    # interface. No loss is added here.
                    (
                        aux_loss,
                        aux_diag,
                    ) = (
                        self._qbound_answer_only_objective(
                            qbound_aux=qbound_aux,
                            labels=labels_device,
                        )
                    )
                    extra_logs[
                        "qbound_answer_only_discrimination_loss"
                    ] = float(
                        aux_loss.item()
                    )
                    extra_logs.update(
                        {
                            f"qbound_{name}": value
                            for name, value
                            in aux_diag.items()
                        }
                    )

                    batch_stats = (
                        self._collect_qbound_stats(
                            eval_wrapper,
                            extra_logs=extra_logs,
                        )
                    )
                    for (
                        name,
                        value,
                    ) in batch_stats.items():
                        qbound_stat_sums[
                            name
                        ] += (
                            float(value)
                            * batch_size
                        )
                    qbound_stat_count += (
                        batch_size
                    )

                video_attention = (
                    model_outputs["fatt"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if (
                    video_attention.ndim == 3
                    and video_attention.shape[-1]
                    == 1
                ):
                    video_attention = (
                        video_attention.squeeze(-1)
                    )
                if video_attention.ndim == 1:
                    video_attention = (
                        video_attention[None, :]
                    )

                batch_ground_pre = None
                batch_keyframe = None
                batch_ori_keyframe = None
                batch_qbound_key = None
                batch_soft_key = None
                batch_cra_key = None

                if isinstance(
                    time_param,
                    dict,
                ):
                    start_time = (
                        time_param["start"]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    end_time = (
                        time_param["end"]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    batch_ground_pre = (
                        np.column_stack(
                            (
                                start_time,
                                end_time,
                            )
                        )
                    )
                    batch_keyframe = (
                        time_param["key"]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    batch_ori_keyframe = (
                        time_param["ori_key"]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    if qbound_aux is not None:
                        q_key = qbound_aux.get(
                            "qbound_key"
                        )
                        c_key = qbound_aux.get(
                            "cra_key"
                        )
                        soft_key = qbound_aux.get(
                            "qbound_soft_key",
                            qbound_aux.get(
                                "soft_temporal_prob"
                            ),
                        )

                        if torch.is_tensor(
                            q_key
                        ):
                            batch_qbound_key = (
                                q_key
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        if torch.is_tensor(
                            c_key
                        ):
                            batch_cra_key = (
                                c_key
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        if torch.is_tensor(
                            soft_key
                        ):
                            soft_key = (
                                self._normalize_distribution(
                                    soft_key.detach()
                                )
                            )
                            batch_soft_key = (
                                soft_key
                                .cpu()
                                .numpy()
                            )

                for (
                    batch_index,
                    qid,
                ) in enumerate(
                    question_id
                ):
                    results[qid] = {
                        "prediction": int(
                            predicted.numpy()[
                                batch_index
                            ]
                        ),
                        "answer": int(
                            labels_cpu.numpy()[
                                batch_index
                            ]
                        ),
                    }

                    ground_res[qid] = (
                        video_attention[
                            batch_index
                        ].tolist()
                    )

                    if (
                        batch_ground_pre
                        is not None
                    ):
                        ground_pre[qid] = (
                            batch_ground_pre[
                                batch_index
                            ].tolist()
                        )
                    if batch_keyframe is not None:
                        keyframe[qid] = (
                            batch_keyframe[
                                batch_index
                            ].tolist()
                        )
                    if (
                        batch_ori_keyframe
                        is not None
                    ):
                        ori_keyframe[qid] = (
                            batch_ori_keyframe[
                                batch_index
                            ].tolist()
                        )
                    if (
                        batch_qbound_key
                        is not None
                    ):
                        qbound_keyframe[qid] = (
                            batch_qbound_key[
                                batch_index
                            ].tolist()
                        )
                    if (
                        batch_soft_key
                        is not None
                    ):
                        qbound_soft_keyframe[
                            qid
                        ] = (
                            batch_soft_key[
                                batch_index
                            ].tolist()
                        )
                    if (
                        batch_cra_key
                        is not None
                    ):
                        cra_keyframe[qid] = (
                            batch_cra_key[
                                batch_index
                            ].tolist()
                        )

                progress = float(
                    index + 1
                ) / len(
                    data_loader
                )
                print(
                    f"\rProgress: "
                    f"{progress:.2%}",
                    end="",
                    flush=True,
                )

        if count == 0:
            raise RuntimeError(
                "Evaluation data loader is empty."
            )

        score["acc"] /= count

        if qbound_stat_count > 0:
            for (
                name,
                total,
            ) in qbound_stat_sums.items():
                score[name] = (
                    total
                    / qbound_stat_count
                )

        split = data_loader.dataset.split
        csv_path = self.cfgs[
            "dataset"
        ]["csv_path"]

        ground_truth = load_file(
            os.path.join(
                csv_path,
                f"gsub_{split}.json",
            )
        )
        segments = load_file(
            os.path.join(
                csv_path,
                f"frame2time_{split}.json",
            )
        )
        questions = load_file(
            os.path.join(
                csv_path,
                f"{split}.csv",
            )
        )

        post_hoc_ground = (
            self.metric.generate_ground(
                ground_res,
                segments,
                questions,
            )
        )
        score["post-hoc"] = self.metric(
            post_hoc_ground,
            ground_truth,
            results,
            gs=False,
        )

        if qbound_keyframe:
            if not hasattr(
                self.metric,
                "temporal_key_metrics",
            ):
                raise AttributeError(
                    "Q-Bound V2.1 requires "
                    "Metric.temporal_key_metrics()."
                )

            score.update(
                self.metric.temporal_key_metrics(
                    qbound_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBound",
                )
            )

        if qbound_soft_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_soft_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundSoft",
                )
            )

        if cra_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    cra_keyframe,
                    ground_truth,
                    segments,
                    prefix="CRA",
                )
            )

        result = {
            "ph": post_hoc_ground,
            "result": results,
            "gt_ground": ground_truth,
            "ph_attn": ground_res,
            "qbound_key": qbound_keyframe,
            "qbound_soft_key":
                qbound_soft_keyframe,
            "cra_key": cra_keyframe,
        }

        if (
            saw_time_param
            and ground_pre
        ):
            score["pre-hoc"] = self.metric(
                ground_pre,
                ground_truth,
                results,
                gs=True,
            )

            merge_ground = (
                self.metric.combine(
                    ground_pre,
                    post_hoc_ground,
                    ground_truth,
                )
            )
            score["merge"] = self.metric(
                merge_ground,
                ground_truth,
                results,
                gs=False,
            )

            result.update(
                {
                    "tg": ground_pre,
                    "key": keyframe,
                    "ori_key":
                        ori_keyframe,
                    "merge": merge_ground,
                }
            )

        return score, result

    def infer(self, epoch=None):
        parameter_count = sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )
        print(
            f"Model Params: "
            f"{parameter_count}"
        )
        print("\nEval Stage")

        val_score, _ = self.eval(
            self.val_dataloader,
            epoch,
            self.cfgs["dataset"]["name"]
            == "star",
        )

        if (
            self.cfgs["dataset"]["name"]
            == "star"
        ):
            test_score = val_score
        else:
            test_score, _ = self.eval(
                self.test_dataloader,
                epoch,
                True,
            )

        score = {
            **{
                f"val_{key}": value
                for key, value
                in val_score.items()
            },
            **{
                f"test_{key}": value
                for key, value
                in test_score.items()
            },
        }

        self.print_score(score)
        return score
