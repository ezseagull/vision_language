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
    CRA training/evaluation pipeline for Q-Bound V2.4.0.

    V2.4 makes one answer-free temporal decision:

        Video + Question
            -> temporal context + early Q->V interaction
            -> Evidence Core a_t + Event Continuity j_t
            -> Core-to-Closure P_Q(s,e)
            -> HARD MAP interval for ordinary CRA VideoQA
            -> SOFT frame membership for Q-Bound grounding training.

    The only Q-Bound-specific objective is Video-Question grounding:

        L_total = L_CRA/main + lambda_G * L_ground

    L_ground compares the differentiable SOFT grounded RAW visual value,
    projected through the host video projector with stop-gradient weights, with
    a stop-gradient QUESTION-ONLY representation.  Answer semantics, temporal GT,
    IoU and IoP are never used to train temporal topology.

    Removed from V2.3: BoundaryRefiner, adaptive-radius candidate search,
    answer-only functional critic, hierarchical ST, length/minimality objectives,
    Gaussian smoothing and event-count logic.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        self.qbound_version = "2.4.0"

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

            stale_keys = (
                "functional_critic",
                "boundary_refiner",
                "relation_weight",
                "boundary_weight",
                "adaptive_radius_ratio",
                "min_radius",
                "functional_composition",
                "boundary_equilibrium",
                "answer_only_supervision",
            )
            stale = sorted(key for key in stale_keys if key in qbound_cfg)
            if stale:
                raise ValueError(
                    "Q-Bound V2.4 config contains removed V2.3/legacy keys: "
                    + ", ".join(stale)
                )

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
        """Compatibility helper: V2.4 has no CRA -> Q-Bound handover."""
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

    def _qbound_v240_loss_settings(self):
        """Return the minimal V2.4 Video-Question grounding settings."""
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        grounding_cfg = qbound_cfg.get("grounding_loss", {})
        if grounding_cfg is None:
            grounding_cfg = {}
        if not isinstance(grounding_cfg, dict):
            raise TypeError(
                "model.qbound.grounding_loss must be a mapping/dict."
            )

        enabled = bool(grounding_cfg.get("enabled", True))
        loss_weight = float(grounding_cfg.get("loss_weight", 1.0))
        temperature = float(grounding_cfg.get("temperature", 0.07))

        if loss_weight < 0.0:
            raise ValueError("grounding_loss.loss_weight must be >= 0.")
        if temperature <= 0.0:
            raise ValueError("grounding_loss.temperature must be > 0.")
        if not enabled and loss_weight != 0.0:
            raise ValueError(
                "grounding_loss.enabled=false requires loss_weight=0."
            )

        return {
            "grounding_enabled": enabled,
            "grounding_weight": loss_weight,
            "grounding_temperature": temperature,
        }

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

    def _project_qbound_value_with_frozen_host(
        self,
        pooled_value,
    ):
        """
        Map raw CLIP VALUE evidence into CRA's language-comparison space while
        stopping gradients into the CRA projection itself.
        """
        if pooled_value.ndim != 2:
            raise ValueError("pooled_value must be [B,D].")

        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        decoder = getattr(model_core, "decoder", None)
        if decoder is None:
            raise AttributeError(
                "Q-Bound V2.4 grounding requires model.decoder."
            )

        projection = getattr(decoder, "video_proj", None)
        if projection is None:
            raise AttributeError(
                "RefineDecoder must expose video_proj for the frozen "
                "grounding metric adapter."
            )

        linear = None
        for module in projection.modules():
            if isinstance(module, nn.Linear):
                linear = module
                break

        if linear is None:
            raise TypeError(
                "decoder.video_proj must contain an nn.Linear layer."
            )

        if pooled_value.size(-1) != linear.in_features:
            raise ValueError(
                "Q-Bound pooled VALUE dimension does not match host "
                f"projection input: {pooled_value.size(-1)} vs "
                f"{linear.in_features}."
            )

        weight = linear.weight.detach()
        bias = (
            linear.bias.detach()
            if linear.bias is not None
            else None
        )

        return F.linear(
            pooled_value,
            weight,
            bias,
        )

    def _qbound_video_question_objective(
        self,
        model_outputs,
        qbound_aux,
        temperature,
    ):
        """
        One-directional in-batch Video->Question contrastive grounding.

        qbound_grounding_pooled_value is the SOFT temporal pooling over raw CLIP
        VALUE tokens.  It is mapped through CRA's existing video projection with
        stop-gradient weights only to obtain a compatible metric space.
        qbound_grounding_question is QUESTION-ONLY and detached.  Therefore the
        new loss trains temporal selection rather than answer/language/projector
        shortcuts.
        """
        if not isinstance(model_outputs, dict):
            raise TypeError("model_outputs must be a dict.")
        if not isinstance(qbound_aux, dict):
            raise TypeError("qbound_aux must be a dict.")

        required = {
            "qbound_grounding_pooled_value",
            "qbound_grounding_question",
            "qbound_grounding_key",
        }
        missing = sorted(required.difference(model_outputs))
        if missing:
            raise KeyError(
                "Q-Bound V2.4 grounding interface is missing: "
                + ", ".join(missing)
            )

        pooled_value = model_outputs["qbound_grounding_pooled_value"]
        question_target = model_outputs["qbound_grounding_question"]
        soft_key = model_outputs["qbound_grounding_key"]

        if pooled_value.ndim != 2 or question_target.ndim != 2:
            raise ValueError(
                "Grounded VALUE and question target must both be [B,D]."
            )

        metric_value = self._project_qbound_value_with_frozen_host(
            pooled_value
        )

        if metric_value.shape != question_target.shape:
            raise ValueError(
                "Frozen host-projected VALUE and question target must have "
                f"the same [B,D]; got {tuple(metric_value.shape)} and "
                f"{tuple(question_target.shape)}."
            )
        if soft_key.ndim != 2 or soft_key.size(0) != pooled_value.size(0):
            raise ValueError("qbound_grounding_key must be [B,T].")
        if question_target.requires_grad:
            raise RuntimeError(
                "V2.4 grounding question target must be stop-gradient."
            )
        if temperature <= 0.0:
            raise ValueError("grounding temperature must be positive.")

        batch_size = pooled_value.size(0)
        if batch_size < 2:
            zero = pooled_value.sum() * 0.0
            return zero, {
                "grounding_batch_size": float(batch_size),
                "grounding_retrieval_acc": 0.0,
                "grounding_positive_similarity": 0.0,
                "grounding_negative_similarity": 0.0,
                "grounding_hardest_negative_similarity": 0.0,
                "grounding_margin": 0.0,
            }

        video_unit = F.normalize(metric_value, p=2, dim=-1, eps=1e-6)
        question_unit = F.normalize(
            question_target, p=2, dim=-1, eps=1e-6
        )

        similarity = torch.matmul(
            video_unit,
            question_unit.transpose(0, 1),
        )
        logits = similarity / float(temperature)
        labels = torch.arange(
            batch_size,
            device=logits.device,
            dtype=torch.long,
        )
        grounding_loss = F.cross_entropy(logits, labels)

        with torch.no_grad():
            positive = similarity.diag()
            off_diagonal = ~torch.eye(
                batch_size,
                dtype=torch.bool,
                device=similarity.device,
            )
            negative = similarity[off_diagonal].reshape(
                batch_size, batch_size - 1
            )
            hardest_negative = negative.max(dim=-1).values
            predicted = logits.argmax(dim=-1)

            diagnostics = {
                "grounding_batch_size": float(batch_size),
                "grounding_retrieval_acc": float(
                    (predicted == labels).float().mean().item()
                ),
                "grounding_positive_similarity": float(
                    positive.mean().item()
                ),
                "grounding_negative_similarity": float(
                    negative.mean().item()
                ),
                "grounding_hardest_negative_similarity": float(
                    hardest_negative.mean().item()
                ),
                "grounding_margin": float(
                    (positive - hardest_negative).mean().item()
                ),
            }

        return grounding_loss, diagnostics

    def _validate_v240_contract(self, qbound_aux):
        if qbound_aux is None:
            raise KeyError("Q-Bound V2.4 must expose qbound_aux.")

        required = {
            "qbound_version",
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
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_value_uses_question_conditioned_states",
            "qbound_value_uses_raw_visual_tokens",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_soft_grounding_feature_exposed",
            "functional_critic_used_in_decoder",
            "answer_features_used_in_temporal_inference",
            "grounding_requires_answer_at_inference",
            "temporal_gt_used_in_training",
            "answer_candidates_used_in_qbound_temporal_path",
            "answer_only_used_in_v24",
            "functional_critic_enabled",
            "vqa_gradient_to_qbound_topology_enabled",
            "question_grounding_objective_expected",
            "qbound_grounding_question_stop_gradient",
            "evidence_core",
            "evidence_core_prob",
            "evidence_core_entropy_ratio",
            "evidence_core_top1_prob",
            "event_continuity",
            "continuity_mask",
            "segment_prob",
            "segment_entropy_ratio",
            "segment_top1_prob",
            "single_frame_prob_mass",
            "short_span_prob_mass",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "selected_span_frame_mask",
            "selected_span_length",
            "selected_span_fraction",
            "soft_expected_span_length",
            "soft_expected_span_fraction",
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
            "cra_key",
            "qbound_value_sequence",
            "qbound_hard_pooled_value",
            "qbound_soft_pooled_value",
            "qbound_training_pooled_value",
            "qbound_hard_video_proj",
            "qbound_soft_video_proj",
            "qbound_training_video_proj",
            "qbound_soft_training_key_max_abs_diff",
        }

        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "Missing Q-Bound V2.4 outputs: " + ", ".join(missing)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "Pipeline/Core Q-Bound version mismatch: "
                f"pipeline={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        def _flag(name, expected):
            value = qbound_aux[name]
            if not torch.is_tensor(value) or value.numel() != 1:
                raise TypeError(
                    f"Q-Bound flag {name!r} must be scalar tensor."
                )
            actual = float(value.detach().item())
            if actual != float(expected):
                raise RuntimeError(
                    f"Q-Bound V2.4 flag {name!r} must be "
                    f"{expected}; got {actual}."
                )

        for name in (
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
            "question_grounding_objective_expected",
            "qbound_grounding_question_stop_gradient",
        ):
            _flag(name, 1.0)

        for name in (
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
            "grounding_requires_answer_at_inference",
            "temporal_gt_used_in_training",
            "answer_candidates_used_in_qbound_temporal_path",
            "answer_only_used_in_v24",
            "functional_critic_enabled",
            "vqa_gradient_to_qbound_topology_enabled",
        ):
            _flag(name, 0.0)

        segment_prob = qbound_aux["segment_prob"]
        if (
            segment_prob.ndim != 3
            or segment_prob.size(1) != segment_prob.size(2)
        ):
            raise ValueError("segment_prob must be [B,T,T].")

        batch_size, time_steps, _ = segment_prob.shape

        for name in (
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
            "selected_span_frame_mask",
            "evidence_core",
        ):
            value = qbound_aux[name]
            if value.shape != (batch_size, time_steps):
                raise ValueError(
                    f"{name} must be [B,T], got {tuple(value.shape)}."
                )

        if qbound_aux["event_continuity"].shape != (
            batch_size,
            max(time_steps - 1, 0),
        ):
            raise ValueError("event_continuity must be [B,T-1].")

        for name in (
            "qbound_key",
            "qbound_training_key",
            "qbound_soft_key",
        ):
            value = qbound_aux[name]
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{name} contains non-finite values.")
            total = value.sum(dim=-1)
            if not torch.allclose(
                total,
                torch.ones_like(total),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise RuntimeError(f"{name} must sum to one.")

        if not torch.allclose(
            qbound_aux["qbound_training_key"].detach(),
            qbound_aux["qbound_soft_key"].detach(),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise RuntimeError(
                "V2.4 training key must equal soft key numerically."
            )

        start = qbound_aux["selected_span_start_frame"]
        end = qbound_aux["selected_span_end_frame"]
        if start.shape != (batch_size,) or end.shape != (batch_size,):
            raise ValueError("selected span endpoints must be [B].")
        if torch.any(start > end):
            raise RuntimeError("Q-Bound V2.4 selected span has start > end.")

        diff = qbound_aux[
            "qbound_soft_training_key_max_abs_diff"
        ].detach()
        if not torch.isfinite(diff).all():
            raise RuntimeError("soft/training key diagnostic is non-finite.")
        if float(diff.max().item()) > 1e-6:
            raise RuntimeError("V2.4 training and soft keys diverged.")

    def _collect_qbound_stats(
        self,
        gqa_output,
        extra_logs=None,
    ):
        stats = {}
        if extra_logs:
            stats.update(extra_logs)

        qbound_aux = self._get_qbound_aux(gqa_output)
        if qbound_aux is None:
            return stats

        for key in (
            "temporal_context_delta_ratio",
            "question_interaction_delta_ratio",
            "evidence_core_entropy_ratio",
            "evidence_core_top1_prob",
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
            "qbound_soft_training_key_max_abs_diff",
        ):
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                log_key = (
                    key
                    if key.startswith("qbound_")
                    else f"qbound_{key}"
                )
                stats[log_key] = scalar

        for name in (
            "evidence_core",
            "event_continuity",
            "visual_continuity_raw",
            "question_continuity_correction_raw",
        ):
            values = qbound_aux.get(name)
            if torch.is_tensor(values) and values.numel() > 0:
                values = values.detach().float()
                stats[f"qbound_{name}_mean"] = values.mean().item()
                stats[f"qbound_{name}_std"] = values.std(
                    unbiased=False
                ).item()
                stats[f"qbound_{name}_min"] = values.min().item()
                stats[f"qbound_{name}_max"] = values.max().item()

        hard_key = qbound_aux.get("qbound_key")
        soft_key = qbound_aux.get("qbound_soft_key")
        cra_key = qbound_aux.get("cra_key")

        if torch.is_tensor(hard_key) and torch.is_tensor(soft_key):
            hard = self._normalize_distribution(hard_key.detach())
            soft = self._normalize_distribution(soft_key.detach())
            stats["qbound_hard_soft_js"] = self._js_divergence(
                hard, soft
            ).mean().item()
            stats["qbound_hard_soft_l1"] = (
                hard - soft
            ).abs().mean().item()

        if torch.is_tensor(hard_key) and torch.is_tensor(cra_key):
            hard = self._normalize_distribution(hard_key.detach())
            cra = self._normalize_distribution(cra_key.detach())
            stats["qbound_cra_hard_js"] = self._js_divergence(
                cra, hard
            ).mean().item()
            stats["qbound_cra_hard_l1"] = (
                cra - hard
            ).abs().mean().item()

        return stats

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
                "qbound_where_input_grad_norm": 0.0,
                "qbound_temporal_context_grad_norm": 0.0,
                "qbound_question_interaction_grad_norm": 0.0,
                "qbound_core_head_grad_norm": 0.0,
                "qbound_visual_continuity_grad_norm": 0.0,
                "qbound_question_continuity_grad_norm": 0.0,
            }

        totals = {
            "where_input": 0.0,
            "temporal": 0.0,
            "question": 0.0,
            "core": 0.0,
            "visual_cont": 0.0,
            "question_cont": 0.0,
        }

        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or "qbound" not in name.lower():
                continue

            lower = name.lower()
            grad_sq = (
                parameter.grad.detach().float().pow(2).sum().item()
            )

            if "where_input_proj" in lower:
                totals["where_input"] += grad_sq
            if any(
                token in lower
                for token in (
                    "temporal_norm",
                    "temporal_attn",
                    "temporal_ff",
                )
            ):
                totals["temporal"] += grad_sq
            if any(
                token in lower
                for token in (
                    "question_token_proj",
                    "qv_norm",
                    "qv_attn",
                    "qv_ff",
                )
            ):
                totals["question"] += grad_sq
            if "core_encoder" in lower or "core_score" in lower:
                totals["core"] += grad_sq
            if "visual_continuity" in lower:
                totals["visual_cont"] += grad_sq
            if "question_continuity" in lower:
                totals["question_cont"] += grad_sq

        return {
            "qbound_where_input_grad_norm":
                float(totals["where_input"] ** 0.5),
            "qbound_temporal_context_grad_norm":
                float(totals["temporal"] ** 0.5),
            "qbound_question_interaction_grad_norm":
                float(totals["question"] ** 0.5),
            "qbound_core_head_grad_norm":
                float(totals["core"] ** 0.5),
            "qbound_visual_continuity_grad_norm":
                float(totals["visual_cont"] ** 0.5),
            "qbound_question_continuity_grad_norm":
                float(totals["question_cont"] ** 0.5),
        }

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_grad_norm",
            "qbound_where_input_grad_norm",
            "qbound_temporal_context_grad_norm",
            "qbound_question_interaction_grad_norm",
            "qbound_core_head_grad_norm",
            "qbound_visual_continuity_grad_norm",
            "qbound_question_continuity_grad_norm",
            "qbound_grounding_weight",
            "qbound_grounding_temperature",
            "qbound_grounding_loss",
            "qbound_grounding_weighted_loss",
            "qbound_grounding_batch_size",
            "qbound_grounding_retrieval_acc",
            "qbound_grounding_positive_similarity",
            "qbound_grounding_negative_similarity",
            "qbound_grounding_hardest_negative_similarity",
            "qbound_grounding_margin",
            "qbound_temporal_context_delta_ratio",
            "qbound_question_interaction_delta_ratio",
            "qbound_evidence_core_entropy_ratio",
            "qbound_evidence_core_top1_prob",
            "qbound_evidence_core_mean",
            "qbound_evidence_core_std",
            "qbound_event_continuity_mean",
            "qbound_event_continuity_std",
            "qbound_visual_continuity_raw_mean",
            "qbound_visual_continuity_raw_std",
            "qbound_question_continuity_correction_raw_mean",
            "qbound_question_continuity_correction_raw_std",
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
            "qbound_cra_hard_js",
            "qbound_cra_hard_l1",
            "qbound_soft_training_key_max_abs_diff",
        ]

    def get_loss(self, batch, qbound_mix=1.0):
        del qbound_mix
        device = self._model_device()

        answer_id = batch["answer_id"]
        answer = batch["answer"].to(device, non_blocking=True)
        video_frames = batch["video_frames"].to(
            device, non_blocking=True
        )
        question = batch["question"].to(
            device, non_blocking=True
        )
        qsns_id = batch["qsns_id"]
        qsns_token_ids = batch["qsns_token_ids"]
        video_len = batch["video_len"]

        settings = self._qbound_v240_loss_settings()

        video_mask = (
            get_mask(
                video_len,
                self.cfgs["dataset"]["max_feats"],
            ).to(device, non_blocking=True)
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_device = self._prepare_labels(
            answer_id, device
        )

        prediction_dict = {}
        target_dict = {}

        # Original CRA VQ objective remains Q-Bound-free.
        if self.cfgs["model"]["vg_loss"]:
            vq_output = self.model(
                video_frames,
                video_mask,
                answer=qsns_token_ids,
                answer_id=qsns_id,
                mode="VQ",
            )

            vq_video_proj = vq_output[0].unsqueeze(-1)
            vq_predicts = torch.bmm(
                vq_output[1],
                vq_video_proj,
            ).squeeze(-1)

            vq_predicts = vq_predicts.reshape(
                batch_size,
                self.cfgs["model"]["prop_num"],
                -1,
            ).mean(dim=1)

            prediction_dict["vq_pred"] = vq_predicts
            target_dict["vq_target"] = qsns_id.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            ).reshape(-1)
            prediction_dict["vq_align_pred"] = (
                vq_output[2]["time_param"]["grounding_feature"]
            )
            target_dict["vq_align_gt"] = vq_output[2]["align_gt"]

        # Main VQA uses the detached HARD MAP temporal key.
        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
            qbound_mix=(
                1.0 if self._qbound_enabled() else 0.0
            ),
            answer_only=None,
        )

        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]
        model_outputs = gqa_output[2]

        main_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=main_video_proj,
            detach_answer=False,
        )

        prediction_dict["vqa_pred"] = main_logits
        target_dict["vqa_target"] = answer_id_device

        prediction_dict["vqa_align_pred"] = (
            model_outputs["time_param"]["grounding_feature"]
        )
        target_dict["vqa_align_gt"] = model_outputs["align_gt"]

        loss = self.criterion(
            prediction_dict,
            target_dict,
        )

        qbound_aux = self._get_qbound_aux(gqa_output)
        qbound_logs = {}

        if self._qbound_enabled():
            self._validate_v240_contract(qbound_aux)

            if (
                settings["grounding_enabled"]
                and settings["grounding_weight"] > 0.0
            ):
                grounding_loss, grounding_diag = (
                    self._qbound_video_question_objective(
                        model_outputs=model_outputs,
                        qbound_aux=qbound_aux,
                        temperature=settings[
                            "grounding_temperature"
                        ],
                    )
                )

                weighted_grounding = (
                    settings["grounding_weight"]
                    * grounding_loss
                )

                loss["qbound_grounding_loss"] = grounding_loss
                loss[
                    "qbound_grounding_weighted_loss"
                ] = weighted_grounding
                loss["total_loss"] = (
                    loss["total_loss"]
                    + weighted_grounding
                )

                qbound_logs.update(
                    {
                        f"qbound_{name}": value
                        for name, value
                        in grounding_diag.items()
                    }
                )

            qbound_logs["qbound_grounding_weight"] = (
                settings["grounding_weight"]
            )
            qbound_logs["qbound_grounding_temperature"] = (
                settings["grounding_temperature"]
            )

        predicted = (
            main_logits.argmax(dim=-1)
            .detach()
            .cpu()
        )
        running_accuracy = (
            predicted
            == answer_id.reshape(-1).cpu()
        ).float().mean().item()

        qbound_stats = self._collect_qbound_stats(
            gqa_output,
            extra_logs=qbound_logs,
        )

        return loss, running_accuracy, qbound_stats

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
                "qbound_grounding_loss"
            )
            self.monitor.reset_kv(
                "qbound_grounding_weighted_loss"
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

    def print_score(self, score):
        def _print_score(score_dict):
            self.monitor.log_info("\n")
            for key in score_dict.keys():
                self.monitor.log_info(f"{key}  ")
            self.monitor.log_info("\n")
            for key in score_dict.keys():
                self.monitor.log_info(
                    f"{score_dict[key]:.1f} \t "
                )

        for split in ("val", "test"):
            acc_key = f"{split}_acc"
            if acc_key in score:
                self.monitor.log_info(
                    f"\n{split.title()} Set\t "
                    f"Q-Bound V2.4 acc: {score[acc_key]:.2%}"
                )

            prehoc_key = f"{split}_pre-hoc"
            if prehoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} V2.4 hard MAP =========="
                )
                _print_score(score[prehoc_key])

            posthoc_key = f"{split}_post-hoc"
            if posthoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} CRA post-hoc =========="
                )
                _print_score(score[posthoc_key])

            merge_key = f"{split}_merge"
            if merge_key in score:
                self.monitor.log_info(
                    f"\n========== {split} merge diagnostic =========="
                )
                _print_score(score[merge_key])

        qbound_keys = [
            key for key in score
            if (
                key.startswith("val_qbound_")
                or key.startswith("test_qbound_")
            )
        ]
        if qbound_keys:
            self.monitor.log_info(
                "\n================ Q-Bound V2.4 diagnostics ================"
            )
            for key in sorted(qbound_keys):
                self.monitor.log_info(
                    f"{key}: {score[key]:.6f}"
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
            key for key in score
            if key.startswith(temporal_prefixes)
        ]
        if temporal_keys:
            self.monitor.log_info(
                "\n============ Temporal-key faithfulness ============"
            )
            for key in sorted(temporal_keys):
                self.monitor.log_info(
                    f"{key}: {score[key]:.6f}"
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
        score = {"acc": 0.0}

        results = {}
        ground_res = {}
        ground_pre = {}

        keyframe = {}
        ori_keyframe = {}
        qbound_keyframe = {}
        qbound_soft_keyframe = {}
        cra_keyframe = {}

        qbound_stat_sums = collections.defaultdict(float)
        qbound_stat_count = 0
        saw_time_param = False

        with torch.no_grad():
            if not self.cfgs["dataset"]["mc"]:
                model_core = (
                    self.model.module
                    if hasattr(self.model, "module")
                    else self.model
                )
                if not hasattr(
                    model_core,
                    "_compute_answer_embedding",
                ):
                    raise AttributeError(
                        "Non-multiple-choice evaluation requires "
                        "_compute_answer_embedding on the model."
                    )
                model_core._compute_answer_embedding(self.a2v)

            for index, batch in enumerate(data_loader):
                answer_id = batch["answer_id"]
                answer = batch["answer"].to(
                    device, non_blocking=True
                )
                video_frames = batch["video_frames"].to(
                    device, non_blocking=True
                )
                question = batch["question"].to(
                    device, non_blocking=True
                )
                question_id = batch["question_id"]
                video_len = batch["video_len"]

                video_mask = get_mask(
                    video_len,
                    video_frames.size(1),
                ).to(device, non_blocking=True)

                batch_size = answer_id.size(0)
                count += batch_size
                labels_cpu = answer_id.reshape(-1).cpu()

                fusion_proj, answer_proj, model_outputs = self.model(
                    video_frames,
                    video_mask,
                    answer=answer,
                    answer_id=answer_id,
                    mode="VQA",
                    question=question,
                    qbound_mix=(
                        1.0 if self._qbound_enabled() else 0.0
                    ),
                    answer_only=None,
                )

                main_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=fusion_proj,
                    detach_answer=False,
                )
                predicted = main_logits.argmax(dim=-1).cpu()

                score["acc"] += (
                    predicted == labels_cpu
                ).sum().item()

                time_param = model_outputs.get("time_param")
                qbound_aux = None
                eval_wrapper = None

                if isinstance(time_param, dict):
                    saw_time_param = True
                    eval_wrapper = (
                        None,
                        None,
                        {"time_param": time_param},
                    )
                    qbound_aux = time_param.get(
                        "qbound_aux",
                        time_param.get("gpeb_aux"),
                    )

                if (
                    self._qbound_enabled()
                    and qbound_aux is not None
                ):
                    self._validate_v240_contract(qbound_aux)
                    batch_stats = self._collect_qbound_stats(
                        eval_wrapper
                    )
                    for name, value in batch_stats.items():
                        qbound_stat_sums[name] += (
                            float(value) * batch_size
                        )
                    qbound_stat_count += batch_size

                video_attention = (
                    model_outputs["fatt"]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if (
                    video_attention.ndim == 3
                    and video_attention.shape[-1] == 1
                ):
                    video_attention = video_attention.squeeze(-1)
                if video_attention.ndim == 1:
                    video_attention = video_attention[None, :]

                batch_ground_pre = None
                batch_keyframe = None
                batch_ori_keyframe = None
                batch_qbound_key = None
                batch_qbound_soft_key = None
                batch_cra_key = None

                if isinstance(time_param, dict):
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
                    batch_ground_pre = np.column_stack(
                        (start_time, end_time)
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
                        hard_key = qbound_aux.get("qbound_key")
                        soft_key = qbound_aux.get("qbound_soft_key")
                        c_key = qbound_aux.get("cra_key")

                        if torch.is_tensor(hard_key):
                            batch_qbound_key = (
                                hard_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(soft_key):
                            batch_qbound_soft_key = (
                                self._normalize_distribution(
                                    soft_key.detach()
                                ).cpu().numpy()
                            )
                        if torch.is_tensor(c_key):
                            batch_cra_key = (
                                c_key.detach().cpu().numpy()
                            )

                for batch_index, qid in enumerate(question_id):
                    results[qid] = {
                        "prediction": int(
                            predicted.numpy()[batch_index]
                        ),
                        "answer": int(
                            labels_cpu.numpy()[batch_index]
                        ),
                    }

                    ground_res[qid] = video_attention[
                        batch_index
                    ].tolist()

                    if batch_ground_pre is not None:
                        ground_pre[qid] = batch_ground_pre[
                            batch_index
                        ].tolist()
                    if batch_keyframe is not None:
                        keyframe[qid] = batch_keyframe[
                            batch_index
                        ].tolist()
                    if batch_ori_keyframe is not None:
                        ori_keyframe[qid] = batch_ori_keyframe[
                            batch_index
                        ].tolist()
                    if batch_qbound_key is not None:
                        qbound_keyframe[qid] = batch_qbound_key[
                            batch_index
                        ].tolist()
                    if batch_qbound_soft_key is not None:
                        qbound_soft_keyframe[qid] = (
                            batch_qbound_soft_key[
                                batch_index
                            ].tolist()
                        )
                    if batch_cra_key is not None:
                        cra_keyframe[qid] = batch_cra_key[
                            batch_index
                        ].tolist()

                progress = float(index + 1) / len(data_loader)
                print(
                    f"\rProgress: {progress:.2%}",
                    end="",
                    flush=True,
                )

        if count == 0:
            raise RuntimeError("Evaluation data loader is empty.")

        score["acc"] /= count

        if qbound_stat_count > 0:
            for name, total in qbound_stat_sums.items():
                score[name] = total / qbound_stat_count

        split = data_loader.dataset.split
        csv_path = self.cfgs["dataset"]["csv_path"]

        ground_truth = load_file(
            os.path.join(csv_path, f"gsub_{split}.json")
        )
        segments = load_file(
            os.path.join(csv_path, f"frame2time_{split}.json")
        )
        questions = load_file(
            os.path.join(csv_path, f"{split}.csv")
        )

        post_hoc_ground = self.metric.generate_ground(
            ground_res,
            segments,
            questions,
        )
        score["post-hoc"] = self.metric(
            post_hoc_ground,
            ground_truth,
            results,
            gs=False,
        )

        if qbound_keyframe:
            if not hasattr(self.metric, "temporal_key_metrics"):
                raise AttributeError(
                    "Q-Bound V2.4 requires "
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
            "qbound_soft_key": qbound_soft_keyframe,
            "cra_key": cra_keyframe,
        }

        if saw_time_param and ground_pre:
            hard_metric = self.metric(
                ground_pre,
                ground_truth,
                results,
                gs=True,
            )
            score["pre-hoc"] = hard_metric
            result.update(
                {
                    "tg": ground_pre,
                    "qbound_tg": ground_pre,
                    "key": keyframe,
                    "ori_key": ori_keyframe,
                }
            )

        # Historical CRA post-hoc merge remains diagnostic only.
        if saw_time_param and ground_pre:
            merge_ground = self.metric.combine(
                ground_pre,
                post_hoc_ground,
                ground_truth,
            )
            score["merge"] = self.metric(
                merge_ground,
                ground_truth,
                results,
                gs=False,
            )
            result["merge"] = merge_ground

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
