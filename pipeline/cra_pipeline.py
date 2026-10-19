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
    CRA training/evaluation pipeline for Q-Bound V2.8.0.

    V2.8 has one continuous Support Field and three deliberately separated
    optimization routes:

        L_total = L_CRA/main + lambda_G * L_ground + lambda_P * L_pred

    * L_CRA/main consumes the QA feature pooled with an identical but detached
      Support Field.  It cannot train temporal relevance or extent topology.
    * L_ground aligns differentiable raw-CLIP evidence with a detached,
      question-only target.  The structural graph is detached, so this route
      trains semantic relevance only.
    * L_pred is the sole training signal for the video-only bidirectional
      predictor that builds adjacent support affinities.

    The Question creates a semantic seed; a detached video-only predictive
    graph propagates that seed into support_temporal_prob.  The same numerical
    support is used for QA, grounding, training, and inference.  Analytic
    boundary/posterior-barycenter spans are metadata and evaluation views only:
    there is no hard-MAP temporal decision or learned span scorer.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        self.qbound_version = "2.8.0"

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
                "hard_map_temperature",
                "span_scorer",
                "event_candidate_selector",
            )
            stale = sorted(key for key in stale_keys if key in qbound_cfg)
            if stale:
                raise ValueError(
                    "Q-Bound V2.8 config contains removed legacy keys: "
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
        """Compatibility helper: V2.8 has no CRA -> Q-Bound handover."""
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

    def _qbound_v280_loss_settings(self):
        """Validate the two mandatory V2.8 auxiliary objectives."""
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        def _read_branch(name, default_temperature=None):
            branch = qbound_cfg.get(name, {})
            if branch is None:
                branch = {}
            if not isinstance(branch, dict):
                raise TypeError(
                    f"model.qbound.{name} must be a mapping/dict."
                )

            enabled = bool(branch.get("enabled", True))
            weight = float(branch.get("loss_weight", 1.0))
            if weight < 0.0:
                raise ValueError(
                    f"model.qbound.{name}.loss_weight must be >= 0."
                )
            if not enabled and weight != 0.0:
                raise ValueError(
                    f"model.qbound.{name}.enabled=false requires "
                    "loss_weight=0."
                )

            result = {"enabled": enabled, "weight": weight}
            if default_temperature is not None:
                temperature = float(
                    branch.get("temperature", default_temperature)
                )
                if temperature <= 0.0:
                    raise ValueError(
                        f"model.qbound.{name}.temperature must be > 0."
                    )
                result["temperature"] = temperature
            return result

        grounding = _read_branch("grounding_loss", default_temperature=0.07)
        predictive = _read_branch("predictive_loss")

        if self._qbound_enabled():
            if not grounding["enabled"] or grounding["weight"] <= 0.0:
                raise ValueError(
                    "Canonical Q-Bound V2.8 requires a positive enabled "
                    "model.qbound.grounding_loss."
                )
            if not predictive["enabled"] or predictive["weight"] <= 0.0:
                raise ValueError(
                    "Canonical Q-Bound V2.8 requires a positive enabled "
                    "model.qbound.predictive_loss."
                )

        return {
            "grounding_enabled": grounding["enabled"],
            "grounding_weight": grounding["weight"],
            "grounding_temperature": grounding["temperature"],
            "predictive_enabled": predictive["enabled"],
            "predictive_weight": predictive["weight"],
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
                "Q-Bound V2.8 grounding requires model.decoder."
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
        One-directional in-batch Video->Question semantic grounding.

        The video side is raw CLIP VALUE pooled by the differentiable V2.8
        Support Field.  It is mapped with detached host-projector weights.
        The target is QUESTION-ONLY and detached.  Since the support graph is
        detached upstream, this loss can train semantic relevance but cannot
        train the video-only support predictor or move the question encoder.
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
                "Q-Bound V2.8 grounding interface is missing: "
                + ", ".join(missing)
            )

        pooled_value = model_outputs["qbound_grounding_pooled_value"]
        question_target = model_outputs["qbound_grounding_question"]
        grounding_key = model_outputs["qbound_grounding_key"]

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
        if grounding_key.ndim != 2 or grounding_key.size(0) != pooled_value.size(0):
            raise ValueError("qbound_grounding_key must be [B,T].")
        if question_target.requires_grad:
            raise RuntimeError(
                "V2.8 grounding question target must be stop-gradient."
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

    def _validate_v280_contract(self, gqa_output):
        """Validate the pipeline-facing V2.8 causal and gradient contract."""
        if gqa_output is None or len(gqa_output) < 3:
            raise TypeError("Q-Bound V2.8 expects a 3-item CRA output.")

        main_video_proj = gqa_output[0]
        model_outputs = gqa_output[2]
        if not isinstance(model_outputs, dict):
            raise TypeError("CRA model outputs must be a dict.")

        qbound_aux = self._get_qbound_aux(gqa_output)
        if not isinstance(qbound_aux, dict):
            raise KeyError(
                "Q-Bound V2.8 must expose time_param['qbound_aux']."
            )

        required_aux = {
            "qbound_version",
            "video_mask",
            "semantic_seed_prob",
            "support_temporal_prob",
            "support_edge_mask",
            "support_edge_affinity",
            "support_graph_laplacian",
            "support_predictive_loss",
            "qbound_support_predictive_loss",
            "segment_prob",
            "qbound_qa_key",
            "qbound_grounding_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
            "qbound_qa_video_proj",
            "qbound_grounding_video_proj",
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
            "cra_key",
            "structured_visual_evidence_enabled",
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
            "qbound_same_support_for_qa_and_grounding_enabled",
            "qbound_qa_support_detached_enabled",
            "qbound_grounding_gradient_to_semantic_seed_enabled",
            "qbound_predictive_loss_exposed",
            "visual_enrichment_only_affects_where",
            "question_conditions_relevance_only",
            "video_only_extent_structure",
            "shared_support_for_qa_and_grounding",
            "qa_support_detached",
            "posterior_barycentric_span_is_metadata_only",
            "question_grounding_objective_expected",
            "support_predictive_objective_expected",
            "grounding_question_stop_gradient",
            "question_direct_span_scoring_enabled",
            "support_predictor_target_leakage_enabled",
            "support_branch_uses_question_enabled",
            "grounding_gradient_to_support_graph_enabled",
            "hard_map_temporal_decision_enabled",
            "neural_span_scorer_enabled",
            "answer_features_used_in_core",
            "temporal_gt_used_in_core",
            "answer_candidates_used_in_qbound_temporal_path",
            "vqa_gradient_to_qbound_topology_enabled",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_grounding_gradient_to_support_graph_enabled",
        }
        missing_aux = sorted(required_aux.difference(qbound_aux))
        if missing_aux:
            raise KeyError(
                "Q-Bound V2.8 auxiliary output is missing: "
                + ", ".join(missing_aux)
            )

        required_outputs = {
            "qbound_grounding_question",
            "qbound_grounding_pooled_value",
            "qbound_grounding_video_proj",
            "qbound_grounding_key",
            "qbound_vqa_video_proj",
            "qbound_vqa_key",
            "qbound_support_predictive_loss",
        }
        missing_outputs = sorted(required_outputs.difference(model_outputs))
        if missing_outputs:
            raise KeyError(
                "CRA/Q-Bound V2.8 integration output is missing: "
                + ", ".join(missing_outputs)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "CRAPipeline/CRA/Q-Bound version mismatch: expected "
                f"{self.qbound_version!r}, got "
                f"{qbound_aux['qbound_version']!r}."
            )

        def _flag(name):
            value = qbound_aux[name]
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"{name} must be a scalar flag.")
                value = value.detach().float().item()
            return float(value)

        expected_on = {
            "structured_visual_evidence_enabled",
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
            "qbound_same_support_for_qa_and_grounding_enabled",
            "qbound_qa_support_detached_enabled",
            "qbound_grounding_gradient_to_semantic_seed_enabled",
            "qbound_predictive_loss_exposed",
            "visual_enrichment_only_affects_where",
            "question_conditions_relevance_only",
            "video_only_extent_structure",
            "shared_support_for_qa_and_grounding",
            "qa_support_detached",
            "posterior_barycentric_span_is_metadata_only",
            "question_grounding_objective_expected",
            "support_predictive_objective_expected",
            "grounding_question_stop_gradient",
        }
        expected_off = {
            "question_direct_span_scoring_enabled",
            "support_predictor_target_leakage_enabled",
            "support_branch_uses_question_enabled",
            "grounding_gradient_to_support_graph_enabled",
            "hard_map_temporal_decision_enabled",
            "neural_span_scorer_enabled",
            "answer_features_used_in_core",
            "temporal_gt_used_in_core",
            "answer_candidates_used_in_qbound_temporal_path",
            "vqa_gradient_to_qbound_topology_enabled",
            "qbound_vqa_gradient_to_topology_enabled",
            "qbound_grounding_gradient_to_support_graph_enabled",
        }
        for name in expected_on:
            actual = _flag(name)
            if actual != 1.0:
                raise RuntimeError(
                    f"V2.8 contract violation: {name}={actual}, expected 1."
                )
        for name in expected_off:
            actual = _flag(name)
            if actual != 0.0:
                raise RuntimeError(
                    f"V2.8 contract violation: {name}={actual}, expected 0."
                )

        def _finite(name, value):
            if not torch.is_tensor(value):
                raise TypeError(f"{name} must be a tensor.")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"{name} contains non-finite values.")

        video_mask = qbound_aux["video_mask"].bool()
        if video_mask.ndim != 2 or not torch.all(video_mask.any(dim=-1)):
            raise ValueError("video_mask must be [B,T] with >=1 valid frame.")
        batch_size, time_steps = video_mask.shape

        def _distribution(name, value):
            if value.shape != video_mask.shape:
                raise ValueError(
                    f"{name} must be [B,T]={tuple(video_mask.shape)}, got "
                    f"{tuple(value.shape)}."
                )
            _finite(name, value)
            if torch.any(value.detach() < -1e-6):
                raise RuntimeError(f"{name} contains negative mass.")
            invalid_mass = (
                value.detach() * (~video_mask).to(value.dtype)
            ).abs().amax()
            if float(invalid_mass.item()) > 1e-5:
                raise RuntimeError(f"{name} assigns mass to padding.")
            total = (
                value.detach() * video_mask.to(value.dtype)
            ).sum(dim=-1)
            if not torch.allclose(
                total,
                torch.ones_like(total),
                atol=1e-4,
                rtol=1e-4,
            ):
                raise RuntimeError(f"{name} must sum to one.")

        distribution_names = (
            "semantic_seed_prob",
            "support_temporal_prob",
            "qbound_qa_key",
            "qbound_grounding_key",
            "qbound_training_key",
            "qbound_soft_key",
            "final_key",
        )
        for name in distribution_names:
            _distribution(name, qbound_aux[name])

        support = qbound_aux["support_temporal_prob"].detach()
        for name in distribution_names[2:]:
            if not torch.allclose(
                qbound_aux[name].detach(),
                support,
                atol=1e-5,
                rtol=1e-5,
            ):
                raise RuntimeError(
                    f"{name} diverges from the single V2.8 Support Field."
                )

        for name in ("qbound_qa_key", "final_key"):
            if qbound_aux[name].requires_grad:
                raise RuntimeError(
                    f"{name} must detach support before the CRA/QA route."
                )

        training_with_grad = self.model.training and torch.is_grad_enabled()
        if training_with_grad:
            for name in ("semantic_seed_prob", "qbound_grounding_key"):
                if not qbound_aux[name].requires_grad:
                    raise RuntimeError(
                        f"{name} must remain differentiable for L_ground."
                    )

        edge_steps = max(time_steps - 1, 0)
        edge_mask = qbound_aux["support_edge_mask"].bool()
        if edge_mask.shape != (batch_size, edge_steps):
            raise ValueError("support_edge_mask has the wrong [B,T-1] shape.")
        edge_affinity = qbound_aux["support_edge_affinity"]
        if edge_affinity.shape != edge_mask.shape:
            raise ValueError("support_edge_affinity must match edge_mask.")
        _finite("support_edge_affinity", edge_affinity)
        if edge_affinity.requires_grad:
            raise RuntimeError(
                "L_ground must see a detached support_edge_affinity."
            )

        laplacian = qbound_aux["support_graph_laplacian"]
        if laplacian.shape != (batch_size, time_steps, time_steps):
            raise ValueError("support_graph_laplacian must be [B,T,T].")
        _finite("support_graph_laplacian", laplacian)
        if laplacian.requires_grad:
            raise RuntimeError(
                "L_ground must see a detached support graph Laplacian."
            )

        predictive_loss = qbound_aux["support_predictive_loss"]
        if not torch.is_tensor(predictive_loss) or predictive_loss.numel() != 1:
            raise TypeError("support_predictive_loss must be a scalar tensor.")
        _finite("support_predictive_loss", predictive_loss)
        if training_with_grad and bool(edge_mask.any()) and not predictive_loss.requires_grad:
            raise RuntimeError(
                "L_pred is detached; the support predictor would not learn."
            )
        for alias in (
            qbound_aux["qbound_support_predictive_loss"],
            model_outputs["qbound_support_predictive_loss"],
        ):
            if not torch.allclose(
                alias.detach(),
                predictive_loss.detach(),
                atol=1e-7,
                rtol=1e-7,
            ):
                raise RuntimeError("Predictive-loss aliases disagree.")

        segment_prob = qbound_aux["segment_prob"]
        if segment_prob.shape != (batch_size, time_steps, time_steps):
            raise ValueError("segment_prob must be [B,T,T].")
        _finite("segment_prob", segment_prob)
        segment_total = segment_prob.detach().sum(dim=(-2, -1))
        if not torch.allclose(
            segment_total,
            torch.ones_like(segment_total),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise RuntimeError("Analytic V2.8 segment_prob must sum to one.")

        qa_feature = qbound_aux["qbound_qa_video_proj"]
        grounding_feature = qbound_aux["qbound_grounding_video_proj"]
        for name, value in (
            ("main_video_proj", main_video_proj),
            ("qbound_qa_video_proj", qa_feature),
            ("qbound_grounding_video_proj", grounding_feature),
            ("model_outputs.qbound_vqa_video_proj", model_outputs["qbound_vqa_video_proj"]),
        ):
            _finite(name, value)
            if value.shape != main_video_proj.shape:
                raise ValueError(f"{name} does not match the CRA QA feature.")
        if not torch.allclose(
            main_video_proj.detach(), qa_feature.detach(), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError("CRA must return the detached-support QA view.")
        if not torch.allclose(
            qa_feature.detach(), grounding_feature.detach(), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError(
                "QA and grounding must use the same numerical Support Field."
            )

        for name in (
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
        ):
            value = qbound_aux[name]
            _finite(name, value)
            if float(value.detach().amax().item()) > 1e-5:
                raise RuntimeError(f"{name} violates the shared-support contract.")

        question_target = model_outputs["qbound_grounding_question"]
        _finite("qbound_grounding_question", question_target)
        if question_target.requires_grad:
            raise RuntimeError(
                "The V2.8 question-only grounding target must be detached."
            )

        return qbound_aux

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

        scalar_keys = (
            "visual_distinctiveness_input_ratio",
            "visual_distinctiveness_delta_ratio",
            "visual_transition_delta_ratio",
            "visual_enrichment_delta_ratio",
            "temporal_context_delta_ratio",
            "relevance_question_attention_entropy_ratio",
            "semantic_seed_entropy_ratio",
            "semantic_seed_top1_prob",
            "semantic_seed_effective_frames",
            "support_edge_surprise_mean",
            "support_edge_surprise_std",
            "support_edge_affinity_mean",
            "support_edge_affinity_std",
            "support_predictive_loss_per_sample",
            "support_predictive_loss",
            "support_entropy_ratio",
            "support_top1_prob",
            "support_effective_frames",
            "support_propagation_l1_delta",
            "semantic_seed_support_cosine",
            "start_boundary_entropy_ratio",
            "end_boundary_entropy_ratio",
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
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
            "qbound_soft_training_key_max_abs_diff",
        )
        for key in scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                log_key = key if key.startswith("qbound_") else f"qbound_{key}"
                stats[log_key] = scalar

        edge_mask = qbound_aux.get("support_edge_mask")
        for name in (
            "support_edge_surprise",
            "support_edge_affinity_raw",
            "support_edge_affinity",
        ):
            value = qbound_aux.get(name)
            if not torch.is_tensor(value) or value.numel() == 0:
                continue
            detached = value.detach().float()
            if torch.is_tensor(edge_mask) and edge_mask.shape == detached.shape:
                detached = detached[edge_mask.detach().bool()]
            if detached.numel() == 0:
                continue
            stats[f"qbound_{name}_mean"] = detached.mean().item()
            stats[f"qbound_{name}_std"] = detached.std(unbiased=False).item()
            stats[f"qbound_{name}_min"] = detached.min().item()
            stats[f"qbound_{name}_max"] = detached.max().item()

        video_mask = qbound_aux.get("video_mask")
        mask = video_mask.detach().bool() if torch.is_tensor(video_mask) else None
        seed_key = qbound_aux.get("semantic_seed_prob")
        support_key = qbound_aux.get("support_temporal_prob")
        qa_key = qbound_aux.get("qbound_qa_key")
        grounding_key = qbound_aux.get("qbound_grounding_key")
        cra_key = qbound_aux.get("cra_key")

        def _compare(prefix, first, second):
            if (
                not torch.is_tensor(first)
                or not torch.is_tensor(second)
                or first.shape != second.shape
                or first.ndim != 2
            ):
                return
            first = self._normalize_distribution(first.detach(), mask=mask)
            second = self._normalize_distribution(second.detach(), mask=mask)
            stats[f"qbound_{prefix}_js"] = self._js_divergence(
                first, second, mask=mask
            ).mean().item()
            stats[f"qbound_{prefix}_l1"] = (
                (first - second).abs().sum(dim=-1).mean().item()
            )

        _compare("seed_support", seed_key, support_key)
        _compare("cra_support", cra_key, support_key)
        _compare("qa_grounding_support", qa_key, grounding_key)

        for new_name, support_name, seed_name in (
            (
                "qbound_support_minus_seed_entropy_ratio",
                "support_entropy_ratio",
                "semantic_seed_entropy_ratio",
            ),
            (
                "qbound_support_minus_seed_effective_frames",
                "support_effective_frames",
                "semantic_seed_effective_frames",
            ),
            (
                "qbound_support_minus_seed_top1_prob",
                "support_top1_prob",
                "semantic_seed_top1_prob",
            ),
        ):
            support_value = qbound_aux.get(support_name)
            seed_value = qbound_aux.get(seed_name)
            if torch.is_tensor(support_value) and torch.is_tensor(seed_value):
                stats[new_name] = (
                    support_value.detach().float()
                    - seed_value.detach().float()
                ).mean().item()

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
        names = (
            "where_input",
            "visual_distinctiveness",
            "visual_transition",
            "temporal_context",
            "semantic_relevance",
            "support_predictor",
        )
        totals = {name: 0.0 for name in names}
        if not self._qbound_enabled():
            return {
                f"qbound_{name}_grad_norm": 0.0
                for name in names
            }

        groups = {
            "where_input": ("where_input_proj",),
            "visual_distinctiveness": (
                "visual_distinctiveness_encoder",
            ),
            "visual_transition": (
                "visual_transition_encoder",
                "visual_transition_frame_delta",
            ),
            "temporal_context": (
                "temporal_norm1",
                "temporal_attn",
                "temporal_attn_delta",
                "temporal_norm2",
                "temporal_ff",
            ),
            "semantic_relevance": (
                "question_token_proj",
                "relevance_visual_norm",
                "relevance_question_attn",
                "relevance_question_context_norm",
                "semantic_seed_encoder",
                "semantic_seed_score",
            ),
            "support_predictor": (
                "support_input_proj",
                "support_forward_cell",
                "support_backward_cell",
                "support_forward_predictor",
                "support_backward_predictor",
            ),
        }

        for parameter_name, parameter in self.model.named_parameters():
            if parameter.grad is None or "qbound" not in parameter_name.lower():
                continue
            lower = parameter_name.lower()
            grad_sq = (
                parameter.grad.detach().float().pow(2).sum().item()
            )
            for group_name, tokens in groups.items():
                if any(token in lower for token in tokens):
                    totals[group_name] += grad_sq
                    break

        result = {
            f"qbound_{name}_grad_norm": float(totals[name] ** 0.5)
            for name in names
        }
        relevance_sq = sum(
            totals[name]
            for name in (
                "where_input",
                "visual_distinctiveness",
                "visual_transition",
                "temporal_context",
                "semantic_relevance",
            )
        )
        result["qbound_relevance_path_grad_norm"] = float(
            relevance_sq ** 0.5
        )
        return result

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_grad_norm",
            "qbound_where_input_grad_norm",
            "qbound_visual_distinctiveness_grad_norm",
            "qbound_visual_transition_grad_norm",
            "qbound_temporal_context_grad_norm",
            "qbound_semantic_relevance_grad_norm",
            "qbound_support_predictor_grad_norm",
            "qbound_relevance_path_grad_norm",
            "qbound_grounding_weight",
            "qbound_grounding_temperature",
            "qbound_predictive_weight",
            "qbound_grounding_loss",
            "qbound_grounding_weighted_loss",
            "qbound_predictive_loss",
            "qbound_predictive_weighted_loss",
            "qbound_grounding_batch_size",
            "qbound_grounding_retrieval_acc",
            "qbound_grounding_positive_similarity",
            "qbound_grounding_negative_similarity",
            "qbound_grounding_hardest_negative_similarity",
            "qbound_grounding_margin",
            "qbound_visual_distinctiveness_input_ratio",
            "qbound_visual_distinctiveness_delta_ratio",
            "qbound_visual_transition_delta_ratio",
            "qbound_visual_enrichment_delta_ratio",
            "qbound_temporal_context_delta_ratio",
            "qbound_relevance_question_attention_entropy_ratio",
            "qbound_semantic_seed_entropy_ratio",
            "qbound_semantic_seed_top1_prob",
            "qbound_semantic_seed_effective_frames",
            "qbound_support_edge_surprise_mean",
            "qbound_support_edge_surprise_std",
            "qbound_support_edge_surprise_min",
            "qbound_support_edge_surprise_max",
            "qbound_support_edge_affinity_raw_mean",
            "qbound_support_edge_affinity_raw_std",
            "qbound_support_edge_affinity_raw_min",
            "qbound_support_edge_affinity_raw_max",
            "qbound_support_edge_affinity_mean",
            "qbound_support_edge_affinity_std",
            "qbound_support_edge_affinity_min",
            "qbound_support_edge_affinity_max",
            "qbound_support_predictive_loss_per_sample",
            "qbound_support_predictive_loss",
            "qbound_support_entropy_ratio",
            "qbound_support_top1_prob",
            "qbound_support_effective_frames",
            "qbound_support_propagation_l1_delta",
            "qbound_semantic_seed_support_cosine",
            "qbound_support_minus_seed_entropy_ratio",
            "qbound_support_minus_seed_effective_frames",
            "qbound_support_minus_seed_top1_prob",
            "qbound_start_boundary_entropy_ratio",
            "qbound_end_boundary_entropy_ratio",
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
            "qbound_seed_support_js",
            "qbound_seed_support_l1",
            "qbound_cra_support_js",
            "qbound_cra_support_l1",
            "qbound_qa_grounding_support_js",
            "qbound_qa_grounding_support_l1",
            "qbound_same_support_max_abs_diff",
            "qbound_qa_grounding_feature_max_abs_diff",
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

        settings = self._qbound_v280_loss_settings()
        video_mask = (
            get_mask(video_len, self.cfgs["dataset"]["max_feats"]).to(
                device, non_blocking=True
            )
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_device = self._prepare_labels(answer_id, device)
        prediction_dict = {}
        target_dict = {}

        # Historical CRA VQ objective remains outside Q-Bound.
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
                vq_output[1], vq_video_proj
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

        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
            qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
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

        # Critical V2.8 firewall: the existing CRA alignment objective belongs
        # to L_CRA and therefore consumes the detached-support QA view.  The
        # differentiable grounding view is reserved exclusively for L_ground.
        prediction_dict["vqa_align_pred"] = main_video_proj
        target_dict["vqa_align_gt"] = model_outputs["align_gt"]

        loss = self.criterion(prediction_dict, target_dict)
        if "total_loss" not in loss:
            raise KeyError("The criterion must return loss['total_loss'].")
        if not torch.isfinite(loss["total_loss"]).all():
            raise RuntimeError("L_CRA/main contains non-finite values.")

        qbound_logs = {}
        if self._qbound_enabled():
            qbound_aux = self._validate_v280_contract(gqa_output)

            grounding_loss, grounding_diag = (
                self._qbound_video_question_objective(
                    model_outputs=model_outputs,
                    qbound_aux=qbound_aux,
                    temperature=settings["grounding_temperature"],
                )
            )
            predictive_loss = qbound_aux["support_predictive_loss"]
            if not torch.isfinite(predictive_loss).all():
                raise RuntimeError("L_pred contains non-finite values.")

            weighted_grounding = (
                settings["grounding_weight"] * grounding_loss
            )
            weighted_predictive = (
                settings["predictive_weight"] * predictive_loss
            )
            loss["qbound_grounding_loss"] = grounding_loss
            loss["qbound_grounding_weighted_loss"] = weighted_grounding
            loss["qbound_predictive_loss"] = predictive_loss
            loss["qbound_predictive_weighted_loss"] = weighted_predictive
            loss["total_loss"] = (
                loss["total_loss"]
                + weighted_grounding
                + weighted_predictive
            )

            qbound_logs.update(
                {
                    f"qbound_{name}": value
                    for name, value in grounding_diag.items()
                }
            )
            qbound_logs["qbound_grounding_weight"] = settings[
                "grounding_weight"
            ]
            qbound_logs["qbound_grounding_temperature"] = settings[
                "grounding_temperature"
            ]
            qbound_logs["qbound_predictive_weight"] = settings[
                "predictive_weight"
            ]

        predicted = main_logits.argmax(dim=-1).detach().cpu()
        running_accuracy = (
            predicted == answer_id.reshape(-1).cpu()
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
            for key in self._qbound_monitor_keys():
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

        
        # R1/R2 diagnostic: save validation predictions per epoch.
        # Does not affect forward, loss, gradients, or model weights.
        if self._qbound_enabled():
            dump_dir = os.path.join(self.save_dir, "val_dumps")
            os.makedirs(dump_dir, exist_ok=True)

            torch.save(
                {
                    "epoch": int(epoch),
                    "score": val_score,
                    "result": val_result,
                },
                os.path.join(
                    dump_dir,
                    f"val_epoch_{epoch:02d}.pth",
                ),
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
                self.monitor.log_info(f"{score_dict[key]:.1f} \t ")

        for split in ("val", "test"):
            acc_key = f"{split}_acc"
            if acc_key in score:
                self.monitor.log_info(
                    f"\n{split.title()} Set\t "
                    f"Q-Bound V2.8 acc: {score[acc_key]:.2%}"
                )

            prehoc_key = f"{split}_pre-hoc"
            if prehoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} V2.8 posterior-barycentric =========="
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
            if key.startswith(("val_qbound_", "test_qbound_"))
        ]
        if qbound_keys:
            self.monitor.log_info(
                "\n================ Q-Bound V2.8 diagnostics ================"
            )
            for key in sorted(qbound_keys):
                self.monitor.log_info(f"{key}: {score[key]:.6f}")

        temporal_prefixes = (
            "val_QBoundSeed",
            "test_QBoundSeed",
            "val_QBoundSupport",
            "test_QBoundSupport",
            "val_CRA-",
            "test_CRA-",
        )
        temporal_keys = [
            key for key in score if key.startswith(temporal_prefixes)
        ]
        if temporal_keys:
            self.monitor.log_info(
                "\n============ Temporal support faithfulness ============"
            )
            for key in sorted(temporal_keys):
                self.monitor.log_info(f"{key}: {score[key]:.6f}")

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
        qbound_seed_keyframe = {}
        qbound_support_keyframe = {}
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
                if not hasattr(model_core, "_compute_answer_embedding"):
                    raise AttributeError(
                        "Non-multiple-choice evaluation requires "
                        "_compute_answer_embedding on the model."
                    )
                model_core._compute_answer_embedding(self.a2v)

            for index, batch in enumerate(data_loader):
                answer_id = batch["answer_id"]
                answer = batch["answer"].to(device, non_blocking=True)
                video_frames = batch["video_frames"].to(
                    device, non_blocking=True
                )
                question = batch["question"].to(
                    device, non_blocking=True
                )
                question_id = batch["question_id"]
                video_len = batch["video_len"]
                video_mask = get_mask(
                    video_len, video_frames.size(1)
                ).to(device, non_blocking=True)

                batch_size = answer_id.size(0)
                count += batch_size
                labels_cpu = answer_id.reshape(-1).cpu()

                gqa_output = self.model(
                    video_frames,
                    video_mask,
                    answer=answer,
                    answer_id=answer_id,
                    mode="VQA",
                    question=question,
                    qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
                    answer_only=None,
                )
                fusion_proj, answer_proj, model_outputs = gqa_output
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
                qbound_aux = self._get_qbound_aux(gqa_output)
                if isinstance(time_param, dict):
                    saw_time_param = True

                if self._qbound_enabled():
                    qbound_aux = self._validate_v280_contract(gqa_output)
                    batch_stats = self._collect_qbound_stats(gqa_output)
                    for name, value in batch_stats.items():
                        qbound_stat_sums[name] += float(value) * batch_size
                    qbound_stat_count += batch_size

                video_attention = model_outputs["fatt"].detach().cpu().numpy()
                if video_attention.ndim == 3 and video_attention.shape[-1] == 1:
                    video_attention = video_attention.squeeze(-1)
                if video_attention.ndim == 1:
                    video_attention = video_attention[None, :]

                batch_ground_pre = None
                batch_keyframe = None
                batch_ori_keyframe = None
                batch_seed_key = None
                batch_support_key = None
                batch_cra_key = None

                if isinstance(time_param, dict):
                    start_time = time_param["start"].detach().cpu().numpy()
                    end_time = time_param["end"].detach().cpu().numpy()
                    batch_ground_pre = np.column_stack((start_time, end_time))
                    batch_keyframe = time_param["key"].detach().cpu().numpy()
                    batch_ori_keyframe = (
                        time_param["ori_key"].detach().cpu().numpy()
                    )

                if qbound_aux is not None:
                    seed_key = qbound_aux.get("semantic_seed_prob")
                    support_key = qbound_aux.get("support_temporal_prob")
                    cra_key = qbound_aux.get("cra_key")
                    mask = qbound_aux.get("video_mask")
                    if torch.is_tensor(seed_key):
                        batch_seed_key = self._normalize_distribution(
                            seed_key.detach(), mask=mask
                        ).cpu().numpy()
                    if torch.is_tensor(support_key):
                        batch_support_key = self._normalize_distribution(
                            support_key.detach(), mask=mask
                        ).cpu().numpy()
                    if torch.is_tensor(cra_key):
                        batch_cra_key = self._normalize_distribution(
                            cra_key.detach(), mask=mask
                        ).cpu().numpy()

                predicted_np = predicted.numpy()
                labels_np = labels_cpu.numpy()
                for batch_index, qid in enumerate(question_id):
                    results[qid] = {
                        "prediction": int(predicted_np[batch_index]),
                        "answer": int(labels_np[batch_index]),
                    }
                    ground_res[qid] = video_attention[batch_index].tolist()
                    if batch_ground_pre is not None:
                        ground_pre[qid] = batch_ground_pre[batch_index].tolist()
                    if batch_keyframe is not None:
                        keyframe[qid] = batch_keyframe[batch_index].tolist()
                    if batch_ori_keyframe is not None:
                        ori_keyframe[qid] = batch_ori_keyframe[batch_index].tolist()
                    if batch_seed_key is not None:
                        qbound_seed_keyframe[qid] = (
                            batch_seed_key[batch_index].tolist()
                        )
                    if batch_support_key is not None:
                        qbound_support_keyframe[qid] = (
                            batch_support_key[batch_index].tolist()
                        )
                    if batch_cra_key is not None:
                        cra_keyframe[qid] = batch_cra_key[batch_index].tolist()

                progress = float(index + 1) / len(data_loader)
                print(f"\rProgress: {progress:.2%}", end="", flush=True)

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
        questions = load_file(os.path.join(csv_path, f"{split}.csv"))

        post_hoc_ground = self.metric.generate_ground(
            ground_res, segments, questions
        )
        score["post-hoc"] = self.metric(
            post_hoc_ground, ground_truth, results, gs=False
        )

        if qbound_support_keyframe:
            if not hasattr(self.metric, "temporal_key_metrics"):
                raise AttributeError(
                    "Q-Bound V2.8 requires Metric.temporal_key_metrics()."
                )
            seed_metrics = self.metric.temporal_key_metrics(
                qbound_seed_keyframe,
                ground_truth,
                segments,
                prefix="QBoundSeed",
            )
            support_metrics = self.metric.temporal_key_metrics(
                qbound_support_keyframe,
                ground_truth,
                segments,
                prefix="QBoundSupport",
            )
            score.update(seed_metrics)
            score.update(support_metrics)

            # Directly expose whether structural propagation improves each
            # temporal metric over the semantic seed.
            for support_name, support_value in support_metrics.items():
                suffix = support_name[len("QBoundSupport"):]
                seed_name = f"QBoundSeed{suffix}"
                seed_value = seed_metrics.get(seed_name)
                if isinstance(support_value, (int, float)) and isinstance(
                    seed_value, (int, float)
                ):
                    score[f"QBoundSupportGain{suffix}"] = (
                        float(support_value) - float(seed_value)
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
            "qbound_seed_key": qbound_seed_keyframe,
            "qbound_support_key": qbound_support_keyframe,
            # Compatibility: qbound_key now means the continuous support.
            "qbound_key": qbound_support_keyframe,
            "cra_key": cra_keyframe,
        }

        if saw_time_param and ground_pre:
            score["pre-hoc"] = self.metric(
                ground_pre, ground_truth, results, gs=True
            )
            result.update(
                {
                    "tg": ground_pre,
                    "qbound_tg": ground_pre,
                    "key": keyframe,
                    "ori_key": ori_keyframe,
                }
            )

            merge_ground = self.metric.combine(
                ground_pre, post_hoc_ground, ground_truth
            )
            score["merge"] = self.metric(
                merge_ground, ground_truth, results, gs=False
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
