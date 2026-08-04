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
    CRA training pipeline with Q-Bound GPEB Ver1.0.

    Ver1.0 removes the fixed-bin middle/end reconstruction objective.
    Learned boundaries are trained by counterfactual necessity:

        A boundary should be strong only when removing it naturally harms
        answer discrimination or grounding agreement.

    The counterfactual harm is detached before it becomes a supervision
    target. This prevents the model from gaming the objective by
    deliberately making the merged counterfactual worse.

    Split counterfactuals are trained to preserve answer and grounding
    predictions, encouraging stability inside an event.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)
        self._active_epoch = 0

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _model_device(self):
        return next(self.model.parameters()).device

    def _get_gpeb_aux(self, gqa_output):
        """Safely extract Q-Bound outputs from a CRA VQA forward."""
        if not self.cfgs["model"].get("gpeb_enabled", False):
            return None

        if gqa_output is None or len(gqa_output) < 3:
            return None

        model_outputs = gqa_output[2]
        if not isinstance(model_outputs, dict):
            return None

        time_param = model_outputs.get("time_param")
        if time_param is None:
            return None

        return time_param.get("gpeb_aux")

    @staticmethod
    def _prepare_labels(answer_id, device):
        labels = answer_id.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).reshape(-1)
        return labels

    def _score_answer_candidates(
        self,
        answer_proj,
        representation,
        detach_answer=True,
        normalize=False,
        logit_scale=1.0,
    ):
        """
        Score answer candidates with one or more representations.

        Args:
            answer_proj:
                [B, C_total, D]. C_total may include proposal copies.

            representation:
                [B, D] or [B, M, D].

        Returns:
            [B, C] for a single representation.
            [B, M, C] for M counterfactual representations.
        """
        if answer_proj.ndim != 3:
            raise ValueError(
                "answer_proj must be [B, C, D], got "
                f"{tuple(answer_proj.shape)}."
            )

        squeeze_counterfactual_dim = False
        if representation.ndim == 2:
            representation = representation.unsqueeze(1)
            squeeze_counterfactual_dim = True
        elif representation.ndim != 3:
            raise ValueError(
                "representation must be [B, D] or [B, M, D], got "
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

        # [B, M, C_total]
        logits = (
            float(logit_scale)
            * torch.einsum(
                "bcd,bmd->bmc",
                candidate_features,
                representation,
            )
        )

        prop_num = int(
            self.cfgs["model"].get("prop_num", 1)
        )
        if prop_num < 1:
            raise ValueError("model.prop_num must be positive.")

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

        if squeeze_counterfactual_dim:
            logits = logits.squeeze(1)

        return logits

    @staticmethod
    def _normalize_distribution(
        values,
        mask=None,
        eps=1e-8,
    ):
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
    def _js_divergence(
        cls,
        first,
        second,
        mask=None,
        eps=1e-8,
    ):
        """
        Jensen-Shannon divergence along the last dimension.

        Broadcasting is allowed, e.g.:
            first:  [B, M, T]
            second: [B, 1, T]
        """
        first = cls._normalize_distribution(
            first,
            mask=mask,
            eps=eps,
        )
        second = cls._normalize_distribution(
            second,
            mask=mask,
            eps=eps,
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

    def _counterfactual_scale(self):
        """
        Warm up ordinary CRA/GPEB learning before activating structural
        counterfactual supervision.
        """
        model_cfg = self.cfgs["model"]

        start_epoch = int(
            model_cfg.get(
                "gpeb_counterfactual_start_epoch",
                2,
            )
        )
        ramp_epochs = int(
            model_cfg.get(
                "gpeb_counterfactual_ramp_epochs",
                3,
            )
        )

        if self._active_epoch <= start_epoch:
            return 0.0

        if ramp_epochs <= 0:
            return 1.0

        return min(
            1.0,
            float(self._active_epoch - start_epoch)
            / float(ramp_epochs),
        )

    # ------------------------------------------------------------------
    # Q-Bound Ver1.0 losses
    # ------------------------------------------------------------------
    def _compute_gpeb_v1_losses(
        self,
        gqa_output,
        answer_proj,
        answer_id,
    ):
        """
        Compute Ver1.0 objectives.

        The boundary-necessity target is derived from the *observed*
        counterfactual harm and then detached:

            harm = answer degradation + grounding degradation

        Boundary strength is calibrated to this target with BCE. Therefore,
        the objective does not explicitly push the merged branch to become
        worse; it only preserves boundaries whose removal is already harmful.
        """
        gpeb_aux = self._get_gpeb_aux(gqa_output)
        if gpeb_aux is None:
            return {}, {}

        required_keys = [
            "bridge_repr",
            "merge_bridge_repr",
            "split_bridge_repr",
            "event_key",
            "cra_key",
            "merge_frame_relevance",
            "split_frame_relevance",
            "boundary_mass",
            "merge_selector",
            "boundary_count",
            "effective_event_count",
            "concurrent_degree_mean",
        ]
        missing = [
            key for key in required_keys
            if key not in gpeb_aux
        ]
        if missing:
            raise KeyError(
                "Missing Q-Bound Ver1.0 outputs: "
                + ", ".join(missing)
            )

        model_cfg = self.cfgs["model"]
        detach_answer = bool(
            model_cfg.get(
                "gpeb_aux_detach_answer",
                True,
            )
        )
        auxiliary_logit_scale = float(
            model_cfg.get(
                "gpeb_aux_logit_scale",
                10.0,
            )
        )
        if auxiliary_logit_scale <= 0.0:
            raise ValueError(
                "gpeb_aux_logit_scale must be positive."
            )

        labels = self._prepare_labels(
            answer_id,
            answer_proj.device,
        )

        original_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=gpeb_aux["bridge_repr"],
            detach_answer=detach_answer,
            normalize=True,
            logit_scale=auxiliary_logit_scale,
        )
        merge_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=gpeb_aux["merge_bridge_repr"],
            detach_answer=detach_answer,
            normalize=True,
            logit_scale=auxiliary_logit_scale,
        )
        split_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=gpeb_aux["split_bridge_repr"],
            detach_answer=detach_answer,
            normalize=True,
            logit_scale=auxiliary_logit_scale,
        )

        if original_logits.size(0) != labels.numel():
            raise ValueError(
                "answer_id size does not match GPEB logits."
            )

        # --------------------------------------------------------------
        # 1. Original bridge answer supervision
        # --------------------------------------------------------------
        auxiliary_qa_loss = F.cross_entropy(
            original_logits,
            labels,
        )
        auxiliary_prediction = original_logits.argmax(
            dim=-1
        )
        auxiliary_accuracy = (
            auxiliary_prediction == labels
        ).float().mean()

        # --------------------------------------------------------------
        # 2. Counterfactual merge necessity
        # --------------------------------------------------------------
        original_ce = F.cross_entropy(
            original_logits,
            labels,
            reduction="none",
        )

        num_merge = merge_logits.size(1)
        merge_labels = labels.unsqueeze(1).expand(
            -1,
            num_merge,
        )
        merge_ce = F.cross_entropy(
            merge_logits.reshape(
                -1,
                merge_logits.size(-1),
            ),
            merge_labels.reshape(-1),
            reduction="none",
        ).reshape(
            labels.size(0),
            num_merge,
        ).mean(dim=1)

        cra_key = gpeb_aux["cra_key"].detach()
        event_key = gpeb_aux["event_key"]

        original_ground_distance = self._js_divergence(
            event_key,
            cra_key,
        )

        merge_frame_relevance = gpeb_aux[
            "merge_frame_relevance"
        ]
        merge_ground_distance = self._js_divergence(
            merge_frame_relevance,
            cra_key.unsqueeze(1),
        ).mean(dim=1)

        answer_harm = merge_ce - original_ce
        grounding_harm = (
            merge_ground_distance
            - original_ground_distance
        )

        grounding_harm_scale = float(
            model_cfg.get(
                "gpeb_merge_ground_harm_scale",
                1.0,
            )
        )
        total_harm = (
            answer_harm
            + grounding_harm_scale * grounding_harm
        )

        harm_temperature = float(
            model_cfg.get(
                "gpeb_necessity_temperature",
                0.25,
            )
        )
        if harm_temperature <= 0.0:
            raise ValueError(
                "gpeb_necessity_temperature must be positive."
            )

        # Zero harm -> target 0. Increasing positive harm approaches 1.
        necessity_target = (
            1.0
            - torch.exp(
                -F.relu(total_harm)
                / harm_temperature
            )
        ).detach()

        merge_selector = gpeb_aux[
            "merge_selector"
        ].squeeze(1)
        boundary_mass = gpeb_aux["boundary_mass"]

        selected_boundary_strength = torch.sum(
            merge_selector * boundary_mass,
            dim=-1,
        ).clamp(
            min=1e-6,
            max=1.0 - 1e-6,
        )

        merge_necessity_loss = F.binary_cross_entropy(
            selected_boundary_strength,
            necessity_target,
        )

        # --------------------------------------------------------------
        # 3. Counterfactual split invariance
        # --------------------------------------------------------------
        distill_temperature = float(
            model_cfg.get(
                "gpeb_split_distill_temperature",
                1.0,
            )
        )
        if distill_temperature <= 0.0:
            raise ValueError(
                "gpeb_split_distill_temperature must be positive."
            )

        original_distribution = F.softmax(
            original_logits.detach()
            / distill_temperature,
            dim=-1,
        )
        split_log_distribution = F.log_softmax(
            split_logits
            / distill_temperature,
            dim=-1,
        )

        split_answer_kl = F.kl_div(
            split_log_distribution,
            original_distribution.unsqueeze(1).expand_as(
                split_log_distribution
            ),
            reduction="none",
        ).sum(dim=-1).mean() * (
            distill_temperature ** 2
        )

        split_ground_distance = self._js_divergence(
            gpeb_aux["split_frame_relevance"],
            event_key.detach().unsqueeze(1),
        ).mean()

        split_ground_weight = float(
            model_cfg.get(
                "gpeb_split_ground_weight",
                1.0,
            )
        )
        split_invariance_loss = (
            split_answer_kl
            + split_ground_weight
            * split_ground_distance
        )

        # --------------------------------------------------------------
        # 4. Grounding preservation
        # --------------------------------------------------------------
        ground_consistency_loss = (
            original_ground_distance.mean()
        )

        # --------------------------------------------------------------
        # 5. Minimum sufficient event structure
        # --------------------------------------------------------------
        max_events = int(
            model_cfg.get(
                "gpeb_num_bins",
                model_cfg.get(
                    "gpeb_max_events",
                    4,
                ),
            )
        )
        max_boundaries = max(max_events - 1, 1)

        boundary_sparsity_loss = (
            gpeb_aux["boundary_count"].float().mean()
            / float(max_boundaries)
        )

        minimum_effective_events = float(
            model_cfg.get(
                "gpeb_min_effective_events",
                1.5,
            )
        )
        effective_event_count = gpeb_aux[
            "effective_event_count"
        ].float()

        minimum_event_loss = F.relu(
            minimum_effective_events
            - effective_event_count
        ).pow(2).mean()

        # Concurrent evidence is taxed so it is used only when the primary
        # monotonic chain cannot explain the useful context.
        concurrent_sparsity_loss = gpeb_aux[
            "concurrent_degree_mean"
        ].float().mean()

        losses = {
            "gpeb_aux_qa_loss": auxiliary_qa_loss,
            "gpeb_merge_necessity_loss": (
                merge_necessity_loss
            ),
            "gpeb_split_invariance_loss": (
                split_invariance_loss
            ),
            "gpeb_ground_consistency_loss": (
                ground_consistency_loss
            ),
            "gpeb_boundary_sparsity_loss": (
                boundary_sparsity_loss
            ),
            "gpeb_min_event_loss": minimum_event_loss,
            "gpeb_concurrent_sparsity_loss": (
                concurrent_sparsity_loss
            ),
        }

        logs = {
            "gpeb_aux_acc": (
                auxiliary_accuracy.detach().item()
            ),
            "gpeb_original_aux_ce": (
                original_ce.detach().mean().item()
            ),
            "gpeb_merge_aux_ce": (
                merge_ce.detach().mean().item()
            ),
            "gpeb_answer_harm": (
                answer_harm.detach().mean().item()
            ),
            "gpeb_original_ground_js": (
                original_ground_distance.detach().mean().item()
            ),
            "gpeb_merge_ground_js": (
                merge_ground_distance.detach().mean().item()
            ),
            "gpeb_grounding_harm": (
                grounding_harm.detach().mean().item()
            ),
            "gpeb_necessity_target": (
                necessity_target.mean().item()
            ),
            "gpeb_selected_boundary_strength": (
                selected_boundary_strength.detach().mean().item()
            ),
            "gpeb_split_answer_kl": (
                split_answer_kl.detach().item()
            ),
            "gpeb_split_ground_js": (
                split_ground_distance.detach().item()
            ),
            "gpeb_counterfactual_scale": (
                self._counterfactual_scale()
            ),
            "gpeb_aux_logit_scale": auxiliary_logit_scale,
        }

        return losses, logs

    def _weighted_add_gpeb_losses(
        self,
        loss,
        gpeb_losses,
    ):
        if not gpeb_losses:
            return

        model_cfg = self.cfgs["model"]
        counterfactual_scale = self._counterfactual_scale()

        weights = {
            "gpeb_aux_qa_loss": float(
                model_cfg.get(
                    "gpeb_aux_qa_weight",
                    0.05,
                )
            ),
            "gpeb_merge_necessity_loss": (
                counterfactual_scale
                * float(
                    model_cfg.get(
                        "gpeb_merge_necessity_weight",
                        0.10,
                    )
                )
            ),
            "gpeb_split_invariance_loss": (
                counterfactual_scale
                * float(
                    model_cfg.get(
                        "gpeb_split_invariance_weight",
                        0.05,
                    )
                )
            ),
            "gpeb_ground_consistency_loss": float(
                model_cfg.get(
                    "gpeb_ground_consistency_weight",
                    0.10,
                )
            ),
            "gpeb_boundary_sparsity_loss": (
                counterfactual_scale
                * float(
                    model_cfg.get(
                        "gpeb_boundary_sparsity_weight",
                        0.005,
                    )
                )
            ),
            # Active from the beginning to prevent one-event collapse.
            "gpeb_min_event_loss": float(
                model_cfg.get(
                    "gpeb_min_event_weight",
                    0.01,
                )
            ),
            "gpeb_concurrent_sparsity_loss": float(
                model_cfg.get(
                    "gpeb_concurrent_sparsity_weight",
                    0.002,
                )
            ),
        }

        for name, value in gpeb_losses.items():
            loss[name] = value
            loss["total_loss"] = (
                loss["total_loss"]
                + weights[name] * value
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _scalar_mean(values):
        if values is None:
            return None

        if not torch.is_tensor(values):
            return float(values)

        if values.numel() == 0:
            return 0.0

        return (
            values.detach()
            .float()
            .mean()
            .item()
        )

    @staticmethod
    def _entropy(weights, dim=-1):
        weights = weights.clamp_min(1e-8)
        return -torch.sum(
            weights * torch.log(weights),
            dim=dim,
        )

    def _collect_gpeb_stats(
        self,
        gqa_output,
        extra_logs=None,
    ):
        stats = {}

        if extra_logs:
            stats.update(extra_logs)

        gpeb_aux = self._get_gpeb_aux(gqa_output)
        if gpeb_aux is None:
            return stats

        scalar_keys = [
            "residual_ratio_mean",
            "residual_ratio_unclipped_mean",
            "residual_scale_mean",
            "residual_max_ratio",
            "branch_cosine_mean",
            "cra_branch_cosine_mean",
            "grounding_scale_used",
            "boundary_prob_mean",
            "boundary_count",
            "assignment_entropy",
            "effective_event_count",
            "token_entropy",
            "concurrent_degree_mean",
            "concurrent_frame_fraction",
            "concurrent_logit_scale_used",
            "concurrent_bias_used",
            "concurrent_context_scale_used",
            "grounding_agreement_mean",
            "ground_refine_mix_mean",
            "grounding_shift_mean",
            "ground_refinement_enabled",
            "ground_refine_max",
            "question_only_conditioning",
        ]

        for key in scalar_keys:
            scalar = self._scalar_mean(
                gpeb_aux.get(key)
            )
            if scalar is not None:
                stats[f"gpeb_{key}"] = scalar

        boundary_prob = gpeb_aux.get("boundary_prob")
        boundary_mask = gpeb_aux.get("boundary_mask")
        if (
            torch.is_tensor(boundary_prob)
            and torch.is_tensor(boundary_mask)
        ):
            valid_boundary = boundary_prob[
                boundary_mask.bool()
            ]
            if valid_boundary.numel() > 0:
                stats["gpeb_boundary_prob_min"] = (
                    valid_boundary.detach().float().min().item()
                )
                stats["gpeb_boundary_prob_max"] = (
                    valid_boundary.detach().float().max().item()
                )

        event_mass = gpeb_aux.get("event_mass")
        if torch.is_tensor(event_mass):
            stats["gpeb_event_mass_min"] = (
                event_mass.detach().float().min(dim=-1)
                .values.mean().item()
            )
            stats["gpeb_event_mass_max"] = (
                event_mass.detach().float().max(dim=-1)
                .values.mean().item()
            )

        fusion_delta = gpeb_aux.get("fusion_delta")
        if torch.is_tensor(fusion_delta):
            stats["gpeb_fusion_delta_norm"] = (
                fusion_delta.detach().float()
                .norm(dim=-1).mean().item()
            )

        merge_selector = gpeb_aux.get(
            "merge_selector"
        )
        if torch.is_tensor(merge_selector):
            stats["gpeb_merge_selector_entropy"] = (
                self._entropy(
                    merge_selector.squeeze(1),
                    dim=-1,
                ).detach().float().mean().item()
            )

        split_selector = gpeb_aux.get(
            "split_selector"
        )
        if torch.is_tensor(split_selector):
            stats["gpeb_split_selector_entropy"] = (
                self._entropy(
                    split_selector.squeeze(1),
                    dim=-1,
                ).detach().float().mean().item()
            )

        return stats

    @staticmethod
    def _gpeb_loss_keys():
        return [
            "gpeb_aux_qa_loss",
            "gpeb_merge_necessity_loss",
            "gpeb_split_invariance_loss",
            "gpeb_ground_consistency_loss",
            "gpeb_boundary_sparsity_loss",
            "gpeb_min_event_loss",
            "gpeb_concurrent_sparsity_loss",
        ]

    @staticmethod
    def _gpeb_monitor_keys():
        return [
            "gpeb_aux_acc",
            "gpeb_original_aux_ce",
            "gpeb_merge_aux_ce",
            "gpeb_answer_harm",
            "gpeb_original_ground_js",
            "gpeb_merge_ground_js",
            "gpeb_grounding_harm",
            "gpeb_necessity_target",
            "gpeb_selected_boundary_strength",
            "gpeb_split_answer_kl",
            "gpeb_split_ground_js",
            "gpeb_counterfactual_scale",
            "gpeb_aux_logit_scale",
            "gpeb_residual_ratio_mean",
            "gpeb_residual_ratio_unclipped_mean",
            "gpeb_residual_scale_mean",
            "gpeb_residual_max_ratio",
            "gpeb_branch_cosine_mean",
            "gpeb_cra_branch_cosine_mean",
            "gpeb_grounding_scale_used",
            "gpeb_boundary_prob_mean",
            "gpeb_boundary_count",
            "gpeb_boundary_prob_min",
            "gpeb_boundary_prob_max",
            "gpeb_assignment_entropy",
            "gpeb_effective_event_count",
            "gpeb_token_entropy",
            "gpeb_event_mass_min",
            "gpeb_event_mass_max",
            "gpeb_concurrent_degree_mean",
            "gpeb_concurrent_frame_fraction",
            "gpeb_concurrent_logit_scale_used",
            "gpeb_concurrent_bias_used",
            "gpeb_concurrent_context_scale_used",
            "gpeb_grounding_agreement_mean",
            "gpeb_ground_refine_mix_mean",
            "gpeb_grounding_shift_mean",
            "gpeb_ground_refinement_enabled",
            "gpeb_ground_refine_max",
            "gpeb_question_only_conditioning",
            "gpeb_fusion_delta_norm",
            "gpeb_merge_selector_entropy",
            "gpeb_split_selector_entropy",
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def get_loss(self, batch):
        device = self._model_device()

        answer_id = batch["answer_id"]
        answer = batch["answer"].to(
            device,
            non_blocking=True,
        )
        video_frames = batch["video_frames"].to(
            device,
            non_blocking=True,
        )
        question = batch["question"].to(
            device,
            non_blocking=True,
        )

        qsns_id = batch["qsns_id"]
        qsns_token_ids = batch["qsns_token_ids"]
        video_len = batch["video_len"]

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

        # Original CRA VQ objective remains unchanged.
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
            target_dict["vq_target"] = (
                qsns_id.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                ).reshape(-1)
            )
            prediction_dict["vq_align_pred"] = (
                vq_output[2]["time_param"][
                    "grounding_feature"
                ]
            )
            target_dict["vq_align_gt"] = (
                vq_output[2]["align_gt"]
            )

        # Ver1.0 explicitly passes question-only tokens to Q-Bound.
        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
        )

        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]

        predicts = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=main_video_proj,
            detach_answer=False,
        )

        prediction_dict["vqa_pred"] = predicts
        target_dict["vqa_target"] = answer_id_device
        prediction_dict["vqa_align_pred"] = (
            gqa_output[2]["time_param"][
                "grounding_feature"
            ]
        )
        target_dict["vqa_align_gt"] = (
            gqa_output[2]["align_gt"]
        )

        loss = self.criterion(
            prediction_dict,
            target_dict,
        )

        gpeb_losses, gpeb_logs = (
            self._compute_gpeb_v1_losses(
                gqa_output=gqa_output,
                answer_proj=answer_proj,
                answer_id=answer_id,
            )
        )
        self._weighted_add_gpeb_losses(
            loss,
            gpeb_losses,
        )

        predicted = predicts.argmax(
            dim=-1
        ).detach().cpu()

        running_accuracy = (
            predicted == answer_id.reshape(-1).cpu()
        ).float().mean().item()

        gpeb_stats = self._collect_gpeb_stats(
            gqa_output,
            extra_logs=gpeb_logs,
        )

        return loss, running_accuracy, gpeb_stats

    def _train_epoch(self, epoch):
        self._active_epoch = int(epoch)

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
        self.monitor.reset_kv("total_loss")

        for key in self.cfgs["optim"]["loss"].keys():
            if key in {"ce", "align"}:
                continue
            self.monitor.reset_kv(f"{key}_loss")

        if self.cfgs["model"].get(
            "gpeb_enabled",
            False,
        ):
            for key in self._gpeb_loss_keys():
                self.monitor.reset_kv(key)
            for key in self._gpeb_monitor_keys():
                self.monitor.reset_kv(key)

        for index, batch in enumerate(
            self.train_dataloader
        ):
            start_time = time.time()

            loss, monitor_metric, gpeb_stats = (
                self.get_loss(batch)
            )

            self.monitor.logkv_mean(
                "acc",
                monitor_metric,
            )
            for name, value in gpeb_stats.items():
                self.monitor.logkv_mean(
                    name,
                    value,
                )

            self.optimizer.zero_grad()
            loss["total_loss"].backward()

            clip_value = self.cfgs["optim"].get(
                "clip",
                0,
            )
            if clip_value:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip_value,
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            progress = float(index + 1) / len(
                self.train_dataloader
            )
            end_time = time.time()
            _, remaining_batch_time, remaining_epoch_time = (
                get_remain_time(
                    start_time,
                    end_time,
                    index,
                    len(self.train_dataloader),
                    epoch,
                    self.cfgs["optim"]["epochs"],
                )
            )

            print_info = (
                f"Progress: {progress:3.2%} "
                f"(rbt-ret: {remaining_batch_time}-"
                f"{remaining_epoch_time}) | "
                f"Train "
                f"{self.cfgs['stat']['monitor']['metric']}: "
                f"{monitor_metric:3.2%}"
            )

            for loss_name, loss_value in loss.items():
                running_loss = (
                    loss_value.detach()
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
        for name, value in self.monitor.name2val.items():
            table[0].append(f"mean {name}")
            table[1].append(f"{value:.4f}")

        self.monitor.log_info(
            f"\n{tabulate(table, headers='firstrow', tablefmt='grid')}",
            end="\n",
        )

        print("Eval Stage")
        val_score, val_result = self.eval(
            self.val_dataloader
        )

        # Retained for compatibility with the current training/checkpoint
        # loop. For final experiments, select checkpoints by validation and
        # report test only once.
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
            test_result = val_result
        else:
            test_score, test_result = self.eval(
                self.test_dataloader
            )

        score = {
            **{
                f"val_{key}": value
                for key, value in val_score.items()
            },
            **{
                f"test_{key}": value
                for key, value in test_score.items()
            },
        }

        self.print_score(score)

        wandb_data = {"epoch": epoch}

        for name, value in self.monitor.name2val.items():
            wandb_data[f"train/{name}"] = value

        for split_name, split_dict in [
            ("val", val_score),
            ("test", test_score),
        ]:
            for key, value in split_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb_data[
                            f"{split_name}/{key}_{sub_key}"
                        ] = sub_value
                else:
                    wandb_data[
                        f"{split_name}/{key}"
                    ] = value

        wandb.log(wandb_data)

        return score, test_result

    # ------------------------------------------------------------------
    # Reporting and visualization
    # ------------------------------------------------------------------
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

        self.monitor.log_info(
            f"\nVal Set\t mean acc: "
            f"{score['val_acc']:.2%}"
        )

        if "val_pre-hoc" in score:
            self.monitor.log_info(
                "\n================ val_attn ================"
            )
            _print_score(score["val_pre-hoc"])

        self.monitor.log_info(
            "\n================ val_post-hoc =================="
        )
        _print_score(score["val_post-hoc"])

        if "val_merge" in score:
            self.monitor.log_info(
                "\n================== val_merge ==================="
            )
            _print_score(score["val_merge"])

        self.monitor.log_info(
            f"\nTest Set\t mean acc: "
            f"{score['test_acc']:.2%}"
        )

        if "test_pre-hoc" in score:
            self.monitor.log_info(
                "\n============== test_attn ================"
            )
            _print_score(score["test_pre-hoc"])

        self.monitor.log_info(
            "\n================ test_post-hoc ================"
        )
        _print_score(score["test_post-hoc"])

        if "test_merge" in score:
            self.monitor.log_info(
                "\n================= test_merge =================="
            )
            _print_score(score["test_merge"])

        gpeb_keys = [
            key for key in score
            if key.startswith("val_gpeb_")
            or key.startswith("test_gpeb_")
        ]

        if gpeb_keys:
            self.monitor.log_info(
                "\n================ Q-Bound diagnostics "
                "================"
            )
            for key in sorted(gpeb_keys):
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

        for index in range(sample_count):
            video_question_id = video_question_ids[index]
            video_id, question_id = (
                video_question_id.split("_")
            )

            duration = gt[video_id]["duration"]
            ground_truth = gt[video_id][
                "location"
            ][question_id]
            posthoc_ground = posthoc[
                video_question_id
            ]
            attention_ground = np.round(
                np.asarray(
                    attn[video_question_id]
                ) * duration,
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
                    probs[video_question_id]
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
                    f"{epoch}_{video_id}_"
                    f"{question_id}.png",
                ),
                dpi=300,
            )
            plt.close(figure)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def eval(self, data_loader, epoch=None, save=False):
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

        gpeb_aux_correct = 0
        gpeb_aux_count = 0
        gpeb_stat_sums = collections.defaultdict(
            float
        )
        gpeb_stat_count = 0

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
                model_core._compute_answer_embedding(
                    self.a2v
                )

            for index, batch in enumerate(data_loader):
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
                question = batch["question"].to(
                    device,
                    non_blocking=True,
                )
                question_id = batch["question_id"]

                video_len = batch["video_len"]
                video_mask = get_mask(
                    video_len,
                    video_frames.size(1),
                ).to(
                    device,
                    non_blocking=True,
                )

                batch_size = answer_id.size(0)
                count += batch_size

                fusion_proj, answer_proj, model_outputs = (
                    self.model(
                        video_frames,
                        video_mask,
                        answer=answer,
                        answer_id=answer_id,
                        mode="VQA",
                        question=question,
                    )
                )

                predicts = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=fusion_proj,
                    detach_answer=False,
                )

                predicted = predicts.argmax(
                    dim=-1
                ).cpu()

                score["acc"] += (
                    predicted
                    == answer_id.reshape(-1).cpu()
                ).sum().item()

                time_param = model_outputs.get(
                    "time_param"
                )
                eval_wrapper = None

                if time_param is not None:
                    saw_time_param = True
                    eval_wrapper = (
                        None,
                        None,
                        {"time_param": time_param},
                    )

                    gpeb_aux = time_param.get(
                        "gpeb_aux"
                    )
                else:
                    gpeb_aux = None

                if gpeb_aux is not None:
                    bridge_repr = gpeb_aux.get(
                        "bridge_repr"
                    )
                    if bridge_repr is not None:
                        auxiliary_logits = (
                            self._score_answer_candidates(
                                answer_proj=answer_proj,
                                representation=bridge_repr,
                                detach_answer=True,
                                normalize=True,
                                logit_scale=float(
                                    self.cfgs["model"].get(
                                        "gpeb_aux_logit_scale",
                                        10.0,
                                    )
                                ),
                            )
                        )
                        auxiliary_predicted = (
                            auxiliary_logits.argmax(
                                dim=-1
                            ).cpu()
                        )
                        gpeb_aux_correct += (
                            auxiliary_predicted
                            == answer_id.reshape(-1).cpu()
                        ).sum().item()
                        gpeb_aux_count += batch_size

                    _, counterfactual_logs = (
                        self._compute_gpeb_v1_losses(
                            gqa_output=eval_wrapper,
                            answer_proj=answer_proj,
                            answer_id=answer_id,
                        )
                    )
                    batch_stats = self._collect_gpeb_stats(
                        eval_wrapper,
                        extra_logs=counterfactual_logs,
                    )

                    for name, value in batch_stats.items():
                        gpeb_stat_sums[name] += (
                            float(value) * batch_size
                        )
                    gpeb_stat_count += batch_size

                video_attention = model_outputs[
                    "fatt"
                ].detach().cpu().numpy()

                if (
                    video_attention.ndim == 3
                    and video_attention.shape[-1] == 1
                ):
                    video_attention = (
                        video_attention.squeeze(-1)
                    )
                if video_attention.ndim == 1:
                    video_attention = (
                        video_attention[None, :]
                    )

                if time_param is not None:
                    start_time = time_param[
                        "start"
                    ].detach().cpu().numpy()
                    end_time = time_param[
                        "end"
                    ].detach().cpu().numpy()
                    batch_ground_pre = np.column_stack(
                        (start_time, end_time)
                    )

                    batch_keyframe = time_param[
                        "key"
                    ].detach().cpu().numpy()
                    batch_ori_keyframe = time_param[
                        "ori_key"
                    ].detach().cpu().numpy()

                    for batch_index, qid in enumerate(
                        question_id
                    ):
                        results[qid] = {
                            "prediction": int(
                                predicted.numpy()[
                                    batch_index
                                ]
                            ),
                            "answer": int(
                                answer_id.reshape(-1)
                                .numpy()[batch_index]
                            ),
                        }
                        ground_res[qid] = (
                            video_attention[
                                batch_index
                            ].tolist()
                        )
                        ground_pre[qid] = (
                            batch_ground_pre[
                                batch_index
                            ].tolist()
                        )
                        keyframe[qid] = (
                            batch_keyframe[
                                batch_index
                            ].tolist()
                        )
                        ori_keyframe[qid] = (
                            batch_ori_keyframe[
                                batch_index
                            ].tolist()
                        )
                else:
                    for batch_index, qid in enumerate(
                        question_id
                    ):
                        results[qid] = {
                            "prediction": int(
                                predicted.numpy()[
                                    batch_index
                                ]
                            ),
                            "answer": int(
                                answer_id.reshape(-1)
                                .numpy()[batch_index]
                            ),
                        }
                        ground_res[qid] = (
                            video_attention[
                                batch_index
                            ].tolist()
                        )

                progress = float(index + 1) / len(
                    data_loader
                )
                print(
                    f"\rProgress: {progress:.2%}",
                    end="",
                    flush=True,
                )

        if count == 0:
            raise RuntimeError(
                "Evaluation data loader is empty."
            )

        score["acc"] /= count

        if gpeb_aux_count > 0:
            score["gpeb_aux_acc"] = (
                gpeb_aux_correct
                / gpeb_aux_count
            )

        if gpeb_stat_count > 0:
            for name, total in gpeb_stat_sums.items():
                score[name] = (
                    total / gpeb_stat_count
                )

        split = data_loader.dataset.split
        csv_path = self.cfgs["dataset"]["csv_path"]

        ground_truth_file = os.path.join(
            csv_path,
            f"gsub_{split}.json",
        )
        ground_truth = load_file(
            ground_truth_file
        )

        segment_file = os.path.join(
            csv_path,
            f"frame2time_{split}.json",
        )
        segments = load_file(segment_file)

        qa_file = os.path.join(
            csv_path,
            f"{split}.csv",
        )
        questions = load_file(qa_file)

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

        result = {
            "ph": post_hoc_ground,
            "result": results,
            "gt_ground": ground_truth,
            "ph_attn": ground_res,
        }

        if saw_time_param:
            score["pre-hoc"] = self.metric(
                ground_pre,
                ground_truth,
                results,
                gs=True,
            )
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
            result.update(
                {
                    "tg": ground_pre,
                    "key": keyframe,
                    "ori_key": ori_keyframe,
                    "merge": merge_ground,
                }
            )

        return score, result

    def infer(self, epoch=None):
        parameter_count = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )
        print(f"Model Params: {parameter_count}")
        print("\nEval Stage")

        val_score, _ = self.eval(
            self.val_dataloader,
            epoch,
            self.cfgs["dataset"]["name"] == "star",
        )

        if self.cfgs["dataset"]["name"] == "star":
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
                for key, value in val_score.items()
            },
            **{
                f"test_{key}": value
                for key, value in test_score.items()
            },
        }

        self.print_score(score)
        return score
