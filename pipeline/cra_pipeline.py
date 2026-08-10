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
    CRA training pipeline with Q-Bound Version 1.1.1.

    V1.1.1 deliberately simplifies the Q-Bound objective.

    Main path:
        question + video
            -> Q-Bound event topology
            -> event temporal evidence
            -> CRA temporal logits + Q-Bound temporal logits
            -> final grounding
            -> SAME main VQA scorer

    Counterfactual path:
        remove the strongest learned EVENT TRANSITION
            -> recompute Q-Bound temporal evidence
            -> SAME grounding path
            -> SAME main VQA scorer

    The only Q-Bound training objective is transition-necessity calibration:

        harm = CE_merge - CE_original
        necessity = 1 - exp(-relu(harm))
        L_qbound = BCE(selected_transition_strength, stopgrad(necessity))

    There is no alpha/beta gate, no split loss, no separate minimality loss,
    no auxiliary answer head, and no CRA-grounding consistency loss.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

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

    def _get_qbound_aux(self, gqa_output):
        """Safely extract Q-Bound V1.1.1 outputs from a CRA VQA forward."""
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

        # V1.1 name first; legacy alias is temporary.
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

    def _score_answer_candidates(
        self,
        answer_proj,
        representation,
        detach_answer=True,
        normalize=False,
        logit_scale=1.0,
    ):
        """
        Score answer candidates with the repository's main dot-product rule.

        answer_proj:
            [B, C_total, D]

        representation:
            [B, D] or [B, M, D]

        Returns:
            [B, C] or [B, M, C]

        `prop_num` aggregation is exactly the same operation used by the main
        VQA path.  V1.1.1 counterfactuals therefore do not use a separate
        auxiliary answer head.
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
            raise ValueError("logit_scale must be positive.")

        logits = float(logit_scale) * torch.einsum(
            "bcd,bmd->bmc",
            candidate_features,
            representation,
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
        Diagnostic only in V1.1.1.

        There is NO CRA-event grounding consistency loss.
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

    # ------------------------------------------------------------------
    # Q-Bound V1.1.1 loss
    # ------------------------------------------------------------------
    def _compute_qbound_v111_loss(
        self,
        gqa_output,
        answer_proj,
        answer_id,
        main_logits,
    ):
        """
        Transition-level necessity calibration.

        Let s be the strength of the strongest learned event transition.
        QBoundCore removes that whole transition and refine.py recomputes the
        merge representation through the SAME main grounding path.

        Per sample:

            L_original = CE(main_logits, y)
            L_merge    = CE(merge_logits, y)

            harm = L_merge - L_original

            necessity =
                1 - exp(-relu(harm))

        `necessity` is detached. Therefore the model cannot reduce this
        objective by intentionally changing the merge risk target itself.

        The single Q-Bound objective is:

            BCE(
                selected_transition_strength,
                stopgrad(necessity)
            )

        This replaces V1.1's split invariance + selected-frame minimality.
        """
        qbound_aux = self._get_qbound_aux(
            gqa_output
        )
        if qbound_aux is None:
            return None, {}

        required_keys = {
            "merge_video_proj",
            "selected_transition_strength",
            "selected_transition_mass",
            "removed_boundary_mass",
            "merge_transition_selector",
            "cra_video_proj",
            "cra_key",
            "event_key",
            "final_key",
            "merge_final_key",
            "merge_key_l1",
        }

        missing = sorted(
            required_keys.difference(
                qbound_aux
            )
        )
        if missing:
            raise KeyError(
                "Missing Q-Bound V1.1.1 outputs: "
                + ", ".join(missing)
            )

        labels = self._prepare_labels(
            answer_id,
            answer_proj.device,
        )

        if main_logits.size(0) != labels.numel():
            raise ValueError(
                "answer_id size does not match main VQA logits."
            )

        # Same answer candidate embeddings and exact same scoring rule
        # as the original VQA path. Answer embeddings are detached only
        # for the counterfactual regularizer.
        merge_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=qbound_aux[
                "merge_video_proj"
            ],
            detach_answer=True,
        )

        original_ce = F.cross_entropy(
            main_logits,
            labels,
            reduction="none",
        )

        if merge_logits.ndim == 2:
            merge_ce = F.cross_entropy(
                merge_logits,
                labels,
                reduction="none",
            )
            merge_pred_logits = merge_logits
        elif merge_logits.ndim == 3:
            # Current core emits one transition-level merge [B,1,D].
            # Keep this robust if later we vectorize several transitions.
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

            merge_pred_logits = merge_logits.mean(
                dim=1
            )
        else:
            raise ValueError(
                "merge logits must be [B,C] or [B,M,C], got "
                f"{tuple(merge_logits.shape)}."
            )

        answer_harm = (
            merge_ce - original_ce
        )

        # Parameter-free [0,1) task-necessity target.
        necessity_target = (
            1.0
            - torch.exp(
                -F.relu(answer_harm)
            )
        ).detach()

        selected_transition_strength = (
            qbound_aux[
                "selected_transition_strength"
            ]
            .reshape(-1)
            .clamp(
                min=1e-6,
                max=1.0 - 1e-6,
            )
        )

        if (
            selected_transition_strength.numel()
            != labels.numel()
        ):
            raise ValueError(
                "selected_transition_strength must contain "
                "one value per sample."
            )

        # The selector is deterministic and non-learned. The loss calibrates
        # the strength of the selected event transition to its observed
        # task-level necessity.
        necessity_loss = F.binary_cross_entropy(
            selected_transition_strength,
            necessity_target,
        )

        with torch.no_grad():
            original_pred = main_logits.argmax(
                dim=-1
            )
            merge_pred = merge_pred_logits.argmax(
                dim=-1
            )

            base_logits = (
                self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=qbound_aux[
                        "cra_video_proj"
                    ],
                    detach_answer=True,
                )
            )
            base_pred = base_logits.argmax(
                dim=-1
            )

            cra_event_js = (
                self._js_divergence(
                    qbound_aux["cra_key"],
                    qbound_aux["event_key"],
                ).mean()
            )

            cra_final_js = (
                self._js_divergence(
                    qbound_aux["cra_key"],
                    qbound_aux["final_key"],
                ).mean()
            )

            merge_final_js = (
                self._js_divergence(
                    qbound_aux["final_key"],
                    qbound_aux[
                        "merge_final_key"
                    ].squeeze(1),
                ).mean()
            )

        logs = {
            "qbound_original_acc": (
                (original_pred == labels)
                .float().mean().item()
            ),
            "qbound_base_only_acc": (
                (base_pred == labels)
                .float().mean().item()
            ),
            "qbound_merge_acc": (
                (merge_pred == labels)
                .float().mean().item()
            ),
            "qbound_original_ce": (
                original_ce.detach()
                .mean().item()
            ),
            "qbound_merge_ce": (
                merge_ce.detach()
                .mean().item()
            ),
            "qbound_answer_harm": (
                answer_harm.detach()
                .mean().item()
            ),
            "qbound_positive_harm_ratio": (
                (answer_harm > 0)
                .float().mean().item()
            ),
            "qbound_necessity": (
                necessity_target
                .mean().item()
            ),
            "qbound_selected_transition_strength": (
                selected_transition_strength
                .detach().mean().item()
            ),
            "qbound_selected_transition_mass": (
                qbound_aux[
                    "selected_transition_mass"
                ]
                .detach().float()
                .mean().item()
            ),
            "qbound_removed_boundary_mass": (
                qbound_aux[
                    "removed_boundary_mass"
                ]
                .detach().float()
                .mean().item()
            ),
            "qbound_merge_key_l1": (
                qbound_aux[
                    "merge_key_l1"
                ]
                .detach().float()
                .mean().item()
            ),
            "qbound_necessity_loss_raw": (
                necessity_loss.detach().item()
            ),
            "qbound_cra_event_js": (
                cra_event_js.detach().item()
            ),
            "qbound_cra_final_js": (
                cra_final_js.detach().item()
            ),
            "qbound_original_merge_js": (
                merge_final_js.detach().item()
            ),
        }

        return necessity_loss, logs

    def _weighted_add_qbound_loss(
        self,
        loss,
        qbound_loss,
    ):
        if qbound_loss is None:
            return

        optim_cfg = self.cfgs.get(
            "optim",
            {},
        )

        # V1.1.1 exposes only one Q-Bound loss weight.
        # The fallback lets the code survive until the YAML is updated.
        qbound_weight = float(
            optim_cfg.get(
                "qbound_weight",
                optim_cfg.get(
                    "qbound_structural_weight",
                    0.10,
                ),
            )
        )

        if qbound_weight < 0.0:
            raise ValueError(
                "optim.qbound_weight must be >= 0."
            )

        loss["qbound_necessity_loss"] = (
            qbound_loss
        )

        loss["total_loss"] = (
            loss["total_loss"]
            + qbound_weight
            * qbound_loss
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

    def _collect_qbound_stats(
        self,
        gqa_output,
        extra_logs=None,
    ):
        stats = {}

        if extra_logs:
            stats.update(
                extra_logs
            )

        qbound_aux = self._get_qbound_aux(
            gqa_output
        )
        if qbound_aux is None:
            return stats

        scalar_keys = [
            "boundary_prob_mean",
            "boundary_count",
            "assignment_entropy",
            "effective_event_count",
            "token_entropy",
            "selected_transition_mass",
            "selected_transition_strength",
            "removed_boundary_mass",
            "merge_key_l1",
            "question_only_conditioning",
        ]

        for key in scalar_keys:
            scalar = self._scalar_mean(
                qbound_aux.get(key)
            )
            if scalar is not None:
                stats[
                    f"qbound_{key}"
                ] = scalar

        temporal_bias = qbound_aux.get(
            "temporal_bias"
        )
        if torch.is_tensor(
            temporal_bias
        ):
            stats[
                "qbound_temporal_bias_abs_mean"
            ] = (
                temporal_bias.detach()
                .float().abs()
                .mean().item()
            )

        boundary_prob = qbound_aux.get(
            "boundary_prob"
        )
        boundary_mask = qbound_aux.get(
            "boundary_mask"
        )

        if (
            torch.is_tensor(boundary_prob)
            and torch.is_tensor(boundary_mask)
        ):
            valid_boundary = boundary_prob[
                boundary_mask.bool()
            ]

            if valid_boundary.numel() > 0:
                stats[
                    "qbound_boundary_prob_min"
                ] = (
                    valid_boundary.detach()
                    .float().min().item()
                )
                stats[
                    "qbound_boundary_prob_max"
                ] = (
                    valid_boundary.detach()
                    .float().max().item()
                )

        event_mass = qbound_aux.get(
            "event_mass"
        )
        if torch.is_tensor(event_mass):
            stats[
                "qbound_event_mass_min"
            ] = (
                event_mass.detach()
                .float()
                .min(dim=-1)
                .values.mean().item()
            )
            stats[
                "qbound_event_mass_max"
            ] = (
                event_mass.detach()
                .float()
                .max(dim=-1)
                .values.mean().item()
            )

        transition_mass = qbound_aux.get(
            "transition_mass"
        )
        transition_strength = qbound_aux.get(
            "transition_strength"
        )
        transition_mask = qbound_aux.get(
            "transition_counterfactual_mask"
        )

        if torch.is_tensor(
            transition_mass
        ):
            if torch.is_tensor(
                transition_mask
            ):
                valid = transition_mask.bool()
                valid_values = transition_mass[
                    valid
                ]
            else:
                valid_values = (
                    transition_mass.reshape(-1)
                )

            if valid_values.numel() > 0:
                stats[
                    "qbound_transition_mass_mean"
                ] = (
                    valid_values.detach()
                    .float().mean().item()
                )
                stats[
                    "qbound_transition_mass_max"
                ] = (
                    valid_values.detach()
                    .float().max().item()
                )

        if torch.is_tensor(
            transition_strength
        ):
            if torch.is_tensor(
                transition_mask
            ):
                valid = transition_mask.bool()
                valid_values = (
                    transition_strength[
                        valid
                    ]
                )
            else:
                valid_values = (
                    transition_strength.reshape(-1)
                )

            if valid_values.numel() > 0:
                stats[
                    "qbound_transition_strength_mean"
                ] = (
                    valid_values.detach()
                    .float().mean().item()
                )
                stats[
                    "qbound_transition_strength_max"
                ] = (
                    valid_values.detach()
                    .float().max().item()
                )

        # Ratio between removed mass and the full boundary count tells us
        # whether the counterfactual is actually large enough to change
        # topology. V1.1's frame-edge merge made this ratio tiny.
        removed_mass = qbound_aux.get(
            "removed_boundary_mass"
        )
        boundary_count = qbound_aux.get(
            "boundary_count"
        )

        if (
            torch.is_tensor(removed_mass)
            and torch.is_tensor(boundary_count)
        ):
            removed_ratio = (
                removed_mass
                / boundary_count.clamp_min(
                    1e-6
                )
            )
            stats[
                "qbound_removed_boundary_ratio"
            ] = (
                removed_ratio.detach()
                .float().mean().item()
            )

        return stats

    @staticmethod
    def _qbound_loss_keys():
        return [
            "qbound_necessity_loss",
        ]

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_original_acc",
            "qbound_base_only_acc",
            "qbound_merge_acc",
            "qbound_original_ce",
            "qbound_merge_ce",
            "qbound_answer_harm",
            "qbound_positive_harm_ratio",
            "qbound_necessity",
            "qbound_selected_transition_strength",
            "qbound_selected_transition_mass",
            "qbound_removed_boundary_mass",
            "qbound_removed_boundary_ratio",
            "qbound_merge_key_l1",
            "qbound_necessity_loss_raw",
            "qbound_cra_event_js",
            "qbound_cra_final_js",
            "qbound_original_merge_js",
            "qbound_temporal_bias_abs_mean",
            "qbound_boundary_prob_mean",
            "qbound_boundary_count",
            "qbound_boundary_prob_min",
            "qbound_boundary_prob_max",
            "qbound_assignment_entropy",
            "qbound_effective_event_count",
            "qbound_token_entropy",
            "qbound_event_mass_min",
            "qbound_event_mass_max",
            "qbound_transition_mass_mean",
            "qbound_transition_mass_max",
            "qbound_transition_strength_mean",
            "qbound_transition_strength_max",
            "qbound_question_only_conditioning",
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

        # Question-only tokens are passed explicitly to Q-Bound.
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
        target_dict["vqa_target"] = (
            answer_id_device
        )

        # Preserve the repository's original CRA alignment objective.
        # V1.1.1 does not invent a pseudo temporal-label loss here because this
        # training batch currently contains no ground-truth temporal segment.
        prediction_dict["vqa_align_pred"] = (
            gqa_output[2]["time_param"][
                "grounding_feature"
            ]
        )
        target_dict["vqa_align_gt"] = (
            gqa_output[2]["align_gt"]
        )

        # L_task: all original CRA task losses.
        loss = self.criterion(
            prediction_dict,
            target_dict,
        )

        # Single Q-Bound V1.1.1 transition-necessity objective.
        qbound_loss, qbound_logs = (
            self._compute_qbound_v111_loss(
                gqa_output=gqa_output,
                answer_proj=answer_proj,
                answer_id=answer_id,
                main_logits=predicts,
            )
        )

        self._weighted_add_qbound_loss(
            loss,
            qbound_loss,
        )

        predicted = predicts.argmax(
            dim=-1
        ).detach().cpu()

        running_accuracy = (
            predicted
            == answer_id.reshape(-1).cpu()
        ).float().mean().item()

        qbound_stats = self._collect_qbound_stats(
            gqa_output,
            extra_logs=qbound_logs,
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
        self.monitor.reset_kv("total_loss")

        for key in self.cfgs["optim"]["loss"].keys():
            if key in {"ce", "align"}:
                continue
            self.monitor.reset_kv(f"{key}_loss")

        if self._qbound_enabled():
            for key in self._qbound_loss_keys():
                self.monitor.reset_kv(key)
            for key in self._qbound_monitor_keys():
                self.monitor.reset_kv(key)

        for index, batch in enumerate(
            self.train_dataloader
        ):
            start_time = time.time()

            loss, monitor_metric, qbound_stats = (
                self.get_loss(batch)
            )

            self.monitor.logkv_mean(
                "acc",
                monitor_metric,
            )
            for name, value in qbound_stats.items():
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

        qbound_keys = [
            key for key in score
            if key.startswith("val_qbound_")
            or key.startswith("test_qbound_")
        ]

        if qbound_keys:
            self.monitor.log_info(
                "\n================ Q-Bound diagnostics "
                "================"
            )
            for key in sorted(qbound_keys):
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

        qbound_stat_sums = collections.defaultdict(
            float
        )
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

                if time_param is not None:
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
                else:
                    eval_wrapper = None
                    qbound_aux = None

                if qbound_aux is not None:
                    _, counterfactual_logs = (
                        self._compute_qbound_v111_loss(
                            gqa_output=eval_wrapper,
                            answer_proj=answer_proj,
                            answer_id=answer_id,
                            main_logits=predicts,
                        )
                    )

                    batch_stats = (
                        self._collect_qbound_stats(
                            eval_wrapper,
                            extra_logs=(
                                counterfactual_logs
                            ),
                        )
                    )

                    for name, value in batch_stats.items():
                        qbound_stat_sums[name] += (
                            float(value) * batch_size
                        )

                    qbound_stat_count += batch_size

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

        if qbound_stat_count > 0:
            for name, total in (
                qbound_stat_sums.items()
            ):
                score[name] = (
                    total / qbound_stat_count
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