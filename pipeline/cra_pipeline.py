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
    CRA training/evaluation pipeline with Q-Bound Version 2.0.0.

    V2.0.0 is a representation-level reset rather than another V1.x temporal
    patch.  Q-Bound no longer commits to a fixed event partition, no longer
    enumerates 528 temporal hypotheses, and no longer learns from a teacher
    distribution.  The active temporal path is:

        generic video tokens
            -> temporal dynamics
            -> video-only transition prior
            -> bounded question-conditioned functional cut residual
            -> query-specific soft cut field
            -> boundary-gated temporal reasoning
            -> direct start/end grounding
            -> hard H* at inference

    The event count is emergent.  There is no semantic ``max_events`` and no
    event-count target.  A cut field c_t^Q only states whether preserving the
    distinction between adjacent temporal states is useful for the current
    question.

    Weak supervision
    ----------------
    QBoundCore itself remains answer-free.  Candidate-answer information is
    used only after the V+Q grounding forward pass by two deterministic,
    stop-gradient CRA verifier interfaces:

      1) Functional composition verifier:
             KEEP(P_t | P_{t+1})  vs  MERGE(P_t,P_{t+1})

         The GT-answer decision-margin change provides a local functional
         signal for whether that temporal distinction should be preserved.
         Small/noisy verifier differences receive little weight rather than
         forcing arbitrary cuts.

      2) Boundary-equilibrium verifier:
             H*, trim-left, trim-right, expand-left, expand-right

         If a one-step boundary perturbation improves GT-answer evidence, that
         action becomes a detached local correction target for the differentiable
         start/end heads.  When no local perturbation improves evidence, the
         boundary is at a local functional equilibrium and this auxiliary loss
         vanishes.

    Thus the Q-Bound-specific objective is:

        L = L_CRA/main
            + lambda_comp * L_functional_composition
            + lambda_eq   * L_boundary_equilibrium

    where L_CRA/main is the repository's ordinary factual VQA/VQ objective.
    No temporal GT, IoU, IoP, pseudo timestamp, teacher KL, span-length target,
    or event-count regularizer is used for training.

    Important gradient rule
    -----------------------
    The functional composition loss uses the *detached* video transition prior
    plus the trainable question residual.  Therefore this auxiliary loss teaches
    the question-conditioned merge/cut adjustment without directly collapsing
    the video-only prior.  The prior itself is learned through the factual QA
    path because the cut field causally gates temporal reasoning.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        self.qbound_version = "2.0.0"
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
        """Compatibility helper: V2.0.0 has no CRA -> Q-Bound handover."""
        return 1.0, 0

    def _train_qbound_mix(self, epoch, batch_index, batches_per_epoch):
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
        return time_param.get("qbound_aux", time_param.get("gpeb_aux"))

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
    def _masked_mean_scalar(values, mask):
        if values.shape != mask.shape:
            raise ValueError("values/mask must have identical shape.")
        mask_f = mask.to(values.dtype)
        denominator = mask_f.sum().clamp_min(1.0)
        return (values * mask_f).sum() / denominator

    @staticmethod
    def _normalize_distribution(values, mask=None, eps=1e-8):
        values = values.clamp_min(0.0)
        if mask is not None:
            while mask.ndim < values.ndim:
                mask = mask.unsqueeze(1)
            values = values * mask.to(values.dtype)
        return values / values.sum(dim=-1, keepdim=True).clamp_min(eps)

    @staticmethod
    def _distribution_entropy(probability, eps=1e-8):
        probability = probability.clamp_min(0.0)
        probability = probability / probability.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(eps)
        return -torch.sum(
            probability * torch.log(probability.clamp_min(eps)),
            dim=-1,
        )

    @classmethod
    def _js_divergence(cls, first, second, mask=None, eps=1e-8):
        first = cls._normalize_distribution(first, mask=mask, eps=eps)
        second = cls._normalize_distribution(second, mask=mask, eps=eps)
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
    # V2.0.0 objective configuration
    # ------------------------------------------------------------------
    def _qbound_v200_loss_settings(self):
        """
        Preferred compact YAML:

            model:
              qbound:
                cut_temperature: 0.7
                functional_composition:
                  loss_weight: 0.10
                  confidence_scale: 0.25
                boundary_equilibrium:
                  loss_weight: 0.10
                  confidence_scale: 0.25

        ``confidence_scale`` does not define a desired answer margin.  It only
        controls how strongly a measured verifier difference is trusted.  Near
        ties receive almost zero auxiliary weight, which avoids forcing noisy
        KEEP/MERGE or trim/expand decisions.
        """
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        comp_cfg = qbound_cfg.get("functional_composition", {})
        if comp_cfg is None:
            comp_cfg = {}
        if not isinstance(comp_cfg, dict):
            raise TypeError(
                "model.qbound.functional_composition must be a mapping/dict."
            )

        eq_cfg = qbound_cfg.get("boundary_equilibrium", {})
        if eq_cfg is None:
            eq_cfg = {}
        if not isinstance(eq_cfg, dict):
            raise TypeError(
                "model.qbound.boundary_equilibrium must be a mapping/dict."
            )

        comp_weight = float(comp_cfg.get("loss_weight", 0.10))
        comp_confidence_scale = float(
            comp_cfg.get("confidence_scale", 0.25)
        )
        eq_weight = float(eq_cfg.get("loss_weight", 0.10))
        eq_confidence_scale = float(
            eq_cfg.get("confidence_scale", 0.25)
        )
        cut_temperature = float(qbound_cfg.get("cut_temperature", 0.7))

        if comp_weight < 0.0:
            raise ValueError(
                "qbound functional_composition.loss_weight must be >= 0."
            )
        if comp_confidence_scale <= 0.0:
            raise ValueError(
                "qbound functional_composition.confidence_scale must be > 0."
            )
        if eq_weight < 0.0:
            raise ValueError(
                "qbound boundary_equilibrium.loss_weight must be >= 0."
            )
        if eq_confidence_scale <= 0.0:
            raise ValueError(
                "qbound boundary_equilibrium.confidence_scale must be > 0."
            )
        if cut_temperature <= 0.0:
            raise ValueError("qbound cut_temperature must be > 0.")

        return {
            "functional_composition_weight": comp_weight,
            "functional_confidence_scale": comp_confidence_scale,
            "boundary_equilibrium_weight": eq_weight,
            "equilibrium_confidence_scale": eq_confidence_scale,
            "cut_temperature": cut_temperature,
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
                f"{answer_proj.size(-1)} vs {representation.size(-1)}."
            )

        candidate_features = (
            answer_proj.detach() if detach_answer else answer_proj
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

        prop_num = int(self.cfgs["model"].get("prop_num", 1))
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

        if squeeze_dim:
            logits = logits.squeeze(1)
        return logits

    # ------------------------------------------------------------------
    # Functional verifier helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _answer_decision_margin(logits, labels):
        """
        GT-answer decision margin:

            z_y - logsumexp_{a != y}(z_a)

        Supports [B,C] or [B,M,C].  Verifier logits are already stop-gradient;
        this helper is therefore used to derive local functional targets only.
        """
        if logits.ndim not in {2, 3}:
            raise ValueError(
                "Decision-margin logits must be [B,C] or [B,M,C]."
            )
        if logits.size(-1) < 2:
            raise ValueError(
                "Functional verification requires at least two answers."
            )

        labels = labels.reshape(-1)
        if labels.numel() != logits.size(0):
            raise ValueError("labels do not match verifier-logit batch size.")

        if logits.ndim == 2:
            gather_index = labels[:, None]
        else:
            gather_index = labels[:, None, None].expand(
                -1,
                logits.size(1),
                1,
            )

        correct = logits.gather(-1, gather_index).squeeze(-1)
        wrong = logits.clone()
        wrong.scatter_(-1, gather_index, float("-inf"))
        wrong_lse = torch.logsumexp(wrong, dim=-1)
        return correct - wrong_lse

    @staticmethod
    def _confidence_from_gap(gap, scale):
        """Map |functional change| to [0,1) without inventing a margin target."""
        return 1.0 - torch.exp(-gap.abs() / float(scale))

    def _qbound_functional_composition_objective(
        self,
        qbound_aux,
        labels,
        settings,
    ):
        """
        Learn query-specific cuts from local KEEP-vs-MERGE consequences.

        A positive GT-answer margin gap means KEEP preserves more useful
        information than MERGE, so the detached target is CUT=1.  A negative
        gap means MERGE is at least as functional, so the target is CUT=0.
        Near ties are down-weighted by a continuous confidence term.

        Crucially, the auxiliary BCE is evaluated on:

            stopgrad(video_prior_logit) + trainable_question_residual

        so this loss cannot directly erase the video-only transition prior.
        The prior remains trainable through the factual QA path, where the cut
        field causally gates temporal reasoning.
        """
        required = {
            "functional_cut_prob",
            "video_boundary_prior_logits",
            "question_cut_residual",
            "functional_pair_valid_mask",
            "functional_keep_answer_logits",
            "functional_merge_answer_logits",
            "functional_left_answer_logits",
            "functional_right_answer_logits",
            "functional_verifier_stop_gradient",
            "teacher_distribution_used",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "V2.0.0 functional-composition interface is missing: "
                + ", ".join(missing)
            )

        pair_valid = qbound_aux["functional_pair_valid_mask"].bool()
        keep_logits = qbound_aux["functional_keep_answer_logits"].detach()
        merge_logits = qbound_aux["functional_merge_answer_logits"].detach()
        left_logits = qbound_aux["functional_left_answer_logits"].detach()
        right_logits = qbound_aux["functional_right_answer_logits"].detach()

        if keep_logits.ndim != 3:
            raise ValueError(
                "functional_keep_answer_logits must be [B,T-1,C]."
            )
        if merge_logits.shape != keep_logits.shape:
            raise ValueError("KEEP/MERGE verifier logits must have same shape.")
        if left_logits.shape != keep_logits.shape or right_logits.shape != keep_logits.shape:
            raise ValueError("LEFT/RIGHT verifier logits must match KEEP logits.")
        if pair_valid.shape != keep_logits.shape[:2]:
            raise ValueError(
                "functional_pair_valid_mask must match verifier [B,T-1]."
            )

        with torch.no_grad():
            keep_margin = self._answer_decision_margin(keep_logits, labels)
            merge_margin = self._answer_decision_margin(merge_logits, labels)
            left_margin = self._answer_decision_margin(left_logits, labels)
            right_margin = self._answer_decision_margin(right_logits, labels)
            functional_gap = keep_margin - merge_margin

            target_cut = (functional_gap > 0.0).to(keep_margin.dtype)
            confidence = self._confidence_from_gap(
                functional_gap,
                settings["functional_confidence_scale"],
            )
            confidence = confidence * pair_valid.to(confidence.dtype)

        # Do not let this local answer-based objective directly rewrite the
        # video-only prior.  It teaches only the question-conditioned residual.
        effective_logits = (
            qbound_aux["video_boundary_prior_logits"].detach()
            + qbound_aux["question_cut_residual"]
        ) / float(settings["cut_temperature"])

        element_loss = F.binary_cross_entropy_with_logits(
            effective_logits,
            target_cut,
            reduction="none",
        )
        denominator = confidence.sum()
        if float(denominator.detach().item()) > 0.0:
            composition_loss = (
                element_loss * confidence
            ).sum() / denominator
        else:
            # Preserve graph connectivity to the question residual.
            composition_loss = qbound_aux["question_cut_residual"].sum() * 0.0

        with torch.no_grad():
            valid_float = pair_valid.to(functional_gap.dtype)
            valid_count = valid_float.sum().clamp_min(1.0)
            predicted_cut = qbound_aux["functional_cut_prob"].detach() >= 0.5
            confident_mask = pair_valid & (confidence > 0.05)
            if confident_mask.any():
                agreement = (
                    predicted_cut[confident_mask]
                    == target_cut.bool()[confident_mask]
                ).float().mean().item()
            else:
                agreement = 0.0

            diagnostics = {
                "functional_gap_mean": (
                    functional_gap * valid_float
                ).sum().item() / valid_count.item(),
                "functional_gap_abs_mean": (
                    functional_gap.abs() * valid_float
                ).sum().item() / valid_count.item(),
                "functional_keep_better_ratio": (
                    ((functional_gap > 0.0) & pair_valid).float().sum()
                    / valid_count
                ).item(),
                "functional_merge_better_ratio": (
                    ((functional_gap < 0.0) & pair_valid).float().sum()
                    / valid_count
                ).item(),
                "functional_target_cut_ratio": (
                    (target_cut * valid_float).sum() / valid_count
                ).item(),
                "functional_confidence_mean": (
                    confidence.sum() / valid_count
                ).item(),
                "functional_cut_target_agreement": agreement,
                "functional_left_right_margin_gap": (
                    ((left_margin - right_margin).abs() * valid_float).sum()
                    / valid_count
                ).item(),
            }

        return composition_loss, diagnostics

    def _qbound_boundary_equilibrium_objective(
        self,
        qbound_aux,
        labels,
        settings,
    ):
        """
        Convert local answer-evidence improvements into endpoint corrections.

        The verifier itself is detached, so directly summing pairwise margin
        penalties would have zero gradient.  Instead, the best *improving*
        one-step action becomes a detached pseudo-action for the differentiable
        start/end heads:

          start: trim-left => s+1, expand-left => s-1
          end  : trim-right => e-1, expand-right => e+1

        If neither local action improves the selected span's GT-answer margin,
        that boundary is already at local functional equilibrium and receives
        zero auxiliary correction.  This avoids self-training the model to keep
        an early, possibly wrong, hard boundary merely because it selected it.
        """
        required = {
            "start_logits",
            "end_logits",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "equilibrium_selected_answer_logits",
            "equilibrium_trim_left_answer_logits",
            "equilibrium_trim_right_answer_logits",
            "equilibrium_expand_left_answer_logits",
            "equilibrium_expand_right_answer_logits",
            "equilibrium_trim_left_valid_mask",
            "equilibrium_trim_right_valid_mask",
            "equilibrium_expand_left_valid_mask",
            "equilibrium_expand_right_valid_mask",
            "equilibrium_verifier_stop_gradient",
            "teacher_distribution_used",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "V2.0.0 boundary-equilibrium interface is missing: "
                + ", ".join(missing)
            )

        start_logits = qbound_aux["start_logits"]
        end_logits = qbound_aux["end_logits"]
        if start_logits.ndim != 2 or end_logits.shape != start_logits.shape:
            raise ValueError("start_logits/end_logits must be matching [B,T].")
        batch_size, time_steps = start_logits.shape

        selected_start = qbound_aux["selected_span_start_frame"].detach().long()
        selected_end = qbound_aux["selected_span_end_frame"].detach().long()
        if selected_start.shape != (batch_size,) or selected_end.shape != (batch_size,):
            raise ValueError("selected span endpoints must be [B].")

        selected_answer = qbound_aux[
            "equilibrium_selected_answer_logits"
        ].detach()
        trim_left_answer = qbound_aux[
            "equilibrium_trim_left_answer_logits"
        ].detach()
        trim_right_answer = qbound_aux[
            "equilibrium_trim_right_answer_logits"
        ].detach()
        expand_left_answer = qbound_aux[
            "equilibrium_expand_left_answer_logits"
        ].detach()
        expand_right_answer = qbound_aux[
            "equilibrium_expand_right_answer_logits"
        ].detach()

        with torch.no_grad():
            selected_margin = self._answer_decision_margin(selected_answer, labels)
            trim_left_margin = self._answer_decision_margin(trim_left_answer, labels)
            trim_right_margin = self._answer_decision_margin(trim_right_answer, labels)
            expand_left_margin = self._answer_decision_margin(expand_left_answer, labels)
            expand_right_margin = self._answer_decision_margin(expand_right_answer, labels)

            tl_valid = qbound_aux["equilibrium_trim_left_valid_mask"].detach().bool()
            tr_valid = qbound_aux["equilibrium_trim_right_valid_mask"].detach().bool()
            el_valid = qbound_aux["equilibrium_expand_left_valid_mask"].detach().bool()
            er_valid = qbound_aux["equilibrium_expand_right_valid_mask"].detach().bool()

            neg_inf = torch.full_like(selected_margin, float("-inf"))
            tl_gain = torch.where(
                tl_valid,
                trim_left_margin - selected_margin,
                neg_inf,
            )
            el_gain = torch.where(
                el_valid,
                expand_left_margin - selected_margin,
                neg_inf,
            )
            tr_gain = torch.where(
                tr_valid,
                trim_right_margin - selected_margin,
                neg_inf,
            )
            er_gain = torch.where(
                er_valid,
                expand_right_margin - selected_margin,
                neg_inf,
            )

            left_gains = torch.stack([tl_gain, el_gain], dim=-1)
            right_gains = torch.stack([tr_gain, er_gain], dim=-1)
            left_best_gain, left_action = left_gains.max(dim=-1)
            right_best_gain, right_action = right_gains.max(dim=-1)

            left_improves = torch.isfinite(left_best_gain) & (left_best_gain > 0.0)
            right_improves = torch.isfinite(right_best_gain) & (right_best_gain > 0.0)

            target_start = selected_start.clone()
            target_start = torch.where(
                left_improves & (left_action == 0),
                target_start + 1,
                target_start,
            )
            target_start = torch.where(
                left_improves & (left_action == 1),
                target_start - 1,
                target_start,
            )

            target_end = selected_end.clone()
            target_end = torch.where(
                right_improves & (right_action == 0),
                target_end - 1,
                target_end,
            )
            target_end = torch.where(
                right_improves & (right_action == 1),
                target_end + 1,
                target_end,
            )

            target_start = target_start.clamp(min=0, max=time_steps - 1)
            target_end = target_end.clamp(min=0, max=time_steps - 1)

            left_positive_gain = torch.where(
                left_improves,
                left_best_gain.clamp_min(0.0),
                torch.zeros_like(left_best_gain),
            )
            right_positive_gain = torch.where(
                right_improves,
                right_best_gain.clamp_min(0.0),
                torch.zeros_like(right_best_gain),
            )
            left_confidence = self._confidence_from_gap(
                left_positive_gain,
                settings["equilibrium_confidence_scale"],
            ) * left_improves.to(left_positive_gain.dtype)
            right_confidence = self._confidence_from_gap(
                right_positive_gain,
                settings["equilibrium_confidence_scale"],
            ) * right_improves.to(right_positive_gain.dtype)

        start_ce = F.cross_entropy(
            start_logits,
            target_start,
            reduction="none",
        )
        end_ce = F.cross_entropy(
            end_logits,
            target_end,
            reduction="none",
        )

        denominator = left_confidence.sum() + right_confidence.sum()
        if float(denominator.detach().item()) > 0.0:
            equilibrium_loss = (
                (start_ce * left_confidence).sum()
                + (end_ce * right_confidence).sum()
            ) / denominator
        else:
            equilibrium_loss = (start_logits.sum() + end_logits.sum()) * 0.0

        with torch.no_grad():
            tl_action = left_improves & (left_action == 0)
            el_action = left_improves & (left_action == 1)
            tr_action = right_improves & (right_action == 0)
            er_action = right_improves & (right_action == 1)
            diagnostics = {
                "equilibrium_selected_margin": selected_margin.mean().item(),
                "equilibrium_active_left_ratio": left_improves.float().mean().item(),
                "equilibrium_active_right_ratio": right_improves.float().mean().item(),
                "equilibrium_active_any_ratio": (
                    left_improves | right_improves
                ).float().mean().item(),
                "equilibrium_trim_left_action_ratio": tl_action.float().mean().item(),
                "equilibrium_expand_left_action_ratio": el_action.float().mean().item(),
                "equilibrium_trim_right_action_ratio": tr_action.float().mean().item(),
                "equilibrium_expand_right_action_ratio": er_action.float().mean().item(),
                "equilibrium_left_gain_mean": left_positive_gain.mean().item(),
                "equilibrium_right_gain_mean": right_positive_gain.mean().item(),
                "equilibrium_left_confidence_mean": left_confidence.mean().item(),
                "equilibrium_right_confidence_mean": right_confidence.mean().item(),
                "equilibrium_start_shift_abs_mean": (
                    target_start - selected_start
                ).abs().float().mean().item(),
                "equilibrium_end_shift_abs_mean": (
                    target_end - selected_end
                ).abs().float().mean().item(),
            }

        return equilibrium_loss, diagnostics

    # ------------------------------------------------------------------
    # V2.0.0 contract + diagnostics
    # ------------------------------------------------------------------
    def _validate_v200_contract(self, qbound_aux):
        if qbound_aux is None:
            raise KeyError("Q-Bound V2.0.0 must expose qbound_aux.")

        required = {
            "qbound_version",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "functional_composition_enabled",
            "dynamic_event_count_enabled",
            "fixed_event_partition_enabled",
            "event_count_cap_enabled",
            "hypothesis_competition_enabled",
            "teacher_distillation_enabled",
            "answer_features_used_in_core",
            "boundary_equilibrium_interface_enabled",
            "functional_answer_verifier_ready",
            "boundary_equilibrium_verifier_ready",
            "functional_verifier_consumer_stop_gradient",
            "teacher_distribution_used",
            "functional_verifier_uses_answer_information",
            "functional_verifier_affects_grounding_forward",
            "grounding_requires_answer_at_inference",
            "temporal_gt_used_in_training",
            "functional_verifier_stop_gradient",
            "equilibrium_verifier_stop_gradient",
            "video_boundary_prior_logits",
            "video_boundary_prior_prob",
            "question_cut_residual",
            "functional_cut_logits",
            "functional_cut_prob",
            "functional_merge_gate",
            "functional_pair_valid_mask",
            "hard_event_count",
            "soft_effective_event_count",
            "one_event_indicator",
            "all_split_indicator",
            "start_logits",
            "end_logits",
            "start_prob",
            "end_prob",
            "training_temporal_prob",
            "main_temporal_prob",
            "selected_span_start_frame",
            "selected_span_end_frame",
            "selected_span_length",
            "selected_span_fraction",
            "functional_keep_answer_logits",
            "functional_merge_answer_logits",
            "equilibrium_selected_answer_logits",
            "equilibrium_trim_left_answer_logits",
            "equilibrium_trim_right_answer_logits",
            "equilibrium_expand_left_answer_logits",
            "equilibrium_expand_right_answer_logits",
            "cra_key",
            "qbound_key",
            "qbound_training_key",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "Missing Q-Bound V2.0.0 outputs: " + ", ".join(missing)
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
                raise TypeError(f"Q-Bound flag {name!r} must be a scalar tensor.")
            actual = float(value.detach().item())
            if actual != float(expected):
                raise RuntimeError(
                    f"Q-Bound V2.0.0 flag {name!r} must be {expected}; "
                    f"got {actual}."
                )

        expected_flags = {
            "qbound_input_is_generic": 1.0,
            "qbound_input_uses_cra_embedding": 0.0,
            "functional_composition_enabled": 1.0,
            "dynamic_event_count_enabled": 1.0,
            "fixed_event_partition_enabled": 0.0,
            "event_count_cap_enabled": 0.0,
            "hypothesis_competition_enabled": 0.0,
            "teacher_distillation_enabled": 0.0,
            "answer_features_used_in_core": 0.0,
            "boundary_equilibrium_interface_enabled": 1.0,
            "functional_answer_verifier_ready": 1.0,
            "boundary_equilibrium_verifier_ready": 1.0,
            "functional_verifier_consumer_stop_gradient": 1.0,
            "functional_verifier_stop_gradient": 1.0,
            "equilibrium_verifier_stop_gradient": 1.0,
            "teacher_distribution_used": 0.0,
            "functional_verifier_uses_answer_information": 1.0,
            "functional_verifier_affects_grounding_forward": 0.0,
            "grounding_requires_answer_at_inference": 0.0,
            "temporal_gt_used_in_training": 0.0,
        }
        for name, expected in expected_flags.items():
            _flag(name, expected)

        cut_prob = qbound_aux["functional_cut_prob"]
        pair_valid = qbound_aux["functional_pair_valid_mask"]
        if cut_prob.ndim != 2 or pair_valid.shape != cut_prob.shape:
            raise ValueError(
                "functional_cut_prob/pair mask must be matching [B,T-1]."
            )
        if qbound_aux["start_logits"].ndim != 2:
            raise ValueError("start_logits must be [B,T].")
        if qbound_aux["end_logits"].shape != qbound_aux["start_logits"].shape:
            raise ValueError("start/end logits must have the same shape.")

    def _collect_qbound_stats(self, gqa_output, extra_logs=None):
        stats = {}
        if extra_logs:
            stats.update(extra_logs)

        qbound_aux = self._get_qbound_aux(gqa_output)
        if qbound_aux is None:
            return stats

        scalar_keys = (
            # Generic input / dynamics.
            "generic_input_adapter_enabled",
            "generic_input_position_enabled",
            "generic_input_raw_rms",
            "generic_input_adapted_rms",
            "generic_input_token_cosine",
            "generic_input_delta_ratio",
            "delta_magnitude",
            "accel_magnitude",
            # Architecture flags.
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "functional_composition_enabled",
            "dynamic_event_count_enabled",
            "fixed_event_partition_enabled",
            "event_count_cap_enabled",
            "hypothesis_competition_enabled",
            "teacher_distillation_enabled",
            "answer_features_used_in_core",
            "teacher_distribution_used",
            "temporal_gt_used_in_training",
            # Composition / anti-collapse.
            "cut_mean",
            "cut_std",
            "prior_cut_mean",
            "question_residual_abs_mean",
            "hard_event_count",
            "soft_effective_event_count",
            "one_event_indicator",
            "all_split_indicator",
            # Reasoning / grounding.
            "anchor_effective_token_count",
            "start_entropy",
            "end_entropy",
            "selected_span_length",
            "selected_span_fraction",
            "selected_pair_score",
        )
        for key in scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                stats[f"qbound_{key}"] = scalar

        # Distribution of functional cuts over valid temporal edges.
        cut_prob = qbound_aux.get("functional_cut_prob")
        pair_valid = qbound_aux.get("functional_pair_valid_mask")
        if torch.is_tensor(cut_prob) and torch.is_tensor(pair_valid):
            values = cut_prob[pair_valid.bool()]
            if values.numel() > 0:
                values = values.detach().float()
                stats["qbound_functional_cut_min"] = values.min().item()
                stats["qbound_functional_cut_max"] = values.max().item()
                stats["qbound_functional_cut_std_valid"] = values.std(
                    unbiased=False
                ).item()
                stats["qbound_functional_cut_hard_ratio"] = (
                    values >= 0.5
                ).float().mean().item()

        prior_prob = qbound_aux.get("video_boundary_prior_prob")
        if (
            torch.is_tensor(prior_prob)
            and torch.is_tensor(cut_prob)
            and prior_prob.shape == cut_prob.shape
        ):
            valid = pair_valid.bool() if torch.is_tensor(pair_valid) else torch.ones_like(
                cut_prob, dtype=torch.bool
            )
            if valid.any():
                stats["qbound_prior_functional_cut_l1"] = (
                    (prior_prob.detach() - cut_prob.detach()).abs()[valid]
                    .float()
                    .mean()
                    .item()
                )

        # Effective event count is emergent; no event-cap histogram is used.
        hard_event_count = qbound_aux.get("hard_event_count")
        if torch.is_tensor(hard_event_count):
            values = hard_event_count.detach().float()
            stats["qbound_hard_event_count_min"] = values.min().item()
            stats["qbound_hard_event_count_max"] = values.max().item()
            stats["qbound_hard_event_count_std"] = values.std(
                unbiased=False
            ).item()

        # Endpoint health.
        for prefix in ("start_prob", "end_prob"):
            probability = qbound_aux.get(prefix)
            if torch.is_tensor(probability):
                p = self._normalize_distribution(probability.detach())
                stats[f"qbound_{prefix}_top1"] = p.max(dim=-1).values.mean().item()

        # Hard span vs differentiable soft temporal support.
        hard_key = qbound_aux.get("qbound_key")
        training_key = qbound_aux.get("qbound_training_key")
        soft_key = qbound_aux.get("soft_temporal_prob")
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
            stats["qbound_cra_qbound_js"] = self._js_divergence(
                cra, hard
            ).mean().item()
            stats["qbound_cra_qbound_key_l1"] = (
                cra - hard
            ).abs().mean().item()
        if torch.is_tensor(training_key) and torch.is_tensor(hard_key):
            # Forward values should be identical under the straight-through bridge.
            stats["qbound_training_hard_forward_l1"] = (
                training_key.detach() - hard_key.detach()
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
        for name, parameter in self.model.named_parameters():
            if "qbound" not in name.lower() or parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            total_sq += grad.pow(2).sum().item()
            found = True
        return float(total_sq ** 0.5) if found else 0.0

    def _qbound_component_grad_norms(self):
        if not self._qbound_enabled():
            return {
                "qbound_input_adapter_grad_norm": 0.0,
                "qbound_dynamics_grad_norm": 0.0,
                "qbound_video_prior_grad_norm": 0.0,
                "qbound_functional_composer_grad_norm": 0.0,
                "qbound_reasoner_grad_norm": 0.0,
                "qbound_endpoint_grad_norm": 0.0,
            }

        totals = {
            "input": 0.0,
            "dynamics": 0.0,
            "prior": 0.0,
            "composer": 0.0,
            "reasoner": 0.0,
            "endpoint": 0.0,
        }
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or "qbound" not in name.lower():
                continue
            lower = name.lower()
            grad_sq = parameter.grad.detach().float().pow(2).sum().item()

            if "generic_input_" in lower:
                totals["input"] += grad_sq
            if any(
                token in lower
                for token in (
                    "state_proj",
                    "delta_proj",
                    "accel_proj",
                    "dynamics_fuse",
                )
            ):
                totals["dynamics"] += grad_sq
            if "video_boundary_encoder" in lower or "video_boundary_score" in lower:
                totals["prior"] += grad_sq
            if any(
                token in lower
                for token in (
                    "functional_composer",
                    "question_cut_residual",
                    "question_proj",
                )
            ):
                totals["composer"] += grad_sq
            if any(
                token in lower
                for token in (
                    "forward_scan_",
                    "backward_scan_",
                    "local_reasoning_fuse",
                    "anchor_",
                    "reasoning_norm",
                )
            ):
                totals["reasoner"] += grad_sq
            if any(
                token in lower
                for token in (
                    "endpoint_encoder",
                    "start_score",
                    "end_score",
                    "frame_evidence_score",
                )
            ):
                totals["endpoint"] += grad_sq

        return {
            "qbound_input_adapter_grad_norm": float(totals["input"] ** 0.5),
            "qbound_dynamics_grad_norm": float(totals["dynamics"] ** 0.5),
            "qbound_video_prior_grad_norm": float(totals["prior"] ** 0.5),
            "qbound_functional_composer_grad_norm": float(
                totals["composer"] ** 0.5
            ),
            "qbound_reasoner_grad_norm": float(totals["reasoner"] ** 0.5),
            "qbound_endpoint_grad_norm": float(totals["endpoint"] ** 0.5),
        }

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_grad_norm",
            "qbound_input_adapter_grad_norm",
            "qbound_dynamics_grad_norm",
            "qbound_video_prior_grad_norm",
            "qbound_functional_composer_grad_norm",
            "qbound_reasoner_grad_norm",
            "qbound_endpoint_grad_norm",
            "qbound_functional_composition_weight",
            "qbound_boundary_equilibrium_weight",
            "qbound_functional_composition_loss",
            "qbound_boundary_equilibrium_loss",
            "qbound_functional_gap_mean",
            "qbound_functional_gap_abs_mean",
            "qbound_functional_keep_better_ratio",
            "qbound_functional_merge_better_ratio",
            "qbound_functional_target_cut_ratio",
            "qbound_functional_confidence_mean",
            "qbound_functional_cut_target_agreement",
            "qbound_functional_left_right_margin_gap",
            "qbound_equilibrium_selected_margin",
            "qbound_equilibrium_active_left_ratio",
            "qbound_equilibrium_active_right_ratio",
            "qbound_equilibrium_active_any_ratio",
            "qbound_equilibrium_trim_left_action_ratio",
            "qbound_equilibrium_expand_left_action_ratio",
            "qbound_equilibrium_trim_right_action_ratio",
            "qbound_equilibrium_expand_right_action_ratio",
            "qbound_equilibrium_left_gain_mean",
            "qbound_equilibrium_right_gain_mean",
            "qbound_equilibrium_left_confidence_mean",
            "qbound_equilibrium_right_confidence_mean",
            "qbound_equilibrium_start_shift_abs_mean",
            "qbound_equilibrium_end_shift_abs_mean",
            "qbound_generic_input_token_cosine",
            "qbound_generic_input_delta_ratio",
            "qbound_cut_mean",
            "qbound_cut_std",
            "qbound_prior_cut_mean",
            "qbound_question_residual_abs_mean",
            "qbound_hard_event_count",
            "qbound_soft_effective_event_count",
            "qbound_one_event_indicator",
            "qbound_all_split_indicator",
            "qbound_functional_cut_hard_ratio",
            "qbound_prior_functional_cut_l1",
            "qbound_start_entropy",
            "qbound_end_entropy",
            "qbound_start_prob_top1",
            "qbound_end_prob_top1",
            "qbound_selected_span_length",
            "qbound_selected_span_fraction",
            "qbound_hard_soft_js",
            "qbound_cra_qbound_js",
            "qbound_training_hard_forward_l1",
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def get_loss(self, batch, qbound_mix=1.0):
        del qbound_mix
        device = self._model_device()

        answer_id = batch["answer_id"]
        answer = batch["answer"].to(device, non_blocking=True)
        video_frames = batch["video_frames"].to(device, non_blocking=True)
        question = batch["question"].to(device, non_blocking=True)
        qsns_id = batch["qsns_id"]
        qsns_token_ids = batch["qsns_token_ids"]
        video_len = batch["video_len"]

        video_mask = (
            get_mask(
                video_len,
                self.cfgs["dataset"]["max_feats"],
            ).to(device, non_blocking=True)
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_device = self._prepare_labels(answer_id, device)
        prediction_dict = {}
        target_dict = {}

        # Original CRA VQ objective remains a testbed objective and keeps
        # Q-Bound disabled, exactly as in prior controlled runs.
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

        # MAIN VQA.  Forward numerics use hard H*, while RefineDecoder's
        # straight-through training key lets the factual QA loss also train
        # Q-Bound start/end distributions.
        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
            qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
        )

        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]
        main_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=main_video_proj,
            detach_answer=False,
        )

        prediction_dict["vqa_pred"] = main_logits
        target_dict["vqa_target"] = answer_id_device
        prediction_dict["vqa_align_pred"] = (
            gqa_output[2]["time_param"]["grounding_feature"]
        )
        target_dict["vqa_align_gt"] = gqa_output[2]["align_gt"]

        # Preserve the repository's factual CRA task objective.
        loss = self.criterion(prediction_dict, target_dict)

        settings = self._qbound_v200_loss_settings()
        qbound_aux = self._get_qbound_aux(gqa_output)
        qbound_logs = {}

        if self._qbound_enabled():
            self._validate_v200_contract(qbound_aux)

            if settings["functional_composition_weight"] > 0.0:
                comp_loss, comp_diag = (
                    self._qbound_functional_composition_objective(
                        qbound_aux=qbound_aux,
                        labels=answer_id_device,
                        settings=settings,
                    )
                )
                weighted_comp = (
                    settings["functional_composition_weight"] * comp_loss
                )
                loss["qbound_functional_composition_loss"] = comp_loss
                loss["qbound_functional_composition_weighted_loss"] = weighted_comp
                loss["total_loss"] = loss["total_loss"] + weighted_comp
                qbound_logs.update(
                    {f"qbound_{name}": value for name, value in comp_diag.items()}
                )

            if settings["boundary_equilibrium_weight"] > 0.0:
                eq_loss, eq_diag = self._qbound_boundary_equilibrium_objective(
                    qbound_aux=qbound_aux,
                    labels=answer_id_device,
                    settings=settings,
                )
                weighted_eq = settings["boundary_equilibrium_weight"] * eq_loss
                loss["qbound_boundary_equilibrium_loss"] = eq_loss
                loss["qbound_boundary_equilibrium_weighted_loss"] = weighted_eq
                loss["total_loss"] = loss["total_loss"] + weighted_eq
                qbound_logs.update(
                    {f"qbound_{name}": value for name, value in eq_diag.items()}
                )

            qbound_logs.update(
                {
                    "qbound_functional_composition_weight": settings[
                        "functional_composition_weight"
                    ],
                    "qbound_boundary_equilibrium_weight": settings[
                        "boundary_equilibrium_weight"
                    ],
                }
            )

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

        self.monitor.reset_kv(f"{self.cfgs['stat']['monitor']['metric']}")
        self.monitor.reset_kv("total_loss")
        for key in self.cfgs["optim"]["loss"].keys():
            if key in {"ce", "align"}:
                continue
            self.monitor.reset_kv(f"{key}_loss")

        if self._qbound_enabled():
            self.monitor.reset_kv("qbound_functional_composition_loss")
            self.monitor.reset_kv("qbound_functional_composition_weighted_loss")
            self.monitor.reset_kv("qbound_boundary_equilibrium_loss")
            self.monitor.reset_kv("qbound_boundary_equilibrium_weighted_loss")
            for key in self._qbound_monitor_keys():
                self.monitor.reset_kv(key)

        for index, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, monitor_metric, qbound_stats = self.get_loss(
                batch,
                qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
            )

            self.monitor.logkv_mean("acc", monitor_metric)
            for name, value in qbound_stats.items():
                self.monitor.logkv_mean(name, value)

            self.optimizer.zero_grad()
            loss["total_loss"].backward()

            if self._qbound_enabled():
                self.monitor.logkv_mean(
                    "qbound_grad_norm",
                    self._qbound_grad_norm(),
                )
                for grad_name, grad_value in (
                    self._qbound_component_grad_norms().items()
                ):
                    self.monitor.logkv_mean(grad_name, grad_value)

            clip_value = self.cfgs["optim"].get("clip", 0)
            if clip_value:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=clip_value,
                )

            self.optimizer.step()
            self.lr_scheduler.step()

            progress = float(index + 1) / len(self.train_dataloader)
            end_time = time.time()
            _, remaining_batch_time, remaining_epoch_time = get_remain_time(
                start_time,
                end_time,
                index,
                len(self.train_dataloader),
                epoch,
                self.cfgs["optim"]["epochs"],
            )

            print_info = (
                f"Progress: {progress:3.2%} "
                f"(rbt-ret: {remaining_batch_time}-{remaining_epoch_time}) | "
                f"Train {self.cfgs['stat']['monitor']['metric']}: "
                f"{monitor_metric:3.2%}"
            )
            for loss_name, loss_value in loss.items():
                running_loss = loss_value.detach().float().cpu().item()
                self.monitor.logkv_mean(loss_name, running_loss)
                print_info += f" {loss_name}: {running_loss:.4f}"
            print(f"\r{print_info}", end="", flush=True)

        table = [[], []]
        for name, value in self.monitor.name2val.items():
            table[0].append(f"mean {name}")
            table[1].append(f"{value:.4f}")
        self.monitor.log_info(
            f"\n{tabulate(table, headers='firstrow', tablefmt='grid')}",
            end="\n",
        )

        print("Eval Stage")
        val_score, val_result = self.eval(self.val_dataloader)
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
            test_result = val_result
        else:
            test_score, test_result = self.eval(self.test_dataloader)

        score = {
            **{f"val_{key}": value for key, value in val_score.items()},
            **{f"test_{key}": value for key, value in test_score.items()},
        }
        self.print_score(score)

        wandb_data = {"epoch": epoch}
        for name, value in self.monitor.name2val.items():
            wandb_data[f"train/{name}"] = value
        for split_name, split_dict in (("val", val_score), ("test", test_score)):
            for key, value in split_dict.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb_data[f"{split_name}/{key}_{sub_key}"] = sub_value
                else:
                    wandb_data[f"{split_name}/{key}"] = value
        wandb.log(wandb_data)
        return score, test_result

    # ------------------------------------------------------------------
    # Reporting / visualization
    # ------------------------------------------------------------------
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
                    f"\n{split.title()} Set\t V2.0.0 functional-composition acc: "
                    f"{score[acc_key]:.2%}"
                )

            prehoc_key = f"{split}_pre-hoc"
            if prehoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} V2.0.0 hard H* =========="
                )
                _print_score(score[prehoc_key])

            posthoc_key = f"{split}_post-hoc"
            if posthoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} post-hoc =========="
                )
                _print_score(score[posthoc_key])

            merge_key = f"{split}_merge"
            if merge_key in score:
                self.monitor.log_info(
                    f"\n========== {split} merge =========="
                )
                _print_score(score[merge_key])

        qbound_keys = [
            key for key in score
            if key.startswith("val_qbound_")
            or key.startswith("test_qbound_")
        ]
        if qbound_keys:
            self.monitor.log_info(
                "\n================ Q-Bound V2.0.0 diagnostics ================"
            )
            for key in sorted(qbound_keys):
                self.monitor.log_info(f"{key}: {score[key]:.6f}")

        temporal_prefixes = (
            "val_QBound-",
            "test_QBound-",
            "val_QBoundSoft-",
            "test_QBoundSoft-",
            "val_CRA-",
            "test_CRA-",
        )
        temporal_keys = [
            key for key in score if key.startswith(temporal_prefixes)
        ]
        if temporal_keys:
            self.monitor.log_info(
                "\n============ Temporal-key faithfulness ============"
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
        image_path = os.path.join(self.save_dir, "record", "grounding")
        os.makedirs(image_path, exist_ok=True)

        video_question_ids = list(posthoc.keys())
        sample_count = min(32, len(video_question_ids))
        shuffle(video_question_ids)

        for index in range(sample_count):
            video_question_id = video_question_ids[index]
            video_id, question_id = video_question_id.split("_")
            duration = gt[video_id]["duration"]
            ground_truth = gt[video_id]["location"][question_id]
            posthoc_ground = posthoc[video_question_id]
            attention_ground = np.round(
                np.asarray(attn[video_question_id]) * duration,
                1,
            )
            merged_ground = merge[video_question_id]

            figure, axis = plt.subplots()
            axis.plot(attention_ground, [3, 3], linewidth=6)
            axis.plot(posthoc_ground, [2, 2], linewidth=6)
            axis.plot(merged_ground, [1, 1], linewidth=6)
            axis.plot(ground_truth[0], [0, 0], linewidth=6)
            if probs is not None:
                probability = np.asarray(probs[video_question_id]) * 3
                axis.plot(
                    np.linspace(0, duration, len(probability)),
                    probability,
                    linewidth=3,
                    label="Temporal relevance",
                )
            plt.tight_layout()
            figure.savefig(
                os.path.join(
                    image_path,
                    f"{epoch}_{video_id}_{question_id}.png",
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
        qbound_keyframe = {}
        qbound_soft_keyframe = {}
        cra_keyframe = {}

        qbound_stat_sums = collections.defaultdict(float)
        qbound_stat_count = 0
        saw_time_param = False
        settings = self._qbound_v200_loss_settings()

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
                    device,
                    non_blocking=True,
                )
                question = batch["question"].to(device, non_blocking=True)
                question_id = batch["question_id"]
                video_len = batch["video_len"]
                video_mask = get_mask(
                    video_len,
                    video_frames.size(1),
                ).to(device, non_blocking=True)

                batch_size = answer_id.size(0)
                count += batch_size
                labels_device = self._prepare_labels(answer_id, device)
                labels_cpu = answer_id.reshape(-1).cpu()

                fusion_proj, answer_proj, model_outputs = self.model(
                    video_frames,
                    video_mask,
                    answer=answer,
                    answer_id=answer_id,
                    mode="VQA",
                    question=question,
                    qbound_mix=(1.0 if self._qbound_enabled() else 0.0),
                )

                main_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=fusion_proj,
                    detach_answer=False,
                )
                predicted = main_logits.argmax(dim=-1).cpu()
                score["acc"] += (predicted == labels_cpu).sum().item()

                time_param = model_outputs.get("time_param")
                qbound_aux = None
                eval_wrapper = None
                if isinstance(time_param, dict):
                    saw_time_param = True
                    eval_wrapper = (None, None, {"time_param": time_param})
                    qbound_aux = time_param.get(
                        "qbound_aux",
                        time_param.get("gpeb_aux"),
                    )

                extra_logs = {}
                if self._qbound_enabled() and qbound_aux is not None:
                    self._validate_v200_contract(qbound_aux)
                    if settings["functional_composition_weight"] > 0.0:
                        comp_loss, comp_diag = (
                            self._qbound_functional_composition_objective(
                                qbound_aux=qbound_aux,
                                labels=labels_device,
                                settings=settings,
                            )
                        )
                        extra_logs["qbound_functional_composition_loss"] = float(
                            comp_loss.item()
                        )
                        extra_logs.update(
                            {f"qbound_{name}": value for name, value in comp_diag.items()}
                        )
                    if settings["boundary_equilibrium_weight"] > 0.0:
                        eq_loss, eq_diag = (
                            self._qbound_boundary_equilibrium_objective(
                                qbound_aux=qbound_aux,
                                labels=labels_device,
                                settings=settings,
                            )
                        )
                        extra_logs["qbound_boundary_equilibrium_loss"] = float(
                            eq_loss.item()
                        )
                        extra_logs.update(
                            {f"qbound_{name}": value for name, value in eq_diag.items()}
                        )

                    batch_stats = self._collect_qbound_stats(
                        eval_wrapper,
                        extra_logs=extra_logs,
                    )
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
                batch_qbound_key = None
                batch_soft_key = None
                batch_cra_key = None

                if isinstance(time_param, dict):
                    start_time = time_param["start"].detach().cpu().numpy()
                    end_time = time_param["end"].detach().cpu().numpy()
                    batch_ground_pre = np.column_stack((start_time, end_time))
                    batch_keyframe = time_param["key"].detach().cpu().numpy()
                    batch_ori_keyframe = time_param["ori_key"].detach().cpu().numpy()

                    if qbound_aux is not None:
                        q_key = qbound_aux.get("qbound_key")
                        c_key = qbound_aux.get("cra_key")
                        soft_key = qbound_aux.get("soft_temporal_prob")
                        if torch.is_tensor(q_key):
                            batch_qbound_key = q_key.detach().cpu().numpy()
                        if torch.is_tensor(c_key):
                            batch_cra_key = c_key.detach().cpu().numpy()
                        if torch.is_tensor(soft_key):
                            soft_key = self._normalize_distribution(
                                soft_key.detach()
                            )
                            batch_soft_key = soft_key.cpu().numpy()

                for batch_index, qid in enumerate(question_id):
                    results[qid] = {
                        "prediction": int(predicted.numpy()[batch_index]),
                        "answer": int(labels_cpu.numpy()[batch_index]),
                    }
                    ground_res[qid] = video_attention[batch_index].tolist()

                    if batch_ground_pre is not None:
                        ground_pre[qid] = batch_ground_pre[batch_index].tolist()
                    if batch_keyframe is not None:
                        keyframe[qid] = batch_keyframe[batch_index].tolist()
                    if batch_ori_keyframe is not None:
                        ori_keyframe[qid] = batch_ori_keyframe[batch_index].tolist()
                    if batch_qbound_key is not None:
                        qbound_keyframe[qid] = batch_qbound_key[batch_index].tolist()
                    if batch_soft_key is not None:
                        qbound_soft_keyframe[qid] = batch_soft_key[batch_index].tolist()
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
                    "Q-Bound V2.0.0 requires Metric.temporal_key_metrics()."
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
            parameter.numel() for parameter in self.model.parameters()
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
            test_score, _ = self.eval(self.test_dataloader, epoch, True)

        score = {
            **{f"val_{key}": value for key, value in val_score.items()},
            **{f"test_{key}": value for key, value in test_score.items()},
        }
        self.print_score(score)
        return score
