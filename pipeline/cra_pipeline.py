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
    CRA training/evaluation pipeline with Q-Bound Version 1.12.

    V1.12 is a structural redesign.  Q-Bound no longer uses the historical
    COMPOSE -> BAPGR -> PPBR -> PSRC -> donor-counterfactual chain.  The active
    temporal path is:

        generic video tokens + question-only features
            -> dynamic DECOMPOSE
            -> ordered event RELATE
            -> frame-exact temporal hypothesis bank
            -> question-only hypothesis scorer
            -> hard winning hypothesis H*
            -> exact temporal grounding

    The key new training signal is *privileged answer-aware temporal
    distillation*.  Candidate-answer information never enters QBoundCore and is
    not required for grounding at inference.  Instead, CRA exposes a
    stop-gradient training-only critic that scores every temporal hypothesis and
    its complement in answer space.

    For a hypothesis H, the critic evaluates four properties without temporal
    ground truth:

      1) sufficiency:
            H should preserve the full-video answer decision;

      2) irredundancy:
            trimming either valid boundary should not improve the answer
            explanation and should preferably weaken it;

      3) completeness:
            expanding either valid boundary should not materially improve the
            answer explanation;

      4) exclusivity:
            H should explain the answer better than its temporal complement.

    These comparisons are computed from per-sample z-normalized answer decision
    margins and combined with equal, fixed weights.  There is deliberately no
    span-length/minimality reward: shorter is not automatically better.  Extra
    evidence must improve the answer explanation to earn its place.

    The detached critic utility induces a listwise target distribution over
    temporal hypotheses.  Q-Bound's answer-free student scorer learns this
    distribution with KL distillation:

        L = L_CRA/main
            + lambda_align * L_weak_align
            + lambda_decomp * L_decomp
            + lambda_rank * KL(q_teacher || p_QBound)

    By default the ranking objective is applied only when the full-video critic
    predicts the ground-truth answer correctly.  This prevents an unreliable
    answer critic from teaching temporal ranking on samples it cannot yet solve.

    Important invariants:
      * no temporal GT / IoU / IoP training loss;
      * no answer features inside QBoundCore;
      * no CRA temporal handover into Q-Bound;
      * no target span length or minimality objective;
      * no donor-video intervention;
      * no differentiable all-span answer marginalization;
      * teacher/critic tensors are stop-gradient;
      * inference grounding uses only video + question.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        self.qbound_version = "1.12"
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
        """Compatibility helper: V1.12 has no CRA -> Q-Bound handover."""
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
    # V1.12 objective configuration
    # ------------------------------------------------------------------
    def _qbound_v112_loss_settings(self):
        """
        Preferred compact YAML:

            model:
              qbound:
                weak_align_weight: 0.10
                weak_align_temperature: 0.07
                decomposition:
                  loss_weight: 0.10
                hypothesis_teacher:
                  enabled: true
                  loss_weight: 0.10
                  correct_only: true

        The four teacher criteria use equal fixed weights by design.  Their
        scales are stabilized by per-sample z-normalization, so V1.12 does not
        introduce separate sufficiency/trim/expand/complement coefficients or
        an answer-logit margin hyperparameter.
        """
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        decomp_cfg = qbound_cfg.get("decomposition", {})
        if not isinstance(decomp_cfg, dict):
            decomp_cfg = {}

        teacher_cfg = qbound_cfg.get("hypothesis_teacher", {})
        if teacher_cfg is None:
            teacher_cfg = {}
        if not isinstance(teacher_cfg, dict):
            raise TypeError(
                "model.qbound.hypothesis_teacher must be a mapping/dict."
            )

        align_weight = float(
            qbound_cfg.get(
                "weak_align_weight",
                model_cfg.get("qbound_weak_align_weight", 0.10),
            )
        )
        align_temperature = float(
            qbound_cfg.get(
                "weak_align_temperature",
                model_cfg.get("qbound_weak_align_temperature", 0.07),
            )
        )
        decomp_weight = float(
            qbound_cfg.get(
                "decomp_weight",
                decomp_cfg.get(
                    "loss_weight",
                    model_cfg.get("qbound_decomp_weight", 0.10),
                ),
            )
        )
        teacher_enabled = bool(teacher_cfg.get("enabled", True))
        teacher_loss_weight = float(teacher_cfg.get("loss_weight", 0.10))
        teacher_correct_only = bool(teacher_cfg.get("correct_only", True))

        if align_weight < 0.0:
            raise ValueError("qbound weak_align_weight must be >= 0.")
        if align_temperature <= 0.0:
            raise ValueError("qbound weak_align_temperature must be > 0.")
        if decomp_weight < 0.0:
            raise ValueError("qbound decomposition loss_weight must be >= 0.")
        if teacher_loss_weight < 0.0:
            raise ValueError(
                "qbound hypothesis_teacher.loss_weight must be >= 0."
            )
        if teacher_enabled and teacher_loss_weight <= 0.0:
            raise ValueError(
                "V1.12 hypothesis teacher is enabled but its loss_weight is zero."
            )

        return {
            "align_weight": align_weight,
            "align_temperature": align_temperature,
            "decomp_weight": decomp_weight,
            "teacher_enabled": teacher_enabled,
            "teacher_loss_weight": teacher_loss_weight,
            "teacher_correct_only": teacher_correct_only,
        }

    # ------------------------------------------------------------------
    # Existing weak supervision retained from V1.11
    # ------------------------------------------------------------------
    @staticmethod
    def _qbound_inbatch_alignment(
        evidence_visual_token,
        question_state,
        temperature,
    ):
        if evidence_visual_token is None or question_state is None:
            raise ValueError(
                "Q-Bound V1.12 weak alignment requires "
                "evidence_visual_token and qa_state."
            )
        if evidence_visual_token.ndim != 2 or question_state.ndim != 2:
            raise ValueError(
                "Q-Bound weak-alignment tensors must both be [B,H]."
            )
        if evidence_visual_token.shape != question_state.shape:
            raise ValueError(
                "Q-Bound weak-alignment shape mismatch: "
                f"{tuple(evidence_visual_token.shape)} vs "
                f"{tuple(question_state.shape)}."
            )

        batch_size = evidence_visual_token.size(0)
        if batch_size < 2:
            return evidence_visual_token.sum() * 0.0, 0.0

        visual = F.normalize(
            evidence_visual_token,
            p=2,
            dim=-1,
            eps=1e-6,
        )
        semantic = F.normalize(
            question_state.detach(),
            p=2,
            dim=-1,
            eps=1e-6,
        )
        logits = torch.matmul(visual, semantic.transpose(0, 1))
        logits = logits / float(temperature)
        labels = torch.arange(batch_size, device=logits.device)
        visual_to_text = F.cross_entropy(logits, labels)
        text_to_visual = F.cross_entropy(logits.transpose(0, 1), labels)
        align_loss = 0.5 * (visual_to_text + text_to_visual)
        retrieval_acc = (
            logits.detach().argmax(dim=-1) == labels
        ).float().mean().item()
        return align_loss, retrieval_acc

    @staticmethod
    def _qbound_decomposition_loss(qbound_aux):
        decomp_loss = qbound_aux.get("decomp_loss")
        if not torch.is_tensor(decomp_loss):
            raise KeyError(
                "Q-Bound V1.12 requires qbound_aux['decomp_loss']."
            )
        value = decomp_loss.mean()
        if not torch.isfinite(value):
            raise FloatingPointError(
                "Q-Bound V1.12 decomposition loss is NaN/Inf."
            )
        return value

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
    # V1.12 answer-aware hypothesis teacher
    # ------------------------------------------------------------------
    @staticmethod
    def _answer_decision_margin(logits, labels):
        """
        Decision margin:

            z_y - logsumexp_{a != y}(z_a)

        Works with logits [B,C] or [B,S,C].
        """
        if logits.ndim not in {2, 3}:
            raise ValueError(
                "Decision-margin logits must be [B,C] or [B,S,C]."
            )
        if logits.size(-1) < 2:
            raise ValueError(
                "V1.12 answer-aware teacher requires at least two answers."
            )

        labels = labels.reshape(-1)
        if labels.numel() != logits.size(0):
            raise ValueError("labels do not match teacher-logit batch size.")

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
        wrong.scatter_(
            -1,
            gather_index,
            float("-inf"),
        )
        wrong_lse = torch.logsumexp(wrong, dim=-1)
        return correct - wrong_lse

    @staticmethod
    def _masked_zscore(values, valid_mask, eps=1e-5):
        if values.ndim != 2 or valid_mask.shape != values.shape:
            raise ValueError(
                "masked z-score expects values/mask with shape [B,S]."
            )
        valid = valid_mask.to(values.dtype)
        count = valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (values * valid).sum(dim=-1, keepdim=True) / count
        variance = (
            ((values - mean).pow(2) * valid).sum(dim=-1, keepdim=True)
            / count
        )
        std = torch.sqrt(variance + eps)
        normalized = (values - mean) / std
        normalized = torch.where(
            valid_mask,
            normalized,
            torch.zeros_like(normalized),
        )
        return normalized, mean.squeeze(-1), std.squeeze(-1)

    @staticmethod
    def _masked_softmax(logits, mask, dim=-1):
        if logits.shape != mask.shape:
            raise ValueError(
                "masked softmax requires logits/mask with identical shape."
            )
        mask = mask.bool()
        if not torch.all(mask.any(dim=dim)):
            raise ValueError(
                "Every sample needs at least one valid temporal hypothesis."
            )
        masked_logits = logits.masked_fill(~mask, -1e4)
        probability = F.softmax(masked_logits, dim=dim)
        probability = probability * mask.to(probability.dtype)
        return probability / probability.sum(
            dim=dim,
            keepdim=True,
        ).clamp_min(1e-8)

    @staticmethod
    def _masked_log_softmax(logits, mask, dim=-1):
        if logits.shape != mask.shape:
            raise ValueError(
                "masked log-softmax requires logits/mask with identical shape."
            )
        mask = mask.bool()
        if not torch.all(mask.any(dim=dim)):
            raise ValueError(
                "Every sample needs at least one valid temporal hypothesis."
            )
        masked_logits = logits.masked_fill(~mask, -1e4)
        return F.log_softmax(masked_logits, dim=dim)

    @staticmethod
    def _gather_neighbor(values, indices):
        if values.ndim != 2 or indices.shape != values.shape:
            raise ValueError(
                "neighbor gather expects values/indices with shape [B,S]."
            )
        safe = indices.clamp_min(0).to(torch.long)
        return values.gather(dim=1, index=safe)

    def _qbound_v112_hypothesis_distillation(
        self,
        model_outputs,
        labels,
        settings,
    ):
        """
        Build the detached privileged teacher and listwise ranking loss.

        The critic logits come from CRA.forward and are already stop-gradient.
        This function converts them into a temporal utility without any GT span.

        Utility terms use equal fixed weights after per-sample margin z-score:

            sufficiency  : log sigmoid(M_H - M_full)
            trim         : log sigmoid(M_H - M_trim)
            expand       : log sigmoid(M_H - M_expand)
            exclusivity  : log sigmoid(M_H - M_complement)

        Only valid comparisons are included, and utility is divided by the
        number of active comparisons for each hypothesis.  Therefore boundary
        hypotheses do not receive a free advantage merely because one trim or
        expansion neighbor does not exist.
        """
        if not isinstance(model_outputs, dict):
            raise TypeError(
                "V1.12 hypothesis distillation requires model_outputs dict."
            )

        required_names = (
            "qbound_hypothesis_logits",
            "qbound_hypothesis_valid_mask",
            "qbound_hypothesis_trim_left_index",
            "qbound_hypothesis_trim_right_index",
            "qbound_hypothesis_expand_left_index",
            "qbound_hypothesis_expand_right_index",
            "qbound_hypothesis_trim_left_valid_mask",
            "qbound_hypothesis_trim_right_valid_mask",
            "qbound_hypothesis_expand_left_valid_mask",
            "qbound_hypothesis_expand_right_valid_mask",
            "qbound_teacher_hypothesis_answer_logits",
            "qbound_teacher_complement_answer_logits",
            "qbound_teacher_full_video_answer_logits",
            "qbound_teacher_hypothesis_valid_mask",
            "qbound_teacher_complement_valid_mask",
            "qbound_teacher_critic_stop_gradient",
            "qbound_teacher_critic_projection_dropout_enabled",
            "qbound_teacher_uses_answer_information",
            "qbound_teacher_affects_grounding_forward",
            "qbound_grounding_requires_answer_at_inference",
            "qbound_selected_hypothesis_index",
        )
        missing = [
            name for name in required_names
            if model_outputs.get(name) is None
        ]
        if missing:
            raise KeyError(
                "V1.12 hypothesis-teacher outputs are missing: "
                + ", ".join(missing)
            )

        student_logits = model_outputs["qbound_hypothesis_logits"]
        valid_mask = model_outputs["qbound_hypothesis_valid_mask"].bool()
        teacher_valid_mask = model_outputs[
            "qbound_teacher_hypothesis_valid_mask"
        ].bool()
        complement_valid = model_outputs[
            "qbound_teacher_complement_valid_mask"
        ].bool()

        hypothesis_answer_logits = model_outputs[
            "qbound_teacher_hypothesis_answer_logits"
        ].detach()
        complement_answer_logits = model_outputs[
            "qbound_teacher_complement_answer_logits"
        ].detach()
        full_video_answer_logits = model_outputs[
            "qbound_teacher_full_video_answer_logits"
        ].detach()

        if student_logits.ndim != 2:
            raise ValueError("qbound_hypothesis_logits must be [B,S].")
        if valid_mask.shape != student_logits.shape:
            raise ValueError(
                "qbound_hypothesis_valid_mask must match student logits."
            )
        if not torch.equal(valid_mask, teacher_valid_mask):
            raise RuntimeError(
                "Student/teacher hypothesis-valid masks disagree in V1.12."
            )
        if hypothesis_answer_logits.shape[:2] != student_logits.shape:
            raise ValueError(
                "Teacher hypothesis logits must match [B,S] student bank."
            )
        if complement_answer_logits.shape != hypothesis_answer_logits.shape:
            raise ValueError(
                "Teacher complement logits must match hypothesis logits."
            )
        if full_video_answer_logits.ndim != 2:
            raise ValueError("Teacher full-video logits must be [B,C].")

        for flag_name, expected in (
            ("qbound_teacher_critic_stop_gradient", 1.0),
            ("qbound_teacher_critic_projection_dropout_enabled", 0.0),
            ("qbound_teacher_uses_answer_information", 1.0),
            ("qbound_teacher_affects_grounding_forward", 0.0),
            ("qbound_grounding_requires_answer_at_inference", 0.0),
        ):
            flag = model_outputs[flag_name]
            if not torch.is_tensor(flag):
                raise TypeError(f"{flag_name} must be a tensor.")
            actual = float(flag.detach().item())
            if actual != expected:
                raise RuntimeError(
                    f"V1.12 invariant {flag_name}={expected} violated; "
                    f"got {actual}."
                )

        labels = labels.reshape(-1)
        if labels.numel() != student_logits.size(0):
            raise ValueError("V1.12 labels do not match hypothesis batch size.")

        with torch.no_grad():
            margin_h = self._answer_decision_margin(
                hypothesis_answer_logits,
                labels,
            )
            margin_comp = self._answer_decision_margin(
                complement_answer_logits,
                labels,
            )
            margin_full = self._answer_decision_margin(
                full_video_answer_logits,
                labels,
            )

            margin_h_z, margin_mean, margin_std = self._masked_zscore(
                margin_h,
                valid_mask,
            )
            margin_full_z = (
                margin_full - margin_mean
            ) / margin_std.clamp_min(1e-5)
            margin_comp_z = (
                margin_comp - margin_mean[:, None]
            ) / margin_std[:, None].clamp_min(1e-5)

            # Fixed equal-weight comparisons.  log-sigmoid is bounded above by
            # zero, scale-robust after z-normalization, and does not require a
            # hand-tuned answer-logit margin.
            sufficiency_score = F.logsigmoid(
                margin_h_z - margin_full_z[:, None]
            )
            sufficiency_score = (
                sufficiency_score * valid_mask.to(sufficiency_score.dtype)
            )
            utility_sum = sufficiency_score.clone()
            utility_count = valid_mask.to(utility_sum.dtype)

            component_sums = {
                "sufficiency": sufficiency_score.clone(),
                "trim": torch.zeros_like(utility_sum),
                "expand": torch.zeros_like(utility_sum),
                "exclusivity": torch.zeros_like(utility_sum),
            }
            component_counts = {
                "sufficiency": valid_mask.to(utility_sum.dtype),
                "trim": torch.zeros_like(utility_sum),
                "expand": torch.zeros_like(utility_sum),
                "exclusivity": torch.zeros_like(utility_sum),
            }

            neighbor_specs = (
                (
                    "trim",
                    "qbound_hypothesis_trim_left_index",
                    "qbound_hypothesis_trim_left_valid_mask",
                ),
                (
                    "trim",
                    "qbound_hypothesis_trim_right_index",
                    "qbound_hypothesis_trim_right_valid_mask",
                ),
                (
                    "expand",
                    "qbound_hypothesis_expand_left_index",
                    "qbound_hypothesis_expand_left_valid_mask",
                ),
                (
                    "expand",
                    "qbound_hypothesis_expand_right_index",
                    "qbound_hypothesis_expand_right_valid_mask",
                ),
            )

            for component_name, index_name, valid_name in neighbor_specs:
                indices = model_outputs[index_name].detach().long()
                compare_valid = (
                    model_outputs[valid_name].detach().bool()
                    & valid_mask
                )
                neighbor_margin = self._gather_neighbor(
                    margin_h_z,
                    indices,
                )
                score = F.logsigmoid(margin_h_z - neighbor_margin)
                score = score * compare_valid.to(score.dtype)
                utility_sum = utility_sum + score
                utility_count = utility_count + compare_valid.to(score.dtype)
                component_sums[component_name] = (
                    component_sums[component_name] + score
                )
                component_counts[component_name] = (
                    component_counts[component_name]
                    + compare_valid.to(score.dtype)
                )

            exclusivity_valid = complement_valid & valid_mask
            exclusivity_score = F.logsigmoid(
                margin_h_z - margin_comp_z
            )
            exclusivity_score = (
                exclusivity_score
                * exclusivity_valid.to(exclusivity_score.dtype)
            )
            utility_sum = utility_sum + exclusivity_score
            utility_count = (
                utility_count
                + exclusivity_valid.to(exclusivity_score.dtype)
            )
            component_sums["exclusivity"] = exclusivity_score
            component_counts["exclusivity"] = exclusivity_valid.to(
                exclusivity_score.dtype
            )

            teacher_utility = utility_sum / utility_count.clamp_min(1.0)
            teacher_utility = teacher_utility.masked_fill(~valid_mask, -1e4)
            teacher_probability = self._masked_softmax(
                teacher_utility,
                valid_mask,
            ).detach()

            full_video_correct = (
                full_video_answer_logits.argmax(dim=-1) == labels
            )
            if bool(settings["teacher_correct_only"]):
                active_sample = full_video_correct
            else:
                active_sample = torch.ones_like(full_video_correct)

            teacher_top1_index = teacher_probability.argmax(dim=-1)
            student_top1_index = student_logits.detach().masked_fill(
                ~valid_mask,
                -1e4,
            ).argmax(dim=-1)
            selected_index = model_outputs[
                "qbound_selected_hypothesis_index"
            ].detach().long().reshape(-1)

            batch_index = torch.arange(
                student_logits.size(0),
                device=student_logits.device,
            )
            selected_teacher_prob = teacher_probability[
                batch_index,
                selected_index,
            ]
            selected_teacher_utility = teacher_utility[
                batch_index,
                selected_index,
            ]
            best_teacher_utility = teacher_utility[
                batch_index,
                teacher_top1_index,
            ]

            # Rank 1 is best.  Percentile 1.0 means teacher-best, 0.0 means
            # worst among valid hypotheses.
            greater = (
                teacher_utility
                > selected_teacher_utility[:, None]
            ) & valid_mask
            selected_rank = 1.0 + greater.sum(dim=-1).to(torch.float32)
            valid_count = valid_mask.sum(dim=-1).clamp_min(1).to(torch.float32)
            selected_rank_percentile = torch.where(
                valid_count > 1.0,
                1.0 - (selected_rank - 1.0) / (valid_count - 1.0),
                torch.ones_like(valid_count),
            )

            teacher_entropy = -torch.sum(
                teacher_probability
                * torch.log(teacher_probability.clamp_min(1e-8)),
                dim=-1,
            )
            max_entropy = torch.log(valid_count.clamp_min(1.0))
            teacher_entropy_ratio = torch.where(
                valid_count > 1.0,
                teacher_entropy / max_entropy.clamp_min(1e-6),
                torch.zeros_like(teacher_entropy),
            )

            teacher_best_answer_logits = hypothesis_answer_logits[
                batch_index,
                teacher_top1_index,
            ]
            selected_answer_logits = hypothesis_answer_logits[
                batch_index,
                selected_index,
            ]
            selected_complement_logits = complement_answer_logits[
                batch_index,
                selected_index,
            ]

            component_means = {}
            for component_name in component_sums:
                numerator = component_sums[component_name].sum(dim=-1)
                denominator = component_counts[component_name].sum(
                    dim=-1
                ).clamp_min(1.0)
                component_means[component_name] = numerator / denominator

        # Student side remains differentiable.
        student_log_probability = self._masked_log_softmax(
            student_logits,
            valid_mask,
        )
        per_sample_kl = torch.sum(
            teacher_probability
            * (
                torch.log(teacher_probability.clamp_min(1e-8))
                - student_log_probability
            ),
            dim=-1,
        )

        active_float = active_sample.to(per_sample_kl.dtype)
        active_count = active_float.sum()
        if float(active_count.detach().item()) > 0.0:
            ranking_loss = (
                per_sample_kl * active_float
            ).sum() / active_count
        else:
            ranking_loss = student_logits.sum() * 0.0

        with torch.no_grad():
            student_probability = self._masked_softmax(
                student_logits.detach(),
                valid_mask,
            )
            student_entropy = -torch.sum(
                student_probability
                * torch.log(student_probability.clamp_min(1e-8)),
                dim=-1,
            )
            valid_count = valid_mask.sum(dim=-1).clamp_min(1).to(
                student_probability.dtype
            )
            student_entropy_ratio = torch.where(
                valid_count > 1.0,
                student_entropy
                / torch.log(valid_count).clamp_min(1e-6),
                torch.zeros_like(student_entropy),
            )

            selected_margin = margin_h[
                batch_index,
                selected_index,
            ]
            best_margin = margin_h[
                batch_index,
                teacher_top1_index,
            ]
            selected_complement_margin = margin_comp[
                batch_index,
                selected_index,
            ]

            diagnostics = {
                "teacher_active_ratio": active_float.mean().item(),
                "teacher_full_video_acc": full_video_correct.float().mean().item(),
                "teacher_top1_probability": teacher_probability.max(
                    dim=-1
                ).values.mean().item(),
                "teacher_entropy": teacher_entropy.mean().item(),
                "teacher_entropy_ratio": teacher_entropy_ratio.mean().item(),
                "teacher_effective_count": torch.exp(
                    teacher_entropy
                ).mean().item(),
                "student_top1_probability": student_probability.max(
                    dim=-1
                ).values.mean().item(),
                "student_entropy": student_entropy.mean().item(),
                "student_entropy_ratio": student_entropy_ratio.mean().item(),
                "teacher_student_top1_agreement": (
                    teacher_top1_index == student_top1_index
                ).float().mean().item(),
                "selected_is_teacher_top1_ratio": (
                    selected_index == teacher_top1_index
                ).float().mean().item(),
                "selected_teacher_probability": selected_teacher_prob.mean().item(),
                "selected_teacher_utility": selected_teacher_utility.mean().item(),
                "teacher_best_utility": best_teacher_utility.mean().item(),
                "teacher_best_minus_selected_utility": (
                    best_teacher_utility - selected_teacher_utility
                ).mean().item(),
                "selected_rank_percentile": selected_rank_percentile.mean().item(),
                "teacher_best_answer_acc": (
                    teacher_best_answer_logits.argmax(dim=-1) == labels
                ).float().mean().item(),
                "teacher_selected_answer_acc": (
                    selected_answer_logits.argmax(dim=-1) == labels
                ).float().mean().item(),
                "teacher_selected_complement_acc": (
                    selected_complement_logits.argmax(dim=-1) == labels
                ).float().mean().item(),
                "teacher_full_margin": margin_full.mean().item(),
                "teacher_selected_margin": selected_margin.mean().item(),
                "teacher_best_margin": best_margin.mean().item(),
                "teacher_selected_complement_margin": (
                    selected_complement_margin.mean().item()
                ),
                "teacher_sufficiency_component": (
                    component_means["sufficiency"].mean().item()
                ),
                "teacher_trim_component": (
                    component_means["trim"].mean().item()
                ),
                "teacher_expand_component": (
                    component_means["expand"].mean().item()
                ),
                "teacher_exclusivity_component": (
                    component_means["exclusivity"].mean().item()
                ),
            }

        return {
            "ranking_loss": ranking_loss,
            "teacher_probability": teacher_probability,
            "teacher_utility": teacher_utility,
            "teacher_top1_index": teacher_top1_index,
            "active_sample_mask": active_sample,
            **diagnostics,
        }

    # ------------------------------------------------------------------
    # V1.12 diagnostics
    # ------------------------------------------------------------------
    def _validate_v112_contract(self, qbound_aux):
        if qbound_aux is None:
            raise KeyError("Q-Bound V1.12 must expose qbound_aux.")

        required = {
            "qbound_version",
            "dynamic_event_count_enabled",
            "hypothesis_competition_enabled",
            "answer_features_used_in_core",
            "legacy_composer_enabled",
            "legacy_refiner_enabled",
            "legacy_counterfactual_donor_enabled",
            "generic_input_adapter_enabled",
            "generic_input_context_reuse",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "event_count",
            "event_capacity",
            "decomp_loss",
            "evidence_visual_token",
            "qa_state",
            "hypothesis_logits",
            "hypothesis_posterior",
            "hypothesis_valid_mask",
            "hypothesis_start_frame",
            "hypothesis_end_frame",
            "hypothesis_frame_weights",
            "hypothesis_complement_frame_weights",
            "selected_hypothesis_index",
            "selected_hypothesis_start_frame",
            "selected_hypothesis_end_frame",
            "selected_hypothesis_span_length",
            "main_temporal_prob",
            "qbound_key",
            "final_key",
            "teacher_hypothesis_answer_logits",
            "teacher_complement_answer_logits",
            "teacher_full_video_answer_logits",
            "teacher_critic_stop_gradient",
            "teacher_critic_projection_dropout_enabled",
            "teacher_uses_answer_information",
            "teacher_affects_grounding_forward",
            "grounding_requires_answer_at_inference",
        }
        missing = sorted(required.difference(qbound_aux))
        if missing:
            raise KeyError(
                "Missing Q-Bound V1.12 outputs: " + ", ".join(missing)
            )

        if str(qbound_aux["qbound_version"]) != self.qbound_version:
            raise RuntimeError(
                "Pipeline/Core Q-Bound version mismatch: "
                f"pipeline={self.qbound_version!r}, "
                f"core={qbound_aux['qbound_version']!r}."
            )

        def _flag(name, expected):
            value = qbound_aux[name]
            if not torch.is_tensor(value):
                raise TypeError(f"Q-Bound flag {name!r} must be a tensor.")
            actual = float(value.detach().item())
            if actual != float(expected):
                raise RuntimeError(
                    f"Q-Bound V1.12 flag {name!r} must be {expected}; "
                    f"got {actual}."
                )

        _flag("dynamic_event_count_enabled", 1.0)
        _flag("hypothesis_competition_enabled", 1.0)
        _flag("answer_features_used_in_core", 0.0)
        _flag("legacy_composer_enabled", 0.0)
        _flag("legacy_refiner_enabled", 0.0)
        _flag("legacy_counterfactual_donor_enabled", 0.0)
        _flag("qbound_input_is_generic", 1.0)
        _flag("qbound_input_uses_cra_embedding", 0.0)
        _flag("generic_input_context_reuse", 1.0)
        _flag("teacher_critic_stop_gradient", 1.0)
        _flag("teacher_critic_projection_dropout_enabled", 0.0)
        _flag("teacher_uses_answer_information", 1.0)
        _flag("teacher_affects_grounding_forward", 0.0)
        _flag("grounding_requires_answer_at_inference", 0.0)

    def _collect_qbound_stats(self, gqa_output, extra_logs=None):
        stats = {}
        if extra_logs:
            stats.update(extra_logs)

        qbound_aux = self._get_qbound_aux(gqa_output)
        if qbound_aux is None:
            return stats

        # Architecture / input contract.
        scalar_keys = (
            "generic_input_adapter_enabled",
            "generic_input_position_enabled",
            "generic_input_context_reuse",
            "generic_input_raw_rms",
            "generic_input_adapted_rms",
            "generic_input_token_cosine",
            "generic_input_delta_ratio",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "dynamic_event_count_enabled",
            "event_capacity",
            "hypothesis_competition_enabled",
            "answer_features_used_in_core",
            "legacy_composer_enabled",
            "legacy_refiner_enabled",
            "legacy_counterfactual_donor_enabled",
            "teacher_critic_stop_gradient",
            "teacher_critic_projection_dropout_enabled",
            "teacher_uses_answer_information",
            "teacher_affects_grounding_forward",
            "grounding_requires_answer_at_inference",
            "qbound_smoothing_applied",

            # Dynamic decomposition.
            "boundary_prob_mean",
            "boundary_count",
            "event_count",
            "effective_event_count",
            "soft_effective_event_count",
            "decomp_loss",
            "decomp_expected_energy",
            "decomp_selected_coherence",
            "decomp_selected_complexity",
            "decomp_selected_complexity_cost",
            "decomp_selected_boundary_reward",
            "decomp_selected_boundary_term",
            "decomp_selected_total_energy",
            "decomp_expected_event_count",
            "partition_entropy",
            "accepted_split_gain",
            "rejected_split_gain",
            "event_mass_max",
            "event_mass_min",
            "event_mass_mean",

            # Event evidence / RELATE.
            "event_evidence_gate_mean",
            "event_evidence_active_count",
            "event_evidence_entropy",
            "event_evidence_effective_count",
            "top1_event_evidence_share",
            "top2_event_evidence_share",
            "multi_event_evidence",
            "ordered_relation_gate_mean",
            "ordered_update_gate_mean",
            "ordered_relation_past_mass",
            "ordered_relation_future_mass",
            "reasoned_event_token_delta_norm",

            # Hypothesis competition.
            "hypothesis_candidate_count",
            "hypothesis_entropy",
            "hypothesis_effective_count",
            "hypothesis_top1_posterior",
            "hypothesis_expected_span_length",
            "selected_hypothesis_span_length",
            "selected_hypothesis_span_fraction",
            "selected_hypothesis_posterior",
            "hypothesis_touched_event_count",
            "hypothesis_touched_event_ratio",
        )
        for key in scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                if key.startswith("qbound_"):
                    stats[key] = scalar
                else:
                    stats[f"qbound_{key}"] = scalar

        raw_rms = self._scalar_mean(qbound_aux.get("generic_input_raw_rms"))
        adapted_rms = self._scalar_mean(
            qbound_aux.get("generic_input_adapted_rms")
        )
        if raw_rms is not None and adapted_rms is not None:
            stats["qbound_generic_input_adapted_to_raw_rms_ratio"] = (
                adapted_rms / max(raw_rms, 1e-8)
            )

        # Event-count histogram uses event_capacity, not a hard-coded 4.
        event_count = qbound_aux.get("event_count")
        capacity = int(
            round(self._scalar_mean(qbound_aux.get("event_capacity")) or 0.0)
        )
        if torch.is_tensor(event_count) and capacity > 0:
            values = event_count.detach().round().long()
            for n_events in range(1, capacity + 1):
                stats[f"qbound_event_count_{n_events}_ratio"] = (
                    values == n_events
                ).float().mean().item()

        # Hypothesis-posterior health.
        posterior = qbound_aux.get("hypothesis_posterior")
        valid = qbound_aux.get("hypothesis_valid_mask")
        if torch.is_tensor(posterior) and torch.is_tensor(valid):
            posterior = posterior.detach().float()
            valid = valid.detach().bool()
            valid_count = valid.sum(dim=-1).clamp_min(1).float()
            entropy = -torch.sum(
                posterior * torch.log(posterior.clamp_min(1e-8)),
                dim=-1,
            )
            max_entropy = torch.log(valid_count.clamp_min(1.0))
            entropy_ratio = torch.where(
                valid_count > 1,
                entropy / max_entropy.clamp_min(1e-6),
                torch.zeros_like(entropy),
            )
            stats["qbound_hypothesis_valid_candidate_count"] = (
                valid_count.mean().item()
            )
            stats["qbound_hypothesis_entropy_ratio"] = (
                entropy_ratio.mean().item()
            )
            sorted_prob = torch.sort(
                posterior,
                dim=-1,
                descending=True,
            ).values
            stats["qbound_hypothesis_top1_share"] = (
                sorted_prob[:, 0].mean().item()
            )
            if sorted_prob.size(-1) >= 2:
                stats["qbound_hypothesis_top2_share"] = (
                    sorted_prob[:, :2].sum(dim=-1).mean().item()
                )

        selected_length = qbound_aux.get("selected_hypothesis_span_length")
        if torch.is_tensor(selected_length):
            values = selected_length.detach().float()
            stats["qbound_selected_hypothesis_span_length_min"] = (
                values.min().item()
            )
            stats["qbound_selected_hypothesis_span_length_max"] = (
                values.max().item()
            )
            stats["qbound_selected_hypothesis_point_span_ratio"] = (
                (values <= 1.0).float().mean().item()
            )

        # Dynamic-topology health.
        event_mass = qbound_aux.get("event_mass")
        event_valid = qbound_aux.get("event_valid_mask")
        if torch.is_tensor(event_mass):
            mass = event_mass.detach().float()
            valid_event = (
                event_valid.detach().bool()
                if torch.is_tensor(event_valid)
                else mass > 0
            )
            if valid_event.any():
                valid_mass = mass[valid_event]
                stats["qbound_event_mass_valid_mean"] = valid_mass.mean().item()
                stats["qbound_event_mass_valid_min"] = valid_mass.min().item()
                stats["qbound_event_mass_valid_max"] = valid_mass.max().item()

        candidate_valid = qbound_aux.get("boundary_candidate_valid_mask")
        if torch.is_tensor(candidate_valid):
            stats["qbound_boundary_candidate_count"] = (
                candidate_valid.detach().float().sum(dim=-1).mean().item()
            )

        boundary_prob = qbound_aux.get("boundary_prob")
        boundary_mask = qbound_aux.get("boundary_mask")
        if torch.is_tensor(boundary_prob) and torch.is_tensor(boundary_mask):
            values = boundary_prob[boundary_mask.bool()]
            if values.numel() > 0:
                values = values.detach().float()
                stats["qbound_boundary_prob_min"] = values.min().item()
                stats["qbound_boundary_prob_max"] = values.max().item()
                stats["qbound_boundary_prob_std"] = values.std(
                    unbiased=False
                ).item()

        # Question-conditioned event reasoning health.
        relation = qbound_aux.get("ordered_relation_gates")
        relation_mask = qbound_aux.get("ordered_relation_valid_mask")
        if torch.is_tensor(relation) and torch.is_tensor(relation_mask):
            values = relation[relation_mask.bool()]
            if values.numel() > 0:
                values = values.detach().float()
                stats["qbound_ordered_relation_gate_valid_mean"] = (
                    values.mean().item()
                )
                stats["qbound_ordered_relation_gate_valid_std"] = (
                    values.std(unbiased=False).item()
                )
                stats["qbound_ordered_relation_active_ratio"] = (
                    (values >= 0.5).float().mean().item()
                )

        # Exact hypothesis key vs CRA reference.
        qbound_key = qbound_aux.get("qbound_key")
        cra_key = qbound_aux.get("cra_key")
        if torch.is_tensor(qbound_key) and torch.is_tensor(cra_key):
            q = self._normalize_distribution(qbound_key.detach())
            c = self._normalize_distribution(cra_key.detach())
            stats["qbound_cra_qbound_js"] = self._js_divergence(
                c, q
            ).mean().item()
            stats["qbound_cra_qbound_key_l1"] = (
                c - q
            ).abs().mean().item()
            stats["qbound_key_entropy"] = self._distribution_entropy(
                q
            ).mean().item()
            stats["qbound_cra_key_entropy"] = self._distribution_entropy(
                c
            ).mean().item()
            stats["qbound_key_max"] = q.max(dim=-1).values.mean().item()
            stats["qbound_cra_key_max"] = c.max(dim=-1).values.mean().item()

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
                "qbound_hypothesis_scorer_grad_norm": 0.0,
                "qbound_decompose_grad_norm": 0.0,
                "qbound_relate_grad_norm": 0.0,
            }

        totals = {
            "input": 0.0,
            "hypothesis": 0.0,
            "decompose": 0.0,
            "relate": 0.0,
        }
        for name, parameter in self.model.named_parameters():
            if parameter.grad is None or "qbound" not in name.lower():
                continue
            lower = name.lower()
            grad_sq = parameter.grad.detach().float().pow(2).sum().item()

            if "generic_input_proj" in lower or "generic_input_norm" in lower:
                totals["input"] += grad_sq
            if (
                "hypothesis_feature_encoder" in lower
                or "hypothesis_score" in lower
            ):
                totals["hypothesis"] += grad_sq
            if (
                "boundary_score" in lower
                or "decomp_" in lower
                or "topology" in lower
            ):
                totals["decompose"] += grad_sq
            if (
                "relation" in lower
                or "ordered_" in lower
                or "event_update" in lower
            ):
                totals["relate"] += grad_sq

        return {
            "qbound_input_adapter_grad_norm": float(totals["input"] ** 0.5),
            "qbound_hypothesis_scorer_grad_norm": float(
                totals["hypothesis"] ** 0.5
            ),
            "qbound_decompose_grad_norm": float(
                totals["decompose"] ** 0.5
            ),
            "qbound_relate_grad_norm": float(totals["relate"] ** 0.5),
        }

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_grad_norm",
            "qbound_input_adapter_grad_norm",
            "qbound_hypothesis_scorer_grad_norm",
            "qbound_decompose_grad_norm",
            "qbound_relate_grad_norm",
            "qbound_weak_align_weight",
            "qbound_decomp_weight",
            "qbound_hypothesis_rank_weight",
            "qbound_hypothesis_teacher_enabled",
            "qbound_hypothesis_teacher_correct_only",
            "qbound_weak_align_retrieval_acc",
            "qbound_hypothesis_rank_loss",
            "qbound_teacher_active_ratio",
            "qbound_teacher_full_video_acc",
            "qbound_teacher_top1_probability",
            "qbound_teacher_entropy",
            "qbound_teacher_entropy_ratio",
            "qbound_teacher_effective_count",
            "qbound_student_top1_probability",
            "qbound_student_entropy",
            "qbound_student_entropy_ratio",
            "qbound_teacher_student_top1_agreement",
            "qbound_selected_is_teacher_top1_ratio",
            "qbound_selected_teacher_probability",
            "qbound_selected_teacher_utility",
            "qbound_teacher_best_utility",
            "qbound_teacher_best_minus_selected_utility",
            "qbound_selected_rank_percentile",
            "qbound_teacher_best_answer_acc",
            "qbound_teacher_selected_answer_acc",
            "qbound_teacher_selected_complement_acc",
            "qbound_teacher_full_margin",
            "qbound_teacher_selected_margin",
            "qbound_teacher_best_margin",
            "qbound_teacher_selected_complement_margin",
            "qbound_teacher_sufficiency_component",
            "qbound_teacher_trim_component",
            "qbound_teacher_expand_component",
            "qbound_teacher_exclusivity_component",
            "qbound_generic_input_token_cosine",
            "qbound_generic_input_delta_ratio",
            "qbound_event_count",
            "qbound_decomp_loss",
            "qbound_partition_entropy",
            "qbound_soft_effective_event_count",
            "qbound_hypothesis_candidate_count",
            "qbound_hypothesis_entropy",
            "qbound_hypothesis_entropy_ratio",
            "qbound_hypothesis_effective_count",
            "qbound_hypothesis_top1_posterior",
            "qbound_hypothesis_expected_span_length",
            "qbound_selected_hypothesis_span_length",
            "qbound_selected_hypothesis_span_fraction",
            "qbound_selected_hypothesis_point_span_ratio",
            "qbound_cra_qbound_js",
            "qbound_cra_qbound_key_l1",
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

        # ----------------------------------------------------------
        # Original CRA VQ objective: unchanged, Q-Bound disabled.
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # MAIN VQA: exact hard winning temporal hypothesis H*.
        # ----------------------------------------------------------
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

        # Preserve the repository's CRA task objective.
        loss = self.criterion(prediction_dict, target_dict)

        settings = self._qbound_v112_loss_settings()
        weak_align_retrieval_acc = None
        teacher_result = None
        qbound_aux = self._get_qbound_aux(gqa_output)

        if self._qbound_enabled():
            self._validate_v112_contract(qbound_aux)

            # Weak question/event semantic alignment.
            if settings["align_weight"] > 0.0:
                qbound_align_loss, weak_align_retrieval_acc = (
                    self._qbound_inbatch_alignment(
                        evidence_visual_token=qbound_aux.get(
                            "evidence_visual_token"
                        ),
                        question_state=qbound_aux.get("qa_state"),
                        temperature=settings["align_temperature"],
                    )
                )
                loss["qbound_align_loss"] = qbound_align_loss
                loss["total_loss"] = (
                    loss["total_loss"]
                    + settings["align_weight"] * qbound_align_loss
                )

            # Dynamic event decomposition objective.
            if settings["decomp_weight"] > 0.0:
                qbound_decomp_loss = self._qbound_decomposition_loss(
                    qbound_aux
                )
                loss["qbound_decomp_loss"] = qbound_decomp_loss
                loss["total_loss"] = (
                    loss["total_loss"]
                    + settings["decomp_weight"] * qbound_decomp_loss
                )

            # Privileged answer-aware temporal ranking.
            if settings["teacher_enabled"]:
                teacher_result = self._qbound_v112_hypothesis_distillation(
                    model_outputs=gqa_output[2],
                    labels=answer_id_device,
                    settings=settings,
                )
                rank_loss = teacher_result["ranking_loss"]
                weighted_rank_loss = (
                    settings["teacher_loss_weight"] * rank_loss
                )
                loss["qbound_hypothesis_rank_loss"] = rank_loss
                loss["qbound_hypothesis_rank_weighted_loss"] = (
                    weighted_rank_loss
                )
                loss["total_loss"] = loss["total_loss"] + weighted_rank_loss

        qbound_logs = {}
        if self._qbound_enabled():
            qbound_logs.update(
                {
                    "qbound_weak_align_weight": settings["align_weight"],
                    "qbound_decomp_weight": settings["decomp_weight"],
                    "qbound_hypothesis_rank_weight": settings[
                        "teacher_loss_weight"
                    ],
                    "qbound_hypothesis_teacher_enabled": float(
                        settings["teacher_enabled"]
                    ),
                    "qbound_hypothesis_teacher_correct_only": float(
                        settings["teacher_correct_only"]
                    ),
                }
            )
            if weak_align_retrieval_acc is not None:
                qbound_logs["qbound_weak_align_retrieval_acc"] = float(
                    weak_align_retrieval_acc
                )
            if teacher_result is not None:
                qbound_logs["qbound_hypothesis_rank_loss"] = float(
                    teacher_result["ranking_loss"].detach().item()
                )
                teacher_log_mapping = {
                    "teacher_active_ratio": "qbound_teacher_active_ratio",
                    "teacher_full_video_acc": "qbound_teacher_full_video_acc",
                    "teacher_top1_probability": "qbound_teacher_top1_probability",
                    "teacher_entropy": "qbound_teacher_entropy",
                    "teacher_entropy_ratio": "qbound_teacher_entropy_ratio",
                    "teacher_effective_count": "qbound_teacher_effective_count",
                    "student_top1_probability": "qbound_student_top1_probability",
                    "student_entropy": "qbound_student_entropy",
                    "student_entropy_ratio": "qbound_student_entropy_ratio",
                    "teacher_student_top1_agreement": "qbound_teacher_student_top1_agreement",
                    "selected_is_teacher_top1_ratio": "qbound_selected_is_teacher_top1_ratio",
                    "selected_teacher_probability": "qbound_selected_teacher_probability",
                    "selected_teacher_utility": "qbound_selected_teacher_utility",
                    "teacher_best_utility": "qbound_teacher_best_utility",
                    "teacher_best_minus_selected_utility": "qbound_teacher_best_minus_selected_utility",
                    "selected_rank_percentile": "qbound_selected_rank_percentile",
                    "teacher_best_answer_acc": "qbound_teacher_best_answer_acc",
                    "teacher_selected_answer_acc": "qbound_teacher_selected_answer_acc",
                    "teacher_selected_complement_acc": "qbound_teacher_selected_complement_acc",
                    "teacher_full_margin": "qbound_teacher_full_margin",
                    "teacher_selected_margin": "qbound_teacher_selected_margin",
                    "teacher_best_margin": "qbound_teacher_best_margin",
                    "teacher_selected_complement_margin": "qbound_teacher_selected_complement_margin",
                    "teacher_sufficiency_component": "qbound_teacher_sufficiency_component",
                    "teacher_trim_component": "qbound_teacher_trim_component",
                    "teacher_expand_component": "qbound_teacher_expand_component",
                    "teacher_exclusivity_component": "qbound_teacher_exclusivity_component",
                }
                for source_name, log_name in teacher_log_mapping.items():
                    qbound_logs[log_name] = float(
                        teacher_result[source_name]
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
            self.monitor.reset_kv("qbound_align_loss")
            self.monitor.reset_kv("qbound_decomp_loss")
            self.monitor.reset_kv("qbound_hypothesis_rank_loss")
            self.monitor.reset_kv("qbound_hypothesis_rank_weighted_loss")
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
                    f"\n{split.title()} Set\t V1.12 exact-hypothesis acc: "
                    f"{score[acc_key]:.2%}"
                )

            prehoc_key = f"{split}_pre-hoc"
            if prehoc_key in score:
                self.monitor.log_info(
                    f"\n========== {split} V1.12 exact H* =========="
                )
                _print_score(score[prehoc_key])

            fixed_key = f"{split}_fixed-readout"
            if fixed_key in score:
                self.monitor.log_info(
                    f"\n========== {split} same-key CRA fixed-window diagnostic =========="
                )
                _print_score(score[fixed_key])

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
                "\n================ Q-Bound V1.12 diagnostics ================"
            )
            for key in sorted(qbound_keys):
                self.monitor.log_info(f"{key}: {score[key]:.6f}")

        temporal_prefixes = (
            "val_QBound-",
            "test_QBound-",
            "val_QBoundSoftHyp-",
            "test_QBoundSoftHyp-",
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
        ground_fixed = {}
        keyframe = {}
        ori_keyframe = {}
        qbound_keyframe = {}
        qbound_soft_hyp_keyframe = {}
        cra_keyframe = {}

        qbound_stat_sums = collections.defaultdict(float)
        qbound_stat_count = 0
        saw_time_param = False

        settings = self._qbound_v112_loss_settings()

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
                    self._validate_v112_contract(qbound_aux)
                    if settings["teacher_enabled"]:
                        teacher_result = (
                            self._qbound_v112_hypothesis_distillation(
                                model_outputs=model_outputs,
                                labels=labels_device,
                                settings=settings,
                            )
                        )
                        extra_logs["qbound_hypothesis_rank_loss"] = float(
                            teacher_result["ranking_loss"].item()
                        )
                        for source_name, log_name in (
                            ("teacher_active_ratio", "qbound_teacher_active_ratio"),
                            ("teacher_full_video_acc", "qbound_teacher_full_video_acc"),
                            ("teacher_top1_probability", "qbound_teacher_top1_probability"),
                            ("teacher_entropy_ratio", "qbound_teacher_entropy_ratio"),
                            ("teacher_effective_count", "qbound_teacher_effective_count"),
                            ("student_top1_probability", "qbound_student_top1_probability"),
                            ("student_entropy_ratio", "qbound_student_entropy_ratio"),
                            ("teacher_student_top1_agreement", "qbound_teacher_student_top1_agreement"),
                            ("selected_is_teacher_top1_ratio", "qbound_selected_is_teacher_top1_ratio"),
                            ("selected_teacher_probability", "qbound_selected_teacher_probability"),
                            ("teacher_best_minus_selected_utility", "qbound_teacher_best_minus_selected_utility"),
                            ("selected_rank_percentile", "qbound_selected_rank_percentile"),
                            ("teacher_best_answer_acc", "qbound_teacher_best_answer_acc"),
                            ("teacher_selected_answer_acc", "qbound_teacher_selected_answer_acc"),
                            ("teacher_selected_complement_acc", "qbound_teacher_selected_complement_acc"),
                        ):
                            extra_logs[log_name] = float(
                                teacher_result[source_name]
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
                batch_fixed = None
                batch_keyframe = None
                batch_ori_keyframe = None
                batch_qbound_key = None
                batch_soft_hyp = None
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
                        soft_hyp = qbound_aux.get("hypothesis_soft_support")
                        if torch.is_tensor(q_key):
                            batch_qbound_key = q_key.detach().cpu().numpy()
                        if torch.is_tensor(c_key):
                            batch_cra_key = c_key.detach().cpu().numpy()
                        if torch.is_tensor(soft_hyp):
                            soft_hyp = self._normalize_distribution(
                                soft_hyp.detach()
                            )
                            batch_soft_hyp = soft_hyp.cpu().numpy()

                        fixed_start = time_param.get(
                            "qbound_legacy_fixed_window_start"
                        )
                        fixed_end = time_param.get(
                            "qbound_legacy_fixed_window_end"
                        )
                        if torch.is_tensor(fixed_start) and torch.is_tensor(
                            fixed_end
                        ):
                            batch_fixed = np.column_stack(
                                (
                                    fixed_start.detach().cpu().numpy(),
                                    fixed_end.detach().cpu().numpy(),
                                )
                            )

                for batch_index, qid in enumerate(question_id):
                    results[qid] = {
                        "prediction": int(predicted.numpy()[batch_index]),
                        "answer": int(labels_cpu.numpy()[batch_index]),
                    }
                    ground_res[qid] = video_attention[batch_index].tolist()

                    if batch_ground_pre is not None:
                        ground_pre[qid] = batch_ground_pre[batch_index].tolist()
                    if batch_fixed is not None:
                        ground_fixed[qid] = batch_fixed[batch_index].tolist()
                    if batch_keyframe is not None:
                        keyframe[qid] = batch_keyframe[batch_index].tolist()
                    if batch_ori_keyframe is not None:
                        ori_keyframe[qid] = batch_ori_keyframe[batch_index].tolist()
                    if batch_qbound_key is not None:
                        qbound_keyframe[qid] = batch_qbound_key[batch_index].tolist()
                    if batch_soft_hyp is not None:
                        qbound_soft_hyp_keyframe[qid] = (
                            batch_soft_hyp[batch_index].tolist()
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
                    "Q-Bound V1.12 requires Metric.temporal_key_metrics()."
                )
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBound",
                )
            )
        if qbound_soft_hyp_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_soft_hyp_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundSoftHyp",
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
            "qbound_soft_hyp_key": qbound_soft_hyp_keyframe,
            "cra_key": cra_keyframe,
        }

        if saw_time_param and ground_pre:
            score["pre-hoc"] = self.metric(
                ground_pre,
                ground_truth,
                results,
                gs=True,
            )
            if ground_fixed:
                score["fixed-readout"] = self.metric(
                    ground_fixed,
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
                    "fixed_tg": ground_fixed,
                    "merge": merge_ground,
                }
            )

            if "fixed-readout" in score:
                main_pre = score["pre-hoc"]
                fixed_score = score["fixed-readout"]
                for metric_name in (
                    "mIoP",
                    "mIoU",
                    "TIoP@0.3",
                    "TIoP@0.5",
                    "TIoU@0.3",
                    "TIoU@0.5",
                    "Acc&GQA",
                ):
                    if metric_name in main_pre and metric_name in fixed_score:
                        safe = metric_name.replace("@", "_at_").replace(
                            "&", "_and_"
                        )
                        score[
                            f"qbound_exact_minus_fixed_{safe}"
                        ] = float(
                            main_pre[metric_name] - fixed_score[metric_name]
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
