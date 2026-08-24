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
    CRA training pipeline with Q-Bound Version 1.2.

    V1.2 preserves the repository task objectives and adds one Q-Bound-specific weak event alignment objective:

        L =
            L_CRA_original
            + lambda_align * L_QBound_weak_align
            + lambda_sparse * L_boundary_sparse   (optional; default 0)

    The weak alignment is event/topology specific: the pure Q-Bound
    evidence_visual_token is aligned with the question-only Q-Bound state
    using in-batch InfoNCE. No answer feature is fed into topology induction.

    There is still deliberately NO:
        * necessity BCE
        * split invariance
        * CRA/Q-Bound consistency loss
        * separate answer head

    Q-Bound learns through the main VQA path. During training a fixed,
    non-learned curriculum hands temporal responsibility from the detached
    CRA temporal teacher to Q-Bound:

        final_key =
            (1-lambda) * stopgrad(cra_key)
            + lambda * qbound_key

    ``lambda`` is scheduled batch-wise from ``handover_start`` to 1.0 over
    ``handover_epochs``. Validation, test, and inference ALWAYS use
    lambda=1.0 so reported/checkpoint metrics are pure Q-Bound temporal
    results and never benefit from the CRA teacher mixture.

    Transition-level MERGE is retained only as a diagnostic counterfactual.

    Dynamic boundary count and event-count histograms are monitored explicitly
    because V1.2 must not regress to either 1-event collapse or max-event saturation.
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

    def _qbound_handover_settings(self):
        """
        Read the deterministic V1.2 CRA -> Q-Bound curriculum.

        Preferred config:
            model:
              qbound:
                handover_start: 0.2
                handover_epochs: 5

        Model-level fallbacks are kept only so older YAML files fail softly
        until the final V1.2 config is installed.
        """
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})

        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        handover_start = float(
            qbound_cfg.get(
                "handover_start",
                model_cfg.get(
                    "qbound_handover_start",
                    0.2,
                ),
            )
        )

        handover_epochs = int(
            qbound_cfg.get(
                "handover_epochs",
                model_cfg.get(
                    "qbound_handover_epochs",
                    5,
                ),
            )
        )

        if not 0.0 <= handover_start <= 1.0:
            raise ValueError(
                "Q-Bound handover_start must be in [0,1]."
            )

        if handover_epochs < 0:
            raise ValueError(
                "Q-Bound handover_epochs must be >= 0."
            )

        return handover_start, handover_epochs

    def _qbound_v12_loss_settings(self):
        """Read the small set of V1.2-specific auxiliary-loss settings."""
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

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
        boundary_sparse_weight = float(
            qbound_cfg.get(
                "boundary_sparse_weight",
                model_cfg.get("qbound_boundary_sparse_weight", 0.0),
            )
        )

        if align_weight < 0.0:
            raise ValueError("qbound weak_align_weight must be >= 0.")
        if align_temperature <= 0.0:
            raise ValueError("qbound weak_align_temperature must be > 0.")
        if boundary_sparse_weight < 0.0:
            raise ValueError("qbound boundary_sparse_weight must be >= 0.")

        return align_weight, align_temperature, boundary_sparse_weight

    @staticmethod
    def _qbound_inbatch_alignment(
        evidence_visual_token,
        question_state,
        temperature,
    ):
        """
        Q-Bound-specific weak event alignment.

        evidence_visual_token: [B,H]
            Pure visual event evidence pooled by the learned dynamic topology.
        question_state: [B,H]
            Question-only Q-Bound state. It is detached and acts as a semantic
            target; answer-choice semantics are never fed into topology.

        The objective is symmetric in retrieval space, but gradients only need
        to move the visual/topology side toward its own semantic target.
        """
        if evidence_visual_token is None or question_state is None:
            raise ValueError(
                "Q-Bound V1.2 weak alignment requires evidence_visual_token "
                "and qa_state."
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
            zero = evidence_visual_token.sum() * 0.0
            return zero, 0.0

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
    def _qbound_boundary_sparse_loss(qbound_aux):
        """
        Optional soft boundary-density penalty. Disabled by default in V1.2.

        This must stay optional because a sparsity force can collapse dynamic
        topology back to one event before event reasoning becomes necessary.
        """
        boundary_prob = qbound_aux.get("boundary_prob")
        boundary_mask = qbound_aux.get("boundary_mask")
        if not torch.is_tensor(boundary_prob) or not torch.is_tensor(boundary_mask):
            raise KeyError(
                "Q-Bound boundary sparsity requires boundary_prob/boundary_mask."
            )
        weights = boundary_mask.to(boundary_prob.dtype)
        return (
            (boundary_prob * weights).sum()
            / weights.sum().clamp_min(1.0)
        )

    def _train_qbound_mix(
        self,
        epoch,
        batch_index,
        batches_per_epoch,
    ):
        """
        Batch-wise deterministic handover.

        BasePipeline epochs are 1-indexed, so:
            epoch=1, batch=0 -> handover_start
            final batch of handover_epochs -> 1.0
            later epochs -> 1.0

        The formula also resumes correctly from checkpoints because it depends
        on the absolute epoch rather than a transient in-memory step counter.
        """
        if not self._qbound_enabled():
            return 0.0

        if batches_per_epoch <= 0:
            raise ValueError(
                "batches_per_epoch must be positive."
            )

        start, handover_epochs = (
            self._qbound_handover_settings()
        )

        if (
            handover_epochs == 0
            or start >= 1.0
        ):
            return 1.0

        epoch_index = max(
            int(epoch) - 1,
            0,
        )

        global_step = (
            epoch_index * batches_per_epoch
            + int(batch_index)
        )

        total_steps = (
            handover_epochs
            * batches_per_epoch
        )

        if total_steps <= 1:
            return 1.0

        progress = min(
            max(
                float(global_step)
                / float(total_steps - 1),
                0.0,
            ),
            1.0,
        )

        return (
            start
            + (1.0 - start) * progress
        )

    def _get_qbound_aux(self, gqa_output):
        """Safely extract Q-Bound V1.2 outputs from a CRA VQA forward."""
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
        VQA path.  V1.2 counterfactuals therefore do not use a separate
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
        Diagnostic only in V1.2.

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
    # Q-Bound V1.2 counterfactual diagnostics
    # ------------------------------------------------------------------
    @staticmethod
    def _distribution_entropy(
        probability,
        eps=1e-8,
    ):
        probability = probability.clamp_min(0.0)
        probability = (
            probability
            / probability.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(eps)
        )

        return -torch.sum(
            probability
            * torch.log(
                probability.clamp_min(eps)
            ),
            dim=-1,
        )

    def _compute_qbound_v12_diagnostics(
        self,
        gqa_output,
        answer_proj,
        answer_id,
        main_logits,
    ):
        """
        Measure Q-Bound behavior WITHOUT adding any optimization loss.

        The strongest learned event transition is removed in QBoundCore.
        refine.py recomputes the merge distribution through:
            Q-Bound probability smoothing
            -> same CRA/Q-Bound handover
            -> same video projection
            -> same main answer scorer

        We log answer harm and temporal changes only.  In particular,
        ``necessity`` remains a descriptive statistic:

            necessity = 1 - exp(-relu(CE_merge - CE_original))

        It remains diagnostic-only in V1.2.
        """
        qbound_aux = self._get_qbound_aux(
            gqa_output
        )

        if qbound_aux is None:
            return {}

        required_keys = {
            "merge_video_proj",
            "selected_transition_strength",
            "selected_transition_mass",
            "removed_boundary_mass",
            "cra_video_proj",
            "cra_key",
            "qbound_key",
            "final_key",
            "merge_qbound_key",
            "merge_final_key",
            "merge_key_l1",
            "pure_qbound_merge_key_l1",
            "cra_qbound_key_l1",
            "qbound_mix",
        }

        missing = sorted(
            required_keys.difference(
                qbound_aux
            )
        )

        if missing:
            raise KeyError(
                "Missing Q-Bound V1.2 outputs: "
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

        # Diagnostics must not create a second training graph.
        with torch.no_grad():
            merge_logits = (
                self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=qbound_aux[
                        "merge_video_proj"
                    ],
                    detach_answer=True,
                )
            )

            original_logits = (
                main_logits.detach()
            )

            original_ce = F.cross_entropy(
                original_logits,
                labels,
                reduction="none",
            )

            if merge_logits.ndim == 2:
                merge_ce = F.cross_entropy(
                    merge_logits,
                    labels,
                    reduction="none",
                )
                merge_pred_logits = (
                    merge_logits
                )

            elif merge_logits.ndim == 3:
                num_merge = (
                    merge_logits.size(1)
                )

                merge_labels = (
                    labels.unsqueeze(1)
                    .expand(-1, num_merge)
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

                merge_pred_logits = (
                    merge_logits.mean(dim=1)
                )

            else:
                raise ValueError(
                    "merge logits must be [B,C] or [B,M,C], got "
                    f"{tuple(merge_logits.shape)}."
                )

            answer_harm = (
                merge_ce - original_ce
            )

            necessity = (
                1.0
                - torch.exp(
                    -F.relu(answer_harm)
                )
            )

            original_pred = (
                original_logits.argmax(
                    dim=-1
                )
            )

            merge_pred = (
                merge_pred_logits.argmax(
                    dim=-1
                )
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

            base_pred = (
                base_logits.argmax(
                    dim=-1
                )
            )

            cra_key = (
                self._normalize_distribution(
                    qbound_aux["cra_key"]
                )
            )

            qbound_key = (
                self._normalize_distribution(
                    qbound_aux["qbound_key"]
                )
            )

            final_key = (
                self._normalize_distribution(
                    qbound_aux["final_key"]
                )
            )

            merge_qbound_key = (
                self._normalize_distribution(
                    qbound_aux[
                        "merge_qbound_key"
                    ].squeeze(1)
                )
            )

            merge_final_key = (
                self._normalize_distribution(
                    qbound_aux[
                        "merge_final_key"
                    ].squeeze(1)
                )
            )

            cra_qbound_js = (
                self._js_divergence(
                    cra_key,
                    qbound_key,
                ).mean()
            )

            cra_final_js = (
                self._js_divergence(
                    cra_key,
                    final_key,
                ).mean()
            )

            qbound_final_js = (
                self._js_divergence(
                    qbound_key,
                    final_key,
                ).mean()
            )

            merge_final_js = (
                self._js_divergence(
                    final_key,
                    merge_final_key,
                ).mean()
            )

            pure_merge_js = (
                self._js_divergence(
                    qbound_key,
                    merge_qbound_key,
                ).mean()
            )

            cra_entropy = (
                self._distribution_entropy(
                    cra_key
                )
            )

            qbound_entropy = (
                self._distribution_entropy(
                    qbound_key
                )
            )

            final_entropy = (
                self._distribution_entropy(
                    final_key
                )
            )

            boundary_mask = (
                qbound_aux.get(
                    "boundary_mask"
                )
            )

            if (
                torch.is_tensor(
                    boundary_mask
                )
                and boundary_mask.ndim == 2
            ):
                valid_length = (
                    boundary_mask.bool()
                    .sum(dim=-1)
                    .to(cra_key.dtype)
                    + 1.0
                )
            else:
                valid_length = (
                    torch.full(
                        (cra_key.size(0),),
                        float(cra_key.size(-1)),
                        dtype=cra_key.dtype,
                        device=cra_key.device,
                    )
                )

            valid_length = (
                valid_length.clamp_min(2.0)
            )

            uniform_entropy = (
                torch.log(valid_length)
            )

            cra_peak = (
                cra_key.max(
                    dim=-1
                ).values
            )
            qbound_peak = (
                qbound_key.max(
                    dim=-1
                ).values
            )
            final_peak = (
                final_key.max(
                    dim=-1
                ).values
            )

            cra_peak_idx = (
                cra_key.argmax(
                    dim=-1
                ).to(cra_key.dtype)
            )
            qbound_peak_idx = (
                qbound_key.argmax(
                    dim=-1
                ).to(cra_key.dtype)
            )
            final_peak_idx = (
                final_key.argmax(
                    dim=-1
                ).to(cra_key.dtype)
            )

            peak_denominator = (
                valid_length - 1.0
            ).clamp_min(1.0)

            cra_qbound_peak_shift = (
                (
                    cra_peak_idx
                    - qbound_peak_idx
                ).abs()
                / peak_denominator
            )

            cra_final_peak_shift = (
                (
                    cra_peak_idx
                    - final_peak_idx
                ).abs()
                / peak_denominator
            )

            qbound_mix = (
                qbound_aux[
                    "qbound_mix"
                ]
                .detach()
                .float()
                .mean()
            )

        return {
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
                original_ce.mean().item()
            ),
            "qbound_merge_ce": (
                merge_ce.mean().item()
            ),
            "qbound_answer_harm": (
                answer_harm.mean().item()
            ),
            "qbound_abs_answer_harm": (
                answer_harm.abs()
                .mean().item()
            ),
            "qbound_positive_harm_ratio": (
                (answer_harm > 0)
                .float().mean().item()
            ),
            "qbound_necessity_diagnostic": (
                necessity.mean().item()
            ),
            "qbound_mix": (
                qbound_mix.item()
            ),

            "qbound_cra_qbound_js": (
                cra_qbound_js.item()
            ),
            "qbound_cra_final_js": (
                cra_final_js.item()
            ),
            "qbound_qbound_final_js": (
                qbound_final_js.item()
            ),
            "qbound_original_merge_js": (
                merge_final_js.item()
            ),
            "qbound_pure_merge_js": (
                pure_merge_js.item()
            ),

            "qbound_cra_key_entropy": (
                cra_entropy.mean().item()
            ),
            "qbound_key_entropy": (
                qbound_entropy.mean().item()
            ),
            "qbound_final_key_entropy": (
                final_entropy.mean().item()
            ),
            "qbound_cra_key_entropy_ratio": (
                (
                    cra_entropy
                    / uniform_entropy.clamp_min(
                        1e-6
                    )
                ).mean().item()
            ),
            "qbound_key_entropy_ratio": (
                (
                    qbound_entropy
                    / uniform_entropy.clamp_min(
                        1e-6
                    )
                ).mean().item()
            ),
            "qbound_final_key_entropy_ratio": (
                (
                    final_entropy
                    / uniform_entropy.clamp_min(
                        1e-6
                    )
                ).mean().item()
            ),

            "qbound_cra_key_max": (
                cra_peak.mean().item()
            ),
            "qbound_key_max": (
                qbound_peak.mean().item()
            ),
            "qbound_final_key_max": (
                final_peak.mean().item()
            ),
            "qbound_cra_peak_ratio": (
                (
                    cra_peak
                    * valid_length
                ).mean().item()
            ),
            "qbound_peak_ratio": (
                (
                    qbound_peak
                    * valid_length
                ).mean().item()
            ),
            "qbound_final_peak_ratio": (
                (
                    final_peak
                    * valid_length
                ).mean().item()
            ),

            "qbound_cra_qbound_peak_shift": (
                cra_qbound_peak_shift
                .mean().item()
            ),
            "qbound_cra_final_peak_shift": (
                cra_final_peak_shift
                .mean().item()
            ),
        }

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
            "event_count",
            "selected_boundary_prob_mean",
            "assignment_entropy",
            "effective_event_count",
            "soft_effective_event_count",
            "event_relevance_entropy",
            "token_entropy",
            "selected_transition_mass",
            "selected_transition_strength",
            "removed_boundary_mass",
            "merge_key_l1",
            "pure_qbound_merge_key_l1",
            "cra_qbound_key_l1",
            "question_only_conditioning",
            "qbound_mix",
        ]

        for key in scalar_keys:
            scalar = self._scalar_mean(
                qbound_aux.get(key)
            )

            if scalar is not None:
                stats[
                    f"qbound_{key}"
                ] = scalar

        boundary_prob = (
            qbound_aux.get(
                "boundary_prob"
            )
        )
        boundary_mask = (
            qbound_aux.get(
                "boundary_mask"
            )
        )

        if (
            torch.is_tensor(boundary_prob)
            and torch.is_tensor(boundary_mask)
        ):
            valid_boundary = (
                boundary_prob[
                    boundary_mask.bool()
                ]
            )

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

        # V1.2 dynamic-topology diagnostics. The histogram is the primary
        # signal for detecting both max-event saturation and 1-event collapse.
        event_count = qbound_aux.get("event_count")
        if torch.is_tensor(event_count):
            event_count_int = event_count.detach().round().long()
            max_events = int(
                self.cfgs.get("model", {})
                .get("qbound", {})
                .get("max_events", 4)
            )
            for number_of_events in range(1, max_events + 1):
                stats[
                    f"qbound_event_count_{number_of_events}_ratio"
                ] = (
                    event_count_int == number_of_events
                ).float().mean().item()

        candidate_mask = qbound_aux.get("boundary_candidate_mask")
        if torch.is_tensor(candidate_mask):
            stats["qbound_boundary_candidate_count"] = (
                candidate_mask.detach().float().sum(dim=-1).mean().item()
            )

        pre_smooth_key = qbound_aux.get("qbound_pre_smooth_key")
        post_smooth_key = qbound_aux.get("qbound_post_smooth_key")
        if torch.is_tensor(pre_smooth_key) and torch.is_tensor(post_smooth_key):
            stats["qbound_pre_post_smooth_l1"] = (
                (pre_smooth_key.detach() - post_smooth_key.detach())
                .abs().mean().item()
            )
            pre_peak = pre_smooth_key.detach().argmax(dim=-1)
            post_peak = post_smooth_key.detach().argmax(dim=-1)
            stats["qbound_smoothing_peak_change_ratio"] = (
                pre_peak != post_peak
            ).float().mean().item()

        frame_semantic_logits = qbound_aux.get("frame_semantic_logits")
        if torch.is_tensor(frame_semantic_logits):
            values = frame_semantic_logits.detach().float()
            stats["qbound_frame_semantic_logit_abs_mean"] = (
                values.abs().mean().item()
            )
            stats["qbound_frame_semantic_logit_std"] = (
                values.std(unbiased=False).item()
            )

        event_mass = (
            qbound_aux.get(
                "event_mass"
            )
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

        event_activity = (
            qbound_aux.get(
                "event_activity"
            )
        )

        if torch.is_tensor(
            event_activity
        ):
            activity = (
                event_activity.detach()
                .float()
            )

            stats[
                "qbound_event_activity_mean"
            ] = activity.mean().item()

            stats[
                "qbound_event_activity_min"
            ] = (
                activity.min(
                    dim=-1
                ).values.mean().item()
            )

            stats[
                "qbound_event_activity_max"
            ] = (
                activity.max(
                    dim=-1
                ).values.mean().item()
            )

        transition_activity = (
            qbound_aux.get(
                "transition_activity"
            )
        )

        if torch.is_tensor(
            transition_activity
        ):
            activity = (
                transition_activity.detach()
                .float()
            )

            stats[
                "qbound_transition_activity_mean"
            ] = (
                activity.mean().item()
            )

            stats[
                "qbound_transition_activity_max"
            ] = (
                activity.max(
                    dim=-1
                ).values.mean().item()
            )

        transition_mass = (
            qbound_aux.get(
                "transition_mass"
            )
        )
        transition_strength = (
            qbound_aux.get(
                "transition_strength"
            )
        )
        transition_mask = (
            qbound_aux.get(
                "transition_counterfactual_mask"
            )
        )

        if torch.is_tensor(
            transition_mass
        ):
            if torch.is_tensor(
                transition_mask
            ):
                valid_values = (
                    transition_mass[
                        transition_mask.bool()
                    ]
                )
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
                valid_values = (
                    transition_strength[
                        transition_mask.bool()
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

        # Monitor semantic score scale; V1.2 no longer applies tanh bounding.
        for source_key, stat_prefix in [
            (
                "event_token_scores",
                "qbound_event_token_score",
            ),
            (
                "transition_token_scores",
                "qbound_transition_token_score",
            ),
        ]:
            values = qbound_aux.get(
                source_key
            )

            if torch.is_tensor(values):
                values = (
                    values.detach()
                    .float().abs()
                )

                stats[
                    f"{stat_prefix}_abs_mean"
                ] = values.mean().item()

                stats[
                    f"{stat_prefix}_abs_max"
                ] = values.max().item()

        for source_key, stat_name in [
            (
                "raw_event_token_logits",
                "qbound_raw_event_token_logit_abs_max",
            ),
            (
                "raw_transition_token_logits",
                "qbound_raw_transition_token_logit_abs_max",
            ),
        ]:
            values = qbound_aux.get(
                source_key
            )

            if torch.is_tensor(values):
                stats[stat_name] = (
                    values.detach()
                    .float().abs()
                    .max().item()
                )

        removed_mass = (
            qbound_aux.get(
                "removed_boundary_mass"
            )
        )
        boundary_count = (
            qbound_aux.get(
                "boundary_count"
            )
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

    def _qbound_grad_norm(self):
        """
        Diagnostic norm of gradients on Q-Bound parameters only.

        This is measured after backward and before gradient clipping.
        """
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
                parameter.grad.detach()
                .float()
            )

            total_sq += (
                grad.pow(2).sum().item()
            )
            found = True

        if not found:
            return 0.0

        return float(
            total_sq ** 0.5
        )

    @staticmethod
    def _qbound_monitor_keys():
        return [
            "qbound_mix",
            "qbound_grad_norm",

            "qbound_original_acc",
            "qbound_base_only_acc",
            "qbound_merge_acc",
            "qbound_original_ce",
            "qbound_merge_ce",
            "qbound_answer_harm",
            "qbound_abs_answer_harm",
            "qbound_positive_harm_ratio",
            "qbound_necessity_diagnostic",

            "qbound_selected_transition_strength",
            "qbound_selected_transition_mass",
            "qbound_removed_boundary_mass",
            "qbound_removed_boundary_ratio",
            "qbound_merge_key_l1",
            "qbound_pure_qbound_merge_key_l1",
            "qbound_cra_qbound_key_l1",

            "qbound_cra_qbound_js",
            "qbound_cra_final_js",
            "qbound_qbound_final_js",
            "qbound_original_merge_js",
            "qbound_pure_merge_js",

            "qbound_cra_key_entropy",
            "qbound_key_entropy",
            "qbound_final_key_entropy",
            "qbound_cra_key_entropy_ratio",
            "qbound_key_entropy_ratio",
            "qbound_final_key_entropy_ratio",
            "qbound_cra_key_max",
            "qbound_key_max",
            "qbound_final_key_max",
            "qbound_cra_peak_ratio",
            "qbound_peak_ratio",
            "qbound_final_peak_ratio",
            "qbound_cra_qbound_peak_shift",
            "qbound_cra_final_peak_shift",

            "qbound_boundary_prob_mean",
            "qbound_boundary_count",
            "qbound_event_count",
            "qbound_event_count_1_ratio",
            "qbound_event_count_2_ratio",
            "qbound_event_count_3_ratio",
            "qbound_event_count_4_ratio",
            "qbound_boundary_candidate_count",
            "qbound_selected_boundary_prob_mean",
            "qbound_boundary_prob_min",
            "qbound_boundary_prob_max",
            "qbound_assignment_entropy",
            "qbound_effective_event_count",
            "qbound_soft_effective_event_count",
            "qbound_event_relevance_entropy",
            "qbound_token_entropy",
            "qbound_pre_post_smooth_l1",
            "qbound_smoothing_peak_change_ratio",
            "qbound_frame_semantic_logit_abs_mean",
            "qbound_frame_semantic_logit_std",
            "qbound_weak_align_retrieval_acc",
            "qbound_weak_align_weight",
            "qbound_boundary_sparse_weight",
            "qbound_event_mass_min",
            "qbound_event_mass_max",

            "qbound_event_activity_mean",
            "qbound_event_activity_min",
            "qbound_event_activity_max",
            "qbound_transition_activity_mean",
            "qbound_transition_activity_max",

            "qbound_transition_mass_mean",
            "qbound_transition_mass_max",
            "qbound_transition_strength_mean",
            "qbound_transition_strength_max",

            "qbound_event_token_score_abs_mean",
            "qbound_event_token_score_abs_max",
            "qbound_transition_token_score_abs_mean",
            "qbound_transition_token_score_abs_max",
            "qbound_raw_event_token_logit_abs_max",
            "qbound_raw_transition_token_logit_abs_max",

            "qbound_question_only_conditioning",
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def get_loss(
        self,
        batch,
        qbound_mix=1.0,
    ):
        device = self._model_device()

        answer_id = batch["answer_id"]

        answer = batch["answer"].to(
            device,
            non_blocking=True,
        )

        video_frames = (
            batch["video_frames"].to(
                device,
                non_blocking=True,
            )
        )

        question = batch["question"].to(
            device,
            non_blocking=True,
        )

        qsns_id = batch["qsns_id"]
        qsns_token_ids = (
            batch["qsns_token_ids"]
        )
        video_len = batch["video_len"]

        video_mask = (
            get_mask(
                video_len,
                self.cfgs[
                    "dataset"
                ]["max_feats"],
            ).to(
                device,
                non_blocking=True,
            )
            if (
                self.cfgs[
                    "dataset"
                ]["max_feats"]
                > 0
            )
            else None
        )

        batch_size = answer_id.size(0)

        answer_id_device = (
            self._prepare_labels(
                answer_id,
                device,
            )
        )

        prediction_dict = {}
        target_dict = {}

        # --------------------------------------------------------------
        # Original CRA VQ objective: unchanged.
        # Q-Bound is disabled in mode="VQ" by models/cra.py.
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

            vq_predicts = (
                vq_predicts.reshape(
                    batch_size,
                    self.cfgs[
                        "model"
                    ]["prop_num"],
                    -1,
                ).mean(dim=1)
            )

            prediction_dict[
                "vq_pred"
            ] = vq_predicts

            target_dict[
                "vq_target"
            ] = (
                qsns_id.to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                ).reshape(-1)
            )

            prediction_dict[
                "vq_align_pred"
            ] = (
                vq_output[2][
                    "time_param"
                ][
                    "grounding_feature"
                ]
            )

            target_dict[
                "vq_align_gt"
            ] = (
                vq_output[2][
                    "align_gt"
                ]
            )

        # --------------------------------------------------------------
        # VQA main path:
        # deterministic CRA -> Q-Bound handover is applied in refine.py.
        # --------------------------------------------------------------
        gqa_output = self.model(
            video_frames,
            video_mask,
            answer=answer,
            answer_id=answer_id,
            mode="VQA",
            question=question,
            qbound_mix=(
                float(qbound_mix)
                if self._qbound_enabled()
                else 0.0
            ),
        )

        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]

        predicts = (
            self._score_answer_candidates(
                answer_proj=answer_proj,
                representation=(
                    main_video_proj
                ),
                detach_answer=False,
            )
        )

        prediction_dict[
            "vqa_pred"
        ] = predicts

        target_dict[
            "vqa_target"
        ] = answer_id_device

        # refine.py keeps grounding_feature = FINAL grounded feature.
        # Thus the original CRA VQA alignment objective now supervises the
        # same mixed/Q-Bound representation used by the main answer scorer.
        prediction_dict[
            "vqa_align_pred"
        ] = (
            gqa_output[2][
                "time_param"
            ][
                "grounding_feature"
            ]
        )

        target_dict[
            "vqa_align_gt"
        ] = (
            gqa_output[2][
                "align_gt"
            ]
        )

        # Repository CRA task loss is preserved exactly; Q-Bound V1.2 auxiliary objectives are added below.
        loss = self.criterion(
            prediction_dict,
            target_dict,
        )

        # --------------------------------------------------------------
        # Q-Bound V1.2 weak event alignment.
        #
        # This is intentionally different from CRA's existing global alignment:
        # it acts on the topology-pooled event evidence inside QBoundCore and a
        # question-only semantic state. Therefore it can shape event assignment
        # even during early CRA->Q-Bound handover without answer leakage.
        # --------------------------------------------------------------
        weak_align_retrieval_acc = None
        align_weight = 0.0
        boundary_sparse_weight = 0.0

        qbound_aux = self._get_qbound_aux(gqa_output)
        if self._qbound_enabled() and qbound_aux is not None:
            (
                align_weight,
                align_temperature,
                boundary_sparse_weight,
            ) = self._qbound_v12_loss_settings()

            if align_weight > 0.0:
                qbound_align_loss, weak_align_retrieval_acc = (
                    self._qbound_inbatch_alignment(
                        evidence_visual_token=qbound_aux.get(
                            "evidence_visual_token"
                        ),
                        question_state=qbound_aux.get("qa_state"),
                        temperature=align_temperature,
                    )
                )
                loss["qbound_align_loss"] = qbound_align_loss
                loss["total_loss"] = (
                    loss["total_loss"]
                    + align_weight * qbound_align_loss
                )

            # Disabled by default for the first V1.2 run. Turn this on only if
            # event_count again saturates at max_events.
            if boundary_sparse_weight > 0.0:
                qbound_boundary_sparse_loss = (
                    self._qbound_boundary_sparse_loss(qbound_aux)
                )
                loss["qbound_boundary_sparse_loss"] = (
                    qbound_boundary_sparse_loss
                )
                loss["total_loss"] = (
                    loss["total_loss"]
                    + boundary_sparse_weight * qbound_boundary_sparse_loss
                )

        # Counterfactuals are diagnostic only. No value returned here is
        # added to loss["total_loss"].
        qbound_logs = (
            self._compute_qbound_v12_diagnostics(
                gqa_output=gqa_output,
                answer_proj=answer_proj,
                answer_id=answer_id,
                main_logits=predicts,
            )
        )

        if self._qbound_enabled():
            qbound_logs["qbound_weak_align_weight"] = float(align_weight)
            qbound_logs["qbound_boundary_sparse_weight"] = float(
                boundary_sparse_weight
            )
            if weak_align_retrieval_acc is not None:
                qbound_logs["qbound_weak_align_retrieval_acc"] = float(
                    weak_align_retrieval_acc
                )

        predicted = (
            predicts.argmax(
                dim=-1
            )
            .detach()
            .cpu()
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
        self.monitor.reset_kv("total_loss")

        for key in self.cfgs["optim"]["loss"].keys():
            if key in {"ce", "align"}:
                continue
            self.monitor.reset_kv(f"{key}_loss")

        if self._qbound_enabled():
            # Loss dict entries are logged separately below, but resetting them
            # here keeps epoch statistics clean when an auxiliary weight changes.
            self.monitor.reset_kv("qbound_align_loss")
            self.monitor.reset_kv("qbound_boundary_sparse_loss")
            for key in self._qbound_monitor_keys():
                self.monitor.reset_kv(key)

        for index, batch in enumerate(
            self.train_dataloader
        ):
            start_time = time.time()

            qbound_mix = (
                self._train_qbound_mix(
                    epoch=epoch,
                    batch_index=index,
                    batches_per_epoch=len(
                        self.train_dataloader
                    ),
                )
            )

            loss, monitor_metric, qbound_stats = (
                self.get_loss(
                    batch,
                    qbound_mix=qbound_mix,
                )
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

            if self._qbound_enabled():
                qbound_grad_norm = (
                    self._qbound_grad_norm()
                )
                self.monitor.logkv_mean(
                    "qbound_grad_norm",
                    qbound_grad_norm,
                )

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

            if self._qbound_enabled():
                print_info += (
                    f" qmix: {qbound_mix:.3f}"
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

        temporal_diagnostic_prefixes = (
            "val_QBound-",
            "test_QBound-",
            "val_CRA-",
            "test_CRA-",
            "val_QBoundPreSmooth-",
            "test_QBoundPreSmooth-",
            "val_QBoundPostSmooth-",
            "test_QBoundPostSmooth-",
        )

        temporal_diagnostic_keys = [
            key
            for key in score
            if key.startswith(temporal_diagnostic_prefixes)
        ]

        if temporal_diagnostic_keys:
            self.monitor.log_info(
                "\n============ Temporal-key faithfulness "
                "============"
            )
            for key in sorted(temporal_diagnostic_keys):
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

        # Q-Bound V1.2 temporal-key diagnostics.
        # Keep the pure Q-Bound and CRA teacher keys separately so we can
        # directly test whether Q-Bound is selective AND temporally correct.
        qbound_keyframe = {}
        cra_keyframe = {}
        qbound_pre_smooth_keyframe = {}
        qbound_post_smooth_keyframe = {}

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
                    # V1.2 checkpoint selection/reporting is ALWAYS
                    # evaluated on the pure Q-Bound temporal path.
                    qbound_mix=(
                        1.0
                        if self._qbound_enabled()
                        else 0.0
                    ),
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
                    counterfactual_logs = (
                        self._compute_qbound_v12_diagnostics(
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

                    # Ver1.2: collect the temporal distributions needed by
                    # Metric.temporal_key_metrics().  Validation/test always
                    # run with qbound_mix=1, so qbound_key is the pure Q-Bound
                    # grounding distribution used for model selection.
                    batch_qbound_keyframe = None
                    batch_cra_keyframe = None
                    batch_qbound_pre_smooth_keyframe = None
                    batch_qbound_post_smooth_keyframe = None

                    if qbound_aux is not None:
                        pure_qbound_key = qbound_aux.get(
                            "qbound_key",
                            qbound_aux.get("event_key"),
                        )
                        pure_cra_key = qbound_aux.get("cra_key")
                        pre_smooth_key = qbound_aux.get(
                            "qbound_pre_smooth_key"
                        )
                        post_smooth_key = qbound_aux.get(
                            "qbound_post_smooth_key",
                            pure_qbound_key,
                        )

                        if torch.is_tensor(pure_qbound_key):
                            batch_qbound_keyframe = (
                                pure_qbound_key.detach().cpu().numpy()
                            )

                        if torch.is_tensor(pure_cra_key):
                            batch_cra_keyframe = (
                                pure_cra_key.detach().cpu().numpy()
                            )

                        if torch.is_tensor(pre_smooth_key):
                            batch_qbound_pre_smooth_keyframe = (
                                pre_smooth_key.detach().cpu().numpy()
                            )

                        if torch.is_tensor(post_smooth_key):
                            batch_qbound_post_smooth_keyframe = (
                                post_smooth_key.detach().cpu().numpy()
                            )

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

                        if batch_qbound_keyframe is not None:
                            qbound_keyframe[qid] = (
                                batch_qbound_keyframe[
                                    batch_index
                                ].tolist()
                            )

                        if batch_cra_keyframe is not None:
                            cra_keyframe[qid] = (
                                batch_cra_keyframe[
                                    batch_index
                                ].tolist()
                            )

                        if batch_qbound_pre_smooth_keyframe is not None:
                            qbound_pre_smooth_keyframe[qid] = (
                                batch_qbound_pre_smooth_keyframe[
                                    batch_index
                                ].tolist()
                            )

                        if batch_qbound_post_smooth_keyframe is not None:
                            qbound_post_smooth_keyframe[qid] = (
                                batch_qbound_post_smooth_keyframe[
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

        # --------------------------------------------------------------
        # Q-Bound V1.2 temporal-faithfulness diagnostics.
        #
        # Official benchmark metrics above remain untouched. These extra
        # diagnostics answer the failure mode seen in V1.1.3:
        #   "the key is selective, but is it selective in the correct GT
        #    temporal region?"
        # --------------------------------------------------------------
        if qbound_keyframe:
            if not hasattr(self.metric, "temporal_key_metrics"):
                raise AttributeError(
                    "Q-Bound V1.2 requires Metric.temporal_key_metrics(). "
                    "Please use metrics_qbound_v1_2.py."
                )

            score.update(
                self.metric.temporal_key_metrics(
                    qbound_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBound",
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

        # Pre/post smoothing diagnostics are useful for identifying whether
        # Gaussian smoothing changes an otherwise-correct Q-Bound peak.
        if qbound_pre_smooth_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_pre_smooth_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundPreSmooth",
                )
            )

        if qbound_post_smooth_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_post_smooth_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundPostSmooth",
                )
            )

        result = {
            "ph": post_hoc_ground,
            "result": results,
            "gt_ground": ground_truth,
            "ph_attn": ground_res,
            "qbound_key": qbound_keyframe,
            "cra_key": cra_keyframe,
            "qbound_pre_smooth_key": qbound_pre_smooth_keyframe,
            "qbound_post_smooth_key": qbound_post_smooth_keyframe,
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