import math
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
    CRA training/evaluation pipeline with Q-Bound Version 1.11.

    V1.11 changes the Q-Bound INPUT contract while retaining the successful
    temporal mechanisms from V1.10.2:

        raw/generic CLIP-L tokens
            -> Q-Bound-owned input adapter
            -> DECOMPOSE -> RELATE -> COMPOSE
            -> BAPGR -> PPBR -> PSRC -> temporal support

    CRA remains only the downstream VideoQA consumer/testbed.  The CRA branch
    still uses its own VideoEmbedding/refinement representation for answer
    scoring, while Q-Bound receives the pre-CRA generic visual tokens through
    an explicit, separately validated route.

    Controlled V1.11-B objective:

        L = L_CRA/main
            + lambda_align * L_weak_align
            + lambda_decomp * L_decomp
            + lambda_suf * L_sufficiency
            + lambda_nec * L_necessity

    Counterfactual supervision uses the mass-preserving differentiable MAIN
    support exposed by QBoundCore and the frozen CRA counterfactual consumer
    exposed by RefineDecoder/CRA:

        X_suff = keep * X + remove * X_donor
        X_nec  = remove * X + keep * X_donor

    Sufficiency requires the preserved-evidence branch to answer the original
    question correctly. Necessity requires removal of selected evidence to
    decrease the correct-answer logit relative to the detached factual branch.
    Necessity is applied only to factual-correct samples by default so early
    training does not reinforce evidence for an already-wrong factual answer.

    The counterfactual answer candidates, CRA sequence, donor sequence, MAIN
    pooling key, and CRA answer projection are detached/frozen on the CF path.
    The added objective therefore supervises Q-Bound temporal evidence rather
    than teaching CRA/language consumers to adapt to feature interventions.

    No temporal GT, IoU/IoP training loss, duration/minimality target,
    event-count forcing, CRA temporal handover, or direct event-token answer
    residual is introduced.
    """

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        self.qbound_version = "1.11"
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
        """
        Compatibility helper.

        V1.4 removes the CRA -> Q-Bound temporal handover entirely.
        The main VQA temporal path is pure Q-Bound whenever Q-Bound is active.
        """
        return 1.0, 0

    def _qbound_v111_loss_settings(self):
        """
        Read the shared V1.11 objective settings.

        Preferred YAML:

            model:
              qbound:
                weak_align_weight: 0.10
                weak_align_temperature: 0.07
                decomposition:
                  loss_weight: 0.10

        V1.11 keeps the V1.10.3 task/alignment/decomposition objective unchanged
        so generic-input decoupling is isolated. No coarse-answer auxiliary,
        minimal-sufficient/span-length, or reliability loss is added.
        """
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        decomp_cfg = qbound_cfg.get("decomposition", {})
        if not isinstance(decomp_cfg, dict):
            decomp_cfg = {}

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

        if align_weight < 0.0:
            raise ValueError("qbound weak_align_weight must be >= 0.")
        if align_temperature <= 0.0:
            raise ValueError("qbound weak_align_temperature must be > 0.")
        if decomp_weight < 0.0:
            raise ValueError("qbound decomp_weight must be >= 0.")

        return (
            align_weight,
            align_temperature,
            decomp_weight,
        )

    def _qbound_v111b_counterfactual_settings(self):
        """
        Read V1.11-B counterfactual evidence-training settings.

        Preferred YAML:

            model:
              qbound:
                counterfactual_evidence_interface: true
                counterfactual:
                  training_enabled: true
                  sufficiency_weight: 0.05
                  necessity_weight: 0.05
                  necessity_margin: 0.20
                  necessity_correct_only: true

        The first controlled V1.11-B run changes only evidence supervision.
        It does not change BAPGR/PPBR/PSRC, span length, topology, or the
        generic-input adapter.
        """
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        interface_enabled = bool(
            qbound_cfg.get(
                "counterfactual_evidence_interface",
                False,
            )
        )

        cf_cfg = qbound_cfg.get("counterfactual", {})
        if cf_cfg is None:
            cf_cfg = {}
        if not isinstance(cf_cfg, dict):
            raise TypeError(
                "model.qbound.counterfactual must be a mapping/dict."
            )

        training_enabled = bool(
            cf_cfg.get(
                "training_enabled",
                qbound_cfg.get(
                    "counterfactual_training_enabled",
                    interface_enabled,
                ),
            )
        )
        sufficiency_weight = float(
            cf_cfg.get(
                "sufficiency_weight",
                qbound_cfg.get(
                    "counterfactual_sufficiency_weight",
                    0.05,
                ),
            )
        )
        necessity_weight = float(
            cf_cfg.get(
                "necessity_weight",
                qbound_cfg.get(
                    "counterfactual_necessity_weight",
                    0.05,
                ),
            )
        )
        necessity_margin = float(
            cf_cfg.get(
                "necessity_margin",
                qbound_cfg.get(
                    "counterfactual_necessity_margin",
                    0.20,
                ),
            )
        )
        necessity_correct_only = bool(
            cf_cfg.get(
                "necessity_correct_only",
                True,
            )
        )

        if training_enabled and not interface_enabled:
            raise RuntimeError(
                "Q-Bound V1.11-B counterfactual training requires "
                "model.qbound.counterfactual_evidence_interface=true."
            )
        if sufficiency_weight < 0.0:
            raise ValueError(
                "counterfactual sufficiency_weight must be >= 0."
            )
        if necessity_weight < 0.0:
            raise ValueError(
                "counterfactual necessity_weight must be >= 0."
            )
        if necessity_margin < 0.0:
            raise ValueError(
                "counterfactual necessity_margin must be >= 0."
            )
        if (
            training_enabled
            and sufficiency_weight <= 0.0
            and necessity_weight <= 0.0
        ):
            raise ValueError(
                "V1.11-B counterfactual training is enabled but both "
                "counterfactual loss weights are zero."
            )

        return {
            "interface_enabled": interface_enabled,
            "training_enabled": training_enabled,
            "sufficiency_weight": sufficiency_weight,
            "necessity_weight": necessity_weight,
            "necessity_margin": necessity_margin,
            "necessity_correct_only": necessity_correct_only,
        }

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
                "Q-Bound V1.11 weak alignment requires evidence_visual_token "
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
    def _qbound_decomposition_loss(qbound_aux):
        """
        Return the mean V1.4 latent-partition objective.

        QBoundCore exposes one expected partition energy per sample.  This is
        the ONLY new topology objective in the first V1.4 run; no boundary
        sparsity, event-balance, entropy, diversity, or necessity losses are
        added.
        """
        decomp_loss = qbound_aux.get("decomp_loss")
        if not torch.is_tensor(decomp_loss):
            raise KeyError(
                "Q-Bound V1.11 decomposition requires qbound_aux['decomp_loss']."
            )

        value = decomp_loss.mean()

        if not torch.isfinite(value):
            raise FloatingPointError(
                "Q-Bound V1.11 decomposition loss is NaN/Inf."
            )

        return value

    def _train_qbound_mix(
        self,
        epoch,
        batch_index,
        batches_per_epoch,
    ):
        """
        Compatibility return value for old CRA.forward callers.

        V1.4 has no handover curriculum:
            Q-Bound enabled  -> 1.0
            Q-Bound disabled -> 0.0

        ``epoch``, ``batch_index`` and ``batches_per_epoch`` are intentionally
        ignored so resumed runs cannot accidentally recreate V1.4 mixing.
        """
        del epoch, batch_index, batches_per_epoch
        return 1.0 if self._qbound_enabled() else 0.0

    def _get_qbound_aux(self, gqa_output):
        """Safely extract Q-Bound V1.11 outputs from a CRA VQA forward."""
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
        VQA path.  V1.4 counterfactuals therefore do not use a separate
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

    def _qbound_v111b_counterfactual_losses(
        self,
        model_outputs,
        labels,
        factual_logits,
        settings,
    ):
        """
        Compute V1.11-B answer-level sufficiency and necessity losses.

        Gradient contract:
          * counterfactual answer candidates are detached by CRA;
          * CF video representations were built from detached CRA features and
            projected by a frozen CRA consumer in RefineDecoder;
          * factual logits are detached inside the necessity target;
          * gradients therefore enter Q-Bound through the differentiable
            keep/remove temporal support.

        Sufficiency:
            CE(f(X_suff, q), y)

        Necessity:
            relu(
                margin
                - [ stopgrad(s_y(X)) - s_y(X_nec) ]
            )

        By default necessity is evaluated only for samples whose factual MAIN
        prediction is already correct.
        """
        if not isinstance(model_outputs, dict):
            raise TypeError(
                "V1.11-B counterfactual training requires model_outputs dict."
            )

        answer_proj = model_outputs.get(
            "qbound_counterfactual_answer_proj"
        )
        sufficient_proj = model_outputs.get(
            "qbound_counterfactual_sufficient_video_proj"
        )
        necessary_proj = model_outputs.get(
            "qbound_counterfactual_necessary_video_proj"
        )
        donor_available = model_outputs.get(
            "qbound_counterfactual_donor_available"
        )
        consumer_frozen = model_outputs.get(
            "qbound_counterfactual_consumer_frozen"
        )

        required = {
            "qbound_counterfactual_answer_proj": answer_proj,
            "qbound_counterfactual_sufficient_video_proj": sufficient_proj,
            "qbound_counterfactual_necessary_video_proj": necessary_proj,
            "qbound_counterfactual_donor_available": donor_available,
            "qbound_counterfactual_consumer_frozen": consumer_frozen,
        }
        missing = [
            name
            for name, value in required.items()
            if value is None
        ]
        if missing:
            raise KeyError(
                "V1.11-B counterfactual outputs are missing: "
                + ", ".join(missing)
            )

        if not torch.is_tensor(answer_proj):
            raise TypeError(
                "qbound_counterfactual_answer_proj must be a tensor."
            )
        if not torch.is_tensor(sufficient_proj):
            raise TypeError(
                "qbound_counterfactual_sufficient_video_proj must be a tensor."
            )
        if not torch.is_tensor(necessary_proj):
            raise TypeError(
                "qbound_counterfactual_necessary_video_proj must be a tensor."
            )
        if not torch.is_tensor(donor_available):
            raise TypeError(
                "qbound_counterfactual_donor_available must be a tensor."
            )
        if not torch.is_tensor(consumer_frozen):
            raise TypeError(
                "qbound_counterfactual_consumer_frozen must be a tensor."
            )

        donor_flag = float(donor_available.detach().item())
        frozen_flag = float(consumer_frozen.detach().item())
        if frozen_flag != 1.0:
            raise RuntimeError(
                "V1.11-B counterfactual consumer must remain frozen."
            )

        sufficient_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=sufficient_proj,
            detach_answer=True,
        )
        necessary_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=necessary_proj,
            detach_answer=True,
        )

        if sufficient_logits.shape != factual_logits.shape:
            raise ValueError(
                "V1.11-B sufficient logits must match factual logits."
            )
        if necessary_logits.shape != factual_logits.shape:
            raise ValueError(
                "V1.11-B necessary logits must match factual logits."
            )

        labels = labels.reshape(-1)
        if labels.numel() != factual_logits.size(0):
            raise ValueError(
                "V1.11-B labels do not match factual batch size."
            )

        # If batch_size==1 there is no valid in-batch donor. Keep graph-valid
        # zero losses and let the caller skip the CF contribution.
        if donor_flag <= 0.0:
            zero = (
                sufficient_logits.sum()
                + necessary_logits.sum()
            ) * 0.0
            return {
                "sufficiency_loss": zero,
                "necessity_loss": zero,
                "sufficient_logits": sufficient_logits,
                "necessary_logits": necessary_logits,
                "necessity_active_ratio": 0.0,
                "donor_available": 0.0,
            }

        sufficiency_per_sample = F.cross_entropy(
            sufficient_logits,
            labels,
            reduction="none",
        )
        sufficiency_loss = sufficiency_per_sample.mean()

        label_index = labels.unsqueeze(-1)
        factual_correct_logit = factual_logits.detach().gather(
            dim=-1,
            index=label_index,
        ).squeeze(-1)
        sufficient_correct_logit = sufficient_logits.gather(
            dim=-1,
            index=label_index,
        ).squeeze(-1)
        necessary_correct_logit = necessary_logits.gather(
            dim=-1,
            index=label_index,
        ).squeeze(-1)

        necessity_gap = (
            factual_correct_logit
            - necessary_correct_logit
        )
        necessity_violation = F.relu(
            float(settings["necessity_margin"])
            - necessity_gap
        )

        if bool(settings["necessity_correct_only"]):
            factual_correct_mask = (
                factual_logits.detach().argmax(dim=-1) == labels
            ).to(necessity_violation.dtype)
        else:
            factual_correct_mask = torch.ones_like(
                necessity_violation
            )

        active_count = factual_correct_mask.sum()
        if float(active_count.detach().item()) > 0.0:
            necessity_loss = (
                necessity_violation
                * factual_correct_mask
            ).sum() / active_count
        else:
            necessity_loss = necessary_logits.sum() * 0.0

        necessity_active_ratio = (
            factual_correct_mask.mean().detach().item()
        )

        with torch.no_grad():
            factual_prob = F.softmax(
                factual_logits.detach(),
                dim=-1,
            ).gather(
                dim=-1,
                index=label_index,
            ).squeeze(-1)
            sufficient_prob = F.softmax(
                sufficient_logits.detach(),
                dim=-1,
            ).gather(
                dim=-1,
                index=label_index,
            ).squeeze(-1)
            necessary_prob = F.softmax(
                necessary_logits.detach(),
                dim=-1,
            ).gather(
                dim=-1,
                index=label_index,
            ).squeeze(-1)

            diagnostics = {
                "factual_correct_logit": (
                    factual_correct_logit.mean().item()
                ),
                "sufficient_correct_logit": (
                    sufficient_correct_logit.detach().mean().item()
                ),
                "necessary_correct_logit": (
                    necessary_correct_logit.detach().mean().item()
                ),
                "factual_minus_sufficient_logit": (
                    factual_correct_logit
                    - sufficient_correct_logit.detach()
                ).mean().item(),
                "necessity_gap": necessity_gap.detach().mean().item(),
                "factual_correct_prob": factual_prob.mean().item(),
                "sufficient_correct_prob": sufficient_prob.mean().item(),
                "necessary_correct_prob": necessary_prob.mean().item(),
                "sufficient_acc": (
                    sufficient_logits.detach().argmax(dim=-1) == labels
                ).float().mean().item(),
                "necessary_acc": (
                    necessary_logits.detach().argmax(dim=-1) == labels
                ).float().mean().item(),
                "necessity_margin_satisfied_ratio": (
                    necessity_gap.detach()
                    >= float(settings["necessity_margin"])
                ).float().mean().item(),
            }

        return {
            "sufficiency_loss": sufficiency_loss,
            "necessity_loss": necessity_loss,
            "sufficient_logits": sufficient_logits,
            "necessary_logits": necessary_logits,
            "necessity_active_ratio": necessity_active_ratio,
            "donor_available": donor_flag,
            **diagnostics,
        }

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
        Diagnostic only in V1.4.

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
    # Q-Bound V1.11 diagnostics
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

    def _compute_qbound_v111_diagnostics(
        self,
        gqa_output,
        answer_proj,
        answer_id,
        main_logits,
        raw_refined_logits=None,
        coarse_logits=None,
    ):
        """
        Diagnostics only; nothing returned here is added to total_loss.

        V1.11 retains the three same-checkpoint temporal branches:

            MAIN        = posterior-stable reliability-calibrated support
            RAW REFINE  = BAPGR + PPBR bounded fine proposal
            COARSE      = hard-top1 COMPOSE diagnostic

        The diagnostics answer:
          1) whether the detached Composer pair prior remains uncertain;
          2) whether REFINE adds information beyond that prior;
          3) how much coarse semantic evidence REFINE preserves;
          4) how much the new reliability suppresses V1.8-style entropy trust;
          5) whether MAIN retains coverage while improving precision;
          6) whether hard-top1 routing and V1.6/V1.7 failures remain disabled.
        """
        qbound_aux = self._get_qbound_aux(gqa_output)
        if qbound_aux is None:
            return {}

        required_keys = {
            # V1.11 generic-input / counterfactual interface.
            "qbound_version",
            "generic_input_adapter_enabled",
            "generic_input_position_enabled",
            "generic_input_context_reuse",
            "generic_input_raw_rms",
            "generic_input_adapted_rms",
            "generic_input_token_cosine",
            "generic_input_delta_ratio",
            "counterfactual_evidence_interface_enabled",
            "counterfactual_mass_preserving_gradient_enabled",
            "counterfactual_raw_keep_support",
            "counterfactual_raw_remove_support",
            "counterfactual_raw_keep_fraction",
            "counterfactual_keep_support",
            "counterfactual_remove_support",
            "counterfactual_keep_fraction",
            "counterfactual_remove_fraction",
            "counterfactual_support_forward_error",
            "counterfactual_support_complement_error",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",

            "qbound_counterfactual_donor_available",
            "qbound_counterfactual_donor_shift",
            "qbound_counterfactual_donor_valid_overlap",
            "qbound_counterfactual_consumer_frozen",

            "cra_video_proj",
            "main_video_proj",
            "qbound_calibrated_video_proj",
            "qbound_raw_refined_video_proj",
            "qbound_coarse_video_proj",

            "cra_key",
            "qbound_calibrated_key",
            "qbound_raw_refined_key",
            "qbound_coarse_key",
            "final_key",
            "qbound_pre_smooth_key",
            "qbound_post_smooth_key",

            "calibration_uncertainty",
            "calibration_information",
            "calibration_confidence",
            "calibration_confidence_detached",

            # V1.11 BAPGR + PPBR routing / reliability.
            "posterior_guided_refinement_enabled",
            "boundary_structured_refinement_enabled",
            "prior_preserving_bounded_refinement_enabled",
            "posterior_stable_reliability_enabled",
            "refine_boundary_anchor_power",
            "refine_correction_cap",
            "refine_hard_coarse_route",
            "calibration_reliability",
            "refiner_information_gain",
            "refined_semantic_retention",
            "base_refiner_confidence",
            "reliability_gain_factor",
            "composer_pair_prior_entropy",
            "composer_pair_prior_entropy_ratio",
            "composer_pair_prior_effective_count",
            "composer_pair_prior_top1_posterior",
            "composer_pair_prior_candidate_count",
            "composer_uniform_pair_prior_entropy",
            "composer_uniform_pair_prior_entropy_ratio",
            "composer_uniform_pair_prior_effective_count",
            "composer_uniform_pair_prior_top1_posterior",
            "composer_pair_prior_expected_span_length",
            "composer_uniform_pair_prior_expected_span_length",
            "composer_pair_prior_length_gain",
            "composer_expected_span_length",
            "main_pair_entropy",
            "main_pair_effective_count",
            "main_pair_top1_posterior",

            # V1.10.2 PPBR diagnostics.
            "refiner_raw_correction_abs_mean",
            "refiner_raw_correction_abs_max",
            "refiner_bounded_correction_abs_mean",
            "refiner_bounded_correction_abs_max",
            "refiner_correction_saturation_ratio",
            "refiner_vs_prior_kl",
            "refiner_vs_prior_js",

            "expected_refined_span_length",
            "raw_vs_prior_expected_length_ratio",
            "expected_refined_length_ratio",
            "calibrated_expected_span_length",
            "calibrated_expected_length_ratio",

            "coarse_span_frame_length",
            "refined_span_frame_length",
            "calibrated_span_frame_length",
            "calibrated_retained_fraction",

            "refine_coarse_detached",
            "all_span_refine_marginal_enabled",
            "uncertainty_calibration_enabled",
            "minimal_sufficient_enabled",

            "qbound_handover_enabled",
            "qbound_direct_reasoning_residual_enabled",
        }

        missing = sorted(required_keys.difference(qbound_aux))
        if missing:
            raise KeyError(
                "Missing Q-Bound V1.11 diagnostic outputs: "
                + ", ".join(missing)
            )

        labels = self._prepare_labels(
            answer_id,
            answer_proj.device,
        )

        if main_logits.size(0) != labels.numel():
            raise ValueError(
                "answer_id size does not match calibrated MAIN logits."
            )

        with torch.no_grad():
            if str(qbound_aux["qbound_version"]) != self.qbound_version:
                raise RuntimeError(
                    "Pipeline/Core Q-Bound version mismatch: "
                    f"pipeline={self.qbound_version!r}, "
                    f"core={qbound_aux['qbound_version']!r}."
                )

            generic_input_adapter_enabled = qbound_aux[
                "generic_input_adapter_enabled"
            ].detach().float()
            generic_input_position_enabled = qbound_aux[
                "generic_input_position_enabled"
            ].detach().float()
            generic_input_context_reuse = qbound_aux[
                "generic_input_context_reuse"
            ].detach().float()
            generic_input_raw_rms = qbound_aux[
                "generic_input_raw_rms"
            ].detach().float()
            generic_input_adapted_rms = qbound_aux[
                "generic_input_adapted_rms"
            ].detach().float()
            generic_input_token_cosine = qbound_aux[
                "generic_input_token_cosine"
            ].detach().float()
            generic_input_delta_ratio = qbound_aux[
                "generic_input_delta_ratio"
            ].detach().float()

            input_is_generic = qbound_aux[
                "qbound_input_is_generic"
            ].detach().float()
            input_uses_cra_embedding = qbound_aux[
                "qbound_input_uses_cra_embedding"
            ].detach().float()

            counterfactual_interface_enabled = qbound_aux[
                "counterfactual_evidence_interface_enabled"
            ].detach().float()
            counterfactual_keep_support = qbound_aux[
                "counterfactual_keep_support"
            ].detach().float()
            counterfactual_remove_support = qbound_aux[
                "counterfactual_remove_support"
            ].detach().float()
            counterfactual_keep_fraction = qbound_aux[
                "counterfactual_keep_fraction"
            ].detach().float()
            counterfactual_remove_fraction = qbound_aux[
                "counterfactual_remove_fraction"
            ].detach().float()
            counterfactual_mass_guard = qbound_aux[
                "counterfactual_mass_preserving_gradient_enabled"
            ].detach().float()
            counterfactual_forward_error = qbound_aux[
                "counterfactual_support_forward_error"
            ].detach().float()
            counterfactual_core_complement_error = qbound_aux[
                "counterfactual_support_complement_error"
            ].detach().float()
            counterfactual_donor_available = qbound_aux[
                "qbound_counterfactual_donor_available"
            ].detach().float()
            counterfactual_donor_overlap = qbound_aux[
                "qbound_counterfactual_donor_valid_overlap"
            ].detach().float()
            counterfactual_consumer_frozen = qbound_aux[
                "qbound_counterfactual_consumer_frozen"
            ].detach().float()

            # Core contract: keep/remove are complementary on valid frames and
            # both are zero outside the valid temporal mask.  Infer validity
            # from their sum so the pipeline does not duplicate dataset logic.
            cf_sum = counterfactual_keep_support + counterfactual_remove_support
            cf_valid = (cf_sum > 0.5).to(cf_sum.dtype)
            counterfactual_complement_error = (
                cf_sum - cf_valid
            ).abs().mean()
            counterfactual_overlap = (
                counterfactual_keep_support
                * counterfactual_remove_support
            ).mean()

            calibrated_logits = main_logits.detach()

            if raw_refined_logits is None:
                raw_refined_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=qbound_aux[
                        "qbound_raw_refined_video_proj"
                    ],
                    detach_answer=True,
                )
            raw_refined_logits = raw_refined_logits.detach()

            if coarse_logits is None:
                coarse_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=qbound_aux[
                        "qbound_coarse_video_proj"
                    ],
                    detach_answer=True,
                )
            coarse_logits = coarse_logits.detach()

            cra_logits = self._score_answer_candidates(
                answer_proj=answer_proj,
                representation=qbound_aux["cra_video_proj"],
                detach_answer=True,
            )

            calibrated_ce = F.cross_entropy(
                calibrated_logits,
                labels,
                reduction="none",
            )
            raw_refined_ce = F.cross_entropy(
                raw_refined_logits,
                labels,
                reduction="none",
            )
            coarse_ce = F.cross_entropy(
                coarse_logits,
                labels,
                reduction="none",
            )
            cra_ce = F.cross_entropy(
                cra_logits,
                labels,
                reduction="none",
            )

            calibrated_pred = calibrated_logits.argmax(dim=-1)
            raw_refined_pred = raw_refined_logits.argmax(dim=-1)
            coarse_pred = coarse_logits.argmax(dim=-1)
            cra_pred = cra_logits.argmax(dim=-1)

            calibrated_key = self._normalize_distribution(
                qbound_aux["qbound_calibrated_key"]
            )
            raw_refined_key = self._normalize_distribution(
                qbound_aux["qbound_raw_refined_key"]
            )
            coarse_key = self._normalize_distribution(
                qbound_aux["qbound_coarse_key"]
            )
            cra_key = self._normalize_distribution(
                qbound_aux["cra_key"]
            )
            final_key = self._normalize_distribution(
                qbound_aux["final_key"]
            )
            pre_smooth_key = self._normalize_distribution(
                qbound_aux["qbound_pre_smooth_key"]
            )
            post_smooth_key = self._normalize_distribution(
                qbound_aux["qbound_post_smooth_key"]
            )

            calibrated_raw_js = self._js_divergence(
                calibrated_key,
                raw_refined_key,
            ).mean()
            calibrated_coarse_js = self._js_divergence(
                calibrated_key,
                coarse_key,
            ).mean()
            raw_coarse_js = self._js_divergence(
                raw_refined_key,
                coarse_key,
            ).mean()
            cra_calibrated_js = self._js_divergence(
                cra_key,
                calibrated_key,
            ).mean()

            calibrated_raw_l1 = (
                calibrated_key - raw_refined_key
            ).abs().mean()
            calibrated_coarse_l1 = (
                calibrated_key - coarse_key
            ).abs().mean()
            raw_coarse_l1 = (
                raw_refined_key - coarse_key
            ).abs().mean()

            pre_post_js = self._js_divergence(
                pre_smooth_key,
                post_smooth_key,
            ).mean()
            final_l1 = (
                calibrated_key - final_key
            ).abs().mean()

            v15_key_raw = qbound_aux.get("v15_diagnostic_key")
            v111_v15_js = None
            v111_v15_l1 = None
            if torch.is_tensor(v15_key_raw):
                v15_key = self._normalize_distribution(v15_key_raw)
                v111_v15_js = self._js_divergence(
                    calibrated_key,
                    v15_key,
                ).mean()
                v111_v15_l1 = (
                    calibrated_key - v15_key
                ).abs().mean()

            v14_key_raw = qbound_aux.get("v14_diagnostic_key")
            v111_v14_js = None
            v111_v14_l1 = None
            if torch.is_tensor(v14_key_raw):
                v14_key = self._normalize_distribution(v14_key_raw)
                v111_v14_js = self._js_divergence(
                    calibrated_key,
                    v14_key,
                ).mean()
                v111_v14_l1 = (
                    calibrated_key - v14_key
                ).abs().mean()

            calibrated_entropy = self._distribution_entropy(
                calibrated_key
            )
            raw_refined_entropy = self._distribution_entropy(
                raw_refined_key
            )
            coarse_entropy = self._distribution_entropy(
                coarse_key
            )
            cra_entropy = self._distribution_entropy(
                cra_key
            )

            valid_length = torch.full(
                (calibrated_key.size(0),),
                float(calibrated_key.size(-1)),
                dtype=calibrated_key.dtype,
                device=calibrated_key.device,
            ).clamp_min(2.0)
            uniform_entropy = torch.log(valid_length)

            calibrated_peak = calibrated_key.max(dim=-1).values
            raw_refined_peak = raw_refined_key.max(dim=-1).values
            coarse_peak = coarse_key.max(dim=-1).values

            uncertainty = qbound_aux[
                "calibration_uncertainty"
            ].detach().float()
            information = qbound_aux[
                "calibration_information"
            ].detach().float()
            # Legacy name ``calibration_confidence`` is an alias of the
            # V1.9 detached reliability; keep both to make V1.8/V1.9 plots
            # directly comparable.
            confidence = qbound_aux[
                "calibration_confidence"
            ].detach().float()
            reliability = qbound_aux[
                "calibration_reliability"
            ].detach().float()
            information_gain = qbound_aux[
                "refiner_information_gain"
            ].detach().float()
            semantic_retention = qbound_aux[
                "refined_semantic_retention"
            ].detach().float()
            base_refiner_confidence = qbound_aux[
                "base_refiner_confidence"
            ].detach().float()
            reliability_gain_factor = qbound_aux[
                "reliability_gain_factor"
            ].detach().float()

            boundary_anchor_power = qbound_aux[
                "refine_boundary_anchor_power"
            ].detach().float()
            correction_cap = qbound_aux[
                "refine_correction_cap"
            ].detach().float()

            refiner_raw_correction_abs_mean = qbound_aux[
                "refiner_raw_correction_abs_mean"
            ].detach().float()
            refiner_raw_correction_abs_max = qbound_aux[
                "refiner_raw_correction_abs_max"
            ].detach().float()
            refiner_bounded_correction_abs_mean = qbound_aux[
                "refiner_bounded_correction_abs_mean"
            ].detach().float()
            refiner_bounded_correction_abs_max = qbound_aux[
                "refiner_bounded_correction_abs_max"
            ].detach().float()
            refiner_correction_saturation_ratio = qbound_aux[
                "refiner_correction_saturation_ratio"
            ].detach().float()
            refiner_vs_prior_kl = qbound_aux[
                "refiner_vs_prior_kl"
            ].detach().float()
            refiner_vs_prior_js = qbound_aux[
                "refiner_vs_prior_js"
            ].detach().float()
            raw_vs_prior_expected_length_ratio = qbound_aux[
                "raw_vs_prior_expected_length_ratio"
            ].detach().float()

            composer_uniform_pair_prior_entropy = qbound_aux[
                "composer_uniform_pair_prior_entropy"
            ].detach().float()
            composer_uniform_pair_prior_entropy_ratio = qbound_aux[
                "composer_uniform_pair_prior_entropy_ratio"
            ].detach().float()
            composer_uniform_pair_prior_effective_count = qbound_aux[
                "composer_uniform_pair_prior_effective_count"
            ].detach().float()
            composer_uniform_pair_prior_top1 = qbound_aux[
                "composer_uniform_pair_prior_top1_posterior"
            ].detach().float()
            composer_pair_prior_expected_span_length = qbound_aux[
                "composer_pair_prior_expected_span_length"
            ].detach().float()
            composer_uniform_pair_prior_expected_span_length = qbound_aux[
                "composer_uniform_pair_prior_expected_span_length"
            ].detach().float()
            composer_pair_prior_length_gain = qbound_aux[
                "composer_pair_prior_length_gain"
            ].detach().float()

            composer_pair_prior_entropy = qbound_aux[
                "composer_pair_prior_entropy"
            ].detach().float()
            composer_pair_prior_entropy_ratio = qbound_aux[
                "composer_pair_prior_entropy_ratio"
            ].detach().float()
            composer_pair_prior_effective_count = qbound_aux[
                "composer_pair_prior_effective_count"
            ].detach().float()
            composer_pair_prior_top1 = qbound_aux[
                "composer_pair_prior_top1_posterior"
            ].detach().float()
            composer_pair_prior_candidate_count = qbound_aux[
                "composer_pair_prior_candidate_count"
            ].detach().float()
            composer_expected_span_length = qbound_aux[
                "composer_expected_span_length"
            ].detach().float()

            main_pair_entropy = qbound_aux[
                "main_pair_entropy"
            ].detach().float()
            main_pair_effective_count = qbound_aux[
                "main_pair_effective_count"
            ].detach().float()
            main_pair_top1 = qbound_aux[
                "main_pair_top1_posterior"
            ].detach().float()

            coarse_length = qbound_aux[
                "coarse_span_frame_length"
            ].detach().float()
            raw_hard_length = qbound_aux[
                "refined_span_frame_length"
            ].detach().float()
            raw_expected_length = qbound_aux[
                "expected_refined_span_length"
            ].detach().float()
            raw_expected_ratio = qbound_aux[
                "expected_refined_length_ratio"
            ].detach().float()

            calibrated_hard_length = qbound_aux[
                "calibrated_span_frame_length"
            ].detach().float()
            calibrated_expected_length = qbound_aux[
                "calibrated_expected_span_length"
            ].detach().float()
            calibrated_expected_ratio = qbound_aux[
                "calibrated_expected_length_ratio"
            ].detach().float()
            calibrated_retained = qbound_aux[
                "calibrated_retained_fraction"
            ].detach().float()

            between_raw_and_coarse = (
                (calibrated_hard_length >= raw_hard_length)
                & (calibrated_hard_length <= coarse_length)
            ).float()

            raw_point_ratio = (
                raw_hard_length <= 1.0
            ).float().mean()
            calibrated_point_ratio = (
                calibrated_hard_length <= 1.0
            ).float().mean()

            coarse_detached = self._scalar_mean(
                qbound_aux.get("refine_coarse_detached")
            )
            hard_coarse_route = self._scalar_mean(
                qbound_aux.get("refine_hard_coarse_route")
            )
            posterior_guided_enabled = self._scalar_mean(
                qbound_aux.get("posterior_guided_refinement_enabled")
            )
            boundary_structured_enabled = self._scalar_mean(
                qbound_aux.get("boundary_structured_refinement_enabled")
            )
            prior_preserving_bounded_enabled = self._scalar_mean(
                qbound_aux.get(
                    "prior_preserving_bounded_refinement_enabled"
                )
            )
            posterior_stable_enabled = self._scalar_mean(
                qbound_aux.get("posterior_stable_reliability_enabled")
            )
            all_span_enabled = self._scalar_mean(
                qbound_aux.get("all_span_refine_marginal_enabled")
            )
            uncertainty_enabled = self._scalar_mean(
                qbound_aux.get("uncertainty_calibration_enabled")
            )
            minimality_enabled = self._scalar_mean(
                qbound_aux.get("minimal_sufficient_enabled")
            )
            confidence_detached = self._scalar_mean(
                qbound_aux.get("calibration_confidence_detached")
            )
            handover_enabled = self._scalar_mean(
                qbound_aux.get("qbound_handover_enabled")
            )
            direct_residual_enabled = self._scalar_mean(
                qbound_aux.get(
                    "qbound_direct_reasoning_residual_enabled"
                )
            )

        diagnostics = {
            # V1.11 generic-input independence contract.
            "qbound_input_is_generic": input_is_generic.mean().item(),
            "qbound_input_uses_cra_embedding": (
                input_uses_cra_embedding.mean().item()
            ),
            "qbound_generic_input_adapter_enabled": (
                generic_input_adapter_enabled.mean().item()
            ),
            "qbound_generic_input_position_enabled": (
                generic_input_position_enabled.mean().item()
            ),
            "qbound_generic_input_context_reuse": (
                generic_input_context_reuse.mean().item()
            ),
            "qbound_generic_input_raw_rms": (
                generic_input_raw_rms.mean().item()
            ),
            "qbound_generic_input_adapted_rms": (
                generic_input_adapted_rms.mean().item()
            ),
            "qbound_generic_input_adapted_to_raw_rms_ratio": (
                (
                    generic_input_adapted_rms
                    / generic_input_raw_rms.clamp_min(1e-8)
                ).mean().item()
            ),
            "qbound_generic_input_token_cosine": (
                generic_input_token_cosine.mean().item()
            ),
            "qbound_generic_input_delta_ratio": (
                generic_input_delta_ratio.mean().item()
            ),

            # Counterfactual evidence interface diagnostics only in V1.11-A.
            "qbound_counterfactual_evidence_interface_enabled": (
                counterfactual_interface_enabled.mean().item()
            ),
            "qbound_counterfactual_mass_preserving_gradient_enabled": (
                counterfactual_mass_guard.mean().item()
            ),
            "qbound_counterfactual_support_forward_error": (
                counterfactual_forward_error.mean().item()
            ),
            "qbound_counterfactual_core_complement_error": (
                counterfactual_core_complement_error.mean().item()
            ),
            "qbound_counterfactual_donor_available": (
                counterfactual_donor_available.mean().item()
            ),
            "qbound_counterfactual_donor_valid_overlap": (
                counterfactual_donor_overlap.mean().item()
            ),
            "qbound_counterfactual_consumer_frozen": (
                counterfactual_consumer_frozen.mean().item()
            ),
            "qbound_counterfactual_keep_fraction": (
                counterfactual_keep_fraction.mean().item()
            ),
            "qbound_counterfactual_remove_fraction": (
                counterfactual_remove_fraction.mean().item()
            ),
            "qbound_counterfactual_support_complement_error": (
                counterfactual_complement_error.item()
            ),
            "qbound_counterfactual_keep_remove_overlap": (
                counterfactual_overlap.item()
            ),

            # Answer behavior.
            "qbound_main_acc": (
                (calibrated_pred == labels).float().mean().item()
            ),
            "qbound_calibrated_acc": (
                (calibrated_pred == labels).float().mean().item()
            ),
            "qbound_raw_refined_acc": (
                (raw_refined_pred == labels).float().mean().item()
            ),
            "qbound_coarse_acc": (
                (coarse_pred == labels).float().mean().item()
            ),
            "qbound_cra_diagnostic_acc": (
                (cra_pred == labels).float().mean().item()
            ),

            "qbound_main_ce": calibrated_ce.mean().item(),
            "qbound_calibrated_ce": calibrated_ce.mean().item(),
            "qbound_raw_refined_ce": raw_refined_ce.mean().item(),
            "qbound_coarse_ce": coarse_ce.mean().item(),
            "qbound_cra_diagnostic_ce": cra_ce.mean().item(),

            "qbound_calibrated_vs_raw_ce_gain": (
                (raw_refined_ce - calibrated_ce).mean().item()
            ),
            "qbound_calibrated_vs_coarse_ce_gain": (
                (coarse_ce - calibrated_ce).mean().item()
            ),
            "qbound_calibrated_vs_cra_ce_gain": (
                (cra_ce - calibrated_ce).mean().item()
            ),
            "qbound_calibrated_beats_raw_ratio": (
                (calibrated_ce < raw_refined_ce).float().mean().item()
            ),
            "qbound_calibrated_beats_coarse_ratio": (
                (calibrated_ce < coarse_ce).float().mean().item()
            ),
            "qbound_calibrated_beats_cra_ratio": (
                (calibrated_ce < cra_ce).float().mean().item()
            ),

            # Calibration behavior.
            "qbound_calibration_uncertainty": uncertainty.mean().item(),
            "qbound_calibration_information": information.mean().item(),
            "qbound_calibration_confidence": confidence.mean().item(),
            "qbound_calibration_confidence_min": confidence.min().item(),
            "qbound_calibration_confidence_max": confidence.max().item(),
            "qbound_calibration_confidence_std": (
                confidence.std(unbiased=False).item()
            ),

            # V1.10.1 reliability decomposition.
            "qbound_calibration_reliability": reliability.mean().item(),
            "qbound_calibration_reliability_min": reliability.min().item(),
            "qbound_calibration_reliability_max": reliability.max().item(),
            "qbound_calibration_reliability_std": (
                reliability.std(unbiased=False).item()
            ),
            "qbound_refiner_information_gain": information_gain.mean().item(),
            "qbound_refiner_information_gain_std": (
                information_gain.std(unbiased=False).item()
            ),
            "qbound_refined_semantic_retention": semantic_retention.mean().item(),
            "qbound_refined_semantic_retention_std": (
                semantic_retention.std(unbiased=False).item()
            ),
            "qbound_base_refiner_confidence": (
                base_refiner_confidence.mean().item()
            ),
            "qbound_reliability_gain_factor": (
                reliability_gain_factor.mean().item()
            ),
            "qbound_reliability_suppression_from_base": (
                (base_refiner_confidence - reliability).mean().item()
            ),
            "qbound_reliability_below_025_ratio": (
                (reliability < 0.25).float().mean().item()
            ),
            "qbound_reliability_above_050_ratio": (
                (reliability > 0.50).float().mean().item()
            ),
            "qbound_semantic_retention_below_050_ratio": (
                (semantic_retention < 0.50).float().mean().item()
            ),

            # V1.10.2 PPBR trust-region diagnostics.
            "qbound_refine_boundary_anchor_power": (
                boundary_anchor_power.mean().item()
            ),
            "qbound_refine_correction_cap": (
                correction_cap.mean().item()
            ),
            "qbound_refiner_raw_correction_abs_mean": (
                refiner_raw_correction_abs_mean.mean().item()
            ),
            "qbound_refiner_raw_correction_abs_max": (
                refiner_raw_correction_abs_max.mean().item()
            ),
            "qbound_refiner_bounded_correction_abs_mean": (
                refiner_bounded_correction_abs_mean.mean().item()
            ),
            "qbound_refiner_bounded_correction_abs_max": (
                refiner_bounded_correction_abs_max.mean().item()
            ),
            "qbound_refiner_correction_saturation_ratio": (
                refiner_correction_saturation_ratio.mean().item()
            ),
            "qbound_refiner_vs_prior_kl": (
                refiner_vs_prior_kl.mean().item()
            ),
            "qbound_refiner_vs_prior_js": (
                refiner_vs_prior_js.mean().item()
            ),
            "qbound_raw_vs_prior_expected_length_ratio": (
                raw_vs_prior_expected_length_ratio.mean().item()
            ),
            "qbound_refiner_correction_compression_ratio": (
                (
                    refiner_bounded_correction_abs_mean
                    / refiner_raw_correction_abs_mean.clamp_min(1e-8)
                ).mean().item()
            ),
            "qbound_refiner_correction_max_compression_ratio": (
                (
                    refiner_bounded_correction_abs_max
                    / refiner_raw_correction_abs_max.clamp_min(1e-8)
                ).mean().item()
            ),

            # V1.10.1 BAPGR anchored-vs-uniform prior behavior retained.
            "qbound_composer_uniform_pair_prior_entropy": (
                composer_uniform_pair_prior_entropy.mean().item()
            ),
            "qbound_composer_uniform_pair_prior_entropy_ratio": (
                composer_uniform_pair_prior_entropy_ratio.mean().item()
            ),
            "qbound_composer_uniform_pair_prior_effective_count": (
                composer_uniform_pair_prior_effective_count.mean().item()
            ),
            "qbound_composer_uniform_pair_prior_top1_posterior": (
                composer_uniform_pair_prior_top1.mean().item()
            ),
            "qbound_composer_pair_prior_expected_span_length": (
                composer_pair_prior_expected_span_length.mean().item()
            ),
            "qbound_composer_uniform_pair_prior_expected_span_length": (
                composer_uniform_pair_prior_expected_span_length.mean().item()
            ),
            "qbound_composer_pair_prior_length_gain": (
                composer_pair_prior_length_gain.mean().item()
            ),
            "qbound_composer_pair_prior_entropy_ratio_reduction_vs_uniform": (
                (
                    composer_uniform_pair_prior_entropy_ratio
                    - composer_pair_prior_entropy_ratio
                ).mean().item()
            ),
            "qbound_composer_pair_prior_effective_count_reduction_vs_uniform": (
                (
                    composer_uniform_pair_prior_effective_count
                    - composer_pair_prior_effective_count
                ).mean().item()
            ),
            "qbound_composer_pair_prior_top1_gain_vs_uniform": (
                (
                    composer_pair_prior_top1
                    - composer_uniform_pair_prior_top1
                ).mean().item()
            ),

            # Active anchored Composer pair-prior / MAIN pair behavior.
            "qbound_composer_pair_prior_entropy": (
                composer_pair_prior_entropy.mean().item()
            ),
            "qbound_composer_pair_prior_entropy_ratio": (
                composer_pair_prior_entropy_ratio.mean().item()
            ),
            "qbound_composer_pair_prior_effective_count": (
                composer_pair_prior_effective_count.mean().item()
            ),
            "qbound_composer_pair_prior_top1_posterior": (
                composer_pair_prior_top1.mean().item()
            ),
            "qbound_composer_pair_prior_candidate_count": (
                composer_pair_prior_candidate_count.mean().item()
            ),
            "qbound_composer_expected_span_length": (
                composer_expected_span_length.mean().item()
            ),
            "qbound_main_pair_entropy": main_pair_entropy.mean().item(),
            "qbound_main_pair_effective_count": (
                main_pair_effective_count.mean().item()
            ),
            "qbound_main_pair_top1_posterior": main_pair_top1.mean().item(),

            # Span hierarchy.  The old top1-relative ratios remain diagnostics
            # only: V1.9 MAIN is allowed to extend outside the hard top-1 chain.
            "qbound_coarse_span_frame_length_diag": (
                coarse_length.mean().item()
            ),
            "qbound_raw_refined_hard_span_length": (
                raw_hard_length.mean().item()
            ),
            "qbound_raw_refined_expected_span_length": (
                raw_expected_length.mean().item()
            ),
            "qbound_raw_refined_expected_length_ratio": (
                raw_expected_ratio.mean().item()
            ),
            "qbound_calibrated_hard_span_length": (
                calibrated_hard_length.mean().item()
            ),
            "qbound_calibrated_expected_span_length": (
                calibrated_expected_length.mean().item()
            ),
            "qbound_calibrated_expected_length_ratio": (
                calibrated_expected_ratio.mean().item()
            ),
            "qbound_calibrated_retained_fraction": (
                calibrated_retained.mean().item()
            ),
            "qbound_calibrated_vs_composer_expected_length_ratio": (
                (
                    calibrated_expected_length
                    / composer_expected_span_length.clamp_min(1.0)
                ).mean().item()
            ),
            "qbound_calibrated_between_raw_coarse_ratio": (
                between_raw_and_coarse.mean().item()
            ),
            "qbound_raw_refined_point_span_ratio": (
                raw_point_ratio.item()
            ),
            "qbound_calibrated_point_span_ratio": (
                calibrated_point_ratio.item()
            ),

            # Key behavior.
            "qbound_cra_calibrated_js": cra_calibrated_js.item(),
            "qbound_calibrated_raw_js": calibrated_raw_js.item(),
            "qbound_calibrated_coarse_js": calibrated_coarse_js.item(),
            "qbound_raw_coarse_js": raw_coarse_js.item(),
            "qbound_calibrated_raw_l1": calibrated_raw_l1.item(),
            "qbound_calibrated_coarse_l1": calibrated_coarse_l1.item(),
            "qbound_raw_coarse_l1": raw_coarse_l1.item(),
            "qbound_pre_post_smooth_js": pre_post_js.item(),
            "qbound_qbound_final_l1": final_l1.item(),

            "qbound_key_entropy": calibrated_entropy.mean().item(),
            "qbound_calibrated_key_entropy": (
                calibrated_entropy.mean().item()
            ),
            "qbound_raw_refined_key_entropy": (
                raw_refined_entropy.mean().item()
            ),
            "qbound_coarse_key_entropy": coarse_entropy.mean().item(),
            "qbound_cra_key_entropy": cra_entropy.mean().item(),

            "qbound_key_entropy_ratio": (
                (
                    calibrated_entropy
                    / uniform_entropy.clamp_min(1e-6)
                ).mean().item()
            ),
            "qbound_calibrated_key_entropy_ratio": (
                (
                    calibrated_entropy
                    / uniform_entropy.clamp_min(1e-6)
                ).mean().item()
            ),
            "qbound_raw_refined_key_entropy_ratio": (
                (
                    raw_refined_entropy
                    / uniform_entropy.clamp_min(1e-6)
                ).mean().item()
            ),
            "qbound_coarse_key_entropy_ratio": (
                (
                    coarse_entropy
                    / uniform_entropy.clamp_min(1e-6)
                ).mean().item()
            ),

            "qbound_key_max": calibrated_peak.mean().item(),
            "qbound_calibrated_key_max": calibrated_peak.mean().item(),
            "qbound_raw_refined_key_max": raw_refined_peak.mean().item(),
            "qbound_coarse_key_max": coarse_peak.mean().item(),

            # V1.9 routing invariants.
            "qbound_posterior_guided_refinement_enabled": (
                0.0
                if posterior_guided_enabled is None
                else float(posterior_guided_enabled)
            ),
            "qbound_boundary_structured_refinement_enabled": (
                0.0
                if boundary_structured_enabled is None
                else float(boundary_structured_enabled)
            ),
            "qbound_prior_preserving_bounded_refinement_enabled": (
                0.0
                if prior_preserving_bounded_enabled is None
                else float(prior_preserving_bounded_enabled)
            ),
            "qbound_posterior_stable_reliability_enabled": (
                0.0
                if posterior_stable_enabled is None
                else float(posterior_stable_enabled)
            ),
            "qbound_refine_hard_coarse_route": (
                0.0
                if hard_coarse_route is None
                else float(hard_coarse_route)
            ),
            "qbound_refine_coarse_detached": (
                0.0 if coarse_detached is None else float(coarse_detached)
            ),
            "qbound_all_span_refine_marginal_enabled": (
                0.0 if all_span_enabled is None else float(all_span_enabled)
            ),
            "qbound_uncertainty_calibration_enabled": (
                0.0
                if uncertainty_enabled is None
                else float(uncertainty_enabled)
            ),
            "qbound_minimal_sufficient_enabled": (
                0.0 if minimality_enabled is None else float(minimality_enabled)
            ),
            "qbound_calibration_confidence_detached": (
                0.0
                if confidence_detached is None
                else float(confidence_detached)
            ),
            "qbound_handover_enabled": (
                0.0 if handover_enabled is None else float(handover_enabled)
            ),
            "qbound_direct_reasoning_residual_enabled": (
                0.0
                if direct_residual_enabled is None
                else float(direct_residual_enabled)
            ),
        }

        if v111_v15_js is not None:
            diagnostics["qbound_v111_v15_key_js"] = v111_v15_js.item()
        if v111_v15_l1 is not None:
            diagnostics["qbound_v111_v15_key_l1"] = v111_v15_l1.item()

        if v111_v14_js is not None:
            diagnostics["qbound_v111_v14_key_js"] = v111_v14_js.item()
        if v111_v14_l1 is not None:
            diagnostics["qbound_v111_v14_key_l1"] = v111_v14_l1.item()

        return diagnostics

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
            stats.update(extra_logs)

        qbound_aux = self._get_qbound_aux(gqa_output)
        if qbound_aux is None:
            return stats

        # ----------------------------------------------------------
        # V1.11 generic-input / counterfactual interface statistics.
        # ----------------------------------------------------------
        v111_interface_scalar_keys = [
            "generic_input_adapter_enabled",
            "generic_input_position_enabled",
            "generic_input_context_reuse",
            "generic_input_raw_rms",
            "generic_input_adapted_rms",
            "generic_input_token_cosine",
            "generic_input_delta_ratio",
            "counterfactual_evidence_interface_enabled",
            "counterfactual_mass_preserving_gradient_enabled",
            "counterfactual_raw_keep_fraction",
            "counterfactual_keep_fraction",
            "counterfactual_remove_fraction",
            "counterfactual_support_forward_error",
            "counterfactual_support_complement_error",
            "qbound_input_is_generic",
            "qbound_input_uses_cra_embedding",
            "qbound_counterfactual_donor_available",
            "qbound_counterfactual_donor_shift",
            "qbound_counterfactual_donor_valid_overlap",
            "qbound_counterfactual_consumer_frozen",
            "qbound_counterfactual_sufficient_factual_cosine",
            "qbound_counterfactual_necessary_factual_cosine",
            "qbound_counterfactual_sufficient_factual_l2",
            "qbound_counterfactual_necessary_factual_l2",
        ]
        for key in v111_interface_scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                if key.startswith("qbound_"):
                    stats[key] = scalar
                else:
                    stats[f"qbound_{key}"] = scalar

        cf_keep = qbound_aux.get("counterfactual_keep_support")
        cf_remove = qbound_aux.get("counterfactual_remove_support")
        if (
            torch.is_tensor(cf_keep)
            and torch.is_tensor(cf_remove)
            and cf_keep.shape == cf_remove.shape
        ):
            keep = cf_keep.detach().float()
            remove = cf_remove.detach().float()
            support_sum = keep + remove
            support_valid = (support_sum > 0.5).to(support_sum.dtype)
            stats["qbound_counterfactual_support_complement_error"] = (
                (support_sum - support_valid).abs().mean().item()
            )
            stats["qbound_counterfactual_keep_remove_overlap"] = (
                (keep * remove).mean().item()
            )
            stats["qbound_counterfactual_keep_support_mean"] = (
                keep.mean().item()
            )
            stats["qbound_counterfactual_keep_support_max"] = (
                keep.max(dim=-1).values.mean().item()
            )

        # ----------------------------------------------------------
        # V1.11 BAPGR + PPBR reliability / routing statistics retained.
        # ----------------------------------------------------------
        v1102_scalar_keys = [
            "stable_refiner_enabled",
            "uncertainty_calibration_enabled",
            "minimal_sufficient_enabled",
            "all_span_refine_marginal_enabled",
            "refine_hard_coarse_route",
            "refine_coarse_detached",

            "calibration_uncertainty",
            "calibration_information",
            "calibration_confidence",
            "calibration_confidence_detached",

            # V1.11 BAPGR + PPBR + reliability decomposition.
            "posterior_guided_refinement_enabled",
            "boundary_structured_refinement_enabled",
            "prior_preserving_bounded_refinement_enabled",
            "posterior_stable_reliability_enabled",
            "refine_boundary_anchor_power",
            "refine_correction_cap",
            "refine_hard_coarse_route",
            "calibration_reliability",
            "refiner_information_gain",
            "refined_semantic_retention",
            "base_refiner_confidence",
            "reliability_gain_factor",
            "composer_pair_prior_entropy",
            "composer_pair_prior_entropy_ratio",
            "composer_pair_prior_effective_count",
            "composer_pair_prior_top1_posterior",
            "composer_pair_prior_candidate_count",
            "composer_uniform_pair_prior_entropy",
            "composer_uniform_pair_prior_entropy_ratio",
            "composer_uniform_pair_prior_effective_count",
            "composer_uniform_pair_prior_top1_posterior",
            "composer_pair_prior_expected_span_length",
            "composer_uniform_pair_prior_expected_span_length",
            "composer_pair_prior_length_gain",
            "composer_expected_span_length",
            "main_pair_entropy",
            "main_pair_effective_count",
            "main_pair_top1_posterior",

            # V1.10.2 PPBR diagnostics.
            "refiner_raw_correction_abs_mean",
            "refiner_raw_correction_abs_max",
            "refiner_bounded_correction_abs_mean",
            "refiner_bounded_correction_abs_max",
            "refiner_correction_saturation_ratio",
            "refiner_vs_prior_kl",
            "refiner_vs_prior_js",
            "raw_vs_prior_expected_length_ratio",

            "refine_selected_pair_candidate_count",
            "refine_selected_pair_entropy_ratio",

            "expected_refined_span_length",
            "expected_refined_length_ratio",
            "expected_refined_span_fraction",
            "expected_refine_trim_fraction",

            "calibrated_expected_span_length",
            "calibrated_expected_length_ratio",
            "calibrated_span_frame_length",
            "calibrated_span_fraction",
            "calibrated_retained_fraction",
            "calibrated_trim_left_frames",
            "calibrated_trim_right_frames",
            "calibrated_trim_total_frames",
            "calibrated_inside_coarse",

            "cra_qbound_key_l1",
            "coarse_calibrated_key_l1",
            "raw_refined_calibrated_key_l1",
            "coarse_raw_refined_key_l1",
        ]

        for key in v1102_scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                stats[f"qbound_{key}"] = scalar

        v1102_prefixed_keys = [
            "qbound_stable_refiner_enabled",
            "qbound_uncertainty_calibration_enabled",
            "qbound_posterior_guided_refinement_enabled",
            "qbound_boundary_structured_refinement_enabled",
            "qbound_prior_preserving_bounded_refinement_enabled",
            "qbound_posterior_stable_reliability_enabled",
            "qbound_minimal_sufficient_enabled",

            "qbound_calibrated_span_enabled",
            "qbound_calibrated_span_length",
            "qbound_calibrated_span_fraction",
            "qbound_calibrated_span_key_mass",

            "qbound_raw_refined_span_enabled",
            "qbound_raw_refined_span_length",
            "qbound_raw_refined_span_fraction",
            "qbound_raw_refined_span_key_mass",

            "qbound_coarse_span_length",
            "qbound_coarse_span_fraction",
            "qbound_coarse_span_key_mass",
        ]

        for key in v1102_prefixed_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                stats[key] = scalar

        scalar_keys = [
            # Proposal / dynamic topology.
            "boundary_prob_mean",
            "boundary_count",
            "event_count",
            "selected_boundary_prob_mean",
            "assignment_entropy",
            "effective_event_count",
            "soft_effective_event_count",

            # DECOMPOSE retained from V1.4.
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

            # Event structure.
            "event_mass_max",
            "event_mass_min",
            "event_mass_mean",

            # Event evidence retained from V1.4.
            "event_evidence_gate_mean",
            "event_evidence_active_count",
            "event_evidence_entropy",
            "event_evidence_effective_count",
            "top1_event_evidence_share",
            "top2_event_evidence_share",
            "multi_event_evidence",

            # Q-OEI.
            "ordered_relation_gate_mean",
            "ordered_update_gate_mean",
            "ordered_relation_past_mass",
            "ordered_relation_future_mass",
            "reasoned_event_token_delta_norm",

            # Integration invariants.
            "cra_qbound_key_l1",
            "question_only_conditioning",
            "qbound_handover_enabled",
        ]

        for key in scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                stats[f"qbound_{key}"] = scalar


        # ----------------------------------------------------------
        # V1.5 COMPOSE: latent contiguous event-span diagnostics.
        # ----------------------------------------------------------
        composer_scalar_keys = [
            "span_entropy",
            "span_effective_count",
            "span_top1_posterior",
            "span_candidate_count",
            "selected_span_event_count",
            "selected_span_frame_length",
            "selected_span_fraction",
            "selected_span_posterior",
        ]

        for key in composer_scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                stats[f"qbound_{key}"] = scalar

        # Adapter-level scalars are already prefixed inside qbound_aux.
        adapter_scalar_keys = [
            "qbound_adaptive_span_enabled",
            "qbound_adaptive_span_length",
            "qbound_adaptive_span_fraction",
            "qbound_adaptive_span_key_mass",
            "qbound_adaptive_span_peak_in_span",
            "qbound_adaptive_span_mask_consistency",
        ]

        model_outputs = (
            gqa_output[2]
            if (
                gqa_output is not None
                and len(gqa_output) >= 3
                and isinstance(gqa_output[2], dict)
            )
            else {}
        )
        time_param = model_outputs.get("time_param", {})
        if not isinstance(time_param, dict):
            time_param = {}

        for key in adapter_scalar_keys:
            value = time_param.get(
                key,
                qbound_aux.get(key),
            )
            scalar = self._scalar_mean(value)
            if scalar is not None:
                stats[key] = scalar

        span_posterior = qbound_aux.get("span_posterior")
        span_valid_mask = qbound_aux.get("span_candidate_valid_mask")

        if (
            torch.is_tensor(span_posterior)
            and torch.is_tensor(span_valid_mask)
        ):
            posterior = span_posterior.detach().float()
            valid_span = span_valid_mask.detach().bool()

            valid_candidate_count = (
                valid_span.sum(dim=-1).float().clamp_min(1.0)
            )
            stats["qbound_span_valid_candidate_count"] = (
                valid_candidate_count.mean().item()
            )

            # Entropy normalized to [0,1] relative to the number of valid
            # candidate spans. 0 = one decisive span, 1 = uniform posterior.
            span_entropy = -torch.sum(
                posterior
                * torch.log(posterior.clamp_min(1e-8)),
                dim=-1,
            )
            max_entropy = torch.log(valid_candidate_count)
            entropy_ratio = torch.where(
                valid_candidate_count > 1.0,
                span_entropy / max_entropy.clamp_min(1e-6),
                torch.zeros_like(span_entropy),
            )
            stats["qbound_span_entropy_ratio"] = (
                entropy_ratio.mean().item()
            )

            sorted_span_prob = torch.sort(
                posterior,
                dim=-1,
                descending=True,
            ).values
            stats["qbound_span_top1_share"] = (
                sorted_span_prob[:, 0].mean().item()
            )
            if sorted_span_prob.size(-1) >= 2:
                stats["qbound_span_top2_share"] = (
                    sorted_span_prob[:, :2].sum(dim=-1).mean().item()
                )

        selected_span_event_count = qbound_aux.get(
            "selected_span_event_count"
        )
        if torch.is_tensor(selected_span_event_count):
            selected_count = (
                selected_span_event_count.detach()
                .round()
                .long()
            )
            max_events = int(
                self.cfgs.get("model", {})
                .get("qbound", {})
                .get("max_events", 4)
            )
            for n_events in range(1, max_events + 1):
                stats[
                    f"qbound_selected_span_{n_events}_event_ratio"
                ] = (
                    selected_count == n_events
                ).float().mean().item()

        selected_span_frame_length = qbound_aux.get(
            "selected_span_frame_length"
        )
        if torch.is_tensor(selected_span_frame_length):
            lengths = selected_span_frame_length.detach().float()
            stats["qbound_selected_span_frame_length_min"] = (
                lengths.min().item()
            )
            stats["qbound_selected_span_frame_length_max"] = (
                lengths.max().item()
            )

        soft_span_support = qbound_aux.get("soft_span_support")
        if torch.is_tensor(soft_span_support):
            support = soft_span_support.detach().float()
            stats["qbound_soft_span_support_mean"] = (
                support.mean().item()
            )
            stats["qbound_soft_span_support_max"] = (
                support.max(dim=-1).values.mean().item()
            )

        selected_event_mask = qbound_aux.get("selected_event_mask")
        if torch.is_tensor(selected_event_mask):
            stats["qbound_selected_event_mask_count"] = (
                selected_event_mask.detach().float()
                .sum(dim=-1).mean().item()
            )

        # ----------------------------------------------------------
        # V1.11 BAPGR + PPBR REFINE + reliability-calibrated boundary diagnostics.
        # ----------------------------------------------------------
        refiner_scalar_keys = [
            "refine_expected_pair_entropy",
            "refine_selected_pair_entropy",
            "refine_selected_pair_effective_count",
            "refine_selected_pair_top1_posterior",
            "refine_selected_pair_candidate_count",
            "refine_selected_pair_entropy_ratio",
            "refine_selected_start_candidate_count",
            "refine_selected_end_candidate_count",
            "refined_span_frame_length",
            "refined_span_fraction",
            "refine_trim_left_frames",
            "refine_trim_right_frames",
            "refine_trim_total_frames",
            "refine_retained_fraction",
            "refined_inside_coarse",

            # V1.11 reliability-calibrated interval.
            "calibrated_expected_span_length",
            "calibrated_expected_length_ratio",
            "calibrated_span_frame_length",
            "calibrated_span_fraction",
            "calibrated_retained_fraction",
            "calibrated_trim_left_frames",
            "calibrated_trim_right_frames",
            "calibrated_trim_total_frames",
            "calibrated_inside_coarse",

            # V1.10.2 reliability calibration.
            "calibration_uncertainty",
            "calibration_information",
            "calibration_confidence",
            "calibration_reliability",
            "refiner_information_gain",
            "refined_semantic_retention",
            "base_refiner_confidence",
            "reliability_gain_factor",
            "composer_pair_prior_entropy",
            "composer_pair_prior_entropy_ratio",
            "composer_pair_prior_effective_count",
            "composer_pair_prior_top1_posterior",
            "composer_pair_prior_candidate_count",
            "composer_uniform_pair_prior_entropy",
            "composer_uniform_pair_prior_entropy_ratio",
            "composer_uniform_pair_prior_effective_count",
            "composer_uniform_pair_prior_top1_posterior",
            "composer_pair_prior_expected_span_length",
            "composer_uniform_pair_prior_expected_span_length",
            "composer_pair_prior_length_gain",
            "composer_expected_span_length",
            "main_pair_entropy",
            "main_pair_effective_count",
            "main_pair_top1_posterior",

            # V1.10.2 PPBR.
            "refine_correction_cap",
            "refiner_raw_correction_abs_mean",
            "refiner_raw_correction_abs_max",
            "refiner_bounded_correction_abs_mean",
            "refiner_bounded_correction_abs_max",
            "refiner_correction_saturation_ratio",
            "refiner_vs_prior_kl",
            "refiner_vs_prior_js",
            "raw_vs_prior_expected_length_ratio",

            "refiner_temperature",
        ]

        for key in refiner_scalar_keys:
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                stats[f"qbound_{key}"] = scalar

        # Adapter-level calibrated/raw/coarse diagnostics.
        refiner_adapter_keys = [
            # Compatibility refined == calibrated MAIN.
            "qbound_refined_span_enabled",
            "qbound_refined_span_length",
            "qbound_refined_span_fraction",
            "qbound_refined_span_key_mass",

            # Explicit calibrated branch.
            "qbound_calibrated_span_enabled",
            "qbound_calibrated_span_length",
            "qbound_calibrated_span_fraction",
            "qbound_calibrated_span_key_mass",

            # Explicit raw-refined branch.
            "qbound_raw_refined_span_enabled",
            "qbound_raw_refined_span_length",
            "qbound_raw_refined_span_fraction",
            "qbound_raw_refined_span_key_mass",

            # Explicit coarse branch.
            "qbound_coarse_span_length",
            "qbound_coarse_span_fraction",
            "qbound_coarse_span_key_mass",

            # Same calibrated key + alternate hard readouts.
            "qbound_coarse_v15_span_length",
            "qbound_coarse_v15_span_fraction",
            "qbound_coarse_v15_span_key_mass",
            "qbound_legacy_fixed_window_span_length",
            "qbound_legacy_fixed_window_key_mass",
            "qbound_refined_vs_coarse_span_length_ratio",
            "qbound_refined_vs_legacy_span_length_ratio",
            "qbound_refined_minus_coarse_key_mass",
            "qbound_refined_minus_legacy_key_mass",
            "qbound_refined_trimmed_frame_count",
        ]

        for key in refiner_adapter_keys:
            value = time_param.get(
                key,
                qbound_aux.get(key),
            )
            scalar = self._scalar_mean(value)
            if scalar is not None:
                stats[key] = scalar

        refined_length = qbound_aux.get("refined_span_frame_length")
        if torch.is_tensor(refined_length):
            values = refined_length.detach().float()
            stats["qbound_refined_span_frame_length_min"] = (
                values.min().item()
            )
            stats["qbound_refined_span_frame_length_max"] = (
                values.max().item()
            )

        calibrated_length = qbound_aux.get(
            "calibrated_span_frame_length"
        )
        if torch.is_tensor(calibrated_length):
            values = calibrated_length.detach().float()
            stats["qbound_calibrated_span_frame_length_min"] = (
                values.min().item()
            )
            stats["qbound_calibrated_span_frame_length_max"] = (
                values.max().item()
            )
            stats["qbound_calibrated_point_span_ratio"] = (
                (values <= 1.0).float().mean().item()
            )

        if torch.is_tensor(refined_length):
            values = refined_length.detach().float()
            stats["qbound_raw_refined_point_span_ratio"] = (
                (values <= 1.0).float().mean().item()
            )

        trim_total = qbound_aux.get("refine_trim_total_frames")
        if torch.is_tensor(trim_total):
            values = trim_total.detach().float()
            stats["qbound_refine_trim_total_frames_min"] = (
                values.min().item()
            )
            stats["qbound_refine_trim_total_frames_max"] = (
                values.max().item()
            )

        pair_posterior = qbound_aux.get(
            "refine_selected_pair_posterior"
        )
        if torch.is_tensor(pair_posterior):
            posterior = pair_posterior.detach().float().reshape(
                pair_posterior.size(0),
                -1,
            )
            sorted_pair_prob = torch.sort(
                posterior,
                dim=-1,
                descending=True,
            ).values
            stats["qbound_refine_pair_top1_share"] = (
                sorted_pair_prob[:, 0].mean().item()
            )
            if sorted_pair_prob.size(-1) >= 2:
                stats["qbound_refine_pair_top2_share"] = (
                    sorted_pair_prob[:, :2]
                    .sum(dim=-1)
                    .mean().item()
                )

        # ----------------------------------------------------------
        # Boundary proposal health.
        # ----------------------------------------------------------
        boundary_prob = qbound_aux.get("boundary_prob")
        boundary_mask = qbound_aux.get("boundary_mask")

        if (
            torch.is_tensor(boundary_prob)
            and torch.is_tensor(boundary_mask)
        ):
            valid_boundary = boundary_prob[boundary_mask.bool()]
            if valid_boundary.numel() > 0:
                valid_boundary = valid_boundary.detach().float()
                stats["qbound_boundary_prob_min"] = (
                    valid_boundary.min().item()
                )
                stats["qbound_boundary_prob_max"] = (
                    valid_boundary.max().item()
                )
                stats["qbound_boundary_prob_std"] = (
                    valid_boundary.std(unbiased=False).item()
                )

        candidate_valid = qbound_aux.get(
            "boundary_candidate_valid_mask"
        )
        if torch.is_tensor(candidate_valid):
            stats["qbound_boundary_candidate_count"] = (
                candidate_valid.detach().float()
                .sum(dim=-1).mean().item()
            )

        candidate_probs = qbound_aux.get(
            "boundary_candidate_probs"
        )
        if (
            torch.is_tensor(candidate_probs)
            and torch.is_tensor(candidate_valid)
        ):
            valid_candidate_probs = candidate_probs[
                candidate_valid.bool()
            ]
            if valid_candidate_probs.numel() > 0:
                values = valid_candidate_probs.detach().float()
                stats["qbound_candidate_prob_mean"] = (
                    values.mean().item()
                )
                stats["qbound_candidate_prob_max"] = (
                    values.max().item()
                )

        boundary_soft = qbound_aux.get(
            "boundary_soft_marginal"
        )
        if torch.is_tensor(boundary_soft):
            values = boundary_soft.detach().float()
            stats["qbound_boundary_soft_marginal_mean"] = (
                values.mean().item()
            )
            stats["qbound_boundary_soft_marginal_max"] = (
                values.max().item()
            )

        # ----------------------------------------------------------
        # Dynamic event-count histogram.
        # ----------------------------------------------------------
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

        # ----------------------------------------------------------
        # Event mass / giant-event-collapse diagnostics.
        # ----------------------------------------------------------
        event_mass = qbound_aux.get("event_mass")
        event_valid_mask = qbound_aux.get("event_valid_mask")

        if torch.is_tensor(event_mass):
            mass = event_mass.detach().float()

            if torch.is_tensor(event_valid_mask):
                valid_event_mask = event_valid_mask.bool()
            else:
                valid_event_mask = mass > 0

            if valid_event_mask.any():
                valid_mass = mass[valid_event_mask]
                stats["qbound_event_mass_valid_mean"] = (
                    valid_mass.mean().item()
                )
                stats["qbound_event_mass_valid_min"] = (
                    valid_mass.min().item()
                )
                stats["qbound_event_mass_valid_max"] = (
                    valid_mass.max().item()
                )

            max_mass_per_sample = (
                mass.masked_fill(~valid_event_mask, -1e4)
                .max(dim=-1).values
            )

            if torch.is_tensor(boundary_mask):
                valid_length = (
                    boundary_mask.detach().bool()
                    .sum(dim=-1).float() + 1.0
                )
            else:
                valid_length = mass.sum(dim=-1).clamp_min(1.0)

            stats["qbound_dominant_event_fraction"] = (
                max_mass_per_sample
                / valid_length.clamp_min(1.0)
            ).mean().item()

        # ----------------------------------------------------------
        # Partition confidence / energy gap.
        # ----------------------------------------------------------
        partition_posterior = qbound_aux.get("partition_posterior")
        partition_valid_mask = qbound_aux.get("partition_valid_mask")
        partition_energy = qbound_aux.get("partition_energy")

        if torch.is_tensor(partition_posterior):
            posterior = partition_posterior.detach().float()
            stats["qbound_partition_max_probability"] = (
                posterior.max(dim=-1).values.mean().item()
            )

        if (
            torch.is_tensor(partition_energy)
            and torch.is_tensor(partition_valid_mask)
        ):
            energy = partition_energy.detach().float().masked_fill(
                ~partition_valid_mask.bool(),
                float("inf"),
            )
            if energy.size(-1) >= 2:
                two_best = torch.topk(
                    energy,
                    k=2,
                    dim=-1,
                    largest=False,
                ).values
                finite = torch.isfinite(two_best).all(dim=-1)
                if finite.any():
                    stats["qbound_partition_energy_gap"] = (
                        (two_best[finite, 1] - two_best[finite, 0])
                        .mean().item()
                    )

        # ----------------------------------------------------------
        # Split-gain distribution.
        # ----------------------------------------------------------
        candidate_split_gain = qbound_aux.get(
            "candidate_split_gain"
        )
        if (
            torch.is_tensor(candidate_split_gain)
            and torch.is_tensor(candidate_valid)
        ):
            values = candidate_split_gain[
                candidate_valid.bool()
            ]
            if values.numel() > 0:
                values = values.detach().float()
                stats["qbound_candidate_split_gain_mean"] = (
                    values.mean().item()
                )
                stats["qbound_candidate_split_gain_positive_ratio"] = (
                    (values > 0).float().mean().item()
                )

        # ----------------------------------------------------------
        # Event-evidence health. Main V1.4 grounding does not use a
        # winner-takes-all event selection.
        # ----------------------------------------------------------
        event_evidence_logits = qbound_aux.get(
            "event_evidence_logits"
        )
        event_evidence_weights = qbound_aux.get(
            "event_evidence_weights"
        )

        if (
            torch.is_tensor(event_evidence_logits)
            and torch.is_tensor(event_valid_mask)
        ):
            valid_logits = event_evidence_logits[
                event_valid_mask.bool()
            ]
            if valid_logits.numel() > 0:
                values = valid_logits.detach().float()
                stats["qbound_event_evidence_logit_mean"] = (
                    values.mean().item()
                )
                stats["qbound_event_evidence_logit_std"] = (
                    values.std(unbiased=False).item()
                )
                stats["qbound_event_evidence_logit_abs_max"] = (
                    values.abs().max().item()
                )

        if (
            torch.is_tensor(event_evidence_weights)
            and torch.is_tensor(event_valid_mask)
        ):
            valid_weights = event_evidence_weights[
                event_valid_mask.bool()
            ]
            if valid_weights.numel() > 0:
                stats["qbound_event_evidence_weight_valid_mean"] = (
                    valid_weights.detach().float().mean().item()
                )

        # ----------------------------------------------------------
        # Q-OEI relation / update diagnostics.
        # ----------------------------------------------------------
        ordered_relation_gates = qbound_aux.get(
            "ordered_relation_gates"
        )
        ordered_relation_mask = qbound_aux.get(
            "ordered_relation_valid_mask"
        )

        if (
            torch.is_tensor(ordered_relation_gates)
            and torch.is_tensor(ordered_relation_mask)
        ):
            valid_relation_values = ordered_relation_gates[
                ordered_relation_mask.bool()
            ]
            if valid_relation_values.numel() > 0:
                values = valid_relation_values.detach().float()
                stats["qbound_ordered_relation_gate_valid_mean"] = (
                    values.mean().item()
                )
                stats["qbound_ordered_relation_gate_valid_std"] = (
                    values.std(unbiased=False).item()
                )
                stats["qbound_ordered_relation_gate_valid_max"] = (
                    values.max().item()
                )
                stats["qbound_ordered_relation_active_ratio"] = (
                    (values >= 0.5).float().mean().item()
                )

        ordered_update_gate = qbound_aux.get(
            "ordered_update_gate"
        )
        if (
            torch.is_tensor(ordered_update_gate)
            and torch.is_tensor(event_valid_mask)
        ):
            valid_update_values = ordered_update_gate[
                event_valid_mask.bool()
            ]
            if valid_update_values.numel() > 0:
                values = valid_update_values.detach().float()
                stats["qbound_ordered_update_gate_valid_mean"] = (
                    values.mean().item()
                )
                stats["qbound_ordered_update_gate_valid_max"] = (
                    values.max().item()
                )

        relation_past_mass = qbound_aux.get(
            "ordered_relation_past_mass"
        )
        relation_future_mass = qbound_aux.get(
            "ordered_relation_future_mass"
        )
        if (
            torch.is_tensor(relation_past_mass)
            and torch.is_tensor(relation_future_mass)
        ):
            stats["qbound_order_direction_asymmetry"] = (
                (
                    relation_past_mass.detach().float()
                    - relation_future_mass.detach().float()
                ).abs().mean().item()
            )

        original_event_tokens = qbound_aux.get("event_tokens")
        reasoned_event_tokens = qbound_aux.get(
            "reasoned_event_tokens"
        )
        if (
            torch.is_tensor(original_event_tokens)
            and torch.is_tensor(reasoned_event_tokens)
            and original_event_tokens.shape == reasoned_event_tokens.shape
        ):
            token_delta = (
                reasoned_event_tokens.detach().float()
                - original_event_tokens.detach().float()
            ).norm(dim=-1)

            if torch.is_tensor(event_valid_mask):
                mask = event_valid_mask.bool()
                if mask.any():
                    stats["qbound_reasoned_event_token_delta_norm"] = (
                        token_delta[mask].mean().item()
                    )
            else:
                stats["qbound_reasoned_event_token_delta_norm"] = (
                    token_delta.mean().item()
                )

        # ----------------------------------------------------------
        # Frame-evidence scale and calibrated-MAIN smoothing.
        # ----------------------------------------------------------
        frame_semantic_logits = qbound_aux.get(
            "frame_semantic_logits"
        )
        if torch.is_tensor(frame_semantic_logits):
            values = frame_semantic_logits.detach().float()
            stats["qbound_frame_semantic_logit_abs_mean"] = (
                values.abs().mean().item()
            )
            stats["qbound_frame_semantic_logit_std"] = (
                values.std(unbiased=False).item()
            )

        pre_smooth_key = qbound_aux.get(
            "qbound_pre_smooth_key"
        )
        post_smooth_key = qbound_aux.get(
            "qbound_post_smooth_key"
        )
        if (
            torch.is_tensor(pre_smooth_key)
            and torch.is_tensor(post_smooth_key)
        ):
            stats["qbound_pre_post_smooth_l1"] = (
                (pre_smooth_key.detach() - post_smooth_key.detach())
                .abs().mean().item()
            )
            pre_peak = pre_smooth_key.detach().argmax(dim=-1)
            post_peak = post_smooth_key.detach().argmax(dim=-1)
            stats["qbound_smoothing_peak_change_ratio"] = (
                pre_peak != post_peak
            ).float().mean().item()

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

    def _qbound_component_grad_norms(self):
        """
        Diagnostic gradient norms for V1.11.

        ``composer`` checks span_composer_score. It should receive the MAIN
        calibrated VQA signal through the differentiable coarse component.

        ``refiner`` checks boundary_refine_encoder/start/end. It should receive
        the same MAIN calibrated VQA signal through the posterior-guided fine component.

        There is no separate coarse task loss and no minimal-sufficient loss.
        """
        if not self._qbound_enabled():
            return {
                "qbound_input_adapter_grad_norm": 0.0,
                "qbound_composer_grad_norm": 0.0,
                "qbound_refiner_grad_norm": 0.0,
            }

        input_adapter_sq = 0.0
        composer_sq = 0.0
        refiner_sq = 0.0

        refiner_tokens = (
            "boundary_refine_encoder",
            "boundary_start_score",
            "boundary_end_score",
        )

        for name, parameter in self.model.named_parameters():
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach().float()
            grad_sq = grad.pow(2).sum().item()

            if (
                "generic_input_proj" in name
                or "generic_input_norm" in name
            ):
                input_adapter_sq += grad_sq

            if "span_composer_score" in name:
                composer_sq += grad_sq

            if any(token in name for token in refiner_tokens):
                refiner_sq += grad_sq

        return {
            "qbound_input_adapter_grad_norm": float(
                input_adapter_sq ** 0.5
            ),
            "qbound_composer_grad_norm": float(composer_sq ** 0.5),
            "qbound_refiner_grad_norm": float(refiner_sq ** 0.5),
        }


    @staticmethod
    def _qbound_monitor_keys():
        return [
                "qbound_grad_norm",
                "qbound_input_adapter_grad_norm",
                "qbound_input_is_generic",
                "qbound_input_uses_cra_embedding",
                "qbound_generic_input_adapter_enabled",
                "qbound_generic_input_position_enabled",
                "qbound_generic_input_context_reuse",
                "qbound_generic_input_raw_rms",
                "qbound_generic_input_adapted_rms",
                "qbound_generic_input_adapted_to_raw_rms_ratio",
                "qbound_generic_input_token_cosine",
                "qbound_generic_input_delta_ratio",
                "qbound_counterfactual_evidence_interface_enabled",
                "qbound_counterfactual_mass_preserving_gradient_enabled",
                "qbound_counterfactual_training_enabled",
                "qbound_counterfactual_loss_weight",
                "qbound_counterfactual_sufficiency_weight",
                "qbound_counterfactual_necessity_weight",
                "qbound_counterfactual_necessity_margin",
                "qbound_counterfactual_necessity_correct_only",
                "qbound_counterfactual_donor_available",
                "qbound_counterfactual_donor_valid_overlap",
                "qbound_counterfactual_consumer_frozen",
                "qbound_counterfactual_sufficiency_loss",
                "qbound_counterfactual_necessity_loss",
                "qbound_counterfactual_total_loss",
                "qbound_counterfactual_factual_correct_logit",
                "qbound_counterfactual_sufficient_correct_logit",
                "qbound_counterfactual_necessary_correct_logit",
                "qbound_counterfactual_factual_minus_sufficient_logit",
                "qbound_counterfactual_necessity_gap",
                "qbound_counterfactual_factual_correct_prob",
                "qbound_counterfactual_sufficient_correct_prob",
                "qbound_counterfactual_necessary_correct_prob",
                "qbound_counterfactual_sufficient_acc",
                "qbound_counterfactual_necessary_acc",
                "qbound_counterfactual_necessity_active_ratio",
                "qbound_counterfactual_necessity_margin_satisfied_ratio",
                "qbound_counterfactual_support_forward_error",
                "qbound_counterfactual_core_complement_error",
                "qbound_counterfactual_keep_fraction",
                "qbound_counterfactual_remove_fraction",
                "qbound_counterfactual_support_complement_error",
                "qbound_counterfactual_keep_remove_overlap",
                "qbound_counterfactual_keep_support_mean",
                "qbound_counterfactual_keep_support_max",
                "qbound_weak_align_weight",
                "qbound_decomp_weight",
                "qbound_weak_align_retrieval_acc",
                "qbound_direct_reasoning_residual_enabled",
                "qbound_handover_enabled",
                "qbound_main_acc",
                "qbound_cra_diagnostic_acc",
                "qbound_main_ce",
                "qbound_cra_diagnostic_ce",
                "qbound_main_vs_cra_acc_gain",
                "qbound_main_vs_cra_ce_gain",
                "qbound_main_beats_cra_ratio",
                "qbound_cra_qbound_js",
                "qbound_pre_post_smooth_js",
                "qbound_qbound_final_l1",
                "qbound_cra_key_entropy",
                "qbound_key_entropy",
                "qbound_pre_smooth_entropy",
                "qbound_post_smooth_entropy",
                "qbound_cra_key_entropy_ratio",
                "qbound_key_entropy_ratio",
                "qbound_cra_key_max",
                "qbound_key_max",
                "qbound_cra_peak_ratio",
                "qbound_peak_ratio",
                "qbound_cra_qbound_peak_shift",
                "qbound_pre_post_smooth_l1",
                "qbound_smoothing_peak_change_ratio",
                "qbound_reasoning_residual_norm",
                "qbound_reasoning_residual_used_norm",
                "qbound_boundary_prob_mean",
                "qbound_boundary_prob_min",
                "qbound_boundary_prob_max",
                "qbound_boundary_prob_std",
                "qbound_boundary_count",
                "qbound_boundary_candidate_count",
                "qbound_candidate_prob_mean",
                "qbound_candidate_prob_max",
                "qbound_boundary_soft_marginal_mean",
                "qbound_boundary_soft_marginal_max",
                "qbound_selected_boundary_prob_mean",
                "qbound_event_count",
                "qbound_event_count_1_ratio",
                "qbound_event_count_2_ratio",
                "qbound_event_count_3_ratio",
                "qbound_event_count_4_ratio",
                "qbound_decomp_loss",
                "qbound_decomp_expected_energy",
                "qbound_decomp_selected_coherence",
                "qbound_decomp_selected_complexity",
                "qbound_decomp_selected_complexity_cost",
                "qbound_decomp_selected_boundary_reward",
                "qbound_decomp_selected_boundary_term",
                "qbound_decomp_selected_total_energy",
                "qbound_decomp_expected_event_count",
                "qbound_partition_entropy",
                "qbound_partition_max_probability",
                "qbound_partition_energy_gap",
                "qbound_accepted_split_gain",
                "qbound_rejected_split_gain",
                "qbound_candidate_split_gain_mean",
                "qbound_candidate_split_gain_positive_ratio",
                "qbound_assignment_entropy",
                "qbound_effective_event_count",
                "qbound_soft_effective_event_count",
                "qbound_event_mass_max",
                "qbound_event_mass_min",
                "qbound_event_mass_mean",
                "qbound_event_mass_valid_mean",
                "qbound_event_mass_valid_min",
                "qbound_event_mass_valid_max",
                "qbound_dominant_event_fraction",
                "qbound_event_evidence_gate_mean",
                "qbound_event_evidence_active_count",
                "qbound_event_evidence_entropy",
                "qbound_event_evidence_effective_count",
                "qbound_top1_event_evidence_share",
                "qbound_top2_event_evidence_share",
                "qbound_multi_event_evidence",
                "qbound_event_evidence_logit_mean",
                "qbound_event_evidence_logit_std",
                "qbound_event_evidence_logit_abs_max",
                "qbound_event_evidence_weight_valid_mean",
                "qbound_ordered_relation_gate_mean",
                "qbound_ordered_update_gate_mean",
                "qbound_ordered_relation_past_mass",
                "qbound_ordered_relation_future_mass",
                "qbound_ordered_relation_gate_valid_mean",
                "qbound_ordered_relation_gate_valid_std",
                "qbound_ordered_relation_gate_valid_max",
                "qbound_ordered_relation_active_ratio",
                "qbound_ordered_update_gate_valid_mean",
                "qbound_ordered_update_gate_valid_max",
                "qbound_order_direction_asymmetry",
                "qbound_reasoned_event_token_delta_norm",
                "qbound_adaptive_span_enabled",
                "qbound_span_entropy",
                "qbound_span_entropy_ratio",
                "qbound_span_effective_count",
                "qbound_span_top1_posterior",
                "qbound_span_top1_share",
                "qbound_span_top2_share",
                "qbound_span_valid_candidate_count",
                "qbound_span_candidate_count",
                "qbound_selected_span_event_count",
                "qbound_selected_span_frame_length",
                "qbound_selected_span_frame_length_min",
                "qbound_selected_span_frame_length_max",
                "qbound_selected_span_fraction",
                "qbound_selected_span_posterior",
                "qbound_selected_event_mask_count",
                "qbound_selected_span_1_event_ratio",
                "qbound_selected_span_2_event_ratio",
                "qbound_selected_span_3_event_ratio",
                "qbound_selected_span_4_event_ratio",
                "qbound_soft_span_support_mean",
                "qbound_soft_span_support_max",
                "qbound_adaptive_span_length",
                "qbound_adaptive_span_fraction",
                "qbound_adaptive_span_key_mass",
                "qbound_legacy_fixed_window_span_length",
                "qbound_frame_semantic_logit_abs_mean",
                "qbound_frame_semantic_logit_std",
                "qbound_cra_qbound_key_l1",
                "qbound_question_only_conditioning",
                "qbound_refined_span_enabled",
                "qbound_refine_expected_pair_entropy",
                "qbound_refine_selected_pair_entropy",
                "qbound_refine_selected_pair_effective_count",
                "qbound_refine_selected_pair_top1_posterior",
                "qbound_refine_pair_top1_share",
                "qbound_refine_pair_top2_share",
                "qbound_refine_selected_start_candidate_count",
                "qbound_refine_selected_end_candidate_count",
                "qbound_refined_span_frame_length",
                "qbound_refined_span_frame_length_min",
                "qbound_refined_span_frame_length_max",
                "qbound_refined_span_fraction",
                "qbound_refine_trim_left_frames",
                "qbound_refine_trim_right_frames",
                "qbound_refine_trim_total_frames",
                "qbound_refine_trim_total_frames_min",
                "qbound_refine_trim_total_frames_max",
                "qbound_refine_retained_fraction",
                "qbound_refined_inside_coarse",
                "qbound_refiner_temperature",
                "qbound_refined_span_length",
                "qbound_refined_span_key_mass",
                "qbound_coarse_v15_span_length",
                "qbound_coarse_v15_span_fraction",
                "qbound_coarse_v15_span_key_mass",
                "qbound_refined_vs_coarse_span_length_ratio",
                "qbound_refined_vs_legacy_span_length_ratio",
                "qbound_refined_minus_coarse_key_mass",
                "qbound_refined_minus_legacy_key_mass",
                "qbound_refined_trimmed_frame_count",
                "qbound_composer_grad_norm",
                "qbound_refiner_grad_norm",
                "qbound_refined_acc",
                "qbound_coarse_acc",
                "qbound_refined_ce",
                "qbound_coarse_ce",
                "qbound_refined_minus_coarse_ce",
                "qbound_refined_vs_coarse_ce_gain",
                "qbound_refined_beats_coarse_ratio",
                "qbound_expected_refined_length_ratio",
                "qbound_expected_refined_span_length",
                "qbound_cra_refined_js",
                "qbound_coarse_refined_js",
                "qbound_coarse_refined_l1",
                "qbound_refined_key_entropy",
                "qbound_coarse_key_entropy",
                "qbound_refined_key_entropy_ratio",
                "qbound_coarse_key_entropy_ratio",
                "qbound_refined_key_max",
                "qbound_coarse_key_max",
                "qbound_refine_coarse_detached",
                "qbound_all_span_refine_marginal_enabled",
                "qbound_stable_refiner_enabled",
                "qbound_refine_hard_coarse_route",
                "qbound_refine_selected_pair_candidate_count",
                "qbound_refine_selected_pair_entropy_ratio",
                "qbound_expected_refined_span_fraction",
                "qbound_expected_refine_trim_fraction",
                "qbound_coarse_span_length",
                "qbound_coarse_span_fraction",
                "qbound_coarse_span_key_mass",
                "qbound_calibrated_acc",
                "qbound_raw_refined_acc",
                "qbound_calibrated_ce",
                "qbound_raw_refined_ce",
                "qbound_calibrated_vs_raw_ce_gain",
                "qbound_calibrated_vs_coarse_ce_gain",
                "qbound_calibrated_vs_cra_ce_gain",
                "qbound_calibrated_beats_raw_ratio",
                "qbound_calibrated_beats_coarse_ratio",
                "qbound_calibrated_beats_cra_ratio",
                "qbound_calibration_uncertainty",
                "qbound_calibration_information",
                "qbound_calibration_confidence",
                "qbound_calibration_confidence_min",
                "qbound_calibration_confidence_max",
                "qbound_calibration_confidence_std",
                "qbound_calibration_confidence_detached",
                "qbound_calibration_reliability",
                "qbound_calibration_reliability_min",
                "qbound_calibration_reliability_max",
                "qbound_calibration_reliability_std",
                "qbound_refiner_information_gain",
                "qbound_refiner_information_gain_std",
                "qbound_refined_semantic_retention",
                "qbound_refined_semantic_retention_std",
                "qbound_base_refiner_confidence",
                "qbound_reliability_gain_factor",
                "qbound_reliability_suppression_from_base",
                "qbound_reliability_below_025_ratio",
                "qbound_reliability_above_050_ratio",
                "qbound_semantic_retention_below_050_ratio",
                "qbound_posterior_guided_refinement_enabled",
                "qbound_boundary_structured_refinement_enabled",
                "qbound_prior_preserving_bounded_refinement_enabled",
                "qbound_posterior_stable_reliability_enabled",
                "qbound_refine_boundary_anchor_power",
                "qbound_refine_correction_cap",
                "qbound_refiner_raw_correction_abs_mean",
                "qbound_refiner_raw_correction_abs_max",
                "qbound_refiner_bounded_correction_abs_mean",
                "qbound_refiner_bounded_correction_abs_max",
                "qbound_refiner_correction_saturation_ratio",
                "qbound_refiner_vs_prior_kl",
                "qbound_refiner_vs_prior_js",
                "qbound_raw_vs_prior_expected_length_ratio",
                "qbound_refiner_correction_compression_ratio",
                "qbound_refiner_correction_max_compression_ratio",
                "qbound_composer_uniform_pair_prior_entropy",
                "qbound_composer_uniform_pair_prior_entropy_ratio",
                "qbound_composer_uniform_pair_prior_effective_count",
                "qbound_composer_uniform_pair_prior_top1_posterior",
                "qbound_composer_pair_prior_expected_span_length",
                "qbound_composer_uniform_pair_prior_expected_span_length",
                "qbound_composer_pair_prior_length_gain",
                "qbound_composer_pair_prior_entropy_ratio_reduction_vs_uniform",
                "qbound_composer_pair_prior_effective_count_reduction_vs_uniform",
                "qbound_composer_pair_prior_top1_gain_vs_uniform",
                "qbound_composer_pair_prior_entropy",
                "qbound_composer_pair_prior_entropy_ratio",
                "qbound_composer_pair_prior_effective_count",
                "qbound_composer_pair_prior_top1_posterior",
                "qbound_composer_pair_prior_candidate_count",
                "qbound_composer_expected_span_length",
                "qbound_main_pair_entropy",
                "qbound_main_pair_effective_count",
                "qbound_main_pair_top1_posterior",
                "qbound_calibrated_vs_composer_expected_length_ratio",
                "qbound_uncertainty_calibration_enabled",
                "qbound_minimal_sufficient_enabled",
                "qbound_coarse_span_frame_length_diag",
                "qbound_raw_refined_hard_span_length",
                "qbound_raw_refined_expected_span_length",
                "qbound_raw_refined_expected_length_ratio",
                "qbound_calibrated_hard_span_length",
                "qbound_calibrated_expected_span_length",
                "qbound_calibrated_expected_length_ratio",
                "qbound_calibrated_span_frame_length",
                "qbound_calibrated_span_frame_length_min",
                "qbound_calibrated_span_frame_length_max",
                "qbound_calibrated_span_fraction",
                "qbound_calibrated_retained_fraction",
                "qbound_calibrated_between_raw_coarse_ratio",
                "qbound_raw_refined_point_span_ratio",
                "qbound_calibrated_point_span_ratio",
                "qbound_calibrated_trim_left_frames",
                "qbound_calibrated_trim_right_frames",
                "qbound_calibrated_trim_total_frames",
                "qbound_calibrated_inside_coarse",
                "qbound_cra_calibrated_js",
                "qbound_calibrated_raw_js",
                "qbound_calibrated_coarse_js",
                "qbound_raw_coarse_js",
                "qbound_calibrated_raw_l1",
                "qbound_calibrated_coarse_l1",
                "qbound_raw_coarse_l1",
                "qbound_calibrated_key_entropy",
                "qbound_raw_refined_key_entropy",
                "qbound_calibrated_key_entropy_ratio",
                "qbound_raw_refined_key_entropy_ratio",
                "qbound_calibrated_key_max",
                "qbound_raw_refined_key_max",
                "qbound_v111_v15_key_js",
                "qbound_v111_v15_key_l1",
                "qbound_v111_v14_key_js",
                "qbound_v111_v14_key_l1",
        ]

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
    def get_loss(
        self,
        batch,
        qbound_mix=1.0,
    ):
        """
        Controlled V1.11-B objective:

            L =
                L_CRA/main_reliability_calibrated_task
                + lambda_align * L_weak_align
                + lambda_decomp * L_decomp
                + lambda_suf * L_sufficiency
                + lambda_nec * L_necessity

        The V1.11-A factual mechanism is unchanged.  V1.11-B only adds
        counterfactual evidence supervision through the mass-preserving MAIN
        support.  The CF consumer and answer candidates are frozen/detached so
        the new gradients target Q-Bound temporal evidence.

        No separate coarse VQA loss, reliability loss, minimal-sufficient loss,
        temporal GT, IoU/IoP loss, entropy loss, or target duration is used.
        """
        del qbound_mix

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
            target_dict["vq_align_gt"] = (
                vq_output[2]["align_gt"]
            )

        # ----------------------------------------------------------
        # MAIN VQA: V1.11 BAPGR + PPBR + posterior-stable reliability-calibrated branch.
        # ----------------------------------------------------------
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
        )

        main_video_proj = gqa_output[0]
        answer_proj = gqa_output[1]

        calibrated_logits = self._score_answer_candidates(
            answer_proj=answer_proj,
            representation=main_video_proj,
            detach_answer=False,
        )

        prediction_dict["vqa_pred"] = calibrated_logits
        target_dict["vqa_target"] = answer_id_device

        prediction_dict["vqa_align_pred"] = (
            gqa_output[2]["time_param"]["grounding_feature"]
        )
        target_dict["vqa_align_gt"] = (
            gqa_output[2]["align_gt"]
        )

        # Preserve the repository CRA task objective for the official MAIN path.
        loss = self.criterion(
            prediction_dict,
            target_dict,
        )

        weak_align_retrieval_acc = None
        align_weight = 0.0
        decomp_weight = 0.0

        cf_settings = {
            "interface_enabled": False,
            "training_enabled": False,
            "sufficiency_weight": 0.0,
            "necessity_weight": 0.0,
            "necessity_margin": 0.0,
            "necessity_correct_only": True,
        }
        cf_result = None

        qbound_aux = self._get_qbound_aux(gqa_output)

        if self._qbound_enabled() and qbound_aux is not None:
            (
                align_weight,
                align_temperature,
                decomp_weight,
            ) = self._qbound_v111_loss_settings()
            cf_settings = (
                self._qbound_v111b_counterfactual_settings()
            )

            # Fail fast if a V1.7 objective/routing mechanism is accidentally on.
            minimality_flag = qbound_aux.get(
                "minimal_sufficient_enabled"
            )
            if torch.is_tensor(minimality_flag):
                if float(minimality_flag.detach().item()) != 0.0:
                    raise RuntimeError(
                        "Q-Bound V1.11 must keep minimal-sufficient routing disabled."
                    )

            uncertainty_flag = qbound_aux.get(
                "uncertainty_calibration_enabled"
            )
            if torch.is_tensor(uncertainty_flag):
                if float(uncertainty_flag.detach().item()) != 1.0:
                    raise RuntimeError(
                        "Q-Bound V1.11 requires calibrated coarse/fine fusion."
                    )

            def _require_scalar_flag(name, expected):
                value = qbound_aux.get(name)
                if not torch.is_tensor(value):
                    raise KeyError(
                        f"Q-Bound V1.11 requires scalar flag {name!r}."
                    )
                actual = float(value.detach().item())
                if actual != float(expected):
                    raise RuntimeError(
                        f"Q-Bound V1.11 flag {name!r} must be {float(expected):.1f}; "
                        f"got {actual:.1f}."
                    )

            # V1.11 input-independence contract.
            if str(qbound_aux.get("qbound_version")) != self.qbound_version:
                raise RuntimeError(
                    "Q-Bound V1.11 version mismatch in training: "
                    f"pipeline={self.qbound_version!r}, "
                    f"core={qbound_aux.get('qbound_version')!r}."
                )
            _require_scalar_flag("qbound_input_is_generic", 1.0)
            _require_scalar_flag("qbound_input_uses_cra_embedding", 0.0)
            _require_scalar_flag("generic_input_context_reuse", 1.0)

            qbound_cfg = self.cfgs.get("model", {}).get("qbound", {})
            if not isinstance(qbound_cfg, dict):
                raise TypeError("model.qbound must be a mapping/dict.")

            expected_adapter = float(
                bool(qbound_cfg.get("use_generic_input_adapter", True))
            )
            _require_scalar_flag(
                "generic_input_adapter_enabled",
                expected_adapter,
            )

            expected_cf_interface = float(
                bool(
                    qbound_cfg.get(
                        "counterfactual_evidence_interface",
                        False,
                    )
                )
            )
            _require_scalar_flag(
                "counterfactual_evidence_interface_enabled",
                expected_cf_interface,
            )
            _require_scalar_flag(
                "counterfactual_mass_preserving_gradient_enabled",
                expected_cf_interface,
            )

            if bool(cf_settings["interface_enabled"]) != bool(
                expected_cf_interface
            ):
                raise RuntimeError(
                    "V1.11-B pipeline/config counterfactual interface mismatch."
                )

            if cf_settings["training_enabled"]:
                _require_scalar_flag(
                    "qbound_counterfactual_consumer_frozen",
                    1.0,
                )

            _require_scalar_flag(
                "posterior_guided_refinement_enabled", 1.0
            )
            _require_scalar_flag(
                "boundary_structured_refinement_enabled", 1.0
            )
            _require_scalar_flag(
                "prior_preserving_bounded_refinement_enabled", 1.0
            )
            _require_scalar_flag(
                "posterior_stable_reliability_enabled", 1.0
            )
            _require_scalar_flag("refine_hard_coarse_route", 0.0)
            _require_scalar_flag("refine_coarse_detached", 1.0)
            _require_scalar_flag("all_span_refine_marginal_enabled", 0.0)
            _require_scalar_flag("calibration_confidence_detached", 1.0)

            refinement_cfg = (
                self.cfgs.get("model", {})
                .get("qbound", {})
                .get("refinement", {})
            )
            if not isinstance(refinement_cfg, dict):
                raise TypeError(
                    "model.qbound.refinement must be a mapping/dict."
                )
            expected_anchor_power = float(
                refinement_cfg.get("boundary_anchor_power", 2.0)
            )
            actual_anchor_power = qbound_aux.get(
                "refine_boundary_anchor_power"
            )
            if not torch.is_tensor(actual_anchor_power):
                raise KeyError(
                    "Q-Bound V1.11 requires "
                    "qbound_aux['refine_boundary_anchor_power']."
                )
            actual_anchor_power = float(
                actual_anchor_power.detach().item()
            )
            if not math.isclose(
                actual_anchor_power,
                expected_anchor_power,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Q-Bound V1.11 boundary_anchor_power mismatch: "
                    f"config={expected_anchor_power}, "
                    f"core={actual_anchor_power}."
                )

            expected_correction_cap = float(
                refinement_cfg.get("correction_cap", 2.0)
            )
            if (
                not math.isfinite(expected_correction_cap)
                or expected_correction_cap <= 0.0
            ):
                raise ValueError(
                    "Q-Bound V1.11 refinement.correction_cap "
                    "must be finite and > 0."
                )

            actual_correction_cap = qbound_aux.get(
                "refine_correction_cap"
            )
            if not torch.is_tensor(actual_correction_cap):
                raise KeyError(
                    "Q-Bound V1.11 requires "
                    "qbound_aux['refine_correction_cap']."
                )
            actual_correction_cap = float(
                actual_correction_cap.detach().item()
            )
            if not math.isclose(
                actual_correction_cap,
                expected_correction_cap,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise RuntimeError(
                    "Q-Bound V1.11 correction_cap mismatch: "
                    f"config={expected_correction_cap}, "
                    f"core={actual_correction_cap}."
                )

            # ------------------------------------------------------
            # Weak question/event semantic alignment.
            # ------------------------------------------------------
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

            # ------------------------------------------------------
            # Dynamic latent DECOMPOSE objective: unchanged.
            # ------------------------------------------------------
            if decomp_weight > 0.0:
                qbound_decomp_loss = (
                    self._qbound_decomposition_loss(qbound_aux)
                )
                loss["qbound_decomp_loss"] = qbound_decomp_loss
                loss["total_loss"] = (
                    loss["total_loss"]
                    + decomp_weight * qbound_decomp_loss
                )

            # ------------------------------------------------------
            # V1.11-B COUNTERFACTUAL EVIDENCE SUPERVISION.
            # ------------------------------------------------------
            if cf_settings["training_enabled"]:
                model_outputs = gqa_output[2]
                cf_result = self._qbound_v111b_counterfactual_losses(
                    model_outputs=model_outputs,
                    labels=answer_id_device,
                    factual_logits=calibrated_logits,
                    settings=cf_settings,
                )

                qbound_sufficiency_loss = cf_result[
                    "sufficiency_loss"
                ]
                qbound_necessity_loss = cf_result[
                    "necessity_loss"
                ]

                weighted_cf_loss = (
                    float(cf_settings["sufficiency_weight"])
                    * qbound_sufficiency_loss
                    + float(cf_settings["necessity_weight"])
                    * qbound_necessity_loss
                )

                loss["qbound_counterfactual_sufficiency_loss"] = (
                    qbound_sufficiency_loss
                )
                loss["qbound_counterfactual_necessity_loss"] = (
                    qbound_necessity_loss
                )
                loss["qbound_counterfactual_total_loss"] = (
                    weighted_cf_loss
                )
                loss["total_loss"] = (
                    loss["total_loss"] + weighted_cf_loss
                )

        # ----------------------------------------------------------
        # Diagnostic answer branches only. These outputs are already
        # detached/no-grad in the decoder and never enter total_loss.
        # ----------------------------------------------------------
        raw_refined_logits = None
        coarse_logits = None

        if self._qbound_enabled() and qbound_aux is not None:
            raw_refined_proj = qbound_aux.get(
                "qbound_raw_refined_video_proj"
            )
            coarse_proj = qbound_aux.get(
                "qbound_coarse_video_proj"
            )

            if torch.is_tensor(raw_refined_proj):
                raw_refined_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=raw_refined_proj,
                    detach_answer=True,
                )

            if torch.is_tensor(coarse_proj):
                coarse_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=coarse_proj,
                    detach_answer=True,
                )

        qbound_logs = self._compute_qbound_v111_diagnostics(
            gqa_output=gqa_output,
            answer_proj=answer_proj,
            answer_id=answer_id,
            main_logits=calibrated_logits,
            raw_refined_logits=raw_refined_logits,
            coarse_logits=coarse_logits,
        )

        if self._qbound_enabled():
            qbound_logs["qbound_counterfactual_training_enabled"] = float(
                bool(cf_settings["training_enabled"])
            )
            qbound_logs["qbound_counterfactual_sufficiency_weight"] = float(
                cf_settings["sufficiency_weight"]
            )
            qbound_logs["qbound_counterfactual_necessity_weight"] = float(
                cf_settings["necessity_weight"]
            )
            qbound_logs["qbound_counterfactual_loss_weight"] = float(
                cf_settings["sufficiency_weight"]
                + cf_settings["necessity_weight"]
            )
            qbound_logs["qbound_counterfactual_necessity_margin"] = float(
                cf_settings["necessity_margin"]
            )
            qbound_logs[
                "qbound_counterfactual_necessity_correct_only"
            ] = float(
                bool(cf_settings["necessity_correct_only"])
            )

            if cf_result is not None:
                qbound_logs[
                    "qbound_counterfactual_sufficiency_loss"
                ] = float(
                    cf_result["sufficiency_loss"].detach().item()
                )
                qbound_logs[
                    "qbound_counterfactual_necessity_loss"
                ] = float(
                    cf_result["necessity_loss"].detach().item()
                )
                qbound_logs[
                    "qbound_counterfactual_total_loss"
                ] = float(
                    (
                        float(cf_settings["sufficiency_weight"])
                        * cf_result["sufficiency_loss"].detach()
                        + float(cf_settings["necessity_weight"])
                        * cf_result["necessity_loss"].detach()
                    ).item()
                )
                qbound_logs[
                    "qbound_counterfactual_necessity_active_ratio"
                ] = float(cf_result["necessity_active_ratio"])
                qbound_logs[
                    "qbound_counterfactual_factual_correct_logit"
                ] = float(cf_result["factual_correct_logit"])
                qbound_logs[
                    "qbound_counterfactual_sufficient_correct_logit"
                ] = float(cf_result["sufficient_correct_logit"])
                qbound_logs[
                    "qbound_counterfactual_necessary_correct_logit"
                ] = float(cf_result["necessary_correct_logit"])
                qbound_logs[
                    "qbound_counterfactual_factual_minus_sufficient_logit"
                ] = float(
                    cf_result["factual_minus_sufficient_logit"]
                )
                qbound_logs[
                    "qbound_counterfactual_necessity_gap"
                ] = float(cf_result["necessity_gap"])
                qbound_logs[
                    "qbound_counterfactual_factual_correct_prob"
                ] = float(cf_result["factual_correct_prob"])
                qbound_logs[
                    "qbound_counterfactual_sufficient_correct_prob"
                ] = float(cf_result["sufficient_correct_prob"])
                qbound_logs[
                    "qbound_counterfactual_necessary_correct_prob"
                ] = float(cf_result["necessary_correct_prob"])
                qbound_logs[
                    "qbound_counterfactual_sufficient_acc"
                ] = float(cf_result["sufficient_acc"])
                qbound_logs[
                    "qbound_counterfactual_necessary_acc"
                ] = float(cf_result["necessary_acc"])
                qbound_logs[
                    "qbound_counterfactual_necessity_margin_satisfied_ratio"
                ] = float(
                    cf_result["necessity_margin_satisfied_ratio"]
                )
                qbound_logs[
                    "qbound_counterfactual_donor_available"
                ] = float(cf_result["donor_available"])

            qbound_logs["qbound_weak_align_weight"] = float(
                align_weight
            )
            qbound_logs["qbound_decomp_weight"] = float(
                decomp_weight
            )

            if weak_align_retrieval_acc is not None:
                qbound_logs[
                    "qbound_weak_align_retrieval_acc"
                ] = float(weak_align_retrieval_acc)

        predicted = calibrated_logits.argmax(dim=-1).detach().cpu()
        running_accuracy = (
            predicted == answer_id.reshape(-1).cpu()
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
            self.monitor.reset_kv("qbound_align_loss")
            self.monitor.reset_kv("qbound_decomp_loss")
            self.monitor.reset_kv(
                "qbound_counterfactual_sufficiency_loss"
            )
            self.monitor.reset_kv(
                "qbound_counterfactual_necessity_loss"
            )
            self.monitor.reset_kv(
                "qbound_counterfactual_total_loss"
            )
            for key in self._qbound_monitor_keys():
                self.monitor.reset_kv(key)

        for index, batch in enumerate(self.train_dataloader):
            start_time = time.time()

            loss, monitor_metric, qbound_stats = self.get_loss(
                batch,
                qbound_mix=(
                    1.0 if self._qbound_enabled() else 0.0
                ),
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
                qbound_grad_norm = self._qbound_grad_norm()
                self.monitor.logkv_mean(
                    "qbound_grad_norm",
                    qbound_grad_norm,
                )

                component_grad_norms = (
                    self._qbound_component_grad_norms()
                )
                for grad_name, grad_value in (
                    component_grad_norms.items()
                ):
                    self.monitor.logkv_mean(
                        grad_name,
                        grad_value,
                    )

            clip_value = self.cfgs["optim"].get("clip", 0)
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
                    .float().cpu().item()
                )
                self.monitor.logkv_mean(
                    loss_name,
                    running_loss,
                )
                print_info += (
                    f" {loss_name}: {running_loss:.4f}"
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

        # Retained for compatibility with the current repository loop.
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

        def _print_readouts(split_prefix):
            display_names = [
                ("pre-hoc", "V1.11-B generic-input + CF evidence + BAPGR/PPBR/PSRC MAIN"),
                (
                    "raw-refined",
                    "BAPGR + PPBR RAW REFINE diagnostic",
                ),
                (
                    "coarse-branch",
                    "COMPOSE coarse diagnostic",
                ),
                (
                    "coarse-readout",
                    "calibrated MAIN answer + coarse hard span",
                ),
                (
                    "fixed-readout",
                    "calibrated MAIN answer + CRA fixed span",
                ),
                ("v15-readout", "full detached V1.5 readout"),
                ("v14-readout", "full detached V1.4 readout"),
            ]

            for score_name, label in display_names:
                full_key = f"{split_prefix}_{score_name}"
                if full_key in score:
                    self.monitor.log_info(
                        f"\n========== {split_prefix} {label} =========="
                    )
                    _print_score(score[full_key])

        self.monitor.log_info(
            f"\nVal Set\t calibrated MAIN acc: "
            f"{score['val_acc']:.2%}"
        )
        if "val_raw_refined_acc" in score:
            self.monitor.log_info(
                f"\nVal Set\t raw-refined acc: "
                f"{score['val_raw_refined_acc']:.2%}"
            )
        if "val_coarse_acc" in score:
            self.monitor.log_info(
                f"\nVal Set\t coarse acc: "
                f"{score['val_coarse_acc']:.2%}"
            )

        _print_readouts("val")

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
            f"\nTest Set\t calibrated MAIN acc: "
            f"{score['test_acc']:.2%}"
        )
        if "test_raw_refined_acc" in score:
            self.monitor.log_info(
                f"\nTest Set\t raw-refined acc: "
                f"{score['test_raw_refined_acc']:.2%}"
            )
        if "test_coarse_acc" in score:
            self.monitor.log_info(
                f"\nTest Set\t coarse acc: "
                f"{score['test_coarse_acc']:.2%}"
            )

        _print_readouts("test")

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
                "\n================ Q-Bound diagnostics ================"
            )
            for key in sorted(qbound_keys):
                self.monitor.log_info(
                    f"{key}: {score[key]:.6f}"
                )

        temporal_diagnostic_prefixes = (
            "val_QBound-",
            "test_QBound-",
            "val_QBoundRawRefined-",
            "test_QBoundRawRefined-",
            "val_QBoundCoarse-",
            "test_QBoundCoarse-",
            "val_CRA-",
            "test_CRA-",
            "val_QBoundPreSmooth-",
            "test_QBoundPreSmooth-",
            "val_QBoundPostSmooth-",
            "test_QBoundPostSmooth-",
            "val_QBoundV15-",
            "test_QBoundV15-",
            "val_QBoundV14-",
            "test_QBoundV14-",
        )

        temporal_diagnostic_keys = [
            key
            for key in score
            if key.startswith(temporal_diagnostic_prefixes)
        ]

        if temporal_diagnostic_keys:
            self.monitor.log_info(
                "\n============ Temporal-key faithfulness ============"
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
        raw_refined_count = 0
        coarse_count = 0

        score = {
            "acc": 0.0,
            "raw_refined_acc": 0.0,
            "coarse_acc": 0.0,
        }

        # Answer predictions.
        results = {}
        raw_refined_results = {}
        coarse_results = {}

        # Grounding outputs.
        ground_res = {}
        ground_pre = {}
        ground_raw_refined = {}
        ground_coarse_branch = {}
        ground_coarse_readout = {}
        ground_fixed_readout = {}
        ground_v15_readout = {}
        ground_v14_readout = {}

        keyframe = {}
        ori_keyframe = {}

        # Temporal-key diagnostics.
        qbound_keyframe = {}
        qbound_raw_refined_keyframe = {}
        qbound_coarse_keyframe = {}
        cra_keyframe = {}
        qbound_pre_smooth_keyframe = {}
        qbound_post_smooth_keyframe = {}
        qbound_v15_keyframe = {}
        qbound_v14_keyframe = {}

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
                )

                # Official V1.11 reliability-calibrated MAIN.
                calibrated_logits = self._score_answer_candidates(
                    answer_proj=answer_proj,
                    representation=fusion_proj,
                    detach_answer=False,
                )
                calibrated_predicted = (
                    calibrated_logits.argmax(dim=-1).cpu()
                )

                labels_cpu = answer_id.reshape(-1).cpu()

                score["acc"] += (
                    calibrated_predicted == labels_cpu
                ).sum().item()

                time_param = model_outputs.get("time_param")

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

                raw_refined_logits = None
                coarse_logits = None
                raw_refined_predicted = None
                coarse_predicted = None

                if qbound_aux is not None:
                    raw_refined_proj = qbound_aux.get(
                        "qbound_raw_refined_video_proj"
                    )
                    if torch.is_tensor(raw_refined_proj):
                        raw_refined_logits = self._score_answer_candidates(
                            answer_proj=answer_proj,
                            representation=raw_refined_proj,
                            detach_answer=True,
                        )
                        raw_refined_predicted = (
                            raw_refined_logits.argmax(dim=-1).cpu()
                        )
                        score["raw_refined_acc"] += (
                            raw_refined_predicted == labels_cpu
                        ).sum().item()
                        raw_refined_count += batch_size

                    coarse_proj = qbound_aux.get(
                        "qbound_coarse_video_proj"
                    )
                    if torch.is_tensor(coarse_proj):
                        coarse_logits = self._score_answer_candidates(
                            answer_proj=answer_proj,
                            representation=coarse_proj,
                            detach_answer=True,
                        )
                        coarse_predicted = (
                            coarse_logits.argmax(dim=-1).cpu()
                        )
                        score["coarse_acc"] += (
                            coarse_predicted == labels_cpu
                        ).sum().item()
                        coarse_count += batch_size

                    diagnostic_logs = self._compute_qbound_v111_diagnostics(
                        gqa_output=eval_wrapper,
                        answer_proj=answer_proj,
                        answer_id=answer_id,
                        main_logits=calibrated_logits,
                        raw_refined_logits=raw_refined_logits,
                        coarse_logits=coarse_logits,
                    )

                    batch_stats = self._collect_qbound_stats(
                        eval_wrapper,
                        extra_logs=diagnostic_logs,
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
                    video_attention = video_attention.squeeze(-1)

                if video_attention.ndim == 1:
                    video_attention = video_attention[None, :]

                if time_param is not None:
                    # Official V1.11 reliability-calibrated hard interval.
                    start_time = (
                        time_param["start"]
                        .detach().cpu().numpy()
                    )
                    end_time = (
                        time_param["end"]
                        .detach().cpu().numpy()
                    )
                    batch_ground_pre = np.column_stack(
                        (start_time, end_time)
                    )

                    batch_keyframe = (
                        time_param["key"]
                        .detach().cpu().numpy()
                    )
                    batch_ori_keyframe = (
                        time_param["ori_key"]
                        .detach().cpu().numpy()
                    )

                    batch_qbound_keyframe = None
                    batch_raw_refined_keyframe = None
                    batch_coarse_keyframe = None
                    batch_cra_keyframe = None
                    batch_pre_smooth_keyframe = None
                    batch_post_smooth_keyframe = None
                    batch_v15_keyframe = None
                    batch_v14_keyframe = None

                    batch_raw_refined = None
                    batch_coarse_branch = None
                    batch_coarse_readout = None
                    batch_fixed_readout = None
                    batch_v15_readout = None
                    batch_v14_readout = None

                    if qbound_aux is not None:
                        calibrated_key = qbound_aux.get(
                            "qbound_calibrated_key",
                            qbound_aux.get("qbound_key"),
                        )
                        raw_refined_key = qbound_aux.get(
                            "qbound_raw_refined_key"
                        )
                        coarse_key = qbound_aux.get(
                            "qbound_coarse_key"
                        )
                        cra_key = qbound_aux.get("cra_key")

                        pre_smooth_key = qbound_aux.get(
                            "qbound_pre_smooth_key"
                        )
                        post_smooth_key = qbound_aux.get(
                            "qbound_post_smooth_key",
                            calibrated_key,
                        )

                        v15_key = qbound_aux.get(
                            "v15_diagnostic_key"
                        )
                        v14_key = qbound_aux.get(
                            "v14_diagnostic_key"
                        )

                        raw_start = time_param.get(
                            "qbound_raw_refined_start"
                        )
                        raw_end = time_param.get(
                            "qbound_raw_refined_end"
                        )

                        coarse_branch_start = time_param.get(
                            "qbound_coarse_start"
                        )
                        coarse_branch_end = time_param.get(
                            "qbound_coarse_end"
                        )

                        # Same calibrated MAIN key, coarse hard interval.
                        coarse_readout_start = time_param.get(
                            "qbound_coarse_v15_start"
                        )
                        coarse_readout_end = time_param.get(
                            "qbound_coarse_v15_end"
                        )

                        fixed_start = time_param.get(
                            "qbound_legacy_fixed_window_start"
                        )
                        fixed_end = time_param.get(
                            "qbound_legacy_fixed_window_end"
                        )

                        v15_start = time_param.get(
                            "qbound_v15_diagnostic_start"
                        )
                        v15_end = time_param.get(
                            "qbound_v15_diagnostic_end"
                        )

                        v14_start = time_param.get(
                            "qbound_v14_diagnostic_start"
                        )
                        v14_end = time_param.get(
                            "qbound_v14_diagnostic_end"
                        )

                        def _stack_interval(start_value, end_value):
                            if (
                                torch.is_tensor(start_value)
                                and torch.is_tensor(end_value)
                            ):
                                return np.column_stack(
                                    (
                                        start_value.detach().cpu().numpy(),
                                        end_value.detach().cpu().numpy(),
                                    )
                                )
                            return None

                        batch_raw_refined = _stack_interval(
                            raw_start,
                            raw_end,
                        )
                        batch_coarse_branch = _stack_interval(
                            coarse_branch_start,
                            coarse_branch_end,
                        )
                        batch_coarse_readout = _stack_interval(
                            coarse_readout_start,
                            coarse_readout_end,
                        )
                        batch_fixed_readout = _stack_interval(
                            fixed_start,
                            fixed_end,
                        )
                        batch_v15_readout = _stack_interval(
                            v15_start,
                            v15_end,
                        )
                        batch_v14_readout = _stack_interval(
                            v14_start,
                            v14_end,
                        )

                        if torch.is_tensor(calibrated_key):
                            batch_qbound_keyframe = (
                                calibrated_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(raw_refined_key):
                            batch_raw_refined_keyframe = (
                                raw_refined_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(coarse_key):
                            batch_coarse_keyframe = (
                                coarse_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(cra_key):
                            batch_cra_keyframe = (
                                cra_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(pre_smooth_key):
                            batch_pre_smooth_keyframe = (
                                pre_smooth_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(post_smooth_key):
                            batch_post_smooth_keyframe = (
                                post_smooth_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(v15_key):
                            batch_v15_keyframe = (
                                v15_key.detach().cpu().numpy()
                            )
                        if torch.is_tensor(v14_key):
                            batch_v14_keyframe = (
                                v14_key.detach().cpu().numpy()
                            )

                    for batch_index, qid in enumerate(question_id):
                        results[qid] = {
                            "prediction": int(
                                calibrated_predicted.numpy()[batch_index]
                            ),
                            "answer": int(
                                labels_cpu.numpy()[batch_index]
                            ),
                        }

                        if raw_refined_predicted is not None:
                            raw_refined_results[qid] = {
                                "prediction": int(
                                    raw_refined_predicted.numpy()[
                                        batch_index
                                    ]
                                ),
                                "answer": int(
                                    labels_cpu.numpy()[batch_index]
                                ),
                            }

                        if coarse_predicted is not None:
                            coarse_results[qid] = {
                                "prediction": int(
                                    coarse_predicted.numpy()[batch_index]
                                ),
                                "answer": int(
                                    labels_cpu.numpy()[batch_index]
                                ),
                            }

                        ground_res[qid] = (
                            video_attention[batch_index].tolist()
                        )
                        ground_pre[qid] = (
                            batch_ground_pre[batch_index].tolist()
                        )
                        keyframe[qid] = (
                            batch_keyframe[batch_index].tolist()
                        )
                        ori_keyframe[qid] = (
                            batch_ori_keyframe[batch_index].tolist()
                        )

                        if batch_qbound_keyframe is not None:
                            qbound_keyframe[qid] = (
                                batch_qbound_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_raw_refined_keyframe is not None:
                            qbound_raw_refined_keyframe[qid] = (
                                batch_raw_refined_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_coarse_keyframe is not None:
                            qbound_coarse_keyframe[qid] = (
                                batch_coarse_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_cra_keyframe is not None:
                            cra_keyframe[qid] = (
                                batch_cra_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_pre_smooth_keyframe is not None:
                            qbound_pre_smooth_keyframe[qid] = (
                                batch_pre_smooth_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_post_smooth_keyframe is not None:
                            qbound_post_smooth_keyframe[qid] = (
                                batch_post_smooth_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_v15_keyframe is not None:
                            qbound_v15_keyframe[qid] = (
                                batch_v15_keyframe[
                                    batch_index
                                ].tolist()
                            )
                        if batch_v14_keyframe is not None:
                            qbound_v14_keyframe[qid] = (
                                batch_v14_keyframe[
                                    batch_index
                                ].tolist()
                            )

                        if batch_raw_refined is not None:
                            ground_raw_refined[qid] = (
                                batch_raw_refined[
                                    batch_index
                                ].tolist()
                            )
                        if batch_coarse_branch is not None:
                            ground_coarse_branch[qid] = (
                                batch_coarse_branch[
                                    batch_index
                                ].tolist()
                            )
                        if batch_coarse_readout is not None:
                            ground_coarse_readout[qid] = (
                                batch_coarse_readout[
                                    batch_index
                                ].tolist()
                            )
                        if batch_fixed_readout is not None:
                            ground_fixed_readout[qid] = (
                                batch_fixed_readout[
                                    batch_index
                                ].tolist()
                            )
                        if batch_v15_readout is not None:
                            ground_v15_readout[qid] = (
                                batch_v15_readout[
                                    batch_index
                                ].tolist()
                            )
                        if batch_v14_readout is not None:
                            ground_v14_readout[qid] = (
                                batch_v14_readout[
                                    batch_index
                                ].tolist()
                            )

                else:
                    for batch_index, qid in enumerate(question_id):
                        results[qid] = {
                            "prediction": int(
                                calibrated_predicted.numpy()[batch_index]
                            ),
                            "answer": int(
                                labels_cpu.numpy()[batch_index]
                            ),
                        }
                        ground_res[qid] = (
                            video_attention[batch_index].tolist()
                        )

                progress = float(index + 1) / len(data_loader)
                print(
                    f"\rProgress: {progress:.2%}",
                    end="",
                    flush=True,
                )

        if count == 0:
            raise RuntimeError("Evaluation data loader is empty.")

        score["acc"] /= count

        if raw_refined_count > 0:
            score["raw_refined_acc"] /= raw_refined_count
        else:
            score.pop("raw_refined_acc", None)

        if coarse_count > 0:
            score["coarse_acc"] /= coarse_count
        else:
            score.pop("coarse_acc", None)

        if qbound_stat_count > 0:
            for name, total in qbound_stat_sums.items():
                score[name] = total / qbound_stat_count

        split = data_loader.dataset.split
        csv_path = self.cfgs["dataset"]["csv_path"]

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

        # --------------------------------------------------------------
        # Temporal-key faithfulness.
        # --------------------------------------------------------------
        if qbound_keyframe:
            if not hasattr(self.metric, "temporal_key_metrics"):
                raise AttributeError(
                    "Q-Bound V1.11 requires "
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

        if qbound_raw_refined_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_raw_refined_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundRawRefined",
                )
            )

        if qbound_coarse_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_coarse_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundCoarse",
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

        if qbound_v15_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_v15_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundV15",
                )
            )

        if qbound_v14_keyframe:
            score.update(
                self.metric.temporal_key_metrics(
                    qbound_v14_keyframe,
                    ground_truth,
                    segments,
                    prefix="QBoundV14",
                )
            )

        result = {
            "ph": post_hoc_ground,
            "result": results,
            "raw_refined_result": raw_refined_results,
            "coarse_result": coarse_results,
            "gt_ground": ground_truth,
            "ph_attn": ground_res,

            "qbound_key": qbound_keyframe,
            "qbound_raw_refined_key": qbound_raw_refined_keyframe,
            "qbound_coarse_key": qbound_coarse_keyframe,
            "cra_key": cra_keyframe,
            "qbound_pre_smooth_key": qbound_pre_smooth_keyframe,
            "qbound_post_smooth_key": qbound_post_smooth_keyframe,
            "qbound_v15_key": qbound_v15_keyframe,
            "qbound_v14_key": qbound_v14_keyframe,

            "raw_refined": ground_raw_refined,
            "coarse_branch": ground_coarse_branch,
            "coarse_readout": ground_coarse_readout,
            "fixed_readout": ground_fixed_readout,
            "v15_readout": ground_v15_readout,
            "v14_readout": ground_v14_readout,
        }

        if saw_time_param:
            # Official calibrated MAIN.
            score["pre-hoc"] = self.metric(
                ground_pre,
                ground_truth,
                results,
                gs=True,
            )

            # BAPGR + PPBR bounded fine REFINE branch.
            if ground_raw_refined and raw_refined_results:
                score["raw-refined"] = self.metric(
                    ground_raw_refined,
                    ground_truth,
                    raw_refined_results,
                    gs=True,
                )

            # Coarse COMPOSE diagnostic branch.
            if ground_coarse_branch and coarse_results:
                score["coarse-branch"] = self.metric(
                    ground_coarse_branch,
                    ground_truth,
                    coarse_results,
                    gs=True,
                )

            # Same calibrated answer prediction, alternative hard interval.
            if ground_coarse_readout:
                score["coarse-readout"] = self.metric(
                    ground_coarse_readout,
                    ground_truth,
                    results,
                    gs=True,
                )

            if ground_fixed_readout:
                score["fixed-readout"] = self.metric(
                    ground_fixed_readout,
                    ground_truth,
                    results,
                    gs=True,
                )

            if ground_v15_readout:
                score["v15-readout"] = self.metric(
                    ground_v15_readout,
                    ground_truth,
                    results,
                    gs=True,
                )

            if ground_v14_readout:
                score["v14-readout"] = self.metric(
                    ground_v14_readout,
                    ground_truth,
                    results,
                    gs=True,
                )

            main_pre = score["pre-hoc"]

            def _record_metric_delta(reference_name, reference_score):
                if not isinstance(reference_score, dict):
                    return

                for metric_name in (
                    "mIoP",
                    "mIoU",
                    "TIoP@0.3",
                    "TIoP@0.5",
                    "TIoU@0.3",
                    "TIoU@0.5",
                    "Acc&GQA",
                ):
                    if (
                        metric_name in main_pre
                        and metric_name in reference_score
                    ):
                        safe_name = (
                            metric_name
                            .replace("@", "_at_")
                            .replace("&", "_and_")
                        )
                        score[
                            f"qbound_calibrated_minus_{reference_name}_"
                            f"{safe_name}"
                        ] = float(
                            main_pre[metric_name]
                            - reference_score[metric_name]
                        )

            _record_metric_delta(
                "rawrefined",
                score.get("raw-refined"),
            )
            _record_metric_delta(
                "coarsebranch",
                score.get("coarse-branch"),
            )
            _record_metric_delta(
                "coarse",
                score.get("coarse-readout"),
            )
            _record_metric_delta(
                "fixed",
                score.get("fixed-readout"),
            )
            _record_metric_delta(
                "v15",
                score.get("v15-readout"),
            )
            _record_metric_delta(
                "v14",
                score.get("v14-readout"),
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
                    "raw_refined_tg": ground_raw_refined,
                    "coarse_branch_tg": ground_coarse_branch,
                    "coarse_tg": ground_coarse_readout,
                    "fixed_tg": ground_fixed_readout,
                    "v15_tg": ground_v15_readout,
                    "v14_tg": ground_v14_readout,
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