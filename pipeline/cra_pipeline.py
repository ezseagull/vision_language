import collections
import json
import os
import re
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
    CRA R2-C / V3.5 pipeline with controlled Q-Bound V3.8-Minimal M0.

    M0 adds an independent L_evQA on the joint answer-event evidence branch.
    It preserves the legacy QA/grounding paths and their verified loss
    configuration for the joint phase. A separate frozen-feature probe trains
    ONLY the new event modules (and reports legacy losses as diagnostics).
    M0 temporal inference never consumes answer labels or temporal ground truth.

    Q-Bound R2-A/B retain three optimization routes; R2-C has two:

        A/B: L_CRA/main + lambda_G * L_ground + lambda_P * L_pred
        C:   L_CRA/main + lambda_G * L_ground

    * L_CRA/main consumes the QA feature pooled with an identical but detached
      Support Field.  It cannot train temporal relevance or extent topology.
    * L_ground aligns differentiable raw-CLIP evidence with a detached,
      question-only target.  The structural graph is detached, so this route
      trains semantic relevance only.
    * V3.5 optionally adds SAME-VIDEO Q1/Q2 evidence discrimination on top.
      It does not modify Q1 QA, R2-C L_ground, fixed propagation, or decoder.
    * A/B use L_pred to train an independent video-only predictor.
      C has no predictor and uses fixed adjacent edges directly.

    The Question creates a semantic seed; a detached video-only predictive
    graph propagates that seed into support_temporal_prob.  The same numerical
    support is used for QA, grounding, training, and inference.  Analytic
    boundary/posterior-barycenter spans are metadata and evaluation views only:
    there is no hard-MAP temporal decision or learned span scorer.
    """

    M1_VIEWS = ("question_only", "cra_answer_conditional", "mil_answer_conditional")
    M1_MONITOR = "v38_m1_cra_acc_gqa"

    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

        # The historical QBoundCore / RefineDecoder wire contract stays 2.8.0.
        self.qbound_version = "2.8.0"
        self.qbound_experiment_version = str(
            self._qbound_config().get("version", self.qbound_version)
        ) if self._qbound_enabled() else self.qbound_version
        self.v38_minimal_enabled = bool(
            self._qbound_enabled() and self.qbound_experiment_version == "3.8-minimal"
        )
        self.v38_m1_enabled = self._qbound_enabled() and self.qbound_experiment_version == "3.8-m1"
        grouped = self.cfgs["dataset"].get("m1_grouped_batch", False)
        if type(grouped) is not bool:
            raise TypeError("dataset.m1_grouped_batch must be boolean.")
        self.m1_grouped_batch = self.v38_m1_enabled or grouped
        if self._qbound_enabled() and self.qbound_experiment_version not in (
            self.qbound_version, "3.8-minimal", "3.8-m1"
        ):
            raise ValueError("Unsupported Q-Bound experiment version.")
        self.v38_settings = self._v38_loss_settings()
        self._v38_grad_step = 0
        self._v38_eligible_rows_this_epoch = 0
        self.r2_graph_mode = (
            self._qbound_config().get("support_graph_mode", "learned")
            if self._qbound_enabled() else None
        )
        self.r2_predictor_enabled = self._qbound_config().get("support_predictor_enabled", True)
        if not isinstance(self.r2_predictor_enabled, bool):
            raise TypeError("support_predictor_enabled must be boolean.")
        if self._qbound_enabled() and not self.r2_predictor_enabled and self.r2_graph_mode != "fixed":
            raise ValueError("R2-C requires fixed graph.")
        self.eval_test_each_epoch = self.cfgs["stat"].get(
            "eval_test_each_epoch", self.v38_minimal_enabled or self.v38_m1_enabled
        )
        self.save_val_predictions = self.cfgs["stat"].get(
            "save_val_predictions", True
        )
        for name, value in (
            ("stat.eval_test_each_epoch", self.eval_test_each_epoch),
            ("stat.save_val_predictions", self.save_val_predictions),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a boolean.")
        if self.v38_minimal_enabled and not self.eval_test_each_epoch:
            raise ValueError(
                "V3.8-Minimal requires stat.eval_test_each_epoch=true. "
                "The test split is diagnostic only, never used for selection."
            )
        if self.v38_minimal_enabled and not self.save_val_predictions:
            raise ValueError(
                "V3.8-Minimal requires stat.save_val_predictions=true so "
                "posterior and timestamp audits are retained for validation."
            )
        if self.v38_minimal_enabled:
            selection_name = str(self.cfgs["stat"].get("monitor", {}).get("metric", ""))
            if selection_name.lower().startswith(("test", "test_", "test/")):
                raise ValueError("M0 model-selection metric must not refer to test split.")
        # Q-Bound core/decoder still advertise 2.8.0 for checkpoint compatibility.
        # V3.5 is a training experiment identifier, NOT a changed core version.
        self.v35_pair_settings = self._v35_pair_loss_settings()
        self._v35_loss_step = 0
        if self.v35_pair_settings["enabled"]:
            if not self._qbound_enabled() or self.r2_graph_mode != "fixed" or self.r2_predictor_enabled:
                raise ValueError("QDTG V3.5 requires enabled R2-C with fixed graph and no predictor.")
        if self.m1_grouped_batch:
            self._configure_v38_m1_pipeline()

        # M2 is an opt-in WHERE overlay on the same M1 checkpoint/version.
        # This step adds evaluation only. It does not change the train loss,
        # parameter groups, legacy checkpoint selection, or CRA answer.
        self._m2_mbr_config, self._m2_mbr = self._configure_m2_mbr_eval()
        if self._m2_mbr is not None:
            self._m2_preflight_full_splits()

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
            if configured_version != self.qbound_experiment_version:
                raise ValueError(
                    "CRAPipeline/Q-Bound experiment version mismatch: "
                    f"expected {self.qbound_experiment_version!r}, got "
                    f"{configured_version!r}."
                )
            if self.v38_minimal_enabled:
                self._configure_v38_phase_and_optimizer()

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

    # ------------------------------------------------------------------
    # V3.8-Minimal (M0): independent QA supervision and controlled phases.
    # ------------------------------------------------------------------
    def _v38_loss_settings(self):
        """Pre-register M0 settings; never infer loss weights from test scores."""
        defaults = dict(enabled=False, weight=0.0, phase="joint",
                        gradient_audit_every=0, dump_joint_matrices=True)
        if not self.v38_minimal_enabled:
            return defaults
        qcfg = self._qbound_config()
        vcfg = qcfg.get("v38", {})
        if not isinstance(vcfg, dict):
            raise TypeError("model.qbound.v38 must be a mapping/dict.")
        if "train_phase" not in vcfg:
            raise ValueError(
                "M0 requires an explicit model.qbound.v38.train_phase "
                "('probe' or 'joint'); do not silently skip P2."
            )
        lcfg = vcfg.get("evidence_loss", {})
        if not isinstance(lcfg, dict):
            raise TypeError("model.qbound.v38.evidence_loss must be a mapping/dict.")
        if "loss_weight" not in lcfg:
            raise ValueError(
                "M0 requires model.qbound.v38.evidence_loss.loss_weight "
                "explicitly, for reproducible comparison."
            )
        enabled = lcfg.get("enabled", True)
        if not isinstance(enabled, bool) or not enabled:
            raise ValueError("M0 requires independently enabled evidence-only CE.")
        weight = float(lcfg.get("loss_weight", 1.0))
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("M0 evidence_loss.loss_weight must be positive and finite.")
        phase = str(vcfg.get("train_phase", "joint"))
        if phase not in ("probe", "joint"):
            raise ValueError("M0 train_phase must be 'probe' or 'joint'.")
        audit_every = vcfg.get("gradient_audit_every", 0)
        if type(audit_every) is not int or audit_every < 0:
            raise TypeError("M0 gradient_audit_every must be int >=0.")
        dump_joint = vcfg.get("dump_joint_matrices", True)
        if not isinstance(dump_joint, bool):
            raise TypeError("M0 dump_joint_matrices must be boolean.")
        return dict(enabled=True, weight=weight, phase=phase,
                    gradient_audit_every=audit_every,
                    dump_joint_matrices=dump_joint)

    def _v38_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _configure_v38_phase_and_optimizer(self):
        """Freeze old paths ONLY in the explicitly selected P2 probe phase."""
        model = self._v38_model()
        if not getattr(model, "qbound_v38_minimal_enabled", False):
            raise RuntimeError("M0 pipeline requires matching V3.8-Minimal CRA wrapper.")
        bank = getattr(model, "qbound_v38_bank", None)
        evidence = getattr(model, "qbound_v38_evidence", None)
        if bank is None or evidence is None:
            raise RuntimeError("M0 event bank/evidence head is absent from CRA.")
        fresh = {id(p) for m in (bank, evidence) for p in m.parameters()}
        opt_ids = {id(p) for group in self.optimizer.param_groups for p in group["params"]}
        if not fresh.issubset(opt_ids):
            raise RuntimeError(
                "The optimizer does not contain ALL new event bank/evidence "
                "parameters. Rebuild it after constructing the V3.8 CRA model."
            )
        if self.v38_settings["phase"] == "probe":
            if self.v35_pair_settings["enabled"]:
                raise ValueError(
                    "M0 frozen-feature probe cannot run V3.5 pair loss; "
                    "disable pair_loss explicitly in the probe YAML."
                )
            for parameter in model.parameters():
                parameter.requires_grad_(id(parameter) in fresh)
        if not any(p.requires_grad for p in bank.parameters()) or not any(
            p.requires_grad for p in evidence.parameters()
        ):
            raise RuntimeError("M0 event bank and evidence scorer must be trainable.")

    def _v38_set_training_mode(self):
        """Make a frozen probe deterministic in the old CRA/encoder modules."""
        if not self.v38_minimal_enabled or self.v38_settings["phase"] != "probe":
            return
        model = self._v38_model()
        model.eval()
        model.qbound_v38_bank.train()
        model.qbound_v38_evidence.train()

    def _v38_outputs(self, model_outputs):
        if not self.v38_minimal_enabled:
            return None
        if not isinstance(model_outputs, dict) or model_outputs.get("qbound_v38_active") is not True:
            raise RuntimeError("M0 forward did not expose an active event branch.")
        required = (
            "qbound_v38_ev_qa_logits", "qbound_v38_ev_qa_loss_mask",
            "qbound_v38_evidence", "qbound_v38_bank",
            "qbound_v38_event_temporal_question",
            "qbound_v38_event_temporal_marginal",
            "qbound_v38_event_temporal_conditional_pred",
            "qbound_v38_grounding_views",
        )
        missing = [key for key in required if key not in model_outputs]
        if missing:
            raise KeyError("M0 forward missing outputs: " + ", ".join(missing))
        logits = model_outputs["qbound_v38_ev_qa_logits"]
        eligible = model_outputs["qbound_v38_ev_qa_loss_mask"]
        if logits.ndim != 2 or logits.size(1) != 5 or eligible.shape != (logits.size(0),):
            raise ValueError("M0 evidence logits/mask must have [B,5]/[B].")
        if eligible.dtype != torch.bool or not torch.isfinite(logits).all():
            raise ValueError("M0 evidence logits must be finite and eligible mask bool.")
        bank = model_outputs["qbound_v38_bank"]
        posterior = model_outputs["qbound_v38_evidence"]["event_joint_posterior"]
        event_mask = bank["event_valid_mask"]
        if posterior.shape != (*event_mask.shape, 5):
            raise ValueError("M0 event joint posterior must be [B,M,5].")
        if not torch.allclose(posterior.sum((1, 2)), torch.ones_like(logits[:, 0]), atol=1e-4):
            raise RuntimeError("M0 joint posterior is not normalized.")
        video_mask = model_outputs.get("qbound_video_mask")
        if video_mask is None:
            raise KeyError("M0 requires qbound_video_mask from RefineDecoder.")
        for key in ("question", "marginal", "conditional_pred"):
            temporal_key = "qbound_v38_event_temporal_" + key
            self._v38_assert_distribution(temporal_key, model_outputs[temporal_key], video_mask)
        return model_outputs

    @staticmethod
    def _v38_assert_distribution(name, values, valid_mask):
        if values.shape != valid_mask.shape or not torch.isfinite(values).all():
            raise ValueError(f"{name} must be finite [B,T] matching the video mask.")
        if (values.detach() < -1e-6).any() or (
            values.detach() * (~valid_mask.bool()).to(values.dtype)
        ).abs().max() > 1e-5:
            raise RuntimeError(f"{name} contains negative/padding probability mass.")
        if not torch.allclose((values * valid_mask).sum(-1),
                              torch.ones_like(values[:, 0]), atol=1e-4):
            raise RuntimeError(f"{name} must sum to one over valid frames.")

    def _v38_evidence_loss(self, model_outputs, labels):
        out = self._v38_outputs(model_outputs)
        logits = out["qbound_v38_ev_qa_logits"].float()
        eligible = out["qbound_v38_ev_qa_loss_mask"]
        if labels.shape != (logits.size(0),):
            raise ValueError("M0 answer labels must be [B].")
        if (labels < 0).any() or (labels >= 5).any():
            raise ValueError("M0 labels must be valid candidate indices [0,4].")
        per_sample = F.cross_entropy(logits, labels, reduction="none")
        # Very short fallback samples contribute no gradient to event QA.
        if eligible.any():
            evidence_loss = per_sample[eligible].mean()
        else:
            evidence_loss = logits.sum() * 0.0
        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            evidence = out["qbound_v38_evidence"]
            marginal = evidence["event_posterior_marginal"]
            question = evidence["event_posterior_question"]
            valid = out["qbound_v38_bank"]["event_valid_mask"]
            diag = {
                "qbound_v38_evqa_loss": float(evidence_loss.detach().item()),
                "qbound_v38_evqa_eligible_fraction": float(eligible.float().mean().item()),
                "qbound_v38_evqa_eligible_count": float(eligible.sum().item()),
                "qbound_v38_evqa_acc": float((predicted == labels).float().mean().item()),
                "qbound_v38_evqa_acc_eligible": (
                    float((predicted[eligible] == labels[eligible]).float().mean().item())
                    if eligible.any() else 0.0
                ),
                "qbound_v38_bank_event_count": float(valid.sum(-1).float().mean().item()),
                "qbound_v38_bank_fallback_fraction": float(out["qbound_v38_bank"]["bank_fallback"].float().mean().item()),
                "qbound_v38_event_prior_l1": float((marginal - question).abs().sum(-1).mean().item()),
                "qbound_v38_event_joint_consistency_max": float(
                    evidence["event_posterior_consistency_max_abs"].max().item()
                ),
                "qbound_v38_event_prior_entropy": float(
                    -(question.clamp_min(1e-30).log() * question).sum(-1).mean().item()
                ),
                "qbound_v38_event_marginal_entropy": float(
                    -(marginal.clamp_min(1e-30).log() * marginal).sum(-1).mean().item()
                ),
                "qbound_v38_interaction_centered_rms": float(
                    evidence["event_interaction_centered"].masked_select(valid.unsqueeze(-1).expand_as(
                        evidence["event_interaction_centered"])).square().mean().sqrt().item()
                ),
            }
        return evidence_loss, diag

    def _v38_gradient_audit(self, event_loss, model_outputs, labels):
        """Exact answer-label gradient check; diagnostic only, never extra loss."""
        evidence = model_outputs["qbound_v38_evidence"]
        g = evidence["event_g"]
        eligible = model_outputs["qbound_v38_ev_qa_loss_mask"]
        if not g.requires_grad or not eligible.any():
            return {"qbound_v38_grad_prior_norm": 0.0,
                    "qbound_v38_grad_prior_identity_max_error": 0.0}
        grad = torch.autograd.grad(event_loss, g, retain_graph=True)[0]
        # d CE / dg_i = [P_marg(i) - P(i|y)] / N_eligible.
        conditional = evidence["event_posterior_conditional_all"]
        correct_cond = conditional.gather(
            2, labels[:, None, None].expand(-1, conditional.size(1), 1)
        ).squeeze(-1)
        expected = (evidence["event_posterior_marginal"] - correct_cond)
        expected = expected * eligible.unsqueeze(-1).to(expected.dtype) / eligible.sum().clamp_min(1)
        valid = model_outputs["qbound_v38_bank"]["event_valid_mask"]
        error = (grad.detach() - expected.detach()).masked_select(valid)
        return {
            "qbound_v38_grad_prior_norm": float(grad.detach().float().norm().item()),
            "qbound_v38_grad_prior_identity_max_error": float(error.abs().max().item()) if error.numel() else 0.0,
        }

    def _v38_parameter_grad_norms(self):
        if not self.v38_minimal_enabled:
            return {}
        model = self._v38_model()
        norms = {}
        for name in ("qbound_v38_bank", "qbound_v38_evidence"):
            module = getattr(model, name)
            norms[f"qbound_v38_{name.removeprefix('qbound_v38_')}_grad_norm"] = float(
                sum(p.grad.detach().float().square().sum().item()
                    for p in module.parameters() if p.grad is not None) ** 0.5
            )
        return norms

    # V3.8-M1 grouped competition. No answer labels or timestamps build masks.
    def _configure_v38_m1_pipeline(self):
        if self.v38_m1_enabled and isinstance(self.model, nn.DataParallel):
            raise ValueError("M1 [B,B] question pools must stay on one device, not DataParallel.")
        align_cfg = self.cfgs["optim"]["loss"].get("align", {}) or {}
        self.m1_mask_cra_alignment = any(float(weight) != 0.0 for weight in align_cfg.values())
        if self.m1_mask_cra_alignment:
            import inspect
            if "align_negative_mask" not in inspect.signature(self.criterion).parameters:
                raise RuntimeError(
                    "Grouped batches require BuildLossFunc(..., align_negative_mask=...). "
                    "Update the criterion before training so CRA cannot sample same-video negatives."
                )
        if not self.v38_m1_enabled:
            return  # legacy-only control with identical sampler and masks
        stat = self.cfgs["stat"]
        if not self.eval_test_each_epoch or not self.save_val_predictions:
            raise ValueError("M1 requires per-epoch test evaluation and saved validation predictions.")
        if stat.get("monitor", {}).get("metric") != self.M1_MONITOR or stat["monitor"].get("mode") != "max":
            raise ValueError(f"M1 requires monitor.metric={self.M1_MONITOR} and monitor.mode=max.")
        if self.v35_pair_settings["enabled"] or self.r2_graph_mode != "fixed" or self.r2_predictor_enabled:
            raise ValueError("M1 uses R2-C legacy and its own question loss, not V3.5 pair loss.")
        cfg = self._qbound_config().get("v38", {})
        settings = {}
        for key in ("question_loss_weight", "answer_loss_weight"):
            val = float(cfg.get(key, 1.0))
            if not np.isfinite(val) or val <= 0:
                raise ValueError(f"M1 {key} must be positive and finite.")
            settings[key] = val
        period = cfg.get("gradient_audit_every", 100)
        if type(period) is not int or period < 0:
            raise ValueError("M1 gradient_audit_every must be a nonnegative integer.")
        settings["gradient_audit_every"] = period
        self.v38_m1_settings = settings
        self._m1_step = 0
        model = self._v38_model()
        head = getattr(model, "qbound_v38_m1_evidence", None)
        bank = getattr(model, "qbound_v38_m1_bank", None)
        if not getattr(model, "qbound_v38_m1_enabled", False) or head is None or bank is None:
            raise RuntimeError("M1 pipeline requires the matching CRA scorer and bank.")
        params = list(head.parameters())
        opt_ids = collections.Counter(id(p) for g in self.optimizer.param_groups for p in g["params"])
        if not params or any(not p.requires_grad or opt_ids[id(p)] != 1 for p in params):
            raise RuntimeError("Every M1 scorer parameter must train and appear once in the optimizer.")
        if any(p.requires_grad for p in bank.parameters()):
            raise RuntimeError("M1 bank geometry must not train the M0 reader.")

    @staticmethod
    def _m1_metadata(batch, name, batch_size):
        value = batch[name]
        if torch.is_tensor(value) and value.ndim == 1:
            value = value.tolist()
        if not isinstance(value, (list, tuple)) or len(value) != batch_size:
            raise ValueError(f"Grouped CRA batch requires one {name} per row.")
        return value

    @staticmethod
    def _m1_question_text_key(question):
        # Preserve every semantic word and its order, including temporal
        # cues. Only remove formatting and common non-semantic wrappers.
        if not isinstance(question, str):
            raise TypeError("question_txt must contain strings.")
        words = re.findall(r"\w+", question.casefold())
        for prefix in (("can", "you", "tell", "me"), ("please", "tell", "me"), ("please",)):
            if tuple(words[:len(prefix)]) == prefix:
                words = words[len(prefix):]
                break
        return tuple(words)

    def _m1_batch_masks(self, batch, question, batch_size, device):
        videos = self._m1_metadata(batch, "video_id", batch_size)
        if any(v is None or not str(v).strip() for v in videos):
            raise ValueError("Grouped CRA batch has a missing video_id.")
        groups = collections.defaultdict(list)
        for i, video in enumerate(videos):
            groups[str(video).strip()].append(i)
        group_ids = torch.empty(batch_size, dtype=torch.long)
        for idx, rows in enumerate(groups.values()):
            group_ids[rows] = idx
        different_video = group_ids[:, None].ne(group_ids[None, :])
        if not self.v38_m1_enabled:
            return different_video.to(device), None, {}

        type_field = "question_type" if "question_type" in batch else "type"
        types = self._m1_metadata(batch, type_field, batch_size)
        texts = self._m1_metadata(batch, "question_txt", batch_size) if "question_txt" in batch else None
        text_keys = [self._m1_question_text_key(s) for s in texts] if texts is not None else [()] * batch_size
        tokens = question.squeeze(1) if question.ndim == 3 and question.size(1) == 1 else question
        if tokens.ndim != 2 or tokens.size(0) != batch_size:
            raise ValueError("M1 question tokens must be [B,L] or [B,1,L].")
        token_keys = [tuple(row) for row in tokens.detach().cpu().tolist()]
        usable = []
        for code in types:
            if not isinstance(code, str) or not code.strip() or code.strip().casefold() in {"null", "nan", "none"}:
                raise ValueError("M1 needs the original NextGQA string question type.")
            usable.append(not code.strip().upper().startswith("D"))

        representatives = {}
        for video, rows in groups.items():
            keep, seen_text, seen_tokens = [], set(), set()
            for i in rows:
                if not usable[i] or token_keys[i] in seen_tokens or (text_keys[i] and text_keys[i] in seen_text):
                    continue
                keep.append(i)
                seen_tokens.add(token_keys[i])
                if text_keys[i]:
                    seen_text.add(text_keys[i])
            representatives[video] = keep

        candidate_mask = torch.zeros((batch_size, batch_size), dtype=torch.bool)
        for video, rows in groups.items():
            positives = representatives[video]
            if not positives:
                continue
            blocked_text = {text_keys[i] for i in positives if text_keys[i]}
            blocked_tokens = {token_keys[i] for i in positives}
            keep = list(positives)
            for other_video, candidates in representatives.items():
                if other_video == video:
                    continue
                for i in candidates:
                    if token_keys[i] in blocked_tokens or (text_keys[i] and text_keys[i] in blocked_text):
                        continue
                    keep.append(i)
                    blocked_tokens.add(token_keys[i])
                    if text_keys[i]:
                        blocked_text.add(text_keys[i])
            # One denominator for all rows in a video, not one per positive.
            # Descriptive/duplicate rows have no diagonal and no question loss.
            candidate_mask[torch.tensor(rows)[:, None], torch.tensor(keep)[None, :]] = True
        positives = int(candidate_mask.diagonal().sum())
        logs = {
            "v38_m1_batch_videos": float(len(groups)),
            "v38_m1_question_positive_count": float(positives),
            "v38_m1_descriptive_count": float(batch_size - sum(usable)),
            "v38_m1_duplicate_count": float(sum(usable) - positives),
        }
        return different_video.to(device), candidate_mask.to(device), logs

    def _m1_outputs(self, model_outputs):
        if model_outputs.get("qbound_v38_m1_active") is not True:
            raise RuntimeError("Expected active M1 outputs from cra.py.")
        if (model_outputs.get("qbound_v38_m1_primary_grounding_view") != "cra_answer_conditional"
                or model_outputs.get("qbound_v38_m1_qa_prediction_source") != "cra"):
            raise RuntimeError("M1 expects CRA QA with CRA-predicted-answer WHERE.")
        return model_outputs["qbound_v38_m1_bank"], model_outputs["qbound_v38_m1_evidence"]

    def _m1_objective(self, model_outputs, labels, candidate_mask, different_video):
        bank, evidence = self._m1_outputs(model_outputs)
        batch_size = labels.numel()
        if labels.ndim != 1 or ((labels < -1) | (labels >= 5)).any():
            raise ValueError("M1 labels must be [B] with values 0..4 or ignored label -1.")
        eligible_q = model_outputs["qbound_v38_m1_question_train_eligible"] & candidate_mask.sum(dim=-1).gt(1)
        eligible_a = bank["bank_train_eligible"] & labels.ge(0)
        q_log = evidence["event_mil_question_log_scores"].float()
        a_log = evidence["event_mil_answer_log_scores"].float()
        not_a_log = evidence["event_mil_answer_log_not_scores"].float()
        if q_log.shape != (batch_size,) or a_log.shape != (batch_size, 5) or not_a_log.shape != a_log.shape:
            raise ValueError("M1 MIL log-score shapes must match the grouped batch and five answers.")
        if not all(bool(torch.isfinite(v).all()) for v in (q_log, a_log, not_a_log)):
            raise RuntimeError("M1 MIL log-scores must be finite before reduction.")
        q_loss = -q_log[eligible_q].sum() / eligible_q.sum().clamp_min(1)
        target = F.one_hot(labels.clamp_min(0), num_classes=5).bool()
        a_per_row = -torch.where(target, a_log, not_a_log).sum(dim=-1)
        a_loss = a_per_row[eligible_a].sum() / eligible_a.sum().clamp_min(1)
        with torch.no_grad():
            paired = (~different_video) & eligible_q[:, None] & eligible_q[None, :]
            paired = paired & candidate_mask & candidate_mask.T
            logs = {
                "v38_m1_question_eligible_count": float(eligible_q.sum()),
                "v38_m1_answer_eligible_count": float(eligible_a.sum()),
                "v38_m1_same_video_pair_count": float(paired.triu(diagonal=1).sum()),
                "v38_m1_mil_acc": float((a_log.argmax(-1).eq(labels) & eligible_a).sum() / eligible_a.sum().clamp_min(1)),
                "v38_m1_fallback_fraction": float(bank["bank_fallback"].float().mean()),
            }
        self._m1_step += 1
        every = self.v38_m1_settings["gradient_audit_every"]
        if every and self._m1_step % every == 0 and torch.is_grad_enabled():
            params = tuple(self._v38_model().qbound_v38_m1_evidence.parameters())
            for name, value in (("question", q_loss), ("answer", a_loss)):
                grads = torch.autograd.grad(value, params, retain_graph=True, allow_unused=True)
                norm_sq = sum(float(g.detach().float().square().sum()) for g in grads if g is not None)
                logs[f"v38_m1_{name}_grad_norm"] = norm_sq ** 0.5
        return q_loss, a_loss, logs

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
        predictor_enabled = bool(getattr(self, "r2_predictor_enabled", True))

        if self._qbound_enabled():
            if not grounding["enabled"] or grounding["weight"] <= 0.0:
                raise ValueError(
                    "Canonical Q-Bound V2.8 requires a positive enabled "
                    "model.qbound.grounding_loss."
                )
            if predictor_enabled:
                if not predictive["enabled"] or predictive["weight"] <= 0.0:
                    raise ValueError("R2-A/B require positive L_pred.")
            elif predictive["enabled"] or predictive["weight"] != 0.0:
                raise ValueError("R2-C requires predictive_loss.enabled=false, loss_weight=0.")

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
        *,
        negative_mask=None,
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
        if negative_mask is not None:
            if (negative_mask.dtype != torch.bool or negative_mask.shape != logits.shape
                    or bool(negative_mask.diagonal().any())):
                raise ValueError("Grouped L_ground expects a boolean different-video negative mask [B,B].")
            keep = negative_mask | torch.eye(batch_size, dtype=torch.bool, device=logits.device)
            eligible = negative_mask.any(dim=-1)
            masked_logits = logits.masked_fill(~keep, -torch.inf)
            row_loss = F.cross_entropy(masked_logits, labels, reduction="none")
            grounding_loss = row_loss[eligible].sum() / eligible.sum().clamp_min(1)
            with torch.no_grad():
                positive = similarity.diag()[eligible]
                negative = similarity[negative_mask]
                hardest = similarity.masked_fill(~negative_mask, -torch.inf).amax(dim=-1)[eligible]
                denom = eligible.sum().clamp_min(1)
                return grounding_loss, {
                    "grounding_batch_size": float(batch_size),
                    "grounding_retrieval_acc": float((masked_logits.argmax(-1).eq(labels) & eligible).sum() / denom),
                    "grounding_positive_similarity": float(positive.sum() / denom),
                    "grounding_negative_similarity": float(negative.sum() / max(negative.numel(), 1)),
                    "grounding_hardest_negative_similarity": float(hardest.sum() / denom),
                    "grounding_margin": float((positive - hardest).sum() / denom),
                    "grounding_different_video_negative_count": float(negative.numel()),
                }
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

    # ------------------------------------------------------------------
    # V3.5 / QDTG — opt-in SAME-VIDEO question discrimination.
    # ------------------------------------------------------------------
    def _v35_pair_loss_settings(self):
        """Strictly gated so an unchanged R2-C YAML runs IDENTICAL code."""
        q_cfg = self._qbound_config() if self._qbound_enabled() else {}
        branch = q_cfg.get("pair_loss", {})
        if branch is None:
            branch = {}
        if not isinstance(branch, dict):
            raise TypeError("model.qbound.pair_loss must be a mapping/dict.")
        enabled = branch.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("model.qbound.pair_loss.enabled must be boolean.")
        weight = float(branch.get("loss_weight", 0.0))
        temperature = float(branch.get("temperature", 0.07))
        margin = float(branch.get("margin", 0.0))
        gradient_audit_every = branch.get("gradient_audit_every", 0)
        if not isinstance(gradient_audit_every, int) or isinstance(gradient_audit_every, bool):
            raise TypeError("model.qbound.pair_loss.gradient_audit_every must be an integer.")
        if weight < 0.0 or not np.isfinite(weight):
            raise ValueError("V3.5 pair loss_weight must be finite and >= 0.")
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("V3.5 pair temperature must be positive and finite.")
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("V3.5 pair margin must be finite and >= 0.")
        if gradient_audit_every < 0:
            raise ValueError("V3.5 pair gradient_audit_every must be >= 0.")
        if enabled != (weight > 0.0):
            raise ValueError(
                "V3.5 requires pair_loss.enabled=true with positive loss_weight "
                "OR enabled=false with loss_weight=0."
            )
        if not enabled and gradient_audit_every != 0:
            raise ValueError("V3.5 gradient audit requires enabled pair loss.")
        return dict(enabled=enabled, weight=weight, temperature=temperature,
                    margin=margin, gradient_audit_every=gradient_audit_every)

    @staticmethod
    def _v35_pair_inputs(batch, batch_size, device):
        """Fail closed: Q2 must be a separate question from the same actual video.

        Required TRAIN-only fields emitted by the future dataset/collate:
          paired_question: LongTensor[B,L2] (question-only ids, PAD-aware)
          pair_valid_mask: BoolTensor[B]
          paired_video_id: list[str] length B (source video of each Q2)
          paired_question_id: list[str] length B (source unique Q2 id)
        Original batch must include video_id and question_id.  Pair validity
        needs to be independently established by the DATASET, not inferred from
        answers.  Identical-video metadata alone cannot prove distinct GT spans.
        """
        required = ("paired_question", "pair_valid_mask", "paired_video_id",
                    "paired_question_id", "video_id", "question_id")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(
                "V3.5 pair_loss is enabled but training batch is missing "
                f"{missing}. Update dataset.py/dataloaders.py before training."
            )
        q2 = batch["paired_question"]
        valid = batch["pair_valid_mask"]
        if not torch.is_tensor(q2) or q2.ndim != 2 or q2.size(0) != batch_size:
            raise ValueError("V3.5 paired_question must be tensor [B,L2].")
        if q2.dtype not in (torch.int32, torch.int64) or q2.size(1) < 1:
            raise TypeError("V3.5 paired_question must be nonempty integer token ids.")
        if not torch.is_tensor(valid) or valid.dtype != torch.bool or valid.shape != (batch_size,):
            raise TypeError("V3.5 pair_valid_mask must be BoolTensor[B].")

        def _metadata(name):
            value = batch[name]
            if not isinstance(value, (list, tuple)) or len(value) != batch_size:
                raise TypeError(f"V3.5 {name} must be a list/tuple of B identifiers.")
            return [str(x) for x in value]

        vid1 = _metadata("video_id")
        vid2 = _metadata("paired_video_id")
        qid1 = _metadata("question_id")
        qid2 = _metadata("paired_question_id")
        valid_cpu = valid.detach().cpu().tolist()
        for i, is_valid in enumerate(valid_cpu):
            if is_valid and vid1[i] != vid2[i]:
                raise ValueError(f"V3.5 row {i}: Q1/Q2 belong to DIFFERENT videos.")
            if is_valid and qid1[i] == qid2[i]:
                raise ValueError(f"V3.5 row {i}: Q1/Q2 refer to the SAME question id.")
        return q2.to(device=device, non_blocking=True), valid.to(device=device, non_blocking=True)

    def _v35_pair_objective(self, model_outputs, settings):
        """Question-discriminative evidence loss (TRAIN ONLY), no new weights.

        Per-frame u_t = normalize(W_host @ raw_CLIP_t + b_host), with detached
        host projection weights and detached video features. Center per-video
        on VALID frames: r_t = u_t - mean_valid(u).  Then e_i=sum_t S_i(t) r_t.
        NOTE: because sum(S1)=sum(S2)=1, centering cancels in (e1-e2).
        Thus centering is numerically redundant for D, NOT independent
        protection against semantic shortcuts or wrong-event localization.

        D = ((e_1-e_2) dot (q_1-q_2))/temperature, with stopgrad normalized
        question-only embeddings. L_pair = mean_valid softplus(margin-D).

        The loss DOES NOT identify GT event; its gradient must be validated
        before a full training run. A wrong Q-specific event is still possible.
        """
        if not self.model.training or not torch.is_grad_enabled():
            raise RuntimeError("V3.5 QDTG pair objective is strictly train-only.")
        required = (
            "qbound_support_temporal_prob", "qbound_pair_support_temporal_prob",
            "qbound_grounding_question", "qbound_pair_grounding_question",
            "qbound_pair_valid_mask", "qbound_value_sequence", "qbound_semantic_seed_logits",
            "qbound_pair_semantic_seed_logits", "qbound_video_mask",
        )
        missing = [key for key in required if key not in model_outputs]
        if missing:
            raise KeyError(f"V3.5 CRA outputs missing pair fields: {missing}")

        s1 = model_outputs["qbound_support_temporal_prob"]
        s2 = model_outputs["qbound_pair_support_temporal_prob"]
        q1 = model_outputs["qbound_grounding_question"]
        q2 = model_outputs["qbound_pair_grounding_question"]
        valid_pairs = model_outputs["qbound_pair_valid_mask"]
        raw = model_outputs["qbound_value_sequence"]
        frame_mask = model_outputs["qbound_video_mask"].bool()

        if s1.ndim != 2 or s1.shape != s2.shape or not s1.requires_grad or not s2.requires_grad:
            raise RuntimeError("V3.5 S1/S2 must be differentiable matching [B,T] Support Fields.")
        b, t = s1.shape
        if raw.ndim != 3 or raw.shape[:2] != (b, t):
            raise ValueError("V3.5 raw VALUE must be [B,T,D].")
        if q1.ndim != 2 or q1.shape != q2.shape or q1.size(0) != b:
            raise ValueError("V3.5 question-global representations must match [B,D].")
        if valid_pairs.shape != (b,) or valid_pairs.dtype != torch.bool:
            raise TypeError("V3.5 pair_valid_mask must be BoolTensor[B].")
        if frame_mask.shape != (b, t):
            raise ValueError("V3.5 qbound_video_mask must be [B,T].")
        if q1.requires_grad or q2.requires_grad:
            raise RuntimeError("V3.5 question targets must be detached.")
        if not (torch.isfinite(s1).all() and torch.isfinite(s2).all()
                and torch.isfinite(q1).all() and torch.isfinite(q2).all()):
            raise RuntimeError("V3.5 input contains non-finite values.")

        # Calculate in float32 for AMP safety.  The sum over valid frames is 1.
        # F.linear on raw features keeps video/projector out of THIS loss graph.
        source = raw.detach().float()
        host = self._v35_host_projection_linear()
        projected = F.linear(source, host.weight.detach().float(),
                             host.bias.detach().float() if host.bias is not None else None)
        u = F.normalize(projected, p=2, dim=-1, eps=1e-6)
        valid_float = frame_mask.unsqueeze(-1).to(dtype=u.dtype)
        mean = (u * valid_float).sum(dim=1, keepdim=True) / valid_float.sum(
            dim=1, keepdim=True).clamp_min(1.0)
        residual = (u - mean) * valid_float
        if residual.size(-1) != q1.size(-1):
            raise ValueError("V3.5 host-projected visual and question dimensions differ.")

        e1 = torch.einsum("bt,btd->bd", s1.float(), residual)
        e2 = torch.einsum("bt,btd->bd", s2.float(), residual)
        question_diff = F.normalize(q1.detach().float(), p=2, dim=-1, eps=1e-6) - F.normalize(
            q2.detach().float(), p=2, dim=-1, eps=1e-6)
        discrimination = ((e1 - e2) * question_diff).sum(dim=-1) / settings["temperature"]
        per_pair = F.softplus(settings["margin"] - discrimination)

        if valid_pairs.any():
            pair_loss = per_pair[valid_pairs].mean()
        else:
            # Zero gradient, no made-up pair. The dataset controls coverage.
            pair_loss = s1.sum() * 0.0

        with torch.no_grad():
            v = valid_pairs
            n_valid = int(v.sum().item())
            info = {
                "qbound_pair_weight": float(settings["weight"]),
                "qbound_pair_temperature": float(settings["temperature"]),
                "qbound_pair_margin": float(settings["margin"]),
                "qbound_pair_valid_fraction": float(v.float().mean().item()),
                "qbound_pair_valid_count": float(n_valid),
                "qbound_pair_discrimination": float(discrimination[v].mean().item()) if n_valid else 0.0,
                "qbound_pair_margin_positive_fraction": float((discrimination[v] > 0).float().mean().item()) if n_valid else 0.0,
                "qbound_pair_support_l1": float((s1[v] - s2[v]).abs().sum(dim=-1).mean().item()) if n_valid else 0.0,
                "qbound_pair_question_cosine": float(F.cosine_similarity(q1[v].float(), q2[v].float(), dim=-1).mean().item()) if n_valid else 0.0,
                "qbound_pair_visual_residual_norm": float(residual[frame_mask].norm(dim=-1).mean().item()) if frame_mask.any() else 0.0,
            }
        if not torch.isfinite(pair_loss).all():
            raise RuntimeError("V3.5 L_pair non-finite.")
        return pair_loss, info

    def _v35_host_projection_linear(self):
        model = self.model.module if hasattr(self.model, "module") else self.model
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "video_proj"):
            raise AttributeError("V3.5 requires CRA RefineDecoder.video_proj.")
        layers = [x for x in decoder.video_proj.modules() if isinstance(x, nn.Linear)]
        if len(layers) != 1:
            raise RuntimeError("V3.5 expects exactly one nn.Linear in video_proj.")
        return layers[0]

    @staticmethod
    def _v35_pair_gradient_diagnostics(pair_loss, grounding_loss, model_outputs,
                                       pair_weight, grounding_weight):
        """Measure loss-gradient agreement at Q1 Seed logits; NOT GT direction."""
        seed1 = model_outputs["qbound_semantic_seed_logits"]
        seed2 = model_outputs["qbound_pair_semantic_seed_logits"]
        if not seed1.requires_grad or not seed2.requires_grad:
            raise RuntimeError("V3.5 Seed logits must retain gradients.")
        gp1, gp2 = torch.autograd.grad(pair_loss, (seed1, seed2),
                                      retain_graph=True, allow_unused=False)
        (gg1,) = torch.autograd.grad(grounding_loss, (seed1,),
                                    retain_graph=True, allow_unused=False)
        p = gp1.detach().float().reshape(gp1.size(0), -1)
        g = gg1.detach().float().reshape(gg1.size(0), -1)
        p_norm, g_norm = p.norm(dim=-1), g.norm(dim=-1)
        cosine = (p*g).sum(dim=-1) / (p_norm*g_norm).clamp_min(1e-12)
        weighted_ratio = (float(pair_weight) * p.norm()) / (
            float(grounding_weight) * g.norm() + 1e-12)
        return {
            "qbound_pair_grad_weighted_ratio": float(weighted_ratio.item()),
            "qbound_pair_grad_seed1_norm": float(p.norm().item()),
            "qbound_pair_grad_seed2_norm": float(gp2.detach().float().norm().item()),
            "qbound_pair_grad_grounding_norm": float(g.norm().item()),
            "qbound_pair_grad_cosine_grounding": float(cosine.mean().item()),
            "qbound_pair_grad_conflict_fraction": float((cosine < 0).float().mean().item()),
        }

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
        if not self.r2_predictor_enabled:
            predictive_flags = {"video_only_support_branch_enabled",
                                "predictive_support_graph_enabled", "qbound_predictive_loss_exposed",
                                "support_predictive_objective_expected"}
            expected_on -= predictive_flags
            expected_off |= predictive_flags
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

        training_with_grad = (self.model.training and torch.is_grad_enabled() and
                              not (self.v38_minimal_enabled and self.v38_settings["phase"] == "probe"))
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
        # Mode contract at pipeline level, in addition to Refine/Base checks.
        # It must remain true in both training and validation inference.
        actual_mode = qbound_aux.get("support_graph_mode", "learned")
        if actual_mode != self.r2_graph_mode:
            raise RuntimeError(
                "R2 graph mode mismatch in VQA outputs: "
                f"config={self.r2_graph_mode!r}, aux={actual_mode!r}."
            )
        expected_used = (
            edge_mask.to(edge_affinity.dtype)
            if actual_mode == "fixed"
            else qbound_aux["support_edge_affinity_raw"].detach()
        )
        if not torch.allclose(
            edge_affinity.detach(), expected_used.detach(), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError(
                f"R2 {actual_mode} propagation received unexpected edges."
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
        if self.r2_predictor_enabled and training_with_grad and bool(edge_mask.any()) and not predictive_loss.requires_grad:
            raise RuntimeError(
                "L_pred is detached; the support predictor would not learn."
            )
        if not self.r2_predictor_enabled and (
            predictive_loss.requires_grad or float(predictive_loss.detach().item()) != 0.0
        ):
            raise RuntimeError("R2-C must not compute L_pred.")
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
            if not getattr(self, "r2_predictor_enabled", True) and (
                key.startswith("support_predictive")
                or key.startswith("support_edge_surprise")
            ):
                continue
            scalar = self._scalar_mean(qbound_aux.get(key))
            if scalar is not None:
                log_key = key if key.startswith("qbound_") else f"qbound_{key}"
                stats[log_key] = scalar

        edge_mask = qbound_aux.get("support_edge_mask")
        edge_names = (
            ("support_edge_surprise", "support_edge_affinity_raw", "support_edge_affinity")
            if getattr(self, "r2_predictor_enabled", True)
            else ("support_edge_affinity",)
        )
        for name in edge_names:
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
            "where_input", "visual_distinctiveness", "visual_transition",
            "temporal_context", "semantic_relevance",
        ) + (("support_predictor",) if getattr(self, "r2_predictor_enabled", True) else ())
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
            # V3.5 / QDTG (populated only when pair_loss enabled)
            "qbound_pair_weight",
            "qbound_pair_temperature",
            "qbound_pair_margin",
            "qbound_pair_loss",
            "qbound_pair_weighted_loss",
            "qbound_pair_valid_fraction",
            "qbound_pair_valid_count",
            "qbound_pair_discrimination",
            "qbound_pair_margin_positive_fraction",
            "qbound_pair_support_l1",
            "qbound_pair_question_cosine",
            "qbound_pair_visual_residual_norm",
            "qbound_pair_grad_weighted_ratio",
            "qbound_pair_grad_seed1_norm",
            "qbound_pair_grad_seed2_norm",
            "qbound_pair_grad_grounding_norm",
            "qbound_pair_grad_cosine_grounding",
            "qbound_pair_grad_conflict_fraction",
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

    @staticmethod
    def _v38_monitor_keys():
        return (
            "qbound_v38_evqa_loss", "qbound_v38_evqa_weighted_loss",
            "qbound_v38_evqa_weight", "qbound_v38_evqa_eligible_fraction",
            "qbound_v38_evqa_eligible_count", "qbound_v38_evqa_acc",
            "qbound_v38_evqa_acc_eligible", "qbound_v38_probe_phase",
            "qbound_v38_bank_event_count", "qbound_v38_bank_fallback_fraction",
            "qbound_v38_event_prior_l1", "qbound_v38_event_joint_consistency_max",
            "qbound_v38_event_prior_entropy", "qbound_v38_event_marginal_entropy",
            "qbound_v38_interaction_centered_rms",
            "qbound_v38_grad_prior_norm", "qbound_v38_grad_prior_identity_max_error",
            "qbound_v38_bank_grad_norm", "qbound_v38_evidence_grad_norm",
        )

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
            get_mask(video_len, video_frames.size(1)).to(
                device, non_blocking=True
            )
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_device = self._prepare_labels(answer_id, device)
        prediction_dict = {}
        target_dict = {}
        negative_mask, candidate_mask, m1_logs = None, None, {}
        v38_m1_kwargs = {}
        if self.m1_grouped_batch:
            negative_mask, candidate_mask, m1_logs = self._m1_batch_masks(
                batch, batch["question"], batch_size, device,
            )
            if self.v38_m1_enabled:
                v38_m1_kwargs["m1_question_candidate_mask"] = candidate_mask

        # V3.5 is train-only. Missing Q2 metadata must be a HARD ERROR, never
        # silently turned into zero pair loss over an unchanged R2-C run.
        v35_pair_kwargs = {}
        if self.v35_pair_settings["enabled"]:
            paired_question, pair_valid_mask = self._v35_pair_inputs(
                batch=batch, batch_size=batch_size, device=device)
            v35_pair_kwargs = {
                "paired_question": paired_question,
                "pair_valid_mask": pair_valid_mask,
            }

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
            **v35_pair_kwargs,
            **v38_m1_kwargs,
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

        criterion_kwargs = (
            {"align_negative_mask": negative_mask}
            if self.m1_grouped_batch and self.m1_mask_cra_alignment else {}
        )
        loss = self.criterion(prediction_dict, target_dict, **criterion_kwargs)
        if "total_loss" not in loss:
            raise KeyError("The criterion must return loss['total_loss'].")
        if not torch.isfinite(loss["total_loss"]).all():
            raise RuntimeError("L_CRA/main contains non-finite values.")

        qbound_logs = {}
        if self.v38_m1_enabled:
            question_loss, answer_loss, stats = self._m1_objective(
                model_outputs, answer_id_device, candidate_mask, negative_mask,
            )
            q_weight = self.v38_m1_settings["question_loss_weight"]
            a_weight = self.v38_m1_settings["answer_loss_weight"]
            loss["v38_m1_question_loss"] = question_loss
            loss["v38_m1_answer_loss"] = answer_loss
            loss["total_loss"] = loss["total_loss"] + q_weight * question_loss + a_weight * answer_loss
            qbound_logs.update(m1_logs)
            qbound_logs.update(stats)
        if self.v38_minimal_enabled:
            ev_loss, ev_logs = self._v38_evidence_loss(model_outputs, answer_id_device)
            weighted_ev = self.v38_settings["weight"] * ev_loss
            legacy_total = loss["total_loss"]
            if self.v38_settings["phase"] == "probe":
                # P2: freeze old encoders/CRA; do NOT train on the old losses.
                loss["qbound_v38_legacy_loss_diagnostic"] = legacy_total.detach()
                loss["total_loss"] = weighted_ev
            else:
                loss["total_loss"] = legacy_total + weighted_ev
            loss["qbound_v38_evqa_loss"] = ev_loss
            loss["qbound_v38_evqa_weighted_loss"] = weighted_ev
            qbound_logs.update(ev_logs)
            qbound_logs["qbound_v38_evqa_weight"] = self.v38_settings["weight"]
            qbound_logs["qbound_v38_probe_phase"] = float(self.v38_settings["phase"] == "probe")
            self._v38_eligible_rows_this_epoch += int(ev_logs["qbound_v38_evqa_eligible_count"])
            self._v38_grad_step += 1
            period = self.v38_settings["gradient_audit_every"]
            if period and self._v38_grad_step % period == 0:
                qbound_logs.update(self._v38_gradient_audit(
                    ev_loss, model_outputs, answer_id_device
                ))
        if self._qbound_enabled():
            qbound_aux = self._validate_v280_contract(gqa_output)

            if self.v38_minimal_enabled and self.v38_settings["phase"] == "probe":
                # P2 intentionally has NO old grounding or pair objectives.
                # The untouched legacy outputs remain available for diagnostics.
                qbound_logs["qbound_grounding_weight"] = settings["grounding_weight"]
            else:
                grounding_loss, grounding_diag = (
                    self._qbound_video_question_objective(
                        model_outputs=model_outputs,
                        qbound_aux=qbound_aux,
                        temperature=settings["grounding_temperature"],
                        **({"negative_mask": negative_mask} if self.m1_grouped_batch else {}),
                    )
                )
                predictive_loss = qbound_aux["support_predictive_loss"]
                if not torch.isfinite(predictive_loss).all():
                    raise RuntimeError("L_pred contains non-finite values.")

                weighted_grounding = (
                    settings["grounding_weight"] * grounding_loss
                )
                loss["qbound_grounding_loss"] = grounding_loss
                loss["qbound_grounding_weighted_loss"] = weighted_grounding
                if self.r2_predictor_enabled:
                    weighted_predictive = settings["predictive_weight"] * predictive_loss
                    loss["qbound_predictive_loss"] = predictive_loss
                    loss["qbound_predictive_weighted_loss"] = weighted_predictive
                    loss["total_loss"] = loss["total_loss"] + weighted_grounding + weighted_predictive
                else:
                    # No predictive loss calculation/gradient in R2-C.
                    loss["total_loss"] = loss["total_loss"] + weighted_grounding

                # V3.5: only the opt-in pair objective can update the SAME shared
                # Q-Bound Seed scorer through Q1 and Q2 Support Fields.  Do not
                # change CRA QA, detached host/video/question sources or L_ground.
                if self.v35_pair_settings["enabled"]:
                    pair_loss, pair_diag = self._v35_pair_objective(
                        model_outputs=model_outputs,
                        settings=self.v35_pair_settings,
                    )
                    weighted_pair = self.v35_pair_settings["weight"] * pair_loss
                    loss["qbound_pair_loss"] = pair_loss
                    loss["qbound_pair_weighted_loss"] = weighted_pair
                    loss["total_loss"] = loss["total_loss"] + weighted_pair
                    qbound_logs.update(pair_diag)
                    self._v35_loss_step += 1
                    period = self.v35_pair_settings["gradient_audit_every"]
                    if period and self._v35_loss_step % period == 0 and pair_diag["qbound_pair_valid_count"] > 0:
                        qbound_logs.update(self._v35_pair_gradient_diagnostics(
                            pair_loss=pair_loss,
                            grounding_loss=grounding_loss,
                            model_outputs=model_outputs,
                            pair_weight=self.v35_pair_settings["weight"],
                            grounding_weight=settings["grounding_weight"],
                        ))

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
                if self.r2_predictor_enabled:
                    qbound_logs["qbound_predictive_weight"] = settings["predictive_weight"]

        predicted = main_logits.argmax(dim=-1).detach().cpu()
        running_accuracy = (
            predicted == answer_id.reshape(-1).cpu()
        ).float().mean().item()
        qbound_stats = self._collect_qbound_stats(
            gqa_output,
            extra_logs=qbound_logs,
        )
        return loss, running_accuracy, qbound_stats

    @staticmethod
    def _r1_posthoc_frame0_fixed(vatt, cur, thd_weight=0.3, dis=10):
        """R1 diagnostic: original find_seg_ada with only left bound >0 -> >=0.

        This is NOT used for primary R2 model selection or the canonical metrics.
        Its threshold, time discount, gap-skipping and right search are unchanged.
        """
        values = np.asarray(vatt, dtype=np.float64).reshape(-1)
        timestamps = np.asarray(cur, dtype=np.float64).reshape(-1)
        if values.size == 0 or timestamps.size != values.size:
            raise ValueError("R1 post-hoc requires matching nonempty key/timestamps.")
        values = (values - values.min()) / (values.max() - values.min() + 1e-12)
        peak = int(np.argmax(values))
        path = [peak]
        mean_s = np.mean(values)
        threshold = mean_s + thd_weight * mean_s
        stamp_s = timestamps[peak]

        i = 1
        while peak - i >= 0:  # ONLY change versus Metric.find_seg_ada
            stamp = timestamps[peak - i]
            elapse = abs(stamp_s - stamp)
            score = values[peak - i] / (1 + elapse / dis)
            if score >= threshold:
                path.append(peak - i)
            elif elapse > dis:
                break
            i += 1

        i = 1
        while peak + i < values.size:
            stamp = timestamps[peak + i]
            elapse = abs(stamp_s - stamp)
            score = values[peak + i] / (1 + elapse / dis)
            if score >= threshold:
                path.append(peak + i)
            elif elapse > dis:
                break
            i += 1
        return [float(timestamps[min(path)]), float(timestamps[max(path)])]

    @staticmethod
    def _r1_key_video_id(question_id):
        """NExT-GQA key: video_id_question_id, video_id may not contain '_'."""
        key = str(question_id)
        if "_" not in key:
            raise ValueError(f"Unexpected NExT-GQA question key: {key!r}")
        return key.split("_", 1)[0]

    @staticmethod
    def _r1_span_diagnostics(predictions, prefix):
        if not predictions:
            return {}
        spans = np.asarray(list(predictions.values()), dtype=np.float64)
        if spans.ndim != 2 or spans.shape[1] != 2:
            raise ValueError("R1 span diagnostics expects predictions of shape [N,2].")
        length = spans[:, 1] - spans[:, 0]
        if np.any(length < -1e-6):
            raise RuntimeError("Temporal decoder emitted reversed interval endpoints.")
        return {
            f"{prefix}ZeroDurationPct": float(np.mean(np.abs(length) <= 1e-9) * 100.0),
            f"{prefix}MeanLengthSeconds": float(np.mean(length)),
        }

    @classmethod
    def _v38_oracle_bank_audit(cls, event_metadata, ground_truth, segments):
        """VALIDATION AUDIT ONLY: best candidate IoU using sampled timestamps.

        This is an approximate diagnostic ceiling, not train supervision or
        an oracle-prediction path. Uses [start, end_exclusive) fixed windows,
        converted to [sampled time(start), sampled time(end-1)].
        """
        ious = []
        best_ranks = []
        duplicate_rate = []
        gt_positive_mass = []
        missing = 0
        for qid, meta in event_metadata.items():
            try:
                vid = cls._r1_key_video_id(qid)
                question_id = str(qid).split("_", 1)[1]
                target = ground_truth[vid]["location"][question_id]
                timeline = np.asarray(segments[vid], dtype=np.float64).reshape(-1)
            except (KeyError, IndexError, TypeError):
                missing += 1
                continue
            gt = np.asarray(target, dtype=np.float64).reshape(-1)
            if gt.size != 2 or timeline.size < int(meta["valid_length"]):
                missing += 1
                continue
            starts = np.asarray(meta["start"], dtype=np.int64)
            ends = np.asarray(meta["end_exclusive"], dtype=np.int64)
            marginal = np.asarray(meta["posterior_marginal"], dtype=np.float64)
            if not (len(starts) == len(ends) == len(marginal)):
                raise ValueError("M0 bank oracle metadata has mismatched window lengths.")
            if not len(starts):
                missing += 1
                continue
            if (starts < 0).any() or (ends <= starts).any() or (ends > timeline.size).any():
                raise RuntimeError(f"M0 bank contains invalid temporal indices: {qid}")
            s_time, e_time = timeline[starts], timeline[ends - 1]
            intersection = np.maximum(0, np.minimum(e_time, gt[1]) - np.maximum(s_time, gt[0]))
            union = np.maximum(e_time, gt[1]) - np.minimum(s_time, gt[0])
            candidate_iou = intersection / np.maximum(union, 1e-12)
            ious.append(float(candidate_iou.max()))
            # Rank only when at least one candidate overlaps GT. A zero-IoU
            # bank has no meaningful "GT candidate" whose rank can be quoted.
            if float(candidate_iou.max()) > 0.0:
                best_index = int(np.argmax(candidate_iou))
                order = np.argsort(-marginal, kind="stable")
                best_ranks.append(int(np.flatnonzero(order == best_index)[0]) + 1)
            duplicate_rate.append(float(np.mean(candidate_iou >= 0.5)))
            gt_positive_mass.append(float(marginal[candidate_iou >= 0.3].sum()))
        if not ious:
            return {"V38BankOracleCount": 0.0,
                    "V38BankOracleUnresolvedCount": float(missing)}
        ious = np.asarray(ious)
        return {
            "V38BankOracleCount": float(len(ious)),
            "V38BankOracleUnresolvedCount": float(missing),
            "V38BankOracleSampledIoUMean": float(ious.mean()),
            "V38BankOracleSampledRecall03": float(np.mean(ious >= 0.3)),
            "V38BankOracleSampledRecall05": float(np.mean(ious >= 0.5)),
            "V38BankGTBestCandidateRankMean": float(np.mean(best_ranks)) if best_ranks else -1.0,
            "V38BankGTBestCandidateRankResolvedCount": float(len(best_ranks)),
            "V38BankMarginalGTCandidateMass03": float(np.mean(gt_positive_mass)),
            "V38BankFractionWindowsIoU05": float(np.mean(duplicate_rate)),
        }

    @classmethod
    def _v38_span_failure_audit(cls, predictions, ground_truth, answer_results, prefix):
        """Evaluate WHERE, span length and QA-correct/IoU-zero without training GT.

        All numbers here are measured strictly after evaluation predictions.
        Missing GT metadata is counted, not treated as a zero-IoU failure.
        """
        valid_ious = []
        lengths = []
        qa_correct_zero = 0
        qa_correct = 0
        missing = 0
        for qid, span in predictions.items():
            try:
                vid = cls._r1_key_video_id(qid)
                question_id = str(qid).split("_", 1)[1]
                gt = np.asarray(ground_truth[vid]["location"][question_id], dtype=np.float64).reshape(-1)
            except (KeyError, TypeError, IndexError):
                missing += 1
                continue
            pred = np.asarray(span, dtype=np.float64).reshape(-1)
            if pred.size != 2 or gt.size != 2:
                missing += 1
                continue
            intersection = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
            union = max(pred[1], gt[1]) - min(pred[0], gt[0])
            iou = intersection / max(union, 1e-12)
            valid_ious.append(iou)
            lengths.append(max(0.0, pred[1] - pred[0]))
            correct = int(answer_results[qid]["prediction"] == answer_results[qid]["answer"])
            qa_correct += correct
            if correct and iou <= 1e-9:
                qa_correct_zero += 1
        if not valid_ious:
            return {f"{prefix}Resolved": 0.0, f"{prefix}Missing": float(missing)}
        return {
            f"{prefix}Resolved": float(len(valid_ious)),
            f"{prefix}Missing": float(missing),
            f"{prefix}IoUZeroFraction": float(np.mean(np.asarray(valid_ious) <= 1e-9)),
            f"{prefix}QACorrectIoUZeroFraction": float(qa_correct_zero / len(valid_ious)),
            f"{prefix}QACorrectIoUZeroAmongCorrect": float(qa_correct_zero / max(qa_correct, 1)),
            f"{prefix}MeanLengthSeconds": float(np.mean(lengths)),
        }

    def _m2_write_full_scores(self, val_score, test_score, epoch=None):
        """Export exactly the same validated C4 scores printed and logged to W&B."""
        if self._m2_mbr is None or not self._m2_mbr_config["references"]:
            return None
        both = {"val": val_score, "test": test_score}
        report = {
            "experiment": "Q-Bound V3.8-M2 C4+MBR seconds",
            "epoch": None if epoch is None else int(epoch),
            "seed": int(self.cfgs["misc"]["seed"]),
            "run": str(self.cfgs["misc"]["running_name"]),
            "selection_rule": "validation only; test observed every epoch",
            "checkpoint_for_inference": (str(self._m2_eval_checkpoint_path)
                                         if self._m2_eval_checkpoint_path else None),
            "scores": {},
        }
        for split, scores in both.items():
            key = "v38_m2_mbr_c4_seconds"
            value = scores.get(key)
            if not isinstance(value, dict):
                raise RuntimeError(f"Missing M2 C4+MBR seconds scores for {split}.")
            needed = 3358 if split == "val" else 5553
            if int(value.get("n_scored", -1)) != needed or int(value.get("n_expected", -1)) != needed:
                raise RuntimeError(f"Refusing a non-full {split} M2 score export.")
            report["scores"][split] = {k: float(v) for k, v in value.items()}
        folder = os.path.join(self.save_dir, "benchmark_scores")
        os.makedirs(folder, exist_ok=True)
        name = (f"m2_c4_full_epoch_{int(epoch):02d}.json" if epoch is not None
                else "m2_c4_full_inference.json")
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
        print(f"[M2 full Val/Test metric export] {path}", flush=True)
        return path

    @staticmethod
    def _m2_flatten_eval_metrics(split_scores):
        """Canonical single-source names for W&B (shared train / infer)."""
        scalars = {}
        for split, metrics in split_scores.items():
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (int, float, np.integer, np.floating, bool)):
                            scalars[f"{split}/{key}_{sub_key}"] = float(sub_value)
                elif isinstance(value, (int, float, np.integer, np.floating, bool)):
                    scalars[f"{split}/{key}"] = float(value)
        return scalars

    def _train_epoch(self, epoch):
        self.monitor.log_info(
            f"\n{'#' * 20} Epoch "
            f"{epoch}/{self.cfgs['optim']['epochs']} "
            f"... {'#' * 20}\n"
        )
        print("Training Stage")
        self.model.train()
        self._v38_set_training_mode()
        self._v38_eligible_rows_this_epoch = 0
        if self.m1_grouped_batch:
            sampler = self.train_dataloader.batch_sampler
            if callable(getattr(sampler, "set_epoch", None)):
                sampler.set_epoch(epoch)

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
                if not self.v35_pair_settings["enabled"] and key.startswith("qbound_pair_"):
                    continue
                if not self.r2_predictor_enabled and (
                    "predictive" in key or "edge_surprise" in key
                    or "edge_affinity_raw" in key or "support_predictor_grad" in key
                ):
                    continue
                self.monitor.reset_kv(key)
            if self.v38_minimal_enabled:
                for key in self._v38_monitor_keys():
                    self.monitor.reset_kv(key)

        v35_valid_pairs_this_epoch = 0
        m1_q_rows, m1_a_rows, m1_pairs = 0, 0, 0
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
            if self.v35_pair_settings["enabled"]:
                v35_valid_pairs_this_epoch += int(qbound_stats.get("qbound_pair_valid_count", 0))
            if self.v38_m1_enabled:
                m1_q_rows += int(qbound_stats["v38_m1_question_eligible_count"])
                m1_a_rows += int(qbound_stats["v38_m1_answer_eligible_count"])
                m1_pairs += int(qbound_stats["v38_m1_same_video_pair_count"])
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

            if self.v38_minimal_enabled:
                for name, value in self._v38_parameter_grad_norms().items():
                    self.monitor.logkv_mean(name, value)
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
                f"{'CRA QA acc' if self.v38_m1_enabled else self.cfgs['stat']['monitor']['metric']}: "
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

        if self.v38_minimal_enabled and self._v38_eligible_rows_this_epoch <= 0:
            raise RuntimeError("M0 trained on zero eligible event-bank examples this epoch.")
        if self.v38_m1_enabled and (min(m1_q_rows, m1_a_rows, m1_pairs) <= 0):
            raise RuntimeError(
                "M1 had no effective question rows, answer rows, or same-video pairs this epoch. "
                "Check the grouped sampler and candidate metadata before interpreting the run."
            )
        if self.v35_pair_settings["enabled"] and v35_valid_pairs_this_epoch <= 0:
            raise RuntimeError(
                "V3.5 enabled but no valid same-video question pairs were used "
                "during this epoch. Inspect dataset pairing/filter before training."
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

        if self.v38_m1_enabled:
            print("Eval Stage: validation (CRA answer + M1 WHERE selection)")
        else:
            print("Eval Stage: validation (canonical R2-C/V3.5 model selection)")
        self._m2_active_eval_epoch = epoch
        val_score, val_result = self.eval(
            self.val_dataloader, save=self.save_val_predictions
        )
        val_result["r2_graph_mode"] = self.r2_graph_mode
        val_result["r2_predictor_enabled"] = self.r2_predictor_enabled
        val_result["r2_variant"] = ("A" if self.r2_graph_mode == "learned"
                                    else "B" if self.r2_predictor_enabled else "C")
        val_result["experiment_variant"] = (
            "V3.8-M1" if self.v38_m1_enabled else
            ("V3.8-Minimal-" + self.v38_settings["phase"])
            if self.v38_minimal_enabled else
            ("V3.5-QDTG" if self.v35_pair_settings["enabled"] else "R2-C")
        )
        val_result["epoch"] = int(epoch)
        val_result["split"] = "val"

        if self.save_val_predictions:
            dump_dir = os.path.join(self.save_dir, "validation_dumps")
            os.makedirs(dump_dir, exist_ok=True)
            dump_path = os.path.join(dump_dir, f"val_epoch_{epoch:02d}.pth")
            # Compatible with previously saved val_epoch_15.pth format:
            # {epoch, score, result}. Each per-question prediction is in result.
            torch.save(
                {
                    "epoch": int(epoch),
                    "score": val_score,
                    "result": val_result,
                    "r2_graph_mode": self.r2_graph_mode,
                    "r2_predictor_enabled": self.r2_predictor_enabled,
                    "r2_variant": val_result["r2_variant"],
                    "experiment_variant": val_result["experiment_variant"],
                },
                dump_path,
            )
            print(f"[R2 {self.r2_graph_mode}] Validation dump: {dump_path}")

        # Always evaluate test each epoch in M0 for OBSERVATION ONLY.
        # The returned validation result and BasePipeline selection contract must
        # NEVER use test numbers to select/update a checkpoint.
        test_score = {}
        if self.eval_test_each_epoch:
            print("Eval Stage: test (diagnostic only; NEVER for selection)")
            if self.cfgs["dataset"]["name"] == "star":
                test_score = val_score
            else:
                test_score, _ = self.eval(self.test_dataloader)

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
        self._m2_write_full_scores(val_score, test_score, epoch=epoch)
        self.print_score(score)

        wandb_data = {
            "epoch": epoch,
            "experiment/v38_minimal": int(self.v38_minimal_enabled),
            "experiment/v38_probe_phase": int(
                self.v38_minimal_enabled and self.v38_settings["phase"] == "probe"
            ),
            "experiment/v38_evqa_weight": float(self.v38_settings["weight"]),
            "experiment/r2_is_fixed_graph": int(self.r2_graph_mode == "fixed")
            if self._qbound_enabled() else 0,
            "experiment/r2_predictor_enabled": int(self.r2_predictor_enabled)
            if self._qbound_enabled() else 0,
            "experiment/v35_pair_loss_enabled": int(self.v35_pair_settings["enabled"]),
            "experiment/v35_pair_loss_weight": float(self.v35_pair_settings["weight"]),
        }
        if self.v38_m1_enabled:
            wandb_data.update({
                "experiment/v38_m1": 1,
                "experiment/v38_m1_question_weight": self.v38_m1_settings["question_loss_weight"],
                "experiment/v38_m1_answer_weight": self.v38_m1_settings["answer_loss_weight"],
                "train/v38_m1_question_eligible_total": m1_q_rows,
                "train/v38_m1_answer_eligible_total": m1_a_rows,
                "train/v38_m1_same_video_pair_total": m1_pairs,
            })
        for (
            name,
            value,
        ) in self.monitor.name2val.items():
            wandb_data[
                f"train/{name}"
            ] = value

        # The same flattening convention is used by train and infer.
        # No partial C4 scores reach this point (coverage guard runs in eval).
        wandb_data.update(self._m2_flatten_eval_metrics({
            "val": val_score, "test": test_score,
        }))

        wandb.log(
            wandb_data
        )
        # BasePipeline._record_best stores this as result.pth; it MUST be val.
        return score, val_result

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
                    f"Q-Bound R2-C/V3.5 acc: {score[acc_key]:.2%}"
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

            for diagnostic_name in (
                "pre-hoc-timestamp", "post-hoc-leftfix", "seed-post-hoc"
            ):
                diagnostic_key = f"{split}_{diagnostic_name}"
                if diagnostic_key in score:
                    self.monitor.log_info(
                        f"\n========== {split} {diagnostic_name} [diagnostic] =========="
                    )
                    _print_score(score[diagnostic_key])

            merge_key = f"{split}_merge"
            if merge_key in score:
                self.monitor.log_info(
                    f"\n========== {split} merge diagnostic =========="
                )
                _print_score(score[merge_key])

        if self.v38_minimal_enabled:
            self.monitor.log_info("\n=============== Q-Bound V3.8-M0 ===============")
            for split in ("val", "test"):
                for name in ("v38_evqa_acc", "v38_qa_agreement",
                             "v38_bank_fallback_fraction",
                             "v38_joint_marginal_post-hoc",
                             "v38_predicted_answer_conditional_post-hoc",
                             "v38_question_only_post-hoc"):
                    value = score.get(f"{split}_{name}")
                    if isinstance(value, dict):
                        self.monitor.log_info(f"\n{split}_{name}: {value}")
                    elif isinstance(value, (float, int)):
                        self.monitor.log_info(f"\n{split}_{name}: {value:.6f}")
        if self.v38_m1_enabled:
            self.monitor.log_info("\n=============== Q-Bound V3.8-M1 ===============")
            for split in ("val", "test"):
                for view in self.M1_VIEWS:
                    name = f"{split}_v38_m1_{view}_post-hoc"
                    if name in score:
                        self.monitor.log_info(f"\n{name} (CRA answers): {score[name]}")
        if self._m2_mbr is not None:
            self.monitor.log_info("\n=============== Q-Bound V3.8-M2: C4 + MBR (SECONDS) ===============")
            primary = ("Acc&GQA", "Acc&VQA", "mIoP", "TIoP@0.3", "TIoP@0.5",
                       "mIoU", "TIoU@0.3", "TIoU@0.5")
            for split in ("val", "test"):
                key = f"{split}_v38_m2_mbr_c4_seconds"
                if key not in score:
                    continue
                values = score[key]
                n = values["n_scored"]
                expected = values["n_expected"]
                if n != expected:
                    raise RuntimeError(f"M2 {split} partial score reached print_score.")
                self.monitor.log_info(
                    f"\n{split.upper()} FULL: {n}/{expected} QA+spans+GT | "
                    "QA=CRA; WHERE=C4 + MBR seconds | metric values are percentages")
                self.monitor.log_info("\n" + " | ".join(
                    f"{metric}: {values[metric]:.4f}" for metric in primary))
                self.monitor.log_info(
                    f"\nW&B prefix: {split}/v38_m2_mbr_c4_seconds_")
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

    def _m1_collect_eval_batch(self, state, model_outputs, qids, predicted, labels, dump):
        bank, evidence = self._m1_outputs(model_outputs)
        cra_answer = model_outputs["qbound_v38_m1_cra_answer"].cpu()
        mil_answer = model_outputs["qbound_v38_m1_mil_answer"].cpu()
        if not torch.equal(cra_answer, predicted):
            raise RuntimeError("M1 WHERE was not conditioned on the actual CRA QA prediction.")
        temporal = model_outputs["qbound_v38_m1_temporal_by_view"]
        posteriors = model_outputs["qbound_v38_m1_event_posteriors"]
        fallback = bank["bank_fallback"]
        tail_windows = bank["event_valid_mask"] & bank["event_end_exclusive"].eq(
            bank["bank_video_lengths"][:, None]
        )
        state["mil_correct"] += int(mil_answer.eq(labels).sum())
        state["mil_agreement"] += int(mil_answer.eq(predicted).sum())
        state["fallback"] += int(fallback.sum())
        state["tail_count"] += int((~fallback).sum())
        arrays = {}
        for view in self.M1_VIEWS:
            arrays[view] = temporal[view].detach().cpu().tolist()
            tail_mass = (posteriors[view] * tail_windows).sum(dim=-1)
            state["tail_sum"][view] += float(tail_mass[~fallback].sum())
        if dump:
            steps = {
                "start": bank["event_start"], "end_exclusive": bank["event_end_exclusive"],
                "width": bank["event_width"], "bank_prior": bank["bank_prior_weights"],
                "ring_valid_steps": bank["event_ring_valid_steps"],
                "prior_question": posteriors["question_only"],
                "posterior_cra_answer": posteriors["cra_answer_conditional"],
                "posterior_mil_answer": posteriors["mil_answer_conditional"],
                "posterior_by_answer": evidence["event_posterior_by_answer"],
                "question_in_scores": evidence["event_question_in_scores"],
                "question_contrast_scores": evidence["event_question_contrast_scores"],
                "answer_in_scores": evidence["event_answer_in_scores"],
                "answer_contrast_scores": evidence["event_answer_contrast_scores"],
            }
            steps = {k: v.detach().cpu().tolist() for k, v in steps.items()}
            mask_rows = bank["event_valid_mask"].cpu().tolist()
            lengths = bank["bank_video_lengths"].cpu().tolist()
            fallbacks = fallback.cpu().tolist()
            mil_scores = evidence["event_mil_answer_scores"].detach().cpu().tolist()
        for i, qid in enumerate(qids):
            for view in self.M1_VIEWS:
                by_qid = state["temporal"][view]
                if qid in by_qid:
                    raise ValueError(f"Duplicate M1 eval question_id: {qid}")
                by_qid[qid] = arrays[view][i]
            if dump:
                valid_idx = [j for j, keep in enumerate(mask_rows[i]) if keep]
                state["metadata"][qid] = {
                    **{k: [value[i][j] for j in valid_idx] for k, value in steps.items()},
                    "valid_mask": [True] * len(valid_idx),
                    "valid_length": int(lengths[i]),
                    "fallback": bool(fallbacks[i]),
                    "cra_predicted_answer": int(cra_answer[i]),
                    "mil_predicted_answer": int(mil_answer[i]),
                    "mil_answer_scores": mil_scores[i],
                }

    @classmethod
    def _m1_bank_rank_audit(cls, metadata, gt, segments):
        """Match M0's midpoint audit, not the legacy sample-endpoint oracle."""
        totals = collections.Counter()
        for qid, meta in metadata.items():
            if meta["fallback"]:
                totals["Fallback"] += 1
                continue
            try:
                vid = cls._r1_key_video_id(qid)
                info = gt[vid]
                target = np.asarray(info["location"][str(qid).split("_", 1)[1]], dtype=np.float64).reshape(-1, 2)
                times = np.asarray(segments[vid], dtype=np.float64).reshape(-1)
                duration = float(info.get("duration", info.get("duration_s")))
            except (KeyError, IndexError, ValueError, TypeError):
                totals["Unresolved"] += 1
                continue
            length = meta["valid_length"]
            if (times.size < length or target.size == 0 or not np.isfinite(target).all()
                    or not np.isfinite(duration) or np.any(target[:, 1] <= target[:, 0])):
                totals["Unresolved"] += 1
                continue
            times = times[:length]
            if (not np.isfinite(times).all() or times[0] < 0 or times[-1] > duration
                    or np.any(np.diff(times) <= 0)):
                totals["Unresolved"] += 1
                continue
            edges = np.r_[0.0, (times[:-1] + times[1:]) / 2.0, duration]
            starts, ends = edges[np.asarray(meta["start"])], edges[np.asarray(meta["end_exclusive"])]
            inter = np.maximum(0.0, np.minimum(ends[:, None], target[:, 1]) - np.maximum(starts[:, None], target[:, 0]))
            union = ends[:, None] - starts[:, None] + target[:, 1] - target[:, 0] - inter
            iou = (inter / np.maximum(union, 1e-12)).max(axis=1)
            good = iou >= 0.3
            totals["Resolved"] += 1
            totals["OracleIoU03"] += int(good.any())
            totals["OracleIoU05"] += int((iou >= 0.5).any())
            for name, key in (("Question", "prior_question"), ("CRA", "posterior_cra_answer"), ("MIL", "posterior_mil_answer")):
                p = np.asarray(meta[key])
                top_idx = int(np.argmax(p))
                top_ties = np.isclose(p, p[top_idx], rtol=1e-7, atol=1e-12)
                if not (good[top_ties].any() and (~good[top_ties]).any()):
                    totals[f"{name}Top1Resolved"] += 1
                    totals[f"{name}Top1IoU03"] += int(good[top_idx])
                totals[f"{name}GTMass03"] += float(p[good].sum())
                if not good.any():
                    continue
                best_good = p[good].max()
                totals[f"{name}Rank"] += 1 + int(((p > best_good) & ~np.isclose(p, best_good, rtol=1e-7, atol=1e-12)).sum())
        n = totals["Resolved"]
        oracle_n = totals["OracleIoU03"]
        out = {
            "V38M1BankResolvedCount": float(n),
            "V38M1BankUnresolvedCount": float(totals["Unresolved"]),
            "V38M1BankFallbackCount": float(totals["Fallback"]),
        }
        if n:
            out["V38M1BankOracleIoU03"] = oracle_n / n
            out["V38M1BankOracleIoU05"] = totals["OracleIoU05"] / n
        if n:
            for name in ("Question", "CRA", "MIL"):
                top_n = totals[f"{name}Top1Resolved"]
                if top_n:
                    out[f"V38M1Bank{name}Top1IoU03"] = totals[f"{name}Top1IoU03"] / top_n
                out[f"V38M1Bank{name}Top1ResolvedCount"] = float(top_n)
                if oracle_n:
                    out[f"V38M1Bank{name}GTRankMean"] = totals[f"{name}Rank"] / oracle_n
                out[f"V38M1Bank{name}GTMass03"] = totals[f"{name}GTMass03"] / n
        return out

    def _m1_finish_eval(self, state, score, count, results, old_spans, gt, segments, questions):
        spans_by_view = {"old_support": old_spans}
        for view, temporal in state["temporal"].items():
            if set(temporal) != set(results):
                raise RuntimeError(f"M1 {view} does not cover the same questions as CRA.")
            spans = self.metric.generate_ground(temporal, segments, questions)
            if set(spans) != set(results):
                raise RuntimeError(f"M1 {view} decoder did not return all evaluation spans.")
            spans_by_view[view] = spans
            # Use CRA's answers for every WHERE view. MIL's own QA accuracy
            # is logged separately, never used for checkpoint selection.
            score[f"v38_m1_{view}_post-hoc"] = self.metric(spans, gt, results, gs=False)
            prefix = "V38M1" + view.title().replace("_", "")
            score.update(self._r1_span_diagnostics(spans, prefix))
            if callable(getattr(self.metric, "temporal_key_metrics", None)):
                score.update(self.metric.temporal_key_metrics(temporal, gt, segments, prefix=prefix))
            if state["tail_count"]:
                score[f"v38_m1_{view}_tail_mass_nonfallback"] = state["tail_sum"][view] / state["tail_count"]
        score[self.M1_MONITOR] = float(score["v38_m1_cra_answer_conditional_post-hoc"]["Acc&GQA"])
        score["v38_m1_mil_acc"] = state["mil_correct"] / count
        score["v38_m1_qa_agreement"] = state["mil_agreement"] / count
        score["v38_m1_bank_fallback_fraction"] = state["fallback"] / count
        if state["metadata"]:
            score.update(self._m1_bank_rank_audit(state["metadata"], gt, segments))
        return spans_by_view

    def _configure_m2_mbr_eval(self):
        options = self._qbound_config().get("m2_mbr", {}) if self._qbound_enabled() else {}
        if not isinstance(options, dict):
            raise TypeError("model.qbound.m2_mbr must be a mapping.")
        defaults = {"enabled": False, "references": True, "compare_steps": True, "learned": False}
        unknown = set(options) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown model.qbound.m2_mbr options: {sorted(unknown)}")
        cfg = {**defaults, **options}
        for key, value in cfg.items():
            if type(value) is not bool:
                raise TypeError(f"model.qbound.m2_mbr.{key} must be a boolean.")
        if not cfg["enabled"]:
            return cfg, None
        if not self.v38_m1_enabled or str(self.cfgs["dataset"]["name"]).lower() != "nextgqa":
            raise ValueError("M2 MBR evaluation requires NExT-GQA and the 3.8-m1 backbone.")
        if not (cfg["references"] or cfg["learned"]):
            raise ValueError("M2 MBR requires at least one evaluation arm.")
        # Deferred import: old configurations do not need the new companion.
        from . import m2_mbr
        return cfg, m2_mbr

    def _m2_preflight_full_splits(self):
        """Cheap fail-before-training audit of native Val/Test timestamp metadata.

        This checks the REAL sample centers from frame2time. It never builds
        replacement timestamps or changes the 32-feature ordering. A decoder
        may still reject noncanonical feature masks at inference; those cases
        are diagnosed by the per-question full-coverage guard.
        """
        if not self._m2_mbr_config["references"]:
            return
        root = self.cfgs["dataset"]["csv_path"]
        output_dir = os.path.join(self.save_dir, "coverage_audits")
        os.makedirs(output_dir, exist_ok=True)
        problems = {}
        for split in ("val", "test"):
            questions = load_file(os.path.join(root, f"{split}.csv"))
            gt = load_file(os.path.join(root, f"gsub_{split}.json"))
            times = load_file(os.path.join(root, f"frame2time_{split}.json"))
            expected = self._m2_split_ids(questions, gt, split)
            if not isinstance(times, dict):
                raise TypeError(f"frame2time_{split}.json must be a JSON dictionary.")
            rows = []
            for vid, frame in questions.groupby("video_id", sort=False):
                key = str(vid)
                status, t = self._m2_mbr.timestamp_status(times.get(key))
                if t is None:
                    rows.append({"video_id": key, "status": status,
                                 "n_questions": len(frame),
                                 "question_ids": [f"{key}_{qid}" for qid in frame["qid"]]})
            report = {
                "split": split, "n_expected": len(expected),
                "n_videos": int(questions["video_id"].nunique()),
                "n_bad_videos": len(rows),
                "n_questions_blocked_by_timestamps": sum(r["n_questions"] for r in rows),
                "bad_videos": rows,
                "policy": "no missing-value interpolation, sorting, or guessed timestamps",
            }
            path = os.path.join(output_dir, f"m2_preflight_{split}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle, ensure_ascii=False, indent=2)
            print(f"[M2 timestamp preflight] {split}: {len(expected)} question IDs; "
                  f"bad_videos={len(rows)}, audit={path}", flush=True)
            if rows:
                problems[split] = path
        if problems:
            raise RuntimeError(
                "M2 cannot certify full native Val/Test before training: "
                f"missing/invalid REAL sampled-frame timestamps; audits={problems}. "
                "Repair frame2time mappings from the actual feature sampler, "
                "not by using a synthetic 32-point linspace."
            )

    def _m2_mbr_collect_batch(self, state, model_outputs, qids, prediction):
        """Collect only inference inputs. QA labels and timestamps do not enter here."""
        mbr = self._m2_mbr
        cfg = self._m2_mbr_config
        bank, evidence = self._m1_outputs(model_outputs)
        batch_size = len(qids)
        mask = bank["bank_video_mask"].detach().cpu().numpy()
        cra_answer = model_outputs["qbound_v38_m1_cra_answer"].detach().cpu()
        if cra_answer.shape != (batch_size,) or not torch.equal(cra_answer, prediction):
            raise RuntimeError("M2 MBR requires the unchanged CRA QA predictions.")
        answers = cra_answer.numpy()
        if mask.shape != (batch_size, mbr.T) or mask.dtype.kind != "b":
            raise ValueError("M2 MBR expects a boolean [B,32] M1 video mask.")

        standard = mask.all(axis=1)
        if cfg["references"]:
            start = bank["event_start"].detach().cpu().numpy()
            end = bank["event_end_exclusive"].detach().cpu().numpy()
            valid = bank["event_valid_mask"].detach().cpu().numpy()
            q = evidence["event_question_in_scores"].detach().cpu().numpy()
            a = evidence["event_answer_in_scores"].detach().cpu().numpy()
            if (
                start.shape != end.shape or start.shape != valid.shape
                or start.ndim != 2 or start.shape[0] != batch_size
                or valid.dtype.kind != "b" or start.dtype.kind not in "iu" or end.dtype.kind not in "iu"
                or q.shape != start.shape or a.shape != (*start.shape, 5)
            ):
                raise ValueError("M2 reference inputs do not match the M1 [B,W,6] inside-score contract.")
        if cfg["learned"]:
            logits_t = model_outputs.get("qbound_v38_m2_span_logits")
            bounds_t = model_outputs.get("qbound_v38_m2_span_bounds")
            if not torch.is_tensor(logits_t) or not torch.is_tensor(bounds_t):
                raise RuntimeError(
                    "M2 learned evaluation is not wired yet: CRA must emit "
                    "qbound_v38_m2_span_logits and qbound_v38_m2_span_bounds. "
                    "Use m2_mbr.learned=false for the C2/C4 reference run."
                )
            logits = logits_t.detach().cpu()
            if logits.dtype in (torch.float16, torch.bfloat16):
                logits = logits.float()
            logits = logits.numpy()
            bounds = bounds_t.detach().cpu().numpy()
            # Shared geometry for all batch rows. No candidate filtering by the scorer.
            if (
                logits.shape != (batch_size, mbr.N_SPANS)
                or bounds.dtype.kind not in "iu"
                or not np.array_equal(bounds, mbr.SPAN_BOUNDS)
            ):
                raise ValueError("M2 scorer must emit [B,496] logits and the canonical [496,2] bounds.")

        for i, raw_kid in enumerate(qids):
            kid = str(raw_kid)
            if kid in state["rows"] or kid in state["excluded"]:
                raise ValueError(f"Duplicate M2 MBR question key: {kid}")
            if not standard[i]:
                # Never silently fill missing M2 results with legacy predictions.
                state["excluded"][kid] = "nonstandard_video_length"
                continue
            row = {"cra_answer": int(answers[i])}
            if cfg["references"]:
                active = valid[i]
                bb = np.column_stack((start[i, active], end[i, active]))
                if not np.array_equal(bb, mbr.BANK_BOUNDS):
                    state["excluded"][kid] = "noncanonical_m1_bank"
                    continue
                inside = np.column_stack((q[i, active], a[i, active]))
                if not np.isfinite(inside).all():
                    raise ValueError(f"Non-finite M1 reference scores at {kid}.")
                row["inside"] = inside
            if cfg["learned"]:
                if not np.isfinite(logits[i]).all():
                    raise ValueError(f"Non-finite M2 span logits at {kid}.")
                row["logits"] = logits[i].copy()
            state["rows"][kid] = row

    def _m2_mbr_decode(self, state, segments, questions):
        """Decode without reading GT or QA labels. Group seconds utilities by video."""
        mbr = self._m2_mbr
        cfg = self._m2_mbr_config
        skipped = dict(state["excluded"])
        names = (["c2", "c4"] if cfg["references"] else []) + (["learned"] if cfg["learned"] else [])
        modes = ("seconds", "steps") if cfg["compare_steps"] else ("seconds",)
        views = {f"{name}_{mode}": {} for name in names for mode in modes}
        detail = {view: {} for view in views}
        question_text = {}
        if cfg["references"]:
            required = {"video_id", "qid", "question"}
            if not required.issubset(questions.columns):
                raise ValueError(f"M2 C2/C4 require CSV columns: {sorted(required)}.")
            for vid, qid, question in questions[["video_id", "qid", "question"]].itertuples(
                index=False, name=None,
            ):
                kid = f"{vid}_{qid}"
                if kid in question_text:
                    raise ValueError(f"Duplicate M2 CSV question key: {kid}")
                question_text[kid] = question

        accepted = []
        video_times = {}
        time_status = {}
        for kid in sorted(state["rows"]):
            vid = self._r1_key_video_id(kid)
            if vid not in video_times:
                status, t = mbr.timestamp_status(segments.get(vid))
                time_status[vid] = status
                video_times[vid] = t
            if video_times[vid] is None:
                skipped[kid] = "timestamp_" + time_status[vid]
                continue
            q = question_text.get(kid)
            if cfg["references"] and (not isinstance(q, str) or not q.strip()):
                skipped[kid] = "missing_question_text"
                continue
            accepted.append((kid, vid))
        if not accepted:
            return views, detail, skipped

        rows = [state["rows"][kid] for kid, _ in accepted]
        p = {}
        if cfg["references"]:
            p.update(mbr.locked_reference_probabilities(
                np.stack([row["inside"] for row in rows]),
                np.asarray([row["cra_answer"] for row in rows], dtype=np.int64),
                [question_text[kid] for kid, _ in accepted],
            ))
        if cfg["learned"]:
            p["learned"] = mbr.interval_probabilities(np.stack([row["logits"] for row in rows]))

        def record(view, indices, bounds, utilities):
            for k, span, utility in zip(indices, bounds, utilities):
                kid, vid = accepted[k]
                s, e = map(int, span)
                ts = video_times[vid]
                seconds = [float(ts[s]), float(ts[e - 1])]
                views[view][kid] = seconds
                detail[view][kid] = {
                    "s_step": s, "e_step_exclusive": e, "span_seconds": seconds,
                    "expected_utility": float(utility), "cra_answer": rows[k]["cra_answer"],
                }

        video_rows = collections.defaultdict(list)
        for i, (_, vid) in enumerate(accepted):
            video_rows[vid].append(i)
        if cfg["compare_steps"]:
            # Keep the batch arithmetic/layout of the locked replay on the step arm.
            for name, probabilities in p.items():
                record(f"{name}_steps", range(len(accepted)), *mbr.select(probabilities))
        for vid, indices in video_rows.items():
            times = video_times[vid]
            together = np.concatenate([p[name][indices] for name in names], axis=0)
            bounds, utilities = mbr.select(together, timestamps=times)
            for i, name in enumerate(names):
                part = slice(i * len(indices), (i + 1) * len(indices))
                record(f"{name}_seconds", indices, bounds[part], utilities[part])
        return views, detail, skipped

    @staticmethod
    def _m2_split_ids(questions, gt, split):
        """Compare native CSV question IDs with native GT annotations.

        Do not infer the benchmark denominator from the subset decoded by MBR.
        This guard is M2-only: old CRA/M1 evaluators remain unchanged.
        """
        if not {"video_id", "qid", "question"}.issubset(questions.columns):
            raise ValueError(f"{split} CSV lacks video_id/qid/question columns.")
        expected = [f"{vid}_{qid}" for vid, qid in
                    questions[["video_id", "qid"]].itertuples(index=False, name=None)]
        gt_ids = [f"{vid}_{qid}" for vid, anno in gt.items()
                  for qid in anno["location"]]
        if len(expected) != len(set(expected)) or len(gt_ids) != len(set(gt_ids)):
            raise RuntimeError(f"M2 {split}: duplicate IDs in native split CSV or GT.")
        reference_n = {"val": 3358, "test": 5553}.get(split)
        if reference_n is not None and len(expected) != reference_n:
            raise RuntimeError(
                f"M2 {split}: split CSV has {len(expected)} questions, expected "
                f"{reference_n} for the native NExT-GQA split. No score exported."
            )
        if set(expected) != set(gt_ids):
            raise RuntimeError(
                f"M2 {split}: CSV/GT ID mismatch: "
                f"missing_GT={sorted(set(expected) - set(gt_ids))[:8]}, "
                f"GT_only={sorted(set(gt_ids) - set(expected))[:8]}."
            )
        return set(expected)

    def _m2_coverage_audit(self, *, split, expected, results, views, skipped):
        """Write the failure reason to disk BEFORE refusing partial benchmark metrics."""
        qa_ids = set(results)
        states = {}
        for view, predictions in views.items():
            observed = set(predictions)
            states[view] = {
                "n_predicted": len(observed),
                "n_missing": len(expected - observed),
                "n_extra": len(observed - expected),
                "missing_examples": sorted(expected - observed)[:50],
                "extra_examples": sorted(observed - expected)[:50],
            }
        report = {
            "split": split,
            "experiment": "Q-Bound V3.8-M2 C4+MBR",
            "expected_n": len(expected),
            "qa_n": len(results),
            "qa_missing_n": len(expected - qa_ids),
            "qa_extra_n": len(qa_ids - expected),
            "qa_missing_examples": sorted(expected - qa_ids)[:50],
            "qa_extra_examples": sorted(qa_ids - expected)[:50],
            "decoded": states,
            "excluded_by_reason": dict(sorted(collections.Counter(skipped.values()).items())),
            "excluded_by_qid": dict(sorted(skipped.items())),
            "checkpoint_selection": "validation only",
            "complete": (qa_ids == expected and all(
                set(predictions) == expected for predictions in views.values())),
        }
        output_dir = os.path.join(self.save_dir, "coverage_audits")
        os.makedirs(output_dir, exist_ok=True)
        epoch = getattr(self, "_m2_active_eval_epoch", None)
        suffix = f"epoch_{int(epoch):02d}" if epoch is not None else "inference"
        path = os.path.join(output_dir, f"m2_{split}_{suffix}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"[M2 full-split coverage] {split}: QA={len(results)}/{len(expected)}, "
              f"span_views={{{', '.join(f'{name}: {len(pred)}' for name, pred in views.items())}}}; "
              f"audit={path}", flush=True)
        if not report["complete"]:
            raise RuntimeError(
                f"M2 {split}: incomplete predictions. Full-split benchmark metrics "
                f"WERE NOT computed or logged to W&B. Audit: {path}. "
                f"Reasons: {report['excluded_by_reason']}"
            )
        return report

    def _m2_mbr_finish_eval(self, state, score, results, gt, segments, questions, split):
        """Score C4 only when QA AND every WHERE arm cover the FULL native split.

        No partial-denominator Acc&GQA is ever emitted as an M2 benchmark.
        Timestamp GT is evaluator-only; MBR never sees it during decoding.
        """
        spans, detail, skipped = self._m2_mbr_decode(state, segments, questions)
        expected = self._m2_split_ids(questions, gt, split)
        report = self._m2_coverage_audit(
            split=split, expected=expected, results=results,
            views=spans, skipped=skipped,
        )
        received = len(state["rows"]) + len(state["excluded"])
        if received != len(expected):
            raise RuntimeError(f"M2 {split}: collected {received} rows vs {len(expected)} IDs.")
        base = {"n_expected": len(expected), "n_received": received,
                "n_qa": len(results), "n_decoded": len(expected),
                "n_scored": len(expected), "full_split": 1,
                "n_excluded_before_scoring": 0,
                "n_decoded_without_gt_row": 0}
        for view, predictions in spans.items():
            cov_hits = 0
            for vid, annotation in gt.items():
                for qid, alternatives in annotation["location"].items():
                    kid = f"{vid}_{qid}"
                    row = detail[view][kid]
                    qa = results[kid]
                    if qa["prediction"] != row["cra_answer"]:
                        raise RuntimeError(f"M2 changed CRA answer at {kid}.")
                    matched = False
                    for loc in alternatives:
                        iou, iop = self.metric.get_tIoU(loc, predictions[kid])
                        if iou >= 0.30 and iop >= 0.50:
                            matched = True
                            break
                    hit = int(matched and qa["prediction"] == qa["answer"])
                    cov_hits += hit
                    row["gqa_cov_hit"] = hit
            official = dict(self.metric(predictions, gt, results, gs=False))
            official["GQA-cov"] = 100.0 * cov_hits / len(expected)
            official["n_expected"] = len(expected)
            official["n_scored"] = len(expected)
            official["n_decoded"] = len(expected)
            score[f"v38_m2_mbr_{view}"] = official
        score["v38_m2_mbr_coverage"] = base
        return {
            "v38_m2_mbr_spans": spans,
            "v38_m2_mbr_diagnostics": detail,
            "v38_m2_mbr_excluded": skipped,
            "v38_m2_mbr_qa_prediction_source": "cra",
            "v38_m2_mbr_full_split": bool(report["complete"]),
        }

    def eval(
        self,
        data_loader,
        epoch=None,
        save=False,
    ):
        del epoch

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
        selected_frame_bounds = {}
        learned_edge_by_qid = {}
        used_edge_by_qid = {}
        edge_surprise_by_qid = {}

        # M0: independent answer and grounding views, NEVER replacing old CRA.
        v38_view_names = (
            "question_only", "joint_marginal", "predicted_answer_conditional"
        )
        v38_temporal_by_view = {name: {} for name in v38_view_names}
        v38_evidence_results = {}
        v38_event_metadata = {}  # Included in validation dumps when save=True.
        v38_correct = 0
        v38_agreement = 0
        v38_fallback = 0
        v38_eligible = 0
        m1_state = None
        if self.v38_m1_enabled:
            m1_state = {
                "temporal": {view: {} for view in self.M1_VIEWS}, "metadata": {},
                "mil_correct": 0, "mil_agreement": 0, "fallback": 0,
                "tail_count": 0, "tail_sum": collections.Counter(),
            }

        m2_state = {"rows": {}, "excluded": {}} if self._m2_mbr is not None else None
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
                if m1_state is not None:
                    self._m1_collect_eval_batch(
                        m1_state, model_outputs, question_id, predicted, labels_cpu,
                        dump=bool(save and data_loader.dataset.split == "val"),
                    )

                if m2_state is not None:
                    self._m2_mbr_collect_batch(m2_state, model_outputs, question_id, predicted)

                if self.v38_minimal_enabled:
                    self._v38_outputs(model_outputs)
                    v38_logits = model_outputs["qbound_v38_ev_qa_logits"]
                    v38_predicted = v38_logits.argmax(dim=-1).detach().cpu()
                    v38_correct += int((v38_predicted == labels_cpu).sum().item())
                    v38_agreement += int((v38_predicted == predicted).sum().item())
                    v38_bank = model_outputs["qbound_v38_bank"]
                    v38_evidence = model_outputs["qbound_v38_evidence"]
                    v38_fallback += int(v38_bank["bank_fallback"].sum().item())
                    v38_eligible += int(v38_bank["bank_train_eligible"].sum().item())
                    v38_arrays = {
                        "question_only": model_outputs["qbound_v38_event_temporal_question"].detach().cpu().numpy(),
                        "joint_marginal": model_outputs["qbound_v38_event_temporal_marginal"].detach().cpu().numpy(),
                        "predicted_answer_conditional": model_outputs[
                            "qbound_v38_event_temporal_conditional_pred"
                        ].detach().cpu().numpy(),
                    }
                    # Store compact posterior/window metadata only for validation
                    # dumps (not for test or gradient computation).
                    if save:
                        v38_meta_arrays = {
                            "valid_mask": v38_bank["event_valid_mask"].detach().cpu().numpy(),
                            "start": v38_bank["event_start"].detach().cpu().numpy(),
                            "end_exclusive": v38_bank["event_end_exclusive"].detach().cpu().numpy(),
                            "width": v38_bank["event_width"].detach().cpu().numpy(),
                            "bank_prior": v38_bank["bank_prior_weights"].detach().cpu().numpy(),
                            "prior_question": v38_evidence["event_posterior_question"].detach().cpu().numpy(),
                            "posterior_marginal": v38_evidence["event_posterior_marginal"].detach().cpu().numpy(),
                            "posterior_conditional": v38_evidence[
                                "event_posterior_conditional_pred"
                            ].detach().cpu().numpy(),
                            "answer_posterior": v38_evidence["event_qa_posterior"].detach().cpu().numpy(),
                            "coverage_bias": v38_bank["bank_prior_temporal_bias"].detach().cpu().numpy(),
                            "fallback": v38_bank["bank_fallback"].detach().cpu().numpy(),
                            "valid_length": v38_bank["bank_video_lengths"].detach().cpu().numpy(),
                        }
                        if self.v38_settings["dump_joint_matrices"]:
                            v38_meta_arrays["joint_posterior"] = (
                                v38_evidence["event_joint_posterior"].detach().cpu().numpy()
                            )
                            v38_meta_arrays["interaction_centered"] = (
                                v38_evidence["event_interaction_centered"].detach().cpu().numpy()
                            )
                    for sample_index, qid in enumerate(question_id):
                        v38_evidence_results[qid] = {
                            "prediction": int(v38_predicted[sample_index].item()),
                            "answer": int(labels_cpu[sample_index].item()),
                        }
                        for view_name, array in v38_arrays.items():
                            v38_temporal_by_view[view_name][qid] = array[sample_index].tolist()
                        if save:
                            valid_count = int(v38_meta_arrays["valid_mask"][sample_index].sum())
                            v38_event_metadata[qid] = {
                                name: (value[sample_index, :valid_count].tolist()
                                       if name in ("valid_mask", "start", "end_exclusive",
                                                   "width", "bank_prior", "prior_question",
                                                   "posterior_marginal", "posterior_conditional",
                                                   "joint_posterior", "interaction_centered")
                                       else value[sample_index].tolist())
                                for name, value in v38_meta_arrays.items()
                            }

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
                batch_frame_bounds = None
                batch_learned_edge = None
                batch_used_edge = None
                batch_edge_surprise = None

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
                    # Preserve both learned edges and edges actually passed to
                    # propagation. This distinguishes graph learning from smoothing.
                    learned_edge = qbound_aux.get("support_edge_affinity_raw")
                    used_edge = qbound_aux.get("support_edge_affinity")
                    surprise = qbound_aux.get("support_edge_surprise")
                    if self.r2_predictor_enabled and torch.is_tensor(learned_edge):
                        batch_learned_edge = learned_edge.detach().cpu().numpy()
                    if torch.is_tensor(used_edge):
                        batch_used_edge = used_edge.detach().cpu().numpy()
                    if self.r2_predictor_enabled and torch.is_tensor(surprise):
                        batch_edge_surprise = surprise.detach().cpu().numpy()
                    start_idx = qbound_aux.get("selected_span_start_frame")
                    end_idx = qbound_aux.get("selected_span_end_frame")
                    if torch.is_tensor(start_idx) and torch.is_tensor(end_idx):
                        batch_frame_bounds = np.column_stack((
                            start_idx.detach().cpu().numpy(),
                            end_idx.detach().cpu().numpy(),
                        ))
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

                # The original CRA post-hoc decoder must see the same Support
                # Field used for ground/QA. Any future divergence invalidates R1/R2.
                if self._qbound_enabled() and batch_support_key is not None:
                    if not np.allclose(
                        video_attention, batch_support_key, atol=1e-5, rtol=1e-5
                    ):
                        raise RuntimeError(
                            "R2 protocol broken: model_outputs['fatt'] differs "
                            "from the Q-Bound Support Field used by QA."
                        )
                predicted_np = predicted.numpy()
                labels_np = labels_cpu.numpy()
                for batch_index, qid in enumerate(question_id):
                    if qid in results:
                        raise RuntimeError(f"Duplicate evaluation question ID: {qid}")
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
                    if batch_frame_bounds is not None:
                        selected_frame_bounds[qid] = [
                            int(batch_frame_bounds[batch_index][0]),
                            int(batch_frame_bounds[batch_index][1]),
                        ]
                    if batch_learned_edge is not None:
                        learned_edge_by_qid[qid] = batch_learned_edge[batch_index].tolist()
                    if batch_used_edge is not None:
                        used_edge_by_qid[qid] = batch_used_edge[batch_index].tolist()
                    if batch_edge_surprise is not None:
                        edge_surprise_by_qid[qid] = batch_edge_surprise[batch_index].tolist()

                progress = float(index + 1) / len(data_loader)
                print(f"\rProgress: {progress:.2%}", end="", flush=True)

        if count == 0:
            raise RuntimeError("Evaluation data loader is empty.")
        score["acc"] /= count
        if self.v38_minimal_enabled:
            score["v38_evqa_acc"] = v38_correct / count
            score["v38_qa_agreement"] = v38_agreement / count
            score["v38_bank_fallback_fraction"] = v38_fallback / count
            score["v38_event_train_eligible_fraction"] = v38_eligible / count
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
        if self._qbound_enabled():
            # R1 diagnostics ONLY. The canonical post-hoc decoder above is
            # unchanged and remains the sole checkpoint-selection reference.
            post_hoc_leftfix = {}
            for qid in post_hoc_ground:
                vid = self._r1_key_video_id(qid)
                post_hoc_leftfix[qid] = self._r1_posthoc_frame0_fixed(
                    ground_res[qid], segments[vid]
                )
            score["post-hoc-leftfix"] = self.metric(
                post_hoc_leftfix, ground_truth, results, gs=False
            )
            score.update(self._r1_span_diagnostics(
                post_hoc_ground, "QBoundPosthoc"
            ))

            seed_post_hoc = self.metric.generate_ground(
                qbound_seed_keyframe, segments, questions
            )
            score["seed-post-hoc"] = self.metric(
                seed_post_hoc, ground_truth, results, gs=False
            )
            score.update(self._r1_span_diagnostics(
                seed_post_hoc, "QBoundSeedPosthoc"
            ))
        else:
            post_hoc_leftfix = {}
            seed_post_hoc = {}

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

        v38_posthoc_spans = {}
        if self.v38_minimal_enabled:
            for view_name, temporal_by_qid in v38_temporal_by_view.items():
                if set(temporal_by_qid) != set(results):
                    raise RuntimeError(f"M0 {view_name} is missing validation/test examples.")
                # EXACT same external R1 generate_ground implementation, same
                # timestamps/queries as old-support post_hoc_ground.
                event_spans = self.metric.generate_ground(
                    temporal_by_qid, segments, questions
                )
                v38_posthoc_spans[view_name] = event_spans
                score[f"v38_{view_name}_post-hoc"] = self.metric(
                    event_spans, ground_truth, v38_evidence_results, gs=False
                )
                prefix = {
                    "question_only": "QBoundV38Question",
                    "joint_marginal": "QBoundV38Marginal",
                    "predicted_answer_conditional": "QBoundV38Conditional",
                }[view_name]
                if hasattr(self.metric, "temporal_key_metrics"):
                    score.update(self.metric.temporal_key_metrics(
                        temporal_by_qid, ground_truth, segments, prefix=prefix
                    ))
                score.update(self._r1_span_diagnostics(
                    event_spans, f"V38{view_name.title().replace('_', '')}R1"
                ))
                score.update(self._v38_span_failure_audit(
                    event_spans, ground_truth, v38_evidence_results,
                    f"V38{view_name.title().replace('_', '')}"
                ))
            # Isolate WHERE improvements from the effect of a different QA head.
            score["v38_marginal_craqa_post-hoc"] = self.metric(
                v38_posthoc_spans["joint_marginal"], ground_truth,
                results, gs=False
            )
            if save:
                score.update(self._v38_oracle_bank_audit(
                    v38_event_metadata, ground_truth, segments
                ))

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
            "ph_r1_leftfix": post_hoc_leftfix,
            "ph_seed": seed_post_hoc,
            "qbound_edge_affinity_learned": learned_edge_by_qid,
            "qbound_edge_affinity_used": used_edge_by_qid,
            "qbound_edge_surprise": edge_surprise_by_qid,
            "qbound_graph_mode": self.r2_graph_mode,
        }

        if m1_state is not None:
            spans = self._m1_finish_eval(
                m1_state, score, count, results, post_hoc_ground,
                ground_truth, segments, questions,
            )
            result.update({
                "v38_m1_temporal_by_view": {"old_support": ground_res, **m1_state["temporal"]},
                "v38_m1_posthoc_spans": spans,
                "v38_m1_event_metadata": m1_state["metadata"],
                "v38_m1_selected_grounding_view": "cra_answer_conditional",
                "v38_m1_qa_prediction_source": "cra",
            })

        if m2_state is not None:
            result.update(self._m2_mbr_finish_eval(
                m2_state, score, results, ground_truth, segments, questions, split,
            ))

        if self.v38_minimal_enabled:
            result.update({
                "v38_evidence_answer_results": v38_evidence_results,
                "v38_temporal_by_view": v38_temporal_by_view,
                "v38_posthoc_spans": v38_posthoc_spans,
                "v38_event_metadata": v38_event_metadata if save else {},
                "v38_experiment_version": "3.8-minimal",
                "v38_training_phase": self.v38_settings["phase"],
                "v38_selected_grounding_posterior": "joint_marginal",
                "v38_qa_prediction_source": "evidence-only",
                "v38_legacy_cra_results_preserved": True,
            })

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

            # R1 timestamp control: use SAME rounded barycenter frame indices
            # that generated tg, mapped to actual sampled-frame timestamps.
            # Do not change the canonical normalized-time pre-hoc metric.
            if self._qbound_enabled():
                ground_pre_timestamp = {}
                for qid, bounds in selected_frame_bounds.items():
                    vid = self._r1_key_video_id(qid)
                    times = np.asarray(segments[vid], dtype=np.float64).reshape(-1)
                    support_size = len(qbound_support_keyframe[qid])
                    if times.size != support_size:
                        raise ValueError(
                            f"R1 timestamps vs Support Field mismatch at {qid}: "
                            f"{times.size} timestamps, {support_size} frames."
                        )
                    start_idx, end_idx = bounds
                    if not (0 <= start_idx <= end_idx < times.size):
                        raise RuntimeError(f"Invalid barycenter indices for {qid}.")
                    ground_pre_timestamp[qid] = [
                        float(times[start_idx]), float(times[end_idx])
                    ]
                if len(ground_pre_timestamp) != len(ground_pre):
                    raise RuntimeError(
                        "R1 timestamp diagnostic lacks some pre-hoc predictions."
                    )
                score["pre-hoc-timestamp"] = self.metric(
                    ground_pre_timestamp, ground_truth, results, gs=False
                )
                result["tg_timestamp"] = ground_pre_timestamp

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

        self._m2_write_full_scores(val_score, test_score, epoch=None)
        self.print_score(score)
        # Frozen checkpoint evaluation must also log the EXACT C4+MBR metrics
        # as the training path, rather than leaving an empty W&B inference run.
        wandb.log({"epoch": int(epoch) if epoch is not None else 0,
                   "experiment/mode_infer": 1,
                   **self._m2_flatten_eval_metrics({"val": val_score,
                                                    "test": test_score})})
        return score
