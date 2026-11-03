# Base Pipeline
import json
import math
import os
import shutil
import time

import pandas as pd
import torch
from numpy import inf

from models import model_fns
from utils import *
from utils.misc.misc import compute_a2v


class BasePipeline(object):
    """Shared training lifecycle with a strict Q-Bound V2.8 contract."""

    QBOUND_VERSION = "2.8.0"

    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs

        # V3.7 preserves per-epoch test observation on WandB. Test results
        # must NEVER be used for checkpoint selection or tuning.
        stat_cfg = cfgs.get("stat")
        if not isinstance(stat_cfg, dict):
            raise TypeError("stat must be a mapping/dict.")
        if self._qbound_enabled():
            if stat_cfg.get("eval_test_each_epoch", True) is not True:
                raise ValueError(
                    "Q-Bound requires stat.eval_test_each_epoch=true: "
                    "test evaluation/WandB logging may not be disabled."
                )
            stat_cfg.setdefault("eval_test_each_epoch", True)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            cfgs["misc"].get("cuda", "0")
        )
        os.environ["CUDA_LAUNCH_BLOCKING"] = (
            "1"
            if cfgs["misc"].get("cuda_launch_blocking", False)
            else "0"
        )

        self.tokenizer = build_tokenizer(cfgs)
        self.a2id, self.id2a, self.a2v = None, None, None
        if not cfgs["dataset"]["mc"]:
            self.a2id, self.id2a, self.a2v = compute_a2v(
                vocab_path=cfgs["dataset"]["vocab_path"],
                bert_tokenizer=self.tokenizer,
                amax_words=cfgs["dataset"]["amax_words"],
            )

        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = build_dataloaders(
            cfgs,
            self.a2id,
            self.tokenizer,
        )

        self.model = model_fns[cfgs["model"]["name"]](
            cfgs,
            self.tokenizer,
        ).cuda()
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.metric = Metric(cfgs)
        self.criterion = BuildLossFunc(cfgs)

        # The repository optimizer is built first. Q-Bound parameters are then
        # moved into one dedicated group before the scheduler is constructed.
        self.optimizer = build_optimizer(cfgs, self.model)
        self._configure_qbound_optimizer_groups()
        self._check_qbound_optimizer()
        self._check_qbound_components()
        self._check_optimizer_coverage()

        self.lr_scheduler = build_lr_scheduler(
            cfgs,
            self.optimizer,
            len(self.train_dataloader),
        )
        self._check_qbound_scheduler()

        self.epochs = int(self.cfgs["optim"]["epochs"])
        self.save_period = int(self.cfgs["optim"]["save_period"])
        if self.epochs < 1:
            raise ValueError("optim.epochs must be positive.")
        if self.save_period < 1:
            raise ValueError("optim.save_period must be positive.")

        # Run the same 15-epoch LR schedule in both R2 arms, but optionally
        # pause after epoch 10.  Resume at epoch 11 with the SAME optim.epochs,
        # optimizer state and LR scheduler state; no schedule reset is needed.
        self.stop_after_epoch = int(
            self.cfgs["optim"].get("stop_after_epoch", self.epochs)
        )
        if not 1 <= self.stop_after_epoch <= self.epochs:
            raise ValueError(
                "optim.stop_after_epoch must be within [1, optim.epochs]."
            )

        # A/B use the same checkpoint policy. Full resumable checkpoints are
        # kept for the current epoch and validation-best selections. Optional
        # lightweight epoch weights are for diagnosis only, not for resuming.
        self.save_epoch_weights = self.cfgs["stat"].get(
            "save_epoch_weights", False
        )
        if not isinstance(self.save_epoch_weights, bool):
            raise TypeError("stat.save_epoch_weights must be a boolean.")

        self.mnt_mode = cfgs["stat"]["monitor"]["mode"]
        if self.mnt_mode not in {"min", "max", "off"}:
            raise ValueError("stat.monitor.mode must be min, max, or off.")

        metric_name = cfgs["stat"]["monitor"]["metric"]
        self.mnt_metric = "val_" + metric_name
        self.mnt_metric_test = "test_" + metric_name
        self.mnt_best = inf if self.mnt_mode == "min" else -inf
        self.early_stop = int(
            self.cfgs["stat"]["monitor"]["early_stop"]
        )

        self.start_epoch = 1
        self.save_dir = os.path.join(
            cfgs["stat"]["record_dir"],
            cfgs["misc"]["running_name"],
        )
        self.checkpoint_dir = os.path.join(
            self.save_dir,
            "checkpoint",
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_recorder = {
            "val": {self.mnt_metric: self.mnt_best},
            "test": {self.mnt_metric_test: self.mnt_best},
        }
        # Unlike the standard QA monitor, these are grounding-specific
        # validation selectors. The TEST split is never used to select a model.
        self.grounding_best = {
            "mIoU": {"value": -inf, "epoch": 0},
            "Acc&GQA": {"value": -inf, "epoch": 0},
        }

        # Weight initialization and training resume are separate operations.
        # Initializing from EQ-PTC E15 must NOT import its Adam moments, LR
        # scheduler position, saved best metrics, or epoch counter.
        resume_path = cfgs["stat"].get("resume")
        init_path = cfgs["stat"].get("init_from_checkpoint")
        if resume_path and init_path:
            raise ValueError(
                "stat.resume and stat.init_from_checkpoint are mutually exclusive."
            )
        if init_path:
            self._initialize_from_checkpoint(init_path)
        if resume_path:
            self._resume_checkpoint(resume_path)

        self.monitor = Monitor(cfgs)
        self.monitor.log_info(
            json.dumps(cfgs, sort_keys=True, indent=4),
            end="\n",
        )

    # ------------------------------------------------------------------
    # Q-Bound V2.8 configuration and module helpers
    # ------------------------------------------------------------------
    def _model_core(self):
        return (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

    def _qbound_config(self):
        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise TypeError("model.qbound must be a mapping/dict.")
        return qbound_cfg

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

    @staticmethod
    def _is_qbound_parameter_name(name):
        return "qbound" in name.lower()

    @staticmethod
    def _normalize_parameter_name(name):
        return name[len("module."):] if name.startswith("module.") else name

    def _named_trainable_qbound_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if (
                parameter.requires_grad
                and self._is_qbound_parameter_name(name)
            )
        ]

    @staticmethod
    def _validate_loss_block(
        block,
        name,
        allowed_keys,
        require_temperature=False,
        allow_disabled=False,
    ):
        if block is None:
            block = {}
        if not isinstance(block, dict):
            raise TypeError(f"model.qbound.{name} must be a mapping/dict.")

        unsupported = sorted(set(block).difference(allowed_keys))
        if unsupported:
            raise ValueError(
                f"model.qbound.{name} contains unsupported keys: "
                + ", ".join(unsupported)
            )

        enabled = bool(block.get("enabled", True))
        weight = float(block.get("loss_weight", 1.0))
        if allow_disabled and not enabled:
            if not math.isfinite(weight) or weight != 0.0:
                raise ValueError(f"Disabled {name} requires loss_weight=0.")
        else:
            if not enabled:
                raise ValueError(f"Canonical Q-Bound V2.8 requires {name}.enabled=true.")
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError(f"model.qbound.{name}.loss_weight must be finite and > 0.")

        result = {"enabled": enabled, "loss_weight": weight}
        if require_temperature:
            temperature = float(block.get("temperature", 0.07))
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(
                    f"model.qbound.{name}.temperature must be finite and > 0."
                )
            result["temperature"] = temperature
        return result

    def _base_v35_pair_settings(self):
        """Validate QDTG loss config, independently of model initialization.

        R2-C without this config is the default; in V3.5 the inference model
        and optimizer parameter groups are unchanged, only the training loss.
        """
        qbound = self._qbound_config()
        cfg = qbound.get("pair_loss", {})
        if cfg is None:
            cfg = {}
        if not isinstance(cfg, dict):
            raise TypeError("model.qbound.pair_loss must be a mapping/dict.")
        allowed = {"enabled", "loss_weight", "temperature", "margin",
                   "gradient_audit_every"}
        unknown = set(cfg).difference(allowed)
        if unknown:
            raise ValueError("Unsupported pair_loss config keys: " + ", ".join(sorted(unknown)))
        enabled = cfg.get("enabled", False)
        if not isinstance(enabled, bool):
            raise TypeError("pair_loss.enabled must be boolean.")
        weight = float(cfg.get("loss_weight", 0.0))
        temperature = float(cfg.get("temperature", 0.07))
        margin = float(cfg.get("margin", 0.0))
        audit_every = cfg.get("gradient_audit_every", 0)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("pair_loss.loss_weight must be finite and nonnegative.")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("pair_loss.temperature must be finite and positive.")
        if not math.isfinite(margin) or margin < 0.0:
            raise ValueError("pair_loss.margin must be finite and nonnegative.")
        if isinstance(audit_every, bool) or not isinstance(audit_every, int) or audit_every < 0:
            raise ValueError("pair_loss.gradient_audit_every must be an integer >= 0.")
        if enabled != (weight > 0.0):
            raise ValueError("pair_loss.enabled=true requires loss_weight>0; disabled requires loss_weight=0.")
        if not enabled and audit_every:
            raise ValueError("Disabled pair_loss requires gradient_audit_every=0.")
        return {"enabled": enabled, "loss_weight": weight,
                "temperature": temperature, "margin": margin,
                "gradient_audit_every": audit_every}

    def _h1_temporal_settings(self):
        """Explicit experiment identity, separate from the stable 2.8.0 ABI.

        Never silently interpret an old V3.5 YAML as the new architecture.
        The same mode/scales must be used in CRA, Refine and QBoundCore.
        """
        config = self._qbound_config()
        if "temporal_context_mode" not in config:
            raise ValueError(
                "H1 requires explicit model.qbound.temporal_context_mode: "
                "'legacy' (V3.5 control) or 'eq_ptc' (new B+C)."
            )
        mode = config["temporal_context_mode"]
        if not isinstance(mode, str) or mode not in {"legacy", "eq_ptc"}:
            raise ValueError("model.qbound.temporal_context_mode must be 'legacy' or 'eq_ptc'.")
        scales = {}
        for key in ("eqptc_early_scale", "eqptc_position_scale", "eqptc_temporal_scale"):
            value = config.get(key, 0.25)
            if isinstance(value, bool):
                raise TypeError(f"model.qbound.{key} must not be boolean.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"model.qbound.{key} must be a number in [0,1].") from exc
            if not math.isfinite(value) or not (0.0 <= value <= 1.0):
                raise ValueError(f"model.qbound.{key} must be finite and in [0,1].")
            scales[key] = value
        return {"mode": mode, "scales": scales}

    def _v37_settings(self):
        """Validate the two independent V3.7 ablation switches.

        The '2.8.0' model ABI is unchanged; experiment identity is NOT.
        Defaults preserve old checkpoints and existing V3.5 configurations.
        """
        cfg = self._qbound_config()
        event_mode = cfg.get("event_grounding_mode", "legacy")
        if not isinstance(event_mode, str) or event_mode not in (
            "legacy", "event_relative"
        ):
            raise ValueError(
                "model.qbound.event_grounding_mode must be 'legacy' "
                "or 'event_relative'."
            )
        pooling_default = "masked" if event_mode == "event_relative" else "legacy"
        pooling_mode = cfg.get("language_pooling_mode", pooling_default)
        if not isinstance(pooling_mode, str) or pooling_mode not in (
            "legacy", "masked"
        ):
            raise ValueError(
                "model.qbound.language_pooling_mode must be 'legacy' or 'masked'."
            )
        scales = {}
        for key, default in (
            ("event_residual_scale", 0.10),
            ("event_relation_temperature", 0.20),
        ):
            raw = cfg.get(key, default)
            if isinstance(raw, bool):
                raise TypeError(f"model.qbound.{key} cannot be boolean.")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"model.qbound.{key} must be numeric.") from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"model.qbound.{key} must be finite and > 0.")
            if key == "event_residual_scale" and value > 1.0:
                raise ValueError("model.qbound.event_residual_scale must be <= 1.")
            scales[key] = value
        max_length = cfg.get("event_candidate_max_length", None)
        if max_length is not None and (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError(
                "model.qbound.event_candidate_max_length must be null or int >= 1."
            )
        if event_mode == "event_relative":
            if self._h1_temporal_settings()["mode"] != "eq_ptc":
                raise ValueError("V3.7 event_relative requires temporal_context_mode=eq_ptc.")
            if (cfg.get("support_graph_mode") != "fixed"
                    or cfg.get("support_predictor_enabled", True) is not False):
                raise ValueError("V3.7 event_relative requires predictor-free R2-C fixed graph.")
        return {
            "event_grounding_mode": event_mode,
            "language_pooling_mode": pooling_mode,
            "event_residual_scale": scales["event_residual_scale"],
            "event_relation_temperature": scales["event_relation_temperature"],
            "event_candidate_max_length": max_length,
        }

    def _qbound_experiment_variant(self):
        if not self._qbound_enabled():
            return "CRA"
        mode = self._h1_temporal_settings()["mode"]
        if mode == "eq_ptc":
            variant = "H1-EQPTC-QDTG" if self._base_v35_pair_settings()["enabled"] else "H1-EQPTC"
        elif self._base_v35_pair_settings()["enabled"]:
            variant = "V3.5-QDTG"
        else:
            variant = "R2-C" if not self._qbound_config().get("support_predictor_enabled", True) else "R2-A/B"
        v37 = self._v37_settings()
        if (v37["language_pooling_mode"] == "legacy"
                and v37["event_grounding_mode"] == "legacy"):
            return variant
        return (f"V3.7-{variant}-pool_{v37['language_pooling_mode']}"
                f"-event_{v37['event_grounding_mode']}")

    # ------------------------------------------------------------------
    # Optimizer grouping
    # ------------------------------------------------------------------
    def _configure_qbound_optimizer_groups(self):
        """Place every QBoundCore parameter in one dedicated LR group."""
        if not self._qbound_enabled():
            return

        optim_cfg = self.cfgs.get("optim", {})
        if "qbound_lr" not in optim_cfg:
            raise ValueError(
                "Q-Bound V2.8 requires an explicit optim.qbound_lr."
            )
        qbound_lr = float(optim_cfg["qbound_lr"])
        if not math.isfinite(qbound_lr) or qbound_lr <= 0.0:
            raise ValueError("optim.qbound_lr must be finite and > 0.")

        named_qbound = self._named_trainable_qbound_parameters()
        if not named_qbound:
            raise RuntimeError(
                "Q-Bound V2.8 is enabled but no trainable Q-Bound "
                "parameter was found."
            )

        qbound_ids = {id(parameter) for _, parameter in named_qbound}
        extracted = []
        extracted_ids = set()
        retained_groups = []

        for group_index, group in enumerate(self.optimizer.param_groups):
            retained = []
            for parameter in group["params"]:
                parameter_id = id(parameter)
                if parameter_id in qbound_ids:
                    if parameter_id not in extracted_ids:
                        extracted.append(parameter)
                        extracted_ids.add(parameter_id)
                else:
                    retained.append(parameter)

            if retained:
                group["params"] = retained
                if not group.get("name") or group.get("name") == "qbound":
                    group["name"] = f"base_{group_index}"
                retained_groups.append(group)

        missing_ids = qbound_ids.difference(extracted_ids)
        if missing_ids:
            missing_names = [
                name
                for name, parameter in named_qbound
                if id(parameter) in missing_ids
            ]
            raise RuntimeError(
                "Q-Bound parameters were absent from build_optimizer():\n"
                + "\n".join(missing_names)
            )

        self.optimizer.param_groups = retained_groups
        qbound_group = {
            "params": extracted,
            "lr": qbound_lr,
            "name": "qbound",
        }

        if "qbound_weight_decay" in optim_cfg:
            qbound_weight_decay = float(
                optim_cfg["qbound_weight_decay"]
            )
            if (
                not math.isfinite(qbound_weight_decay)
                or qbound_weight_decay < 0.0
            ):
                raise ValueError(
                    "optim.qbound_weight_decay must be finite and >= 0."
                )
            qbound_group["weight_decay"] = qbound_weight_decay

        self.optimizer.add_param_group(qbound_group)
        if self.optimizer.param_groups[-1].get("name") != "qbound":
            raise RuntimeError(
                "Failed to create the Q-Bound optimizer group."
            )

        total_parameters = sum(parameter.numel() for parameter in extracted)
        actual_group = self.optimizer.param_groups[-1]
        print("\n[Q-Bound V2.8 Optimizer Group Created]")
        print(f"  lr: {qbound_lr}")
        print(
            "  weight_decay:",
            actual_group.get(
                "weight_decay",
                self.optimizer.defaults.get("weight_decay"),
            ),
        )
        print(f"  trainable parameters: {total_parameters:,}")
        print(f"  tensors: {len(extracted)}\n")

    def _qbound_optimizer_group(self):
        groups = [
            group
            for group in self.optimizer.param_groups
            if group.get("name") == "qbound"
        ]
        if len(groups) != 1:
            raise RuntimeError(
                "Q-Bound V2.8 requires exactly one optimizer group named "
                f"'qbound'; found {len(groups)}."
            )
        return groups[0]

    def _check_qbound_optimizer(self, check_current_lr=True):
        if not self._qbound_enabled():
            return

        named_qbound = self._named_trainable_qbound_parameters()
        if not named_qbound:
            raise RuntimeError(
                "Q-Bound V2.8 has no trainable Q-Bound parameters."
            )

        optimizer_counts = {}
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                optimizer_counts[parameter_id] = (
                    optimizer_counts.get(parameter_id, 0) + 1
                )

        missing = []
        duplicated = []
        for name, parameter in named_qbound:
            count = optimizer_counts.get(id(parameter), 0)
            if count == 0:
                missing.append(name)
            elif count > 1:
                duplicated.append(name)

        if missing:
            raise RuntimeError(
                "Q-Bound parameters missing from optimizer:\n"
                + "\n".join(missing)
            )
        if duplicated:
            raise RuntimeError(
                "Q-Bound parameters duplicated in optimizer:\n"
                + "\n".join(duplicated)
            )

        qbound_group = self._qbound_optimizer_group()
        expected_ids = {id(parameter) for _, parameter in named_qbound}
        actual_ids = {id(parameter) for parameter in qbound_group["params"]}
        if actual_ids != expected_ids:
            absent = [
                name
                for name, parameter in named_qbound
                if id(parameter) not in actual_ids
            ]
            extra_count = len(actual_ids.difference(expected_ids))
            raise RuntimeError(
                "The dedicated Q-Bound group does not exactly match "
                "QBoundCore parameters. Missing:\n"
                + "\n".join(absent)
                + f"\nUnexpected tensors: {extra_count}"
            )

        if check_current_lr:
            expected_lr = float(self.cfgs["optim"]["qbound_lr"])
            actual_lr = float(qbound_group["lr"])
            if not math.isclose(
                actual_lr,
                expected_lr,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Q-Bound optimizer LR mismatch: "
                    f"expected {expected_lr}, got {actual_lr}."
                )

        total_parameters = sum(
            parameter.numel() for _, parameter in named_qbound
        )
        print("[Q-Bound V2.8 Optimizer Check]")
        print(f"  trainable parameters: {total_parameters:,}")
        print(f"  tensors: {len(named_qbound)}")
        print("  every tensor registered exactly once: YES\n")

    # ------------------------------------------------------------------
    # V2.8 cross-file component contract
    # ------------------------------------------------------------------
    def _check_qbound_v280_components(self, qbound_cfg):
        version = str(qbound_cfg.get("version", ""))
        if version != self.QBOUND_VERSION:
            raise ValueError(
                "BasePipeline only supports active Q-Bound V2.8.0; "
                f"got model.qbound.version={version!r}."
            )

        allowed_qbound_keys = {
            "enabled",
            "version",
            "require_question_only",
            "require_token_question",
            "detach_backbone_inputs",
            "hidden_dim",
            "dropout",
            "num_question_heads",
            "question_ff_multiplier",
            "support_graph_mode",
            "support_predictor_enabled",
            "grounding_loss",
            "predictive_loss",
            "pair_loss",  # V3.5 train-only objective.
            "temporal_context_mode",  # H1 controlled architecture: legacy or eq_ptc.
            "eqptc_early_scale",
            "eqptc_position_scale",
            "eqptc_temporal_scale",
            # V3.7 controlled pooling and event-relative comparator.
            "language_pooling_mode",
            "event_grounding_mode",
            "event_residual_scale",
            "event_relation_temperature",
            "event_candidate_max_length",
        }
        unsupported = sorted(
            set(qbound_cfg).difference(allowed_qbound_keys)
        )
        if unsupported:
            raise ValueError(
                "Q-Bound V2.8 config contains unsupported/legacy keys: "
                + ", ".join(unsupported)
            )

        h1 = self._h1_temporal_settings()
        temporal_mode, eqptc_scales = h1["mode"], h1["scales"]
        v37 = self._v37_settings()
        event_mode = v37["event_grounding_mode"]
        pooling_mode = v37["language_pooling_mode"]
        if temporal_mode == "eq_ptc" and (
            qbound_cfg.get("support_graph_mode") != "fixed"
            or qbound_cfg.get("support_predictor_enabled", True)
        ):
            raise ValueError("H1 EQ-PTC experiment requires R2-C fixed graph, predictor disabled.")

        if not bool(qbound_cfg.get("enabled", True)):
            raise ValueError(
                "_check_qbound_v280_components called while Q-Bound is disabled."
            )
        if not bool(qbound_cfg.get("require_question_only", True)):
            raise ValueError(
                "Q-Bound V2.8 requires require_question_only=true."
            )
        if not bool(qbound_cfg.get("require_token_question", True)):
            raise ValueError(
                "Q-Bound V2.8 requires require_token_question=true."
            )
        if not bool(qbound_cfg.get("detach_backbone_inputs", True)):
            raise ValueError(
                "Q-Bound V2.8 requires detach_backbone_inputs=true."
            )
        graph_mode = qbound_cfg.get("support_graph_mode", "learned")
        if not isinstance(graph_mode, str) or graph_mode not in ("learned", "fixed"):
            raise ValueError(
                "model.qbound.support_graph_mode must be exactly "
                "'learned' (R2-A) or 'fixed' (R2-B)."
            )
        predictor_enabled = qbound_cfg.get("support_predictor_enabled", True)
        if not isinstance(predictor_enabled, bool):
            raise TypeError("model.qbound.support_predictor_enabled must be boolean.")
        if not predictor_enabled and graph_mode != "fixed":
            raise ValueError("R2-C requires support_graph_mode=fixed.")
        if bool(self.cfgs["model"].get("do", False)):
            raise ValueError(
                "Q-Bound V2.8 requires model.do=false so VisualIntervention "
                "cannot confound temporal grounding."
            )

        hidden_dim = int(qbound_cfg.get("hidden_dim", 256))
        dropout = float(
            qbound_cfg.get(
                "dropout",
                self.cfgs["model"].get(
                    "qbound_dropout",
                    self.cfgs["model"].get("dropout", 0.1),
                ),
            )
        )
        num_heads = int(qbound_cfg.get("num_question_heads", 4))
        ff_multiplier = int(
            qbound_cfg.get("question_ff_multiplier", 2)
        )
        if hidden_dim < 1:
            raise ValueError("model.qbound.hidden_dim must be positive.")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("model.qbound.dropout must be in [0,1).")
        if num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError(
                "model.qbound.hidden_dim must be divisible by "
                "num_question_heads."
            )
        if ff_multiplier < 1:
            raise ValueError(
                "model.qbound.question_ff_multiplier must be >= 1."
            )

        grounding_cfg = self._validate_loss_block(
            qbound_cfg.get("grounding_loss", {}),
            "grounding_loss",
            {"enabled", "loss_weight", "temperature"},
            require_temperature=True,
        )
        predictive_cfg = self._validate_loss_block(
            qbound_cfg.get("predictive_loss", {}),
            "predictive_loss",
            {"enabled", "loss_weight"},
            require_temperature=False,
            allow_disabled=not predictor_enabled,
        )
        if not predictor_enabled and predictive_cfg["enabled"]:
            raise ValueError("R2-C must disable predictive_loss with weight 0.")

        pair_settings = self._base_v35_pair_settings()
        pair_enabled = pair_settings["enabled"]
        if pair_enabled and (graph_mode != "fixed" or predictor_enabled):
            raise ValueError("V3.5 QDTG requires R2-C fixed graph without a support predictor.")

        max_feats = int(self.cfgs.get("dataset", {}).get("max_feats", 0))
        if max_feats == 1:
            raise ValueError(
                "Q-Bound V2.8 predictive support requires at least two "
                "video timesteps; dataset.max_feats=1 cannot form an edge."
            )

        model_core = self._model_core()
        model_version = str(getattr(model_core, "qbound_version", ""))
        if model_version != version:
            raise RuntimeError(
                "BasePipeline/CRA version mismatch: "
                f"config={version!r}, model={model_version!r}."
            )

        decoder = getattr(model_core, "decoder", None)
        if decoder is None:
            raise AttributeError(
                "Q-Bound V2.8 requires model.decoder."
            )
        decoder_version = str(getattr(decoder, "qbound_version", ""))
        if decoder_version != version:
            raise RuntimeError(
                "BasePipeline/RefineDecoder version mismatch: "
                f"config={version!r}, decoder={decoder_version!r}."
            )

        qbound_core = getattr(decoder, "qbound", None)
        if qbound_core is None:
            raise AttributeError(
                "Q-Bound V2.8 requires RefineDecoder.qbound."
            )
        core_version = str(getattr(qbound_core, "qbound_version", ""))
        if core_version != version:
            raise RuntimeError(
                "BasePipeline/QBoundCore version mismatch: "
                f"config={version!r}, core={core_version!r}."
            )

        decoder_graph_mode = getattr(decoder, "qbound_support_graph_mode", None)
        core_graph_mode = getattr(qbound_core, "support_graph_mode", None)
        if decoder_graph_mode != graph_mode or core_graph_mode != graph_mode:
            raise RuntimeError(
                "R2 A/B/C graph mode mismatch between YAML, RefineDecoder "
                "and QBoundCore. Expected all to be "
                f"{graph_mode!r}; got decoder={decoder_graph_mode!r}, "
                f"core={core_graph_mode!r}. Install both R2 files, "
                "not a mixture of R2 and legacy V2.8."
            )

        if bool(getattr(decoder, "qbound_support_predictor_enabled", True)) != predictor_enabled or bool(
            getattr(qbound_core, "support_predictor_enabled", True)
        ) != predictor_enabled:
            raise RuntimeError("R2-C predictor mode mismatch across YAML/Refine/Core.")

        # H1: detect the frequent source-mixing failure before training. The
        # architecture mode is NOT encoded in qbound_version='2.8.0'.
        for owner_name, owner in (("CRA", model_core), ("RefineDecoder", decoder),
                                  ("QBoundCore", qbound_core)):
            attribute = ("temporal_context_mode" if owner_name == "QBoundCore"
                         else "qbound_temporal_context_mode")
            actual = getattr(owner, attribute, None)
            if actual != temporal_mode:
                raise RuntimeError(
                    f"H1 {owner_name}.{attribute}={actual!r}, but YAML requests "
                    f"{temporal_mode!r}. Update core.py/refine.py/cra.py together."
                )
        for key, value in eqptc_scales.items():
            for owner_name, owner in (("CRA", model_core), ("RefineDecoder", decoder)):
                if key not in getattr(owner, "qbound_eqptc_scales", {}):
                    raise RuntimeError(f"H1 {owner_name} does not expose {key}.")
                actual = float(owner.qbound_eqptc_scales[key])
                if not math.isclose(actual, value, rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(f"H1 {owner_name}.{key}={actual} != YAML {value}.")
            actual_core = float(getattr(qbound_core, key, float("nan")))
            if not math.isclose(actual_core, value, rel_tol=1e-10, abs_tol=1e-12):
                raise RuntimeError(f"H1 QBoundCore.{key}={actual_core} != YAML {value}.")

        # V3.7 identity checks: no silent mixture of CRA/Refine/Core code.
        for owner_name, owner in (("CRA", model_core), ("RefineDecoder", decoder),
                                   ("QBoundCore", qbound_core)):
            attr = ("event_grounding_mode" if owner_name == "QBoundCore"
                    else "qbound_event_grounding_mode")
            if getattr(owner, attr, None) != event_mode:
                raise RuntimeError(
                    f"V3.7 {owner_name}.{attr} mismatch: expected {event_mode!r}."
                )
        language = getattr(model_core, "lang_model", None)
        if language is None:
            raise RuntimeError("V3.7 CRA has no LanModel.")
        for owner_name, actual in (
            ("CRA", getattr(model_core, "qbound_language_pooling_mode", None)),
            ("LanModel", getattr(language, "pooling_mode", None)),
        ):
            if actual != pooling_mode:
                raise RuntimeError(
                    f"V3.7 {owner_name} pooling mode {actual!r} != {pooling_mode!r}."
                )
        for name in ("event_residual_scale", "event_relation_temperature",
                     "event_candidate_max_length"):
            expected = v37[name]
            actual = getattr(qbound_core, name, "missing")
            if expected is None:
                same = actual is None
            elif isinstance(expected, float):
                try:
                    same = math.isclose(float(actual), expected, rel_tol=1e-7, abs_tol=1e-8)
                except (TypeError, ValueError):
                    same = False
            else:
                same = actual == expected
            if not same:
                raise RuntimeError(f"V3.7 QBoundCore.{name}={actual!r} != YAML {expected!r}.")
        if event_mode == "event_relative" and (
            graph_mode != "fixed" or predictor_enabled or temporal_mode != "eq_ptc"
        ):
            raise RuntimeError("V3.7 requires EQ-PTC with R2-C fixed propagation.")

        if pair_enabled:
            # Fail before optimizer/training on a mixture of R2-C source and
            # V3.5 pair-aware modules; no new nn.Parameters are expected.
            import inspect
            interfaces = (
                ("CRA.refine_forward", getattr(model_core, "refine_forward", None),
                 {"paired_question", "pair_valid_mask"}),
                ("RefineDecoder.forward", getattr(decoder, "forward", None),
                 {"paired_question_feature", "paired_question_mask", "pair_valid_mask"}),
                ("QBoundCore.forward_paired", getattr(qbound_core, "forward_paired", None),
                 {"paired_question_features", "paired_question_mask"}),
            )
            for name, fn, required in interfaces:
                if not callable(fn):
                    raise RuntimeError(f"V3.5 source mismatch: {name} is not callable.")
                parameters = set(inspect.signature(fn).parameters)
                if not required.issubset(parameters):
                    raise RuntimeError(f"V3.5 source mismatch: {name} lacks {sorted(required - parameters)}.")

        for owner_name, owner in (
            ("CRA", model_core),
            ("RefineDecoder", decoder),
        ):
            required_true = {
                "qbound_require_question_only": True,
                "qbound_require_token_question": True,
                "qbound_detach_backbone_inputs": True,
            }
            for attribute, expected in required_true.items():
                if bool(getattr(owner, attribute, False)) != expected:
                    raise RuntimeError(
                        f"{owner_name}.{attribute} violates V2.8."
                    )

        expected_core_values = {
            "input_dim": int(self.cfgs["model"]["d_model"]),
            "hidden_dim": hidden_dim,
            "num_question_heads": num_heads,
            "question_ff_multiplier": ff_multiplier,
        }
        for attribute, expected in expected_core_values.items():
            actual = int(getattr(qbound_core, attribute, -1))
            if actual != expected:
                raise RuntimeError(
                    f"QBoundCore.{attribute} mismatch: expected "
                    f"{expected}, got {actual}."
                )
        actual_dropout = float(getattr(qbound_core, "dropout_p", -1.0))
        if not math.isclose(
            actual_dropout,
            dropout,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "QBoundCore.dropout_p mismatch: "
                f"expected {dropout}, got {actual_dropout}."
            )

        relevance_modules = {
            "where_input_proj",
            "visual_distinctiveness_encoder",
            "visual_transition_encoder",
            "visual_transition_frame_delta",
            "temporal_norm1",
            "temporal_attn",
            "temporal_attn_delta",
            "temporal_norm2",
            "temporal_ff",
            "question_token_proj",
            "relevance_visual_norm",
            "relevance_question_attn",
            "relevance_question_context_norm",
            "semantic_seed_encoder",
            "semantic_seed_score",
        }
        # Exactly six trainable modules belong to the additional H1 B+C path.
        # Sinusoidal position is deterministic, so has no trainable parameters.
        eqptc_modules = {
            "eqptc_early_visual_norm",
            "eqptc_early_question_attn",
            "eqptc_early_delta",
            "eqptc_temporal_norm",
            "eqptc_temporal_attn",
            "eqptc_temporal_delta",
        }
        if temporal_mode == "eq_ptc":
            relevance_modules.update(eqptc_modules)
        else:
            forbidden_eq = sorted(name for name in eqptc_modules if hasattr(qbound_core, name))
            if forbidden_eq:
                raise RuntimeError(
                    "H1 legacy control contains EQ-PTC modules: " + ", ".join(forbidden_eq)
                )
        # The optional Event Comparator is a relevance/Seed module; it must
        # receive the same qbound_lr and have no access to QA answers or GT.
        if event_mode == "event_relative":
            relevance_modules.add("event_comparator")
        elif hasattr(qbound_core, "event_comparator"):
            raise RuntimeError("V3.7 legacy mode contains a trainable Event Comparator.")

        support_modules = {
            "support_input_proj",
            "support_forward_cell",
            "support_backward_cell",
            "support_forward_predictor",
            "support_backward_predictor",
        }
        expected_modules = relevance_modules | (support_modules if predictor_enabled else set())
        if not predictor_enabled:
            present = sorted(name for name in support_modules if hasattr(qbound_core, name))
            if present:
                raise RuntimeError("R2-C predictor modules must be ABSENT: " + ", ".join(present))

        missing_modules = sorted(
            name
            for name in expected_modules
            if not isinstance(getattr(qbound_core, name, None), torch.nn.Module)
        )
        if missing_modules:
            raise RuntimeError(
                "QBoundCore is missing V2.8 trainable modules:\n"
                + "\n".join(missing_modules)
            )

        forbidden_attributes = {
            "query_route_visual_proj",
            "query_route_question_proj",
            "query_route_attn",
            "core_encoder",
            "core_score",
            "joint_continuity_encoder",
            "continuity_direction",
            "boundary_refiner",
            "span_scorer",
            "start_head",
            "end_head",
            "event_selector",
            "functional_critic",
            "answer_critic",
        }
        present_forbidden = sorted(
            name for name in forbidden_attributes if hasattr(qbound_core, name)
        )
        if present_forbidden:
            raise RuntimeError(
                "QBoundCore contains removed pre-V2.8 modules: "
                + ", ".join(present_forbidden)
            )
        if hasattr(decoder, "qbound_boundary_refiner"):
            raise RuntimeError(
                "RefineDecoder must not instantiate qbound_boundary_refiner."
            )

        named_qbound = self._named_trainable_qbound_parameters()
        qbound_group = self._qbound_optimizer_group()
        qbound_group_ids = {
            id(parameter) for parameter in qbound_group["params"]
        }

        relevance_ids = set()
        support_ids = set()
        trainable_roots = set()
        unexpected_roots = []
        wrong_prefix = []
        for name, parameter in named_qbound:
            normalized = self._normalize_parameter_name(name)
            marker = "decoder.qbound."
            if marker not in normalized:
                wrong_prefix.append(name)
                continue
            relative = normalized.split(marker, 1)[1]
            root = relative.split(".", 1)[0]
            if root in relevance_modules:
                trainable_roots.add(root)
                relevance_ids.add(id(parameter))
            elif root in support_modules:
                trainable_roots.add(root)
                support_ids.add(id(parameter))
            else:
                unexpected_roots.append(name)

        if wrong_prefix:
            raise RuntimeError(
                "V2.8 trainable Q-Bound tensors must live under "
                "decoder.qbound.*:\n"
                + "\n".join(wrong_prefix)
            )
        if unexpected_roots:
            raise RuntimeError(
                "Unexpected trainable modules exist in QBoundCore:\n"
                + "\n".join(unexpected_roots)
            )
        missing_trainable_roots = sorted(
            expected_modules.difference(trainable_roots)
        )
        if missing_trainable_roots:
            raise RuntimeError(
                "V2.8 modules unexpectedly have no trainable parameters:\n"
                + "\n".join(missing_trainable_roots)
            )
        if not relevance_ids:
            raise RuntimeError(
                "The semantic-relevance branch has no trainable parameters."
            )
        if predictor_enabled and not support_ids:
            raise RuntimeError(
                "The video-only support predictor has no trainable parameters."
            )
        if relevance_ids.intersection(support_ids):
            raise RuntimeError(
                "Relevance and support predictor parameter sets overlap."
            )
        all_qbound_ids = {id(parameter) for _, parameter in named_qbound}
        if relevance_ids | support_ids != all_qbound_ids:
            raise RuntimeError(
                "Not every Q-Bound parameter belongs to an explicit V2.8 "
                "branch."
            )
        if not all_qbound_ids.issubset(qbound_group_ids):
            raise RuntimeError(
                "Some V2.8 Q-Bound parameters are outside qbound_lr."
            )

        video_proj_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if (
                parameter.requires_grad
                and "decoder.video_proj." in self._normalize_parameter_name(name)
            )
        ]
        if not video_proj_parameters:
            raise RuntimeError(
                "RefineDecoder.video_proj trainable parameters were not found."
            )
        misplaced_video_proj = [
            name
            for name, parameter in video_proj_parameters
            if id(parameter) in qbound_group_ids
        ]
        if misplaced_video_proj:
            raise RuntimeError(
                "Shared CRA video_proj must remain in the base optimizer "
                "group:\n"
                + "\n".join(misplaced_video_proj)
            )

        print("[Q-Bound V2.8 R2-A/B/C + H1 Component Check]")
        print(f"  H1 temporal mode: {temporal_mode}")
        print(f"  V3.7 language pooling: {pooling_mode}")
        print(f"  V3.7 event grounding: {event_mode}")
        if event_mode == "event_relative":
            print(f"  V3.7 Event Comparator tensors: "
                  f"{len(list(qbound_core.event_comparator.parameters()))}")
        print(f"  H1 EQ-PTC scales: {eqptc_scales if temporal_mode == 'eq_ptc' else 'inactive'}")
        print(f"  version: {version}")
        print(f"  propagation graph mode: {graph_mode}")
        print(f"  semantic-relevance tensors: {len(relevance_ids)}")
        print(f"  support-predictor tensors: {len(support_ids)}")
        print("  shared support for training/inference: REQUIRED")
        print("  VQA -> topology gradient: BLOCKED by integration contract")
        print("  L_ground -> support graph gradient: BLOCKED")
        print("  L_pred: active ONLY in A/B" if predictor_enabled
              else "  L_pred: removed in R2-C")
        print(
            "  loss weights: "
            f"grounding={grounding_cfg['loss_weight']}, "
            f"predictive={predictive_cfg['loss_weight']}"
        )
        print("  shared CRA video_proj remains a base parameter: YES\n")

    def _check_qbound_components(self):
        if not self._qbound_enabled():
            return
        qbound_cfg = self._qbound_config()
        version = str(qbound_cfg.get("version", ""))
        if version != self.QBOUND_VERSION:
            raise ValueError(
                "This BasePipeline has no legacy Q-Bound dispatch. "
                f"Expected {self.QBOUND_VERSION!r}, got {version!r}."
            )
        self._check_qbound_v280_components(qbound_cfg)

    def _check_optimizer_coverage(self):
        """Ensure every trainable model tensor occurs exactly once."""
        optimizer_counts = {}
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                optimizer_counts[parameter_id] = (
                    optimizer_counts.get(parameter_id, 0) + 1
                )

        missing = []
        duplicated = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            count = optimizer_counts.get(id(parameter), 0)
            if count == 0:
                missing.append(name)
            elif count > 1:
                duplicated.append(name)

        if missing:
            raise RuntimeError(
                "Trainable parameters missing from optimizer:\n"
                + "\n".join(missing)
            )
        if duplicated:
            raise RuntimeError(
                "Trainable parameters duplicated in optimizer:\n"
                + "\n".join(duplicated)
            )

        print(
            "[Optimizer Coverage Check] Every trainable parameter is "
            "registered exactly once.\n"
        )

    def _check_qbound_scheduler(self, check_base_lr=True):
        if not self._qbound_enabled():
            return

        qbound_group_index = None
        for group_index, group in enumerate(self.optimizer.param_groups):
            if group.get("name") == "qbound":
                qbound_group_index = group_index
                break
        if qbound_group_index is None:
            raise RuntimeError(
                "Q-Bound optimizer group is missing before scheduler use."
            )

        if not hasattr(self.lr_scheduler, "base_lrs"):
            print(
                "[Q-Bound V2.8 Scheduler Check] Scheduler exposes no "
                "base_lrs; optimizer grouping order was still validated.\n"
            )
            return

        base_lrs = list(self.lr_scheduler.base_lrs)
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise RuntimeError(
                "Scheduler/optimizer parameter-group mismatch: "
                f"scheduler={len(base_lrs)}, "
                f"optimizer={len(self.optimizer.param_groups)}."
            )

        if check_base_lr:
            expected_lr = float(self.cfgs["optim"]["qbound_lr"])
            actual_lr = float(base_lrs[qbound_group_index])
            if not math.isclose(
                actual_lr,
                expected_lr,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Scheduler registered an unexpected Q-Bound base LR: "
                    f"expected {expected_lr}, got {actual_lr}."
                )

        print(
            "[Q-Bound V2.8 Scheduler Check] "
            f"group_index={qbound_group_index}, "
            f"base_lr={base_lrs[qbound_group_index]}\n"
        )

    # ------------------------------------------------------------------
    # Checkpoint metadata and resume safety
    # ------------------------------------------------------------------
    def _optimizer_group_signature(self):
        return [
            {
                "name": group.get("name", f"group_{index}"),
                "tensor_count": len(group["params"]),
                "parameter_count": int(
                    sum(parameter.numel() for parameter in group["params"])
                ),
            }
            for index, group in enumerate(self.optimizer.param_groups)
        ]

    def _qbound_checkpoint_metadata(self):
        enabled = self._qbound_enabled()
        qbound = self._qbound_config() if enabled else {}
        pair = self._base_v35_pair_settings() if enabled else None
        h1 = self._h1_temporal_settings() if enabled else None
        v37 = self._v37_settings() if enabled else None
        predictor = bool(qbound.get("support_predictor_enabled", True)) if enabled else False
        objective = None
        if enabled:
            objective = ("L_CRA/main + lambda_G*L_ground + lambda_P*L_pred"
                         if predictor else "L_CRA/main + lambda_G*L_ground")
            if pair["enabled"]:
                objective += " + lambda_pair*L_pair"
        return {
            "enabled": enabled,
            "version": self.QBOUND_VERSION if enabled else None,
            "experiment_variant": self._qbound_experiment_variant() if enabled else None,
            "temporal_context_mode": h1["mode"] if enabled else None,
            "eqptc_scales": h1["scales"] if enabled else None,
            "h1_architecture_revision": "EQ-PTC-BC-v1" if enabled and h1["mode"] == "eq_ptc" else "legacy-V3.5",
            "v37_architecture_revision": (
                "PI-ERG-v1" if enabled and (
                    v37["event_grounding_mode"] != "legacy" or
                    v37["language_pooling_mode"] != "legacy"
                ) else None
            ),
            "v37_settings": v37,
            "init_from_checkpoint": self.cfgs["stat"].get("init_from_checkpoint"),
            "support_predictor_enabled": qbound.get("support_predictor_enabled", True) if enabled else None,
            "support_graph_mode": qbound.get("support_graph_mode", "learned") if enabled else None,
            "planned_epochs": self.epochs,
            "run_seed": self.cfgs["misc"].get("seed"),
            "dataset_name": self.cfgs["dataset"].get("name"),
            "optimizer_groups": self._optimizer_group_signature(),
            "loss_weights": ({
                "grounding": float(qbound.get("grounding_loss", {}).get("loss_weight", 1.0)),
                "predictive": float(qbound.get("predictive_loss", {}).get("loss_weight", 1.0)),
                "pair": float(pair["loss_weight"]),
            } if enabled else None),
            # Logging cadence does not change loss, so it is intentionally not
            # included in the optimizer-compatibility objective signature.
            "pair_loss": ({name: pair[name] for name in
                           ("enabled", "loss_weight", "temperature", "margin")}
                          if enabled else None),
            "loss_contract": objective,
        }

    def _resume_optimizer_is_compatible(
        self,
        checkpoint,
        resume_strict,
    ):
        if not self._qbound_enabled():
            return True

        metadata = checkpoint.get("qbound")
        if not isinstance(metadata, dict):
            message = (
                "Checkpoint has no Q-Bound V2.8 metadata. Model weights may "
                "be loaded non-strictly, but optimizer/scheduler state cannot "
                "be trusted."
            )
            if resume_strict or self._h1_temporal_settings()["mode"] == "eq_ptc":
                raise RuntimeError(message)
            print("Warning:", message)
            return False

        # V3.7 modifies model behavior with no ABI version bump. A saved
        # optimizer is valid only for EXACTLY the same pooling/event architecture
        # and hyperparameters; non-strict resume must not bypass this check.
        current_v37 = self._v37_settings()
        saved_v37 = metadata.get("v37_settings")
        if saved_v37 is None:
            # Old V3.5 metadata is unambiguously legacy for BOTH switches.
            saved_v37 = {
                "event_grounding_mode": "legacy",
                "language_pooling_mode": "legacy",
                "event_residual_scale": 0.10,
                "event_relation_temperature": 0.20,
                "event_candidate_max_length": None,
            }
        if not isinstance(saved_v37, dict) or set(saved_v37) != set(current_v37):
            raise RuntimeError("V3.7 resume checkpoint has invalid event/pooling metadata.")
        for key, expected in current_v37.items():
            actual = saved_v37[key]
            if (isinstance(expected, float) and not isinstance(expected, bool)
                    and isinstance(actual, (int, float)) and not isinstance(actual, bool)):
                matches = math.isclose(float(actual), expected, rel_tol=1e-10, abs_tol=1e-12)
            else:
                matches = actual == expected
            if not matches:
                raise RuntimeError(
                    f"Cannot resume optimizer across V3.7 {key}: "
                    f"checkpoint={actual!r}, current={expected!r}. "
                    "Use stat.init_from_checkpoint for weights-only transfer."
                )
        expected_revision = self._qbound_checkpoint_metadata()["v37_architecture_revision"]
        saved_revision = metadata.get("v37_architecture_revision")
        if saved_revision != expected_revision:
            raise RuntimeError(
                "V3.7 resume architecture revision mismatch: "
                f"saved={saved_revision!r}, expected={expected_revision!r}."
            )

        # Old V3.5 checkpoints have no temporal_context_mode metadata: they
        # are LEGACY by definition, not a legitimate EQ-PTC optimizer state.
        current_h1 = self._h1_temporal_settings()
        saved_h1_mode = metadata.get("temporal_context_mode", "legacy")
        if saved_h1_mode != current_h1["mode"]:
            raise RuntimeError(
                "H1 checkpoint architecture mismatch: "
                f"saved={saved_h1_mode!r}, current={current_h1['mode']!r}. "
                "Do not resume optimizer/scheduler across V3.5 and EQ-PTC, "
                "even with resume_strict=false. Start a NEW training run."
            )
        if saved_h1_mode == "eq_ptc":
            if metadata.get("h1_architecture_revision") != "EQ-PTC-BC-v1":
                raise RuntimeError("H1 checkpoint has unknown EQ-PTC architecture revision.")
            saved_scales = metadata.get("eqptc_scales")
            if not isinstance(saved_scales, dict) or set(saved_scales) != set(current_h1["scales"]):
                raise RuntimeError("H1 checkpoint lacks the matching EQ-PTC scale metadata.")
            for key, actual in current_h1["scales"].items():
                if not math.isclose(float(saved_scales[key]), actual,
                                    rel_tol=1e-10, abs_tol=1e-12):
                    raise RuntimeError(
                        f"H1 checkpoint {key}: saved={saved_scales[key]}, current={actual}. "
                        "A coefficient change is a NEW experiment, not a resume."
                    )

        if not bool(metadata.get("enabled", False)):
            raise RuntimeError(
                "Cannot resume active Q-Bound V2.8 from a checkpoint in "
                "which Q-Bound was disabled."
            )
        checkpoint_version = str(metadata.get("version", ""))
        if checkpoint_version != self.QBOUND_VERSION:
            raise RuntimeError(
                "Q-Bound checkpoint version mismatch: "
                f"expected {self.QBOUND_VERSION!r}, got "
                f"{checkpoint_version!r}."
            )

        # Legacy V2.8 checkpoints did not record graph mode. They can only
        # belong to the original learned arm, never the fixed arm.
        saved_predictor = metadata.get("support_predictor_enabled", True)
        current_predictor = self._qbound_config().get("support_predictor_enabled", True)
        if saved_predictor != current_predictor:
            raise RuntimeError(
                "Cannot resume R2-C from A/B (or reverse): predictor mode mismatch. "
                "Start R2-C from fresh initialization."
            )
        saved_mode = metadata.get("support_graph_mode", "learned")
        current_mode = self._qbound_config().get("support_graph_mode", "learned")
        if saved_mode != current_mode:
            raise RuntimeError(
                "R2-A/B checkpoint graph-mode mismatch: "
                f"saved={saved_mode!r}, current={current_mode!r}. "
                "Never resume fixed from learned (or vice versa), even with "
                "resume_strict=False: this would contaminate the comparison."
            )
        if "support_graph_mode" not in metadata:
            print("Warning: legacy V2.8 checkpoint assumed learned graph mode.")

        saved_planned_epochs = metadata.get("planned_epochs")
        if (saved_planned_epochs is not None
                and int(saved_planned_epochs) != self.epochs):
            raise RuntimeError(
                "R2 checkpoint was trained with a different planned "
                "optim.epochs/LR schedule: "
                f"saved={saved_planned_epochs}, current={self.epochs}. "
                "For 10->15 continuation keep optim.epochs=15 and change "
                "only optim.stop_after_epoch from 10 to 15."
            )

        for saved_key, current_value in (
            ("run_seed", self.cfgs["misc"].get("seed")),
            ("dataset_name", self.cfgs["dataset"].get("name")),
        ):
            if saved_key in metadata and metadata[saved_key] != current_value:
                raise RuntimeError(
                    f"R2 checkpoint {saved_key} mismatch: "
                    f"saved={metadata[saved_key]!r}, "
                    f"current={current_value!r}."
                )

        # Non-strict model-weight loads are allowed, but CROSS-OBJECTIVE
        # optimizer/scheduler continuation is forbidden, even non-strictly.
        current = self._qbound_checkpoint_metadata()
        current_pair = current["pair_loss"]
        saved_pair = metadata.get("pair_loss")
        if saved_pair is None:
            saved_pair = metadata.get("v35_pair_loss")
        if saved_pair is None:
            saved_pair = {"enabled": False, "loss_weight": 0.0,
                          "temperature": 0.07, "margin": 0.0}
        else:
            saved_pair = {key: saved_pair.get(key) for key in
                          ("enabled", "loss_weight", "temperature", "margin")}
        if saved_pair != current_pair:
            raise RuntimeError(
                "Cannot RESUME Q-Bound across R2-C/V3.5 objectives: "
                f"saved pair_loss={saved_pair}, current={current_pair}. "
                "Start a fresh optimizer schedule."
            )
        saved_variant = metadata.get("experiment_variant")
        if saved_variant is not None and saved_variant != current["experiment_variant"]:
            raise RuntimeError(
                "Cannot RESUME Q-Bound from a different experiment variant: "
                f"saved={saved_variant}, current={current['experiment_variant']}."
            )

        saved_loss_weights = metadata.get("loss_weights")
        current_loss_weights = self._qbound_checkpoint_metadata()[
            "loss_weights"
        ]
        if isinstance(saved_loss_weights, dict) and "pair" not in saved_loss_weights:
            saved_loss_weights = {**saved_loss_weights, "pair": 0.0}
        if (
            isinstance(saved_loss_weights, dict)
            and saved_loss_weights != current_loss_weights
        ):
            if current_pair["enabled"] or saved_pair["enabled"]:
                raise RuntimeError(
                    "Cannot RESUME V3.5 with different loss weights: "
                    f"saved={saved_loss_weights}, current={current_loss_weights}."
                )
            message = (
                "Checkpoint Q-Bound loss weights differ from the current "
                f"configuration: saved={saved_loss_weights}, "
                f"current={current_loss_weights}."
            )
            if resume_strict:
                raise RuntimeError(message)
            print("Warning:", message)
            return False

        saved_signature = metadata.get("optimizer_groups")
        current_signature = self._optimizer_group_signature()
        if saved_signature != current_signature:
            message = (
                "Checkpoint optimizer-group signature does not match the "
                "current V2.8 model."
            )
            if resume_strict or self._h1_temporal_settings()["mode"] == "eq_ptc":
                raise RuntimeError(
                    message
                    + f"\nSaved: {saved_signature}"
                    + f"\nCurrent: {current_signature}"
                )
            print("Warning:", message)
            return False
        return True

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------
    def _update_grounding_best(self, epoch, score):
        """Select additional checkpoints ONLY by validation grounding scores."""
        if not self._qbound_enabled():
            return ()

        post_hoc = score.get("val_post-hoc")
        if not isinstance(post_hoc, dict):
            raise RuntimeError(
                "R2 A/B evaluation must provide score['val_post-hoc'] as "
                "a dict; selection by test metrics is forbidden."
            )

        improved = []
        for key in ("mIoU", "Acc&GQA"):
            if key not in post_hoc:
                raise KeyError(f"Validation post-hoc score has no {key!r}.")
            value = float(post_hoc[key])
            if not math.isfinite(value):
                raise RuntimeError(f"Validation post-hoc {key} is not finite.")
            if value > self.grounding_best[key]["value"]:
                self.grounding_best[key] = {"value": value, "epoch": int(epoch)}
                improved.append(key)
        return tuple(improved)

    def train(self):
        if self.start_epoch > self.stop_after_epoch:
            raise ValueError(
                f"Resume starts at epoch {self.start_epoch}, but "
                f"optim.stop_after_epoch={self.stop_after_epoch}. Increase "
                "stop_after_epoch (without changing optim.epochs) or use a "
                "checkpoint from an earlier epoch."
            )
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.stop_after_epoch + 1):
            score, result = self._train_epoch(epoch)

            if score is None:
                self._save_checkpoint(epoch)
                continue

            log = {"epoch": epoch}
            log.update(score)

            if self.mnt_mode != "off" and self.mnt_metric in log:
                self._record_best(log, result)

            best_grounding = self._update_grounding_best(epoch, score)
            best = False
            if self.mnt_mode != "off":
                if self.mnt_metric not in log:
                    print(
                        "Warning: Metric "
                        f"{self.mnt_metric!r} is not found. "
                        "Model performance monitoring is disabled."
                    )
                    self.mnt_mode = "off"
                else:
                    metric_value = log[self.mnt_metric]
                    improved = (
                        self.mnt_mode == "min"
                        and metric_value <= self.mnt_best
                    ) or (
                        self.mnt_mode == "max"
                        and metric_value >= self.mnt_best
                    )

                    if improved:
                        self.mnt_best = metric_value
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

            # Save one full resumable checkpoint, and snapshot every improved
            # validation selector from the SAME state. Never select by test.
            if best or best_grounding or epoch % self.save_period == 0:
                self._save_checkpoint(
                    epoch,
                    save_best=best,
                    save_grounding_best=best_grounding,
                )

            if (
                self.mnt_mode != "off"
                and not_improved_count >= self.early_stop
            ):
                print(
                    "Validation performance did not improve for "
                    f"{self.early_stop} epochs. Training stops."
                )
                break

        if self._qbound_enabled():
            print("[R2 A/B/C] Validation post-hoc best checkpoint epochs:")
            for key, info in self.grounding_best.items():
                print(f"  {key}: epoch {info['epoch']}, value={info['value']:.4f}")
            if self.stop_after_epoch < self.epochs:
                print(
                    f"[R2 A/B/C] Stopped at epoch {self.stop_after_epoch} of "
                    f"planned {self.epochs}. Resume from current_checkpoint.pth "
                    "with unchanged optim.epochs and stop_after_epoch updated."
                )
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        current_time = time.asctime(time.localtime(time.time()))
        # A/B screening intentionally skips test, so never record a fake test row.
        test_was_evaluated = bool(
            self.cfgs["stat"].get("eval_test_each_epoch", False)
        )
        self.best_recorder["val"]["time"] = current_time
        if test_was_evaluated:
            self.best_recorder["test"]["time"] = current_time

        seed = self.cfgs["misc"]["seed"]
        self.best_recorder["val"]["seed"] = seed
        self.best_recorder["val"]["best_model_from"] = "val"
        if test_was_evaluated:
            self.best_recorder["test"]["seed"] = seed
            self.best_recorder["test"]["best_model_from"] = "val"

        record_dir = self.cfgs["stat"]["record_dir"]
        dataset_name = self.cfgs["dataset"]["name"]
        os.makedirs(record_dir, exist_ok=True)
        record_path = os.path.join(record_dir, f"{dataset_name}.csv")

        rows = [self.best_recorder["val"]]
        if test_was_evaluated:
            rows.append(self.best_recorder["test"])
        new_records = pd.DataFrame(rows)
        if os.path.exists(record_path):
            old_records = pd.read_csv(record_path)
            record_table = pd.concat(
                [old_records, new_records],
                ignore_index=True,
            )
        else:
            record_table = new_records

        record_table.to_csv(record_path, index=False)
        print(f"Best results saved to: {record_path}")

    def _save_checkpoint(self, epoch, save_best=False, save_grounding_best=()):
        state = {
            "epoch": int(epoch),
            "model_only": False,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "best_recorder": self.best_recorder,
            "grounding_best": self.grounding_best,
            "qbound": self._qbound_checkpoint_metadata(),
        }
        filename = os.path.join(self.checkpoint_dir, "current_checkpoint.pth")
        torch.save(state, filename)
        print(f"Saving checkpoint: {filename} ...")

        # These snapshots are resumable: weights + optimizer + LR scheduler.
        snapshots = []
        if save_best:
            snapshots.append("model_best.pth")  # configured val monitor, often QA
        for metric_name in save_grounding_best:
            if metric_name == "mIoU":
                snapshots.append("model_best_posthoc_miou.pth")
            elif metric_name == "Acc&GQA":
                snapshots.append("model_best_posthoc_gqa.pth")
            else:
                raise ValueError(f"Unknown R2 grounding checkpoint selector: {metric_name}")

        for snapshot in snapshots:
            out = os.path.join(self.checkpoint_dir, snapshot)
            shutil.copyfile(filename, out)
            print(f"  validation-best snapshot: {out}")

        # Optional diagnostic epoch-by-epoch weights, deliberately omitting
        # optimizer/scheduler so files are smaller. DO NOT resume from these.
        if self.save_epoch_weights:
            weights_path = os.path.join(
                self.checkpoint_dir, f"weights_epoch_{int(epoch):02d}.pth"
            )
            torch.save(
                {
                    "epoch": int(epoch),
                    "model_only": True,
                    "state_dict": self.model.state_dict(),
                    "qbound": self._qbound_checkpoint_metadata(),
                },
                weights_path,
            )
            print(f"  diagnostic model-only snapshot: {weights_path}")

    def _initialize_from_checkpoint(self, checkpoint_path):
        """Initialize V3.7 from matching older Q-Bound weights, NOT optimizer.

        This is an opt-in transfer experiment distinct from training resume.
        All pre-existing model tensors must match shape and be present; only
        new Event Comparator tensors may be randomly initialized. E15/other
        pretrained weights are never treated as a resumed V3.7 run.
        """
        if not self._qbound_enabled():
            raise ValueError("stat.init_from_checkpoint requires active Q-Bound.")
        source = str(checkpoint_path)
        print(f"[V3.7] Initializing weights ONLY from: {source}")
        checkpoint = torch.load(
            source,
            map_location=next(self.model.parameters()).device,
            weights_only=False,  # user-supplied trusted local checkpoint
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Initialization checkpoint must be a dict.")
        state = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state, dict):
            raise TypeError("Initialization checkpoint state_dict must be a dict.")
        saved_metadata = checkpoint.get("qbound")
        if isinstance(saved_metadata, dict):
            if not saved_metadata.get("enabled", False):
                raise RuntimeError("Cannot initialize Q-Bound from a Q-Bound-disabled checkpoint.")
            if str(saved_metadata.get("version", "")) != self.QBOUND_VERSION:
                raise RuntimeError("Initialization checkpoint Q-Bound ABI mismatch.")
            if saved_metadata.get("temporal_context_mode", "legacy") != self._h1_temporal_settings()["mode"]:
                raise RuntimeError("Initialization checkpoint temporal context mode mismatch.")
            if saved_metadata.get("support_graph_mode", "learned") != self._qbound_config().get("support_graph_mode", "learned"):
                raise RuntimeError("Initialization checkpoint support graph mode mismatch.")
            if bool(saved_metadata.get("support_predictor_enabled", True)) != bool(
                self._qbound_config().get("support_predictor_enabled", True)
            ):
                raise RuntimeError("Initialization checkpoint predictor mode mismatch.")
            current_scales = self._h1_temporal_settings()["scales"]
            saved_scales = saved_metadata.get("eqptc_scales")
            if (self._h1_temporal_settings()["mode"] == "eq_ptc"
                    and isinstance(saved_scales, dict)):
                for key, expected in current_scales.items():
                    if key not in saved_scales or not math.isclose(
                        float(saved_scales[key]), expected, rel_tol=1e-10, abs_tol=1e-12
                    ):
                        raise RuntimeError(f"Initialization checkpoint EQ-PTC scale mismatch: {key}.")

        # Strip an optional DataParallel 'module.' prefix for a name-based
        # comparison, then restore the current model's own prefix convention.
        current_state = self.model.state_dict()
        current_names = {self._normalize_parameter_name(k): k for k in current_state}
        source_names = {}
        for key, value in state.items():
            normalized = self._normalize_parameter_name(str(key))
            if normalized in source_names:
                raise RuntimeError(f"Duplicate normalized checkpoint key: {normalized}.")
            source_names[normalized] = value
        remapped = {}
        missing = []
        for normalized, actual_name in current_names.items():
            if normalized not in source_names:
                missing.append(normalized)
                continue
            expected = current_state[actual_name]
            supplied = source_names[normalized]
            if not torch.is_tensor(supplied) or supplied.shape != expected.shape:
                raise RuntimeError(
                    f"Initialization tensor mismatch: {normalized}: "
                    f"expected {tuple(expected.shape)}, got "
                    f"{tuple(supplied.shape) if torch.is_tensor(supplied) else type(supplied)}."
                )
            remapped[actual_name] = supplied
        unexpected = sorted(set(source_names).difference(current_names))
        allowed_missing = [
            name for name in missing
            if name.startswith("decoder.qbound.event_comparator.")
        ]
        forbidden_missing = sorted(set(missing).difference(allowed_missing))
        if forbidden_missing or unexpected:
            raise RuntimeError(
                "V3.7 weight initialization must match ALL existing tensors. "
                f"Missing existing: {forbidden_missing[:12]}; "
                f"Unexpected: {unexpected[:12]}."
            )
        if allowed_missing and self._v37_settings()["event_grounding_mode"] != "event_relative":
            raise RuntimeError("Unexpected Event Comparator missing keys in legacy mode.")
        result = self.model.load_state_dict(remapped, strict=not bool(missing))
        if sorted(self._normalize_parameter_name(x) for x in result.missing_keys) != sorted(missing):
            raise RuntimeError("State load returned unexpected missing keys.")
        if result.unexpected_keys:
            raise RuntimeError(f"State load returned unexpected keys: {result.unexpected_keys}.")
        self.start_epoch = 1
        self._check_optimizer_coverage()
        self._check_qbound_optimizer(check_current_lr=True)
        self._check_qbound_scheduler(check_base_lr=True)
        print(
            "[V3.7] Weights loaded; new Event Comparator tensors: "
            f"{len(allowed_missing)}. Fresh optimizer/scheduler at epoch 1."
        )

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print(f"Loading checkpoint: {resume_path} ...")
        map_location = next(self.model.parameters()).device
        checkpoint = torch.load(
            resume_path,
            map_location=map_location,
            weights_only=False,  # local checkpoint trusted by the user
        )
        if checkpoint.get("model_only", False) or not all(
            key in checkpoint for key in ("optimizer", "lr_scheduler")
        ):
            raise RuntimeError(
                "This checkpoint has model weights but no complete training "
                "state. Resume R2 training only from current_checkpoint.pth "
                "or a model_best*.pth full checkpoint, never from "
                "weights_epoch_XX.pth."
            )

        resume_strict = bool(
            self.cfgs["stat"].get("resume_strict", True)
        )
        optimizer_compatible = self._resume_optimizer_is_compatible(
            checkpoint,
            resume_strict=resume_strict,
        )

        incompatible_keys = self.model.load_state_dict(
            checkpoint["state_dict"],
            strict=resume_strict,
        )
        if not resume_strict:
            print("Missing model keys:", incompatible_keys.missing_keys)
            print(
                "Unexpected model keys:",
                incompatible_keys.unexpected_keys,
            )

        if optimizer_compatible:
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.mnt_best = checkpoint.get(
                "monitor_best",
                self.mnt_best,
            )
            saved_best_recorder = checkpoint.get("best_recorder")
            if isinstance(saved_best_recorder, dict):
                self.best_recorder = saved_best_recorder
            saved_grounding_best = checkpoint.get("grounding_best")
            if isinstance(saved_grounding_best, dict):
                self.grounding_best = saved_grounding_best
        else:
            # A non-strict legacy/transfer load initializes model weights only.
            # Reusing its epoch, optimizer, scheduler, or best metric would mix
            # objectives and invalidate a controlled V2.8 run.
            self.start_epoch = 1
            print(
                "Starting a fresh optimization schedule at epoch 1 after "
                "the non-strict model-only load."
            )

        optimizer_strict = bool(
            self.cfgs["stat"].get(
                "resume_optimizer_strict",
                resume_strict,
            )
        )
        if optimizer_compatible and "optimizer" in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as error:
                if optimizer_strict:
                    raise RuntimeError(
                        "Cannot load optimizer state from checkpoint."
                    ) from error
                print("Warning: cannot load optimizer state:", error)

        if optimizer_compatible and "lr_scheduler" in checkpoint:
            try:
                self.lr_scheduler.load_state_dict(
                    checkpoint["lr_scheduler"]
                )
            except Exception as error:
                if optimizer_strict:
                    raise RuntimeError(
                        "Cannot load scheduler state from checkpoint."
                    ) from error
                print("Warning: cannot load scheduler state:", error)

        self._check_optimizer_coverage()
        self._check_qbound_optimizer(check_current_lr=False)
        self._check_qbound_scheduler(check_base_lr=False)
        print(
            "Checkpoint loaded. Resume training from epoch "
            f"{self.start_epoch}"
        )

    def _record_best(self, log, result):
        """Select checkpoints by validation only; never by test metrics."""
        current_best = self.best_recorder["val"][self.mnt_metric]
        improved_val = (
            self.mnt_mode == "min"
            and log[self.mnt_metric] <= current_best
        ) or (
            self.mnt_mode == "max"
            and log[self.mnt_metric] >= current_best
        )
        if not improved_val:
            return

        self.best_recorder["val"].update(log)
        # Do NOT mirror val metrics into the test recorder when test evaluation
        # is disabled. The result artifact below is validation predictions.
        if self.cfgs["stat"].get("eval_test_each_epoch", False):
            self.best_recorder["test"].update(log)

        filename = os.path.join(
            self.monitor.record_dir,
            "result.pth",
        )
        torch.save(result, filename)

    def _print_best(self):
        metric_name = self.cfgs["stat"]["monitor"]["metric"]
        print(
            "Best results (w.r.t {}) in validation set:".format(
                metric_name
            )
        )
        for key, value in self.best_recorder["val"].items():
            print("\t{:15s}: {}".format(str(key), value))

        if self.cfgs["stat"].get("eval_test_each_epoch", False):
            print(
                "Test results paired with the validation-best checkpoint "
                "(w.r.t {}):".format(metric_name)
            )
            for key, value in self.best_recorder["test"].items():
                print("\t{:15s}: {}".format(str(key), value))
        else:
            print("Test split NOT evaluated during R2-A/B screening.")
