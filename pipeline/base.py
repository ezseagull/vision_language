# Base Pipeline
import json
import math
import os
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

        resume_path = cfgs["stat"].get("resume")
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
        if not enabled:
            raise ValueError(
                f"Canonical Q-Bound V2.8 requires {name}.enabled=true."
            )
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"model.qbound.{name}.loss_weight must be finite and > 0."
            )

        result = {"enabled": enabled, "loss_weight": weight}
        if require_temperature:
            temperature = float(block.get("temperature", 0.07))
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(
                    f"model.qbound.{name}.temperature must be finite and > 0."
                )
            result["temperature"] = temperature
        return result

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
            "grounding_loss",
            "predictive_loss",
        }
        unsupported = sorted(
            set(qbound_cfg).difference(allowed_qbound_keys)
        )
        if unsupported:
            raise ValueError(
                "Q-Bound V2.8 config contains unsupported/legacy keys: "
                + ", ".join(unsupported)
            )

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
        )

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
        support_modules = {
            "support_input_proj",
            "support_forward_cell",
            "support_backward_cell",
            "support_forward_predictor",
            "support_backward_predictor",
        }
        expected_modules = relevance_modules | support_modules

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
        if not support_ids:
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

        print("[Q-Bound V2.8 Component Check]")
        print(f"  version: {version}")
        print(f"  semantic-relevance tensors: {len(relevance_ids)}")
        print(f"  support-predictor tensors: {len(support_ids)}")
        print("  shared support for training/inference: REQUIRED")
        print("  VQA -> topology gradient: BLOCKED by integration contract")
        print("  L_ground -> support graph gradient: BLOCKED")
        print("  L_pred is sole support-predictor objective: REQUIRED")
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
        loss_weights = None
        if enabled:
            qbound_cfg = self._qbound_config()
            loss_weights = {
                "grounding": float(
                    qbound_cfg.get("grounding_loss", {}).get(
                        "loss_weight",
                        1.0,
                    )
                ),
                "predictive": float(
                    qbound_cfg.get("predictive_loss", {}).get(
                        "loss_weight",
                        1.0,
                    )
                ),
            }
        return {
            "enabled": enabled,
            "version": self.QBOUND_VERSION if enabled else None,
            "optimizer_groups": self._optimizer_group_signature(),
            "loss_weights": loss_weights,
            "loss_contract": (
                "L_CRA/main + lambda_G*L_ground + lambda_P*L_pred"
                if enabled
                else None
            ),
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
            if resume_strict:
                raise RuntimeError(message)
            print("Warning:", message)
            return False

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

        saved_loss_weights = metadata.get("loss_weights")
        current_loss_weights = self._qbound_checkpoint_metadata()[
            "loss_weights"
        ]
        if (
            isinstance(saved_loss_weights, dict)
            and saved_loss_weights != current_loss_weights
        ):
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
            if resume_strict:
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
    def train(self):
        not_improved_count = 0

        for epoch in range(self.start_epoch, self.epochs + 1):
            score, result = self._train_epoch(epoch)

            if score is None:
                self._save_checkpoint(epoch)
                continue

            log = {"epoch": epoch}
            log.update(score)

            if self.mnt_mode != "off" and self.mnt_metric in log:
                self._record_best(log, result)

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

            # Save a validation-best checkpoint immediately, even when the
            # epoch is not a periodic checkpoint boundary.
            if best:
                self._save_checkpoint(epoch, save_best=True)
            elif epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=False)

            if (
                self.mnt_mode != "off"
                and not_improved_count >= self.early_stop
            ):
                print(
                    "Validation performance did not improve for "
                    f"{self.early_stop} epochs. Training stops."
                )
                break

        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
        current_time = time.asctime(time.localtime(time.time()))
        self.best_recorder["val"]["time"] = current_time
        self.best_recorder["test"]["time"] = current_time

        seed = self.cfgs["misc"]["seed"]
        self.best_recorder["val"]["seed"] = seed
        self.best_recorder["test"]["seed"] = seed
        self.best_recorder["val"]["best_model_from"] = "val"
        self.best_recorder["test"]["best_model_from"] = "val"

        record_dir = self.cfgs["stat"]["record_dir"]
        dataset_name = self.cfgs["dataset"]["name"]
        os.makedirs(record_dir, exist_ok=True)
        record_path = os.path.join(record_dir, f"{dataset_name}.csv")

        new_records = pd.DataFrame(
            [
                self.best_recorder["val"],
                self.best_recorder["test"],
            ]
        )
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

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            "epoch": int(epoch),
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "best_recorder": self.best_recorder,
            "qbound": self._qbound_checkpoint_metadata(),
        }
        filename = os.path.join(
            self.checkpoint_dir,
            "current_checkpoint.pth",
        )
        torch.save(state, filename)
        print(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = os.path.join(
                self.checkpoint_dir,
                "model_best.pth",
            )
            torch.save(state, best_path)
            print(
                "*************** Saving current best: "
                "model_best.pth ... ***************"
            )

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print(f"Loading checkpoint: {resume_path} ...")
        map_location = next(self.model.parameters()).device
        checkpoint = torch.load(
            resume_path,
            map_location=map_location,
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

        print(
            "Test results paired with the validation-best checkpoint "
            "(w.r.t {}):".format(metric_name)
        )
        for key, value in self.best_recorder["test"].items():
            print("\t{:15s}: {}".format(str(key), value))
