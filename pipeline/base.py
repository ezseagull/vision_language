# Base Pipeline
import copy
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
    """Shared CRA lifecycle, with isolated V2.8/V3.5, V3.8-M0 and M1 contracts.

    V3.8 is an *experiment* version. QBoundCore/RefineDecoder still speak
    their original V2.8.0 wire interface and keep their checkpoint keys.
    """

    QBOUND_VERSION = "2.8.0"
    V38_EXPERIMENT_VERSION = "3.8-minimal"
    V38_M1_EXPERIMENT_VERSION = "3.8-m1"
    M1_MONITOR = "v38_m1_cra_acc_gqa"
    M1_EVENT_PREFIXES = ("qbound_v38_m1_bank.", "qbound_v38_m1_evidence.")

    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs
        self._m2_eval_checkpoint_path = self._preflight_m2_eval_checkpoint()

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

        # M0 phase setup / M1 weight-only transfer must precede the optimizer.
        # In a probe, freezing after optimizer construction would silently
        # retain stale V3.5 groups and invalidate optimizer/resume signatures.
        self.v38_initialization = None
        self._preflight_v38_lifecycle()
        initialization = cfgs["stat"].get("init_from_checkpoint")
        if self._v38_enabled():
            self._prepare_v38_parameter_freeze()
            if initialization:
                self._initialize_v38_from_checkpoint(initialization)
        elif self._base_v38_m1_enabled():
            if initialization:
                self._initialize_v38_m1_from_checkpoint(initialization)

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
        self.grounding_best = {} if self._base_v38_m1_enabled() else {
            "mIoU": {"value": -inf, "epoch": 0},
            "Acc&GQA": {"value": -inf, "epoch": 0},
        }
        if self._v38_enabled():
            # Independent VAL-ONLY selectors; never mixed with old-support.
            self.grounding_best.update({
                "v38_joint_marginal_mIoU": {"value": -inf, "epoch": 0},
                "v38_joint_marginal_Acc&GQA": {"value": -inf, "epoch": 0},
            })

        resume_path = cfgs["stat"].get("resume")
        if resume_path:
            self._resume_checkpoint(resume_path)
        elif self._m2_eval_checkpoint_path is not None:
            self._load_m2_eval_checkpoint(self._m2_eval_checkpoint_path)

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

    # ------------------------------------------------------------------
    # V3.8-Minimal lifecycle / initialization policy
    # ------------------------------------------------------------------
    def _v38_enabled(self):
        return (self._qbound_enabled() and
                str(self._qbound_config().get("version", self.QBOUND_VERSION))
                == self.V38_EXPERIMENT_VERSION)

    def _v38_settings(self):
        if not (self._v38_enabled() or self._base_v38_m1_enabled()):
            return {}
        cfg = self._qbound_config().get("v38", {})
        if not isinstance(cfg, dict):
            raise TypeError("model.qbound.v38 must be a mapping/dict.")
        return cfg

    def _v38_phase(self):
        return str(self._v38_settings().get("train_phase", ""))

    def _base_v38_m1_enabled(self):
        return (self._qbound_enabled() and
                self._qbound_config().get("version") == self.V38_M1_EXPERIMENT_VERSION)

    def _base_m1_grouped_batch(self):
        return self._base_v38_m1_enabled() or self.cfgs["dataset"].get("m1_grouped_batch", False)

    def _preflight_v38_m1_lifecycle(self):
        stat = self.cfgs["stat"]
        if stat.get("resume") and stat.get("init_from_checkpoint"):
            raise ValueError("Choose one of stat.resume and stat.init_from_checkpoint.")
        if stat.get("resume") and (
                stat.get("resume_strict", True) is not True or
                stat.get("resume_optimizer_strict", True) is not True):
            raise ValueError("M1 requires strict model and optimizer resume.")
        for flag in ("eval_test_each_epoch", "save_val_predictions"):
            if stat.get(flag) is not True:
                raise ValueError(f"M1 requires stat.{flag}=true.")
        monitor = stat.get("monitor", {})
        if monitor.get("metric") != self.M1_MONITOR or monitor.get("mode") != "max":
            raise ValueError(f"M1 selects checkpoints only by validation {self.M1_MONITOR} (max).")
        qcfg = self._qbound_config()
        if (qcfg.get("support_graph_mode") != "fixed" or
                qcfg.get("support_predictor_enabled") is not False or
                self._base_v35_pair_settings()["enabled"]):
            raise ValueError("M1 requires fixed R2-C, predictor off, and V3.5 pair loss off.")
        settings = self._v38_settings()
        allowed = {"widths", "prior_mode", "include_tail", "detach_answer_embeddings",
                   "projection_dim", "gamma", "window_temperature", "candidate_temperature",
                   "question_loss_weight", "answer_loss_weight", "gradient_audit_every"}
        if set(settings) - allowed:
            raise ValueError("Unused M1 v38 config keys: " + ", ".join(sorted(set(settings) - allowed)))
        if isinstance(self.model, torch.nn.DataParallel):
            raise ValueError("M1 question pools are [B,B]; configure one GPU, not DataParallel.")

    def _check_qbound_v38_m1_components(self, qbound_cfg):
        model = self._model_core()
        bank = getattr(model, "qbound_v38_m1_bank", None)
        scorer = getattr(model, "qbound_v38_m1_evidence", None)
        if (not getattr(model, "qbound_v38_m1_enabled", False) or
                not isinstance(bank, torch.nn.Module) or not isinstance(scorer, torch.nn.Module) or
                not callable(getattr(bank, "geometry", None))):
            raise RuntimeError("M1 requires its matching CRA, geometry bank, and MIL scorer.")
        if any(p.requires_grad for p in bank.parameters()) or not all(
                p.requires_grad for p in scorer.parameters()):
            raise RuntimeError("M1 must train its MIL scorer and keep the bank reader frozen.")
        # Reuse the full legacy R2-C assertions on M1's unchanged CRA/Core path.
        # The opt-in, parameter-free MBR evaluator is not a V2.8 Core option.
        legacy_cfg = {k: v for k, v in qbound_cfg.items() if k not in ("v38", "m2_mbr")}
        legacy_cfg["version"] = self.QBOUND_VERSION
        self._check_qbound_v280_components(legacy_cfg, model_experiment_version=self.V38_M1_EXPERIMENT_VERSION)

    def _m1_model_training_signature(self):
        """Normalize only diagnostic flags when comparing M1 architectures."""
        config = copy.deepcopy(self.cfgs["model"])
        config["qbound"].pop("m2_mbr", None)
        config["qbound"].get("v38", {}).pop("gradient_audit_every", None)
        return config

    def _preflight_m2_eval_checkpoint(self):
        """An optional weight-only path for a frozen M1 MBR evaluation."""
        stat = self.cfgs["stat"]
        requested = stat.get("m2_eval_checkpoint")
        if requested is None:
            return None
        if not isinstance(requested, (str, os.PathLike)) or not os.fspath(requested).strip():
            raise TypeError("stat.m2_eval_checkpoint must be a nonempty checkpoint path.")
        mbr = self._qbound_config().get("m2_mbr")
        if (
            not self._base_v38_m1_enabled()
            or not isinstance(mbr, dict) or mbr.get("enabled") is not True
            or str(self.cfgs["dataset"].get("name", "")).lower() != "nextgqa"
        ):
            raise ValueError(
                "stat.m2_eval_checkpoint requires NextGQA, Q-Bound 3.8-m1 "
                "and model.qbound.m2_mbr.enabled=true."
            )
        if stat.get("resume") or stat.get("init_from_checkpoint"):
            raise ValueError(
                "Choose stat.m2_eval_checkpoint alone, without stat.resume or "
                "stat.init_from_checkpoint."
            )
        path = os.path.abspath(os.path.expanduser(os.fspath(requested)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"stat.m2_eval_checkpoint does not exist: {path}")
        return path

    def _load_m2_eval_checkpoint(self, path):
        """Load native M1 weights strictly; do not restore any training state."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        meta = checkpoint.get("qbound") if isinstance(checkpoint, dict) else None
        if (
            not isinstance(meta, dict) or meta.get("enabled") is not True
            or meta.get("version") != self.V38_M1_EXPERIMENT_VERSION
            or meta.get("core_wire_version") != self.QBOUND_VERSION
            or meta.get("support_graph_mode") != "fixed"
            or meta.get("support_predictor_enabled") is not False
            or meta.get("dataset_name") != self.cfgs["dataset"].get("name")
        ):
            raise RuntimeError("M2 evaluation requires a native M1 R2-C checkpoint for this dataset.")

        training = meta.get("training_signature")
        saved_model = training.get("model") if isinstance(training, dict) else None
        if not isinstance(saved_model, dict) or saved_model != self._m1_model_training_signature():
            raise RuntimeError(
                "M2 evaluation checkpoint model configuration differs from the M1 YAML."
            )
        saved_data = training.get("dataset")
        if isinstance(saved_data, dict):
            # Paths, workers and batch size may differ for an inference run.
            for key in ("feat_type", "max_feats", "mc", "qmax_words", "amax_words"):
                if key in saved_data and saved_data[key] != self.cfgs["dataset"].get(key):
                    raise RuntimeError(f"M2 evaluation checkpoint dataset setting differs: {key}.")

        state = self._normalized_state_dict(checkpoint.get("state_dict"))
        self._model_core().load_state_dict(state, strict=True)
        self.model.eval()
        print(
            f"[V3.8-M2 MBR evaluation] Loaded M1 weights from {path} "
            f"(epoch={checkpoint.get('epoch', 'unknown')}); "
            "optimizer, scheduler and model-selection state not restored."
        )

    def _initialize_v38_m1_from_checkpoint(self, path):
        """Copy R2-C/V3.5 or M0 shared weights, never M0 event modules/state."""
        path = os.path.abspath(os.path.expanduser(os.fspath(path)))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"stat.init_from_checkpoint does not exist: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict):
            raise RuntimeError("M1 initialization requires a checkpoint with state_dict and Q-Bound metadata.")
        meta = checkpoint.get("qbound")
        if (not isinstance(meta, dict) or not meta.get("enabled", False) or
                meta.get("version") not in (self.QBOUND_VERSION, self.V38_EXPERIMENT_VERSION) or
                meta.get("support_graph_mode") != "fixed" or
                meta.get("support_predictor_enabled") is not False):
            raise RuntimeError("M1 weight-only initialization requires a V3.5/R2-C or M0 checkpoint.")
        if meta.get("dataset_name") != self.cfgs["dataset"].get("name"):
            raise RuntimeError("M1 initialization checkpoint dataset does not match the destination.")
        source_version = meta["version"]
        from_m0 = source_version == self.V38_EXPERIMENT_VERSION
        declared_pool = self.cfgs["stat"].get("init_source_pooling_mode")
        source_pool = meta.get("language_pooling_mode")
        if from_m0:
            m0_meta = meta.get("v38")
            m0_arch = m0_meta.get("architecture") if isinstance(m0_meta, dict) else None
            if not isinstance(m0_arch, dict):
                raise RuntimeError("M0 checkpoint lacks source architecture metadata.")
            source_pool = m0_arch.get("language_pooling_mode", source_pool)
        if source_pool is not None and declared_pool is not None and source_pool != declared_pool:
            raise RuntimeError("Declared source pooling contradicts checkpoint metadata.")
        source_pool = source_pool if source_pool is not None else declared_pool
        if source_pool not in ("legacy", "masked"):
            raise RuntimeError("Set stat.init_source_pooling_mode to the actual source pooling mode.")
        if source_pool != getattr(self._model_core(), "language_pooling_mode", None):
            raise RuntimeError("Keep source CRA language pooling when transferring legacy weights to M1.")

        source = self._normalized_state_dict(checkpoint.get("state_dict"))
        destination_model = self._model_core()
        dest = destination_model.state_dict()
        m0_prefixes = ("qbound_v38_bank.", "qbound_v38_evidence.")
        discarded = {k for k in source if k.startswith(m0_prefixes)} if from_m0 else set()
        if from_m0 and not all(any(k.startswith(p) for k in discarded) for p in m0_prefixes):
            raise RuntimeError("M0 source is missing its event bank or evidence scorer keys.")
        fresh = {k for k in dest if k.startswith(self.M1_EVENT_PREFIXES)}
        legacy = {k: v for k, v in source.items() if k not in discarded}
        missing, unexpected = set(dest) - set(legacy), set(legacy) - set(dest)
        if missing != fresh or unexpected:
            raise RuntimeError(f"M1 shared checkpoint keys mismatch: missing={sorted(missing - fresh)}, "
                               f"unexpected={sorted(unexpected | (fresh - missing))}.")
        for name, tensor in legacy.items():
            if not torch.is_tensor(tensor) or tensor.shape != dest[name].shape:
                raise RuntimeError(f"M1 initialization tensor shape mismatch: {name}")
        incompatible = destination_model.load_state_dict(legacy, strict=False)
        if set(incompatible.missing_keys) != fresh or incompatible.unexpected_keys:
            raise RuntimeError("M1 initialization did not load exactly the shared CRA/Core state.")
        self.v38_initialization = {
            "source_path": path, "source_version": source_version,
            "source_variant": meta.get("experiment_variant"), "source_epoch": checkpoint.get("epoch"),
            "shared_state_keys": len(legacy), "m0_state_keys_ignored": len(discarded),
            "m1_state_keys_initialized_fresh": len(fresh), "source_pooling_mode": source_pool,
        }
        print("[V3.8-M1 weight-only initialization]", self.v38_initialization)

    def _preflight_v38_lifecycle(self):
        if self._base_v38_m1_enabled():
            return self._preflight_v38_m1_lifecycle()
        if not self._v38_enabled():
            if self.cfgs["stat"].get("init_from_checkpoint"):
                raise ValueError("stat.init_from_checkpoint is reserved for V3.8-M0/M1.")
            return
        if self._v38_phase() not in ("probe", "joint"):
            raise ValueError("M0 requires v38.train_phase='probe' or 'joint'.")
        if (self.cfgs["stat"].get("resume") and
                self.cfgs["stat"].get("init_from_checkpoint")):
            raise ValueError("Choose exactly one of stat.resume and stat.init_from_checkpoint.")
        if self.cfgs["stat"].get("resume") and not self.cfgs["stat"].get("resume_strict", True):
            raise ValueError("M0 continuation requires stat.resume_strict=true. "
                             "Use stat.init_from_checkpoint for a fresh optimizer instead.")
        for name in ("eval_test_each_epoch", "save_val_predictions"):
            if self.cfgs["stat"].get(name) is not True:
                raise ValueError(f"V3.8 requires stat.{name}=true.")
        monitor = str(self.cfgs["stat"]["monitor"]["metric"])
        if monitor.lower().startswith(("test", "test_", "test/")):
            raise ValueError("Checkpoint selection MUST use validation, not test.")
        if self._v38_phase() == "probe" and self._base_v35_pair_settings()["enabled"]:
            raise ValueError("M0 frozen probe requires pair_loss.enabled=false, weight=0.")

    def _prepare_v38_parameter_freeze(self):
        model = self._model_core()
        if not getattr(model, "qbound_v38_minimal_enabled", False):
            raise RuntimeError("V3.8 YAML requires the matching CRA model source.")
        bank = getattr(model, "qbound_v38_bank", None)
        evidence = getattr(model, "qbound_v38_evidence", None)
        if not isinstance(bank, torch.nn.Module) or not isinstance(evidence, torch.nn.Module):
            raise RuntimeError("V3.8 model lacks event bank/evidence module.")
        event_ids = {id(p) for mod in (bank, evidence) for p in mod.parameters()}
        if not event_ids:
            raise RuntimeError("V3.8 must have trainable event parameters.")
        # For P3, use the pre-existing requires_grad flags; do not force
        # unfreezing an unrelated backbone that was frozen by CRA's design.
        if self._v38_phase() == "probe":
            for p in model.parameters():
                p.requires_grad_(id(p) in event_ids)
        if not all(p.requires_grad for mod in (bank, evidence) for p in mod.parameters()):
            raise RuntimeError("All event bank/scorer parameters must be trainable.")

    @staticmethod
    def _normalized_state_dict(state):
        if not isinstance(state, dict) or not state:
            raise TypeError("Checkpoint state_dict must be a nonempty mapping.")
        normalized = {}
        for name, tensor in state.items():
            key = name[len("module."):] if name.startswith("module.") else name
            if key in normalized:
                raise RuntimeError(f"Duplicate checkpoint parameter after DP normalization: {key}")
            normalized[key] = tensor
        return normalized

    def _v38_architecture_signature(self):
        cfg = self._v38_settings()
        qcfg = self._qbound_config()
        return {
            "widths": tuple(int(w) for w in cfg.get("widths", (2, 4, 8, 16))),
            "context_radius": int(cfg.get("context_radius", 1)),
            "prior_mode": cfg.get("prior_mode", "scale_balanced"),
            "include_tail": cfg.get("include_tail", True),
            "centering": cfg.get("centering", "double"),
            "prior_temperature": float(cfg.get("prior_temperature", 1.0)),
            "interaction_temperature": float(cfg.get("interaction_temperature", 1.0)),
            "dropout": float(cfg.get("dropout", 0.1)),
            "detach_answer_embeddings": cfg.get("detach_answer_embeddings", True),
            "language_pooling_mode": qcfg.get("language_pooling_mode",
                self.cfgs["model"].get("language_pooling_mode", "masked")),
        }

    def _v38_training_signature(self):
        """Loss/data knobs that must NOT silently change on exact resume."""
        qcfg = self._qbound_config()
        model = self.cfgs["model"]
        ds = self.cfgs["dataset"]
        optim = self.cfgs["optim"]
        return {
            "grounding_loss": dict(qcfg.get("grounding_loss", {})),
            "predictive_loss": dict(qcfg.get("predictive_loss", {})),
            "pair_loss": dict(qcfg.get("pair_loss", {})),
            "evidence_loss": dict(self._v38_settings().get("evidence_loss", {})),
            "cra_loss": dict(optim.get("loss", {})),
            "cra_vg_loss": bool(model.get("vg_loss", False)),
            "d_model": int(model["d_model"]),
            "core_hidden_dim": int(qcfg.get("hidden_dim", 256)),
            "core_dropout": float(qcfg.get("dropout", model.get("dropout", 0.1))),
            "core_question_heads": int(qcfg.get("num_question_heads", 4)),
            "core_question_ff": int(qcfg.get("question_ff_multiplier", 2)),
            "dataset_features_path": ds.get("features_path"),
            "dataset_csv_path": ds.get("csv_path"),
            "dataset_mc": ds.get("mc"),
            "dataset_batch_size": ds.get("batch_size"),
            "dataset_max_feats": ds.get("max_feats"),
            "dataset_qmax_words": ds.get("qmax_words"),
            "dataset_amax_words": ds.get("amax_words"),
            "optim_qbound_lr": float(optim["qbound_lr"]),
            "optim_event_lr": float(optim.get("qbound_v38_lr", optim["qbound_lr"])),
            "optim_clip": optim.get("clip"),
        }

    def _v38_optimizer_parameter_names(self):
        names_by_id = {id(parameter): self._normalize_parameter_name(name)
                       for name, parameter in self.model.named_parameters()}
        return [
            {"name": group.get("name", f"group_{index}"),
             "parameters": [names_by_id.get(id(p), "<unknown>") for p in group["params"]]}
            for index, group in enumerate(self.optimizer.param_groups)
        ]

    def _initialize_v38_from_checkpoint(self, path):
        """Transfer ONLY model weights; always start epoch 1 + new optimizer.

        From R2-C/V3.5: EXACT missing keys are the new event modules.
        From an M0 probe: ALL keys must match; phase may change to joint.
        This is NOT stat.resume, which restores optimizer/scheduler/epoch.
        """
        path = str(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
            raise RuntimeError("Initialization checkpoint must contain state_dict.")
        metadata = checkpoint.get("qbound")
        if not isinstance(metadata, dict) or not metadata.get("enabled", False):
            raise RuntimeError("Safe V3.8 initialization requires checkpoint Q-Bound metadata.")
        source_version = str(metadata.get("version", ""))
        if source_version not in (self.QBOUND_VERSION, self.V38_EXPERIMENT_VERSION):
            raise RuntimeError(f"Unsupported initialization Q-Bound version {source_version!r}.")
        if (metadata.get("support_graph_mode") != "fixed" or
                metadata.get("support_predictor_enabled") is not False):
            raise RuntimeError("M0 only accepts R2-C sources (fixed graph, no predictor).")
        dataset = metadata.get("dataset_name")
        if dataset != self.cfgs["dataset"].get("name"):
            raise RuntimeError(f"Initialization dataset mismatch: source={dataset!r}.")
        if source_version == self.QBOUND_VERSION:
            # V3.5 metadata predates masked-pooling provenance. Require the
            # operator to identify the source mode rather than silently
            # changing question/answer embeddings during M0 comparison.
            declared_pool = self.cfgs["stat"].get("init_source_pooling_mode")
            source_pool = metadata.get("language_pooling_mode", declared_pool)
            if source_pool not in ("legacy", "masked"):
                raise RuntimeError("V3.5 -> M0 requires stat.init_source_pooling_mode "
                                   "('legacy' or 'masked') when source checkpoint "
                                   "does not record language pooling mode.")
            if (declared_pool is not None and source_pool != declared_pool):
                raise RuntimeError("User-declared source pooling contradicts checkpoint metadata.")
            if source_pool != self._v38_architecture_signature()["language_pooling_mode"]:
                raise RuntimeError("V3.5 source pooling mode and M0 pooling differ. "
                                   "Keep the baseline mode for a controlled comparison; "
                                   "run a separately labeled pooling ablation if desired.")
        if source_version == self.V38_EXPERIMENT_VERSION:
            source_v38 = metadata.get("v38")
            if not isinstance(source_v38, dict):
                raise RuntimeError("M0 source lacks v38 architecture metadata.")
            source_arch = source_v38.get("architecture")
            if source_arch is None or {
                k: tuple(v) if k == "widths" else v for k, v in source_arch.items()
            } != self._v38_architecture_signature():
                raise RuntimeError("M0 initialization architecture differs from source. "
                                   "Use an explicitly matched config/weights.")
        source_state = self._normalized_state_dict(checkpoint["state_dict"])
        dest = self._normalized_state_dict(self.model.state_dict())
        missing = set(dest).difference(source_state)
        unexpected = set(source_state).difference(dest)
        event_prefixes = ("qbound_v38_bank.", "qbound_v38_evidence.")
        expected_event = {k for k in dest if k.startswith(event_prefixes)}
        if unexpected:
            raise RuntimeError("Unexpected parameters in initialization: " + ", ".join(sorted(unexpected)[:20]))
        if source_version == self.QBOUND_VERSION:
            if missing != expected_event:
                raise RuntimeError("V3.5 -> M0 must lack EXACTLY new event keys; "
                                   f"missing={sorted(missing)}, expected={sorted(expected_event)}")
        elif missing:
            raise RuntimeError("M0 -> M0 transfer must contain ALL parameters: " + ", ".join(sorted(missing)))
        for name in set(source_state).intersection(dest):
            if not torch.is_tensor(source_state[name]) or source_state[name].shape != dest[name].shape:
                raise RuntimeError(f"Initialization tensor/shape mismatch: {name}")
        target_prefix = "module." if all(k.startswith("module.") for k in self.model.state_dict()) else ""
        target_keys = {target_prefix + k: v for k, v in source_state.items()}
        incompatible = self.model.load_state_dict(target_keys, strict=False)
        expected_missing = {target_prefix + k for k in missing}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError("Unexpected model incompatibility during initialization.")
        self.v38_initialization = {
            "source_path": os.path.abspath(path),
            "source_version": source_version,
            "source_variant": metadata.get("experiment_variant"),
            "source_phase": metadata.get("v38", {}).get("phase") if isinstance(metadata.get("v38"), dict) else None,
            "source_epoch": checkpoint.get("epoch"),
            "new_event_parameters": len(expected_missing),
        }
        print("[V3.8 weight-only initialization]", self.v38_initialization)
        print("  Starting epoch 1 with NEW optimizer/scheduler; no prior best or scheduler state.")

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
                and not (self._base_v38_m1_enabled() and
                         self._normalize_parameter_name(name).startswith(self.M1_EVENT_PREFIXES))
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

    def _qbound_experiment_variant(self):
        if not self._qbound_enabled():
            return "CRA"
        if self._base_v35_pair_settings()["enabled"]:
            return "V3.5-QDTG"
        return "R2-C" if not self._qbound_config().get("support_predictor_enabled", True) else "R2-A/B"

    # ------------------------------------------------------------------
    # Optimizer grouping
    # ------------------------------------------------------------------
    def _configure_qbound_optimizer_groups(self):
        """Place every QBoundCore parameter in one dedicated LR group."""
        if not self._qbound_enabled():
            return
        if self._v38_enabled():
            return self._configure_v38_optimizer_groups()

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
        if self._base_v38_m1_enabled():
            self._configure_m1_event_optimizer_group()

    def _m1_event_optimizer_options(self):
        optim = self.cfgs["optim"]
        lr = float(optim.get("qbound_v38_lr", optim["qbound_lr"]))
        wd = float(optim.get("qbound_v38_weight_decay", optim.get(
            "qbound_weight_decay", self.optimizer.defaults.get("weight_decay", 0.0))))
        if not math.isfinite(lr) or lr <= 0 or not math.isfinite(wd) or wd < 0:
            raise ValueError("M1 event optimizer needs positive LR and nonnegative weight decay.")
        return {"lr": lr, "weight_decay": wd}

    def _configure_m1_event_optimizer_group(self):
        model = self._model_core()
        head = getattr(model, "qbound_v38_m1_evidence", None)
        bank = getattr(model, "qbound_v38_m1_bank", None)
        if not isinstance(head, torch.nn.Module) or not isinstance(bank, torch.nn.Module):
            raise RuntimeError("Missing M1 CRA scorer/bank before optimizer setup.")
        params = list(head.parameters())
        head_ids = {id(p) for p in params}
        bank_ids = {id(p) for p in bank.parameters()}
        if (not params or not all(p.requires_grad for p in params) or head_ids & bank_ids or
                any(p.requires_grad for p in bank.parameters())):
            raise RuntimeError("M1 scorer must train independently of the frozen bank reader.")
        groups, extracted = [], set()
        for group in self.optimizer.param_groups:
            if group.get("name") == "qbound_v38_m1":
                raise RuntimeError("M1 event optimizer group already exists.")
            retained = []
            for p in group["params"]:
                if id(p) in head_ids:
                    if id(p) in extracted:
                        raise RuntimeError("M1 scorer parameter appears twice in the optimizer.")
                    extracted.add(id(p))
                elif id(p) not in bank_ids:
                    retained.append(p)
            if retained:
                group["params"] = retained
                groups.append(group)
        if extracted != head_ids:
            raise RuntimeError("build_optimizer did not register all M1 scorer parameters.")
        self.optimizer.param_groups = groups
        options = self._m1_event_optimizer_options()
        self.optimizer.add_param_group({"params": params, "name": "qbound_v38_m1", **options})
        print(f"[V3.8-M1 Optimizer] scorer_tensors={len(params)}, {options}")

    def _check_m1_event_optimizer(self, check_current_lr):
        model = self._model_core()
        params = list(model.qbound_v38_m1_evidence.parameters())
        group = [g for g in self.optimizer.param_groups if g.get("name") == "qbound_v38_m1"]
        if (len(group) != 1 or len(group[0]["params"]) != len(params) or
                {id(p) for p in group[0]["params"]} != {id(p) for p in params}):
            raise RuntimeError("M1 requires exactly one optimizer group containing only its MIL scorer.")
        bank_ids = {id(p) for p in model.qbound_v38_m1_bank.parameters()}
        if any(id(p) in bank_ids for g in self.optimizer.param_groups for p in g["params"]):
            raise RuntimeError("Frozen M1 bank reader must not be in the optimizer.")
        options = self._m1_event_optimizer_options()
        if not math.isclose(float(group[0]["weight_decay"]), options["weight_decay"], rel_tol=1e-9):
            raise RuntimeError("M1 event weight decay mismatch.")
        if check_current_lr and not math.isclose(float(group[0]["lr"]), options["lr"], rel_tol=1e-9):
            raise RuntimeError("M1 event learning rate mismatch.")

    def _configure_v38_optimizer_groups(self):
        """Separate new event params from legacy QBoundCore; phase-specific groups."""
        optim_cfg = self.cfgs.get("optim", {})
        if "qbound_lr" not in optim_cfg:
            raise ValueError("M0 requires explicit optim.qbound_lr.")
        core_lr = float(optim_cfg["qbound_lr"])
        event_lr = float(optim_cfg.get("qbound_v38_lr", core_lr))
        if not all(math.isfinite(v) and v > 0 for v in (core_lr, event_lr)):
            raise ValueError("M0 optimizer learning rates must be positive and finite.")
        event_wd = float(optim_cfg.get("qbound_v38_weight_decay",
                                        optim_cfg.get("qbound_weight_decay",
                                                      self.optimizer.defaults.get("weight_decay", 0.0))))
        if not math.isfinite(event_wd) or event_wd < 0:
            raise ValueError("M0 event weight_decay must be finite and >=0.")

        core_params = [(n, p) for n, p in self._named_trainable_qbound_parameters()
                       if "decoder.qbound." in self._normalize_parameter_name(n)]
        new_params = [(n, p) for n, p in self._named_trainable_qbound_parameters()
                      if self._normalize_parameter_name(n).startswith(
                          ("qbound_v38_bank.", "qbound_v38_evidence."))]
        expected = {id(p) for _, p in core_params + new_params}
        if len(expected) != len(core_params) + len(new_params) or not new_params:
            raise RuntimeError("Duplicated/missing V3.8 event parameters.")
        if self._v38_phase() == "probe" and core_params:
            raise RuntimeError("Probe must freeze QBoundCore BEFORE optimizer setup.")
        extracted = {}
        groups = []
        for group_index, group in enumerate(self.optimizer.param_groups):
            keep = []
            for p in group["params"]:
                if id(p) in expected:
                    if id(p) in extracted:
                        raise RuntimeError("Optimizer contains duplicated Q-Bound parameter.")
                    extracted[id(p)] = p
                elif self._v38_phase() == "probe" and not p.requires_grad:
                    # Remove frozen legacy parameters to keep the probe optimizer pure.
                    continue
                else:
                    keep.append(p)
            if keep:
                group["params"] = keep
                if not group.get("name") or group.get("name") in ("qbound", "qbound_v38"):
                    group["name"] = f"base_{group_index}"
                groups.append(group)
        if expected != set(extracted):
            missing = [n for n, p in core_params + new_params if id(p) not in extracted]
            raise RuntimeError("build_optimizer omitted V3.8 tensors: " + ", ".join(missing))
        self.optimizer.param_groups = groups
        if core_params:
            core_group = {"params": [p for _, p in core_params],
                          "lr": core_lr, "name": "qbound"}
            if "qbound_weight_decay" in optim_cfg:
                core_group["weight_decay"] = float(optim_cfg["qbound_weight_decay"])
            self.optimizer.add_param_group(core_group)
        self.optimizer.add_param_group({
            "params": [p for _, p in new_params], "lr": event_lr,
            "weight_decay": event_wd, "name": "qbound_v38"})
        print(f"[V3.8-M0 Optimizer] phase={self._v38_phase()}, "
              f"legacy_core_tensors={len(core_params)}, event_tensors={len(new_params)}, "
              f"core_lr={core_lr}, event_lr={event_lr}")

    def _check_v38_optimizer(self, check_current_lr=True):
        groups = {g.get("name"): g for g in self.optimizer.param_groups}
        if "qbound_v38" not in groups or sum(g.get("name") == "qbound_v38"
                                               for g in self.optimizer.param_groups) != 1:
            raise RuntimeError("M0 requires exactly one qbound_v38 optimizer group.")
        expected_event = {id(p) for name, p in self.model.named_parameters()
                          if p.requires_grad and self._normalize_parameter_name(name).startswith(
                              ("qbound_v38_bank.", "qbound_v38_evidence."))}
        actual_event = {id(p) for p in groups["qbound_v38"]["params"]}
        if not expected_event or expected_event != actual_event:
            raise RuntimeError("M0 event optimizer group has missing/extra parameters.")
        expected_core = {id(p) for name, p in self.model.named_parameters()
                         if p.requires_grad and "decoder.qbound." in self._normalize_parameter_name(name)}
        core_groups = [g for g in self.optimizer.param_groups if g.get("name") == "qbound"]
        if bool(expected_core) != bool(core_groups) or len(core_groups) > 1:
            raise RuntimeError("M0 QBoundCore optimizer group disagrees with train phase.")
        if expected_core and {id(p) for p in core_groups[0]["params"]} != expected_core:
            raise RuntimeError("M0 core optimizer group has missing/extra parameters.")
        if check_current_lr:
            for name, expected in (("qbound_v38", self.cfgs["optim"].get(
                                      "qbound_v38_lr", self.cfgs["optim"]["qbound_lr"])),
                                   ("qbound", self.cfgs["optim"]["qbound_lr"])):
                if name in groups and not math.isclose(float(groups[name]["lr"]),
                                                       float(expected), rel_tol=1e-9):
                    raise RuntimeError(f"M0 {name} learning rate mismatch.")
        self._check_optimizer_coverage()

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
        if self._v38_enabled():
            return self._check_v38_optimizer(check_current_lr=check_current_lr)
        if self._base_v38_m1_enabled():
            self._check_m1_event_optimizer(check_current_lr)

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
    def _check_qbound_v280_components(self, qbound_cfg, model_experiment_version=None):
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
            "pair_loss",  # V3.5 train-only objective; no new model parameters.
            "language_pooling_mode",  # V3.7 masked/legacy pooling checkpoint contract.
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
        if model_version != (model_experiment_version or version):
            raise RuntimeError(
                "BasePipeline/CRA version mismatch: "
                f"config={model_experiment_version or version!r}, model={model_version!r}."
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

        print("[Q-Bound V2.8 R2-A/B/C Component Check]")
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

    def _check_qbound_v38_components(self, qbound_cfg):
        """Fail closed on mixed M0/legacy source, phases, masks and modules."""
        model = self._model_core()
        decoder = getattr(model, "decoder", None)
        core = getattr(decoder, "qbound", None)
        if (getattr(model, "qbound_version", None) != self.V38_EXPERIMENT_VERSION or
                not getattr(model, "qbound_v38_minimal_enabled", False) or
                not getattr(decoder, "qbound_v38_minimal_enabled", False) or
                getattr(decoder, "qbound_version", None) != self.QBOUND_VERSION or
                getattr(core, "qbound_version", None) != self.QBOUND_VERSION):
            raise RuntimeError("M0 experiment version/wire-version mismatch across CRA/Refine/Core.")
        allowed = {"enabled", "version", "require_question_only", "require_token_question",
                   "detach_backbone_inputs", "hidden_dim", "dropout", "num_question_heads",
                   "question_ff_multiplier", "support_graph_mode", "support_predictor_enabled",
                   "grounding_loss", "predictive_loss", "pair_loss", "language_pooling_mode", "v38"}
        unknown = set(qbound_cfg) - allowed
        if unknown:
            raise ValueError("Unknown M0 model.qbound keys: " + ", ".join(sorted(unknown)))
        v38_cfg = self._v38_settings()
        allowed_v38 = {"widths", "context_radius", "prior_mode", "include_tail", "centering",
                       "prior_temperature", "interaction_temperature", "detach_answer_embeddings",
                       "dropout", "train_phase", "gradient_audit_every", "dump_joint_matrices",
                       "evidence_loss"}
        if set(v38_cfg) - allowed_v38:
            raise ValueError("Unknown M0 v38 keys: " + ", ".join(sorted(set(v38_cfg) - allowed_v38)))
        loss_cfg = v38_cfg.get("evidence_loss", {})
        if (not isinstance(loss_cfg, dict) or
                set(loss_cfg) - {"enabled", "loss_weight"} or
                loss_cfg.get("enabled") is not True or
                not math.isfinite(float(loss_cfg.get("loss_weight", 0))) or
                float(loss_cfg.get("loss_weight", 0)) <= 0):
            raise ValueError("M0 requires positive evidence_loss with enabled=true.")
        if not all(bool(qbound_cfg.get(key, True)) for key in
                   ("require_question_only", "require_token_question", "detach_backbone_inputs")):
            raise ValueError("M0 must preserve question-only, token-level, detached-input firewall.")
        if qbound_cfg.get("support_graph_mode") != "fixed" or qbound_cfg.get("support_predictor_enabled") is not False:
            raise ValueError("M0 requires R2-C (fixed graph, support predictor disabled).")
        if bool(self.cfgs["model"].get("do", False)):
            raise ValueError("M0 requires model.do=false.")
        if int(self.cfgs["dataset"].get("max_feats", 0)) < 3:
            raise ValueError("M0 needs dataset.max_feats>=3 for non-fallback event proposals.")
        self._validate_loss_block(qbound_cfg.get("grounding_loss"), "grounding_loss",
                                  {"enabled", "loss_weight", "temperature"}, require_temperature=True)
        self._validate_loss_block(qbound_cfg.get("predictive_loss"), "predictive_loss",
                                  {"enabled", "loss_weight"}, allow_disabled=True)
        if qbound_cfg.get("predictive_loss", {}).get("enabled", True):
            raise ValueError("R2-C predictive loss must be explicitly disabled.")
        if self._v38_phase() == "probe" and self._base_v35_pair_settings()["enabled"]:
            raise ValueError("M0 probe cannot train V3.5 pair loss.")
        if self._base_v35_pair_settings()["enabled"]:
            for obj, name in ((model, "refine_forward"), (decoder, "forward"),
                              (core, "forward_paired")):
                if not callable(getattr(obj, name, None)):
                    raise RuntimeError(f"M0 pair-loss source is missing {name}.")
        if any(hasattr(core, n) for n in ("support_input_proj", "support_forward_cell",
                                              "support_backward_cell", "support_forward_predictor",
                                              "support_backward_predictor")):
            raise RuntimeError("M0 R2-C must have NO predictor parameters.")
        for key in ("input_dim", "hidden_dim", "num_question_heads", "question_ff_multiplier"):
            expected = (int(self.cfgs["model"]["d_model"]) if key == "input_dim" else
                        int(qbound_cfg.get(key, 256 if key == "hidden_dim" else
                                            4 if key == "num_question_heads" else 2)))
            if int(getattr(core, key, -1)) != expected:
                raise RuntimeError(f"M0 QBoundCore.{key} differs from YAML.")
        for name in ("forward_v38_features", "decode_v38_event_support"):
            if not callable(getattr(core, name, None)):
                raise RuntimeError(f"M0 requires QBoundCore.{name}().")
        for name in ("forward_v38_encoder_features", "decode_v38_grounding_views"):
            if not callable(getattr(decoder, name, None)):
                raise RuntimeError(f"M0 requires RefineDecoder.{name}().")
        for name in ("_forward_v38_minimal",):
            if not callable(getattr(model, name, None)):
                raise RuntimeError(f"M0 requires CRA.{name}().")
        for name in ("qbound_v38_bank", "qbound_v38_evidence"):
            if not isinstance(getattr(model, name, None), torch.nn.Module):
                raise RuntimeError(f"M0 requires CRA.{name} module.")
        if getattr(model.lang_model, "pooling_mode", None) != self._v38_architecture_signature()["language_pooling_mode"]:
            raise RuntimeError("M0 language pooling mode mismatch; compare against checkpoint baseline.")
        bank, scorer = model.qbound_v38_bank, model.qbound_v38_evidence
        arch = self._v38_architecture_signature()
        if (tuple(bank.widths) != arch["widths"] or
                bank.context_radius != arch["context_radius"] or
                bank.prior_mode != arch["prior_mode"] or
                bank.include_tail != arch["include_tail"] or
                scorer.centering != arch["centering"] or
                not math.isclose(scorer.prior_temperature, arch["prior_temperature"]) or
                not math.isclose(scorer.interaction_temperature, arch["interaction_temperature"]) or
                not math.isclose(scorer.question_prior[2].p, arch["dropout"]) or
                scorer.answer_proj.in_features != int(self.cfgs["model"]["d_model"])):
            raise RuntimeError("M0 Event Bank/Evidence attributes differ from YAML.")
        unexpected_trainable = [name for name, p in model.named_parameters()
                                if p.requires_grad and "qbound" in name.lower()
                                and not name.startswith(("decoder.qbound.",
                                                         "qbound_v38_bank.",
                                                         "qbound_v38_evidence."))]
        if unexpected_trainable:
            raise RuntimeError("Unexpected trainable Q-Bound module path: " +
                               ", ".join(unexpected_trainable[:20]))
        if self._v38_phase() == "probe":
            frozen_in_optimizer = [p for group in self.optimizer.param_groups
                                   for p in group["params"] if not p.requires_grad]
            if frozen_in_optimizer:
                raise RuntimeError("M0 probe optimizer still contains frozen legacy parameters.")
        event_ids = {id(p) for m in (bank, scorer) for p in m.parameters()}
        if not event_ids or not all(p.requires_grad for m in (bank, scorer) for p in m.parameters()):
            raise RuntimeError("M0 event modules are not fully trainable.")
        if self._v38_phase() == "probe":
            leaked = [n for n, p in model.named_parameters() if id(p) not in event_ids and p.requires_grad]
            if leaked:
                raise RuntimeError("M0 probe has unexpectedly trainable old parameters: " + ", ".join(leaked[:20]))
        self._check_v38_optimizer()
        print(f"[V3.8-M0 Component Check] phase={self._v38_phase()}, "
              f"bank_widths={bank.widths}, new_parameters={sum(p.numel() for m in (bank, scorer) for p in m.parameters()):,}")

    def _check_qbound_components(self):
        if not self._qbound_enabled():
            return
        qbound_cfg = self._qbound_config()
        version = str(qbound_cfg.get("version", ""))
        if version == self.V38_EXPERIMENT_VERSION:
            return self._check_qbound_v38_components(qbound_cfg)
        if version == self.V38_M1_EXPERIMENT_VERSION:
            return self._check_qbound_v38_m1_components(qbound_cfg)
        if version != self.QBOUND_VERSION:
            raise ValueError(f"Unsupported Q-Bound experiment version: {version!r}.")
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
        if self._v38_enabled():
            if hasattr(self.lr_scheduler, "base_lrs"):
                rates = list(self.lr_scheduler.base_lrs)
                groups = self.optimizer.param_groups
                if len(rates) != len(groups):
                    raise RuntimeError("V3.8 scheduler/optimizer group count mismatch.")
                if check_base_lr:
                    for index, group in enumerate(groups):
                        if group.get("name") in ("qbound", "qbound_v38"):
                            wanted = self.cfgs["optim"].get("qbound_v38_lr",
                                        self.cfgs["optim"]["qbound_lr"]) if group["name"] == "qbound_v38" else self.cfgs["optim"]["qbound_lr"]
                            if not math.isclose(float(rates[index]), float(wanted), rel_tol=1e-9):
                                raise RuntimeError("V3.8 scheduler captured an incorrect learning rate.")
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
            if self._base_v38_m1_enabled():
                idx = next(i for i, g in enumerate(self.optimizer.param_groups)
                           if g.get("name") == "qbound_v38_m1")
                if not math.isclose(float(base_lrs[idx]), self._m1_event_optimizer_options()["lr"], rel_tol=1e-9):
                    raise RuntimeError("Scheduler did not capture the M1 scorer base LR.")

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
        if self._base_v38_m1_enabled():
            optim = self.cfgs["optim"]
            qcfg = self._qbound_config()
            # Stop/pause and diagnostics can change without changing the
            # optimizer objective, sampler or planned LR schedule.
            model_config = self._m1_model_training_signature()
            optim_config = copy.deepcopy({k: v for k, v in optim.items()
                                          if k not in ("stop_after_epoch", "save_period")})
            return {
                "enabled": True, "version": self.V38_M1_EXPERIMENT_VERSION,
                "core_wire_version": self.QBOUND_VERSION, "experiment_variant": "V3.8-M1",
                "support_graph_mode": "fixed", "support_predictor_enabled": False,
                "planned_epochs": self.epochs, "run_seed": self.cfgs["misc"].get("seed"),
                "dataset_name": self.cfgs["dataset"].get("name"),
                "optimizer_groups": self._optimizer_group_signature(),
                "optimizer_group_param_names": self._v38_optimizer_parameter_names(),
                "training_signature": {
                    "model": model_config, "optim": optim_config,
                    "dataset": copy.deepcopy(self.cfgs["dataset"]),
                    "updates_per_epoch": len(self.train_dataloader),
                    "legacy_alignment": "different_video_32_negatives",
                },
                "loss_weights": {
                    "grounding": float(qcfg.get("grounding_loss", {}).get("loss_weight", 1.0)),
                    "question": float(self._v38_settings().get("question_loss_weight", 1.0)),
                    "answer": float(self._v38_settings().get("answer_loss_weight", 1.0)),
                },
                "m1_grouped_batch": True,
                "v38_m1": {"initialization": self.v38_initialization,
                           "monitor": self.M1_MONITOR, "where": "cra_answer_conditional"},
                "loss_contract": "L_CRA/main + lambda_G*L_ground + lambda_Q*L_Q + lambda_A*L_A",
            }
        if self._v38_enabled():
            qbound = self._qbound_config()
            pair = self._base_v35_pair_settings()
            objective = "lambda_ev*L_evQA" if self._v38_phase() == "probe" else (
                "L_CRA/main + lambda_G*L_ground" +
                (" + lambda_pair*L_pair" if pair["enabled"] else "") +
                " + lambda_ev*L_evQA")
            return {
                "enabled": True,
                "version": self.V38_EXPERIMENT_VERSION,
                "core_wire_version": self.QBOUND_VERSION,
                "experiment_variant": "V3.8-Minimal-" + self._v38_phase(),
                "support_predictor_enabled": False,
                "support_graph_mode": "fixed",
                "planned_epochs": self.epochs,
                "run_seed": self.cfgs["misc"].get("seed"),
                "dataset_name": self.cfgs["dataset"].get("name"),
                "optimizer_groups": self._optimizer_group_signature(),
                "optimizer_group_param_names": self._v38_optimizer_parameter_names(),
                "training_signature": self._v38_training_signature(),
                "loss_weights": {
                    "grounding": float(qbound.get("grounding_loss", {}).get("loss_weight", 1.0)),
                    "predictive": 0.0,
                    "pair": pair["loss_weight"],
                    "evidence": float(self._v38_settings()["evidence_loss"]["loss_weight"]),
                },
                "pair_loss": {k: pair[k] for k in
                              ("enabled", "loss_weight", "temperature", "margin")},
                "v38": {"phase": self._v38_phase(),
                        "architecture": self._v38_architecture_signature(),
                        "initialization": self.v38_initialization},
                "loss_contract": objective,
            }
        enabled = self._qbound_enabled()
        qbound = self._qbound_config() if enabled else {}
        pair = self._base_v35_pair_settings() if enabled else None
        predictor = bool(qbound.get("support_predictor_enabled", True)) if enabled else False
        objective = None
        if enabled:
            objective = ("L_CRA/main + lambda_G*L_ground + lambda_P*L_pred"
                         if predictor else "L_CRA/main + lambda_G*L_ground")
            if pair["enabled"]:
                objective += " + lambda_pair*L_pair"
        return {
            **({"m1_grouped_batch": True} if self._base_m1_grouped_batch() else {}),
            "enabled": enabled,
            "version": self.QBOUND_VERSION if enabled else None,
            "experiment_variant": self._qbound_experiment_variant() if enabled else None,
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
        if self._base_v38_m1_enabled():
            return self._resume_v38_m1_optimizer_is_compatible(checkpoint, resume_strict)
        metadata = checkpoint.get("qbound")
        saved_grouped = bool(metadata.get("m1_grouped_batch", False)) if isinstance(metadata, dict) else False
        if saved_grouped != self._base_m1_grouped_batch():
            raise RuntimeError("Do not RESUME across grouped and ungrouped CRA objectives.")
        if isinstance(metadata, dict) and metadata.get("version") == self.V38_M1_EXPERIMENT_VERSION:
            raise RuntimeError("An M1 full checkpoint can only RESUME in M1.")
        if self._v38_enabled():
            return self._resume_v38_optimizer_is_compatible(checkpoint, resume_strict)
        if not self._qbound_enabled():
            return True

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
            if resume_strict:
                raise RuntimeError(
                    message
                    + f"\nSaved: {saved_signature}"
                    + f"\nCurrent: {current_signature}"
                )
            print("Warning:", message)
            return False
        return True

    def _resume_v38_m1_optimizer_is_compatible(self, checkpoint, resume_strict):
        saved = checkpoint.get("qbound")
        if (not resume_strict or not isinstance(saved, dict) or
                saved.get("version") != self.V38_M1_EXPERIMENT_VERSION):
            raise RuntimeError("M1 resume needs a full M1 checkpoint. For R2-C/M0 weights use stat.init_from_checkpoint.")
        current = self._qbound_checkpoint_metadata()
        for key, value in current.items():
            if key != "v38_m1" and saved.get(key) != value:
                raise RuntimeError(f"M1 resume metadata mismatch: {key}.")
        saved_m1 = saved.get("v38_m1")
        if not isinstance(saved_m1, dict) or any(
                saved_m1.get(k) != current["v38_m1"][k] for k in ("monitor", "where")):
            raise RuntimeError("M1 resume validation-selection contract changed.")
        self.v38_initialization = saved_m1.get("initialization")
        return True

    def _resume_v38_optimizer_is_compatible(self, checkpoint, resume_strict):
        """Only SAME-PHASE, SAME-OBJECTIVE M0 can resume optimizer + epoch.

        Cross-version (V3.5 -> M0) or cross-phase (probe -> joint) MUST use
        stat.init_from_checkpoint to start a fresh schedule.
        """
        if not resume_strict:
            raise RuntimeError("M0 requires strict full-state resume.")
        saved = checkpoint.get("qbound")
        if not isinstance(saved, dict) or saved.get("version") != self.V38_EXPERIMENT_VERSION:
            raise RuntimeError("V3.8 resume needs a full V3.8 checkpoint. "
                               "Use stat.init_from_checkpoint for V3.5 transfer.")
        current = self._qbound_checkpoint_metadata()
        for name in ("enabled", "version", "core_wire_version", "experiment_variant",
                     "support_graph_mode", "support_predictor_enabled", "planned_epochs",
                     "run_seed", "dataset_name", "optimizer_groups",
                     "optimizer_group_param_names", "training_signature", "loss_weights",
                     "pair_loss", "loss_contract"):
            if saved.get(name) != current.get(name):
                raise RuntimeError(f"M0 resume metadata {name} mismatch: "
                                   f"saved={saved.get(name)!r}, current={current.get(name)!r}. "
                                   "Use weight-only init for a new experiment.")
        saved_v38 = saved.get("v38")
        now_v38 = current["v38"]
        if not isinstance(saved_v38, dict):
            raise RuntimeError("M0 checkpoint lacks phase/architecture metadata.")
        if saved_v38.get("phase") != now_v38["phase"]:
            raise RuntimeError("M0 resume cannot switch probe/joint phase. "
                               "Use stat.init_from_checkpoint and a NEW optimizer.")
        saved_arch = saved_v38.get("architecture")
        if not isinstance(saved_arch, dict) or {
            k: tuple(v) if k == "widths" else v for k, v in saved_arch.items()
        } != now_v38["architecture"]:
            raise RuntimeError("M0 resume architecture changed. Start a new controlled run.")
        # Initialization provenance must survive exact continuation.
        if saved_v38.get("initialization") != now_v38["initialization"]:
            # During full resume this process has not performed weight-only init;
            # recover provenance from saved metadata without changing weights.
            if now_v38["initialization"] is not None:
                raise RuntimeError("M0 resume/initialization provenance conflict.")
            self.v38_initialization = saved_v38.get("initialization")
        return True

    # ------------------------------------------------------------------
    # Training lifecycle
    # ------------------------------------------------------------------
    def _update_grounding_best(self, epoch, score):
        """Select additional checkpoints ONLY by validation grounding scores."""
        if not self._qbound_enabled() or self._base_v38_m1_enabled():
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
        if self._v38_enabled():
            event = score.get("val_v38_joint_marginal_post-hoc")
            if not isinstance(event, dict):
                raise RuntimeError("M0 checkpoint selection requires VAL joint-marginal R1 results.")
            for metric in ("mIoU", "Acc&GQA"):
                if metric not in event:
                    raise KeyError(f"VAL joint-marginal metrics missing {metric!r}.")
                value = float(event[metric])
                if not math.isfinite(value):
                    raise RuntimeError(f"VAL joint-marginal {metric} is nonfinite.")
                selector = f"v38_joint_marginal_{metric}"
                if value > self.grounding_best[selector]["value"]:
                    self.grounding_best[selector] = {"value": value, "epoch": int(epoch)}
                    improved.append(selector)
        return tuple(improved)

    def train(self):
        if self._m2_eval_checkpoint_path is not None:
            raise RuntimeError(
                "stat.m2_eval_checkpoint is for frozen evaluation only. "
                "Call infer() instead of train()."
            )
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

            if self._base_v38_m1_enabled():
                if score is None or self.mnt_metric not in score:
                    raise RuntimeError(f"Missing M1 validation selection metric {self.mnt_metric!r}.")
                if not math.isfinite(float(score[self.mnt_metric])):
                    raise RuntimeError(f"Nonfinite M1 validation selection metric {self.mnt_metric!r}.")

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
            if self.grounding_best:
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
            elif metric_name == "v38_joint_marginal_mIoU":
                snapshots.append("model_best_v38_marginal_miou.pth")
            elif metric_name == "v38_joint_marginal_Acc&GQA":
                snapshots.append("model_best_v38_marginal_gqa.pth")
            else:
                raise ValueError(f"Unknown validation grounding checkpoint selector: {metric_name}")

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

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        if self._base_v38_m1_enabled() and not os.path.isfile(resume_path):
            raise FileNotFoundError(f"stat.resume does not exist: {resume_path}")
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

        if self._v38_enabled() or self._base_v38_m1_enabled():
            # Every model-selection decision above depends ONLY on val_*.
            # Store test metrics alongside the val-selected checkpoint, without
            # copying test results into the validation record.
            val_log = {k: v for k, v in log.items() if k.startswith("val_")}
            test_log = {k: v for k, v in log.items() if k.startswith("test_")}
            self.best_recorder["val"].update({"epoch": log["epoch"], **val_log})
            self.best_recorder["test"].update({"epoch": log["epoch"], **test_log,
                                               "selected_by": self.mnt_metric})
        else:
            self.best_recorder["val"].update(log)
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
