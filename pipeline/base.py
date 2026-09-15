# Base Pipeline
import os
import time
import torch
import json
import math
import pandas as pd
from numpy import inf
from models import model_fns
from utils import *

from utils.misc.misc import compute_a2v


class BasePipeline(object):
    def __init__(self, cfgs) -> None:
        self.cfgs = cfgs
        # set cuda device
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
        # build dataloader
        self.train_dataloader, self.val_dataloader, self.test_dataloader = build_dataloaders(cfgs, self.a2id, self.tokenizer)
        # build model
        self.model = model_fns[cfgs["model"]["name"]](cfgs, self.tokenizer).cuda()
        # DDP
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        # metric
        self.metric = Metric(cfgs)
        # loss
        self.criterion = BuildLossFunc(cfgs)
        # optim
        #
        # Build the repository's original optimizer first so that all
        # existing optimizer options remain unchanged. Then move the
        # Q-Bound parameters to their own parameter group.
        self.optimizer = build_optimizer(cfgs, self.model)
        self._configure_qbound_optimizer_groups()
        self._check_qbound_optimizer()
        self._check_qbound_components()
        self._check_optimizer_coverage()

        # Build the scheduler only after the parameter groups are final.
        self.lr_scheduler = build_lr_scheduler(
            cfgs,
            self.optimizer,
            len(self.train_dataloader),
        )
        self._check_qbound_scheduler()

        self.epochs = self.cfgs["optim"]["epochs"]
        self.save_period = self.cfgs["optim"]["save_period"]

        self.mnt_mode = cfgs["stat"]["monitor"]["mode"]
        self.mnt_metric = 'val_' + cfgs["stat"]["monitor"]["metric"]
        self.mnt_metric_test = 'test_' + cfgs["stat"]["monitor"]["metric"]
        assert self.mnt_mode in ['min', 'max']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = self.cfgs["stat"]["monitor"]["early_stop"]

        self.start_epoch = 1
        self.save_dir = os.path.join(cfgs["stat"]["record_dir"], cfgs["misc"]["running_name"])
        self.checkpoint_dir = os.path.join(self.save_dir, "checkpoint")

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        resume_path = cfgs["stat"].get("resume")
        if resume_path:
            self._resume_checkpoint(resume_path)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        # monitor
        self.monitor = Monitor(cfgs)
        self.monitor.log_info(json.dumps(cfgs,sort_keys=True,indent=4), end='\n')

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
        """
        Q-Bound trainable parameters live under:
            decoder.qbound.*

        DataParallel may prefix names with ``module.``; substring matching is
        therefore intentional.
        """
        return "qbound" in name.lower()

    def _configure_qbound_optimizer_groups(self):
        """
        Move all trainable Q-Bound core/adapter parameters into one dedicated
        optimizer group while preserving the repository's original optimizer.

        Required only when Q-Bound is enabled:
            optim.qbound_lr

        The grounding module and shared CRA projection remain base-model
        parameters. Only QBoundCore trainable parameters use the dedicated
        Q-Bound learning rate. In V2.0.0 this includes the generic input
        adapter, temporal-dynamics encoder, video-only boundary prior,
        query-adaptive functional composer, boundary-gated reasoner, sparse
        query-anchor reasoner, and direct start/end grounding heads.
        """
        if not self._qbound_enabled():
            return

        optim_cfg = self.cfgs.get("optim", {})

        qbound_lr = float(
            optim_cfg.get(
                "qbound_lr",
                optim_cfg.get("gpeb_lr", 0.0),
            )
        )

        if qbound_lr <= 0.0:
            raise ValueError(
                "Q-Bound is enabled but optim.qbound_lr is missing or "
                "non-positive."
            )

        named_qbound_parameters = [
            (name, parameter)
            for name, parameter
            in self.model.named_parameters()
            if (
                parameter.requires_grad
                and self._is_qbound_parameter_name(name)
            )
        ]

        if not named_qbound_parameters:
            raise RuntimeError(
                "Q-Bound is enabled, but no trainable parameter whose "
                "name contains 'qbound' was found. Check that the Q-Bound "
                "refine decoder is instantiated."
            )

        qbound_param_ids = {
            id(parameter)
            for _, parameter
            in named_qbound_parameters
        }

        retained_groups = []
        extracted_qbound_parameters = []
        extracted_ids = set()

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            retained_parameters = []

            for parameter in group["params"]:
                parameter_id = id(parameter)

                if parameter_id in qbound_param_ids:
                    if parameter_id not in extracted_ids:
                        extracted_qbound_parameters.append(
                            parameter
                        )
                        extracted_ids.add(
                            parameter_id
                        )
                else:
                    retained_parameters.append(
                        parameter
                    )

            if retained_parameters:
                group["params"] = retained_parameters
                group.setdefault(
                    "name",
                    f"base_{group_index}",
                )
                retained_groups.append(group)

        self.optimizer.param_groups = (
            retained_groups
        )

        missing_ids = (
            qbound_param_ids
            - extracted_ids
        )
        if missing_ids:
            missing_names = [
                name
                for name, parameter
                in named_qbound_parameters
                if id(parameter) in missing_ids
            ]
            raise RuntimeError(
                "Some Q-Bound parameters were not present in the "
                "optimizer returned by build_optimizer:\n"
                + "\n".join(missing_names)
            )

        qbound_group = {
            "params": extracted_qbound_parameters,
            "lr": qbound_lr,
            "name": "qbound",
        }

        if "qbound_weight_decay" in optim_cfg:
            qbound_group["weight_decay"] = float(
                optim_cfg["qbound_weight_decay"]
            )

        self.optimizer.add_param_group(
            qbound_group
        )

        # The last group must be the dedicated Q-Bound group.
        if self.optimizer.param_groups[-1].get("name") != "qbound":
            raise RuntimeError(
                "Failed to create the dedicated Q-Bound optimizer group."
            )

        total_qbound_params = sum(
            parameter.numel()
            for parameter
            in extracted_qbound_parameters
        )

        print("\n[Q-Bound Optimizer Group Created]")
        print(f"  lr: {qbound_lr}")
        print(
            "  weight_decay:",
            self.optimizer.param_groups[-1].get(
                "weight_decay",
                self.optimizer.defaults.get(
                    "weight_decay",
                    None,
                ),
            ),
        )
        print(
            "  trainable parameters: "
            f"{total_qbound_params:,}"
        )
        print(
            "  tensors: "
            f"{len(extracted_qbound_parameters)}\n"
        )

    def _check_qbound_optimizer(self):
        """
        Verify that every trainable Q-Bound parameter is registered exactly
        once and that the dedicated learning rate is correct.
        """
        if not self._qbound_enabled():
            return

        optimizer_parameter_counts = {}

        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                optimizer_parameter_counts[
                    parameter_id
                ] = (
                    optimizer_parameter_counts.get(
                        parameter_id,
                        0,
                    )
                    + 1
                )

        qbound_parameters = []
        missing_parameters = []
        duplicated_parameters = []

        for name, parameter in (
            self.model.named_parameters()
        ):
            if not self._is_qbound_parameter_name(
                name
            ):
                continue

            if not parameter.requires_grad:
                continue

            qbound_parameters.append(
                (name, parameter.numel())
            )

            registration_count = (
                optimizer_parameter_counts.get(
                    id(parameter),
                    0,
                )
            )

            if registration_count == 0:
                missing_parameters.append(name)
            elif registration_count > 1:
                duplicated_parameters.append(name)

        total_qbound_params = sum(
            numel
            for _, numel
            in qbound_parameters
        )

        print("\n[Q-Bound Optimizer Check]")
        print(
            "Trainable Q-Bound parameters: "
            f"{total_qbound_params:,}"
        )
        print(
            "Q-Bound tensors registered:",
            len(qbound_parameters),
        )
        print(
            "Optimizer parameter groups:",
            len(self.optimizer.param_groups),
        )

        qbound_group_indices = []

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            group_parameter_ids = {
                id(parameter)
                for parameter in group["params"]
            }

            num_qbound_tensors = sum(
                id(parameter)
                in group_parameter_ids
                for parameter_name, parameter
                in self.model.named_parameters()
                if (
                    parameter.requires_grad
                    and self._is_qbound_parameter_name(
                        parameter_name
                    )
                )
            )

            group_name = group.get(
                "name",
                f"group_{group_index}",
            )

            print(
                f"  Group {group_index} "
                f"({group_name}): "
                f"lr={group.get('lr')}, "
                f"num_tensors={len(group['params'])}, "
                f"qbound_tensors={num_qbound_tensors}"
            )

            if num_qbound_tensors > 0:
                qbound_group_indices.append(
                    group_index
                )

        if missing_parameters:
            raise RuntimeError(
                "The following Q-Bound parameters are missing "
                "from the optimizer:\n"
                + "\n".join(missing_parameters)
            )

        if duplicated_parameters:
            raise RuntimeError(
                "The following Q-Bound parameters occur more than once "
                "in the optimizer:\n"
                + "\n".join(duplicated_parameters)
            )

        if len(qbound_group_indices) != 1:
            raise RuntimeError(
                "Expected all Q-Bound tensors in exactly one optimizer "
                "group, but found them in groups: "
                f"{qbound_group_indices}."
            )

        actual_qbound_lr = (
            self.optimizer.param_groups[
                qbound_group_indices[0]
            ]["lr"]
        )

        expected_qbound_lr = float(
            self.cfgs["optim"].get(
                "qbound_lr",
                self.cfgs["optim"].get(
                    "gpeb_lr",
                    actual_qbound_lr,
                ),
            )
        )

        if not math.isclose(
            float(actual_qbound_lr),
            expected_qbound_lr,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "Q-Bound optimizer LR mismatch: "
                f"expected {expected_qbound_lr}, "
                f"got {actual_qbound_lr}."
            )

        print(
            "All Q-Bound parameters are registered exactly once."
        )
        print(
            "Verified Q-Bound learning rate: "
            f"{actual_qbound_lr}"
        )

        # Debug summary: all trainable Q-Bound tensors should live under
        # decoder.qbound.*. This catches accidental trainable Q-Bound tensors
        # added outside the dedicated core while remaining DataParallel-safe.
        prefix_counts = {}
        for name, parameter in self.model.named_parameters():
            if (
                parameter.requires_grad
                and self._is_qbound_parameter_name(name)
            ):
                normalized_name = (
                    name[len("module."):]
                    if name.startswith("module.")
                    else name
                )
                if ".qbound." in normalized_name:
                    prefix = normalized_name.split(".qbound.", 1)[0] + ".qbound"
                else:
                    prefix = normalized_name.rsplit(".", 1)[0]

                prefix_counts[prefix] = (
                    prefix_counts.get(prefix, 0) + 1
                )

        print("Q-Bound trainable tensor groups:")
        for prefix, count in sorted(prefix_counts.items()):
            print(f"  {prefix}: {count} tensors")
        print("")

    def _check_qbound_v200_components(self, qbound_cfg):
        """
        Structural safety check for Q-Bound V2.0.0.

        V2.0.0 is an architectural reset rather than another V1.x patch:

            raw/generic video
                -> temporal dynamics
                -> video-only boundary prior
                -> bounded question-conditioned residual
                -> query-specific soft cut field
                -> boundary-gated local reasoning + sparse query anchors
                -> direct start/end grounding
                -> local boundary-equilibrium supervision

        Core invariants checked here:
          1) no fixed event partition and no event-count capacity/target;
          2) no exhaustive temporal-hypothesis bank / teacher distillation;
          3) Q-Bound inference is video + question only;
          4) answer-aware KEEP/MERGE and equilibrium verifiers are
             parameter-free, stop-gradient training consumers;
          5) every trainable V2 mechanism is present and lives in the single
             ``qbound`` optimizer group;
          6) shared CRA ``decoder.video_proj`` remains a base-model parameter.

        Temporal GT is never a training input in this contract.
        """
        model_cfg = self.cfgs.get("model", {})
        optim_cfg = self.cfgs.get("optim", {})

        if str(model_cfg.get("baseline", "")) != "refine":
            raise RuntimeError(
                "Q-Bound V2.0.0 requires model.baseline='refine'."
            )

        if bool(model_cfg.get("do", False)):
            raise RuntimeError(
                "Controlled Q-Bound V2.0.0 requires model.do=false. "
                "CRA VisualIntervention would confound the functional "
                "KEEP/MERGE and boundary-equilibrium verifier signals."
            )

        if not bool(qbound_cfg.get("require_question_only", True)):
            raise RuntimeError(
                "Q-Bound V2.0.0 requires question-only grounding semantics. "
                "Candidate answers may be used by detached training verifiers "
                "but must not enter QBoundCore."
            )

        if not bool(qbound_cfg.get("detach_backbone_inputs", True)):
            raise RuntimeError(
                "Controlled Q-Bound V2.0.0 expects "
                "model.qbound.detach_backbone_inputs=true."
            )

        if not bool(qbound_cfg.get("use_generic_input_adapter", True)):
            raise RuntimeError(
                "Q-Bound V2.0.0 requires the Q-Bound-owned generic input "
                "adapter."
            )

        if not bool(qbound_cfg.get("input_adapter_use_position", True)):
            raise RuntimeError(
                "Controlled Q-Bound V2.0.0 expects positional encoding in "
                "the reasoning stream of the generic input adapter."
            )

        # --------------------------------------------------------------
        # Reject stale V1.x architecture knobs rather than silently ignoring
        # them.  In V2 event count is emergent from the cut field.
        # --------------------------------------------------------------
        forbidden_config_keys = (
            "max_events",
            "proposal_candidates",
            "use_temporal_prior",
            "question_modulation_scale",
            "boundary_temperature",
            "boundary_radius",
            "boundary_bias_init",
            "min_event_mass",
            "decomposition",
            "reasoning",
            "hypothesis_teacher",
            "weak_align_weight",
            "weak_align_temperature",
            "composer",
            "refinement",
            "counterfactual",
            "counterfactual_evidence_interface",
        )
        stale_keys = [
            key for key in forbidden_config_keys
            if key in qbound_cfg and qbound_cfg.get(key) not in (None, {}, False)
        ]
        if stale_keys:
            raise RuntimeError(
                "Q-Bound V2.0.0 YAML still contains V1.x configuration "
                "keys that must be removed: " + ", ".join(stale_keys)
            )

        hidden_dim = int(qbound_cfg.get("hidden_dim", 256))
        dropout = float(qbound_cfg.get("dropout", model_cfg.get("dropout", 0.1)))
        input_adapter_dropout = float(
            qbound_cfg.get("input_adapter_dropout", dropout)
        )
        cut_temperature = float(qbound_cfg.get("cut_temperature", 1.0))
        cut_threshold = float(qbound_cfg.get("cut_threshold", 0.5))
        question_residual_scale = float(
            qbound_cfg.get("question_residual_scale", 1.0)
        )
        num_anchor_heads = int(qbound_cfg.get("num_anchor_heads", 4))
        num_local_reasoning_layers = int(
            qbound_cfg.get("num_local_reasoning_layers", 2)
        )

        if hidden_dim < 1:
            raise ValueError("Q-Bound V2.0.0 hidden_dim must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("Q-Bound V2.0.0 dropout must be in [0,1).")
        if not (0.0 <= input_adapter_dropout < 1.0):
            raise ValueError(
                "Q-Bound V2.0.0 input_adapter_dropout must be in [0,1)."
            )
        if not math.isfinite(cut_temperature) or cut_temperature <= 0.0:
            raise ValueError(
                "Q-Bound V2.0.0 cut_temperature must be finite and > 0."
            )
        if not (0.0 < cut_threshold < 1.0):
            raise ValueError(
                "Q-Bound V2.0.0 cut_threshold must be in (0,1)."
            )
        if (
            not math.isfinite(question_residual_scale)
            or question_residual_scale <= 0.0
            or question_residual_scale > 1.0
        ):
            raise ValueError(
                "Controlled Q-Bound V2.0.0 requires "
                "0 < question_residual_scale <= 1 so the question can "
                "adjust, but not arbitrarily dominate, the video prior."
            )
        if num_anchor_heads < 1:
            raise ValueError(
                "Q-Bound V2.0.0 num_anchor_heads must be >= 1."
            )
        if num_local_reasoning_layers < 1:
            raise ValueError(
                "Q-Bound V2.0.0 num_local_reasoning_layers must be >= 1."
            )

        # Training objectives are intentionally compact.  These are objective
        # coefficients, not architectural event-count controls.
        comp_cfg = qbound_cfg.get("functional_composition", {})
        if not isinstance(comp_cfg, dict):
            raise TypeError(
                "model.qbound.functional_composition must be a mapping/dict."
            )
        eq_cfg = qbound_cfg.get("boundary_equilibrium", {})
        if not isinstance(eq_cfg, dict):
            raise TypeError(
                "model.qbound.boundary_equilibrium must be a mapping/dict."
            )

        comp_weight = float(comp_cfg.get("loss_weight", 0.10))
        comp_confidence_scale = float(comp_cfg.get("confidence_scale", 0.25))
        eq_weight = float(eq_cfg.get("loss_weight", 0.10))
        eq_confidence_scale = float(eq_cfg.get("confidence_scale", 0.25))

        for name, value in (
            ("functional_composition.loss_weight", comp_weight),
            ("boundary_equilibrium.loss_weight", eq_weight),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Q-Bound V2.0.0 {name} must be finite and >= 0."
                )
        for name, value in (
            ("functional_composition.confidence_scale", comp_confidence_scale),
            ("boundary_equilibrium.confidence_scale", eq_confidence_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"Q-Bound V2.0.0 {name} must be finite and > 0."
                )
        if comp_weight <= 0.0:
            raise RuntimeError(
                "The controlled V2.0.0 run requires a positive functional "
                "composition loss weight."
            )
        if eq_weight <= 0.0:
            raise RuntimeError(
                "The controlled V2.0.0 run requires a positive boundary "
                "equilibrium loss weight."
            )

        # --------------------------------------------------------------
        # Cross-file V2.0.0 contract.
        # --------------------------------------------------------------
        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )
        version = "2.0.0"

        model_version = str(getattr(model_core, "qbound_version", ""))
        if model_version != version:
            raise RuntimeError(
                "BasePipeline/model Q-Bound version mismatch: "
                f"config={version!r}, model={model_version!r}."
            )

        decoder = getattr(model_core, "decoder", None)
        decoder_version = str(getattr(decoder, "qbound_version", ""))
        if decoder_version != version:
            raise RuntimeError(
                "BasePipeline/RefineDecoder Q-Bound version mismatch: "
                f"config={version!r}, decoder={decoder_version!r}."
            )

        qbound_core = getattr(decoder, "qbound", None)
        if qbound_core is None:
            raise RuntimeError(
                "Q-Bound V2.0.0 is enabled but RefineDecoder.qbound is missing."
            )
        core_version = str(getattr(qbound_core, "qbound_version", ""))
        if core_version != version:
            raise RuntimeError(
                "BasePipeline/QBoundCore version mismatch: "
                f"config={version!r}, core={core_version!r}."
            )

        # V2 uses two detached, answer-space consequence scorers.  They are
        # methods on the CRA decoder, not trainable Q-Bound modules.
        for method_name in (
            "score_functional_composition_pairs",
            "score_boundary_equilibrium_views",
        ):
            if not callable(getattr(decoder, method_name, None)):
                raise RuntimeError(
                    f"RefineDecoder V2.0.0 must expose {method_name}()."
                )

        # The old V1.12 teacher interface must not survive the reset.
        if callable(getattr(decoder, "score_privileged_hypothesis_bank", None)):
            raise RuntimeError(
                "RefineDecoder V2.0.0 still exposes the V1.12 privileged "
                "hypothesis-bank teacher interface. Remove it."
            )

        expected_core_settings = {
            "hidden_dim": hidden_dim,
            "cut_temperature": cut_temperature,
            "cut_threshold": cut_threshold,
            "question_residual_scale": question_residual_scale,
            "num_anchor_heads": num_anchor_heads,
            "num_local_reasoning_layers": num_local_reasoning_layers,
        }
        for attribute_name, expected_value in expected_core_settings.items():
            actual_value = getattr(qbound_core, attribute_name, None)
            if actual_value is None:
                raise RuntimeError(
                    f"QBoundCore V2.0.0 is missing attribute {attribute_name!r}."
                )
            if isinstance(expected_value, float):
                if not math.isclose(
                    float(actual_value),
                    float(expected_value),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise RuntimeError(
                        "QBoundCore V2.0.0 setting mismatch for "
                        f"{attribute_name}: config={expected_value}, "
                        f"core={actual_value}."
                    )
            elif int(actual_value) != int(expected_value):
                raise RuntimeError(
                    "QBoundCore V2.0.0 setting mismatch for "
                    f"{attribute_name}: config={expected_value}, "
                    f"core={actual_value}."
                )

        if not bool(getattr(qbound_core, "use_generic_input_adapter", False)):
            raise RuntimeError(
                "QBoundCore V2.0.0 generic input adapter is not active."
            )
        if bool(getattr(qbound_core, "input_adapter_use_position", False)) != bool(
            qbound_cfg.get("input_adapter_use_position", True)
        ):
            raise RuntimeError(
                "QBoundCore V2.0.0 positional-input setting does not match "
                "the YAML configuration."
            )

        # No fixed K, partition bank, event-list reasoner, or 528-hypothesis
        # scorer may remain in the V2 core.
        forbidden_core_attributes = (
            "max_events",
            "proposal_candidates",
            "partition_patterns",
            "partition_split_count",
            "hypothesis_feature_encoder",
            "hypothesis_score",
            "ordered_relation_score",
            "event_encoder",
        )
        surviving_attributes = [
            name for name in forbidden_core_attributes
            if hasattr(qbound_core, name)
        ]
        if surviving_attributes:
            raise RuntimeError(
                "QBoundCore V2.0.0 still contains V1.x architecture state: "
                + ", ".join(surviving_attributes)
            )

        # --------------------------------------------------------------
        # Trainable-module / optimizer contract.
        # --------------------------------------------------------------
        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        qbound_named_parameters = [
            (name, parameter)
            for name, parameter in trainable_named_parameters
            if self._is_qbound_parameter_name(name)
        ]
        if not qbound_named_parameters:
            raise RuntimeError(
                "Q-Bound V2.0.0 has no trainable Q-Bound parameters."
            )

        qbound_group = next(
            (
                group
                for group in self.optimizer.param_groups
                if group.get("name") == "qbound"
            ),
            None,
        )
        if qbound_group is None:
            raise RuntimeError(
                "Q-Bound V2.0.0 optimizer group 'qbound' is missing."
            )
        qbound_group_ids = {
            id(parameter) for parameter in qbound_group["params"]
        }

        required_component_tokens = {
            "GENERIC_INPUT": (
                "generic_input_proj",
                "generic_input_norm",
            ),
            "TEMPORAL_DYNAMICS": (
                "state_proj",
                "delta_proj",
                "accel_proj",
                "dynamics_fuse",
            ),
            "VIDEO_BOUNDARY_PRIOR": (
                "video_boundary_encoder",
                "video_boundary_score",
            ),
            "FUNCTIONAL_COMPOSITION": (
                "question_proj",
                "functional_composer",
                "question_cut_residual",
            ),
            "LOCAL_BOUNDARY_GATED_REASONING": (
                "forward_scan_updates",
                "forward_scan_gates",
                "backward_scan_updates",
                "backward_scan_gates",
                "local_reasoning_fuse",
            ),
            "SPARSE_QUERY_ANCHORS": (
                "anchor_score",
                "anchor_query_proj",
                "anchor_key_proj",
                "anchor_value_proj",
                "anchor_update",
                "anchor_update_gate",
                "reasoning_norm",
            ),
            "DIRECT_GROUNDING": (
                "endpoint_encoder",
                "start_score",
                "end_score",
                "frame_evidence_score",
            ),
        }

        missing_tokens = []
        misplaced_parameters = []
        component_summary = {}

        for component_name, tokens in required_component_tokens.items():
            component_parameter_ids = set()
            component_tensor_names = set()

            for token in tokens:
                matched = [
                    (name, parameter)
                    for name, parameter in qbound_named_parameters
                    if token in name
                ]
                if not matched:
                    missing_tokens.append(
                        f"{component_name}: missing trainable token {token!r}"
                    )
                    continue

                for name, parameter in matched:
                    component_parameter_ids.add(id(parameter))
                    component_tensor_names.add(name)
                    if id(parameter) not in qbound_group_ids:
                        misplaced_parameters.append(name)

            component_summary[component_name] = {
                "tensors": len(component_tensor_names),
                "parameters": sum(
                    parameter.numel()
                    for _, parameter in qbound_named_parameters
                    if id(parameter) in component_parameter_ids
                ),
            }

        if missing_tokens:
            raise RuntimeError(
                "Q-Bound V2.0.0 expected trainable components are missing:\n"
                + "\n".join(missing_tokens)
            )
        if misplaced_parameters:
            raise RuntimeError(
                "Q-Bound V2.0.0 trainable tensors were found outside the "
                "'qbound' optimizer group:\n"
                + "\n".join(sorted(set(misplaced_parameters)))
            )

        # Reject trainable remnants of the V1.x event-partition, hypothesis,
        # calibration, teacher, or answer-input machinery.  Functional verifier
        # methods themselves are parameter-free and therefore do not appear here.
        forbidden_qbound_tokens = (
            "decomp_question_gate",
            "topology_proj",
            "topology_qa_proj",
            "boundary_feature_encoder",
            "boundary_classifier",
            "event_encoder",
            "ordered_relation",
            "event_evidence",
            "hypothesis_",
            "span_composer",
            "refiner_",
            "calibration_",
            "reliability_",
            "counterfactual_",
            "donor_",
            "teacher_",
            "critic_",
            "answer_proj",
            "answer_encoder",
            "answer_feature",
        )
        unexpected_parameters = [
            name
            for name, _ in qbound_named_parameters
            if any(
                token in name.lower()
                for token in forbidden_qbound_tokens
            )
        ]
        if unexpected_parameters:
            raise RuntimeError(
                "Q-Bound V2.0.0 contains trainable V1.x/privileged "
                "parameters that should not exist:\n"
                + "\n".join(unexpected_parameters)
            )

        expected_qbound_lr = float(optim_cfg.get("qbound_lr", 0.0))
        actual_qbound_lr = float(qbound_group["lr"])
        if expected_qbound_lr <= 0.0:
            raise ValueError(
                "Q-Bound V2.0.0 requires optim.qbound_lr > 0."
            )
        if not math.isclose(
            actual_qbound_lr,
            expected_qbound_lr,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "Q-Bound V2.0.0 component-group LR mismatch: "
                f"expected {expected_qbound_lr}, got {actual_qbound_lr}."
            )

        shared_video_proj_in_qbound_group = []
        for name, parameter in trainable_named_parameters:
            normalized_name = (
                name[len("module."):]
                if name.startswith("module.")
                else name
            )
            if (
                "decoder.video_proj" in normalized_name
                and ".qbound." not in normalized_name
                and id(parameter) in qbound_group_ids
            ):
                shared_video_proj_in_qbound_group.append(name)
        if shared_video_proj_in_qbound_group:
            raise RuntimeError(
                "Shared CRA video_proj must remain in the base optimizer "
                "group, but these tensors were found in Q-Bound:\n"
                + "\n".join(shared_video_proj_in_qbound_group)
            )

        # There must be no separately trainable verifier/teacher module outside
        # QBoundCore.  The verifier is a detached consumer of CRA answer space.
        forbidden_global_trainable_tokens = (
            "functional_verifier",
            "equilibrium_verifier",
            "hypothesis_teacher",
            "privileged_hypothesis",
        )
        unexpected_global = [
            name for name, _ in trainable_named_parameters
            if any(token in name.lower() for token in forbidden_global_trainable_tokens)
        ]
        if unexpected_global:
            raise RuntimeError(
                "V2.0.0 verifier/teacher logic must be parameter-free, but "
                "trainable tensors were found:\n" + "\n".join(unexpected_global)
            )

        max_feats = int(self.cfgs.get("dataset", {}).get("max_feats", 0))
        print("[Q-Bound V2.0.0 Component Check]")
        print("  fixed event partition: NO")
        print("  event-count cap / target: NO")
        print("  exhaustive hypothesis bank: NO")
        print("  teacher distribution / KL distillation: NO")
        print("  answer features inside QBoundCore: NO")
        print("  temporal GT training supervision: NO")
        print(
            "  video-only prior -> bounded question residual -> soft cut field: YES"
        )
        print(
            "  functional KEEP/MERGE verifier: YES (parameter-free, stop-gradient)"
        )
        print(
            "  boundary-equilibrium verifier: YES (parameter-free, stop-gradient)"
        )
        print(
            "  local composition complexity: O(T); sparse global reasoning: "
            f"O(T*R), R={num_anchor_heads}"
        )
        if max_feats > 0:
            print(
                "  dataset temporal resolution: "
                f"T<={max_feats}; semantic event count remains emergent"
            )
        print(
            "  question_residual_scale: "
            f"{question_residual_scale}"
        )
        print(
            "  cut_temperature / threshold: "
            f"{cut_temperature} / {cut_threshold}"
        )
        print(
            "  functional composition loss weight / confidence scale: "
            f"{comp_weight} / {comp_confidence_scale}"
        )
        print(
            "  boundary equilibrium loss weight / confidence scale: "
            f"{eq_weight} / {eq_confidence_scale}"
        )
        for component_name, summary in component_summary.items():
            print(
                f"  {component_name}: "
                f"{summary['tensors']} tensors, "
                f"{summary['parameters']:,} parameters"
            )
        print("  shared CRA video_proj remains in base optimizer group: YES")
        print(
            "  all V2.0.0 Q-Bound trainable components use qbound_lr: "
            f"{actual_qbound_lr}\n"
        )

    def _check_qbound_v112_components(self, qbound_cfg):
        """
        Structural safety check for Q-Bound V1.12.

        V1.12 is a deliberate architectural reset:

            generic visual input
                -> dynamic DECOMPOSE
                -> Q-OEI RELATE
                -> exact frame-level temporal hypothesis bank
                -> question-only intra-video hypothesis scorer
                -> hard winning hypothesis

        Candidate-answer semantics are privileged TRAINING-ONLY supervision.
        They are consumed by RefineDecoder's stop-gradient critic and must not
        appear as trainable Q-Bound inputs or parameters.

        This check therefore verifies four invariants:
          1) event count is capacity-bounded but dynamic;
          2) the hypothesis scorer is present and trainable with qbound_lr;
          3) legacy COMPOSE/BAPGR/PPBR/PSRC/donor machinery is absent;
          4) the answer-aware critic is parameter-free and answer information
             is not part of QBoundCore's inference path.
        """
        model_cfg = self.cfgs.get("model", {})
        optim_cfg = self.cfgs.get("optim", {})

        if str(model_cfg.get("baseline", "")) != "refine":
            raise RuntimeError(
                "Q-Bound V1.12 requires model.baseline='refine'."
            )

        if bool(model_cfg.get("do", False)):
            raise RuntimeError(
                "Controlled Q-Bound V1.12 requires model.do=false. "
                "CRA VisualIntervention would confound the privileged "
                "hypothesis critic."
            )

        if not bool(qbound_cfg.get("require_question_only", True)):
            raise RuntimeError(
                "Q-Bound V1.12 requires question-only grounding semantics. "
                "Answer features may supervise the training-only critic but "
                "must not enter QBoundCore."
            )

        if bool(qbound_cfg.get("use_temporal_prior", False)):
            raise RuntimeError(
                "Q-Bound V1.12 must not consume a CRA temporal prior."
            )

        if not bool(qbound_cfg.get("detach_backbone_inputs", True)):
            raise RuntimeError(
                "Controlled Q-Bound V1.12 expects "
                "model.qbound.detach_backbone_inputs=true."
            )

        if not bool(qbound_cfg.get("use_generic_input_adapter", True)):
            raise RuntimeError(
                "Q-Bound V1.12 requires the Q-Bound-owned generic input "
                "adapter."
            )

        if not bool(qbound_cfg.get("input_adapter_use_position", True)):
            raise RuntimeError(
                "Q-Bound V1.12 requires positional encoding in the generic "
                "input adapter."
            )

        max_events = int(qbound_cfg.get("max_events", 6))
        proposal_candidates = int(qbound_cfg.get("proposal_candidates", 8))
        if max_events < 2:
            raise ValueError(
                "Q-Bound V1.12 max_events must be >= 2. It is a capacity "
                "ceiling, not a forced event count."
            )
        if proposal_candidates < max_events - 1:
            raise ValueError(
                "Q-Bound V1.12 proposal_candidates must be >= max_events-1 "
                "so every event count up to the capacity can be represented."
            )

        decomposition_cfg = qbound_cfg.get("decomposition", {})
        if not isinstance(decomposition_cfg, dict):
            raise TypeError(
                "model.qbound.decomposition must be a mapping/dict."
            )
        if not bool(decomposition_cfg.get("enabled", True)):
            raise RuntimeError(
                "Q-Bound V1.12 requires dynamic DECOMPOSE to remain enabled."
            )

        reasoning_cfg = qbound_cfg.get("reasoning", {})
        if not isinstance(reasoning_cfg, dict):
            raise TypeError("model.qbound.reasoning must be a mapping/dict.")
        if not bool(reasoning_cfg.get("enabled", True)):
            raise RuntimeError(
                "Q-Bound V1.12 requires Q-OEI RELATE to remain enabled."
            )
        reasoning_type = str(
            reasoning_cfg.get("type", "ordered_event_interaction")
        )
        if reasoning_type != "ordered_event_interaction":
            raise RuntimeError(
                "Q-Bound V1.12 reasoning.type must be "
                "'ordered_event_interaction'; got "
                f"{reasoning_type!r}."
            )

        teacher_cfg = qbound_cfg.get("hypothesis_teacher", {})
        if not isinstance(teacher_cfg, dict):
            raise TypeError(
                "model.qbound.hypothesis_teacher must be a mapping/dict."
            )
        if not bool(teacher_cfg.get("enabled", True)):
            raise RuntimeError(
                "The controlled Q-Bound V1.12 run requires the privileged "
                "answer-aware hypothesis teacher."
            )
        teacher_loss_weight = float(teacher_cfg.get("loss_weight", 0.10))
        if (
            not math.isfinite(teacher_loss_weight)
            or teacher_loss_weight <= 0.0
        ):
            raise ValueError(
                "Q-Bound V1.12 hypothesis_teacher.loss_weight must be "
                "finite and > 0."
            )

        # V1.12 intentionally removes these V1.11 mechanism blocks.  Reject
        # stale YAML rather than silently ignoring it.
        for legacy_block in ("composer", "refinement", "counterfactual"):
            legacy_value = qbound_cfg.get(legacy_block)
            if legacy_value not in (None, {}):
                raise RuntimeError(
                    "Q-Bound V1.12 YAML still contains legacy block "
                    f"model.qbound.{legacy_block}. Remove it for a clean "
                    "V1.12 run."
                )

        if bool(qbound_cfg.get("counterfactual_evidence_interface", False)):
            raise RuntimeError(
                "Q-Bound V1.12 must keep the V1.11 donor-counterfactual "
                "interface disabled."
            )

        # --------------------------------------------------------------
        # Cross-file V1.12 contract.
        # --------------------------------------------------------------
        model_core = (
            self.model.module
            if hasattr(self.model, "module")
            else self.model
        )

        version = "1.12"
        model_version = str(getattr(model_core, "qbound_version", ""))
        if model_version != version:
            raise RuntimeError(
                "BasePipeline/model Q-Bound version mismatch: "
                f"config={version!r}, model={model_version!r}."
            )

        decoder = getattr(model_core, "decoder", None)
        decoder_version = str(getattr(decoder, "qbound_version", ""))
        if decoder_version != version:
            raise RuntimeError(
                "BasePipeline/RefineDecoder Q-Bound version mismatch: "
                f"config={version!r}, decoder={decoder_version!r}."
            )

        qbound_core = getattr(decoder, "qbound", None)
        if qbound_core is None:
            raise RuntimeError(
                "Q-Bound V1.12 is enabled but RefineDecoder.qbound is missing."
            )
        core_version = str(getattr(qbound_core, "qbound_version", ""))
        if core_version != version:
            raise RuntimeError(
                "BasePipeline/QBoundCore version mismatch: "
                f"config={version!r}, core={core_version!r}."
            )

        if not callable(
            getattr(decoder, "score_privileged_hypothesis_bank", None)
        ):
            raise RuntimeError(
                "RefineDecoder V1.12 must expose "
                "score_privileged_hypothesis_bank()."
            )

        actual_max_events = int(getattr(qbound_core, "max_events", -1))
        actual_proposals = int(
            getattr(qbound_core, "proposal_candidates", -1)
        )
        if actual_max_events != max_events:
            raise RuntimeError(
                "QBoundCore V1.12 max_events mismatch: "
                f"config={max_events}, core={actual_max_events}."
            )
        if actual_proposals != proposal_candidates:
            raise RuntimeError(
                "QBoundCore V1.12 proposal_candidates mismatch: "
                f"config={proposal_candidates}, core={actual_proposals}."
            )
        if bool(getattr(qbound_core, "use_temporal_prior", True)):
            raise RuntimeError(
                "QBoundCore V1.12 unexpectedly enables a temporal prior."
            )
        if not bool(
            getattr(qbound_core, "use_generic_input_adapter", False)
        ):
            raise RuntimeError(
                "QBoundCore V1.12 generic input adapter is not active."
            )
        if bool(
            getattr(qbound_core, "input_adapter_use_position", False)
        ) != bool(qbound_cfg.get("input_adapter_use_position", True)):
            raise RuntimeError(
                "QBoundCore V1.12 positional-input setting does not match "
                "the YAML configuration."
            )

        # The partition bank must contain every split count from 0 to
        # K_max-1.  This verifies that max_events is a capacity ceiling rather
        # than a hidden fixed-event assignment.
        partition_split_count = getattr(
            qbound_core, "partition_split_count", None
        )
        if not torch.is_tensor(partition_split_count):
            raise RuntimeError(
                "QBoundCore V1.12 partition_split_count buffer is missing."
            )
        observed_split_counts = set(
            int(value)
            for value in partition_split_count.detach().cpu().tolist()
        )
        expected_split_counts = set(range(max_events))
        if observed_split_counts != expected_split_counts:
            raise RuntimeError(
                "QBoundCore V1.12 does not represent all dynamic event "
                "counts. Expected split counts "
                f"{sorted(expected_split_counts)}, got "
                f"{sorted(observed_split_counts)}."
            )

        expected_partition_count = sum(
            math.comb(proposal_candidates, split_count)
            for split_count in range(max_events)
        )
        actual_partition_count = int(partition_split_count.numel())
        if actual_partition_count != expected_partition_count:
            raise RuntimeError(
                "QBoundCore V1.12 partition-bank size mismatch: "
                f"expected {expected_partition_count}, "
                f"got {actual_partition_count}."
            )

        # Numeric decomposition settings must agree with the instantiated core.
        expected_decomp = {
            "decomp_complexity_weight": float(
                decomposition_cfg.get("complexity_weight", 1.25)
            ),
            "decomp_boundary_weight": float(
                decomposition_cfg.get("boundary_weight", 0.50)
            ),
            "decomp_temperature": float(
                decomposition_cfg.get("temperature", 0.20)
            ),
        }
        for attribute_name, expected_value in expected_decomp.items():
            actual_value = float(
                getattr(qbound_core, attribute_name, float("nan"))
            )
            if not math.isclose(
                actual_value,
                expected_value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "QBoundCore V1.12 decomposition setting mismatch for "
                    f"{attribute_name}: config={expected_value}, "
                    f"core={actual_value}."
                )

        # --------------------------------------------------------------
        # Trainable-module / optimizer contract.
        # --------------------------------------------------------------
        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        qbound_named_parameters = [
            (name, parameter)
            for name, parameter in trainable_named_parameters
            if self._is_qbound_parameter_name(name)
        ]
        if not qbound_named_parameters:
            raise RuntimeError(
                "Q-Bound V1.12 has no trainable Q-Bound parameters."
            )

        qbound_group = next(
            (
                group
                for group in self.optimizer.param_groups
                if group.get("name") == "qbound"
            ),
            None,
        )
        if qbound_group is None:
            raise RuntimeError(
                "Q-Bound V1.12 optimizer group 'qbound' is missing."
            )
        qbound_group_ids = {
            id(parameter) for parameter in qbound_group["params"]
        }

        # Every named token below represents an intentional V1.12 trainable
        # mechanism.  Requiring all of them catches partial file replacement.
        required_component_tokens = {
            "GENERIC_INPUT": (
                "generic_input_proj",
                "generic_input_norm",
            ),
            "DECOMPOSE": (
                "decomp_question_gate",
                "topology_proj",
                "topology_qa_proj",
                "boundary_feature_encoder",
                "boundary_classifier",
            ),
            "EVENT_ENCODING": (
                "raw_proj",
                "context_proj",
                "qa_proj",
                "qa_film",
                "event_encoder",
            ),
            "RELATE": (
                "ordered_relation_score",
                "ordered_value_proj",
                "ordered_event_update",
                "ordered_update_gate",
                "ordered_event_norm",
            ),
            "EVIDENCE": (
                "event_evidence_score",
                "frame_evidence_encoder",
                "frame_evidence_score",
            ),
            "HYPOTHESIS_SCORER": (
                "hypothesis_feature_encoder",
                "hypothesis_score",
            ),
        }

        missing_tokens = []
        misplaced_parameters = []
        component_summary = {}

        for component_name, tokens in required_component_tokens.items():
            component_parameter_ids = set()
            component_tensor_names = set()

            for token in tokens:
                matched = [
                    (name, parameter)
                    for name, parameter in qbound_named_parameters
                    if token in name
                ]
                if not matched:
                    missing_tokens.append(
                        f"{component_name}: missing trainable token {token!r}"
                    )
                    continue

                for name, parameter in matched:
                    component_parameter_ids.add(id(parameter))
                    component_tensor_names.add(name)
                    if id(parameter) not in qbound_group_ids:
                        misplaced_parameters.append(name)

            component_summary[component_name] = {
                "tensors": len(component_tensor_names),
                "parameters": sum(
                    parameter.numel()
                    for _, parameter in qbound_named_parameters
                    if id(parameter) in component_parameter_ids
                ),
            }

        if missing_tokens:
            raise RuntimeError(
                "Q-Bound V1.12 expected trainable components are missing:\n"
                + "\n".join(missing_tokens)
            )
        if misplaced_parameters:
            raise RuntimeError(
                "Q-Bound V1.12 trainable tensors were found outside the "
                "'qbound' optimizer group:\n"
                + "\n".join(sorted(set(misplaced_parameters)))
            )

        # These names correspond to mechanisms explicitly removed in V1.12,
        # or to a forbidden trainable privileged critic/answer input.
        forbidden_qbound_tokens = (
            "span_composer",
            "boundary_refine_encoder",
            "boundary_start_score",
            "boundary_end_score",
            "refiner_",
            "calibration_",
            "reliability_",
            "counterfactual_",
            "donor_",
            "teacher_",
            "critic_",
            "answer_proj",
            "answer_encoder",
            "answer_feature",
        )
        unexpected_parameters = [
            name
            for name, _ in qbound_named_parameters
            if any(
                token in name.lower()
                for token in forbidden_qbound_tokens
            )
        ]
        if unexpected_parameters:
            raise RuntimeError(
                "Q-Bound V1.12 contains trainable legacy/privileged "
                "parameters that should not exist:\n"
                + "\n".join(unexpected_parameters)
            )

        expected_qbound_lr = float(optim_cfg.get("qbound_lr", 0.0))
        actual_qbound_lr = float(qbound_group["lr"])
        if expected_qbound_lr <= 0.0:
            raise ValueError(
                "Q-Bound V1.12 requires optim.qbound_lr > 0."
            )
        if not math.isclose(
            actual_qbound_lr,
            expected_qbound_lr,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "Q-Bound V1.12 component-group LR mismatch: "
                f"expected {expected_qbound_lr}, got {actual_qbound_lr}."
            )

        shared_video_proj_in_qbound_group = []
        for name, parameter in trainable_named_parameters:
            normalized_name = (
                name[len("module."):]
                if name.startswith("module.")
                else name
            )
            if (
                "decoder.video_proj" in normalized_name
                and ".qbound." not in normalized_name
                and id(parameter) in qbound_group_ids
            ):
                shared_video_proj_in_qbound_group.append(name)
        if shared_video_proj_in_qbound_group:
            raise RuntimeError(
                "Shared CRA video_proj must remain in the base optimizer "
                "group, but these tensors were found in Q-Bound:\n"
                + "\n".join(shared_video_proj_in_qbound_group)
            )

        max_feats = int(self.cfgs.get("dataset", {}).get("max_feats", 0))
        exact_hypothesis_count = (
            max_feats * (max_feats + 1) // 2
            if max_feats > 0
            else None
        )

        print("[Q-Bound V1.12 Component Check]")
        print(
            "  dynamic event capacity: "
            f"K_i in [1,{max_events}] (not fixed to K_max)"
        )
        print(
            "  boundary proposal slots: "
            f"{proposal_candidates}"
        )
        print(
            "  latent partition patterns: "
            f"{actual_partition_count}"
        )
        if exact_hypothesis_count is not None:
            print(
                "  frame-exact contiguous hypotheses at dataset max_feats="
                f"{max_feats}: {exact_hypothesis_count}"
            )
        for component_name, summary in component_summary.items():
            print(
                f"  {component_name}: "
                f"{summary['tensors']} tensors, "
                f"{summary['parameters']:,} parameters"
            )
        print(
            "  privileged answer-aware critic trainable parameters: 0 "
            "(stop-gradient consumer)"
        )
        print("  answer features inside QBoundCore: NO")
        print("  legacy COMPOSE/BAPGR/PPBR/PSRC/donor path: DISABLED")
        print("  shared CRA video_proj remains in base optimizer group: YES")
        print(
            "  hypothesis teacher: enabled, "
            f"loss_weight={teacher_loss_weight}, "
            f"correct_only={bool(teacher_cfg.get('correct_only', True))}"
        )
        print(
            "  all V1.12 Q-Bound trainable components use qbound_lr: "
            f"{actual_qbound_lr}\n"
        )

    def _check_qbound_components(self):
        """
        Structural safety check for Q-Bound experiments.

        V2.0.0 has its own clean architecture contract and is dispatched to
        ``_check_qbound_v200_components``. V1.12 and older checks are retained
        below only so historical experiment configs remain loadable.

        V1.11 keeps the successful V1.10.2 temporal mechanism:

            DECOMPOSE -> RELATE -> COMPOSE -> BAPGR -> PPBR -> PSRC -> GROUND

        and adds:
          * a Q-Bound-owned generic visual input adapter;
          * optionally (V1.11-B) a parameter-free counterfactual evidence
            interface with answer-level sufficiency/necessity supervision.

        Important optimizer contract:
          * V1.11-B adds NO new trainable counterfactual module;
          * all trainable Q-Bound tensors remain in the existing ``qbound``
            optimizer group and use ``optim.qbound_lr``;
          * shared CRA ``decoder.video_proj`` remains in the base group;
          * BAPGR anchor power and PPBR correction cap remain fixed structural
            constants rather than learned gates;
          * the CF evidence route must remain parameter-free and Visual
            Intervention must remain disabled in the controlled V1.11-B run.
        """
        if not self._qbound_enabled():
            return

        model_cfg = self.cfgs.get("model", {})
        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            qbound_cfg = {}

        version = str(qbound_cfg.get("version", ""))

        if version == "2.0.0":
            self._check_qbound_v200_components(qbound_cfg)
            return

        if version == "1.12":
            self._check_qbound_v112_components(qbound_cfg)
            return

        refinement_cfg = qbound_cfg.get("refinement", {})
        if not isinstance(refinement_cfg, dict):
            refinement_cfg = {}

        # Keep BasePipeline reusable for older/other experimental versions.
        if version not in {"1.8", "1.9", "1.10.1", "1.10.2", "1.11"}:
            return

        if version in {"1.9", "1.10.1", "1.10.2", "1.11"}:
            expected_routing_flags = {
                "hard_coarse_route": False,
                "detach_coarse": True,
                "posterior_guided": True,
                "posterior_stable_reliability": True,
            }

            if version in {"1.10.1", "1.10.2", "1.11"}:
                expected_routing_flags["boundary_structured"] = True

            for key, expected in expected_routing_flags.items():
                if version == "1.11":
                    # V1.11 modules already default to these fixed routing
                    # choices. Allow omission in a future compact/paper YAML,
                    # but fail if an explicitly supplied value changes them.
                    actual = bool(
                        refinement_cfg.get(key, expected)
                    )
                else:
                    if key not in refinement_cfg:
                        raise RuntimeError(
                            f"Q-Bound V{version} config is missing "
                            f"refinement.{key}. Expected {expected!r}."
                        )
                    actual = bool(refinement_cfg[key])

                if actual != expected:
                    raise RuntimeError(
                        f"Q-Bound V{version} routing config mismatch for "
                        f"refinement.{key}: expected {expected!r}, "
                        f"got {actual!r}."
                    )

        if version == "1.10.1":
            expected_refinement_type = (
                "boundary_anchored_posterior_guided_reliability_refiner"
            )
            actual_refinement_type = str(
                refinement_cfg.get("type", "")
            )
            if actual_refinement_type != expected_refinement_type:
                raise RuntimeError(
                    "Q-Bound V1.10.1 refinement.type mismatch: "
                    f"expected {expected_refinement_type!r}, "
                    f"got {actual_refinement_type!r}."
                )

            if "boundary_anchor_power" not in refinement_cfg:
                raise RuntimeError(
                    "Q-Bound V1.10.1 config is missing "
                    "refinement.boundary_anchor_power."
                )

            boundary_anchor_power = float(
                refinement_cfg["boundary_anchor_power"]
            )
            if (
                not math.isfinite(boundary_anchor_power)
                or boundary_anchor_power <= 0.0
            ):
                raise ValueError(
                    "Q-Bound V1.10.1 refinement.boundary_anchor_power "
                    "must be finite and > 0; got "
                    f"{boundary_anchor_power}."
                )

            # Controlled first V1.10.1 run uses p=2.0. Fail fast if this
            # config is accidentally changed, so V1.9 -> V1.10.1 remains a
            # clean one-factor comparison.
            if not math.isclose(
                boundary_anchor_power,
                2.0,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.10.1 expects "
                    "refinement.boundary_anchor_power=2.0; got "
                    f"{boundary_anchor_power}."
                )

            # CRA wrapper and RefineDecoder already check this themselves;
            # BasePipeline verifies it again after optional DataParallel
            # wrapping so silent cross-version pairing is impossible.
            model_core = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            model_version = str(
                getattr(model_core, "qbound_version", "")
            )
            if model_version != version:
                raise RuntimeError(
                    "BasePipeline/model Q-Bound version mismatch: "
                    f"config={version!r}, model={model_version!r}."
                )

            decoder = getattr(model_core, "decoder", None)
            decoder_version = str(
                getattr(decoder, "qbound_version", "")
            )
            if decoder_version != version:
                raise RuntimeError(
                    "BasePipeline/RefineDecoder Q-Bound version mismatch: "
                    f"config={version!r}, decoder={decoder_version!r}."
                )

        if version == "1.10.2":
            expected_refinement_type = (
                "prior_preserving_bounded_refiner"
            )
            actual_refinement_type = str(
                refinement_cfg.get("type", "")
            )
            if actual_refinement_type != expected_refinement_type:
                raise RuntimeError(
                    "Q-Bound V1.10.2 refinement.type mismatch: "
                    f"expected {expected_refinement_type!r}, "
                    f"got {actual_refinement_type!r}."
                )

            if "boundary_anchor_power" not in refinement_cfg:
                raise RuntimeError(
                    "Q-Bound V1.10.2 config is missing "
                    "refinement.boundary_anchor_power."
                )
            boundary_anchor_power = float(
                refinement_cfg["boundary_anchor_power"]
            )
            if (
                not math.isfinite(boundary_anchor_power)
                or boundary_anchor_power <= 0.0
            ):
                raise ValueError(
                    "Q-Bound V1.10.2 refinement.boundary_anchor_power "
                    "must be finite and > 0; got "
                    f"{boundary_anchor_power}."
                )

            if "correction_cap" not in refinement_cfg:
                raise RuntimeError(
                    "Q-Bound V1.10.2 config is missing "
                    "refinement.correction_cap."
                )
            correction_cap = float(
                refinement_cfg["correction_cap"]
            )
            if (
                not math.isfinite(correction_cap)
                or correction_cap <= 0.0
            ):
                raise ValueError(
                    "Q-Bound V1.10.2 refinement.correction_cap "
                    "must be finite and > 0; got "
                    f"{correction_cap}."
                )

            if not math.isclose(
                boundary_anchor_power,
                2.0,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.10.2 expects "
                    "refinement.boundary_anchor_power=2.0; got "
                    f"{boundary_anchor_power}."
                )
            if not math.isclose(
                correction_cap,
                2.0,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.10.2 expects "
                    "refinement.correction_cap=2.0; got "
                    f"{correction_cap}."
                )

            model_core = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            model_version = str(
                getattr(model_core, "qbound_version", "")
            )
            if model_version != version:
                raise RuntimeError(
                    "BasePipeline/model Q-Bound version mismatch: "
                    f"config={version!r}, model={model_version!r}."
                )

            decoder = getattr(model_core, "decoder", None)
            decoder_version = str(
                getattr(decoder, "qbound_version", "")
            )
            if decoder_version != version:
                raise RuntimeError(
                    "BasePipeline/RefineDecoder Q-Bound version mismatch: "
                    f"config={version!r}, decoder={decoder_version!r}."
                )

            qbound_core = getattr(decoder, "qbound", None)
            qbound_core_version = str(
                getattr(qbound_core, "qbound_version", "")
            )
            if qbound_core_version != version:
                raise RuntimeError(
                    "BasePipeline/QBoundCore version mismatch: "
                    f"config={version!r}, core={qbound_core_version!r}."
                )

            actual_core_cap = float(
                getattr(
                    qbound_core,
                    "refiner_correction_cap",
                    float("nan"),
                )
            )
            if not math.isclose(
                actual_core_cap,
                correction_cap,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "BasePipeline/QBoundCore correction-cap mismatch: "
                    f"config={correction_cap}, core={actual_core_cap}."
                )

            actual_core_anchor_power = float(
                getattr(
                    qbound_core,
                    "refiner_boundary_anchor_power",
                    float("nan"),
                )
            )
            if not math.isclose(
                actual_core_anchor_power,
                boundary_anchor_power,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "BasePipeline/QBoundCore boundary-anchor mismatch: "
                    f"config={boundary_anchor_power}, "
                    f"core={actual_core_anchor_power}."
                )

        if version == "1.11":
            # ----------------------------------------------------------
            # V1.11 family contract
            # ----------------------------------------------------------
            expected_refinement_type = "prior_preserving_bounded_refiner"
            actual_refinement_type = str(
                refinement_cfg.get("type", expected_refinement_type)
            )
            if actual_refinement_type != expected_refinement_type:
                raise RuntimeError(
                    "Q-Bound V1.11 refinement.type mismatch: "
                    f"expected {expected_refinement_type!r}, "
                    f"got {actual_refinement_type!r}."
                )

            # These are fixed structural constants for the controlled V1.11
            # family, not free/tuned hyperparameters. They may be omitted from
            # a future compact YAML because all V1.11 modules already default
            # to the same values.
            boundary_anchor_power = float(
                refinement_cfg.get("boundary_anchor_power", 2.0)
            )
            correction_cap = float(
                refinement_cfg.get("correction_cap", 2.0)
            )

            if (
                not math.isfinite(boundary_anchor_power)
                or boundary_anchor_power <= 0.0
            ):
                raise ValueError(
                    "Q-Bound V1.11 refinement.boundary_anchor_power "
                    "must be finite and > 0; got "
                    f"{boundary_anchor_power}."
                )
            if (
                not math.isfinite(correction_cap)
                or correction_cap <= 0.0
            ):
                raise ValueError(
                    "Q-Bound V1.11 refinement.correction_cap "
                    "must be finite and > 0; got "
                    f"{correction_cap}."
                )

            if not math.isclose(
                boundary_anchor_power,
                2.0,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.11 expects fixed "
                    "refinement.boundary_anchor_power=2.0; got "
                    f"{boundary_anchor_power}."
                )
            if not math.isclose(
                correction_cap,
                2.0,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.11 expects fixed "
                    "refinement.correction_cap=2.0; got "
                    f"{correction_cap}."
                )

            if not bool(
                qbound_cfg.get("use_generic_input_adapter", True)
            ):
                raise RuntimeError(
                    "Q-Bound V1.11 requires "
                    "model.qbound.use_generic_input_adapter=true."
                )

            if not bool(
                qbound_cfg.get("input_adapter_use_position", True)
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.11 expects positional encoding "
                    "inside the generic input adapter."
                )

            # ------------------------------------------------------
            # Counterfactual settings.
            #
            # V1.11-A:
            #   interface=false -> generic-input baseline.
            #
            # V1.11-B:
            #   interface=true + counterfactual.training_enabled=true.
            #
            # The CF weights/margin are objective coefficients, not new
            # architectural modules. Defaults are centralized here so a
            # future compact YAML may omit them without changing behavior.
            # ------------------------------------------------------
            configured_cf_interface = bool(
                qbound_cfg.get(
                    "counterfactual_evidence_interface",
                    False,
                )
            )

            counterfactual_cfg = qbound_cfg.get(
                "counterfactual",
                {},
            )
            if counterfactual_cfg is None:
                counterfactual_cfg = {}
            if not isinstance(counterfactual_cfg, dict):
                raise TypeError(
                    "model.qbound.counterfactual must be a mapping/dict."
                )

            cf_training_enabled = bool(
                counterfactual_cfg.get(
                    "training_enabled",
                    configured_cf_interface,
                )
            )
            cf_sufficiency_weight = float(
                counterfactual_cfg.get(
                    "sufficiency_weight",
                    0.05,
                )
            )
            cf_necessity_weight = float(
                counterfactual_cfg.get(
                    "necessity_weight",
                    0.05,
                )
            )
            cf_necessity_margin = float(
                counterfactual_cfg.get(
                    "necessity_margin",
                    0.20,
                )
            )
            cf_necessity_correct_only = bool(
                counterfactual_cfg.get(
                    "necessity_correct_only",
                    True,
                )
            )

            if cf_training_enabled and not configured_cf_interface:
                raise RuntimeError(
                    "Q-Bound V1.11 counterfactual training cannot be enabled "
                    "while counterfactual_evidence_interface=false."
                )
            if cf_sufficiency_weight < 0.0:
                raise ValueError(
                    "Q-Bound V1.11 counterfactual.sufficiency_weight "
                    "must be >= 0."
                )
            if cf_necessity_weight < 0.0:
                raise ValueError(
                    "Q-Bound V1.11 counterfactual.necessity_weight "
                    "must be >= 0."
                )
            if cf_necessity_margin < 0.0:
                raise ValueError(
                    "Q-Bound V1.11 counterfactual.necessity_margin "
                    "must be >= 0."
                )
            if (
                cf_training_enabled
                and cf_sufficiency_weight <= 0.0
                and cf_necessity_weight <= 0.0
            ):
                raise RuntimeError(
                    "Q-Bound V1.11-B enables counterfactual training but "
                    "both CF loss weights are zero."
                )

            # Controlled V1.11-B keeps VisualIntervention OFF so the
            # counterfactual consumer is not confounded by a second causal
            # intervention mechanism.
            if (
                configured_cf_interface
                and bool(model_cfg.get("do", False))
            ):
                raise RuntimeError(
                    "Controlled Q-Bound V1.11-B requires model.do=false."
                )

            # ------------------------------------------------------
            # Cross-file version/interface consistency.
            # ------------------------------------------------------
            model_core = (
                self.model.module
                if hasattr(self.model, "module")
                else self.model
            )
            model_version = str(
                getattr(model_core, "qbound_version", "")
            )
            if model_version != version:
                raise RuntimeError(
                    "BasePipeline/model Q-Bound version mismatch: "
                    f"config={version!r}, model={model_version!r}."
                )

            decoder = getattr(model_core, "decoder", None)
            decoder_version = str(
                getattr(decoder, "qbound_version", "")
            )
            if decoder_version != version:
                raise RuntimeError(
                    "BasePipeline/RefineDecoder Q-Bound version mismatch: "
                    f"config={version!r}, decoder={decoder_version!r}."
                )

            qbound_core = getattr(decoder, "qbound", None)
            qbound_core_version = str(
                getattr(qbound_core, "qbound_version", "")
            )
            if qbound_core_version != version:
                raise RuntimeError(
                    "BasePipeline/QBoundCore version mismatch: "
                    f"config={version!r}, core={qbound_core_version!r}."
                )

            actual_core_cap = float(
                getattr(
                    qbound_core,
                    "refiner_correction_cap",
                    float("nan"),
                )
            )
            if not math.isclose(
                actual_core_cap,
                correction_cap,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "BasePipeline/QBoundCore correction-cap mismatch: "
                    f"config={correction_cap}, core={actual_core_cap}."
                )

            actual_core_anchor_power = float(
                getattr(
                    qbound_core,
                    "refiner_boundary_anchor_power",
                    float("nan"),
                )
            )
            if not math.isclose(
                actual_core_anchor_power,
                boundary_anchor_power,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "BasePipeline/QBoundCore boundary-anchor mismatch: "
                    f"config={boundary_anchor_power}, "
                    f"core={actual_core_anchor_power}."
                )

            if not bool(
                getattr(
                    qbound_core,
                    "use_generic_input_adapter",
                    False,
                )
            ):
                raise RuntimeError(
                    "QBoundCore V1.11 generic input adapter is not active."
                )

            actual_position_setting = bool(
                getattr(
                    qbound_core,
                    "input_adapter_use_position",
                    False,
                )
            )
            if actual_position_setting != bool(
                qbound_cfg.get("input_adapter_use_position", True)
            ):
                raise RuntimeError(
                    "QBoundCore V1.11 positional-input setting does not "
                    "match the YAML configuration."
                )

            actual_core_cf_interface = bool(
                getattr(
                    qbound_core,
                    "counterfactual_evidence_interface",
                    False,
                )
            )
            if actual_core_cf_interface != configured_cf_interface:
                raise RuntimeError(
                    "QBoundCore V1.11 counterfactual interface setting "
                    "does not match the YAML configuration."
                )

            actual_decoder_cf_interface = bool(
                getattr(
                    decoder,
                    "qbound_counterfactual_evidence_interface",
                    False,
                )
            )
            if actual_decoder_cf_interface != configured_cf_interface:
                raise RuntimeError(
                    "RefineDecoder V1.11 counterfactual interface setting "
                    "does not match the YAML configuration."
                )

            actual_model_cf_interface = bool(
                getattr(
                    model_core,
                    "qbound_counterfactual_evidence_interface",
                    False,
                )
            )
            if actual_model_cf_interface != configured_cf_interface:
                raise RuntimeError(
                    "CRA V1.11 counterfactual interface setting does not "
                    "match the YAML configuration."
                )

        required_component_tokens = {
            "DECOMPOSE": ("boundary_classifier",),
            "RELATE": ("ordered_relation_score",),
            "GROUND": ("event_evidence_score",),
            "COMPOSE": ("span_composer_score",),
        }

        if bool(refinement_cfg.get("enabled", True)):
            required_component_tokens["REFINE"] = (
                "boundary_refine_encoder",
                "boundary_start_score",
                "boundary_end_score",
            )

        if version == "1.11":
            required_component_tokens["GENERIC_INPUT"] = (
                "generic_input_proj",
                "generic_input_norm",
            )

        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]

        qbound_group = None
        for group in self.optimizer.param_groups:
            if group.get("name") == "qbound":
                qbound_group = group
                break

        if qbound_group is None:
            raise RuntimeError(
                f"Q-Bound V{version} is enabled but optimizer group "
                "'qbound' is missing."
            )

        qbound_group_ids = {
            id(parameter)
            for parameter in qbound_group["params"]
        }

        missing_components = []
        misplaced_parameters = []
        component_summary = {}

        for component_name, tokens in required_component_tokens.items():
            matched = [
                (name, parameter)
                for name, parameter in trainable_named_parameters
                if (
                    self._is_qbound_parameter_name(name)
                    and any(token in name for token in tokens)
                )
            ]

            if not matched:
                missing_components.append(
                    f"{component_name}: expected one of {tokens}"
                )
                continue

            component_summary[component_name] = {
                "tensors": len(matched),
                "parameters": sum(
                    parameter.numel()
                    for _, parameter in matched
                ),
            }

            for name, parameter in matched:
                if id(parameter) not in qbound_group_ids:
                    misplaced_parameters.append(name)

        if missing_components:
            raise RuntimeError(
                f"Q-Bound V{version} expected trainable components are missing:\n"
                + "\n".join(missing_components)
            )

        if misplaced_parameters:
            raise RuntimeError(
                f"Q-Bound V{version} trainable tensors found outside "
                "optimizer group 'qbound':\n"
                + "\n".join(misplaced_parameters)
            )

        expected_qbound_lr = float(self.cfgs["optim"]["qbound_lr"])
        actual_qbound_lr = float(qbound_group["lr"])

        if not math.isclose(
            actual_qbound_lr,
            expected_qbound_lr,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"Q-Bound V{version} component group LR mismatch: "
                f"expected {expected_qbound_lr}, got {actual_qbound_lr}."
            )

        forbidden_trainable_tokens = (
            "calibration_gate",
            "calibration_score",
            "uncertainty_gate",
            "uncertainty_score",
            "reliability_gate",
            "reliability_score",
            "posterior_gate",
            "posterior_score",
        )

        if version == "1.11":
            # V1.11-B counterfactual validation is intentionally parameter-free:
            # the only trainable objects are the existing Q-Bound modules.
            forbidden_trainable_tokens = (
                *forbidden_trainable_tokens,
                "counterfactual_gate",
                "counterfactual_score",
                "counterfactual_proj",
                "counterfactual_adapter",
            )

        if version in {"1.10.1", "1.10.2", "1.11"}:
            # BAPGR's boundary anchor is intentionally a fixed structural
            # prior. A trainable "boundary_anchor*" tensor would silently
            # turn the controlled experiment into a different model.
            forbidden_trainable_tokens = (
                *forbidden_trainable_tokens,
                "boundary_anchor",
            )

        if version in {"1.10.2", "1.11"}:
            # PPBR's cap is a fixed structural trust-region setting.
            forbidden_trainable_tokens = (
                *forbidden_trainable_tokens,
                "correction_cap",
                "correction_gate",
                "correction_score",
            )
        unexpected_routing_parameters = [
            name
            for name, parameter in trainable_named_parameters
            if (
                self._is_qbound_parameter_name(name)
                and any(
                    token in name.lower()
                    for token in forbidden_trainable_tokens
                )
            )
        ]

        if unexpected_routing_parameters:
            raise RuntimeError(
                f"Q-Bound V{version} controlled routing/calibration is "
                "intended to be parameter-free, but trainable routing "
                "parameters were found:\n"
                + "\n".join(unexpected_routing_parameters)
            )

        shared_video_proj_in_qbound_group = []
        for name, parameter in trainable_named_parameters:
            normalized_name = (
                name[len("module."):]
                if name.startswith("module.")
                else name
            )

            if (
                "decoder.video_proj" in normalized_name
                and ".qbound." not in normalized_name
                and id(parameter) in qbound_group_ids
            ):
                shared_video_proj_in_qbound_group.append(name)

        if shared_video_proj_in_qbound_group:
            raise RuntimeError(
                "Shared CRA video_proj must remain in the base optimizer group, "
                "but these tensors were found in the Q-Bound group:\n"
                + "\n".join(shared_video_proj_in_qbound_group)
            )

        print(f"[Q-Bound V{version} Component Check]")
        for component_name, summary in component_summary.items():
            print(
                f"  {component_name}: "
                f"{summary['tensors']} tensors, "
                f"{summary['parameters']:,} parameters"
            )

        if version == "1.9":
            print("  detached posterior-guided refinement config: YES")
            print("  posterior-stable reliability config: YES")

        if version == "1.10.1":
            print("  detached posterior-guided refinement config: YES")
            print("  boundary-structured BAPGR config: YES")
            print(
                "  boundary_anchor_power: "
                f"{float(refinement_cfg['boundary_anchor_power'])}"
            )
            print("  posterior-stable reliability config: YES")

        if version == "1.10.2":
            print("  detached posterior-guided refinement config: YES")
            print("  boundary-structured BAPGR config: YES")
            print(
                "  boundary_anchor_power: "
                f"{float(refinement_cfg['boundary_anchor_power'])}"
            )
            print("  prior-preserving bounded refinement (PPBR): YES")
            print(
                "  correction_cap: "
                f"{float(refinement_cfg['correction_cap'])}"
            )
            print("  posterior-stable reliability config: YES")

        if version == "1.11":
            cf_cfg = qbound_cfg.get("counterfactual", {})
            if not isinstance(cf_cfg, dict):
                cf_cfg = {}
            cf_interface = bool(
                qbound_cfg.get(
                    "counterfactual_evidence_interface",
                    False,
                )
            )
            cf_training = bool(
                cf_cfg.get(
                    "training_enabled",
                    cf_interface,
                )
            )

            print("  generic Q-Bound input adapter: YES")
            print(
                "  input_adapter_use_position: "
                f"{bool(qbound_cfg.get('input_adapter_use_position', True))}"
            )
            print(
                "  counterfactual_evidence_interface: "
                f"{cf_interface}"
            )
            print(
                "  counterfactual_training_enabled: "
                f"{cf_training}"
            )
            if cf_training:
                print(
                    "  CF weights (suf/nec): "
                    f"{float(cf_cfg.get('sufficiency_weight', 0.05))}/"
                    f"{float(cf_cfg.get('necessity_weight', 0.05))}"
                )
                print(
                    "  CF necessity_margin: "
                    f"{float(cf_cfg.get('necessity_margin', 0.20))}"
                )
                print(
                    "  CF necessity_correct_only: "
                    f"{bool(cf_cfg.get('necessity_correct_only', True))}"
                )
            print("  detached posterior-guided refinement config: YES")
            print("  boundary-structured BAPGR config: YES")
            print(
                "  boundary_anchor_power (fixed): "
                f"{float(refinement_cfg.get('boundary_anchor_power', 2.0))}"
            )
            print("  prior-preserving bounded refinement (PPBR): YES")
            print(
                "  correction_cap (fixed): "
                f"{float(refinement_cfg.get('correction_cap', 2.0))}"
            )
            print("  posterior-stable reliability config: YES")

        print("  routing/calibration is parameter-free: YES")
        print("  shared CRA video_proj remains in base optimizer group: YES")
        print(
            f"  all V{version} Q-Bound trainable components use qbound_lr: "
            f"{actual_qbound_lr}\n"
        )

    # Backward-compatible alias for older debug code.
    def _check_qbound_v18_components(self):
        return self._check_qbound_components()


    def _check_optimizer_coverage(self):
        """
        Verify that every trainable model parameter appears exactly once.

        The Q-Bound-specific check is necessary but not sufficient: mutating
        optimizer groups must not accidentally drop or duplicate ordinary
        CRA, language, video, or grounding parameters.
        """
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
            "[Optimizer Coverage Check] "
            "Every trainable parameter is registered exactly once.\n"
        )

    def _check_qbound_scheduler(self):
        """
        Verify that the scheduler was constructed after the dedicated Q-Bound
        optimizer group and sees the same number of parameter groups.
        """
        if not self._qbound_enabled():
            return

        qbound_group_index = None

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            if group.get("name") == "qbound":
                qbound_group_index = group_index
                break

        if qbound_group_index is None:
            raise RuntimeError(
                "Q-Bound is enabled but optimizer group 'qbound' "
                "does not exist."
            )

        if hasattr(self.lr_scheduler, "base_lrs"):
            base_lrs = list(
                self.lr_scheduler.base_lrs
            )

            if len(base_lrs) != len(
                self.optimizer.param_groups
            ):
                raise RuntimeError(
                    "Scheduler/optimizer parameter-group mismatch: "
                    f"scheduler={len(base_lrs)}, "
                    f"optimizer={len(self.optimizer.param_groups)}."
                )

            expected_lr = float(
                self.cfgs["optim"].get(
                    "qbound_lr",
                    self.cfgs["optim"].get(
                        "gpeb_lr",
                    ),
                )
            )
            scheduler_qbound_lr = float(
                base_lrs[qbound_group_index]
            )

            if not math.isclose(
                scheduler_qbound_lr,
                expected_lr,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "Scheduler registered an unexpected Q-Bound "
                    f"base LR: expected {expected_lr}, "
                    f"got {scheduler_qbound_lr}."
                )

            print(
                "[Q-Bound Scheduler Check] "
                f"base_lr={scheduler_qbound_lr}, "
                f"group_index={qbound_group_index}\n"
            )
        else:
            print(
                "[Q-Bound Scheduler Check] Scheduler has no "
                "'base_lrs' attribute. Parameter groups were still "
                "created before scheduler construction.\n"
            )

    def train(self):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            score, result = self._train_epoch(epoch)

            if score is None:
                self._save_checkpoint(epoch)
                continue

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(score)
            self._record_best(log, result)

            # print logged informations to the screen
            # for key, value in log.items():
            #     print('\t{:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                        self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count >= self.early_stop:
                    print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                        self.early_stop))
                    break

            if epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
        self._print_best()
        self._print_best_to_file()

    def _print_best_to_file(self):
      crt_time = time.asctime(
          time.localtime(time.time())
      )

      self.best_recorder["val"]["time"] = crt_time
      self.best_recorder["test"]["time"] = crt_time

      seed = self.cfgs["misc"]["seed"]

      self.best_recorder["val"]["seed"] = seed
      self.best_recorder["test"]["seed"] = seed

      self.best_recorder["val"]["best_model_from"] = "val"
      self.best_recorder["test"]["best_model_from"] = "val"

      record_dir = self.cfgs["stat"]["record_dir"]
      dataset_name = self.cfgs["dataset"]["name"]

      if not os.path.exists(record_dir):
          os.makedirs(
              record_dir,
              exist_ok=True,
          )

      record_path = os.path.join(
          record_dir,
          f"{dataset_name}.csv",
      )

      new_records = pd.DataFrame([
          self.best_recorder["val"],
          self.best_recorder["test"],
      ])

      if os.path.exists(record_path):
          old_records = pd.read_csv(record_path)

          record_table = pd.concat(
              [
                  old_records,
                  new_records,
              ],
              ignore_index=True,
          )
      else:
          record_table = new_records

      record_table.to_csv(
          record_path,
          index=False,
      )

      print(
          f"Best results saved to: {record_path}"
      )

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'monitor_best': self.mnt_best
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("*************** Saving current best: model_best.pth ... ***************")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))
        map_location = next(
            self.model.parameters()
        ).device
        checkpoint = torch.load(
            resume_path,
            map_location=map_location,
        )
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        resume_strict = self.cfgs["stat"].get(
            "resume_strict",
            True,
        )

        incompatible_keys = self.model.load_state_dict(
            checkpoint["state_dict"],
            strict=resume_strict,
        )

        if not resume_strict:
            print(
                "Missing model keys:",
                incompatible_keys.missing_keys,
            )
            print(
                "Unexpected model keys:",
                incompatible_keys.unexpected_keys,
            )

        try:
            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )
        except Exception as error:
            print(
                "Cannot load optimizer state from checkpoint:",
                error,
            )
        if "lr_scheduler" in checkpoint:
          try:
              self.lr_scheduler.load_state_dict(
                  checkpoint["lr_scheduler"]
              )
          except Exception as error:
              print(
                  "Cannot load scheduler state:",
                  error,
              )
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log, result):
        """
        Record only the checkpoint selected by validation performance.

        The old code independently tracked the best test epoch, which leaks
        test-set information into model selection. The `test` record below
        is now the test result paired with the validation-best checkpoint.
        """
        improved_val = (
            (
                self.mnt_mode == "min"
                and log[self.mnt_metric]
                <= self.best_recorder["val"][self.mnt_metric]
            )
            or (
                self.mnt_mode == "max"
                and log[self.mnt_metric]
                >= self.best_recorder["val"][self.mnt_metric]
            )
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
        # print('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        print('Best results (w.r.t {}) in validation set:'.format(self.cfgs["stat"]["monitor"]["metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print(
            'Test results paired with the validation-best checkpoint '
            '(w.r.t {}):'.format(
                self.cfgs["stat"]["monitor"]["metric"]
            )
        )
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))