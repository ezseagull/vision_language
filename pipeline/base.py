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
        os.environ['CUDA_VISIBLE_DEVICES'] = cfgs["misc"]["cuda"]
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
        # Ver0.5 GPEB parameters to their own parameter group.
        self.optimizer = build_optimizer(cfgs, self.model)
        self._configure_gpeb_optimizer_groups()
        self._check_gpeb_optimizer()

        # Build the scheduler only after the parameter groups are final.
        self.lr_scheduler = build_lr_scheduler(
            cfgs,
            self.optimizer,
            len(self.train_dataloader),
        )
        self._check_gpeb_scheduler()

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

        if cfgs["stat"]["resume"] != None:
            self._resume_checkpoint(cfgs["stat"]["resume"])

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best},
                              'test': {self.mnt_metric_test: self.mnt_best}}

        # monitor
        self.monitor = Monitor(cfgs)
        self.monitor.log_info(json.dumps(cfgs,sort_keys=True,indent=4), end='\n')

    @staticmethod
    def _is_gpeb_parameter_name(name):
        """
        Return True for all Ver0.5 GPEB-related parameters.

        This matches:
            decoder.gpeb.*
            decoder.gpeb_gate.*
            decoder.gpeb_*_norm.*
            decoder.gpeb_qa_proj.*

        It also works when DataParallel prefixes names with "module.".
        """
        return "gpeb" in name.lower()

    def _configure_gpeb_optimizer_groups(self):
        """
        Move trainable GPEB parameters to a dedicated optimizer group.

        Why this is done after build_optimizer:
            - The repository may already configure Adam/AdamW, amsgrad,
              weight decay, or other optimizer-specific options.
            - Reusing build_optimizer preserves those settings.
            - The new group only overrides the learning rate (and,
              optionally, weight decay) for GPEB.

        Expected config:
            optim:
              lr: 1.e-5
              gpeb_lr: 1.e-4
              # Optional:
              # gpeb_weight_decay: 0

        Notes:
            - This method must run before build_lr_scheduler.
            - It is safe here because no optimizer step has occurred yet.
        """
        model_cfg = self.cfgs.get("model", {})
        optim_cfg = self.cfgs.get("optim", {})

        if not model_cfg.get("gpeb_enabled", False):
            return

        if "gpeb_lr" not in optim_cfg:
            print(
                "[GPEB Optimizer] optim.gpeb_lr is not set; "
                "GPEB will use the repository's default optimizer LR."
            )
            return

        gpeb_lr = float(optim_cfg["gpeb_lr"])
        if gpeb_lr <= 0:
            raise ValueError(
                "optim.gpeb_lr must be positive, "
                f"got {gpeb_lr}."
            )

        named_gpeb_parameters = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if (
                parameter.requires_grad
                and self._is_gpeb_parameter_name(name)
            )
        ]

        if not named_gpeb_parameters:
            raise RuntimeError(
                "model.gpeb_enabled=True, but no trainable parameter "
                "whose name contains 'gpeb' was found. Check whether "
                "the Ver0.5 refine decoder is actually instantiated."
            )

        gpeb_param_ids = {
            id(parameter)
            for _, parameter in named_gpeb_parameters
        }

        # Remove GPEB tensors from every group created by build_optimizer.
        # They will be re-added exactly once in the dedicated group.
        retained_groups = []
        extracted_gpeb_parameters = []
        extracted_ids = set()

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            retained_parameters = []

            for parameter in group["params"]:
                parameter_id = id(parameter)

                if parameter_id in gpeb_param_ids:
                    if parameter_id not in extracted_ids:
                        extracted_gpeb_parameters.append(parameter)
                        extracted_ids.add(parameter_id)
                else:
                    retained_parameters.append(parameter)

            if retained_parameters:
                group["params"] = retained_parameters
                group.setdefault(
                    "name",
                    f"base_{group_index}",
                )
                retained_groups.append(group)

        self.optimizer.param_groups = retained_groups

        missing_ids = gpeb_param_ids - extracted_ids
        if missing_ids:
            missing_names = [
                name
                for name, parameter in named_gpeb_parameters
                if id(parameter) in missing_ids
            ]
            raise RuntimeError(
                "Some trainable GPEB parameters were not present in "
                "the optimizer returned by build_optimizer:\n"
                + "\n".join(missing_names)
            )

        gpeb_group = {
            "params": extracted_gpeb_parameters,
            "lr": gpeb_lr,
            "name": "gpeb",
        }

        if "gpeb_weight_decay" in optim_cfg:
            gpeb_group["weight_decay"] = float(
                optim_cfg["gpeb_weight_decay"]
            )

        self.optimizer.add_param_group(gpeb_group)

        total_gpeb_params = sum(
            parameter.numel()
            for parameter in extracted_gpeb_parameters
        )

        print("\n[GPEB Optimizer Group Created]")
        print(f"  lr: {gpeb_lr}")
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
            f"  trainable parameters: "
            f"{total_gpeb_params:,}"
        )
        print(
            f"  tensors: "
            f"{len(extracted_gpeb_parameters)}\n"
        )

    def _check_gpeb_optimizer(self):
        """
        Verify that all trainable GPEB parameters are registered exactly
        once and report the actual optimizer learning rates.
        """
        if not self.cfgs["model"].get("gpeb_enabled", False):
            return

        optimizer_parameter_counts = {}

        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                parameter_id = id(parameter)
                optimizer_parameter_counts[parameter_id] = (
                    optimizer_parameter_counts.get(
                        parameter_id,
                        0,
                    )
                    + 1
                )

        gpeb_parameters = []
        missing_parameters = []
        duplicated_parameters = []

        for name, parameter in self.model.named_parameters():
            if not self._is_gpeb_parameter_name(name):
                continue

            if not parameter.requires_grad:
                continue

            gpeb_parameters.append(
                (name, parameter.numel())
            )

            registration_count = optimizer_parameter_counts.get(
                id(parameter),
                0,
            )

            if registration_count == 0:
                missing_parameters.append(name)
            elif registration_count > 1:
                duplicated_parameters.append(name)

        total_gpeb_params = sum(
            numel
            for _, numel in gpeb_parameters
        )

        print("\n[GPEB Optimizer Check]")
        print(
            f"Trainable GPEB parameters: "
            f"{total_gpeb_params:,}"
        )
        print(
            "GPEB tensors registered:",
            len(gpeb_parameters),
        )
        print(
            "Optimizer parameter groups:",
            len(self.optimizer.param_groups),
        )

        gpeb_group_indices = []

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            group_parameter_ids = {
                id(parameter)
                for parameter in group["params"]
            }

            num_gpeb_tensors = sum(
                id(parameter) in group_parameter_ids
                for _, parameter in self.model.named_parameters()
                if (
                    parameter.requires_grad
                    and self._is_gpeb_parameter_name(_)
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
                f"gpeb_tensors={num_gpeb_tensors}"
            )

            if num_gpeb_tensors > 0:
                gpeb_group_indices.append(group_index)

        if missing_parameters:
            raise RuntimeError(
                "The following GPEB parameters are missing "
                "from the optimizer:\n"
                + "\n".join(missing_parameters)
            )

        if duplicated_parameters:
            raise RuntimeError(
                "The following GPEB parameters occur more than once "
                "in the optimizer:\n"
                + "\n".join(duplicated_parameters)
            )

        if len(gpeb_group_indices) != 1:
            raise RuntimeError(
                "Expected all GPEB tensors to be in exactly one "
                "optimizer group, but found GPEB tensors in groups: "
                f"{gpeb_group_indices}."
            )

        actual_gpeb_lr = self.optimizer.param_groups[
            gpeb_group_indices[0]
        ]["lr"]

        expected_gpeb_lr = float(
            self.cfgs["optim"].get(
                "gpeb_lr",
                actual_gpeb_lr,
            )
        )

        if not math.isclose(
            float(actual_gpeb_lr),
            expected_gpeb_lr,
            rel_tol=1e-9,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                "GPEB optimizer LR mismatch: "
                f"expected {expected_gpeb_lr}, "
                f"got {actual_gpeb_lr}."
            )

        print(
            "All GPEB parameters are registered exactly once."
        )
        print(
            f"Verified GPEB learning rate: "
            f"{actual_gpeb_lr}\n"
        )

    def _check_gpeb_scheduler(self):
        """
        Check that the scheduler was built after the GPEB optimizer
        group and can see every optimizer parameter group.
        """
        if not self.cfgs["model"].get("gpeb_enabled", False):
            return

        gpeb_group_index = None

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            if group.get("name") == "gpeb":
                gpeb_group_index = group_index
                break

        if gpeb_group_index is None:
            # This can occur only when no separate gpeb_lr was set.
            return

        if hasattr(self.lr_scheduler, "base_lrs"):
            base_lrs = list(self.lr_scheduler.base_lrs)

            if len(base_lrs) != len(
                self.optimizer.param_groups
            ):
                raise RuntimeError(
                    "The LR scheduler does not contain the same "
                    "number of parameter groups as the optimizer: "
                    f"scheduler={len(base_lrs)}, "
                    f"optimizer={len(self.optimizer.param_groups)}."
                )

            expected_lr = float(
                self.cfgs["optim"]["gpeb_lr"]
            )
            scheduler_gpeb_lr = float(
                base_lrs[gpeb_group_index]
            )

            if not math.isclose(
                scheduler_gpeb_lr,
                expected_lr,
                rel_tol=1e-9,
                abs_tol=0.0,
            ):
                raise RuntimeError(
                    "The scheduler registered an unexpected GPEB "
                    f"base LR: expected {expected_lr}, "
                    f"got {scheduler_gpeb_lr}."
                )

            print(
                "[GPEB Scheduler Check] "
                f"base_lr={scheduler_gpeb_lr}, "
                f"group_index={gpeb_group_index}\n"
            )
        else:
            print(
                "[GPEB Scheduler Check] Scheduler has no "
                "'base_lrs' attribute; optimizer groups were created "
                "before scheduler construction, but verify the LR "
                "values during the first training iterations.\n"
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

                if not_improved_count > self.early_stop:
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
      self.best_recorder["test"]["best_model_from"] = "test"

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
        checkpoint = torch.load(resume_path)
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
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)
            # save the result from the best model
            filename = os.path.join(self.monitor.record_dir, 'result.pth')
            torch.save(result, filename)


        improved_test = (self.mnt_mode == 'min' and log[self.mnt_metric_test] <= self.best_recorder['test'][
            self.mnt_metric_test]) or \
                        (self.mnt_mode == 'max' and log[self.mnt_metric_test] >= self.best_recorder['test'][
                            self.mnt_metric_test])
        if improved_test:
            self.best_recorder['test'].update(log)

    def _print_best(self):
        # print('Best results (w.r.t {}) in validation set:'.format(self.args["monitor_metric"]))
        print('Best results (w.r.t {}) in validation set:'.format(self.cfgs["stat"]["monitor"]["metric"]))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

        print('Best results (w.r.t {}) in test set:'.format(self.cfgs["stat"]["monitor"]["metric"]))
        for key, value in self.best_recorder['test'].items():
            print('\t{:15s}: {}'.format(str(key), value))
