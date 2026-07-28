# Base Pipeline
import os
import time
import torch
import json
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
        self.optimizer = build_optimizer(cfgs, self.model)
        # Verify that the new GPEB branch is included.
        self._check_gpeb_optimizer()
        # lr_scheduler
        self.lr_scheduler = build_lr_scheduler(cfgs, self.optimizer, len(self.train_dataloader))

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

    def _check_gpeb_optimizer(self):
        """
        Verify that all trainable GPEB parameters are registered
        in the optimizer.
        """
        if not self.cfgs["model"].get("gpeb_enabled", False):
            return

        optimizer_param_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }

        gpeb_parameters = []
        missing_parameters = []

        for name, parameter in self.model.named_parameters():
            if "gpeb" not in name:
                continue

            if not parameter.requires_grad:
                continue

            gpeb_parameters.append(
                (name, parameter.numel())
            )

            if id(parameter) not in optimizer_param_ids:
                missing_parameters.append(name)

        total_gpeb_params = sum(
            numel for _, numel in gpeb_parameters
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

        for group_index, group in enumerate(
            self.optimizer.param_groups
        ):
            print(
                f"  Group {group_index}: "
                f"lr={group.get('lr')}, "
                f"num_tensors={len(group['params'])}"
            )

        if missing_parameters:
            raise RuntimeError(
                "The following GPEB parameters are missing "
                "from the optimizer:\n"
                + "\n".join(missing_parameters)
            )

        print(
            "All GPEB parameters are registered "
            "in the optimizer.\n"
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
        self.model.load_state_dict(checkpoint['state_dict']) # , strict=False)
        try: self.optimizer.load_state_dict(checkpoint['optimizer'])
        except: print("Can not load the optimizer from checkpoint")
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
