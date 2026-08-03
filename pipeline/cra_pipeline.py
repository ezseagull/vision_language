import collections
import os
import numpy as np
from tabulate import tabulate
import torch
from torch import nn
import torch.nn.functional as F
from .base import BasePipeline
from utils.misc.misc import get_mask, load_file
import matplotlib.pyplot as plt
from random import shuffle
from scipy.special import softmax as np_softmax
import time
from utils.misc.misc import get_remain_time
import wandb

class CRAPipeline(BasePipeline):
    def __init__(self, cfgs) -> None:
        super().__init__(cfgs)

    def _get_gpeb_aux(self, gqa_output):
        """Safely extract GPEB auxiliary outputs from CRA VQA output."""
        if not self.cfgs["model"].get("gpeb_enabled", False):
            return None

        if len(gqa_output) < 3:
            return None

        model_outputs = gqa_output[2]
        time_param = model_outputs.get("time_param")
        if time_param is None:
            return None

        return time_param.get("gpeb_aux")

    def _compute_gpeb_loss(self, gqa_output):
        """
        Compute midpoint and endpoint reconstruction losses
        for Grounded Predictive Event Bridge.

        Args:
            gqa_output:
                Output returned by CRA in mode="VQA".

        Returns:
            bridge_loss:
                Scalar tensor, or None if GPEB is disabled.
            loss_logs:
                Dictionary of detached component losses.
        """
        gpeb_aux = self._get_gpeb_aux(gqa_output)
        if gpeb_aux is None:
            return None, {}

        required_keys = [
            "predicted_middles",
            "target_middles",
            "predicted_ends",
            "target_ends",
            "event_valid_mask",
        ]

        missing_keys = [
            key
            for key in required_keys
            if key not in gpeb_aux
        ]

        if missing_keys:
            raise KeyError(
                "Missing GPEB outputs: "
                + ", ".join(missing_keys)
            )

        event_mask = gpeb_aux[
            "event_valid_mask"
        ].bool()

        # Defensive guard, although with 32 frames
        # and 4 bins this should normally be True.
        if not event_mask.any():
            zero_loss = gpeb_aux[
                "predicted_middles"
            ].sum() * 0.0

            return zero_loss, {
                "gpeb_middle_loss": 0.0,
                "gpeb_end_loss": 0.0,
            }

        predicted_middles = gpeb_aux[
            "predicted_middles"
        ][event_mask]

        target_middles = gpeb_aux[
            "target_middles"
        ][event_mask].detach()

        predicted_ends = gpeb_aux[
            "predicted_ends"
        ][event_mask]

        target_ends = gpeb_aux[
            "target_ends"
        ][event_mask].detach()

        middle_loss = F.smooth_l1_loss(
            predicted_middles,
            target_middles,
        )

        end_loss = F.smooth_l1_loss(
            predicted_ends,
            target_ends,
        )

        # Average instead of sum to keep the scale stable.
        bridge_loss = 0.5 * (
            middle_loss + end_loss
        )

        loss_logs = {
            "gpeb_middle_loss": (
                middle_loss.detach().item()
            ),
            "gpeb_end_loss": (
                end_loss.detach().item()
            ),
        }

        return bridge_loss, loss_logs

    def _compute_gpeb_aux_qa_loss(
        self,
        gqa_output,
        answer_proj,
        answer_id,
        batch_size,
    ):
        """
        Score each answer candidate using only the GPEB representation.

        The existing answer embeddings are reused so this branch introduces
        no new classifier parameters. By default they are detached, ensuring
        that this auxiliary objective primarily trains GPEB rather than
        duplicating gradients into the language/answer encoder.
        """
        gpeb_aux = self._get_gpeb_aux(gqa_output)
        if gpeb_aux is None:
            return None, None, None

        bridge_repr = gpeb_aux.get("bridge_repr")
        if bridge_repr is None:
            raise KeyError(
                "GPEB Ver0.5 requires gpeb_aux['bridge_repr']."
            )

        if bridge_repr.ndim != 2:
            raise ValueError(
                "bridge_repr must be [B, D], got "
                f"{tuple(bridge_repr.shape)}."
            )

        detach_answer = self.cfgs["model"].get(
            "gpeb_aux_detach_answer",
            True,
        )
        aux_answer_proj = (
            answer_proj.detach()
            if detach_answer
            else answer_proj
        )

        if aux_answer_proj.ndim != 3:
            raise ValueError(
                "answer_proj must be [B, C, D], got "
                f"{tuple(aux_answer_proj.shape)}."
            )

        if aux_answer_proj.size(0) != bridge_repr.size(0):
            raise ValueError(
                "answer_proj and bridge_repr batch sizes differ: "
                f"{aux_answer_proj.size(0)} vs {bridge_repr.size(0)}."
            )

        if aux_answer_proj.size(-1) != bridge_repr.size(-1):
            raise ValueError(
                "answer_proj and bridge_repr feature dimensions differ: "
                f"{aux_answer_proj.size(-1)} vs {bridge_repr.size(-1)}."
            )

        aux_logits = torch.bmm(
            aux_answer_proj,
            bridge_repr.unsqueeze(-1),
        ).squeeze(-1)

        prop_num = self.cfgs["model"].get("prop_num", 1)
        aux_logits = aux_logits.view(
            batch_size,
            prop_num,
            -1,
        ).mean(dim=1)

        answer_id_cuda = answer_id.to(
            device=aux_logits.device,
            dtype=torch.long,
        ).reshape(-1)
        if answer_id_cuda.numel() != aux_logits.size(0):
            raise ValueError(
                "answer_id size does not match GPEB auxiliary logits."
            )
        aux_loss = F.cross_entropy(
            aux_logits,
            answer_id_cuda,
        )
        aux_predicted = aux_logits.argmax(dim=1)
        aux_accuracy = (
            aux_predicted == answer_id_cuda
        ).float().mean()

        return aux_loss, aux_logits, aux_accuracy

    @staticmethod
    def _masked_scalar_mean(values, mask=None):
        if values is None:
            return None
        if not torch.is_tensor(values):
            return float(values)
        if mask is not None and values.ndim >= 2:
            values = values[mask]
        if values.numel() == 0:
            return 0.0
        return values.detach().float().mean().item()

    def _collect_gpeb_stats(self, gqa_output, loss_logs=None):
        """Collect detached diagnostics for terminal and W&B logging."""
        stats = {}
        if loss_logs:
            stats.update(loss_logs)

        gpeb_aux = self._get_gpeb_aux(gqa_output)
        if gpeb_aux is None:
            return stats

        direct_scalar_keys = [
            "gate_mean",
            "gate_std",
            "gate_min",
            "gate_max",
            "residual_ratio_mean",
            "residual_ratio_unclipped_mean",
            "residual_clip_scale_mean",
            "residual_max_ratio",
            "branch_cosine_mean",
            "grounding_scale_used",
        ]
        for key in direct_scalar_keys:
            value = gpeb_aux.get(key)
            scalar = self._masked_scalar_mean(value)
            if scalar is not None:
                stats[f"gpeb_{key}"] = scalar

        event_mask = gpeb_aux.get("event_valid_mask")
        if event_mask is not None:
            event_mask = event_mask.bool()

        event_keys = {
            "event_prediction_error": "gpeb_prediction_error",
            "event_weight_entropy": "gpeb_event_entropy",
            "event_effective_count": "gpeb_effective_event_count",
        }
        for source_key, log_key in event_keys.items():
            value = gpeb_aux.get(source_key)
            scalar = self._masked_scalar_mean(
                value,
                event_mask if source_key == "event_prediction_error" else None,
            )
            if scalar is not None:
                stats[log_key] = scalar

        fusion_delta = gpeb_aux.get("fusion_delta")
        if torch.is_tensor(fusion_delta):
            stats["gpeb_fusion_delta_norm"] = (
                fusion_delta.detach().float().norm(dim=-1).mean().item()
            )

        return stats

    def _gpeb_monitor_keys(self):
        return [
            "gpeb_aux_acc",
            "gpeb_middle_loss",
            "gpeb_end_loss",
            "gpeb_gate_mean",
            "gpeb_gate_std",
            "gpeb_gate_min",
            "gpeb_gate_max",
            "gpeb_residual_ratio_mean",
            "gpeb_residual_ratio_unclipped_mean",
            "gpeb_residual_clip_scale_mean",
            "gpeb_residual_max_ratio",
            "gpeb_branch_cosine_mean",
            "gpeb_grounding_scale_used",
            "gpeb_prediction_error",
            "gpeb_event_entropy",
            "gpeb_effective_event_count",
            "gpeb_fusion_delta_norm",
        ]
    
    def get_loss(self, batch):
        answer_id, answer, video_frames, question = (
            batch["answer_id"],
            batch["answer"],
            batch["video_frames"].cuda(),
            batch["question"].cuda(),
        )

        qsns_id, qsns_token_ids, qsns_seq_len = (
            batch["qsns_id"],
            batch["qsns_token_ids"],
            batch["qsns_seq_len"],
        )

        video_len = batch["video_len"]
        video_mask = (
            get_mask(
                video_len,
                self.cfgs["dataset"]["max_feats"],
            ).cuda()
            if self.cfgs["dataset"]["max_feats"] > 0
            else None
        )

        batch_size = answer_id.size(0)
        answer_id_cuda = answer_id.cuda()
        pred = {}
        target = {}

        # Find video moments relevant to the question.
        if self.cfgs["model"]["vg_loss"]:
            vq_output = self.model(
                video_frames,
                video_mask,
                qsns_token_ids,
                qsns_id,
                mode="VQ",
            )
            vq_video_proj = vq_output[0].unsqueeze(-1)
            vq_predicts = torch.bmm(
                vq_output[1],
                vq_video_proj,
            ).squeeze(-1)
            vq_predicts = vq_predicts.view(
                batch_size,
                self.cfgs["model"]["prop_num"],
                -1,
            ).mean(dim=1)

            pred["vq_pred"] = vq_predicts
            target["vq_target"] = qsns_id.cuda()
            pred["vq_align_pred"] = (
                vq_output[2]["time_param"]["grounding_feature"]
            )
            target["vq_align_gt"] = vq_output[2]["align_gt"]

        # Refine moments according to candidate answers.
        gqa_output = self.model(
            video_frames,
            video_mask,
            answer,
            answer_id,
            mode="VQA",
        )
        main_video_proj = gqa_output[0].unsqueeze(-1)
        answer_proj = gqa_output[1]
        vqa_predicts = torch.bmm(
            answer_proj,
            main_video_proj,
        ).squeeze(-1)
        predicts = vqa_predicts.view(
            batch_size,
            self.cfgs["model"]["prop_num"],
            -1,
        ).mean(dim=1)

        pred["vqa_pred"] = predicts
        target["vqa_target"] = answer_id_cuda
        pred["vqa_align_pred"] = (
            gqa_output[2]["time_param"]["grounding_feature"]
        )
        target["vqa_align_gt"] = gqa_output[2]["align_gt"]

        loss = self.criterion(pred, target)

        # GPEB predictive reconstruction objective.
        gpeb_bridge_loss, bridge_logs = self._compute_gpeb_loss(
            gqa_output
        )
        if gpeb_bridge_loss is not None:
            bridge_weight = self.cfgs["model"].get(
                "gpeb_bridge_loss_weight",
                0.05,
            )
            loss["gpeb_bridge_loss"] = gpeb_bridge_loss
            loss["total_loss"] = (
                loss["total_loss"]
                + bridge_weight * gpeb_bridge_loss
            )

        # GPEB-only answer objective. This explicitly forces bridge_repr
        # to contain information capable of distinguishing answer choices.
        gpeb_aux_loss, _, gpeb_aux_acc = (
            self._compute_gpeb_aux_qa_loss(
                gqa_output=gqa_output,
                answer_proj=answer_proj,
                answer_id=answer_id,
                batch_size=batch_size,
            )
        )
        if gpeb_aux_loss is not None:
            aux_weight = self.cfgs["model"].get(
                "gpeb_aux_qa_weight",
                0.30,
            )
            loss["gpeb_aux_qa_loss"] = gpeb_aux_loss
            loss["total_loss"] = (
                loss["total_loss"]
                + aux_weight * gpeb_aux_loss
            )

        predicted = predicts.argmax(dim=1).detach().cpu()
        running_acc = (
            predicted == answer_id.cpu()
        ).float().mean().item()

        gpeb_stats = self._collect_gpeb_stats(
            gqa_output,
            bridge_logs,
        )
        if gpeb_aux_acc is not None:
            gpeb_stats["gpeb_aux_acc"] = (
                gpeb_aux_acc.detach().item()
            )

        return loss, running_acc, gpeb_stats

    def _train_epoch(self, epoch):
        self.monitor.log_info(f"\n{'#'*20} Epoch {epoch}/{self.cfgs['optim']['epochs']} ... {'#'*20}\n")
        #########################################
        # Train Stage
        #########################################
        print("Training Stage")
        self.model.train()
        # reset monitor
        self.monitor.reset_kv(f"{self.cfgs['stat']['monitor']['metric']}")
        self.monitor.reset_kv(f"total_loss")
        for key in self.cfgs["optim"]["loss"].keys():
            if key == "ce" or key == "align":
                continue
            self.monitor.reset_kv(f"{key}_loss")

        if self.cfgs["model"].get("gpeb_enabled", False):
            self.monitor.reset_kv("gpeb_bridge_loss")
            self.monitor.reset_kv("gpeb_aux_qa_loss")
            for key in self._gpeb_monitor_keys():
                self.monitor.reset_kv(key)
        # train iter

        for idx, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            loss, monitor_metric, gpeb_stats = self.get_loss(batch)
            
            self.monitor.logkv_mean("acc", monitor_metric)
            for name, value in gpeb_stats.items():
                self.monitor.logkv_mean(name, value)
            self.optimizer.zero_grad()
            loss["total_loss"].backward()
            if self.cfgs["optim"]["clip"]:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.cfgs["optim"]["clip"])
            self.optimizer.step()
            self.lr_scheduler.step()

            progress = float(idx + 1) / len(self.train_dataloader)
            end_time = time.time()
            ct, rbt, ret = get_remain_time(start_time, end_time, idx, len(self.train_dataloader), epoch, self.cfgs["optim"]['epochs'])
            print_info = f"Progress: {progress:3.2%} (rbt-ret: {rbt}-{ret}) | Train {self.cfgs['stat']['monitor']['metric']}: {monitor_metric:3.2%}"

            for _, (loss_fns, _loss) in enumerate(loss.items()):
                running_loss = _loss.detach().cpu().numpy()
                self.monitor.logkv_mean(f"{loss_fns}", running_loss)
                print_info += f" {loss_fns}: {running_loss:.4f}"

            # print info
            print(f'\r{print_info}', end='', flush=True)
        
        print_info = [[], []]
        for _, (name, val) in enumerate(self.monitor.name2val.items()):
            print_info[0].append(f"mean {name}")
            print_info[1].append(f"{val:.4f}")

        self.monitor.log_info(f'\n{tabulate(print_info, headers="firstrow", tablefmt="grid")}', end="\n")

        #########################################
        # Eval Stage
        #########################################
        print("Eval Stage")
        # val set
        val_score, val_result = self.eval(self.val_dataloader)
        # test set
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
            test_result = val_result
        else:
            test_score, test_result = self.eval(self.test_dataloader)
        score = {}
        score.update({f"val_{k}": val_score[k] for k in val_score.keys()})
        score.update({f"test_{k}": test_score[k] for k in test_score.keys()})
        self.print_score(score)

        # === BỔ SUNG CODE BẮN LOG LÊN WANDB ===
        wandb_data = {"epoch": epoch}
        
        # 1. Gom toàn bộ chỉ số huấn luyện vào thư mục "train/..."
        for name, val in self.monitor.name2val.items():
            wandb_data[f"train/{name}"] = val

        # 2. Phân rã và gom nhóm Val / Test vào đúng thư mục "val/..." và "test/..."
        for split_name, split_dict in [("val", val_score), ("test", test_score)]:
            for k, v in split_dict.items():
                if isinstance(v, dict):
                    # Nếu là metric cụm (ví dụ: post-hoc chứa R@0.3, R@0.5...)
                    for sub_k, sub_v in v.items():
                        wandb_data[f"{split_name}/{k}_{sub_k}"] = sub_v
                else:
                    # Nếu là metric đơn (ví dụ: acc)
                    wandb_data[f"{split_name}/{k}"] = v

        wandb.log(wandb_data)
        # ======================================

        return score, test_result
    
    def print_score(self, score):
        def _print_score(_score):
            self.monitor.log_info(f"\n")
            for k in _score.keys():
                self.monitor.log_info(f"{k}  ")
            self.monitor.log_info(f"\n")
            for k in _score.keys():
                self.monitor.log_info(f"{_score[k]:.1f} \t ")

        self.monitor.log_info(f"\nVal Set\t mean acc: {score['val_acc']:.2%}")
        if 'val_pre-hoc' in score.keys():
            self.monitor.log_info("\n================ val_attn ================")
            _print_score(score['val_pre-hoc'])
        self.monitor.log_info("\n================ val_post-hoc ==================")
        _print_score(score['val_post-hoc'])
        if 'val_merge' in score.keys():
            self.monitor.log_info("\n================== val_merge ===================")
            _print_score(score['val_merge'])

        self.monitor.log_info(f"\nTest Set\t mean acc: {score['test_acc']:.2%}")
        if 'test_pre-hoc' in score.keys():
            self.monitor.log_info("\n============== test_attn ================")
            _print_score(score['test_pre-hoc'])
        self.monitor.log_info("\n================ test_post-hoc ================")
        _print_score(score['test_post-hoc'])
        if 'test_merge' in score.keys():
            self.monitor.log_info("\n================= test_merge ==================")
            _print_score(score['test_merge'])

        gpeb_keys = [
            key for key in score
            if key.startswith("val_gpeb_")
            or key.startswith("test_gpeb_")
        ]
        if gpeb_keys:
            self.monitor.log_info(
                "\n================ GPEB diagnostics ================"
            )
            for key in gpeb_keys:
                self.monitor.log_info(
                    f"{key}: {score[key]:.6f}"
                )
        print("\n")
    
    def draw_grounding(self, gt, posthoc, attn, merge, epoch, probs=None):
        img_path = os.path.join(self.save_dir, "record", "grounding")
        if os.path.exists(img_path):
            # os.remove(img_path)
            pass
        else:
            os.mkdir(img_path)

        vid_qid = list(posthoc.keys())
        # if probs is not None:
        #    probs = {vid_qid[i]: probs[i] for i in range(len(vid_qid))}

        nums = 32  # len(vid_qid)
        shuffle(vid_qid)
        for i in range(nums):
            _vid_qid = vid_qid[i]
            _vid, _qid = _vid_qid.split('_')
            duration = gt[_vid]['duration']
            gt_g = gt[_vid]["location"][_qid]
            posthoc_g = posthoc[f'{_vid}_{_qid}']
            attn_g = np.round(np.asarray(attn[f'{_vid}_{_qid}'])*duration, 1)
            merge_g = merge[f'{_vid}_{_qid}']

            # 创建一个图和轴
            fig, ax = plt.subplots()
            
            # 对于每个时间区间，使用对应的颜色绘制一个条形图
            ax.plot(attn_g, [3, 3], color="blue", linewidth=6)
            ax.plot(posthoc_g, [2, 2], color="yellow", linewidth=6)
            ax.plot(merge_g, [1, 1], color="green", linewidth=6)
            ax.plot(gt_g[0], [0, 0], color="red", linewidth=6)  # 调整 linewidth 来改变时间条的粗细
            if probs is not None:
                _probs = probs[_vid_qid]  # [length]
                _probs = np.array(_probs)*3
                ax.plot(np.linspace(0, duration, 32), _probs, color="gray", linewidth=3, label="ATT-RLSTM")

            # 优化图表布局
            # plt.xticks(rotation=45)  # 旋转时间标签，使其更易读
            plt.tight_layout()       # 自动调整子图参数，使之填充整个图像区域

            fig.savefig(os.path.join(img_path, f'{epoch}_{_vid}_{_qid}.png'), dpi=300)
            del fig, ax
        
    def eval(self, data_loader, epoch=None, save=False):
        self.model.eval()
        count = 0
        score = {"acc": .0}
        results = {}
        ground_res = {}
        ground_pre = {}
        keyframe = {}
        ori_keyframe = {}
        gpeb_aux_correct = 0
        gpeb_aux_count = 0
        gpeb_stat_sums = collections.defaultdict(float)
        gpeb_stat_count = 0

        with torch.no_grad():
            if not self.cfgs["dataset"]["mc"]:
                self.model.module._compute_answer_embedding(self.a2v)
            
            for idx, batch in enumerate(data_loader):
                answer_id, answer, video_frames, question, question_id = (
                    batch["answer_id"],
                    batch["answer"].to("cuda"),
                    batch["video_frames"].to("cuda"),
                    batch["question"].to("cuda"),
                    batch['question_id'],    
                )
                
                video_len = batch["video_len"]
                if self.cfgs["model"]["lan"] in ['DistilBERET','BERT', 'DeBERTa']:
                    pad_id = 0
                elif self.cfgs["model"]["lan"] == 'RoBERTa':
                    pad_id = 1
                
                video_mask = get_mask(video_len, video_frames.size(1)).to("cuda")
                count += answer_id.size(0)
                
                fusion_proj, answer_proj, kargs = self.model(video_frames, video_mask, answer)
                
                fusion_proj = fusion_proj.unsqueeze(2)
                predicts = torch.bmm(answer_proj, fusion_proj).squeeze(-1) # Sửa .squeeze() thành .squeeze(-1) an toàn hơn
                
                # --- SỬA LỖI DIMENSION OUT OF RANGE Ở ĐÂY ---
                if predicts.dim() == 1:
                    predicts = predicts.unsqueeze(0)
                # ---------------------------------------------

                # GPEB-only validation accuracy and diagnostics.
                if "time_param" in kargs:
                    eval_gpeb_aux = kargs["time_param"].get("gpeb_aux")
                else:
                    eval_gpeb_aux = None

                if eval_gpeb_aux is not None:
                    bridge_repr = eval_gpeb_aux.get("bridge_repr")
                    if bridge_repr is not None:
                        aux_logits = torch.bmm(
                            answer_proj,
                            bridge_repr.unsqueeze(-1),
                        ).squeeze(-1)
                        aux_logits = aux_logits.view(
                            answer_id.size(0),
                            self.cfgs["model"]["prop_num"],
                            -1,
                        ).mean(dim=1)
                        aux_predicted = aux_logits.argmax(dim=1).cpu()
                        gpeb_aux_correct += (
                            aux_predicted == answer_id.cpu()
                        ).sum().item()
                        gpeb_aux_count += answer_id.size(0)

                    # Build a CRA-like wrapper for the shared collector.
                    eval_wrapper = (
                        None,
                        None,
                        {"time_param": kargs["time_param"]},
                    )
                    batch_stats = self._collect_gpeb_stats(eval_wrapper)
                    for name, value in batch_stats.items():
                        gpeb_stat_sums[name] += (
                            float(value) * answer_id.size(0)
                        )
                    gpeb_stat_count += answer_id.size(0)
                
                predicted = torch.max(predicts, dim=1).indices.cpu()
                score["acc"] += (predicted == answer_id.cpu()).sum().item()
                
                video_attn = kargs['fatt'].squeeze().cpu().numpy()
                if video_attn.ndim == 1: video_attn = video_attn[np.newaxis, :] # Đảm bảo luôn có 2 chiều

                if 'time_param' in kargs.keys():
                    time_param = kargs['time_param']
                    start_time = time_param["start"].detach().cpu().numpy()
                    end_time = time_param["end"].detach().cpu().numpy()
                    _ground_pre = np.column_stack((start_time, end_time))
                    _keyframe = time_param["key"].detach().cpu().numpy()
                    _ori_keyframe = time_param["ori_key"].detach().cpu().numpy()
                    for bs, qid in enumerate(question_id):
                        results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer': int(answer_id.numpy()[bs])}
                        ground_res[qid] = video_attn[bs].tolist()
                        ground_pre[qid] = _ground_pre[bs].tolist()
                        keyframe[qid] = _keyframe[bs].tolist()
                        ori_keyframe[qid] = _ori_keyframe[bs].tolist()
                else:
                    for bs, qid in enumerate(question_id):
                        results[qid] = {'prediction': int(predicted.numpy()[bs]), 'answer': int(answer_id.numpy()[bs])}
                        ground_res[qid] = video_attn[bs].tolist()

                progress = float(idx + 1) / len(data_loader)
                print(f"\rProgress: {progress:.2%}", end='', flush=True)

        # --- TÍNH TOÁN METRIC SAU KHI XONG VÒNG LẶP ---
        score["acc"] /= count
        if gpeb_aux_count > 0:
            score["gpeb_aux_acc"] = (
                gpeb_aux_correct / gpeb_aux_count
            )
        if gpeb_stat_count > 0:
            for name, value_sum in gpeb_stat_sums.items():
                score[name] = value_sum / gpeb_stat_count

        gt_file = os.path.join(self.cfgs["dataset"]["csv_path"], f'gsub_{data_loader.dataset.split}.json')
        gt_ground = load_file(gt_file)
        seg_file = os.path.join(self.cfgs["dataset"]["csv_path"], f'frame2time_{data_loader.dataset.split}.json')
        segs = load_file(seg_file)
        qa_file = os.path.join(self.cfgs["dataset"]["csv_path"], f'{data_loader.dataset.split}.csv')
        qas = load_file(qa_file)
        
        post_hoc_ground = self.metric.generate_ground(ground_res, segs, qas)
        score["post-hoc"] = self.metric(post_hoc_ground, gt_ground, results, gs=False)
        result = {"ph": post_hoc_ground, "result": results, "gt_ground": gt_ground, "ph_attn": ground_res}

        if 'time_param' in kargs.keys():
            score["pre-hoc"] = self.metric(ground_pre, gt_ground, results, gs=True)
            merge_ground = self.metric.combine(ground_pre, post_hoc_ground, gt_ground)
            score["merge"] = self.metric(merge_ground, gt_ground, results, gs=False)
            result.update({"tg": ground_pre, "key": keyframe, "ori_key": ori_keyframe, "merge": merge_ground})
            
        return score, result
    
    def infer(self, epoch=None):
        count_param = sum(p.numel() for p in self.model.parameters())
        print(f"Model Params: {count_param}")
        print("\nEval Stage")
        # val set
        val_score, _ = self.eval(self.val_dataloader, epoch, True if self.cfgs["dataset"]["name"] == "star" else False)
        # test set
        if self.cfgs["dataset"]["name"] == "star":
            test_score = val_score
        else:
            test_score, _ = self.eval(self.test_dataloader, epoch, True)
        score = {}
        score.update({f"val_{k}": val_score[k] for k in val_score.keys()})
        score.update({f"test_{k}": test_score[k] for k in test_score.keys()})
        self.print_score(score)
        return score