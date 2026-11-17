import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


def compute_ce_loss(pred, target, mask=None):
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    loss = criterion(pred, target)
    return loss


def _masked_align_loss(grounding_feature, qsn, negative_mask):
    """Keep CRA's 32 negative draws with replacement, only from allowed rows."""
    if grounding_feature.ndim != 2 or qsn.shape != grounding_feature.shape:
        raise ValueError("Masked CRA alignment expects two matching [B,D] tensors.")
    batch_size = grounding_feature.size(0)
    if (not torch.is_tensor(negative_mask) or negative_mask.dtype != torch.bool
            or negative_mask.shape != (batch_size, batch_size)):
        raise ValueError("align_negative_mask must be boolean [B,B].")
    device = grounding_feature.device
    allowed = negative_mask.to(device) & ~torch.eye(batch_size, device=device, dtype=torch.bool)
    eligible = allowed.any(dim=1)
    if not eligible.any():
        return (grounding_feature.sum() + qsn.sum()) * 0.0

    indices = torch.multinomial(allowed[eligible].float(), 32, replacement=True)
    # FP16/BF16 normalization and dot products use FP32 in this opt-in path.
    dtype = torch.promote_types(grounding_feature.dtype, qsn.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    visual = F.normalize(grounding_feature.to(dtype), dim=-1)
    query = F.normalize(qsn[eligible].to(dtype), dim=-1)
    positive = (query * visual[eligible]).sum(dim=-1, keepdim=True)
    negative = (query[:, None, :] * visual[indices]).sum(dim=-1)
    logits = torch.cat((positive, negative), dim=1) / 0.07
    labels = torch.zeros(query.size(0), device=device, dtype=torch.long)
    return F.cross_entropy(logits, labels)


def align_loss(grounding_feature, qsn, neg_sample=None, *, align_negative_mask=None):
    if align_negative_mask is not None:
        if neg_sample is not None:
            raise ValueError("Provide neg_sample or align_negative_mask, not both.")
        return _masked_align_loss(grounding_feature, qsn, align_negative_mask)
        
    def _info_nce_loss(x, x_pos, x_neg, temperature=0.07):
        """
        Compute the InfoNCE loss given query x, positive sample x_pos, and negative samples x_neg.
        
        Args:
        - x (torch.Tensor): Query tensor of shape [bs, n]
        - x_pos (torch.Tensor): Positive sample tensor of shape [bs, n]
        - x_neg (torch.Tensor): Negative samples tensor of shape [bs, k, n]
        - temperature (float): Temperature scaling factor
        
        Returns:
        - loss (torch.Tensor): InfoNCE loss
        """
        # Normalize the input vectors
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_neg = F.normalize(x_neg, dim=-1)
        
        # Calculate positive similarities
        pos_sim = torch.sum(x * x_pos, dim=-1, keepdim=True) / temperature
        
        # Calculate negative similarities
        neg_sim = torch.bmm(x.unsqueeze(1), x_neg.transpose(1, 2)).squeeze(1) / temperature
        
        # Concatenate positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=-1)
        
        # Create labels for the positive samples
        labels = torch.zeros(x.size(0), dtype=torch.long).to(x.device)
        
        # Compute the cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss
        
    def _sample_negatives(x_pos, k):
        """
        Sample k negative examples for each positive example from x_pos.
        
        Args:
        - x_pos (torch.Tensor): Positive sample tensor of shape [bs, n]
        - k (int): Number of negative samples to draw for each positive example
        
        Returns:
        - x_neg (torch.Tensor): Negative samples tensor of shape [bs, k, n]
        """
        bs, n = x_pos.size()
        x_neg = torch.zeros(bs, k, n).to(x_pos.device)
        for i in range(bs):
            indices = list(range(bs))
            indices.remove(i)
            neg_indices = torch.tensor(indices).to(x_pos.device)
            sampled_indices = neg_indices[torch.randint(0, len(neg_indices), (k,))]
            x_neg[i] = x_pos[sampled_indices]
        return x_neg
    
    if neg_sample is None:
        x_neg = _sample_negatives(grounding_feature, 32)
    else:
        x_neg = neg_sample
    info_nce_loss = _info_nce_loss(qsn, grounding_feature, x_neg)

    return info_nce_loss



LOSS_FNS = {"ce": compute_ce_loss, "align": align_loss}

class BuildLossFunc():
    def __init__(self, cfgs):
        self.loss_fns = cfgs["optim"]["loss"]
        self.grouped_alignment = (
            cfgs.get("dataset", {}).get("m1_grouped_batch", False)
            or cfgs.get("model", {}).get("qbound", {}).get("version") == "3.8-m1"
        )

    def __call__(self, pred, target, *, align_negative_mask=None):
        loss = {}
        total_loss = 0
        for loss_fn in self.loss_fns.keys():
            if loss_fn == "ce":
                # vq loss
                if "vq_pred" in pred.keys() and "vq" in self.loss_fns["ce"].keys():
                    _loss = LOSS_FNS[loss_fn](pred["vq_pred"], target["vq_target"])
                    total_loss = total_loss + _loss * self.loss_fns[loss_fn]["vq"]
                    loss[f"vq_loss"] = _loss
                # vqa loss
                _loss = LOSS_FNS[loss_fn](pred["vqa_pred"], target["vqa_target"])
                total_loss = total_loss + _loss * self.loss_fns[loss_fn]["vqa"]
                loss[f"vqa_loss"] = _loss
            elif loss_fn == "align":
                # global
                if "g" in self.loss_fns[loss_fn].keys():
                    weight = self.loss_fns[loss_fn]["g"]
                    if self.grouped_alignment and float(weight) == 0.0:
                        # A disabled loss shouldn't sample unmasked negatives.
                        loss["ag_loss"] = pred["vqa_align_pred"].sum() * 0.0
                        continue
                    if self.grouped_alignment and align_negative_mask is None:
                        raise ValueError("Grouped CRA alignment requires align_negative_mask.")
                    align_kwargs = ({"align_negative_mask": align_negative_mask}
                                    if align_negative_mask is not None else {})
                    if "vq_align_pred" in pred.keys():
                        base = LOSS_FNS[loss_fn](pred["vq_align_pred"], target["vq_align_gt"], **align_kwargs)
                    else:
                        # Preserve the historical CE contribution when the
                        # standard CE block precedes align and VQ is disabled.
                        # If it hasn't run, there is no prior term to reuse.
                        base = loss.get("vqa_loss", 0)
                    _loss = base + LOSS_FNS[loss_fn](
                        pred["vqa_align_pred"], target["vqa_align_gt"], **align_kwargs
                    )
                    total_loss = total_loss + _loss * weight
                    loss[f"ag_loss"] = _loss

        loss["total_loss"] = total_loss
        return loss
