import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .dataset import VideoQADataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler


def _pad_candidate_tokens(batch, key):
    """
    Pad a multiple-choice token tensor stored as [N, L] per sample
    to the maximum L in the current batch.

    This helper is used for both:
        - answer:      CRA's existing [Question SEP Answer] candidates
        - answer_only: V2.1's answer-only candidates
    """
    if key not in batch[0]:
        return

    value = batch[0][key]
    if not torch.is_tensor(value) or value.ndim != 2:
        return

    max_len = max(x[key].size(1) for x in batch)

    for i in range(len(batch)):
        cur = batch[i][key]
        if cur.size(1) < max_len:
            batch[i][key] = torch.cat(
                [
                    cur,
                    torch.zeros(
                        (
                            cur.size(0),
                            max_len - cur.size(1),
                        ),
                        dtype=cur.dtype,
                    ),
                ],
                dim=1,
            )


def videoqa_collate_fn(batch):
    """
    Tensorize a VideoQA batch.

    V2.1 change:
        In addition to the original `answer` tensor, pad `answer_only`
        when it is present. No existing CRA field is modified.
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))

    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(
                        qmax_len - len(batch[i]["question"]),
                        dtype=torch.long,
                    ),
                ],
                0,
            )

    # Original CRA candidate view: [Question SEP Answer].
    if "answer" in batch[0] and not isinstance(batch[0]["answer"], int):
        _pad_candidate_tokens(batch, "answer")

    # V2.1 auxiliary candidate view: Answer only.
    _pad_candidate_tokens(batch, "answer_only")

    return default_collate(batch)


def build_dataloaders(cfgs, a2id, tokenizer, ddp=False):
    train_dataset = VideoQADataset(cfgs, "train", tokenizer, a2id)
    val_dataset = VideoQADataset(cfgs, "val", tokenizer, a2id)
    test_dataset = (
        VideoQADataset(cfgs, "test", tokenizer, a2id)
        if cfgs["dataset"]["name"] != "star"
        else None
    )

    if ddp:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
        test_sampler = DistributedSampler(test_dataset)

        # NOTE: Kept identical to the user's current runtime code.
        # If DDP is enabled later, sampler/batch_sampler wiring should be
        # reviewed separately rather than changing it as part of V2.1.
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False if ddp else True,
        drop_last=True,
        collate_fn=videoqa_collate_fn,
        batch_sampler=train_sampler,
        pin_memory=True if ddp else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfgs["dataset"]["batch_size"],
        num_workers=cfgs["dataset"]["num_thread_reader"],
        shuffle=False,
        collate_fn=videoqa_collate_fn,
        batch_sampler=val_sampler,
        pin_memory=True if ddp else False,
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=cfgs["dataset"]["batch_size"],
            num_workers=cfgs["dataset"]["num_thread_reader"],
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
            batch_sampler=test_sampler,
            pin_memory=True if ddp else False,
        )
        if cfgs["dataset"]["name"] != "STAR"
        else None
    )

    return train_loader, val_loader, test_loader
