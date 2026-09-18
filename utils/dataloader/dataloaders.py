from functools import partial

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from .dataset import VideoQADataset


def _resolve_pad_token_id(tokenizer) -> int:
    """
    Return the tokenizer-specific padding id.

    Q-Bound V2.3 must not assume PAD=0 because the language encoder can be
    changed independently (e.g. RoBERTa, BERT, DeBERTa, etc.).

    If a tokenizer family does not define a pad token, configure that once in
    build_tokenizer(); the dataloader must not silently guess a padding id.
    """
    pad_token_id = getattr(tokenizer, "pad_token_id", None)

    if pad_token_id is None:
        raise ValueError(
            "Tokenizer does not define pad_token_id. "
            "Configure a valid pad token in build_tokenizer() before "
            "building the dataloaders."
        )

    return int(pad_token_id)


def _pad_candidate_tokens(batch, key, pad_token_id: int):
    """
    Pad a multiple-choice token tensor stored as [N, L] per sample to the
    maximum sequence length L in the current batch.

    Used for:
        - answer:
            CRA's existing [Question SEP Answer] candidates.

        - answer_only:
            Q-Bound V2.3 training-only answer candidates for the detached
            functional critic.

    The padding id comes from the active tokenizer, so this helper is language
    model agnostic.
    """
    if not batch:
        return

    if key not in batch[0]:
        return

    value = batch[0][key]

    if not torch.is_tensor(value) or value.ndim != 2:
        return

    max_len = max(
        sample[key].size(1)
        for sample in batch
    )

    for sample in batch:
        cur = sample[key]

        if not torch.is_tensor(cur) or cur.ndim != 2:
            raise TypeError(
                f"Expected batch['{key}'] to be a 2-D tensor [N,L] "
                f"for every sample; got {type(cur)!r} / "
                f"shape={getattr(cur, 'shape', None)}."
            )

        pad_len = max_len - cur.size(1)

        if pad_len <= 0:
            continue

        padding = torch.full(
            (
                cur.size(0),
                pad_len,
            ),
            fill_value=pad_token_id,
            dtype=cur.dtype,
            device=cur.device,
        )

        sample[key] = torch.cat(
            [cur, padding],
            dim=1,
        )


def videoqa_collate_fn(
    batch,
    pad_token_id: int,
):
    """
    Tensorize one VideoQA batch with tokenizer-aware padding.

    V2.3 contract
    -------------
    - question is padded with tokenizer.pad_token_id;
    - answer is padded with the same id;
    - answer_only is padded with the same id when present;
    - no RoBERTa/BERT-specific constant is hard-coded.

    This keeps:

        collate padding id
            ==
        language-model attention-mask padding id
            ==
        Q-Bound question-mask padding id.

    answer_only remains optional at collation time.  The V2.3 training pipeline
    requires it only when the functional critic is enabled, while official
    validation/test temporal inference does not require it.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    # --------------------------------------------------------------
    # Question-only tokens: [L] per sample -> [B, L_max].
    # --------------------------------------------------------------
    if "question" not in batch[0]:
        raise KeyError(
            "VideoQA sample is missing required field 'question'."
        )

    qmax_len = max(
        len(sample["question"])
        for sample in batch
    )

    for sample in batch:
        question = sample["question"]

        if not torch.is_tensor(question):
            raise TypeError(
                "Expected sample['question'] to be a torch.Tensor."
            )

        if question.ndim != 1:
            raise ValueError(
                "Expected sample['question'] to be [L], got "
                f"{tuple(question.shape)}."
            )

        pad_len = qmax_len - question.size(0)

        if pad_len <= 0:
            continue

        padding = torch.full(
            (pad_len,),
            fill_value=pad_token_id,
            dtype=question.dtype,
            device=question.device,
        )

        sample["question"] = torch.cat(
            [question, padding],
            dim=0,
        )

    # --------------------------------------------------------------
    # Existing CRA QA-candidate view: [Question SEP Answer_i].
    # In non-MC settings `answer` may be an integer/id, so only pad tensors.
    # --------------------------------------------------------------
    if (
        "answer" in batch[0]
        and not isinstance(batch[0]["answer"], int)
    ):
        _pad_candidate_tokens(
            batch,
            "answer",
            pad_token_id,
        )

    # --------------------------------------------------------------
    # V2.3 training-only privileged view: [Answer_i].
    # Optional for val/test and for configurations with critic disabled.
    # --------------------------------------------------------------
    _pad_candidate_tokens(
        batch,
        "answer_only",
        pad_token_id,
    )

    return default_collate(batch)


def build_dataloaders(
    cfgs,
    a2id,
    tokenizer,
    ddp=False,
):
    """
    Build train/validation/test dataloaders.

    Padding is tokenizer-agnostic and therefore does not need to be changed
    when switching among supported language encoders/tokenizers.

    DDP note
    --------
    DistributedSampler is passed through DataLoader(..., sampler=...), not
    batch_sampler=.  The latter expects a sampler that yields batches of
    indices and is mutually exclusive with batch_size.
    """
    pad_token_id = _resolve_pad_token_id(
        tokenizer
    )

    collate_fn = partial(
        videoqa_collate_fn,
        pad_token_id=pad_token_id,
    )

    train_dataset = VideoQADataset(
        cfgs,
        "train",
        tokenizer,
        a2id,
    )
    val_dataset = VideoQADataset(
        cfgs,
        "val",
        tokenizer,
        a2id,
    )

    dataset_name = str(
        cfgs["dataset"]["name"]
    ).lower()

    test_dataset = (
        VideoQADataset(
            cfgs,
            "test",
            tokenizer,
            a2id,
        )
        if dataset_name != "star"
        else None
    )

    # --------------------------------------------------------------
    # Distributed samplers.
    # --------------------------------------------------------------
    if ddp:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            shuffle=False,
        )
        test_sampler = (
            DistributedSampler(
                test_dataset,
                shuffle=False,
            )
            if test_dataset is not None
            else None
        )
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    batch_size = cfgs["dataset"][
        "batch_size"
    ]
    num_workers = cfgs["dataset"][
        "num_thread_reader"
    ]

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(
            False
            if ddp
            else True
        ),
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=bool(ddp),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=bool(ddp),
    )

    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            sampler=test_sampler,
            drop_last=False,
            collate_fn=collate_fn,
            pin_memory=bool(ddp),
        )
        if test_dataset is not None
        else None
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
