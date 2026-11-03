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
    # V3.5 / EQ-PTC: Q2 is training-only, and always padded independently
    # of Q1. Fail closed if *any* row includes incomplete pair metadata.
    # Otherwise default_collate() can silently discard Q2 when row 0 lacks it.
    # --------------------------------------------------------------
    pair_fields = ("paired_question", "pair_valid_mask", "paired_video_id",
                   "paired_question_id")
    presence = [any(key in sample for key in pair_fields) for sample in batch]
    if any(presence):
        for sample_index, sample in enumerate(batch):
            missing = [key for key in pair_fields if key not in sample]
            if missing:
                raise KeyError(f"V3.5 batch row {sample_index} has incomplete Q2: {missing}")

        pair_max_len = 0
        for sample_index, sample in enumerate(batch):
            question2 = sample["paired_question"]
            valid2 = sample["pair_valid_mask"]
            if (not torch.is_tensor(question2) or question2.ndim != 1
                    or question2.dtype != torch.long or question2.numel() < 1):
                raise TypeError("V3.5 paired_question must be LongTensor[L>=1].")
            if (not torch.is_tensor(valid2) or valid2.dtype != torch.bool
                    or valid2.ndim != 0):
                raise TypeError("V3.5 pair_valid_mask must be a bool scalar tensor.")
            if not isinstance(sample["paired_video_id"], str) or not isinstance(
                    sample["paired_question_id"], str):
                raise TypeError("V3.5 paired_video_id/paired_question_id must be strings.")
            # A valid Q2 must not be the same question or another video.
            if bool(valid2):
                if sample.get("video_id") != sample["paired_video_id"]:
                    raise ValueError(f"Q1/Q2 video ID mismatch at batch row {sample_index}.")
                if sample.get("question_id") == sample["paired_question_id"]:
                    raise ValueError(f"Q1/Q2 question ID identical at batch row {sample_index}.")
                # Trim only trailing padding: a leading special token may
                # legitimately share PAD=0 in some tokenizer families.
                def _trim_trailing_pad(tokens):
                    end = tokens.numel()
                    while end > 0 and int(tokens[end - 1]) == pad_token_id:
                        end -= 1
                    return tokens[:end]
                if torch.equal(
                    _trim_trailing_pad(sample["question"]),
                    _trim_trailing_pad(question2),
                ):
                    raise ValueError(f"Q1/Q2 token IDs identical at batch row {sample_index}.")
            pair_max_len = max(pair_max_len, question2.numel())

        for sample in batch:
            question2 = sample["paired_question"]
            if question2.size(0) < pair_max_len:
                padding = torch.full(
                    (pair_max_len - question2.size(0),), pad_token_id,
                    dtype=question2.dtype, device=question2.device,
                )
                sample["paired_question"] = torch.cat([question2, padding], dim=0)

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

    # H1 fair comparison: independently seeded train/val/test generators.
    # EQ-PTC consumes extra torch RNG at initialization; do not let that
    # perturb data order or DataLoader worker seeds relative to legacy.
    qbound = cfgs.get("model", {}).get("qbound", {})
    controlled_mode = qbound.get("temporal_context_mode") if isinstance(qbound, dict) else None
    h1_controlled = controlled_mode in ("legacy", "eq_ptc")
    data_seed = cfgs.get("misc", {}).get("seed", 0)
    if h1_controlled and (not isinstance(data_seed, int) or isinstance(data_seed, bool)):
        raise TypeError("H1 dataset RNG requires integer misc.seed.")

    def _rng(offset):
        return (torch.Generator().manual_seed(int(data_seed) + offset)
                if h1_controlled else None)

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
        generator=_rng(101),
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
        generator=_rng(202),
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
            generator=_rng(303),
        )
        if test_dataset is not None
        else None
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )
