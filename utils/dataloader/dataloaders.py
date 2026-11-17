from collections import defaultdict
from functools import partial
from heapq import heapify, heappop, heappush
from random import Random

import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from .dataset import VideoQADataset, _m1_data_flags


class M1GroupedBatchSampler(Sampler):
    """Shuffle within videos, then pack at most 16 distinct video chunks per batch.

    Each chunk contains up to 4 rows. Never repeat or drop rows to fill a batch,
    and never read a label or a timestamp. Batch size is therefore a ceiling.
    """

    QUESTIONS_PER_VIDEO = 4
    MAX_VIDEOS_PER_BATCH = 16

    def __init__(self, video_ids, batch_size, seed=0):
        if type(batch_size) is not int or batch_size < 2 * self.QUESTIONS_PER_VIDEO:
            raise ValueError("M1 grouped batching requires integer dataset.batch_size >= 8.")
        self.videos_per_batch = min(self.MAX_VIDEOS_PER_BATCH, batch_size // self.QUESTIONS_PER_VIDEO)
        self.seed = int(seed)
        self.epoch = 0
        groups = defaultdict(list)
        for index, video in enumerate(video_ids):
            groups[str(video).strip()].append(index)
        if len(groups) < 2:
            raise ValueError("M1 grouped batching needs at least two distinct train videos.")
        self.groups = dict(groups)
        self.num_rows = sum(map(len, self.groups.values()))
        width = self.QUESTIONS_PER_VIDEO
        chunks_per_video = [(len(rows) + width - 1) // width for rows in self.groups.values()]
        self.num_chunks = sum(chunks_per_video)
        self.num_batches = max(max(chunks_per_video),
                               (self.num_chunks + self.videos_per_batch - 1) // self.videos_per_batch)

    def __len__(self):
        return self.num_batches

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = Random(self.seed + self.epoch)
        video_groups = list(self.groups.values())
        rng.shuffle(video_groups)
        batches = [[] for _ in range(self.num_batches)]
        batch_slots = list(range(self.num_batches))
        rng.shuffle(batch_slots)
        # Assign one video's chunks to distinct least-loaded batches. Remove
        # every chosen slot before reinserting any slot from this video.
        slots = [(0, rank, index) for rank, index in enumerate(batch_slots)]
        heapify(slots)
        width = self.QUESTIONS_PER_VIDEO
        for rows in video_groups:
            indices = rows.copy()
            rng.shuffle(indices)
            chunks = [indices[i:i + width] for i in range(0, len(indices), width)]
            chosen = [heappop(slots) for _ in chunks]
            for chunk, (load, rank, index) in zip(chunks, chosen):
                batches[index].extend(chunk)
                heappush(slots, (load + 1, rank, index))

        # Load is balanced by number of chunks, not by question count. The
        # number of emitted batches stays identical across epochs for LR/resume.
        rng.shuffle(batches)
        for batch in batches:
            rng.shuffle(batch)
            yield batch


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
    v38_minimal: bool = False,
    v38_m1: bool = False,
    m1_grouped_train: bool = False,
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

    if v38_minimal or v38_m1:
        # Detect data-contract errors before the costly
        # RoBERTa/video forward. No labels or timestamps are added to the model.
        version = "V3.8-M1" if v38_m1 else "V3.8-M0"
        for sample_index, sample in enumerate(batch):
            qid = sample.get("question_id", f"batch_row_{sample_index}")
            video = sample.get("video_frames")
            if not isinstance(video, (torch.Tensor,)):
                # Original H5-backed dataset returns numpy arrays. Allow both.
                if not hasattr(video, "shape"):
                    raise TypeError(f"{version} {qid}: video_frames is not array-like.")
            if len(video.shape) != 2 or video.shape[0] != int(sample["video_len"]):
                raise ValueError(
                    f"{version} {qid}: video_len must equal sampled [T,D] "
                    "feature count, not the raw video duration or unique-index count."
                )
            for key in ("question", "answer"):
                if key not in sample or not torch.is_tensor(sample[key]):
                    raise TypeError(f"{version} {qid}: {key} must contain token IDs.")
            answers = sample["answer"]
            if answers.ndim != 2 or answers.size(0) != 5:
                raise ValueError(f"{version} {qid}: answer must be [5,L].")
            if not bool(answers.ne(pad_token_id).any(dim=-1).all()):
                raise ValueError(f"{version} {qid}: an answer candidate is all-PAD.")
            question = sample["question"]
            if question.ndim != 1 or not bool(question.ne(pad_token_id).any()):
                raise ValueError(f"{version} {qid}: question must be nonempty [L].")
            label = int(sample["answer_id"])
            minimum_label = -1 if v38_m1 and m1_grouped_train else 0
            if not minimum_label <= label < 5:
                raise ValueError(f"{version} {qid}: answer_id must be in [{minimum_label},4].")
            if not v38_minimal:
                continue
            indexes = sample.get("v38_sampled_feature_indices")
            if (not torch.is_tensor(indexes) or indexes.ndim != 1
                    or indexes.numel() != video.shape[0]):
                raise ValueError(f"V3.8-M0 {qid}: missing sampled-feature audit indices.")
            source_steps = int(sample["v38_source_feature_steps"])
            if source_steps < 1 or bool(((indexes < 0) | (indexes >= source_steps)).any()):
                raise ValueError(f"V3.8-M0 {qid}: sampled feature indices out of range.")

    if m1_grouped_train:
        for sample in batch:
            required = ("video_id", "question_type", "question_txt") if v38_m1 else ("video_id",)
            for name in required:
                value = sample.get(name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Grouped train batch needs a nonempty string {name} per row.")

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
    # V3.5 QDTG: independently PAD optional train-only Q2 question tokens.
    # Do not alter Q1 padding length or touch validation/test batches.
    # --------------------------------------------------------------
    if "paired_question" in batch[0]:
        if not all("paired_question" in sample and "pair_valid_mask" in sample
                   for sample in batch):
            raise KeyError("V3.5 pair fields must be present in every TRAIN row.")
        pair_max_len = max(sample["paired_question"].numel() for sample in batch)
        for sample in batch:
            question2 = sample["paired_question"]
            valid2 = sample["pair_valid_mask"]
            if (not torch.is_tensor(question2) or question2.ndim != 1
                    or question2.dtype != torch.long or question2.numel() < 1):
                raise TypeError("V3.5 paired_question must be LongTensor[L>=1].")
            if (not torch.is_tensor(valid2) or valid2.dtype != torch.bool
                    or valid2.numel() != 1):
                raise TypeError("V3.5 pair_valid_mask must be a bool scalar tensor.")
            if not isinstance(sample.get("paired_video_id"), str) or not isinstance(
                    sample.get("paired_question_id"), str):
                raise TypeError("V3.5 paired_video_id/paired_question_id must be strings.")
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

    qbound_cfg = cfgs.get("model", {}).get("qbound", {})
    v38_m1, m1_grouped = _m1_data_flags(cfgs)
    v38_minimal = bool(
        isinstance(qbound_cfg, dict)
        and qbound_cfg.get("enabled", False)
        and str(qbound_cfg.get("version", "")) == "3.8-minimal"
    )
    if v38_minimal:
        if int(cfgs["dataset"].get("mc", 0)) != 5:
            raise ValueError("V3.8-M0 requires exactly five MC candidates.")
        if int(cfgs["dataset"].get("max_feats", 0)) < 3:
            raise ValueError("V3.8-M0 requires at least three sampled feature steps.")
    if m1_grouped and ddp:
        raise ValueError("M1 grouped batching is single-process; it cannot use DistributedSampler.")
    if v38_m1:
        if cfgs["dataset"].get("mc") != 5:
            raise ValueError("V3.8-M1 requires dataset.mc=5.")
        max_feats = cfgs["dataset"].get("max_feats", 0)
        if type(max_feats) is not int or max_feats < 3:
            raise ValueError("V3.8-M1 requires integer dataset.max_feats>=3.")

    collate_fn = partial(
        videoqa_collate_fn,
        pad_token_id=pad_token_id,
        v38_minimal=v38_minimal,
        v38_m1=v38_m1,
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
    if v38_minimal:
        if not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("V3.8-M0 retained L_ground needs batch_size>=2.")
        if len(train_dataset) < batch_size:
            raise ValueError(
                "V3.8-M0 train split has fewer samples than batch_size; "
                "drop_last=True would produce zero training steps."
            )
        if not isinstance(num_workers, int) or num_workers < 0:
            raise ValueError("dataset.num_thread_reader must be a nonnegative integer.")

    if m1_grouped:
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("dataset.num_thread_reader must be a nonnegative integer.")
        grouped_sampler = M1GroupedBatchSampler(
            train_dataset.data["video_id"], batch_size, cfgs.get("misc", {}).get("seed", 0)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=grouped_sampler,
            num_workers=num_workers,
            collate_fn=partial(collate_fn, m1_grouped_train=True),
            pin_memory=False,
        )
        print(f"[M1 grouped sampler] {len(grouped_sampler.groups)} videos, "
              f"{grouped_sampler.num_rows} rows, {len(train_loader)} batches/epoch, "
              f"mean batch size={grouped_sampler.num_rows / len(train_loader):.2f}, "
              f"up to {grouped_sampler.videos_per_batch} videos x "
              f"{grouped_sampler.QUESTIONS_PER_VIDEO} rows/video; no repeats or dropped rows.")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=(False if ddp else True),
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
