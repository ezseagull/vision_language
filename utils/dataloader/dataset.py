import os.path as osp
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
import re

# ============================================================================
# V3.5 QDTG pair selection (TRAIN ONLY).
# Conservative weak-label heuristic; NOT a temporal-GT/pseudo-label oracle.
# The loss assumes distinct temporal evidence, which cannot be proved from text.
# Do not silently fall back to arbitrary same-video pairs when a sample fails.
# ============================================================================
_QDTG_ACTIONS = (
    "walk", "run", "open", "close", "sit", "stand", "hold", "pick", "take",
    "eat", "drink", "throw", "jump", "move", "enter", "exit", "leave", "arrive",
    "approach", "lift", "drop", "kick", "hit", "push", "pull", "climb",
    "fall", "carry", "turn", "ride", "drive", "park", "cook", "cut", "wash",
    "clean", "place", "grab", "pass", "catch", "bring", "dance", "sing",
    "talk", "speak", "point", "wave", "touch", "hug", "kiss", "shake",
    "bend", "pour", "stir", "fold", "unfold", "write", "read", "draw",
    "paint", "put", "remove", "attach", "detach", "unlock", "lock",
    "wear", "dress", "undress", "clap", "swim", "fly", "chase", "follow",
    "hide", "search", "find", "bite", "feed", "pet", "brush", "sweep",
    "pack", "unpack", "plant", "dig", "build", "break", "repair", "tear",
    "slide", "crawl", "roll", "bow", "nod", "swing", "hang", "step",
    "shoot", "play", "jump", "reach", "collect", "select", "press", "type",
)
_QDTG_IRREGULAR = {
    "ran": "run", "running": "run",
    "sat": "sit", "sitting": "sit", "stood": "stand", "standing": "stand",
    "held": "hold", "holding": "hold", "took": "take", "taken": "take",
    "ate": "eat", "eaten": "eat", "drank": "drink", "drunk": "drink",
    "threw": "throw", "thrown": "throw", "left": "leave",
    "fell": "fall", "fallen": "fall", "rode": "ride", "ridden": "ride",
    "drove": "drive", "driven": "drive", "caught": "catch",
    "brought": "bring", "wore": "wear", "worn": "wear",
    "putting": "put", "cutting": "cut", "hitting": "hit",
    "bit": "bite", "bitten": "bite", "shot": "shoot", "shotting": "shoot",
    "broke": "break", "broken": "break", "found": "find",
}

def _qdtg_forms():
    result = dict(_QDTG_IRREGULAR)
    for base in set(_QDTG_ACTIONS):
        result[base] = base
        result[base + "s"] = base
        result[base + "ed"] = base
        result[base + "ing"] = base
        if base.endswith("e"):
            result[base[:-1] + "ing"] = base
            result[base + "d"] = base
        if base.endswith("y") and len(base) >= 2 and base[-2] not in "aeiou":
            result[base[:-1] + "ies"] = base
            result[base[:-1] + "ied"] = base
        if base.endswith(("s", "sh", "ch", "x")):
            result[base + "es"] = base
        if len(base) > 2 and base[-1] not in "aeiouwxy" and base[-2] in "aeiou":
            result[base + base[-1] + "ing"] = base
            result[base + base[-1] + "ed"] = base
    return result

_QDTG_ACTION_FORMS = _qdtg_forms()
_QDTG_TOKENS = re.compile(r"[a-z]+")
_QDTG_GENERIC = {
    "do", "make", "have", "get", "use", "look", "see", "be", "go", "show",
    "start", "stop", "begin", "finish", "appear", "seem", "happen",
}
_QDTG_ACTION_EXCLUSIONS = {
    # Frequent verbs that can apply to the same visual instant/whole clip.
    "look", "watch", "see", "move", "play", "find", "follow", "reach",
}

def _qdtg_normalize_question(value):
    return " ".join(_QDTG_TOKENS.findall(str(value).casefold()))

def _qdtg_action_anchors(value):
    anchors = set()
    for token in _QDTG_TOKENS.findall(str(value).casefold()):
        base = _QDTG_ACTION_FORMS.get(token)
        if base and base not in _QDTG_GENERIC and base not in _QDTG_ACTION_EXCLUSIONS:
            anchors.add(base)
    return frozenset(anchors)

from utils.misc.misc import load_file, group, get_qsn_type, tokenize
from utils.dataloader.prepare_video import video_sampling, prepare_input


def _m1_data_flags(cfgs):
    qbound = cfgs.get("model", {}).get("qbound", {})
    m1 = (isinstance(qbound, dict) and bool(qbound.get("enabled", False))
          and qbound.get("version") == "3.8-m1")
    grouped = cfgs["dataset"].get("m1_grouped_batch", False)
    if type(grouped) is not bool:
        raise TypeError("dataset.m1_grouped_batch must be boolean.")
    return m1, m1 or grouped


class BaseDataset(Dataset):
    def __init__(self, cfgs, split, transform=None):
        pass

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self):
        pass



class VideoQADataset(Dataset):
    def __init__(self, cfgs, split, tokenizer, a2id):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features_path: dictionary  to video frames
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param tokenizer: BERT tokenizer
        :param a2id: answer to index mapping
        :param ivqa: whether to use iVQA or not
        :param max_feats: maximum frames to sample from a video
        """
        # self.anno_path = osp.dirname(csv_path)
        self.split = split
        csv_path = osp.join(cfgs["dataset"]["csv_path"], f"{split}.csv")
        self.data = pd.read_csv(csv_path, keep_default_na=False)
        self.dset = csv_path.split('/')[-2]
        self.v38_m1_enabled, grouped = _m1_data_flags(cfgs)
        self.m1_grouped_train = grouped and split == "train"
        if self.v38_m1_enabled or self.m1_grouped_train:
            self._m1_check_csv_metadata(csv_path)
        
        self.video_feature_path = osp.join(cfgs["dataset"]["features_path"], f"{split}.h5")
        self.feat_type = cfgs["dataset"]["feat_type"]
        self.use_frame = True
        self.use_mot =  False
        self.qmax_words = cfgs["dataset"]["qmax_words"]
        self.amax_words = cfgs["dataset"]["amax_words"]
        self.a2id = a2id
        self.tokenizer = tokenizer
        
        # NOTE gsub
        # self.gsub = load_file(f"data/nextgqa/gsub_{split}.json") if split != "train" else None
        # self.gsub = load_file(f"data/{cfgs['dataset']['name']}/gsub_{split}.json") if split != "train" else None
        self.gsub = (
            load_file(
                osp.join(
                    cfgs["dataset"]["csv_path"],
                    f"gsub_{split}.json",
                )
            )
            if split != "train"
            else None
        )
        
        # Inference and the R2-C control path never create a Q2 sample.
        qbound_cfg = cfgs.get("model", {}).get("qbound", {})
        pair_cfg = qbound_cfg.get("pair_loss", {}) if isinstance(qbound_cfg, dict) else {}
        if pair_cfg is None:
            pair_cfg = {}
        if not isinstance(pair_cfg, dict):
            raise TypeError("model.qbound.pair_loss must be a mapping.")
        pair_enabled = pair_cfg.get("enabled", False)
        if not isinstance(pair_enabled, bool):
            raise TypeError("model.qbound.pair_loss.enabled must be boolean.")
        if pair_enabled and not qbound_cfg.get("enabled", False):
            raise ValueError("V3.5 requires model.qbound.enabled=true.")
        self.qdtg_pair_enabled = bool(pair_enabled and split == "train")
        self._qdtg_pair_candidates = {}

        self.v_questions = {}
        self.max_feats = cfgs["dataset"]["max_feats"]
        self.mc = cfgs["dataset"]["mc"]
        self.vg = cfgs["model"]["vg_loss"]

        # V3.8-M0 DATA-ONLY audit; the original V3.5 input tensors and temporal
        # sampling remain unchanged. No temporal GT is read on TRAIN.
        self.v38_minimal_enabled = bool(
            isinstance(qbound_cfg, dict)
            and qbound_cfg.get("enabled", False)
            and str(qbound_cfg.get("version", "")) == "3.8-minimal"
        )
        if self.v38_minimal_enabled:
            v38_cfg = qbound_cfg.get("v38", {})
            if not isinstance(v38_cfg, dict):
                raise TypeError("model.qbound.v38 must be a mapping/dict.")
            phase = v38_cfg.get("train_phase")
            if phase not in ("probe", "joint"):
                raise ValueError("V3.8-M0 requires train_phase='probe' or 'joint'.")
            if self.mc != 5:
                raise ValueError("V3.8-M0 Joint Answer-Event requires dataset.mc=5.")
            if not isinstance(self.max_feats, int) or self.max_feats < 3:
                raise ValueError("V3.8-M0 requires dataset.max_feats>=3.")
            if split == "train" and phase == "probe" and pair_enabled:
                raise ValueError(
                    "V3.8-M0 probe freezes V3.5/QDTG; set "
                    "model.qbound.pair_loss.enabled=false and loss_weight=0."
                )
        self.mode = osp.basename(csv_path).split('.')[0] #train, val or test
        
        # NOTE 除了NextGQA都要False
        # if osp.exists(osp.join(anno_path, 'train_gpt4_sub.json')):
        self.agu = True
        
        if self.mode not in ['val', 'test']:
            self.all_answers = set(self.data['answer'])
            self.all_questions = set(self.data['question'])
            self.ans_group, self.qsn_group = group(self.data, gt=False)

            if self.agu:
                anno_path = osp.dirname(csv_path)
                agu_file = osp.join(anno_path, 'train_gpt4_sub.json')
                if osp.exists(osp.join(anno_path, 'train_gpt4_sub.json')) is False:
                    self.agu = False
                else:
                    self.qsn_agu = load_file(agu_file)

        self._gather_by_v()
        if self.qdtg_pair_enabled:
            self._qdtg_build_pairs()
        print('Load {}...'.format(self.video_feature_path))
        self.frame_feats = {}
        with h5py.File(self.video_feature_path, 'r') as fp:
            vids = fp['vid']
            feat_key = f'{self.feat_type}_I' if self.feat_type != 'Swin' else 'swin_2d'
            feats = fp[feat_key]
            # print(feats.shape) #v_num, clip_num, feat_dim
            for id, (vid, feat) in enumerate(zip(vids, feats)):
                vid = vid.decode("utf-8")
                self.frame_feats[str(vid)] = feat
         
        if self.v38_minimal_enabled or self.v38_m1_enabled:
            self._v38_check_feature_and_time_contract(cfgs)

        # with h5py.File(app_feat_file, 'r') as fp:
        #     vqids = fp['qid']
        #     feat_key = f'{feat_type}_I'
        #     feats = fp[feat_key]
        #     print(feats.shape) #v_num, clip_num, feat_dim
        #     for id, (vqid, feat) in enumerate(zip(vqids, feats)):
        #         vqid = vqid.decode("utf-8")
        #         self.frame_feats[str(vqid)] = feat


    def _m1_check_csv_metadata(self, path):
        """Read only row identity/question metadata, never temporal GT."""
        required = {"video_id", "qid", "question", "type"}
        if not required.issubset(self.data):
            raise ValueError(f"M1 CSV {path} lacks columns: {sorted(required - set(self.data))}.")
        if self.data.empty:
            raise ValueError(f"M1 CSV is empty: {path}")
        for name in ("video_id", "qid", "question", "type"):
            value = self.data[name].astype(str).str.strip()
            invalid = value.str.casefold().isin({"", "null", "none", "nan"})
            if invalid.any():
                raise ValueError(f"M1 CSV {path}: {int(invalid.sum())} rows with missing {name}.")
        if self.m1_grouped_train:
            counts = self.data["type"].astype(str).str.strip().value_counts().sort_index().to_dict()
            descriptive = sum(count for code, count in counts.items() if code.upper().startswith("D"))
            print(f"[M1 grouped data] train raw question types={counts}; "
                  f"descriptive rows kept for CRA: {descriptive}/{len(self.data)}")

    def _v38_check_feature_and_time_contract(self, cfgs):
        """Fail before training/eval on V3.8 feature/timestamp incompatibilities.

        This is an audit, not a new sampling policy. `frame2time_{split}` is
        validation/test-only; reading it on TRAIN would violate the weak-label
        data boundary. The official metric uses these same per-step timestamps.
        """
        version = "V3.8-M1" if self.v38_m1_enabled else "V3.8-M0"
        videos = {str(video_id) for video_id in self.data["video_id"]}
        if self.split in ("val", "test"):
            # CRAPipeline.eval stores predictions in a dict keyed by
            # video_id + '_' + qid. Duplicate IDs silently overwrite rows,
            # corrupting QA/grounding metrics. TRAIN duplicates remain legal:
            # the existing V3.5 pair index already excludes ambiguous pairs.
            qkeys = [
                (str(video_id), str(question_id))
                for video_id, question_id in zip(
                    self.data["video_id"], self.data["qid"]
                )
            ]
            if len(set(qkeys)) != len(qkeys):
                counts = {}
                for qkey in qkeys:
                    counts[qkey] = counts.get(qkey, 0) + 1
                examples = [key for key, count in counts.items() if count > 1][:5]
                raise ValueError(
                    f"{version} {self.split}: duplicate (video_id,qid) keys "
                    f"would overwrite eval predictions; examples: {examples}"
                )
        missing = sorted(videos.difference(self.frame_feats))
        if missing:
            raise KeyError(
                f"{version} {self.split}: {len(missing)} CSV video IDs have "
                f"no {self.feat_type} features; first IDs: {missing[:5]}"
            )

        expected_dim = int(cfgs["model"]["d_model"])
        repeated_videos = []
        raw_lengths = []
        for video_id in videos:
            feat = self.frame_feats[video_id]
            if feat.ndim != 2 or feat.shape[1] != expected_dim:
                raise ValueError(
                    f"{version} {video_id}: expected raw feature [F,{expected_dim}], "
                    f"got {tuple(feat.shape)}."
                )
            raw_len = int(feat.shape[0])
            if raw_len < 1:
                raise ValueError(f"{version} {video_id}: zero feature steps.")
            raw_lengths.append(raw_len)
            if raw_len < self.max_feats:
                repeated_videos.append(video_id)

        if self.split in ("val", "test"):
            timestamps_path = osp.join(
                cfgs["dataset"]["csv_path"], f"frame2time_{self.split}.json"
            )
            timestamps = load_file(timestamps_path)
            if not isinstance(timestamps, dict):
                raise TypeError(
                    f"{version} {timestamps_path} must contain video_id -> "
                    "per-feature-step timestamps."
                )
            missing_times = sorted(videos.difference(timestamps))
            if missing_times:
                raise KeyError(
                    f"{version} {self.split}: timestamps missing for "
                    f"{len(missing_times)} videos; first IDs: {missing_times[:5]}"
                )
            for video_id in videos:
                times = np.asarray(timestamps[video_id], dtype=np.float64).reshape(-1)
                if times.size != self.max_feats:
                    raise ValueError(
                        f"{version} {self.split}/{video_id}: frame2time has "
                        f"{times.size} positions but the existing sampler outputs "
                        f"{self.max_feats}. Do NOT silently interpolate the "
                        "temporal labels; fix the feature/timestamp mapping "
                        "before running official R1 evaluation."
                    )
                if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
                    raise ValueError(
                        f"{version} {self.split}/{video_id}: frame2time "
                        "must be finite and non-decreasing."
                    )
            # These timestamps are for evaluation only. Do not pass them into
            # Event Bank/Scorer or TRAIN dataset outputs.

        print(
            f"[{version} DATA AUDIT] split={self.split}; "
            f"videos={len(videos)}, sampled_steps={self.max_feats}, "
            f"raw_feature_steps_min={min(raw_lengths)}, "
            f"raw_feature_steps_max={max(raw_lengths)}, "
            f"videos_with_repeated_sample_indices={len(repeated_videos)}. "
            "Sampler unchanged; repeated sampled steps are not PAD."
        )

    def __len__(self):
        return len(self.data)
    
    def _gather_by_v(self):
        for idx, row in self.data.iterrows():
            vid, qsn, qtype = str(row['video_id']), row['question'], row['type']
            if qtype[0] == 'D': continue #omit descriptive question
            if vid not in self.v_questions:
                self.v_questions[vid] = [qsn]
            else:
                self.v_questions[vid].append(qsn)
        
    def _qdtg_build_pairs(self):
        """Pre-index same-video, different-action question pairs from TRAIN CSV.

        No temporal GT, validation data, answer labels, CLIP teacher or Q2
        augmentation is used.  Disjoint action-word anchors are an intentionally
        conservative *proxy* for distinct events, not a proof of disjoint GT.
        Duplicate (video_id, qid) identifiers and paraphrases are excluded.
        """
        # The source NExT-GQA CSV can contain duplicate (video_id, qid)
        # identifiers with different text. Exclude ALL colliding IDs from the
        # auxiliary pair index; do not treat them as unambiguous evidence.
        id_counts = {}
        for _, row in self.data.iterrows():
            key = (str(row["video_id"]), str(row["qid"]))
            id_counts[key] = id_counts.get(key, 0) + 1
        duplicated = {key for key, count in id_counts.items() if count > 1}

        by_video = {}
        anchors = {}
        normalized = {}
        qids = {}
        for index, row in self.data.iterrows():
            vid = str(row["video_id"])
            qid = str(row["qid"])
            qtext = str(row["question"])
            qtype = str(row["type"]) if "type" in row else ""
            if qtype.startswith("D") or (vid, qid) in duplicated:
                continue
            by_video.setdefault(vid, []).append(int(index))
            qids[int(index)] = f"{vid}_{qid}"
            normalized[int(index)] = _qdtg_normalize_question(qtext)
            anchors[int(index)] = _qdtg_action_anchors(qtext)

        for idx, row in self.data.iterrows():
            idx = int(idx)
            vid = str(row["video_id"])
            candidates = []
            primary_actions = anchors.get(idx, frozenset())
            if (vid, str(row["qid"])) not in duplicated and primary_actions and normalized.get(idx):
                for other in by_video.get(vid, ()):
                    if other == idx or qids.get(idx) == qids.get(other):
                        continue
                    second_actions = anchors.get(other, frozenset())
                    if not second_actions or not primary_actions.isdisjoint(second_actions):
                        continue
                    # No same-text question or whitespace/punctuation paraphrase.
                    if normalized[idx] == normalized[other]:
                        continue
                    # Reject near duplicate token bags; relation words alone are
                    # not evidence that temporal GT should be disjoint.
                    words1 = set(normalized[idx].split())
                    words2 = set(normalized[other].split())
                    overlap = len(words1 & words2) / max(1, len(words1 | words2))
                    if overlap >= 0.75:
                        continue
                    candidates.append(other)
            self._qdtg_pair_candidates[idx] = tuple(candidates)

        eligible = sum(bool(pairs) for pairs in self._qdtg_pair_candidates.values())
        total = len(self.data)
        print(f"[V3.5 QDTG] train pair eligibility: {eligible}/{total} "
              f"({100.0 * eligible / max(1,total):.2f}%), "
              f"duplicate_vid_qid_excluded={len(duplicated)}, "
              "rule=distinct-action-anchors; NO GT/teacher")
        if eligible == 0:
            raise RuntimeError(
                "V3.5 has zero eligible same-video pairs under the conservative "
                "distinct-action filter. Do NOT silently fall back to arbitrary "
                "pairs; audit train.csv and explicitly revise the pair policy."
            )

    def _qdtg_get_pair(self, index, video_id, qid, question_tensor, question_txt):
        """Return Q2, validity, and identity without modifying main RNG state.

        Called AFTER the entire R2-C __getitem__ body to preserve its random
        augmentation and negative-answer sampling. Restores Python RNG state so
        sampling Q2 does not perturb subsequent examples in an R2-C control run.
        """
        # R2-C augments Q1 without temporal-label guarantees. To avoid turning
        # an augmented paraphrase into a contradictory pair, conservatively
        # disable the new loss on rows whose final Q1 differs from CSV Q1.
        original_text = _qdtg_normalize_question(self.data.loc[int(index), "question"])
        if _qdtg_normalize_question(question_txt) != original_text:
            return (question_tensor.clone(), False, str(video_id),
                    f"{video_id}_{qid}", str(question_txt), 0)

        options = self._qdtg_pair_candidates.get(int(index), ())
        q1_text = _qdtg_normalize_question(question_txt)
        eligible = [j for j in options if
                    _qdtg_normalize_question(self.data.loc[j, "question"]) != q1_text]
        if not eligible:
            # Pipeline/CRA mask invalid examples; the placeholder Q2 must still
            # have valid tokens for LanModel's per-row attention mask.
            return (question_tensor.clone(), False, str(video_id),
                    f"{video_id}_{qid}", str(question_txt), 0)

        rng_state = random.getstate()
        try:
            partner = random.choice(eligible)
        finally:
            random.setstate(rng_state)
        paired_row = self.data.loc[partner]
        paired_video_id = str(paired_row["video_id"])
        if paired_video_id != str(video_id):
            raise RuntimeError("V3.5 pair index violates same-video provenance.")
        paired_qid = f"{paired_video_id}_{str(paired_row['qid'])}"
        if paired_qid == f"{video_id}_{qid}":
            raise RuntimeError("V3.5 pair index selected identical question ID.")
        paired_text = str(paired_row["question"])
        encoded = self.tokenizer.encode(
            paired_text, add_special_tokens=True, padding="longest",
            max_length=self.qmax_words, truncation=True,
        )
        paired_question = torch.tensor(encoded, dtype=torch.long)
        if paired_question.numel() < 1:
            raise RuntimeError("V3.5 selected an empty tokenized Q2.")
        # Distinct raw texts can become identical under tokenizer truncation.
        if (paired_question.shape == question_tensor.shape and
                torch.equal(paired_question, question_tensor)):
            return (question_tensor.clone(), False, str(video_id),
                    f"{video_id}_{qid}", str(question_txt), 0)
        return (paired_question, True, paired_video_id, paired_qid,
                paired_text, len(eligible))

    def get_vid_frames(self, vid_id):
        #deprecated as extracting features offline is much more effcient
        sp_mode = 'uniC' if self.mode == 'train' else 'uniC'

        vid_path = osp.join(self.video_feature_path, vid_id)
        frames = video_sampling(vid_path, mode=sp_mode, frame_num=self.max_feats)
        video_inputs = prepare_input(frames)

        return video_inputs
    

    def get_vid_feats(self, vid_id, return_sampling_metadata=False):
        # Preserve the canonical CRA/V3.5 uniform-index sampler EXACTLY.
        # A source with fewer than max_feats positions repeats indices; this
        # is resampling, NOT padding. Changing the mask would invalidate the
        # old-support vs Event-Bank comparison and frame2time evaluation.
        feat = self.frame_feats[vid_id]
        fnum = feat.shape[0]
        if self.v38_minimal_enabled and fnum < 1:
            raise ValueError(f"V3.8-M0 {vid_id}: empty video features.")
        sp_fids = np.linspace(0, fnum-1, self.max_feats, dtype=int)
        feat = feat[sp_fids]
        if (self.v38_minimal_enabled or self.v38_m1_enabled) and not np.isfinite(feat).all():
            version = "V3.8-M1" if self.v38_m1_enabled else "V3.8-M0"
            raise ValueError(f"{version} {vid_id}: sampled visual features contain NaN/Inf.")
        if return_sampling_metadata:
            return feat, sp_fids, int(fnum)
        return feat
    
    def get_vqid_feats(self, vqid):

        feat = self.frame_feats[vqid]
        vlen = feat.shape[0]
        
        return feat, vlen


    def __getitem__(self, index):
        
        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video_id"]
        vid_id = str(vid_id)
        qid =  str(cur_sample['qid'])
        if self.split != "train":
            gsub = self.gsub[vid_id]
            duration = gsub["duration"]
            gsub = torch.tensor(gsub["location"][qid][0]) / float(duration)
        else:
            gsub = None

        # vid_frames = self.get_vid_frames(vid_id)

        vid_qid = f'{vid_id}_{qid}'
        # vid_frames = self.frame_feats[vid_qid]
        if self.v38_minimal_enabled:
            vid_frames, sampled_indices, source_feature_steps = self.get_vid_feats(
                vid_id, return_sampling_metadata=True
            )
            if vid_frames.ndim != 2 or vid_frames.shape != (
                self.max_feats, int(self.frame_feats[vid_id].shape[-1])
            ):
                raise ValueError("V3.8-M0 sampled video features must remain [T,D].")
        else:
            vid_frames = self.get_vid_feats(vid_id)
        # All T returned positions are valid resampled features, including
        # repeated positions; do not invent a shorter PAD length here.
        vlen = self.max_feats
        
        # --------------------------------------------------------------
        # FIX 1: Keep all question representations synchronized.
        # Augment first, then tokenize the FINAL question used by this
        # sample. Preserve the original question so that an augmented
        # paraphrase is not paired against its original form as a negative.
        # --------------------------------------------------------------
        original_question_txt = cur_sample['question']
        question_txt = original_question_txt

        if self.mode == 'train' and self.agu and vid_qid in self.qsn_agu:
            agus = self.qsn_agu[vid_qid].get('gen', [])
            if agus and random.random() < 0.3:
                question_txt = random.choice(agus).rstrip('?')

        if self.mc >= 0:
            qsn_tk_id = torch.tensor(
                self.tokenizer.encode(
                    question_txt,
                    add_special_tokens=True,
                    padding="longest",
                    max_length=self.qmax_words,
                    truncation=True,
                ),
                dtype=torch.long
            )
            q_len = torch.tensor([len(qsn_tk_id)], dtype=torch.long)
        else:
            qsn_tk_id = torch.tensor([0], dtype=torch.long)
            q_len = torch.tensor([0], dtype=torch.long)

        qtype, ans_token_ids, answer_len = 0, 0, 0
        # max_seg_num = self.amax_words
        # seg_feats = torch.zeros(self.mc, max_seg_num, 2048)
        # seg_num = torch.LongTensor(self.mc)

        qsns_id, qsns_token_ids, qsns_seq_len = 0, 0, 0
        raw_qtype = 'null' if 'type' not in cur_sample else cur_sample['type']
        qtype = raw_qtype

        if self.vg and self.mode not in ['val','test']:
            try:
                qtype = get_qsn_type(question_txt, qtype)
            except:
                print(vid_qid, question_txt)
            neg_num = 5
            if qtype not in self.qsn_group or len(self.qsn_group[qtype]) < neg_num-1:
                valid_qsncans = self.all_questions
            else:
                valid_qsncans = self.qsn_group[qtype]
            
            # ----------------------------------------------------------
            # FIX 2: Exclude the positive question correctly.
            # `set(question_txt)` is a set of characters, not a set
            # containing the full question. If augmentation is active,
            # exclude both the original and the paraphrased positive.
            # ----------------------------------------------------------
            positive_questions = {original_question_txt, question_txt}

            if random.random() < 0.3:
                same_v_qsn = set(self.v_questions.get(vid_id, ()) if self.m1_grouped_train
                                 else self.v_questions[vid_id])
                same_v_other_qsn = list(same_v_qsn - positive_questions)
                num_other = len(same_v_other_qsn)

                if num_other >= self.mc - 1:
                    qchoices = random.sample(
                        same_v_other_qsn,
                        self.mc - 1
                    )
                else:
                    needed = self.mc - 1 - num_other
                    cand_qsn = (
                        set(valid_qsncans)
                        - same_v_qsn
                        - positive_questions
                    )

                    if len(cand_qsn) < needed:
                        cand_qsn = (
                            set(self.all_questions)
                            - same_v_qsn
                            - positive_questions
                        )

                    if len(cand_qsn) < needed:
                        raise RuntimeError(
                            f"Not enough negative questions for {vid_qid}: "
                            f"need {needed}, found {len(cand_qsn)}"
                        )

                    qchoices = same_v_other_qsn + random.sample(
                        list(cand_qsn),
                        needed
                    )
            else:
                cand_qsns = set(valid_qsncans) - positive_questions

                if len(cand_qsns) < self.mc - 1:
                    cand_qsns = (
                        set(self.all_questions)
                        - positive_questions
                    )

                if len(cand_qsns) < self.mc - 1:
                    raise RuntimeError(
                        f"Not enough negative questions for {vid_qid}: "
                        f"need {self.mc - 1}, found {len(cand_qsns)}"
                    )

                qchoices = random.sample(
                    list(cand_qsns),
                    self.mc - 1
                )
            """
            same_v_qsn = set(self.v_questions[vid_id])
            same_v_other_qsn = list(same_v_qsn - set(question_txt))
            num_other = len(same_v_other_qsn)
            cand_qsn = valid_qsncans - same_v_qsn
            
            if num_other >= 2:
                qchoices = rd.sample(same_v_other_qsn, 2)
            else:
                add = rd.sample(list(cand_qsn), 2-num_other)
                qchoices = same_v_other_qsn + add
                cand_qsn = cand_qsn - set(add)
                
            qchoices.extend(rd.sample(list(cand_qsn), 2))
            """ 
            qchoices.append(question_txt)
            random.shuffle(qchoices)
            qsns_id = qchoices.index(question_txt)
            qsns_token_ids, qsn_tokens = tokenize(
                    qchoices,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.qmax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            qsns_seq_len = torch.tensor([len(qsn) for qsn in qsns_token_ids], dtype=torch.long)
        
        question_id = vid_id +'_'+str(cur_sample["qid"])

        # --------------------------------------------------------------
        # V2.1 DATA CONTRACT
        # Keep CRA's existing Question[SEP]Answer candidates unchanged,
        # but expose a second answer-only view for Q-Bound supervision.
        #
        # IMPORTANT:
        #   - Q-Bound grounding must still use Video + Question only.
        #   - `answer_only` is training supervision / answer evidence only;
        #     it must never be fed into the temporal segment scorer.
        # --------------------------------------------------------------
        answer_only_txts = None
        answer_only_token_ids = None
        answer_only_len = None

        if self.mc:
            ans = cur_sample['answer']
            choices = [str(cur_sample["a" + str(i)]) for i in range(self.mc)]
            answer_id = choices.index(ans) if ans in choices else -1
            if self.v38_minimal_enabled and answer_id < 0:
                raise ValueError(
                    f"V3.8-M0 {vid_qid}: GT answer is absent from the five "
                    "answer candidates. Cannot train/evaluate evidence-only QA."
                )

            if self.mode not in ['val', 'test'] and random.random() < 0.3:
                try:                    
                    qtype = get_qsn_type(question_txt, qtype)
                except:
                    print(vid_qid, question_txt)
                if qtype not in self.ans_group or len(self.ans_group[qtype]) < self.mc-1:
                    valid_anscans = self.all_answers
                else:
                    valid_anscans = self.ans_group[qtype]
                
                cand_answers = valid_anscans - {ans}
                choices = random.sample(sorted(cand_answers), self.mc-1)
                choices.append(ans)

                random.shuffle(choices)
                answer_id = choices.index(ans)
                
                # print(question_txt, choices, ans)
        
            answer_txts = [question_txt+f' {self.tokenizer.sep_token} '+ opt for opt in choices]
               
            try:
                ans_token_ids, answer_tokens = tokenize(
                    answer_txts,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.amax_words,
                    dynamic_padding=False,
                    truncation=True
                )
                
            except Exception as e:
                print("Fail to tokenize:", answer_txts)
                raise e
            qas_len = torch.tensor([len(ans) for ans in ans_token_ids], dtype=torch.long)
            if self.v38_minimal_enabled:
                if (not torch.is_tensor(ans_token_ids) or ans_token_ids.ndim != 2
                        or ans_token_ids.size(0) != 5):
                    raise ValueError(
                        f"V3.8-M0 {vid_qid}: answer input must be [5,L] "
                        "token IDs, preserving CRA's [Question SEP Answer_i]."
                    )
                pad_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_id is None or not bool(ans_token_ids.ne(int(pad_id)).any(dim=1).all()):
                    raise ValueError(f"V3.8-M0 {vid_qid}: empty/PAD-only answer candidate.")
                if torch.unique(ans_token_ids, dim=0).size(0) == 1:
                    raise ValueError(
                        f"V3.8-M0 {vid_qid}: all five answer candidates "
                        "become identical token sequences after tokenization "
                        "(possible question-prefix truncation)."
                    )

            # V2.1: tokenize the same candidate choices WITHOUT prepending
            # the question. This avoids the Q <-> [Q SEP A] language shortcut
            # when answer candidates supervise the temporal evidence.
            answer_only_txts = list(choices)
            try:
                answer_only_token_ids, _ = tokenize(
                    answer_only_txts,
                    self.tokenizer,
                    add_special_tokens=True,
                    max_length=self.amax_words,
                    dynamic_padding=False,
                    truncation=True
                )
            except Exception as e:
                print("Fail to tokenize answer-only candidates:", answer_only_txts)
                raise e

            answer_only_len = torch.tensor(
                [len(ans) for ans in answer_only_token_ids],
                dtype=torch.long
            )
        else:
            answer_txts = cur_sample["answer"]
            answer_id = self.a2id.get(answer_txts, -1)  # answer_id -1 if not in top answers, that will be considered as wrong prediction during evaluation
        
        sample = {
            "video_id": vid_id,
            "video_frames": vid_frames,
            "video_len": vlen,
            "question": qsn_tk_id,
            "question_txt": question_txt,
            "type": qtype,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": ans_token_ids,
            "qas_len": qas_len,
            "question_id": question_id,
            "qsns_id": qsns_id,
            "qsns_token_ids": qsns_token_ids,
            "qsns_seq_len": qsns_seq_len,
            "q_len": q_len,
        }

        if self.m1_grouped_train:
            # CRA may rewrite `type` during VQ/answer augmentation. M1 must
            # decide descriptive eligibility from the original CSV type.
            sample["question_type"] = str(raw_qtype).strip()

        if self.v38_minimal_enabled:
            # Auditable sampling geometry only. These fields are NOT oracle
            # labels, are never sent to Event Scorer, and must not change the
            # sampled [T,D] video or its video_len. Pipeline R1 still reads
            # frame2time_{val,test}.json independently.
            unique_count = int(np.unique(sampled_indices).size)
            sample.update({
                "v38_source_feature_steps": int(source_feature_steps),
                "v38_sampled_feature_indices": torch.from_numpy(
                    sampled_indices.astype(np.int64, copy=True)
                ),
                "v38_unique_sampled_feature_steps": unique_count,
                "v38_repeated_sampled_step_fraction": float(
                    1.0 - unique_count / self.max_feats
                ),
            })

        # V2.1-specific fields. For NExT-GQA (mc=5), these are always present.
        # Keeping them separate preserves the original CRA input exactly.
        if self.mc:
            sample.update({
                "answer_only_txt": answer_only_txts,
                "answer_only": answer_only_token_ids,
                "answer_only_len": answer_only_len,
            })

        if self.qdtg_pair_enabled:
            # Q2 is TRAIN-only: no changes to R2-C QA/negative candidates or
            # validation/test data. No separate Q2 answer, features, or GT.
            q2, valid, pair_vid, pair_qid, pair_text, count = self._qdtg_get_pair(
                index, vid_id, qid, qsn_tk_id, question_txt,
            )
            sample.update({
                "paired_question": q2,
                "pair_valid_mask": torch.tensor(valid, dtype=torch.bool),
                "paired_video_id": pair_vid,
                "paired_question_id": pair_qid,
                "paired_question_txt": pair_text,  # audit-only
                "pair_candidate_count": int(count),  # audit-only
            })

        if self.split != "train":
            sample["gsub"] = gsub

        return sample
