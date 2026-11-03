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
# Audit only: contrast the paired question *text* for ordinal cases.
# Do NOT use this pattern to manufacture a Q2 negative without GT proof.
_H1_ORDINAL_TOKEN = re.compile(r"\b(first|second|third|fourth)\b")
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
        # H1's architectural ablation MUST keep the V3.5 Q2 sampling policy.
        # Distinct action anchors are a weak heuristic, not GT-disjoint proof.
        self.qdtg_pair_policy = "v35_distinct_action_anchors"
        self.qdtg_pair_audit = {}
        # Both H1 arms use stable candidate ordering. This changes only the
        # RNG-to-candidate mapping, not the candidate pool or sampling weights.
        self.h1_deterministic_sampling = qbound_cfg.get("temporal_context_mode") in ("legacy", "eq_ptc")
        if self.qdtg_pair_enabled:
            mode = qbound_cfg.get("temporal_context_mode", "legacy")
            if mode not in ("legacy", "eq_ptc"):
                raise ValueError("model.qbound.temporal_context_mode must be legacy or eq_ptc.")
            if (qbound_cfg.get("support_graph_mode") != "fixed" or
                    qbound_cfg.get("support_predictor_enabled") is not False):
                raise ValueError("V3.5 Q1/Q2 pairing requires R2-C fixed graph without predictor.")

        # Preserve the exact V3.5 sample indices, order and output shape.
        self._qbound_input_dim = (
            int(cfgs["model"]["d_model"])
            if bool(qbound_cfg.get("enabled", False)) else None
        )
        if int(cfgs["dataset"]["max_feats"]) < 2 and self._qbound_input_dim is not None:
            raise ValueError("Active Q-Bound requires dataset.max_feats >= 2.")

        self.v_questions = {}
        self.max_feats = cfgs["dataset"]["max_feats"]
        self.mc = cfgs["dataset"]["mc"]
        self.vg = cfgs["model"]["vg_loss"]
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
            feat_key = f'{self.feat_type}_I' if self.feat_type != 'Swin' else 'swin_2d'
            if 'vid' not in fp or feat_key not in fp:
                raise KeyError(
                    f"Missing HDF5 'vid' or feature key {feat_key!r} in {self.video_feature_path}"
                )
            vids = fp['vid']
            feats = fp[feat_key]
            if len(vids) != len(feats):
                raise ValueError("Feature HDF5 vid/feature row counts differ.")
            for vid, feat in zip(vids, feats):
                vid = vid.decode("utf-8") if isinstance(vid, bytes) else str(vid)
                if vid in self.frame_feats:
                    raise ValueError(f"Duplicate video id {vid!r} in feature HDF5.")
                self.frame_feats[vid] = feat
        # Existing np.linspace sampling repeats source rows when fnum<max_feats.
        # Preserve this behaviour for fair V3.5 comparison, but disclose it.
        if self._qbound_input_dim is not None:
            shorter = sum(f.shape[0] < self.max_feats for f in self.frame_feats.values())
            print(f"[H1 data audit] split={self.split}, feature_key={feat_key}, "
                  f"sampled_frames={self.max_feats}, short_source_videos={shorter}/"
                  f"{len(self.frame_feats)}; sampling=original_np_linspace")
         
        # with h5py.File(app_feat_file, 'r') as fp:
        #     vqids = fp['qid']
        #     feat_key = f'{feat_type}_I'
        #     feats = fp[feat_key]
        #     print(feats.shape) #v_num, clip_num, feat_dim
        #     for id, (vqid, feat) in enumerate(zip(vqids, feats)):
        #         vqid = vqid.decode("utf-8")
        #         self.frame_feats[str(vqid)] = feat


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
        # Audit the exact H1 blind spot without altering pair candidates.
        # E.g., first/second variants usually share action anchors and are not
        # positive training pairs under the existing policy. Importantly, this
        # text-only audit does NOT certify visually distinct GT intervals.
        contrast_count = 0
        contrast_qids = set()
        contrast_selected_as_candidates = 0
        for vid, indices in by_video.items():
            groups = {}
            for idx in indices:
                normalized_q = normalized[idx]
                ordinals = set(_H1_ORDINAL_TOKEN.findall(normalized_q))
                if len(ordinals) != 1:
                    continue
                context = _H1_ORDINAL_TOKEN.sub("<ordinal>", normalized_q)
                ordinal = next(iter(ordinals))
                groups.setdefault(context, []).append((idx, ordinal))
            for occurrences in groups.values():
                for a in range(len(occurrences)):
                    for b in range(a + 1, len(occurrences)):
                        if occurrences[a][1] != occurrences[b][1]:
                            contrast_count += 1
                            idx_a, idx_b = occurrences[a][0], occurrences[b][0]
                            contrast_qids.update((idx_a, idx_b))
                            if (idx_b in self._qdtg_pair_candidates.get(idx_a, ()) or
                                    idx_a in self._qdtg_pair_candidates.get(idx_b, ())):
                                contrast_selected_as_candidates += 1
        contrast_eligible = sum(bool(self._qdtg_pair_candidates.get(i)) for i in contrast_qids)
        self.qdtg_pair_audit = {
            "pair_policy": self.qdtg_pair_policy,
            "eligible_primary_questions": eligible,
            "total_training_questions": total,
            "duplicated_video_question_ids": len(duplicated),
            "text_only_ordinal_contrast_pairs": contrast_count,
            "ordinal_contrast_pairs_in_v35_candidate_index": contrast_selected_as_candidates,
            "ordinal_contrast_question_rows": len(contrast_qids),
            "ordinal_contrast_rows_with_v35_pair": contrast_eligible,
            "uses_temporal_gt": False,
        }
        print(f"[V3.5 QDTG/H1] train pair eligibility: {eligible}/{total} "
              f"({100.0 * eligible / max(1,total):.2f}%), "
              f"duplicate_vid_qid_excluded={len(duplicated)}, "
              "rule=distinct-action-anchors; NO GT/teacher")
        print(f"[H1 Q2 coverage] text-only first/second contrasts={contrast_count}; "
              f"ordinal_pairs_in_original_Q2_candidates={contrast_selected_as_candidates}; "
              f"question_rows={len(contrast_qids)}; "
              f"rows_with_any_original_V35_pair={contrast_eligible}; "
              "DO NOT interpret as verified GT-disjoint evidence")
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
    

    def get_vid_feats(self, vid_id):
        # Preserve V3.5's EXACT sampling indices: changing them would confound
        # the EQ-PTC vs legacy H1 experiment and frame2time_* evaluation.
        feat = self.frame_feats[vid_id]
        if feat.ndim != 2 or feat.shape[0] < 1:
            raise ValueError(f"Video {vid_id!r}: expected nonempty [F,D] features, "
                             f"got {tuple(feat.shape)}")
        if self._qbound_input_dim is not None and feat.shape[-1] != self._qbound_input_dim:
            raise ValueError(f"Video {vid_id!r}: Q-Bound expects feature dim "
                             f"{self._qbound_input_dim}, got {feat.shape[-1]}.")
        fnum = feat.shape[0]
        sp_fids = np.linspace(0, fnum - 1, self.max_feats, dtype=int)
        selected = feat[sp_fids]
        if (self._qbound_input_dim is not None and
                not np.isfinite(selected).all()):
            raise ValueError(f"Video {vid_id!r}: selected visual features contain NaN/Inf.")
        return selected
    
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
        vid_frames = self.get_vid_feats(vid_id)
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
        qtype = 'null' if 'type' not in cur_sample else cur_sample['type']

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
                same_v_qsn = set(self.v_questions[vid_id])
                same_v_other_qsn = (sorted(same_v_qsn - positive_questions)
                                    if self.h1_deterministic_sampling
                                    else list(same_v_qsn - positive_questions))
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
                        (sorted(cand_qsn) if self.h1_deterministic_sampling else list(cand_qsn)),
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
                    (sorted(cand_qsns) if self.h1_deterministic_sampling else list(cand_qsns)),
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
