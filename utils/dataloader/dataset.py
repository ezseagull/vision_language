import os.path as osp
import torch
import h5py
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import random
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
        
    def get_vid_frames(self, vid_id):
        #deprecated as extracting features offline is much more effcient
        sp_mode = 'uniC' if self.mode == 'train' else 'uniC'

        vid_path = osp.join(self.video_feature_path, vid_id)
        frames = video_sampling(vid_path, mode=sp_mode, frame_num=self.max_feats)
        video_inputs = prepare_input(frames)

        return video_inputs
    

    def get_vid_feats(self, vid_id):
        feat =  self.frame_feats[vid_id]
        fnum = feat.shape[0]
        sp_fids = np.linspace(0, fnum-1, self.max_feats, dtype=int)
        feat = feat[sp_fids]
        
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

        if self.split != "train":
            sample["gsub"] = gsub

        return sample
