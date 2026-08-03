import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_cra.only_trans import LanModel
from modules.m_tempclip.transformer import VideoEmbedding
from modules.m_cra.only_trans import Decoder as BaselineDecoder
from modules.m_cra.refine import Decoder as RefineDecoder
from modules.m_cra.causal_module import LinguisticInterventionGraph


class CRA(nn.Module):
    def __init__(self, cfgs, tokenizer) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.tokenizer = tokenizer
        self.baseline = cfgs["model"]["baseline"]
        self.qmax_words = cfgs["dataset"]["qmax_words"]
        self.max_feats = cfgs["dataset"]["max_feats"]
        self.n_negs = cfgs["model"]["n_negs"]

        d_model = cfgs["model"]["d_model"]

        # Language and video encoders.
        self.lang_model = LanModel(
            tokenizer,
            lan=cfgs["model"]["lan"],
            out_dim=d_model,
        )
        self.video_model = VideoEmbedding(cfgs)
        self.do = LinguisticInterventionGraph(cfgs)

        if self.baseline == "baseline":
            self.decoder = BaselineDecoder()
            self.forward = self.baseline_forward
        elif self.baseline == "refine":
            # Keep the same decoder construction used by the CRA baseline
            # and Ver0 so that Ver0.5 changes only the GPEB path. Note that
            # RefineDecoder's internal default dropout remains 0.1 here.
            self.decoder = RefineDecoder(
                num_layer=cfgs["model"]["num_layers"],
                cfgs=cfgs,
            )
            self.forward = self.refine_forward
        else:
            raise ValueError(
                "Unknown model.baseline value: "
                f"{self.baseline!r}. Expected 'baseline' or 'refine'."
            )

    @staticmethod
    def _prepare_answer_indices(
        answer_id,
        batch_size: int,
        device: torch.device,
    ):
        """Return candidate labels as a robust [B] LongTensor."""
        if answer_id is None:
            return None

        answer_indices = answer_id.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        ).reshape(-1)

        if answer_indices.numel() != batch_size:
            raise ValueError(
                "answer_id must contain one candidate index per sample. "
                f"Expected {batch_size} values, got "
                f"{answer_indices.numel()}."
            )

        return answer_indices

    @staticmethod
    def _select_candidate_features(
        candidate_features: torch.Tensor,
        candidate_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Select one [D] feature from [B, C, D] for every sample."""
        batch_indices = torch.arange(
            candidate_features.size(0),
            device=candidate_features.device,
        )
        return candidate_features[
            batch_indices,
            candidate_indices,
        ]

    def _encode_answers(self, answer, device):
        if answer is None:
            raise ValueError("answer tokens must not be None.")

        answer = answer.to(
            device=device,
            non_blocking=True,
        )
        answer_g, answer_w = self.lang_model(answer)

        if answer_g.ndim != 3:
            raise ValueError(
                "LanModel must return answer_g with shape [B, C, D], "
                f"got {tuple(answer_g.shape)}."
            )

        return answer_g, answer_w

    def refine_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
    ):
        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        answer_g, answer_w = self._encode_answers(
            answer,
            device,
        )

        batch_size = answer_g.size(0)
        answer_indices = self._prepare_answer_indices(
            answer_id,
            batch_size,
            answer_g.device,
        )

        video_f = self.video_model(video)
        qsn_gt = None

        if mode == "VQ" and answer_indices is not None:
            # Question-only branch in the original training pipeline passes
            # qsns_token_ids and qsns_id through the same language interface.
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
            qsn_gt = qsn_g
        else:
            # Preserve the original CRA behavior for a fair Ver0/Ver0.5
            # comparison. This is an element-wise maximum over candidate QA
            # representations; it is therefore QA-candidate-conditioned, not
            # a pure question-only representation.
            qsn_g = answer_g.max(dim=1).values

            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(
                    answer_g,
                    answer_indices,
                )

        use_gpeb = (
            self.cfgs["model"].get("gpeb_enabled", False)
            and mode == "VQA"
        )

        video_proj, video_attn, time_param = self.decoder(
            video_f,
            qsn_g,
            video_mask,
            gsub=gsub,
            do=self.cfgs["model"].get("do", False),
            window_sizes=self.cfgs["model"].get(
                "window_sizes",
                [1, 3, 5],
            ),
            use_gpeb=use_gpeb,
        )

        # Candidate answer embeddings are reused by both the main QA scorer
        # and the GPEB-only auxiliary QA loss in cra_pipeline.py.
        answer_proj = answer_g

        outs = {
            "fatt": video_attn,
            "time_param": time_param,
            "align_gt": qsn_gt,
            "qa_condition": qsn_g,
            "gpeb_active": use_gpeb,
        }
        return video_proj, answer_proj, outs

    def baseline_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
    ):
        device = video.device
        answer_g, answer_w = self._encode_answers(
            answer,
            device,
        )

        answer_indices = self._prepare_answer_indices(
            answer_id,
            answer_g.size(0),
            answer_g.device,
        )

        video_f = self.video_model(video)

        if answer_indices is not None:
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
        else:
            qsn_g = answer_g.max(dim=1).values

        video_proj, video_attn = self.decoder(
            video_f,
            answer_w,
            video_mask,
        )
        answer_proj = answer_g

        outs = {
            "fatt": video_attn,
            "qa_condition": qsn_g,
        }
        return video_proj, answer_proj, outs

    def align_loss(self, qsn, grounding_feature, neg_sample=None):
        def _sample_negatives(x_pos, k):
            """Sample k negatives for each item, with replacement."""
            batch_size, feature_dim = x_pos.size()

            if batch_size < 2:
                raise ValueError(
                    "InfoNCE negative sampling requires batch_size >= 2."
                )

            x_neg = torch.zeros(
                batch_size,
                k,
                feature_dim,
                device=x_pos.device,
                dtype=x_pos.dtype,
            )

            for index in range(batch_size):
                candidate_indices = torch.cat(
                    [
                        torch.arange(
                            0,
                            index,
                            device=x_pos.device,
                        ),
                        torch.arange(
                            index + 1,
                            batch_size,
                            device=x_pos.device,
                        ),
                    ]
                )
                sampled_positions = torch.randint(
                    low=0,
                    high=candidate_indices.numel(),
                    size=(k,),
                    device=x_pos.device,
                )
                x_neg[index] = x_pos[
                    candidate_indices[sampled_positions]
                ]

            return x_neg

        if neg_sample is None:
            negative_count = max(
                1,
                self.cfgs["dataset"]["batch_size"] // 2,
            )
            x_neg = _sample_negatives(
                grounding_feature,
                negative_count,
            )
        else:
            x_neg = neg_sample

        return self._info_nce_loss(
            qsn,
            grounding_feature,
            x_neg,
        )

    @staticmethod
    def _info_nce_loss(
        x,
        x_pos,
        x_neg,
        temperature=0.07,
    ):
        """Compute InfoNCE with one positive and K negatives."""
        x = F.normalize(x, dim=-1)
        x_pos = F.normalize(x_pos, dim=-1)
        x_neg = F.normalize(x_neg, dim=-1)

        pos_sim = torch.sum(
            x * x_pos,
            dim=-1,
            keepdim=True,
        ) / temperature

        neg_sim = torch.bmm(
            x.unsqueeze(1),
            x_neg.transpose(1, 2),
        ).squeeze(1) / temperature

        logits = torch.cat(
            [pos_sim, neg_sim],
            dim=-1,
        )
        labels = torch.zeros(
            x.size(0),
            dtype=torch.long,
            device=x.device,
        )
        return F.cross_entropy(logits, labels)
