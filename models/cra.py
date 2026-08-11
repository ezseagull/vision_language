import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_cra.only_trans import LanModel
from modules.m_tempclip.transformer import VideoEmbedding
from modules.m_cra.only_trans import Decoder as BaselineDecoder
from modules.m_cra.refine import Decoder as RefineDecoder
from modules.m_cra.causal_module import LinguisticInterventionGraph


class CRA(nn.Module):
    """
    CRA wrapper with optional Q-Bound Version 1.1.3 temporal reasoning.

    Responsibilities of this wrapper are deliberately limited:

    * preserve the original CRA candidate-conditioned VQ/VQA language path;
    * encode a separate question-only representation for Q-Bound;
    * pass the deterministic CRA -> Q-Bound handover weight ``qbound_mix``
      to the refine decoder;
    * expose the decoder's final temporal key and diagnostics unchanged.

    Q-Bound never receives answer-choice-conditioned features for topology
    induction.  The actual CRA/Q-Bound temporal handover is implemented in
    modules.m_cra.refine.Decoder, not in this wrapper.
    """

    def __init__(self, cfgs, tokenizer) -> None:
        super().__init__()

        self.cfgs = cfgs
        self.tokenizer = tokenizer

        model_cfg = cfgs["model"]
        dataset_cfg = cfgs["dataset"]

        qbound_cfg = model_cfg.get("qbound", {})
        if not isinstance(qbound_cfg, dict):
            raise ValueError("model.qbound must be a mapping/dict.")

        self.baseline = model_cfg["baseline"]
        self.qmax_words = dataset_cfg["qmax_words"]
        self.max_feats = dataset_cfg["max_feats"]
        self.n_negs = model_cfg["n_negs"]

        self.qbound_enabled = bool(
            qbound_cfg.get(
                "enabled",
                model_cfg.get(
                    "qbound_enabled",
                    model_cfg.get("gpeb_enabled", False),
                ),
            )
        )
        self.qbound_version = str(
            qbound_cfg.get(
                "version",
                model_cfg.get(
                    "qbound_version",
                    model_cfg.get("gpeb_version", "1.1.3"),
                ),
            )
        )
        self.qbound_require_question_only = bool(
            qbound_cfg.get(
                "require_question_only",
                model_cfg.get(
                    "qbound_require_question_only",
                    model_cfg.get(
                        "gpeb_require_question_only",
                        True,
                    ),
                ),
            )
        )

        # Temporary compatibility aliases for old checkpoints/logging only.
        self.gpeb_enabled = self.qbound_enabled
        self.gpeb_version = self.qbound_version
        self.gpeb_require_question_only = (
            self.qbound_require_question_only
        )

        d_model = model_cfg["d_model"]

        self.lang_model = LanModel(
            tokenizer,
            lan=model_cfg["lan"],
            out_dim=d_model,
        )
        self.video_model = VideoEmbedding(cfgs)
        self.do = LinguisticInterventionGraph(cfgs)

        if self.baseline == "baseline":
            self.decoder = BaselineDecoder()
            self.forward = self.baseline_forward
        elif self.baseline == "refine":
            self.decoder = RefineDecoder(
                num_layer=model_cfg["num_layers"],
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
        if candidate_features.ndim != 3:
            raise ValueError(
                "candidate_features must be [B, C, D], "
                f"got {tuple(candidate_features.shape)}."
            )

        if candidate_indices.ndim != 1:
            raise ValueError(
                "candidate_indices must be [B], "
                f"got {tuple(candidate_indices.shape)}."
            )

        batch_indices = torch.arange(
            candidate_features.size(0),
            device=candidate_features.device,
        )

        return candidate_features[
            batch_indices,
            candidate_indices,
        ]

    def _encode_answers(
        self,
        answer,
        device: torch.device,
    ):
        """
        Encode candidate question-answer sequences.

        Expected input:
            answer: [B, C, L]

        Returns:
            answer_global: [B, C, D]
            answer_tokens: [B, C, L, D]
        """
        if answer is None:
            raise ValueError("answer tokens must not be None.")

        answer = answer.to(
            device=device,
            non_blocking=True,
        )

        if answer.ndim != 3:
            raise ValueError(
                "Candidate answer tokens must have shape [B, C, L], "
                f"got {tuple(answer.shape)}."
            )

        if hasattr(self.lang_model, "encode_candidates"):
            answer_global, answer_tokens = (
                self.lang_model.encode_candidates(answer)
            )
        else:
            # Compatibility fallback for the original LanModel.
            answer_global, answer_tokens = self.lang_model(answer)

        if answer_global.ndim != 3:
            raise ValueError(
                "LanModel must return candidate features [B, C, D], "
                f"got {tuple(answer_global.shape)}."
            )

        return answer_global, answer_tokens

    def _encode_question(
        self,
        question,
        device: torch.device,
    ):
        """
        Encode question-only tokens for Q-Bound.

        Accepted input:
            [B, L]
            [B, 1, L]  (the singleton candidate dimension is removed)

        Returns:
            question_global: [B, D]
            question_tokens: [B, L, D]
        """
        if question is None:
            return None, None

        question = question.to(
            device=device,
            non_blocking=True,
        )

        if question.ndim == 3 and question.size(1) == 1:
            question = question.squeeze(1)

        if question.ndim != 2:
            raise ValueError(
                "Question-only tokens must have shape [B, L] "
                "or [B, 1, L], "
                f"got {tuple(question.shape)}."
            )

        if hasattr(self.lang_model, "encode_question"):
            question_global, question_tokens = (
                self.lang_model.encode_question(question)
            )
        else:
            # Compatibility fallback for the original LanModel.
            question_global, question_tokens = self.lang_model(question)

        if question_global.ndim != 2:
            raise ValueError(
                "LanModel must return question features [B, D], "
                f"got {tuple(question_global.shape)}."
            )

        return question_global, question_tokens

    @staticmethod
    def _move_video_mask(
        video_mask,
        device: torch.device,
    ):
        if video_mask is None:
            return None

        return video_mask.to(
            device=device,
            non_blocking=True,
        )

    def refine_forward(
        self,
        video,
        video_mask,
        answer=None,
        answer_id=None,
        mode="VQA",
        gsub=None,
        question=None,
        qbound_mix=1.0,
    ):
        """
        Refined CRA/Q-Bound V1.1.3 forward.

        ``question`` is required by default whenever Q-Bound is active,
        because topology induction must remain question-only.

        ``qbound_mix`` is a deterministic, non-learned handover value in
        [0,1].  It is consumed by RefineDecoder:

            final_key =
                (1 - qbound_mix) * stopgrad(cra_key)
                + qbound_mix * qbound_key

        The training pipeline schedules this value. Validation/test should
        explicitly use qbound_mix=1.0.
        """
        mode = mode.upper()

        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask,
            device,
        )

        answer_g, _ = self._encode_answers(
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
            # Preserve the original VQ behavior.
            qsn_g = self._select_candidate_features(
                answer_g,
                answer_indices,
            )
            qsn_gt = qsn_g
        else:
            # Preserve the original CRA candidate-conditioned context.
            qsn_g = answer_g.max(dim=1).values

            if answer_indices is not None:
                qsn_gt = self._select_candidate_features(
                    answer_g,
                    answer_indices,
                )

        use_qbound = (
            self.qbound_enabled
            and mode == "VQA"
        )

        question_g = None
        question_tokens = None

        if use_qbound:
            if question is None:
                if self.qbound_require_question_only:
                    raise ValueError(
                        "Q-Bound V1.1.3 requires question-only token ids. "
                        "Pass them through CRA.forward(..., question=...). "
                        "Do not silently reuse candidate answer features, "
                        "because that would leak answer-choice semantics "
                        "into boundary learning."
                    )
            else:
                question_g, question_tokens = (
                    self._encode_question(
                        question,
                        device,
                    )
                )

                if question_g.size(0) != batch_size:
                    raise ValueError(
                        "Question and answer batch sizes do not match: "
                        f"{question_g.size(0)} vs {batch_size}."
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
            use_qbound=use_qbound,
            question_feature=question_g,
            # Fixed curriculum value supplied by CRAPipeline.
            # RefineDecoder detaches/validates it and performs the actual
            # probability-space CRA -> Q-Bound handover.
            qbound_mix=(
                qbound_mix
                if use_qbound
                else 0.0
            ),
        )

        # Candidate embeddings are still scored by the repository's
        # original main dot-product rule. Q-Bound changes temporal grounding,
        # not the answer-scoring head.
        answer_proj = answer_g

        outs = {
            "fatt": video_attn,
            "time_param": time_param,
            "align_gt": qsn_gt,
            "qa_condition": qsn_g,
            "question_condition": question_g,
            "question_tokens": question_tokens,
            "question_only_conditioning": torch.as_tensor(
                question_g is not None,
                device=video_proj.device,
                dtype=video_proj.dtype,
            ),
            "qbound_active": use_qbound,
            "qbound_mix": (
                time_param.get("qbound_mix")
                if isinstance(time_param, dict)
                else None
            ),
            # Temporary compatibility alias for old logging/checkpoints.
            "gpeb_active": use_qbound,
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
        question=None,
        qbound_mix=1.0,
    ):
        # Keep a common call signature with refine_forward.
        del gsub, question, qbound_mix

        mode = mode.upper()
        if mode not in {"VQ", "VQA"}:
            raise ValueError(
                f"Unsupported CRA mode {mode!r}; expected 'VQ' or 'VQA'."
            )

        device = video.device
        video_mask = self._move_video_mask(
            video_mask,
            device,
        )

        answer_g, answer_tokens = self._encode_answers(
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
            answer_tokens,
            video_mask,
        )

        outs = {
            "fatt": video_attn,
            "qa_condition": qsn_g,
        }

        return video_proj, answer_g, outs

    def align_loss(
        self,
        qsn,
        grounding_feature,
        neg_sample=None,
    ):
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

        return F.cross_entropy(
            logits,
            labels,
        )