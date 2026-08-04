import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_tempclip.transformer import LayerNorm
from modules.m_tempclip.language_model import Bert


class LanModel(nn.Module):
    """
    Language embedding module.

    Version 1.0 keeps the original pooling behavior for compatibility,
    while exposing explicit question-only and candidate encoding methods.

    Supported inputs:
        question token ids:
            [B, L]

        multiple-choice candidate token ids:
            [B, N, L]

    Returned representations:
        question:
            pooled: [B, D]
            tokens: [B, L, D]

        candidates:
            pooled: [B, N, D]
            tokens: [B, N, L, D]
    """

    def __init__(
        self,
        tokenizer,
        lan,
        word_dim=768,
        out_dim=512,
        debug_device=False,
    ):
        super().__init__()

        self.bert = Bert(tokenizer, lan)
        self.linear_text = nn.Linear(word_dim, out_dim)

        # Keep the original learned word-attention pooling so that
        # the CRA baseline behavior is not changed by Ver1.0.
        self.word_attn = nn.Sequential(
            nn.Linear(out_dim, 1),
            nn.Softmax(dim=-2),
        )

        self.out_dim = out_dim
        self.debug_device = bool(debug_device)

    def _check_device(self, x):
        """
        Optional device diagnostic.

        The old implementation iterated over every BERT parameter and
        printed on every forward pass. That is unnecessarily expensive
        during training, so the check is now disabled by default.
        """
        if not self.debug_device:
            return

        mismatches = []
        for name, param in self.bert.named_parameters():
            if param.device != x.device:
                mismatches.append(
                    f"{name}: parameter={param.device}, input={x.device}"
                )

        if mismatches:
            raise RuntimeError(
                "Language encoder device mismatch:\n"
                + "\n".join(mismatches[:10])
            )

    def _encode_2d(self, token_ids):
        """
        Encode a batch of token sequences.

        Args:
            token_ids: [B, L]

        Returns:
            pooled: [B, D]
            token_features: [B, L, D]
        """
        if token_ids.ndim != 2:
            raise ValueError(
                "_encode_2d expects token ids with shape [B, L], "
                f"got {tuple(token_ids.shape)}."
            )

        self._check_device(token_ids)

        token_features = self.bert(token_ids)
        token_features = self.linear_text(token_features)

        word_weights = self.word_attn(token_features)
        pooled = torch.sum(
            token_features * word_weights,
            dim=1,
        )

        return pooled, token_features

    def encode_question(self, question):
        """
        Encode question-only tokens.

        Args:
            question: [B, L]

        Returns:
            question_global: [B, D]
            question_tokens: [B, L, D]
        """
        if question.ndim != 2:
            raise ValueError(
                "encode_question expects [B, L], "
                f"got {tuple(question.shape)}."
            )

        return self._encode_2d(question)

    def encode_candidates(self, answers):
        """
        Encode multiple-choice candidate sequences.

        Args:
            answers: [B, N, L]

        Returns:
            answer_global: [B, N, D]
            answer_tokens: [B, N, L, D]
        """
        if answers.ndim != 3:
            raise ValueError(
                "encode_candidates expects [B, N, L], "
                f"got {tuple(answers.shape)}."
            )

        batch_size, num_answers, seq_len = answers.shape

        flat_answers = answers.reshape(
            batch_size * num_answers,
            seq_len,
        )

        flat_global, flat_tokens = self._encode_2d(
            flat_answers
        )

        answer_global = flat_global.reshape(
            batch_size,
            num_answers,
            self.out_dim,
        )

        answer_tokens = flat_tokens.reshape(
            batch_size,
            num_answers,
            seq_len,
            self.out_dim,
        )

        return answer_global, answer_tokens

    def forward(self, text):
        """
        Backward-compatible dispatch.

        [B, L]    -> question-style encoding
        [B, N, L] -> multiple-choice candidate encoding
        """
        if text.ndim == 2:
            return self.encode_question(text)

        if text.ndim == 3:
            return self.encode_candidates(text)

        raise ValueError(
            "LanModel expects [B, L] or [B, N, L], "
            f"got {tuple(text.shape)}."
        )


def clones(module, num_copies):
    return nn.ModuleList(
        [copy.deepcopy(module) for _ in range(num_copies)]
    )


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_param=None):
        del time_param
        return self.w_2(
            self.dropout(
                F.relu(self.w_1(x))
            )
        )


class SublayerConnection(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(
            sublayer(self.norm(x))
        )


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()

        if d_model % h != 0:
            raise ValueError(
                "d_model must be divisible by the number of heads."
            )

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(
            nn.Linear(d_model, d_model),
            4,
        )
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.attention = self.ori_attention

        # Kept for checkpoint and module compatibility.
        self.selector = nn.Linear(self.d_k, 2)
        self.time_reparam = nn.Linear(2, self.d_k)

    def ori_attention(
        self,
        query,
        key,
        value,
        mask=None,
        dropout=None,
    ):
        d_k = query.size(-1)

        scores = torch.matmul(
            query,
            key.transpose(-2, -1),
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0,
                -1e9,
            )

        attention = F.softmax(
            scores,
            dim=-1,
        )

        if dropout is not None:
            attention = dropout(attention)

        return (
            torch.matmul(attention, value),
            attention,
        )

    def forward(
        self,
        query,
        key,
        value,
        mask=None,
    ):
        batch_size = query.size(0)

        query, key, value = [
            linear(tensor)
            .reshape(
                batch_size,
                -1,
                self.h,
                self.d_k,
            )
            .transpose(1, 2)
            for linear, tensor in zip(
                self.linears,
                (query, key, value),
            )
        ]

        x, self.attn = self.attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
        )

        x = (
            x.transpose(1, 2)
            .contiguous()
            .reshape(
                batch_size,
                -1,
                self.h * self.d_k,
            )
        )

        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim,
        dropout,
    ):
        super().__init__()

        self.self_attn = MultiHeadedAttention(
            num_heads,
            embed_dim,
            dropout,
        )
        self.feed_forward = PositionwiseFeedForward(
            embed_dim,
            ff_dim,
            dropout,
        )
        self.sublayer_self = SublayerConnection(
            embed_dim,
            dropout,
        )
        self.sublayer_ff = SublayerConnection(
            embed_dim,
            dropout,
        )

    def forward(self, x):
        x = self.sublayer_self(
            x,
            lambda value: self.self_attn(
                value,
                value,
                value,
            ),
        )

        x = self.sublayer_ff(
            x,
            lambda value: self.feed_forward(value),
        )

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_layer=2,
        num_heads=8,
        ff_dim=768,
        dropout=0.1,
    ):
        super().__init__()

        self.layers = clones(
            DecoderLayer(
                embed_dim,
                num_heads,
                ff_dim,
                dropout,
            ),
            num_layer,
        )
        self.norm = LayerNorm(embed_dim)

        self.attention_video_feature = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=-2),
        )

        self.video_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        x,
        qa,
        self_mask=None,
    ):
        # Keep the original baseline behavior. This decoder does not use
        # qa or self_mask directly; the arguments remain for compatibility.
        del qa, self_mask

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        video_attn = self.attention_video_feature(x)

        x = torch.sum(
            x * video_attn,
            dim=1,
        )

        x = self.video_proj(x)

        return x, video_attn
