import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.m_cra.only_trans import (
    clones,
    PositionwiseFeedForward,
    LayerNorm,
    SublayerConnection,
    MultiHeadedAttention,
)
from modules.m_cra.grounding import GroundingModule
from modules.m_cra.causal_module import VisualIntervention
from modules.m_cra.event_bridge import GroundedPredictiveEventBridge


class RefineMHA(MultiHeadedAttention):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__(h, d_model, dropout)
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = self.refined_attention

        self.attn = None
        self.refined_mask = None

    def ori_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def refined_attention(self, query, key, value, mask=None, dropout=None):
        bs, n_head, n_query, d_k = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)

        if mask is not None:
            mask = (
                mask.unsqueeze(1)
                .unsqueeze(1)
                .expand(-1, n_head, n_query, -1)
                .reshape(*attn.shape)
            )
            attn = attn * (mask + 1e-10)
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        qkv = torch.matmul(attn, value)
        return qkv

    def causal_attention(self, query, key, value, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        neg_scores = torch.matmul(query, -key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            neg_scores = neg_scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        neg_p_attn = F.softmax(neg_scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return (
            torch.matmul(p_attn, value) + torch.matmul(neg_p_attn, value),
            (p_attn, neg_p_attn),
        )

    def forward(self, query, key, value, mask=None, alpha=(0.125, 0.5)):
        nbatches = query.size(0)
        query, key, value = [
            linear(tensor)
            .view(nbatches, -1, self.h, self.d_k)
            .transpose(1, 2)
            for linear, tensor in zip(self.linears, (query, key, value))
        ]

        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        return self.linears[-1](x)


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = RefineMHA(num_heads, embed_dim, dropout)
        self.sublayer_ff = SublayerConnection(embed_dim, dropout)
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        self.sublayer_self = SublayerConnection(embed_dim, dropout)

    def forward(self, x, mask=None, wm_mode=False):
        x = self.sublayer_self(x, lambda value: self.self_attn(value, value, value, mask))
        x = self.sublayer_ff(x, lambda value: self.feed_forward(value))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_layer=2,
        num_heads=8,
        ff_dim=768,
        dropout=0.1,
        cfgs=None,
    ):
        super().__init__()

        if cfgs is None:
            raise ValueError("cfgs must not be None.")

        self.embed_dim = embed_dim
        self.cfgs = cfgs
        self.layers = clones(
            DecoderLayer(embed_dim, num_heads, ff_dim, dropout),
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

        model_cfg = self.cfgs["model"]

        self.grounding_module = GroundingModule(
            d_model=embed_dim,
            dropout=model_cfg["dropout"],
            initial_sigma=model_cfg["sigma"],
            window_sizes=model_cfg["window_sizes"],
        )
        self.do = VisualIntervention(self.cfgs)

        # -------------------------------------------------------------
        # GPEB Version 0.5
        # -------------------------------------------------------------
        self.gpeb_enabled = model_cfg.get("gpeb_enabled", False)

        if self.gpeb_enabled:
            self.gpeb = GroundedPredictiveEventBridge(
                input_dim=embed_dim,
                hidden_dim=model_cfg.get("gpeb_hidden_dim", 256),
                num_bins=model_cfg.get("gpeb_num_bins", 4),
                use_grounding_prior=model_cfg.get(
                    "gpeb_use_grounding_prior",
                    True,
                ),
                dropout=model_cfg.get("gpeb_dropout", dropout),
                grounding_prior_clip=model_cfg.get(
                    "gpeb_grounding_prior_clip",
                    4.0,
                ),
                grounding_scale_max=model_cfg.get(
                    "gpeb_grounding_scale_max",
                    5.0,
                ),
            )

            gate_hidden_dim = int(
                model_cfg.get(
                    "gpeb_gate_hidden_dim",
                    max(embed_dim // 2, 1),
                )
            )
            gate_dropout = float(
                model_cfg.get("gpeb_gate_dropout", dropout)
            )
            gate_bias_init = float(
                model_cfg.get("gpeb_gate_bias_init", -0.5)
            )

            if gate_hidden_dim < 1:
                raise ValueError("gpeb_gate_hidden_dim must be positive.")

            self.gpeb_cra_norm = nn.LayerNorm(embed_dim)
            self.gpeb_bridge_norm = nn.LayerNorm(embed_dim)
            self.gpeb_qa_proj = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )

            # Gate input:
            #   CRA feature
            #   GPEB feature
            #   QA summary
            #   CRA-GPEB agreement
            #   CRA-GPEB absolute difference
            # Shape: [B, 5 * D]
            self.gpeb_gate = nn.Sequential(
                nn.Linear(embed_dim * 5, gate_hidden_dim),
                nn.GELU(),
                nn.Dropout(gate_dropout),
                nn.Linear(gate_hidden_dim, embed_dim),
            )

            # Start near a moderate gate value: sigmoid(-0.5) ~= 0.378.
            # A tiny final weight keeps gates almost constant initially,
            # while still allowing sample-specific learning later.
            nn.init.normal_(
                self.gpeb_gate[-1].weight,
                mean=0.0,
                std=1e-3,
            )
            nn.init.constant_(
                self.gpeb_gate[-1].bias,
                gate_bias_init,
            )

            self.gpeb_residual_dropout = nn.Dropout(
                model_cfg.get("gpeb_residual_dropout", dropout)
            )

            # Ver0.5b: isolate GPEB auxiliary gradients from the shared
            # CRA grounding/temporal backbone, and explicitly bound how
            # much the GPEB residual can modify the CRA representation.
            self.gpeb_detach_backbone_inputs = bool(
                model_cfg.get("gpeb_detach_backbone_inputs", True)
            )
            self.gpeb_detach_gate_context = bool(
                model_cfg.get("gpeb_detach_gate_context", True)
            )
            self.gpeb_residual_max_ratio = float(
                model_cfg.get("gpeb_residual_max_ratio", 0.15)
            )
            if self.gpeb_residual_max_ratio <= 0.0:
                raise ValueError(
                    "gpeb_residual_max_ratio must be positive."
                )

    def _summarize_qa_for_gpeb(self, qa: torch.Tensor) -> torch.Tensor:
        """
        Return one QA vector per sample for the fusion gate.

        Current compatibility behavior matches event_bridge.py:
        all dimensions between batch and feature are averaged.
        After inspecting the outer CRA model, this should preferably be
        replaced by a pure question representation if one is available.
        """
        if qa.ndim < 2:
            raise ValueError("qa must have at least two dimensions.")

        if qa.size(-1) != self.embed_dim:
            raise ValueError(
                "The last qa dimension must equal embed_dim. "
                f"Expected {self.embed_dim}, got {qa.size(-1)}."
            )

        if qa.ndim == 2:
            return qa

        return qa.reshape(qa.size(0), -1, self.embed_dim).mean(dim=1)

    def forward(
        self,
        x,
        qa,
        self_mask=None,
        gsub=None,
        do=False,
        window_sizes=(1, 3, 5),
        use_gpeb=False,
    ):
        # Preserve raw timestep features. Decoder layers return new tensors,
        # so this reference is not overwritten in place.
        x_raw = x

        # Temporal refinement.
        for layer in self.layers:
            x = layer(x, self_mask)
        x_bar = self.norm(x)

        # CRA grounding.
        time_param = self.grounding_module(
            x_bar,
            qa,
            window_sizes=window_sizes,
        )

        recon_feature = None

        # Original CRA grounded video representation.
        video_proj = torch.sum(
            x_bar * time_param["key"].unsqueeze(dim=-1),
            dim=1,
        )
        video_proj = self.video_proj(video_proj)

        # Optional visual causal intervention remains unchanged.
        if do:
            x_star = torch.mean(x_bar, dim=1)
            x_star = self.video_proj(x_star)
            video_proj = self.do(x_star, video_proj)

        # Preserve the original CRA representation for grounding/alignment
        # losses. GPEB must not silently change the old CRA objectives.
        cra_video_proj = video_proj
        time_param["grounding_feature"] = cra_video_proj

        # -------------------------------------------------------------
        # GPEB Version 0.5 branch
        # -------------------------------------------------------------
        gpeb_aux = None

        if self.gpeb_enabled and use_gpeb:
            # Ver0.5b grounding-safe inputs. Detaching these tensors does
            # not freeze GPEB parameters; it only prevents GPEB-specific
            # losses from pulling the shared CRA temporal/grounding path
            # away from its original objectives.
            if self.gpeb_detach_backbone_inputs:
                raw_for_gpeb = x_raw.detach()
                contextual_for_gpeb = x_bar.detach()
                qa_for_gpeb = qa.detach()
                grounding_for_gpeb = time_param["key"].detach()
            else:
                raw_for_gpeb = x_raw
                contextual_for_gpeb = x_bar
                qa_for_gpeb = qa
                grounding_for_gpeb = time_param["key"]

            gpeb_aux = self.gpeb(
                raw_video=raw_for_gpeb,
                contextual_video=contextual_for_gpeb,
                qa_features=qa_for_gpeb,
                video_mask=self_mask,
                grounding_weights=grounding_for_gpeb,
            )

            bridge_repr = gpeb_aux["bridge_repr"]
            fusion_delta = gpeb_aux["fusion_delta"]

            qa_summary = self._summarize_qa_for_gpeb(qa)

            # The direct CRA path below still receives the main QA and
            # grounding gradients. Only the extra gate-conditioning path
            # is detached, avoiding a second unwanted gradient route into
            # CRA and the language encoder.
            cra_gate_source = (
                cra_video_proj.detach()
                if self.gpeb_detach_gate_context
                else cra_video_proj
            )
            qa_gate_source = (
                qa_summary.detach()
                if self.gpeb_detach_gate_context
                else qa_summary
            )

            cra_for_gate = self.gpeb_cra_norm(cra_gate_source)
            bridge_for_gate = self.gpeb_bridge_norm(bridge_repr)
            qa_for_gate = self.gpeb_qa_proj(qa_gate_source)

            gate_input = torch.cat(
                [
                    cra_for_gate,
                    bridge_for_gate,
                    qa_for_gate,
                    cra_for_gate * bridge_for_gate,
                    torch.abs(cra_for_gate - bridge_for_gate),
                ],
                dim=-1,
            )

            gate_logits = self.gpeb_gate(gate_input)
            gate = torch.sigmoid(gate_logits)

            # Candidate residual before safety clipping.
            unbounded_residual = gate * fusion_delta
            unbounded_residual = self.gpeb_residual_dropout(
                unbounded_residual
            )

            # Ver0.5b bounded residual fusion. The residual norm is capped
            # relative to each sample's CRA norm, preventing GPEB from
            # becoming a second full-strength representation as observed
            # in Ver0.5.
            eps = 1e-6
            cra_norm = cra_video_proj.norm(
                p=2,
                dim=-1,
                keepdim=True,
            ).clamp_min(eps)
            unbounded_norm = unbounded_residual.norm(
                p=2,
                dim=-1,
                keepdim=True,
            )
            residual_ratio_unclipped = (
                unbounded_norm / cra_norm
            ).squeeze(-1)

            max_residual_norm = (
                self.gpeb_residual_max_ratio * cra_norm
            )
            residual_clip_scale = torch.clamp(
                max_residual_norm
                / unbounded_norm.clamp_min(eps),
                max=1.0,
            )
            gated_residual = (
                unbounded_residual * residual_clip_scale
            )

            video_proj = cra_video_proj + gated_residual

            residual_ratio = (
                gated_residual.norm(p=2, dim=-1)
                / cra_video_proj.norm(p=2, dim=-1).clamp_min(eps)
            )
            branch_cosine = F.cosine_similarity(
                cra_video_proj,
                bridge_repr,
                dim=-1,
                eps=eps,
            )

            # Keep live tensors required by later losses, and scalar/vector
            # diagnostics for logging. Do not call .item() inside forward.
            gpeb_aux.update(
                {
                    "gate": gate,
                    "gate_logits": gate_logits,
                    "gated_residual": gated_residual,
                    "gate_mean": gate.mean(),
                    "gate_std": gate.std(unbiased=False),
                    "gate_min": gate.min(),
                    "gate_max": gate.max(),
                    "residual_ratio": residual_ratio,
                    "residual_ratio_mean": residual_ratio.mean(),
                    "residual_ratio_unclipped": residual_ratio_unclipped,
                    "residual_ratio_unclipped_mean": (
                        residual_ratio_unclipped.mean()
                    ),
                    "residual_clip_scale": residual_clip_scale.squeeze(-1),
                    "residual_clip_scale_mean": residual_clip_scale.mean(),
                    "residual_max_ratio": torch.as_tensor(
                        self.gpeb_residual_max_ratio,
                        device=video_proj.device,
                        dtype=video_proj.dtype,
                    ),
                    "branch_cosine_similarity": branch_cosine,
                    "branch_cosine_mean": branch_cosine.mean(),
                    "cra_video_proj": cra_video_proj,
                    "fused_video_proj": video_proj,
                }
            )
        else:
            video_proj = cra_video_proj

        time_param["gpeb_aux"] = gpeb_aux

        # Kept for backward compatibility with the current pipeline.
        # Despite the old key name, this is the final CRA/GPEB feature.
        time_param["gpeb_feature"] = video_proj

        return video_proj, time_param["key"], time_param
