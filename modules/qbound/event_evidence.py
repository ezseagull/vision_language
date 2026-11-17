"""Q-Bound V3.8-Minimal: joint answer x event evidence model.

The same joint P(event,answer) supplies evidence-only QA and temporal
posteriors; no answer labels or temporal ground-truth may enter forward().
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointAnswerEventEvidence(nn.Module):
    def __init__(self, event_dim: int, answer_dim: int, *,
                 centering: str = "double", prior_temperature: float = 1.0,
                 interaction_temperature: float = 1.0, dropout: float = 0.1) -> None:
        super().__init__()
        if event_dim < 1 or answer_dim < 1:
            raise ValueError("Event/answer dims must be positive.")
        if centering not in ("raw", "row", "double"):
            raise ValueError("Centering must be 'raw', 'row' or 'double'.")
        if prior_temperature <= 0 or interaction_temperature <= 0:
            raise ValueError("Temperatures must be > 0.")
        self.centering = centering
        self.prior_temperature = float(prior_temperature)
        self.interaction_temperature = float(interaction_temperature)
        self.event_dim = int(event_dim)
        self.question_prior = nn.Sequential(
            nn.Linear(4 * event_dim + 2, event_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(event_dim, 1),
        )
        self.event_proj = nn.Linear(event_dim, event_dim)
        self.answer_proj = nn.Linear(answer_dim, event_dim)
        # Neutral event prior at initialization; interaction retains learnable variation.
        nn.init.normal_(self.question_prior[-1].weight, std=0.01)
        nn.init.zeros_(self.question_prior[-1].bias)

    def forward(self, bank: Dict[str, torch.Tensor], question_embedding: torch.Tensor,
                answer_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        event = bank["event_states"]  # [B,M,d]
        valid = bank["event_valid_mask"]
        footprints = bank["event_footprint"]
        prior = bank["bank_prior_weights"]
        position = bank["event_position_features"]
        batch, n_events, dim = event.shape
        if question_embedding.shape != (batch, dim):
            raise ValueError("question_embedding must be [B,event_dim].")
        if answer_embeddings.ndim != 3 or answer_embeddings.shape[:2] != (batch, 5):
            raise ValueError("answer_embeddings must be [B,5,answer_dim].")
        if answer_embeddings.size(-1) != self.answer_proj.in_features:
            raise ValueError("Answer feature dimension mismatch.")
        if valid.shape != (batch, n_events) or prior.shape != valid.shape:
            raise ValueError("Event masks/prior have wrong shape.")
        if footprints.shape[:2] != valid.shape:
            raise ValueError("Event footprint shape mismatch.")
        if not bool(valid.any(-1).all()):
            raise ValueError("Missing event/fallback for a sample.")
        if not bool(torch.isfinite(event).all()) or not bool(torch.isfinite(answer_embeddings).all()):
            raise ValueError("Nonfinite input to event evidence.")
        if not bool(torch.allclose(prior.sum(-1), torch.ones_like(prior.sum(-1)), atol=1e-5)):
            raise ValueError("Bank prior must be normalized.")
        q = question_embedding[:, None, :].expand_as(event)
        prior_input = torch.cat((event, q, event * q, torch.abs(event - q), position), dim=-1)
        u = self.question_prior(prior_input).squeeze(-1).float()
        # Geometry prior is constant and does not contain any GT timestamps.
        g = u / self.prior_temperature + prior.float().clamp_min(1e-30).log()
        g = g.masked_fill(~valid, -torch.inf)
        p_question = torch.softmax(g, dim=1)

        interaction = torch.einsum("bmd,bad->bma", self.event_proj(event),
                                   self.answer_proj(answer_embeddings)) / math.sqrt(dim)
        interaction = interaction.float()
        rows = interaction.mean(dim=-1, keepdim=True)
        mask_f = valid.unsqueeze(-1).to(interaction.dtype)
        column_mean = (interaction * mask_f).sum(dim=1, keepdim=True) / mask_f.sum(dim=1, keepdim=True).clamp_min(1)
        global_mean = (column_mean.mean(dim=-1, keepdim=True))
        if self.centering == "raw":
            centered = interaction
        elif self.centering == "row":
            centered = interaction - rows
        else:
            centered = interaction - rows - column_mean + global_mean
        centered = centered.masked_fill(~valid.unsqueeze(-1), 0.0)
        joint_score = (g.unsqueeze(-1) + centered / self.interaction_temperature)
        joint_log_norm = torch.logsumexp(joint_score.flatten(1), dim=-1)
        log_joint = joint_score - joint_log_norm[:, None, None]
        joint = torch.exp(log_joint).masked_fill(~valid.unsqueeze(-1), 0.0)
        # Marginal answer logits are derived from THE SAME J used by WHERE.
        evidence_logits = torch.logsumexp(joint_score, dim=1)
        answer_posterior = torch.softmax(evidence_logits, dim=-1)
        marginal_event = joint.sum(dim=-1)
        # P(i|a), then P(i|a_hat). a_hat is a MODEL prediction, never GT.
        # Softmax over EVENTS for each answer is the stable equivalent of
        # P(i,a) / P(a). Unlike division it remains normalized when P(a) is
        # extremely small (e.g. an uncompetitive candidate under AMP).
        conditional_all = torch.softmax(joint_score, dim=1)
        conditional_all = conditional_all.masked_fill(~valid.unsqueeze(-1), 0.0)
        predicted_answer = evidence_logits.argmax(dim=-1)
        selected_conditional = conditional_all.gather(
            2, predicted_answer.view(batch, 1, 1).expand(batch, n_events, 1)
        ).squeeze(-1)
        selected_conditional = selected_conditional / selected_conditional.sum(dim=-1, keepdim=True).clamp_min(1e-30)
        temporal_question = torch.bmm(p_question[:, None, :], footprints.float()).squeeze(1)
        temporal_marginal = torch.bmm(marginal_event[:, None, :], footprints.float()).squeeze(1)
        temporal_conditional = torch.bmm(selected_conditional[:, None, :], footprints.float()).squeeze(1)
        temporal_all_conditionals = torch.einsum("bma,bmt->bat", conditional_all, footprints.float())
        # No new learned WHERE head: all three time distributions are push-forwards.
        # The no-evidence single-frame fallback contributes no evidence CE (mask below).
        train_eligible = bank["bank_train_eligible"]
        return {
            "event_question_prior_logits": u,
            "event_g": g,
            "event_interaction_raw": interaction,
            "event_interaction_centered": centered,
            "event_joint_scores": joint_score,
            "event_joint_posterior": joint,
            "event_qa_logits": evidence_logits,
            "event_qa_posterior": answer_posterior,
            "event_predicted_answer_index": predicted_answer,
            "event_posterior_question": p_question,
            "event_posterior_marginal": marginal_event,
            "event_posterior_conditional_pred": selected_conditional,
            "event_posterior_conditional_all": conditional_all,
            "event_temporal_question": temporal_question,
            "event_temporal_marginal": temporal_marginal,
            "event_temporal_conditional_pred": temporal_conditional,
            "event_temporal_conditional_all": temporal_all_conditionals,
            "event_train_eligible": train_eligible,
            "event_posterior_consistency_max_abs": (
                marginal_event - (conditional_all * answer_posterior[:, None, :]).sum(-1)
            ).abs().max(-1).values,
            # Predicted-answer contrast, NOT the label-conditioned CE gradient.
            # The pipeline must compute the true H2/H3.8 gradient using y.
            "event_pred_answer_prior_contrast": (
                marginal_event - selected_conditional
            ),
        }


class FactorizedAnswerEventMIL(nn.Module):
    """V3.8-M1. Independent frame-text cosine scores, then fixed window reads.

    The visual projection operates on each step independently. The text
    projection is shared between question-only and [Question SEP Answer]
    inputs. Both input streams are detached, but the new projections train.
    The scorer only reads bank geometry, not M0's context states or learned
    positional features. It never receives answer labels, GT or timestamps.
    """

    def __init__(self, *, visual_dim: int, text_dim: int, projection_dim: int,
                 gamma: float = 0.25, window_temperature: float = 0.1,
                 candidate_temperature: float = 0.1) -> None:
        super().__init__()
        if min(visual_dim, text_dim, projection_dim) < 1:
            raise ValueError("M1 projection dimensions must be positive.")
        if not 0.0 <= gamma <= 0.5:
            raise ValueError("M1 gamma must be in [0, 0.5].")
        for name, value in (("window_temperature", window_temperature),
                            ("candidate_temperature", candidate_temperature)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"M1 {name} must be positive and finite.")
        self.visual_dim = int(visual_dim)
        self.text_dim = int(text_dim)
        self.projection_dim = int(projection_dim)
        self.gamma = float(gamma)
        self.window_temperature = float(window_temperature)
        self.candidate_temperature = float(candidate_temperature)
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, projection_dim), nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.text_proj = nn.Linear(text_dim, projection_dim)

    def _read_scores(self, field, inner, ring, width_scale, has_ring):
        score_in = torch.bmm(inner, field)
        score_out = torch.bmm(ring, field)
        contrast = (score_in - score_out) * width_scale.unsqueeze(-1)
        # No comparison is possible for a one-step fallback with no exterior.
        return score_in, contrast.masked_fill(~has_ring.unsqueeze(-1), 0.0)

    def _where(self, contrast, log_prior, valid):
        logits = log_prior.unsqueeze(-1) + contrast / self.window_temperature
        return F.log_softmax(logits.masked_fill(~valid.unsqueeze(-1), -torch.inf), dim=1)

    def forward(self, *, bank: Dict[str, torch.Tensor], frame_features: torch.Tensor,
                question_embeddings: torch.Tensor, answer_embeddings: torch.Tensor,
                question_candidate_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if frame_features.ndim != 3 or frame_features.size(-1) != self.visual_dim:
            raise ValueError("M1 frame_features must be [B,T,visual_dim].")
        batch, time_steps, _ = frame_features.shape
        if question_embeddings.shape != (batch, self.text_dim):
            raise ValueError("M1 question_embeddings must be [B,text_dim].")
        if answer_embeddings.shape != (batch, 5, self.text_dim):
            raise ValueError("M1 answer_embeddings must be [B,5,text_dim].")

        video_mask = bank["bank_video_mask"]
        valid = bank["event_valid_mask"]
        prior = bank["bank_prior_weights"]
        widths = bank["event_width"]
        inner = bank["event_footprint"]
        ring = bank["event_ring_footprint"]
        ring_counts = bank["event_ring_valid_steps"]
        if video_mask.dtype != torch.bool or video_mask.shape != (batch, time_steps):
            raise ValueError("M1 video mask must be boolean [B,T].")
        if valid.ndim != 2 or valid.dtype != torch.bool or valid.size(0) != batch:
            raise ValueError("M1 event mask must be boolean [B,M].")
        if not bool(valid.any(dim=1).all()):
            raise ValueError("M1 requires a valid window or fallback in every row.")
        if any(t.shape != valid.shape for t in (prior, widths, ring_counts)):
            raise ValueError("M1 prior, width and outer-ring count must match [B,M].")
        if inner.shape != (*valid.shape, time_steps) or ring.shape != inner.shape:
            raise ValueError("M1 inner and outer footprints must match [B,M,T].")
        if question_candidate_mask is not None:
            if (question_candidate_mask.dtype != torch.bool
                    or question_candidate_mask.shape != (batch, batch)):
                raise ValueError("M1 question candidate mask must be boolean [B,B].")
            if not bool(question_candidate_mask.any(dim=1).all()):
                raise ValueError("M1 needs a safe candidate softmax for every scored row.")

        # Float32 scoring/reductions prevent AMP from discarding small window
        # or class probabilities. Disable autocast for the fixed reads as well.
        with torch.autocast(device_type=frame_features.device.type, enabled=False):
            v = frame_features.detach().to(dtype=self.visual_proj[0].weight.dtype)
            v = v.masked_fill(~video_mask.unsqueeze(-1), 0.0)
            v = F.normalize(self.visual_proj(v).float(), dim=-1, eps=1e-8)
            v = v.masked_fill(~video_mask.unsqueeze(-1), 0.0)
            q = F.normalize(self.text_proj(question_embeddings.detach().to(self.text_proj.weight.dtype)).float(),
                            dim=-1, eps=1e-8)
            a = F.normalize(self.text_proj(answer_embeddings.detach().to(self.text_proj.weight.dtype)).float(),
                            dim=-1, eps=1e-8)
            inner, ring = inner.float(), ring.float()
            has_ring = ring_counts.gt(0)
            width_scale = (widths.float().clamp_min(1) / 2).pow(self.gamma)
            log_prior = prior.float().clamp_min(1e-30).log()

            q_field = (v * q.unsqueeze(1)).sum(dim=-1)
            q_in, q_cs = self._read_scores(q_field.unsqueeze(-1), inner, ring, width_scale, has_ring)
            q_log_alpha = self._where(q_cs, log_prior, valid).squeeze(-1)

            a_field = torch.bmm(v, a.transpose(1, 2))
            a_in, a_cs = self._read_scores(a_field, inner, ring, width_scale, has_ring)
            a_log_alpha = self._where(a_cs, log_prior, valid)
            a_logits = a_in / self.candidate_temperature
            a_denom = torch.logsumexp(a_logits, dim=-1, keepdim=True)
            a_log_beta = a_logits - a_denom
            a_log_scores = torch.logsumexp(a_log_alpha + a_log_beta, dim=1)

            # 1-s_a = sum_i alpha_ia * (1-beta_ia). Compute this in log
            # space from the other four logits, never by subtracting a
            # saturated score from 1. The pipeline applies the label loss.
            eye5 = torch.eye(5, dtype=torch.bool, device=v.device)
            other_logits = a_logits.unsqueeze(-2).expand(-1, -1, 5, -1).masked_fill(eye5, -torch.inf)
            a_log_not_beta = torch.logsumexp(other_logits, dim=-1) - a_denom
            a_log_not_scores = torch.logsumexp(a_log_alpha + a_log_not_beta, dim=1)

            out = {
                "event_posterior_question": q_log_alpha.exp(),
                "event_posterior_by_answer": a_log_alpha.exp(),
                "event_mil_answer_scores": a_log_scores.exp(),
                "event_mil_answer_log_scores": a_log_scores,
                "event_mil_answer_log_not_scores": a_log_not_scores,
                "event_question_in_scores": q_in.squeeze(-1),
                "event_question_contrast_scores": q_cs.squeeze(-1),
                "event_answer_in_scores": a_in,
                "event_answer_contrast_scores": a_cs,
                "event_answer_candidate_posterior": a_log_beta.exp().masked_fill(~valid.unsqueeze(-1), 0),
            }
            if question_candidate_mask is not None:
                # The same [B,B] candidate set per video is supplied by CRA's
                # caller. Each row's WHERE uses its own question. Its WHAT
                # compares that question with all retained batch questions.
                q_fields = v @ q.transpose(0, 1)  # [B,T,B]
                # Use the identical self-score as Q-only for the diagonal,
                # avoiding a separate accumulation order for that candidate.
                row_ids = torch.arange(batch, device=v.device)
                q_in_pool = torch.bmm(inner, q_fields)
                q_in_pool = torch.where(
                    torch.eye(batch, dtype=torch.bool, device=v.device)[:, None, :],
                    q_in, q_in_pool,
                )
                q_logits = q_in_pool / self.candidate_temperature
                q_log_beta = F.log_softmax(
                    q_logits.masked_fill(~question_candidate_mask[:, None, :], -torch.inf), dim=-1,
                )
                q_self = q_log_beta[row_ids, :, row_ids]
                # A dropped diagonal (duplicate/descriptive) has no loss.
                # Use a neutral finite score here; CRA exports the true
                # eligibility, including the original empty-pool rows.
                q_self = q_self.masked_fill(~question_candidate_mask.diagonal().unsqueeze(-1), 0.0)
                q_log_scores = torch.logsumexp(q_log_alpha + q_self, dim=1)
                out["event_mil_question_log_scores"] = q_log_scores
                out["event_mil_question_scores"] = q_log_scores.exp()
                out["event_question_candidate_posterior"] = q_log_beta.exp().masked_fill(
                    ~valid.unsqueeze(-1), 0.0,
                )
            return out
