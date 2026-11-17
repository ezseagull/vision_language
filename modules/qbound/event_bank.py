"""Q-Bound V3.8-Minimal: fixed, masked temporal hypothesis bank.

No timestamp labels, learned boundary, selection Top-K, or full-video proposal.
The bank is fixed in feature steps, not seconds. Each valid proposal has a
unit-mass TARGET footprint; surrounding context never becomes grounding mass.
"""
from typing import Dict, Sequence

import torch
import torch.nn as nn


class FixedTemporalEventBank(nn.Module):
    """Fixed width/half-stride windows and a small residual context reader.

    For T=32, widths (2,4,8,16) create 31+15+7+3=56 proposals.
    All windows are valid only within each example's LEFT-aligned video mask.
    For T=1 or T=2, frame-level fallback windows are provided for safe
    inference; event QA supervision MUST exclude these rows (train_eligible).
    """

    def __init__(
        self,
        hidden_dim: int,
        widths: Sequence[int] = (2, 4, 8, 16),
        context_radius: int = 1,
        prior_mode: str = "scale_balanced",
        include_tail: bool = True,
    ) -> None:
        super().__init__()
        widths = tuple(int(w) for w in widths)
        if hidden_dim < 1 or not widths or len(set(widths)) != len(widths):
            raise ValueError("hidden_dim>0 and distinct nonempty widths are required.")
        if any(w < 2 for w in widths) or tuple(sorted(widths)) != widths:
            raise ValueError("Bank widths must be increasing integers >= 2.")
        if context_radius not in (0, 1, 2):
            raise ValueError("M0 context_radius must be pre-registered as 0, 1, or 2.")
        if prior_mode not in ("scale_balanced", "uniform_window"):
            raise ValueError("Unknown prior_mode: use scale_balanced or uniform_window.")
        if not isinstance(include_tail, bool):
            raise TypeError("include_tail must be bool.")
        self.hidden_dim = hidden_dim
        self.widths = widths
        self.context_radius = context_radius
        self.prior_mode = prior_mode
        self.include_tail = include_tail
        self.context_projection = nn.Linear(2 * hidden_dim + 2, hidden_dim)
        self.event_norm = nn.LayerNorm(hidden_dim)
        nn.init.zeros_(self.context_projection.weight)
        nn.init.zeros_(self.context_projection.bias)

    def _one_sample_windows(self, length: int):
        windows = []
        for w in self.widths:
            if w >= length:  # no whole-video proposal in the main bank
                continue
            stride = max(1, w // 2)
            starts = list(range(0, length - w + 1, stride))
            if self.include_tail and starts and starts[-1] != length - w:
                starts.append(length - w)
            windows.extend((s, s + w, w) for s in starts)
        fallback = (length <= 2)
        if not windows:  # explicit frame-level fallback, never a trained event
            windows = [(i, i + 1, 1) for i in range(length)]
            fallback = True
        return windows, fallback

    def geometry(self, video_mask: torch.Tensor, *, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
        """Parameter-free bank and M1 readout footprints on the M0 window grid.

        The outer ring targets a total of w steps, split equally for the
        even main scales. If one side is short, borrow from the other. If
        fewer than w valid exterior steps exist, use all that are available.
        When no exterior exists (one-frame fallback), the ring is all zero.
        The ring does not contribute mass to the exported grounding span.

        This API does not call the M0 residual reader or its LayerNorm.
        The existing forward() is kept unchanged for M0 and old checkpoints.
        """
        if video_mask.ndim != 2 or video_mask.numel() == 0:
            raise ValueError("Event bank requires a nonempty video_mask [B,T].")
        if video_mask.dtype != torch.bool:
            raise TypeError("video_mask must be boolean.")
        if not dtype.is_floating_point:
            raise TypeError("Event footprints require a floating-point dtype.")
        if not bool(video_mask.any(dim=-1).all()):
            raise ValueError("Every video needs at least one valid timestep.")
        if bool(((~video_mask[:, :-1]) & video_mask[:, 1:]).any()):
            raise ValueError("Fixed event windows require left-aligned masks without holes.")

        batch, time_steps = video_mask.shape
        device = video_mask.device
        lengths = video_mask.sum(dim=-1)
        samples = [self._one_sample_windows(int(t)) for t in lengths.tolist()]
        max_events = max(len(windows) for windows, _ in samples)
        padded_windows = torch.tensor(
            [windows + [(0, 0, 0)] * (max_events - len(windows)) for windows, _ in samples],
            dtype=torch.long, device=device,
        )
        start, end, widths = padded_windows.unbind(dim=-1)
        valid = widths.gt(0)
        fallback = torch.tensor([flag for _, flag in samples], dtype=torch.bool, device=device)
        frame_ids = torch.arange(time_steps, device=device).view(1, 1, -1)
        valid_frames = valid.unsqueeze(-1) & video_mask[:, None, :]
        target = valid_frames & (frame_ids >= start.unsqueeze(-1)) & (frame_ids < end.unsqueeze(-1))
        footprint = target.to(dtype) / widths.clamp_min(1).unsqueeze(-1).to(dtype)

        left_limit = start
        right_limit = lengths[:, None] - end
        left_goal = widths // 2
        right_goal = widths - left_goal
        left = torch.minimum(left_limit, left_goal + (right_goal - right_limit).clamp_min(0))
        right = torch.minimum(right_limit, right_goal + (left_goal - left_limit).clamp_min(0))
        ring = valid_frames & (
            ((frame_ids >= (start - left).unsqueeze(-1)) & (frame_ids < start.unsqueeze(-1)))
            | ((frame_ids >= end.unsqueeze(-1)) & (frame_ids < (end + right).unsqueeze(-1)))
        )
        ring_counts = ring.sum(dim=-1)
        ring_footprint = ring.to(dtype) / ring_counts.clamp_min(1).unsqueeze(-1).to(dtype)

        event_count = valid.sum(dim=-1)
        if self.prior_mode == "uniform_window":
            prior = valid.to(dtype) / event_count[:, None].to(dtype)
        else:
            scales = torch.tensor((1, *self.widths), device=device)
            scale_membership = widths.unsqueeze(-1).eq(scales)
            per_scale = scale_membership.sum(dim=1)
            num_scales = per_scale.gt(0).sum(dim=-1, keepdim=True)
            per_window = (scale_membership * per_scale[:, None, :]).sum(dim=-1)
            prior = valid.to(dtype) / (num_scales * per_window).clamp_min(1).to(dtype)

        return {
            "event_valid_mask": valid,
            "event_start": start,
            "event_end_exclusive": end,
            "event_width": widths,
            "event_footprint": footprint,
            "event_ring_footprint": ring_footprint,
            "event_ring_valid_steps": ring_counts,
            "bank_prior_weights": prior,
            "bank_prior_temporal_bias": (prior.unsqueeze(-1) * footprint).sum(dim=1),
            "bank_event_count": event_count,
            "bank_fallback": fallback,
            "bank_train_eligible": (~fallback) & event_count.ge(2),
            "bank_video_lengths": lengths,
            "bank_video_mask": video_mask,
        }

    def forward(self, frame_states: torch.Tensor, video_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        if frame_states.ndim != 3:
            raise ValueError("frame_states must be [B,T,d].")
        batch, time_steps, dim = frame_states.shape
        if time_steps < 1 or dim != self.hidden_dim or video_mask.shape != (batch, time_steps):
            raise ValueError("Event bank frame/mask dimensions mismatch.")
        if video_mask.dtype != torch.bool:
            raise TypeError("video_mask must be boolean.")
        if not bool(video_mask.any(dim=-1).all()):
            raise ValueError("Every video needs at least one valid timestep.")
        if time_steps > 1 and bool(((~video_mask[:, :-1]) & video_mask[:, 1:]).any()):
            raise ValueError("M0 bank needs left-aligned masks without internal holes.")
        lengths = video_mask.sum(dim=-1).tolist()
        per_sample = [self._one_sample_windows(int(t)) for t in lengths]
        max_events = max(len(pair[0]) for pair in per_sample)
        device = frame_states.device
        start = torch.zeros(batch, max_events, device=device, dtype=torch.long)
        end = torch.zeros_like(start)
        widths = torch.zeros_like(start)
        valid = torch.zeros(batch, max_events, device=device, dtype=torch.bool)
        frame_ids = torch.arange(time_steps, device=device).view(1, 1, -1)
        fallback = torch.tensor([p[1] for p in per_sample], device=device, dtype=torch.bool)
        for b, (windows, _) in enumerate(per_sample):
            n = len(windows)
            valid[b, :n] = True
            start[b, :n] = torch.tensor([w[0] for w in windows], device=device)
            end[b, :n] = torch.tensor([w[1] for w in windows], device=device)
            widths[b, :n] = torch.tensor([w[2] for w in windows], device=device)
        footprint = ((frame_ids >= start.unsqueeze(-1)) &
                     (frame_ids < end.unsqueeze(-1)) & valid.unsqueeze(-1) &
                     video_mask[:, None, :]).to(frame_states.dtype)
        footprint = footprint / widths.clamp_min(1).unsqueeze(-1).to(frame_states.dtype)
        # [B,M,T] -> [B,M,d], with autograd to H.
        event_target = torch.bmm(footprint, frame_states)
        left_context = torch.zeros_like(event_target)
        right_context = torch.zeros_like(event_target)
        left_flag = torch.zeros(batch, max_events, device=device, dtype=torch.bool)
        right_flag = torch.zeros_like(left_flag)
        if self.context_radius:
            k = self.context_radius
            left_begin = (start - k).clamp_min(0)
            right_end = torch.minimum(end + k, torch.tensor(lengths, device=device).unsqueeze(-1))
            left_select = ((frame_ids >= left_begin.unsqueeze(-1)) &
                           (frame_ids < start.unsqueeze(-1)) & valid.unsqueeze(-1) &
                           video_mask[:, None, :])
            right_select = ((frame_ids >= end.unsqueeze(-1)) &
                            (frame_ids < right_end.unsqueeze(-1)) & valid.unsqueeze(-1) &
                            video_mask[:, None, :])
            left_count = left_select.sum(dim=-1)
            right_count = right_select.sum(dim=-1)
            left_flag = left_count > 0
            right_flag = right_count > 0
            left_weights = left_select.to(frame_states.dtype) / left_count.clamp_min(1).unsqueeze(-1)
            right_weights = right_select.to(frame_states.dtype) / right_count.clamp_min(1).unsqueeze(-1)
            left_context = torch.bmm(left_weights, frame_states)
            right_context = torch.bmm(right_weights, frame_states)
        flags = torch.stack((left_flag, right_flag), dim=-1).to(frame_states.dtype)
        residual = self.context_projection(torch.cat((left_context, right_context, flags), dim=-1))
        event_states = self.event_norm(event_target + residual) * valid.unsqueeze(-1).to(frame_states.dtype)
        bank_prior = torch.zeros(batch, max_events, device=device, dtype=frame_states.dtype)
        for b in range(batch):
            active = valid[b]
            if self.prior_mode == "uniform_window":
                bank_prior[b, active] = 1.0 / int(active.sum())
            else:
                used_widths = torch.unique(widths[b, active])
                for w in used_widths:
                    scale_mask = active & (widths[b] == w)
                    bank_prior[b, scale_mask] = 1.0 / (len(used_widths) * int(scale_mask.sum()))
        # This location bias is a *prior geometry* diagnostic, not temporal GT.
        coverage_bias = torch.bmm(bank_prior.unsqueeze(1), footprint).squeeze(1)
        token_lengths = torch.as_tensor(lengths, device=device, dtype=frame_states.dtype).unsqueeze(-1)
        positions = torch.stack(
            ((start + end).to(frame_states.dtype) / (2.0 * token_lengths),
             widths.to(frame_states.dtype) / token_lengths), dim=-1
        ) * valid.unsqueeze(-1)
        return {
            "event_states": event_states,
            "event_target_states": event_target,
            "left_context_states": left_context,
            "right_context_states": right_context,
            "left_context_valid": left_flag,
            "right_context_valid": right_flag,
            "event_valid_mask": valid,
            "event_start": start,
            "event_end_exclusive": end,
            "event_width": widths,
            "event_position_features": positions,
            "event_footprint": footprint,
            "bank_prior_weights": bank_prior,
            "bank_prior_temporal_bias": coverage_bias,
            "bank_event_count": valid.sum(dim=-1),
            "bank_fallback": fallback,
            "bank_train_eligible": (~fallback) & (valid.sum(dim=-1) >= 2),
            "bank_video_lengths": torch.as_tensor(lengths, device=device, dtype=torch.long),
        }
