from typing import Optional

import torch
import torch.nn as nn


class CRAQBoundAdapter(nn.Module):
    """
    Thin CRA <-> Q-Bound integration layer for Version 1.1.

    Q-Bound itself stays backbone-agnostic. This adapter owns the two
    backbone-specific learnable contributions:

        z_final = z_base + alpha * A(z_event)
        l_final = l_base + beta  * l_event

    alpha and beta start at zero, so enabling Q-Bound preserves the CRA
    baseline at initialization.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        dropout: float = 0.1,
        event_scale_init: float = 0.0,
        temporal_scale_init: float = 0.0,
    ) -> None:
        super().__init__()

        self.event_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        nn.init.eye_(self.event_proj.weight)

        # No adapter-level dropout in V1.1. Counterfactual necessity
        # must not be contaminated by different random masks for original,
        # merge, and split representations. CRA's existing backbone dropout
        # remains untouched.
        del dropout

        self.event_scale = nn.Parameter(
            torch.tensor(float(event_scale_init))
        )
        self.temporal_scale = nn.Parameter(
            torch.tensor(float(temporal_scale_init))
        )

    @property
    def alpha(self) -> torch.Tensor:
        return torch.tanh(self.event_scale)

    @property
    def beta(self) -> torch.Tensor:
        return torch.tanh(self.temporal_scale)

    def project_event_repr(
        self,
        event_repr: torch.Tensor,
    ) -> torch.Tensor:
        return self.event_proj(event_repr)

    def fuse_video(
        self,
        base_repr: torch.Tensor,
        event_repr: torch.Tensor,
    ):
        """
        base_repr:  [B, D]
        event_repr: [B, D]
        """
        event_delta = self.project_event_repr(event_repr)
        fused = base_repr + self.alpha * event_delta

        return fused, {
            "event_delta": event_delta,
            "event_scale": self.event_scale,
            "alpha": self.alpha,
        }

    def fuse_counterfactual_video(
        self,
        base_repr: torch.Tensor,
        event_repr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse merge/split event representations through the SAME adapter
        used by the original path.

        base_repr:  [B, D]
        event_repr: [B, N, D] or [B, D]
        """
        if event_repr.ndim == 2:
            return self.fuse_video(base_repr, event_repr)[0]

        if event_repr.ndim != 3:
            raise ValueError(
                "counterfactual event_repr must be [B,D] or [B,N,D]."
            )

        projected = self.event_proj(event_repr)
        return base_repr.unsqueeze(1) + self.alpha * projected

    def fuse_temporal_logits(
        self,
        base_logits: torch.Tensor,
        event_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        """
        Fuse BEFORE key/start/end are computed.

        base_logits:  [B, T]
        event_logits: [B, T]
        """
        if base_logits.shape != event_logits.shape:
            raise ValueError(
                "base_logits and event_logits must have the same shape."
            )

        fused = base_logits + self.beta * event_logits

        if valid_mask is not None:
            if valid_mask.shape != fused.shape:
                raise ValueError(
                    "valid_mask must have the same [B,T] shape as logits."
                )
            fused = fused.masked_fill(~valid_mask.bool(), -1e4)

        return fused, {
            "temporal_scale": self.temporal_scale,
            "beta": self.beta,
        }

    def fuse_counterfactual_temporal_logits(
        self,
        base_logits: torch.Tensor,
        event_logits: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        base_logits:  [B, T]
        event_logits: [B, N, T] or [B, T]
        """
        if event_logits.ndim == 2:
            return self.fuse_temporal_logits(
                base_logits,
                event_logits,
                valid_mask,
            )[0]

        if event_logits.ndim != 3:
            raise ValueError(
                "counterfactual event_logits must be [B,T] or [B,N,T]."
            )

        if (
            event_logits.size(0) != base_logits.size(0)
            or event_logits.size(-1) != base_logits.size(-1)
        ):
            raise ValueError(
                "counterfactual event logits do not match base logits."
            )

        fused = (
            base_logits.unsqueeze(1)
            + self.beta * event_logits
        )

        if valid_mask is not None:
            fused = fused.masked_fill(
                ~valid_mask.bool().unsqueeze(1),
                -1e4,
            )

        return fused
