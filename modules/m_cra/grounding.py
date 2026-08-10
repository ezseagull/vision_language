import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveGaussianFilter(nn.Module):
    """
    Gaussian smoothing used by CRA grounding.

    Version 1.1 keeps the original CRA behavior:
        input temporal distribution
        -> Gaussian Conv1d
        -> temporal softmax

    valid_mask is optional and only prevents padded positions from receiving
    probability mass. When valid_mask is None (or all True), behavior matches
    the original implementation.
    """

    def __init__(
        self,
        kernel_size=5,
        initial_sigma=1.0,
        max_sigma=5.0,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.sigma = nn.Parameter(
            torch.tensor(
                float(initial_sigma),
                dtype=torch.float32,
            )
        )
        self.max_sigma = float(max_sigma)

    @staticmethod
    def _masked_softmax(
        logits,
        valid_mask=None,
        dim=1,
    ):
        if valid_mask is None:
            return F.softmax(logits, dim=dim)

        if valid_mask.ndim == 2:
            valid_mask = valid_mask.unsqueeze(-1)

        if valid_mask.shape != logits.shape:
            raise ValueError(
                "valid_mask must match temporal logits. "
                f"Got logits={tuple(logits.shape)}, "
                f"mask={tuple(valid_mask.shape)}."
            )

        valid_mask = valid_mask.bool()

        masked_logits = logits.masked_fill(
            ~valid_mask,
            -1e4,
        )
        prob = F.softmax(
            masked_logits,
            dim=dim,
        )
        prob = prob * valid_mask.to(prob.dtype)

        return (
            prob
            / prob.sum(
                dim=dim,
                keepdim=True,
            ).clamp_min(1e-6)
        )

    def forward(
        self,
        weights,
        valid_mask=None,
    ):
        """
        weights:
            [B, T, 1]

        returns:
            [B, T, 1]
        """
        if weights.ndim != 3 or weights.size(-1) != 1:
            raise ValueError(
                "weights must have shape [B, T, 1]. "
                f"Got {tuple(weights.shape)}."
            )

        kernel = self.create_gaussian_kernel(
            self.kernel_size,
            self.sigma,
            device=weights.device,
            dtype=weights.dtype,
        )

        weights = weights.permute(0, 2, 1)
        smoothed = F.conv1d(
            weights,
            kernel,
            padding=self.padding,
        )
        smoothed = smoothed.permute(0, 2, 1)

        smoothed = self._masked_softmax(
            smoothed,
            valid_mask=valid_mask,
            dim=1,
        )

        return smoothed

    def create_gaussian_kernel(
        self,
        kernel_size,
        sigma,
        device,
        dtype,
    ):
        x = (
            torch.arange(
                kernel_size,
                dtype=dtype,
                device=device,
            )
            - kernel_size // 2
        )

        # Preserve the learnable CRA sigma while avoiding division by zero.
        sigma_used = sigma.to(
            device=device,
            dtype=dtype,
        ).abs().clamp(
            min=1e-4,
            max=self.max_sigma,
        )

        gaussian_kernel = torch.exp(
            -0.5 * (x / sigma_used).pow(2)
        )
        gaussian_kernel = (
            gaussian_kernel
            / gaussian_kernel.sum().clamp_min(1e-6)
        )

        return gaussian_kernel.view(1, 1, -1)


class GroundingModule(nn.Module):
    """
    CRA grounding with pre-grounding Q-Bound temporal-logit integration.

    Version 1.1 principle
    ---------------------
    Q-Bound must affect temporal evidence BEFORE `key`, `start`, and `end`
    are produced.

    To preserve the CRA baseline at beta=0, the original CRA frame
    distribution is first computed exactly as before:

        raw score
        -> softmax over time = ori_key

    We then use log(ori_key) as a numerically stable temporal-logit
    representation:

        cra_temporal_logits = log(ori_key)

    The CRA adapter may provide:

        temporal_bias = beta * qbound_event_temporal_logits

    and this module computes:

        fused_temporal_logits =
            cra_temporal_logits + temporal_bias

        pre_smooth_key =
            softmax(fused_temporal_logits)

        final key =
            GaussianFilter(pre_smooth_key)

    Therefore:
        temporal_bias == 0
        => pre_smooth_key == ori_key
        => the old CRA grounding path is preserved.

    Public API
    ----------
    compute_base_temporal(v, qa, valid_mask=None)
        Compute CRA temporal evidence once. This is useful because Q-Bound
        may optionally consume the detached CRA key as a prior.

    finalize_grounding(base_temporal, temporal_bias=None, ...)
        Add Q-Bound evidence and only then produce key/start/end.

    forward(...)
        Backward-compatible convenience wrapper.
    """

    def __init__(
        self,
        d_model=768,
        dropout=0.3,
        initial_sigma=1.0,
        window_sizes=(1, 3, 5),
    ):
        super().__init__()

        self.qa_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        self.v_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # IMPORTANT:
        # Version 1.0 had Softmax(dim=-2) as the last item here.
        # It has no parameters, so removing it does not change checkpoint
        # parameter names. We explicitly apply the same softmax in
        # compute_base_temporal(), which lets us access the pre-softmax score.
        self.grounding = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        self.gs_filter = AdaptiveGaussianFilter(
            initial_sigma=float(initial_sigma),
        )

        self.window_sizes = list(window_sizes)

    @staticmethod
    def _normalize_valid_mask(
        valid_mask,
        batch_size,
        time_steps,
        device,
    ):
        if valid_mask is None:
            return torch.ones(
                batch_size,
                time_steps,
                dtype=torch.bool,
                device=device,
            )

        while (
            valid_mask.ndim > 2
            and valid_mask.size(1) == 1
        ):
            valid_mask = valid_mask.squeeze(1)

        if valid_mask.shape != (
            batch_size,
            time_steps,
        ):
            raise ValueError(
                "valid_mask must be [B, T]. "
                f"Expected {(batch_size, time_steps)}, "
                f"got {tuple(valid_mask.shape)}."
            )

        return valid_mask.to(
            device=device,
        ).bool()

    @staticmethod
    def _masked_softmax(
        logits,
        valid_mask,
        dim=-1,
    ):
        if logits.shape != valid_mask.shape:
            raise ValueError(
                "logits and valid_mask must have the same shape. "
                f"Got {tuple(logits.shape)} and "
                f"{tuple(valid_mask.shape)}."
            )

        masked_logits = logits.masked_fill(
            ~valid_mask,
            -1e4,
        )

        prob = F.softmax(
            masked_logits,
            dim=dim,
        )
        prob = (
            prob
            * valid_mask.to(prob.dtype)
        )

        return (
            prob
            / prob.sum(
                dim=dim,
                keepdim=True,
            ).clamp_min(1e-6)
        )

    @staticmethod
    def _normalize_temporal_bias(
        temporal_bias,
        batch_size,
        time_steps,
        device,
        dtype,
    ):
        if temporal_bias is None:
            return torch.zeros(
                batch_size,
                time_steps,
                device=device,
                dtype=dtype,
            )

        if (
            temporal_bias.ndim == 3
            and temporal_bias.size(-1) == 1
        ):
            temporal_bias = temporal_bias.squeeze(-1)

        if temporal_bias.shape != (
            batch_size,
            time_steps,
        ):
            raise ValueError(
                "temporal_bias must be [B, T]. "
                f"Expected {(batch_size, time_steps)}, "
                f"got {tuple(temporal_bias.shape)}."
            )

        return temporal_bias.to(
            device=device,
            dtype=dtype,
        )

    def compute_base_temporal(
        self,
        v,
        qa,
        valid_mask=None,
    ):
        """
        Compute CRA temporal evidence WITHOUT Q-Bound modification.

        Returns tensors needed both for:
          1) the optional detached CRA prior passed into Q-Bound, and
          2) the final Q-Bound-biased grounding computation.

        v:
            [B, T, D]
        qa:
            [B, D]
        """
        if v.ndim != 3:
            raise ValueError(
                "v must have shape [B, T, D]."
            )
        if qa.ndim != 2:
            raise ValueError(
                "qa must have shape [B, D]."
            )

        batch_size, time_steps, _ = v.shape
        valid_mask = self._normalize_valid_mask(
            valid_mask,
            batch_size,
            time_steps,
            v.device,
        )

        v_state = self.v_proj(v)
        qa_state = self.qa_proj(qa)

        gate = torch.matmul(
            v_state,
            qa_state.unsqueeze(-1),
        ).tanh()

        raw_temporal_logits = (
            self.grounding(
                v_state * gate
            ).squeeze(-1)
        )

        # This explicitly reproduces the Softmax(dim=-2) that used to be
        # the final layer of self.grounding.
        ori_key = self._masked_softmax(
            raw_temporal_logits,
            valid_mask,
            dim=-1,
        )

        # log(probability) is equivalent to logits up to an additive
        # per-sample constant, which softmax ignores.
        cra_temporal_logits = torch.log(
            ori_key.clamp_min(1e-8)
        )

        # CRA-only key after the same Gaussian path used by the baseline.
        cra_key = self.gs_filter(
            ori_key.unsqueeze(-1),
            valid_mask=valid_mask,
        ).squeeze(-1)

        return {
            "raw_temporal_logits": (
                raw_temporal_logits
            ),
            "cra_temporal_logits": (
                cra_temporal_logits
            ),
            "ori_key": ori_key,
            "cra_key": cra_key,
            "valid_mask": valid_mask,
        }

    def finalize_grounding(
        self,
        base_temporal,
        temporal_bias=None,
        window_sizes=None,
    ):
        """
        Fuse Q-Bound temporal evidence before producing key/start/end.

        base_temporal:
            output of compute_base_temporal()

        temporal_bias:
            [B, T], usually
                beta * qbound_out["event_temporal_logits"]
        """
        cra_temporal_logits = (
            base_temporal[
                "cra_temporal_logits"
            ]
        )
        valid_mask = base_temporal[
            "valid_mask"
        ]

        batch_size, time_steps = (
            cra_temporal_logits.shape
        )

        temporal_bias = (
            self._normalize_temporal_bias(
                temporal_bias,
                batch_size,
                time_steps,
                cra_temporal_logits.device,
                cra_temporal_logits.dtype,
            )
        )

        # Invalid positions must never enter the fused temporal evidence.
        temporal_bias = (
            temporal_bias
            * valid_mask.to(
                temporal_bias.dtype
            )
        )

        fused_temporal_logits = (
            cra_temporal_logits
            + temporal_bias
        )

        pre_smooth_key = self._masked_softmax(
            fused_temporal_logits,
            valid_mask,
            dim=-1,
        )

        keyframe_prob = self.gs_filter(
            pre_smooth_key.unsqueeze(-1),
            valid_mask=valid_mask,
        ).squeeze(-1)

        time_param = self.gen_grounding(
            keyframe_prob,
            window_sizes=window_sizes,
        )

        # Baseline / diagnostic signals.
        time_param["ori_key"] = (
            base_temporal["ori_key"]
        )
        time_param["cra_key"] = (
            base_temporal["cra_key"]
        )

        # Logit-level signals used by Ver1.1 analysis.
        time_param[
            "raw_temporal_logits"
        ] = base_temporal[
            "raw_temporal_logits"
        ]
        time_param[
            "cra_temporal_logits"
        ] = cra_temporal_logits
        time_param[
            "temporal_bias"
        ] = temporal_bias
        time_param[
            "fused_temporal_logits"
        ] = fused_temporal_logits
        time_param[
            "pre_smooth_key"
        ] = pre_smooth_key

        # Useful scalar diagnostic. This is zero at Q-Bound initialization
        # because beta starts from zero in CRAQBoundAdapter.
        time_param[
            "temporal_bias_abs_mean"
        ] = temporal_bias.abs().mean()

        return time_param

    def gen_grounding(
        self,
        keyframe_prob,
        window_sizes=None,
    ):
        bs, length = keyframe_prob.size()[:2]

        if window_sizes is None:
            window_sizes = self.window_sizes

        max_indices = self.find_best_interval(
            keyframe_prob,
            window_sizes=window_sizes,
        )

        start_indices = (
            max_indices[:, 0]
            .unsqueeze(1)
            .expand(-1, length)
        )
        end_indices = (
            max_indices[:, 1]
            .unsqueeze(1)
            .expand(-1, length)
        )

        range_tensor = torch.arange(
            length,
            device=keyframe_prob.device,
        ).expand(bs, length)

        keyframe_mask = (
            (range_tensor >= start_indices)
            & (range_tensor <= end_indices)
        )

        denominator = float(
            max(length - 1, 1)
        )

        start_time = (
            max_indices[:, 0].float()
            / denominator
        )
        end_time = (
            max_indices[:, 1].float()
            / denominator
        )

        start_time = start_time.clamp(
            0.0,
            1.0,
        )
        end_time = end_time.clamp(
            0.0,
            1.0,
        )

        return {
            "key": keyframe_prob,
            "max_indices": max_indices,
            "start": start_time,
            "end": end_time,
            "mask": keyframe_mask,
        }

    def find_best_interval(
        self,
        keyframe_prob,
        window_sizes=None,
    ):
        """
        Kept behaviorally aligned with the current CRA implementation for
        the first Ver1.1 experiment.

        NOTE:
        The original code uses:
            contains_max = (max_positions >= start) & (max_positions < end)

        even though `end` is later treated as inclusive. This makes the
        window-size-1 case ineffective. We intentionally do NOT change that
        unrelated baseline behavior in the first Ver1.1 run, so any metric
        change can be attributed to Q-Bound. It can be fixed in a controlled
        ablation later.
        """
        if window_sizes is None:
            window_sizes = self.window_sizes

        bs, length = keyframe_prob.shape
        max_interval_size = length // 2

        max_indices = torch.zeros(
            (bs, 2),
            dtype=torch.long,
            device=keyframe_prob.device,
        )

        best_scores = torch.full(
            (bs,),
            float("-inf"),
            dtype=keyframe_prob.dtype,
            device=keyframe_prob.device,
        )

        max_positions = keyframe_prob.argmax(
            dim=1
        )

        for window_size in window_sizes:
            if (
                window_size <= 0
                or window_size > max_interval_size
            ):
                continue

            sliding_sums = F.conv1d(
                keyframe_prob.unsqueeze(1),
                weight=torch.ones(
                    (1, 1, window_size),
                    device=keyframe_prob.device,
                    dtype=keyframe_prob.dtype,
                ),
                padding=0,
                stride=1,
            ).squeeze(1)

            for start in range(
                length - window_size + 1
            ):
                end = (
                    start
                    + window_size
                    - 1
                )

                # Preserved from the current CRA code for apples-to-apples
                # Ver1.0 -> Ver1.1 comparison. See docstring above.
                contains_max = (
                    (max_positions >= start)
                    & (max_positions < end)
                )

                window_scores = (
                    sliding_sums[:, start]
                    .clone()
                )

                window_scores = (
                    window_scores.masked_fill(
                        ~contains_max,
                        float("-inf"),
                    )
                )

                better_scores = (
                    window_scores
                    > best_scores
                )

                best_scores = torch.where(
                    better_scores,
                    window_scores,
                    best_scores,
                )

                if better_scores.any():
                    max_indices[
                        better_scores,
                        0,
                    ] = start
                    max_indices[
                        better_scores,
                        1,
                    ] = end

        return max_indices

    def forward(
        self,
        v,
        qa,
        window_sizes=None,
        temporal_bias=None,
        valid_mask=None,
        return_base_temporal=False,
    ):
        """
        Backward-compatible forward.

        Old CRA call still works:
            grounding_module(v, qa, window_sizes=...)

        Ver1.1 call:
            grounding_module(
                v,
                qa,
                window_sizes=...,
                temporal_bias=beta * event_logits,
                valid_mask=video_mask,
            )
        """
        base_temporal = (
            self.compute_base_temporal(
                v,
                qa,
                valid_mask=valid_mask,
            )
        )

        time_param = self.finalize_grounding(
            base_temporal,
            temporal_bias=temporal_bias,
            window_sizes=window_sizes,
        )

        if return_base_temporal:
            time_param[
                "base_temporal"
            ] = base_temporal

        return time_param
