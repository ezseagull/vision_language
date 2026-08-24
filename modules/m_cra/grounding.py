import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveGaussianFilter(nn.Module):
    """
    Gaussian smoothing used by CRA and Q-Bound.

    CRA path (UNCHANGED):
        input temporal distribution
        -> Gaussian Conv1d
        -> temporal softmax

    Q-Bound V1.2 path:
        input temporal probability
        -> mask-aware Gaussian normalized convolution
        -> sum normalization

    The Q-Bound path deliberately does NOT apply a second softmax. It also
    detaches the CRA Gaussian sigma by default so VQA/Q-Bound gradients cannot
    alter the CRA smoothing parameter.

    The mask-aware normalized convolution is important: a uniform Q-Bound
    distribution must remain uniform after smoothing. Plain zero-padded
    Conv1d would otherwise create an artificial center preference even when
    Q-Bound topology has collapsed to one event.
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

    def smooth_probability(
        self,
        weights,
        valid_mask=None,
        detach_sigma=True,
        sigma_override=None,
        eps=1e-6,
    ):
        """
        Probability-preserving Gaussian smoothing for Q-Bound V1.2.

        Args:
            weights:
                [B, T, 1] non-negative temporal probabilities.
            valid_mask:
                optional [B, T] or [B, T, 1] valid-frame mask.
            detach_sigma:
                Used only when ``sigma_override`` is None. If True, gradients
                do not update this filter's learnable sigma.
            sigma_override:
                Optional fixed scalar/tensor sigma. Ver1.2 uses this for the
                Q-Bound branch so its smoothing width is not inherited from
                CRA's learnable Gaussian parameter.
            eps:
                numerical stability constant.

        Returns:
            [B, T, 1] probability distribution.

        Unlike ``forward()``, this method NEVER applies softmax after the
        convolution.

        We use normalized convolution:

            y = conv(p * mask, G) / conv(mask, G)

        before the final sum-normalization. This preserves constant/uniform
        signals at sequence boundaries and for variable valid lengths.
        """
        if weights.ndim != 3 or weights.size(-1) != 1:
            raise ValueError(
                "weights must have shape [B, T, 1]. "
                f"Got {tuple(weights.shape)}."
            )

        batch_size, time_steps, _ = weights.shape

        if valid_mask is None:
            mask = torch.ones(
                batch_size,
                time_steps,
                1,
                dtype=weights.dtype,
                device=weights.device,
            )
        else:
            if valid_mask.ndim == 2:
                valid_mask = valid_mask.unsqueeze(-1)

            if valid_mask.shape != weights.shape:
                raise ValueError(
                    "valid_mask must match weights. "
                    f"Got weights={tuple(weights.shape)}, "
                    f"mask={tuple(valid_mask.shape)}."
                )

            mask = valid_mask.to(
                device=weights.device,
                dtype=weights.dtype,
            )

        # Defensive probability normalization before smoothing.
        weights = weights.clamp_min(0.0)
        weights = weights * mask

        input_mass = weights.sum(
            dim=1,
            keepdim=True,
        )

        uniform_fallback = mask / mask.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1.0)

        weights = torch.where(
            input_mass > eps,
            weights / input_mass.clamp_min(eps),
            uniform_fallback,
        )

        if sigma_override is None:
            sigma = (
                self.sigma.detach()
                if detach_sigma
                else self.sigma
            )
        else:
            # Fixed Q-Bound smoothing width. This deliberately has no
            # gradient path to CRA's learnable sigma.
            sigma = torch.as_tensor(
                sigma_override,
                device=weights.device,
                dtype=weights.dtype,
            )

        kernel = self.create_gaussian_kernel(
            self.kernel_size,
            sigma,
            device=weights.device,
            dtype=weights.dtype,
        )

        # [B,T,1] -> [B,1,T]
        weights_1d = weights.permute(0, 2, 1)
        mask_1d = mask.permute(0, 2, 1)

        numerator = F.conv1d(
            weights_1d,
            kernel,
            padding=self.padding,
        )

        # Correct the truncated Gaussian support near valid boundaries.
        support = F.conv1d(
            mask_1d,
            kernel,
            padding=self.padding,
        )

        smoothed = numerator / support.clamp_min(eps)
        smoothed = smoothed.permute(0, 2, 1)

        # Invalid frames receive exactly zero probability.
        smoothed = smoothed.clamp_min(0.0) * mask

        output_mass = smoothed.sum(
            dim=1,
            keepdim=True,
        )

        smoothed = torch.where(
            output_mass > eps,
            smoothed / output_mass.clamp_min(eps),
            uniform_fallback,
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
    CRA grounding plus a separate Q-Bound V1.2 probability path.

    CRA behavior is preserved exactly:
        raw CRA score
        -> softmax = ori_key
        -> Gaussian Conv1d
        -> second softmax = cra_key

    Q-Bound V1.2 does NOT use CRA temporal-logit fusion. Its main temporal
    distribution comes directly from QBoundCore.event_temporal_prob:

        Q-Bound temporal probability
        -> mask-aware Gaussian normalized convolution
        -> sum normalization
        -> qbound_key

    CRA and Q-Bound keys are mixed later in refine.py during the deterministic
    handover schedule. This module does not contain alpha, beta, a learnable
    CRA/Q-Bound gate, or a handover parameter.

    The old Version 1.1 ``finalize_grounding(..., temporal_bias=...)`` API is
    intentionally kept for checkpoint/code compatibility, but Ver1.1.3 must
    not use that method for its Q-Bound main path.

    Legacy Version 1.1 principle
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
        qbound_sigma=None,
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

        # Ver1.2 decouples Q-Bound smoothing from CRA's learnable sigma.
        # The value defaults to the same initial width for backward-compatible
        # behavior, but it is a non-persistent buffer rather than a parameter.
        qbound_sigma_value = (
            float(initial_sigma)
            if qbound_sigma is None
            else float(qbound_sigma)
        )
        if qbound_sigma_value <= 0.0:
            raise ValueError("qbound_sigma must be positive.")
        self.register_buffer(
            "qbound_sigma",
            torch.tensor(qbound_sigma_value, dtype=torch.float32),
            persistent=False,
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

    @staticmethod
    def _normalize_probability(
        probability,
        valid_mask,
        eps=1e-6,
    ):
        """
        Normalize a [B,T] non-negative temporal distribution under a mask.
        """
        if probability.ndim == 3 and probability.size(-1) == 1:
            probability = probability.squeeze(-1)

        if probability.ndim != 2:
            raise ValueError(
                "probability must be [B,T] or [B,T,1]. "
                f"Got {tuple(probability.shape)}."
            )

        if valid_mask.shape != probability.shape:
            raise ValueError(
                "probability and valid_mask must have the same [B,T] shape. "
                f"Got {tuple(probability.shape)} and "
                f"{tuple(valid_mask.shape)}."
            )

        probability = probability.clamp_min(0.0)
        probability = (
            probability
            * valid_mask.to(probability.dtype)
        )

        mass = probability.sum(
            dim=-1,
            keepdim=True,
        )

        fallback = valid_mask.to(probability.dtype)
        fallback = fallback / fallback.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)

        return torch.where(
            mass > eps,
            probability / mass.clamp_min(eps),
            fallback,
        )

    def finalize_probability_grounding(
        self,
        temporal_probability,
        valid_mask=None,
        window_sizes=None,
        detach_sigma=True,
        apply_smoothing=True,
    ):
        """
        Build Q-Bound V1.2 grounding directly from temporal probabilities.

        Expected main input:
            QBoundCore["event_temporal_prob"]      [B,T]

        Counterfactual input:
            QBoundCore["merge_event_temporal_prob"][:, 0]  [B,T]

        Pipeline:
            probability
            -> normalize under valid mask
            -> Gaussian smooth_probability()
            -> sum-normalized qbound_key
            -> the SAME CRA interval extraction (gen_grounding)

        There is deliberately:
            * no CRA temporal-logit addition,
            * no alpha/beta gate,
            * no second temporal softmax,
            * no Q-Bound gradient into CRA sigma when detach_sigma=True.
        """
        if temporal_probability.ndim == 3 and temporal_probability.size(-1) == 1:
            temporal_probability = temporal_probability.squeeze(-1)

        if temporal_probability.ndim != 2:
            raise ValueError(
                "temporal_probability must be [B,T] or [B,T,1]. "
                f"Got {tuple(temporal_probability.shape)}."
            )

        batch_size, time_steps = temporal_probability.shape

        valid_mask = self._normalize_valid_mask(
            valid_mask,
            batch_size,
            time_steps,
            temporal_probability.device,
        )

        pre_smooth_key = self._normalize_probability(
            temporal_probability,
            valid_mask,
        )

        if apply_smoothing:
            qbound_key = self.gs_filter.smooth_probability(
                pre_smooth_key.unsqueeze(-1),
                valid_mask=valid_mask,
                detach_sigma=detach_sigma,
                sigma_override=self.qbound_sigma,
            ).squeeze(-1)
        else:
            # Useful controlled ablation: QBoundCore already returns a valid
            # probability distribution, so smoothing is not mathematically
            # required for the Q-Bound branch.
            qbound_key = pre_smooth_key

        # Defensive normalization after smoothing. smooth_probability already
        # guarantees this, but keeping the invariant explicit here makes the
        # refine.py contract robust.
        qbound_key = self._normalize_probability(
            qbound_key,
            valid_mask,
        )

        time_param = self.gen_grounding(
            qbound_key,
            window_sizes=window_sizes,
        )

        time_param["pre_smooth_key"] = pre_smooth_key
        time_param["post_smooth_key"] = qbound_key
        time_param["qbound_key"] = qbound_key
        time_param["qbound_smoothing_applied"] = torch.as_tensor(
            float(apply_smoothing),
            dtype=qbound_key.dtype,
            device=qbound_key.device,
        )
        time_param["qbound_sigma"] = self.qbound_sigma.to(
            device=qbound_key.device,
            dtype=qbound_key.dtype,
        )
        time_param["valid_mask"] = valid_mask

        return time_param

    def finalize_grounding(
        self,
        base_temporal,
        temporal_bias=None,
        window_sizes=None,
    ):
        """
        LEGACY V1.1 temporal-logit fusion path.

        Kept unchanged for compatibility and CRA baseline reproduction.
        Q-Bound V1.2 must use ``finalize_probability_grounding()`` instead.

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

        Legacy Ver1.1 call:
            grounding_module(
                v,
                qa,
                window_sizes=...,
                temporal_bias=beta * event_logits,
                valid_mask=video_mask,
            )

        Q-Bound V1.2 does not use this convenience forward for its main
        temporal path. It calls ``finalize_probability_grounding()`` directly.
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
