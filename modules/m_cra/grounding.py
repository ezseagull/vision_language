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

    Q-Bound V1.6 probability path:
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
        Probability-preserving Gaussian smoothing for Q-Bound V1.6.

        Args:
            weights:
                [B, T, 1] non-negative temporal probabilities.
            valid_mask:
                optional [B, T] or [B, T, 1] valid-frame mask.
            detach_sigma:
                Used only when ``sigma_override`` is None. If True, gradients
                do not update this filter's learnable sigma.
            sigma_override:
                Optional fixed scalar/tensor sigma. Ver1.6 uses this for the
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
    CRA grounding plus Q-Bound V1.6 coarse-to-fine grounding.

    CRA path is preserved exactly:
        raw CRA score
        -> softmax = ori_key
        -> Gaussian Conv1d
        -> second softmax = cra_key
        -> legacy fixed-window interval extraction

    Q-Bound V1.6 main path:
        QBoundCore.refined_temporal_prob
        -> probability-preserving Gaussian smoothing
        -> qbound_key
        -> refined frame-level start/end supplied by QBoundCore

    V1.6 keeps the V1.5 coarse event-chain span as a diagnostic, but the main
    pre-hoc interval is the frame-refined span. The legacy CRA [1,3,5] fixed
    window is also retained only as a controlled diagnostic.

    This yields three readouts on the same V1.6 temporal key:
        1) refined V1.6 interval        -- MAIN
        2) coarse V1.5 event span      -- diagnostic
        3) CRA fixed short window      -- diagnostic

    The Q-Bound probability path still:
        * applies no CRA temporal-logit fusion;
        * uses no CRA->Q-Bound handover;
        * applies no second temporal softmax after QBoundCore;
        * uses a fixed/detached Q-Bound Gaussian width.

    Public API
    ----------
    compute_base_temporal(v, qa, valid_mask=None)
        Original CRA temporal evidence.

    finalize_probability_grounding(...)
        Legacy V1.4 Q-Bound readout.  Kept for controlled ablations.

    finalize_composed_grounding(...)
        Legacy V1.5 adaptive event-chain readout, retained for ablation.

    finalize_refined_grounding(...)
        V1.6 main readout. Uses the refined Q-Bound probability for grounded
        pooling and the frame-level refined interval for pre-hoc grounding.

    finalize_grounding(...)
        Legacy V1.1 CRA/Q-Bound temporal-logit fusion, compatibility only.

    forward(...)
        Backward-compatible CRA convenience wrapper.
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

        # V1.6 keeps Q-Bound smoothing decoupled from CRA's learnable sigma.
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


    def gen_adaptive_grounding(
        self,
        keyframe_prob,
        selected_span_start_frame,
        selected_span_end_frame,
        valid_mask=None,
        selected_frame_mask=None,
    ):
        """
        Convert explicit Q-Bound frame indices into a contiguous interval.

        Unlike ``gen_grounding()``, this helper performs NO fixed-window
        search. It is shared by the V1.5 coarse Composer diagnostic and the
        V1.6 frame-refined main readout.

        Args
        ----
        keyframe_prob:
            [B,T] temporal probability used for downstream grounded pooling.
        selected_span_start_frame / selected_span_end_frame:
            [B] integer frame indices from QBoundCore.
        valid_mask:
            optional [B,T].
        selected_frame_mask:
            optional detached [B,T] mask from QBoundCore.  It is retained as a
            diagnostic consistency signal; the returned main mask is rebuilt
            from start/end so it is guaranteed contiguous.

        Returns
        -------
        Dict with the same main contract as ``gen_grounding()``:
            key, max_indices, start, end, mask
        plus adaptive-span diagnostics.
        """
        if keyframe_prob.ndim != 2:
            raise ValueError(
                "keyframe_prob must have shape [B,T], got "
                f"{tuple(keyframe_prob.shape)}."
            )

        batch_size, time_steps = keyframe_prob.shape
        device = keyframe_prob.device

        valid_mask = self._normalize_valid_mask(
            valid_mask,
            batch_size,
            time_steps,
            device,
        )

        def _prepare_index(index, name):
            if not torch.is_tensor(index):
                index = torch.as_tensor(index, device=device)
            index = index.to(device=device).reshape(-1).long().detach()
            if index.numel() != batch_size:
                raise ValueError(
                    f"{name} must contain one index per sample. "
                    f"Expected {batch_size}, got {index.numel()}."
                )
            return index

        start_index = _prepare_index(
            selected_span_start_frame,
            "selected_span_start_frame",
        )
        end_index = _prepare_index(
            selected_span_end_frame,
            "selected_span_end_frame",
        )

        frame_index = torch.arange(
            time_steps,
            device=device,
            dtype=torch.long,
        ).view(1, time_steps).expand(batch_size, -1)

        # Last valid temporal index per sample.  This equals valid_length-1
        # for the contiguous-prefix masks used by NExT-GQA, but also avoids
        # assuming every sample uses all T slots.
        last_valid_index = torch.where(
            valid_mask,
            frame_index,
            torch.full_like(frame_index, -1),
        ).max(dim=-1).values.clamp_min(0)

        start_index = start_index.clamp(min=0, max=max(time_steps - 1, 0))
        end_index = end_index.clamp(min=0, max=max(time_steps - 1, 0))
        start_index = torch.minimum(start_index, last_valid_index)
        end_index = torch.minimum(end_index, last_valid_index)

        invalid_order = end_index < start_index
        if invalid_order.any():
            # Defensive fallback to the single selected start frame.  A valid
            # contiguous event chain from QBoundCore should never trigger this.
            end_index = torch.where(
                invalid_order,
                start_index,
                end_index,
            )

        main_mask = (
            (frame_index >= start_index.unsqueeze(1))
            & (frame_index <= end_index.unsqueeze(1))
            & valid_mask
        )

        if selected_frame_mask is None:
            composer_mask = main_mask
            composer_mask_consistency = torch.ones(
                batch_size,
                dtype=keyframe_prob.dtype,
                device=device,
            )
        else:
            if not torch.is_tensor(selected_frame_mask):
                selected_frame_mask = torch.as_tensor(
                    selected_frame_mask,
                    device=device,
                )
            if selected_frame_mask.shape != (batch_size, time_steps):
                raise ValueError(
                    "selected_frame_mask must be [B,T], got "
                    f"{tuple(selected_frame_mask.shape)}."
                )
            composer_mask = selected_frame_mask.to(device=device).bool().detach()
            composer_mask = composer_mask & valid_mask
            composer_mask_consistency = (
                composer_mask == main_mask
            ).all(dim=-1).to(keyframe_prob.dtype)

        max_indices = torch.stack(
            [start_index, end_index],
            dim=-1,
        )

        # Normalize by the last valid feature index so padded suffixes do not
        # compress the predicted interval.  For the current NExT-GQA setup,
        # every sample has 32 valid temporal tokens, so this is identical to
        # the legacy T-1 normalization.
        denominator = last_valid_index.to(keyframe_prob.dtype).clamp_min(1.0)
        start_time = (
            start_index.to(keyframe_prob.dtype) / denominator
        ).clamp(0.0, 1.0)
        end_time = (
            end_index.to(keyframe_prob.dtype) / denominator
        ).clamp(0.0, 1.0)

        span_length = main_mask.sum(dim=-1).to(keyframe_prob.dtype)
        valid_length = valid_mask.sum(dim=-1).to(keyframe_prob.dtype).clamp_min(1.0)
        span_fraction = span_length / valid_length

        span_key_mass = (
            keyframe_prob * main_mask.to(keyframe_prob.dtype)
        ).sum(dim=-1)

        peak_index = keyframe_prob.argmax(dim=-1)
        peak_in_span = (
            (peak_index >= start_index)
            & (peak_index <= end_index)
        ).to(keyframe_prob.dtype)

        return {
            "key": keyframe_prob,
            "max_indices": max_indices,
            "start": start_time,
            "end": end_time,
            "mask": main_mask,

            "adaptive_span_used": torch.ones(
                (),
                dtype=keyframe_prob.dtype,
                device=device,
            ),
            "adaptive_span_start_frame": start_index,
            "adaptive_span_end_frame": end_index,
            "adaptive_span_length": span_length,
            "adaptive_span_fraction": span_fraction,
            "adaptive_span_key_mass": span_key_mass,
            "adaptive_span_peak_in_span": peak_in_span,
            "adaptive_span_composer_mask": composer_mask,
            "adaptive_span_mask_consistency": composer_mask_consistency,
            "valid_mask": valid_mask,
        }

    def finalize_composed_grounding(
        self,
        temporal_probability,
        selected_span_start_frame,
        selected_span_end_frame,
        valid_mask=None,
        selected_frame_mask=None,
        window_sizes=None,
        detach_sigma=True,
        apply_smoothing=True,
    ):
        """
        Legacy Q-Bound V1.5 coarse event-span grounding adapter.

        Inputs
        ------
        temporal_probability:
            QBoundCore["composed_temporal_prob"] / ["event_temporal_prob"],
            shape [B,T].
        selected_span_start_frame / selected_span_end_frame:
            Detached hard latent event-span boundaries selected by the
            QBoundCore composer.
        selected_frame_mask:
            Optional detached exact selected-chain mask from QBoundCore.

        Main path
        ---------
            composed temporal probability
                -> normalize under valid mask
                -> optional probability-preserving Gaussian smoothing
                -> qbound_key used for grounded video pooling

            selected latent event chain
                -> adaptive start/end/mask used for pre-hoc grounding

        The legacy CRA fixed-window extractor is evaluated in parallel only as
        a diagnostic and does NOT define the V1.5 main start/end.
        """
        if (
            temporal_probability.ndim == 3
            and temporal_probability.size(-1) == 1
        ):
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
            qbound_key = pre_smooth_key

        qbound_key = self._normalize_probability(
            qbound_key,
            valid_mask,
        )

        # --------------------------------------------------------------
        # Legacy V1.5 coarse event-chain span, no [1,3,5] re-selection.
        # --------------------------------------------------------------
        time_param = self.gen_adaptive_grounding(
            qbound_key,
            selected_span_start_frame=selected_span_start_frame,
            selected_span_end_frame=selected_span_end_frame,
            valid_mask=valid_mask,
            selected_frame_mask=selected_frame_mask,
        )

        # --------------------------------------------------------------
        # Controlled V1.4/CRA-style readout diagnostic on the SAME key.
        # --------------------------------------------------------------
        legacy_time = self.gen_grounding(
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

        time_param["legacy_fixed_window_key"] = legacy_time["key"]
        time_param["legacy_fixed_window_max_indices"] = legacy_time[
            "max_indices"
        ]
        time_param["legacy_fixed_window_start"] = legacy_time["start"]
        time_param["legacy_fixed_window_end"] = legacy_time["end"]
        time_param["legacy_fixed_window_mask"] = legacy_time["mask"]
        time_param["legacy_fixed_window_span_length"] = (
            legacy_time["mask"].sum(dim=-1).to(qbound_key.dtype)
        )

        adaptive_length = time_param["adaptive_span_length"].clamp_min(1.0)
        legacy_length = time_param[
            "legacy_fixed_window_span_length"
        ].clamp_min(1.0)
        time_param["adaptive_vs_legacy_span_length_ratio"] = (
            adaptive_length / legacy_length
        )

        # Useful direct diagnostic for the bottleneck observed in V1.4:
        # how much probability mass is captured by each interval extractor?
        time_param["legacy_fixed_window_key_mass"] = (
            qbound_key
            * legacy_time["mask"].to(qbound_key.dtype)
        ).sum(dim=-1)
        time_param["adaptive_minus_legacy_key_mass"] = (
            time_param["adaptive_span_key_mass"]
            - time_param["legacy_fixed_window_key_mass"]
        )

        return time_param


    def finalize_refined_grounding(
        self,
        temporal_probability,
        refined_span_start_frame,
        refined_span_end_frame,
        valid_mask=None,
        refined_frame_mask=None,
        coarse_span_start_frame=None,
        coarse_span_end_frame=None,
        coarse_frame_mask=None,
        window_sizes=None,
        detach_sigma=True,
        apply_smoothing=True,
    ):
        """
        Main Q-Bound V1.6 coarse-to-fine grounding adapter.

        Inputs
        ------
        temporal_probability:
            QBoundCore["refined_temporal_prob"] / ["main_temporal_prob"],
            shape [B,T].
        refined_span_start_frame / refined_span_end_frame:
            Detached frame-level boundaries returned by QBoundCore REFINE.
        refined_frame_mask:
            Optional detached [B,T] refined support mask.
        coarse_span_start_frame / coarse_span_end_frame / coarse_frame_mask:
            Optional V1.5 Composer span. These are diagnostic only.

        Main path
        ---------
            refined temporal probability
                -> normalize under valid mask
                -> probability-preserving Gaussian smoothing
                -> qbound_key used for grounded video pooling

            refined frame boundaries
                -> final V1.6 start/end/mask used for pre-hoc grounding

        Controlled diagnostics on the SAME qbound_key
        ---------------------------------------------
            * coarse V1.5 event-chain interval
            * legacy CRA [1,3,5] fixed-window interval

        There is deliberately no extra temporal softmax in this adapter.
        """
        if (
            temporal_probability.ndim == 3
            and temporal_probability.size(-1) == 1
        ):
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
            qbound_key = pre_smooth_key

        qbound_key = self._normalize_probability(
            qbound_key,
            valid_mask,
        )

        # ----------------------------------------------------------
        # V1.6 MAIN: refined frame-level interval.
        # ----------------------------------------------------------
        refined_time = self.gen_adaptive_grounding(
            qbound_key,
            selected_span_start_frame=refined_span_start_frame,
            selected_span_end_frame=refined_span_end_frame,
            valid_mask=valid_mask,
            selected_frame_mask=refined_frame_mask,
        )

        # Preserve the generic GroundingModule contract.
        time_param = dict(refined_time)

        # Rename/add explicit V1.6 semantics without removing compatibility
        # aliases such as adaptive_span_length.
        time_param["refined_span_used"] = torch.ones(
            (),
            dtype=qbound_key.dtype,
            device=qbound_key.device,
        )
        time_param["refined_span_start_frame"] = refined_time[
            "adaptive_span_start_frame"
        ]
        time_param["refined_span_end_frame"] = refined_time[
            "adaptive_span_end_frame"
        ]
        time_param["refined_span_length"] = refined_time[
            "adaptive_span_length"
        ]
        time_param["refined_span_fraction"] = refined_time[
            "adaptive_span_fraction"
        ]
        time_param["refined_span_key_mass"] = refined_time[
            "adaptive_span_key_mass"
        ]
        time_param["refined_span_peak_in_span"] = refined_time[
            "adaptive_span_peak_in_span"
        ]
        time_param["refined_span_core_mask"] = refined_time[
            "adaptive_span_composer_mask"
        ]
        time_param["refined_span_mask_consistency"] = refined_time[
            "adaptive_span_mask_consistency"
        ]

        # ----------------------------------------------------------
        # V1.5 coarse Composer diagnostic on the SAME V1.6 key.
        # ----------------------------------------------------------
        coarse_available = (
            coarse_span_start_frame is not None
            and coarse_span_end_frame is not None
        )

        if coarse_available:
            coarse_time = self.gen_adaptive_grounding(
                qbound_key,
                selected_span_start_frame=coarse_span_start_frame,
                selected_span_end_frame=coarse_span_end_frame,
                valid_mask=valid_mask,
                selected_frame_mask=coarse_frame_mask,
            )

            time_param["coarse_v15_start"] = coarse_time["start"]
            time_param["coarse_v15_end"] = coarse_time["end"]
            time_param["coarse_v15_mask"] = coarse_time["mask"]
            time_param["coarse_v15_max_indices"] = coarse_time[
                "max_indices"
            ]
            time_param["coarse_v15_span_start_frame"] = coarse_time[
                "adaptive_span_start_frame"
            ]
            time_param["coarse_v15_span_end_frame"] = coarse_time[
                "adaptive_span_end_frame"
            ]
            time_param["coarse_v15_span_length"] = coarse_time[
                "adaptive_span_length"
            ]
            time_param["coarse_v15_span_fraction"] = coarse_time[
                "adaptive_span_fraction"
            ]
            time_param["coarse_v15_span_key_mass"] = coarse_time[
                "adaptive_span_key_mass"
            ]
            time_param["coarse_v15_span_peak_in_span"] = coarse_time[
                "adaptive_span_peak_in_span"
            ]
            time_param["coarse_v15_span_mask_consistency"] = coarse_time[
                "adaptive_span_mask_consistency"
            ]
        else:
            zeros = torch.zeros(
                batch_size,
                dtype=qbound_key.dtype,
                device=qbound_key.device,
            )
            false_mask = torch.zeros(
                batch_size,
                time_steps,
                dtype=torch.bool,
                device=qbound_key.device,
            )
            time_param["coarse_v15_start"] = zeros
            time_param["coarse_v15_end"] = zeros
            time_param["coarse_v15_mask"] = false_mask
            time_param["coarse_v15_max_indices"] = torch.zeros(
                batch_size,
                2,
                dtype=torch.long,
                device=qbound_key.device,
            )
            time_param["coarse_v15_span_start_frame"] = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=qbound_key.device,
            )
            time_param["coarse_v15_span_end_frame"] = torch.zeros(
                batch_size,
                dtype=torch.long,
                device=qbound_key.device,
            )
            time_param["coarse_v15_span_length"] = zeros
            time_param["coarse_v15_span_fraction"] = zeros
            time_param["coarse_v15_span_key_mass"] = zeros
            time_param["coarse_v15_span_peak_in_span"] = zeros
            time_param["coarse_v15_span_mask_consistency"] = zeros

        # ----------------------------------------------------------
        # CRA/V1.4 fixed short-window diagnostic on the SAME V1.6 key.
        # ----------------------------------------------------------
        legacy_time = self.gen_grounding(
            qbound_key,
            window_sizes=window_sizes,
        )

        time_param["legacy_fixed_window_key"] = legacy_time["key"]
        time_param["legacy_fixed_window_max_indices"] = legacy_time[
            "max_indices"
        ]
        time_param["legacy_fixed_window_start"] = legacy_time["start"]
        time_param["legacy_fixed_window_end"] = legacy_time["end"]
        time_param["legacy_fixed_window_mask"] = legacy_time["mask"]
        time_param["legacy_fixed_window_span_length"] = (
            legacy_time["mask"].sum(dim=-1).to(qbound_key.dtype)
        )
        time_param["legacy_fixed_window_key_mass"] = (
            qbound_key
            * legacy_time["mask"].to(qbound_key.dtype)
        ).sum(dim=-1)

        # ----------------------------------------------------------
        # Direct coarse-to-fine diagnostics.
        # ----------------------------------------------------------
        refined_length = time_param["refined_span_length"].clamp_min(1.0)
        coarse_length = time_param[
            "coarse_v15_span_length"
        ].clamp_min(1.0)
        legacy_length = time_param[
            "legacy_fixed_window_span_length"
        ].clamp_min(1.0)

        time_param["refined_vs_coarse_span_length_ratio"] = (
            refined_length / coarse_length
        )
        time_param["refined_vs_legacy_span_length_ratio"] = (
            refined_length / legacy_length
        )
        time_param["coarse_vs_legacy_span_length_ratio"] = (
            coarse_length / legacy_length
        )

        time_param["refined_minus_coarse_key_mass"] = (
            time_param["refined_span_key_mass"]
            - time_param["coarse_v15_span_key_mass"]
        )
        time_param["refined_minus_legacy_key_mass"] = (
            time_param["refined_span_key_mass"]
            - time_param["legacy_fixed_window_key_mass"]
        )

        # How much of the V1.5 coarse interval survived REFINE.
        time_param["refined_retained_fraction_of_coarse"] = (
            refined_length / coarse_length
        )
        time_param["refined_trimmed_frame_count"] = (
            time_param["coarse_v15_span_length"]
            - time_param["refined_span_length"]
        ).clamp_min(0.0)

        # Smoothing / key contract.
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

    def finalize_probability_grounding(
        self,
        temporal_probability,
        valid_mask=None,
        window_sizes=None,
        detach_sigma=True,
        apply_smoothing=True,
    ):
        """
        Legacy Q-Bound V1.4 grounding directly from temporal probabilities.

        Expected main input:
            QBoundCore["event_temporal_prob"]      [B,T]

        Compatibility input:
            QBoundCore["legacy_v14_event_temporal_prob"] or any [B,T] prob.

        Pipeline:
            probability
            -> normalize under valid mask
            -> Gaussian smooth_probability()
            -> sum-normalized qbound_key
            -> legacy CRA interval extraction (gen_grounding)

        There is deliberately:
            * no CRA temporal-logit addition,
            * no alpha/beta gate,
            * no CRA->Q-Bound handover,
            * no second temporal softmax,
            * no Q-Bound gradient into CRA sigma when detach_sigma=True.

        Therefore the only normalization after QBoundCore is sum
        normalization before/after probability-preserving Gaussian smoothing.
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
        Q-Bound V1.6 main path must use ``finalize_refined_grounding()`` instead.

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

        Q-Bound V1.6 does not use this convenience forward for its main
        temporal path. It calls ``finalize_refined_grounding()`` directly.
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
