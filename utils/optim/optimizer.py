import torch


def build_optimizer(cfgs, model):
    """
    Build the repository's base optimizer.

    Q-Bound V1.1 parameter grouping is intentionally handled later by
    BasePipeline._configure_qbound_optimizer_groups().  This function therefore
    creates only one initial group containing every trainable parameter.

    Keeping the responsibility in one place avoids stale GPEB-specific logic
    and guarantees that the final optimizer groups are:

        base / CRA parameters : optim.lr
        Q-Bound parameters    : optim.qbound_lr

    The Q-Bound group is created before the LR scheduler is built.
    """
    optim_cfg = cfgs["optim"]

    optim_name = optim_cfg["optimizer"]
    base_lr = float(optim_cfg["lr"])
    weight_decay = float(
        optim_cfg.get("weight_decay", 0.0)
    )
    amsgrad = bool(
        optim_cfg.get("amsgrad", False)
    )

    trainable_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_params:
        raise RuntimeError(
            "No trainable model parameters were found."
        )

    parameter_groups = [
        {
            "params": trainable_params,
            "lr": base_lr,
            "name": "base_0",
        }
    ]

    optimizer_class = getattr(
        torch.optim,
        optim_name,
    )

    optimizer_kwargs = {
        "weight_decay": weight_decay,
    }

    if optim_name in {"Adam", "AdamW"}:
        optimizer_kwargs["amsgrad"] = amsgrad

    optimizer = optimizer_class(
        parameter_groups,
        **optimizer_kwargs,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in trainable_params
    )

    print("\n[Initial Optimizer Group]")
    print(
        "Group 0: base_0 | "
        f"lr={base_lr:.2e} | "
        f"parameters={parameter_count:,}"
    )

    return optimizer
