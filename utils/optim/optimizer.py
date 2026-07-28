import torch
import math 

def build_optimizer(cfgs, model):
    optim_name = cfgs["optim"]["optimizer"]
    weight_decay = cfgs["optim"]["weight_decay"]
    base_lr = cfgs["optim"]["lr"]
    amsgrad = cfgs["optim"].get("amsgrad", False)

    gpeb_lr = cfgs["optim"].get(
        "gpeb_lr",
        base_lr * 10,
    )

    gpeb_alpha_lr = cfgs["optim"].get(
        "gpeb_alpha_lr",
        base_lr * 100,
    )

    base_params = []
    gpeb_params = []
    gpeb_alpha_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        # Check alpha first because its name also contains "gpeb".
        if "gpeb_alpha" in name:
            gpeb_alpha_params.append(parameter)

        elif "gpeb" in name:
            gpeb_params.append(parameter)

        else:
            base_params.append(parameter)

    parameter_groups = []

    if base_params:
        parameter_groups.append({
            "params": base_params,
            "lr": base_lr,
            "group_name": "cra_base",
        })

    if gpeb_params:
        parameter_groups.append({
            "params": gpeb_params,
            "lr": gpeb_lr,
            "group_name": "gpeb",
        })

    if gpeb_alpha_params:
        parameter_groups.append({
            "params": gpeb_alpha_params,
            "lr": gpeb_alpha_lr,
            "weight_decay": 0.0,
            "group_name": "gpeb_alpha",
        })

    optimizer_class = getattr(
        torch.optim,
        optim_name,
    )

    optimizer_kwargs = {
        "weight_decay": weight_decay,
    }

    # Adam and AdamW accept amsgrad.
    if optim_name in ["Adam", "AdamW"]:
        optimizer_kwargs["amsgrad"] = amsgrad

    optimizer = optimizer_class(
        parameter_groups,
        **optimizer_kwargs,
    )

    print("\n[Optimizer Groups]")

    for index, group in enumerate(
        optimizer.param_groups
    ):
        parameter_count = sum(
            parameter.numel()
            for parameter in group["params"]
        )

        print(
            f"Group {index}: "
            f"{group.get('group_name', 'unnamed')} | "
            f"lr={group['lr']:.2e} | "
            f"parameters={parameter_count:,}"
        )

    return optimizer