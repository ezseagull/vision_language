"""Run a VideoQA experiment from its YAML, using the installed pipeline registry."""

import argparse
import inspect
import os
from pathlib import Path

import yaml


def main(args):
    cfg_path = Path(args.cfg).expanduser().resolve()
    with cfg_path.open(encoding="utf-8") as stream:
        cfgs = yaml.safe_load(stream)
    if not isinstance(cfgs, dict):
        raise ValueError(f"Expected a YAML mapping in {cfg_path}.")
    for key in ("misc", "optim", "model"):
        if not isinstance(cfgs.get(key), dict):
            raise ValueError(f"Expected a '{key}' mapping in {cfg_path}.")

    misc = cfgs["misc"]
    if args.cuda not in (None, ""):
        misc["cuda"] = args.cuda
    if args.R:
        misc["running_name"] = args.R

    # Match Base's contract: a non-null checkpoint is for frozen evaluation.
    # Base still validates the actual value and loads the checkpoint.
    stat = cfgs.get("stat")
    frozen_m2_eval = isinstance(stat, dict) and stat.get("m2_eval_checkpoint") is not None
    run_infer = args.infer or frozen_m2_eval
    if frozen_m2_eval:
        mode = "infer (stat.m2_eval_checkpoint)"
    else:
        mode = "infer (--infer)" if args.infer else "train"

    # Set device visibility before importing anything that can load PyTorch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(misc.get("cuda", "0"))

    from pipeline import pipeline_fns

    pipeline_key = cfgs["optim"].get("pipeline")
    if not isinstance(pipeline_key, str) or pipeline_key not in pipeline_fns:
        choices = ", ".join(sorted(pipeline_fns))
        raise ValueError(f"Unknown optim.pipeline={pipeline_key!r}. Available: {choices}")
    pipeline_cls = pipeline_fns[pipeline_key]
    source = inspect.getsourcefile(pipeline_cls)
    model = cfgs["model"]
    qbound = model.get("qbound", {})
    if not isinstance(qbound, dict):
        raise ValueError("model.qbound must be a YAML mapping when specified.")
    # This value is for diagnostics only. The pipeline owns compatibility
    # checks. Do not rewrite the YAML version or maintain a version list here.
    version = qbound.get("version", model.get("qbound_version", model.get("gpeb_version")))

    print(f"[VideoQA] YAML: {cfg_path}", flush=True)
    print(f"[VideoQA] optim.pipeline={pipeline_key!r} -> "
          f"{pipeline_cls.__module__}.{pipeline_cls.__name__}", flush=True)
    print(f"[VideoQA] Pipeline source: {Path(source).resolve() if source else '<unknown>'}", flush=True)
    print(f"[VideoQA] Q-Bound version from YAML: {version!r}", flush=True)
    print(f"[VideoQA] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}", flush=True)
    print(f"[VideoQA] Mode: {mode}", flush=True)
    if args.check_entrypoint:
        return  # Report imports only. No model construction or W&B run.

    import numpy as np
    import wandb
    from utils.misc import setup_seed

    if misc["seed"] == -1:
        misc["seed"] = int(np.random.randint(0, 23333))
    setup_seed(misc["seed"])

    # Initialize the pipeline first. Invalid version/configuration should
    # fail with the original traceback without creating an empty W&B run.
    pipeline = pipeline_cls(cfgs)
    wandb.init(
        entity="chinhez",
        project="CRA-GQA",
        name=misc["running_name"],
        config=cfgs,
    )
    if run_infer:
        return pipeline.infer()
    return pipeline.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="config/CRA/CRA_NextGQA.yml")
    parser.add_argument("--cuda", default=None, help="Override misc.cuda from YAML.")
    parser.add_argument("-R", default="", help="Override misc.running_name from YAML.")
    parser.add_argument(
        "--infer", action="store_true",
        help="Run inference instead of training. Automatic when stat.m2_eval_checkpoint is set.",
    )
    parser.add_argument("--check-entrypoint", action="store_true",
                        help="Print YAML version and loaded pipeline path without building the model.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
