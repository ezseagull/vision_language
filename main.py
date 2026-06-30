import argparse
from utils.misc import setup_seed, load_file
import numpy as np
import yaml
from pipeline import *
import wandb

def main(args):
    # -------------------------------
    # load hyper-param
    # -------------------------------
    cfgs = load_file(args.cfg)
    if args.cuda != "":
        cfgs["misc"]["cuda"] = args.cuda
    if args.R != "":
        cfgs["misc"]["running_name"] = args.R
    # -------------------------------
    # fix random seeds
    # -------------------------------
    if cfgs["misc"]["seed"] == -1:
        cfgs["misc"]["seed"] = np.random.randint(0, 23333)
    setup_seed(cfgs["misc"]["seed"])

    wandb.init(
        entity="chinhez",  # Tên định danh (username hoặc team) của bạn trên WandB
        project="CRA-GQA",  # Tên Thư mục dự án trên web
        name=cfgs["misc"]["running_name"],  # Tên của riêng biểu đồ lần chạy này
        config=cfgs,  # Bê nguyên toàn bộ siêu tham số từ YAML ném lên cloud
    )
    
    # -------------------------------
    # Run!
    # -------------------------------
    pipeline = pipeline_fns[cfgs['optim']['pipeline']](cfgs)
    if args.infer:
        pipeline.infer()
    else:
        pipeline.train()
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--cfg", type=str, default="config/CRA/CRA_NextGQA.yml")
    parse.add_argument("--cuda", type=str, default="1")
    parse.add_argument("-R", type=str, default="")
    # parse.add_argument("--infer", type=bool, default=False)
    parse.add_argument("--infer", action="store_true", help="Bật chế độ Inference")
    args = parse.parse_args()

    main(args)
    