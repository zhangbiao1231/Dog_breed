import argparse
import torch
import sys
from pathlib import Path

import logging
import os

from utils.inference import do_evaluation
from utils.config import cfg
from utils import modelLoader

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory#
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def evaluation(cfg,args):
    print(f'========== load model ==========')
    model = modelLoader.load(cfg,args)
    # print(model.device)
    preds = model(torch.zeros((32, 3, 224, 224)))
    print(preds.shape)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)
    print(f'========== inference configuratiton ==========')
    print(f'load_model: {args.load_model_filename}')
    print(f'========== inference start on {device} ==========')
    do_evaluation(cfg,args,model)
def main():
    parser = argparse.ArgumentParser(description='ResNet Evaluation and Dog-breed-classify dataset.')
    parser.add_argument(
        "--load_model_filename",
        default="resnet34.pt",
        type=str,
        help="",
    )
    parser.add_argument("--data_dir", type=str, default=ROOT / "data/datasets/train_valid_test", help="dataset path")
    parser.add_argument("--export_csv_filename", type=str, default='submission.csv', help="export .csv filename")
    args = parser.parse_args()
    # path = ROOT / ''
    evaluation(cfg,args)

if __name__ == '__main__':
    main()


