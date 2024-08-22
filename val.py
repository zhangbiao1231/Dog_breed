# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""
Validate a trained YOLOv5 classification model on a classification dataset.

Usage:
    $ bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
    $ python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate ImageNet

Usage - formats:
    $ python classify/val.py --weights yolov5s-cls.pt                 # PyTorch
                                       yolov5s-cls.torchscript        # TorchScript
                                       yolov5s-cls.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                       yolov5s-cls_openvino_model     # OpenVINO
                                       yolov5s-cls.engine             # TensorRT
                                       yolov5s-cls.mlmodel            # CoreML (macOS-only)
                                       yolov5s-cls_saved_model        # TensorFlow SavedModel
                                       yolov5s-cls.pb                 # TensorFlow GraphDef
                                       yolov5s-cls.tflite             # TensorFlow Lite
                                       yolov5s-cls_edgetpu.tflite     # TensorFlow Edge TPU
                                       yolov5s-cls_paddle_model       # PaddlePaddle
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.dataloaders import create_classification_dataloader
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    TEXT_LABELS,
    Profile,
    colorstr,
    intersect_dicts,
    increment_path,
    print_args,
    save_to_csv,
)
from utils.torch_utils import select_device, smart_inference_mode
from models.get_Net import get_net


@smart_inference_mode()
def run(
        data="",  # dataset dir
        weights="",  # model.pt path(s)
        batch_size=64,  # batch size
        imgsz=224,  # inference size (pixels)
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        workers=8,  # max dataloader workers (per RANK in DDP mode)
        verbose=False,  # verbose output
        project=ROOT / "runs/val-cls",   # save to project/name
        name="exp",  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        criterion=None,
        pbar=None,
        is_test=False,
):
    """Validates a resnet classification model on a dataset, computing metrics like top1 and top5 accuracy."""
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        half &= device.type != "cpu"  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        if is_test:
            project = ROOT / "runs/test-cls"
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir
        csv = save_dir/"results.csv"

        # Load model
        # model = nn.ModuleList()
        # file = Path(str(weights).strip().replace("'", ""))
        # ckpt = torch.load(file, map_location="cpu")  # load
        # ckpt = (ckpt["model"]).to(device).float()  # FP32 model
        # model.append(ckpt.eval())
        # model = model[-1]

        # Load model
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak
        model = get_net(name="resnet34").to(device)  # create
        exclude = []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report

        # Dataloader
        data = Path(data)#
        valid_dir = data / "valid" if not is_test else data / "test"  # data/test or data/val
        dataloader = create_classification_dataloader(
            path=valid_dir,
            imgsz=imgsz,
            batch_size=batch_size,
            augment=False,
            rank=-1,
            workers=workers,
            shuffle=False,
        )
    model.eval()
    pred, targets, loss, dt = [], [], 0, (Profile(device=device), Profile(device=device), Profile(device=device))
    n = len(dataloader)  # number of batches
    action = "validating" if dataloader.dataset.root.stem == "valid" else "testing"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)
    with torch.cuda.amp.autocast(enabled=device.type != "cpu"):
        for images, labels in bar:
            with dt[0]:
                images, labels = images.to(device, non_blocking=True), labels.to(device)  # (64,3,224,224) (64,)
            # Inference
            with dt[1]:
                y = model(images)  # (64,120)

            with dt[2]:
                if not is_test:
                    pred.append(y.argsort(1, descending=True)[:, :5])  # (11,64,5) +（1，16，5）
                    targets.append(labels)  # (11,64) +(1,16)
                    if criterion:
                        loss += criterion(y, labels).sum() / len(labels)  # (64,120) (64,) #最后一个batch 是16，不是64
                else:
                    pred.extend(F.softmax(y, dim=1).cpu().detach().numpy())
    if not is_test:
        loss /= n
        pred, targets = torch.cat(pred), torch.cat(targets)  # preds.shape ->(11*64+1*16,5) = (720,5) ;#targets.shape ->(11*64+1*16,) = (720,)
        correct = (targets[:, None] == pred).float()  # (720,5)
        acc = torch.stack((correct[:, 0], correct.max(1).values),
                          dim=1)  # (top1, top5) accuracy ;acc.shape -> (720,2) 第一列是top1,第二列是top5
        top1, top5 = acc.mean(0).tolist()  # 按dim0求均值并分解成两个数据列表，存储top1和top5数据
        if pbar:
            pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
        if verbose:  # all classes
            LOGGER.info(f"{'Class':>24}{'Images':>12}{'top1_acc':>12}{'top5_acc':>12}")
            LOGGER.info(f"{'all':>24}{targets.shape[0]:>12}{top1:>12.3g}{top5:>12.3g}")
            for i, c in enumerate(TEXT_LABELS):  # model.names 其实就是列表 [0:dog1,1:dog2,...,120]
                acc_i = acc[targets == i]
                top1i, top5i = acc_i.mean(0).tolist()
                LOGGER.info(f"{c:>24}{acc_i.shape[0]:>12}{top1i:>12.3g}{top5i:>12.3g}")

            # Log
                metrics = {
                    "Labels": c,
                    "Images": acc_i.shape[0],
                    "metrics/accuracy_top1": top1i,
                    "metrics/accuracy_top5": top5i,
                }  # learning rate

                keys, vals = list(metrics.keys()), list(metrics.values())
                n = len(metrics) +1# number of cols
                s = "" if csv.exists() else (("%30s," * n % tuple(["Classes"]+keys)).rstrip(",") + "\n")  # header
                with open(csv, "a") as f:
                    f.write(s + ("%30.5g," % i) +
                            ("%30s," % vals[0]) +
                            ("%30.5g," * (n-2) % tuple(vals[1:])).rstrip(",") + "\n")
        return top1, top5, loss
    else:
        save_to_csv(valid_dir / "unknown",
                    csv,
                    pred,
                    TEXT_LABELS)
    # Print results
    t = tuple(x.t / len(dataloader.dataset.samples) * 1e3 for x in dt)  # speeds per image
    shape = (1, 3, imgsz, imgsz)
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms post-process per image at shape {shape}" % t)
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}")


def parse_opt():
    """Parses and returns command line arguments for YOLOv5 model evaluation and inference settings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/datasets/dog-breed-identification",
                        help="dataset path")
    # parser.add_argument("--model", type=str, default="resnet34", help="initial weights path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "runs/train-cls/resnet34-resume-test11/weights/last.pt",
                        help="model.pt path(s)")
    parser.add_argument("--batch-size", type=int, default=64, help="batch size")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="inference size (pixels)")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--verbose", nargs="?", const=True, default=True, help="verbose output")
    parser.add_argument("--project", default=ROOT / "runs/val-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    # parser.add_argument("--dnn", action="store_true", help="use OpenCV DNN for ONNX inference")
    parser.add_argument("--is-test", default=False, help="")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes the YOLOv5 model prediction workflow, handling argument parsing and requirement checks."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
