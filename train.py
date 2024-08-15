# dog-breed 🐶, 1.0.0 license
"""
Train a dog-greed classifier model on a classification dataset.

Usage - Single-GPU training:
    $ python classify/train.py --model yolov5s-cls.pt --data imagenette160 --epochs 5 --img 224

Datasets:           --data , tiny-dog, cifar10, or 'path/to/data'
dong-greed-cls models:  --model resnet34-cls.pt
Torchvision models: --model resnet34, vgg19, etc. See https://pytorch.org/vision/stable/models.html
"""
import argparse
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.hub as hub
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
from torch.cuda import amp
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # dog-greed root directory#
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate
from models.experimental import attempt_load
from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader
from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    check_requirements,
    colorstr,
    download,
    increment_path,
    init_seeds,
    print_args,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import imshow_cls
from utils.torch_utils import (
    model_info,
    reshape_classifier_output,
    select_device,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def train(opt,device):
    """Trains a dog-greed classify model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir,data,bs,epochs,nw,imgsz,pretrained = (
        opt.save_dir,
        Path(opt.data),
        opt.batch_size,
        opt.epochs,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
    )
    cuda = device.type != "cpu"

    # Directories
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / "last.pt", wdir / "best.pt"

    # Save run settings
    yaml_save(save_dir / "opt.yaml", vars(opt))

    # Logger
    logger = GenericLogger(opt=opt, console_logger=LOGGER) if RANK in {-1, 0} else None

    # Download Dataset
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        data_dir = data if data.is_dir() else (DATASETS_DIR/data)
        if not data_dir.is_dir():
            LOGGER.info(f"\nDataset not found ⚠️, missing path {data_dir}, attempting download...")
            t = time.time()
            if str(data) == "kaggle_dog_tiny":
                subprocess.run(args=["bash",str(ROOT / "data/scripts/get_tinydog.sh")], shell=True,check=True)
                #TODO reorg 数据集
            else:
                url = f"http://d2l-data.s3-accelerate.amazonaws.com/{data}.zip"
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Dataloaders
    nc = len([x for x in (data_dir / "train").glob("*") if x.is_dir()])  # number of classes
    trainloader = create_classification_dataloader(
        path=data_dir / "train",
        imgsz=imgsz,
        batch_size=bs // WORLD_SIZE,
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=nw,
    )

    test_dir = data_dir / "test" if (data_dir / "test").exists() else data_dir / "valid"  # data/test or data/val
    if RANK in {-1, 0}:
        testloader = create_classification_dataloader(
            path=test_dir,
            imgsz=imgsz,
            batch_size=bs // WORLD_SIZE * 2,
            augment=False,
            cache=opt.cache,
            rank=-1,
            workers=nw,
        )



    loss = nn.CrossEntropyLoss(reduction = "none")
    num_epochs = opt.epochs
    device = cfg.MODEL.DEVICE
    # lr, wd, lr_period, lr_decay = cfg.SOLVER.LR,cfg.SOLVER.WEIGHT_DECAY,cfg.SOLVER.LR_PERIOD,cfg.SOLVER.LR_DECAY
    lr, wd, lr_period, lr_decay = opt.lr,opt.wd,opt.lr_period,opt.lr_decay
    momentum = cfg.SOLVER.MOMENTUM
    print(f'========== training configuratiton ==========')
    print(f'num_epochs: {opt.epochs} \n'
          f'batch_size: {opt.batch_size} \n'
          f'learning_rate: {opt.lr} \n'
          f'weight_decay: {opt.wd} \n'
          f'lr_period: {opt.lr_period} \n'
          f'lr_decay: {lr_decay}')
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
    for X,y in train_iter:
        print(X.shape[0])
        break
    imgsz = X.shape[-1]
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        # f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('green', opt.save_dir)}\n"
        f'Starting training for {num_epochs} epochs...'
        f'Train on {device}'
    )
    # ==============start train================
    for epoch in range(num_epochs):
        metric = Accumulator(2)
        for i,(features,labels) in enumerate(train_iter):
            timer.start()
            features,labels = features.to(device),labels.to(device)
            trainer.zero_grad()
            output = net(features)
            l = loss(output,labels).sum()
            l.backward()
            trainer.step()
            metric.add(l,labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                            (metric[0] / metric[1],None))
        measures = f'train loss {metric[0] / metric[1]:.3f}'
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, device)
            print(f'epoch {epoch + 1}/{num_epochs}:\n'
            f'train loss {metric[0] / metric[1]:.3f}, valid_loss:{valid_loss:.3f}')
        else:
            # animator.add(epoch +1,(None,valid_loss.detach().cpu()))
            print(f'epoch {epoch + 1}/{num_epochs}:\n'
            f'train loss {metric[0] / metric[1]:.3f}')
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(device)}')
    # ==============train done================
    print('========== train done ==========')

    output_folder = os.path.join(cfg.OUTPUT_DIR, "train")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f'export {opt.export_pt_filename}...')
    save_model(model=net,
               save_path=os.path.join(output_folder, opt.export_pt_filename))
    save_model(model=net,
               save_path=os.path.join(opt.save_dir, opt.export_pt_filename))
    return net
def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov5s-cls.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="kaggle_dog_tiny", help="kaggle_cifar10_tiny, banana-detection, VOCtrainval_11-May-2012, , ...")
    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=True, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="Adam", help="optimizer")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--decay", type=float, default=5e-5, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_requirements(ROOT / "requirements.txt")
    device = select_device(opt.device, batch_size=opt.batch_size)
    # Parameters
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run

    # Train
    train(opt, device)
def run(**kwargs):
    """
    Executes YOLOv5 model training or inference with specified parameters, returning updated options.

    Example: from yolov5 import classify; classify.train.run(data=mnist, imgsz=320, model='yolov5m')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


