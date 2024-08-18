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
import torch.nn as nn
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
# from models.yolo import ClassificationModel, DetectionModel
from utils.dataloaders import create_classification_dataloader

from utils.general import (
    DATASETS_DIR,
    LOGGER,
    TQDM_BAR_FORMAT,
    WorkingDirectory,
    TEXT_LABELS,
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
    de_parallel,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def train(opt,device):
    """Trains a dog-greed classify model, managing datasets, model optimization, logging, and saving checkpoints."""
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    save_dir,data,bs,epochs,nw,imgsz,pretrained,is_train= (
        opt.save_dir,
        Path(opt.data),
        opt.batch_size,
        opt.epochs,
        min(os.cpu_count() - 1, opt.workers),
        opt.imgsz,
        str(opt.pretrained).lower() == "true",
        opt.is_train,
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
            else:
                url = f"http://d2l-data.s3-accelerate.amazonaws.com/{data}.zip"
                download(url, dir=data_dir.parent)
            s = f"Dataset download success ✅ ({time.time() - t:.1f}s), saved to {colorstr('bold', data_dir)}\n"
            LOGGER.info(s)

    # Dataloaders
    # train_dir = data_dir / "train"
    nc = len(TEXT_LABELS)  # number of classes
    train_dir = data_dir / "train" if not is_train else data_dir / "train_valid"
    trainloader = create_classification_dataloader(
        path=train_dir,
        imgsz=imgsz,
        batch_size=bs // WORLD_SIZE,
        augment=True,
        cache=opt.cache,
        rank=LOCAL_RANK,
        workers=nw,
        shuffle=True,
    )

    valid_dir = data_dir / "valid"
    if RANK in {-1, 0}:
        validloader = create_classification_dataloader(
            path=valid_dir,
            imgsz=imgsz,
            batch_size=bs // WORLD_SIZE *2,
            augment=False,
            cache=opt.cache,
            rank=-1,
            workers=nw,
            shuffle=False,
        )

    # model
    with torch_distributed_zero_first(LOCAL_RANK), WorkingDirectory(ROOT):
        if Path(opt.model).is_file() or opt.model.endswith(".pt"): #TODO 需要修改attempt_load 函数
            model = attempt_load(opt.model, device="cpu", fuse=False)
        elif opt.model in torchvision.models.__dict__:  # TorchVision models i.e. resnet50, efficientnet_b0
            finetune_net = nn.Sequential()
            finetune_net.features = torchvision.models.__dict__[opt.model](weights="DEFAULT" if pretrained else None)
        else:
            m = hub.list("ultralytics/yolov5")  # + hub.list('pytorch/vision')  # models
            raise ModuleNotFoundError(f"--model {opt.model} not found. Available models are: \n" + "\n".join(m))
        finetune_net.output_new = nn.Sequential(nn.Linear(1000, 256),
                                                nn.ReLU(),
                                                nn.Linear(256, 120))
    # Freeze
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    net = finetune_net.to(device)

    # Optimizer
    optimizer = smart_optimizer(model=net,name=opt.optimizer,
                                lr=opt.lr0,momentum=0.9,
                                decay=opt.decay)
    # Scheduler
    lrf = 0.01  # final lr (fraction of lr0)
    # lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
    def lf(x):
        """Linear learning rate scheduler function, scaling learning rate from initial value to `lrf` over `epochs`."""
        return (1 - x / epochs) * (1 - lrf) + lrf  # linear
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) #余弦退火
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_period, opt.lr_decay)

    # Train
    t0 = time.time()
    # criterion = smartCrossEntropyLoss(label_smoothing=opt.label_smoothing)  # loss function
    criterion = nn.CrossEntropyLoss(reduction="none")
    best_fitness = 0.0
    # scaler = amp.GradScaler(enabled=cuda)
    val = valid_dir.stem  # 'valid' or 'test'
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} test\n'
        f'Using {nw * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting {opt.model} training on {data} dataset with {nc} classes for {epochs} epochs...\n\n'
        f"{'Epoch':>10}{'GPU_mem':>10}{'train_loss':>12}{f'{val}_loss':>12}{'top1_acc':>12}{'top5_acc':>12}"
    )
    for epoch in range(epochs):  # loop over the dataset multiple times
        tloss, vloss, fitness = 0.0, 0.0, 0.0  # train loss, val loss, fitness
        net.train()
        pbar = enumerate(trainloader)
        if RANK in {-1, 0}:
            pbar = tqdm(enumerate(trainloader), total=len(trainloader), bar_format=TQDM_BAR_FORMAT)
        for i, (images, labels) in pbar:  # progress bar
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            optimizer.zero_grad()

            # Forward
            l = criterion(net(images), labels).sum()

            # Backward
            l.backward()

            # Optimize
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()

            if RANK in {-1, 0}:
                # Print
                tloss = (tloss * i + l.item()/labels.shape[0]) / (i + 1)  # update mean losses
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 36
                # validate #
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate.run(
                        model=net, dataloader=validloader, criterion=criterion, pbar=pbar
                    )  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy
        # Scheduler
        scheduler.step()

        # Log metrics
        if RANK in {-1, 0}:
            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Log
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss,
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]["lr"],
            }  # learning rate
            logger.log_metrics(metrics, epoch)

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(net).half(),  # deepcopy(de_parallel(model)).half(),
                    "ema": None,  # deepcopy(ema.ema).half(),
                    # "updates": ema.updates,
                    "optimizer": None,  # optimizer.state_dict(),
                    "opt": vars(opt),
                    # "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

    # Train complete
    if RANK in {-1, 0} and final_epoch:
        LOGGER.info(
            f'\nTraining complete ({(time.time() - t0) / 3600:.3f} hours)'
            f"\nResults saved to {colorstr('bold', save_dir)}"
            f'\nPredict:         python classify/predict.py --weights {best} --source im.jpg'
            f'\nValidate:        python classify/val.py --weights {best} --data {data_dir}'
            f'\nExport:          python export.py --weights {best} --include onnx'
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{best}')"
            f'\nVisualize:       https://netron.app\n'
        )

        # Plot examples
        images, labels = (x[:25] for x in next(iter(validloader)))  # first 25 images and labels
        pred = torch.max(net(images.to(device)), 1)[1]

        file = imshow_cls(images, labels, pred, names=TEXT_LABELS, verbose=False, f=save_dir / "validimages.jpg") #TODO names需要修改

        # Log results
        meta = {"epochs": epochs, "top1_acc": best_fitness, "date": datetime.now().isoformat()}
        logger.log_images(file, name="Test Examples (true-predicted)", epoch=epoch) #TODO 需要调试，保证tn可用
        # logger.log_model(best, epochs, metadata=meta)
        logger.log_metrics()



def parse_opt(known=False):
    """Parses command line arguments for YOLOv5 training including model path, dataset, epochs, and more, returning
    parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default= "resnet34", help="initial weights path")
    parser.add_argument("--data", type=str, default="dog-breed-identification", help="kaggle_cifar10_tiny, kaggle_dog_tiny，banana-detection, VOCtrainval_11-May-2012, , ...")
    parser.add_argument("--epochs", type=int, default=10, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="total batch size for all GPUs")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=224, help="train, val image size (pixels)")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help='--cache images in "ram" (default) or "disk"')
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train-cls", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--pretrained", nargs="?", const=True, default=True, help="start from i.e. --pretrained False")
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW", "RMSProp"], default="SGD", help="optimizer")
    parser.add_argument("--lr0", type=float, default=1e-3, help="initial learning rate")
    parser.add_argument("--lr_period", type=int, default=10, help="learning rate period")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="learning rate * decay over period per")
    parser.add_argument("--decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing epsilon")
    parser.add_argument("--cutoff", type=int, default=None, help="Model layer cutoff index for Classify() head")
    parser.add_argument("--dropout", type=float, default=None, help="Dropout (fraction)")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    parser.add_argument("--is_trian", default=False, help="")
    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    """Executes YOLOv5 training with given options, handling device setup and DDP mode; includes pre-training checks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        # check_requirements(ROOT / "requirements.txt")
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


