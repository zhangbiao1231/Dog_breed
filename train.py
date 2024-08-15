# dog-greed 🐶, 1.0.0 license
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
    ModelEMA,
    de_parallel,
    model_info,
    reshape_classifier_output,
    select_device,
    smart_DDP,
    smart_optimizer,
    smartCrossEntropyLoss,
    torch_distributed_zero_first,
)

def train(net, train_iter, valid_iter, opt,cfg):
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
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=ROOT / "data/datasets/train_valid_test", help="dataset path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--epochs", type=int, default=128, help="total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="total batch size")
    parser.add_argument("--is_Full", type=bool, default=False, help="")
    parser.add_argument("--lr", type=float, default=1e-4, help="")
    parser.add_argument("--wd", type=float, default=1e-4, help="")
    parser.add_argument("--lr_period", type=int, default=2, help="")
    parser.add_argument("--lr_decay", type=float, default=0.9, help="")
    parser.add_argument("--export_pt_filename", type=str, default='resnet34.pt', help="export .pt filename")

    parser.add_argument("--project", default=ROOT / "outputs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")

    #parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    opt = parser.parse_args()
    return opt
def main(opt):
    # print(opt.epochs)

    if not opt.is_Full:
        train_iter = dataLoader(cfg=cfg,
                                data_dir=opt.data_dir,
                                batch_size=opt.batch_size,
                                folder= 'train',
                                is_Train=True,
                                is_Test=False)
        valid_iter = dataLoader(cfg=cfg,
                                data_dir=opt.data_dir,
                                batch_size=opt.batch_size,
                                folder= 'valid',
                                is_Train=False,
                                is_Test=False)
    else:
        train_iter = dataLoader(cfg=cfg,
                                data_dir=opt.data_dir,
                                batch_size=opt.batch_size,
                                folder='train_valid',
                                is_Train=True,
                                is_Test=False)
        valid_iter = None
    device = cfg.MODEL.DEVICE
    net = get_net(cfg).to(device)
    # print(valid_iter)

    for X, y in train_iter:
        print(X.shape, y.shape)
        break

    X = torch.zeros((opt.batch_size,3,224,224)).to(device)
    print(net(X).shape)

    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    net = train(net, train_iter, valid_iter, opt,cfg)
    return net

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


