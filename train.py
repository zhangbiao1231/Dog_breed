import os
import sys
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from utils.config import cfg

from utils.dataloaders import dataLoader
from models.get_Net import get_net
from utils.plots import *
from utils.general import Accumulator
from utils.loss import *
from utils.Time import Timer

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory#
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def train(net, train_iter, valid_iter, opt,cfg):
    loss = nn.CrossEntropyLoss(reduction = "none")
    num_epochs = opt.epochs
    device = cfg.MODEL.DEVICE
    lr, wd, lr_period, lr_decay = cfg.SOLVER.LR,cfg.SOLVER.WEIGHT_DECAY,cfg.SOLVER.LR_PERIOD,cfg.SOLVER.LR_DECAY
    trainer = torch.optim.SGD((param for param in net.parameters()
                               if param.requires_grad), lr=lr,
                              momentum=0.9,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_period, lr_decay)
    num_batches, timer = len(train_iter), Timer()
    legend = ['train loss']
    if valid_iter is not None:
        legend.append('valid loss')
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=legend)
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
            valid_loss = evaluate_loss(valid_iter,net,device)
            animator.add(epoch +1,(None,valid_loss.detach().cpu()))
        scheduler.step()
    if valid_iter is not None:
        measures += f', valid loss {valid_loss:.3f}'
    print(measures + f'\n{metric[1] * num_epochs / timer.sum():.1f}'
          f' examples/sec on {str(device)}')
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=ROOT / "data/datasets/train_valid_test", help="dataset path")
    parser.add_argument("--epochs", type=int, default=128, help="total training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="total batch size")
    #parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    opt = parser.parse_args()
    return opt
def main(opt):
    # print(opt.epochs)
    train_iter = dataLoader(cfg, opt,
                            'train',
                            is_Train=True,
                            is_Test=False)
    valid_iter = dataLoader(cfg, opt,
                            'valid',
                            is_Train=False,
                            is_Test=False)
    device = cfg.MODEL.DEVICE
    net = get_net(device).to(device)

    for X, y in train_iter:
        print(X.shape, y.shape)
        break
    batch_size = opt.batch_size
    X = torch.zeros((opt.batch_size,3,224,224))
    print(net(X).shape)

    train(net, train_iter, valid_iter, opt,cfg)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


