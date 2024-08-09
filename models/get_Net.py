import torch.nn as nn
import torchvision

def get_net(cfg):
    # device = cfg.MODEL.DEVICE
    pretrained = cfg.MODEL.PRETRAINED
    finetune_net = nn.Sequential()
    if pretrained:
        finetune_net.features = torchvision.models.resnet34(weights='DEFAULT') #加载与预训练模型 weights = None
    else:
        finetune_net.features = torchvision.models.resnet34(weights=None)

    finetune_net.output_new = nn.Sequential(nn.Linear(1000,256),
                                           nn.ReLU(),
                                           nn.Linear(256,120))
    finetune_net.output_new = nn.Sequential(nn.Linear(1000,256),
                                           nn.ReLU(),
                                           nn.Linear(256,120))
    # finetune_net = finetune_net.to(device)
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
