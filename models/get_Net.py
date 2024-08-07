import torch.nn as nn
import torchvision



def get_net(device):
    # device = cfg.MODEL.DEVICE
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet34(weights = 'DEFAULT')#加载与预训练模型 weights = None

    #堆叠两个全连接层
    #定义一个新网络,输出类别120
    finetune_net.output_new = nn.Sequential(nn.Linear(1000,256),
                                           nn.ReLU(),
                                           nn.Linear(256,120))
    # 将模型参数分配给用于计算的CPU或GPU
    finetune_net = finetune_net.to(device)
    # 冻结参数
    for param in finetune_net.features.parameters():
        param.requires_grad = False
    return finetune_net
