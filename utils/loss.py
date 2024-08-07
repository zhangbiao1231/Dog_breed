# import torch
import torch.nn as nn

loss = nn.CrossEntropyLoss(reduction="none")

def evaluate_loss(data_iter,net,device):
    l_sum , n = 0.0 , 0.0
    for features,labels in data_iter:
        features,labels = features.to(device),labels.to(device)
        outputs = net(features)
        l = loss(outputs,labels)
        l_sum += l.sum()
        n += labels.numel()
    return (l_sum/n).to('cpu')