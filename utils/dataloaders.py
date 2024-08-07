import torch
import os
import torchvision

from utils.augmentations import classify_augmentations

def dataSets(cfg,opt,folder,is_train = True):
    augment = is_train
    augs = classify_augmentations(augment,cfg)
    # data_dir = cfg.DATASETS.DATA_DIR
    data_dir = opt.data_dir
    return  torchvision.datasets.ImageFolder(
        os.path.join(data_dir, folder),
        transform=augs)
def dataLoader(cfg,opt,folder,is_Train = True,is_Test = False):
    dataset = dataSets(cfg,opt,folder,is_Train)
    shuffle = is_Train
    drop_last = False if is_Test else True
    batch_size = opt.batch_size
    # batch_size = cfg.SOLVER.BATCH_SIZE
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle = shuffle, drop_last=drop_last)

