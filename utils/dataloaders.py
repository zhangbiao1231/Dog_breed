import torch
import os
import torchvision

from utils.augmentations import classify_augmentations

def make_dataSets(cfg,data_dir,folder,is_Train = True):
    augment = is_Train
    augs = classify_augmentations(augment,cfg)
    # data_dir = cfg.DATASETS.DATA_DIR
    return  torchvision.datasets.ImageFolder(
        os.path.join(data_dir, folder),
        transform=augs)
def dataLoader(cfg,batch_size,data_dir,folder,is_Train = True,is_Test = False):
    dataset = make_dataSets(cfg,data_dir,folder,is_Train)
    shuffle = is_Train
    drop_last = False if is_Test else True
    # batch_size = cfg.SOLVER.BATCH_SIZE
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle = shuffle, drop_last=drop_last)

