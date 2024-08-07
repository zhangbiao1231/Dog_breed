import torch
import os
import torchvision

from utils.augmentations import classify_augmentations

def dataSets(data_dir,folder,augment = True):
    augs = classify_augmentations(augment)
    return  torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=augs)
def dataLoader(dataset,batch_size,shuffle = True,drop_last = True):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle = shuffle, drop_last=drop_last)
