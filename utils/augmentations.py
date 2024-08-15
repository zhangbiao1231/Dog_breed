# Zebulon zhang dog-greed, 1.0 license
"""Image augmentation functions."""

import torch
import torchvision
# from utils.config import cfg
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]) # RGB mean
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]) # RGB standard deviation

def classify_augmentations(
    augment,
    cfg
):
    # dog-greed classification Argumentations
    """Sets up and returns Argumentations transforms for dog-greed classification tasks depending on augmentation
    settings.
    """
    size   = (cfg.INPUT.IMAGE_SIZE,cfg.INPUT.IMAGE_SIZE)
    scale  = cfg.DATASETS.SCALE
    ratio  = cfg.DATASETS.RATIO
    jitter = cfg.DATASETS.JITTER
    mean   = cfg.DATASETS.IMAGENET_MEAN
    std    = cfg.DATASETS.IMAGENET_STD
    if augment:  # Resize and crop for train set

        T = [torchvision.transforms.RandomResizedCrop(size=size,
                                                      scale=scale,
                                                      ratio=ratio),
             torchvision.transforms.RandomHorizontalFlip()]
        color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
        T += [torchvision.transforms.ColorJitter(*color_jitter,hue=0)]
    else:  # Use fixed crop for eval set (reproducibility)
        T = [torchvision.transforms.Resize(256),
             torchvision.transforms.CenterCrop(size=size)]
    T += [torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=mean, std=std)]  # Normalize and convert to Tensor
    return torchvision.transforms.Compose(T)
def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverses ImageNet normalization for BCHW format RGB images by applying `x = x * std + mean`."""
    for i in range(3):
        x[:, i] = x[:, i] * std[i] + mean[i]
    return x
