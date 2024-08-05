# Zebulon zhang dog-greed, 1.0 license
"""Image augmentation functions."""

import torch
import torchvision

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]) # RGB mean
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]) # RGB standard deviation

def classify_augmentations(
    is_Train=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD
):
    # dog-greed classification Argumentations
    """Sets up and returns Argumentations transforms for dog-greed classification tasks depending on augmentation
    settings.
    """
    if is_Train:  # Resize and crop for train set
        T = [torchvision.transforms.RandomResizedCrop(size=size,
                                                      scale=scale,
                                                      ratio=ratio),
             torchvision.transforms.RandomHorizontalFlip()]
        color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
        T += [torchvision.transforms.ColorJitter(*color_jitter,hue=0)]
    else:  # Use fixed crop for eval set (reproducibility)
        T = [torchvision.transforms.Resize(256),
             torchvision.transforms.CenterCrop(size = size)]
    T += [torchvision.transforms.ToTensor(),
          torchvision.transforms.Normalize(mean=mean, std=std)]  # Normalize and convert to Tensor
    return torchvision.transforms.Compose(T)