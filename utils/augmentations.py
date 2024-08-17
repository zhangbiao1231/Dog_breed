# dog-greed, 1.1.0 license
"""Image augmentation functions."""

import math
import random

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from utils.general import LOGGER,colorstr

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

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
def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    ratio=(0.75, 1.0 / 0.75),  # 0.75, 1.33
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
    auto_aug=False,
):
    # YOLOv5 classification Albumentations (optional, only used if package is installed)
    """Sets up and returns Albumentations transforms for YOLOv5 classification tasks depending on augmentation
    settings.
    """
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio)]
            if auto_aug:
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    color_jitter = (float(jitter),) * 3  # repeat value for brightness, contrast, saturation, 0 hue
                    T += [A.ColorJitter(*color_jitter, 0)]
        else:  # Use fixed crop for eval set (reproducibility)
            T = [A.SmallestMaxSize(max_size=size), A.CenterCrop(height=size, width=size)]
        T += [A.Normalize(mean=mean, std=std), ToTensorV2()]  # Normalize and convert to Tensor
        LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        LOGGER.warning(f"{prefix}⚠️ not found, install with `pip install albumentations` (recommended)")
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


def classify_transforms(size=224):
    """Applies a series of transformations including center crop, ToTensor, and normalization for classification."""
    assert isinstance(size, int), f"ERROR: classify_transforms size {size} must be integer, not (list, tuple)"
    # T.Compose([T.ToTensor(), T.Resize(size), T.CenterCrop(size), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return T.Compose([CenterCrop(size), ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
class CenterCrop:
    # YOLOv5 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Initializes CenterCrop for image preprocessing, accepting single int or tuple for size, defaults to 640."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """
        Applies center crop to the input image and resizes it to a specified size, maintaining aspect ratio.

        im = np.array HWC
        """
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(im[top : top + m, left : left + m], (self.w, self.h), interpolation=cv2.INTER_LINEAR)


class ToTensor:
    # YOLOv5 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initializes ToTensor for YOLOv5 image preprocessing, with optional half precision (half=True for FP16)."""
        super().__init__()
        self.half = half

    def __call__(self, im):
        """
        Converts BGR np.array image from HWC to RGB CHW format, and normalizes to [0, 1], with support for FP16 if
        `half=True`.

        im = np.array HWC in BGR order
        """
        im = np.ascontiguousarray(im.transpose((2, 0, 1))[::-1])  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im