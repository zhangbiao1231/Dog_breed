# dog-greed 🐶, 1.0.0 license
"""Plotting utils."""
import contextlib
import math
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch

from utils.general import LOGGER, increment_path

# Settings
RANK = int(os.getenv("RANK", -1))
matplotlib.rc("font", **{"size": 11})
matplotlib.use("Agg")  # for writing to files only
def imshow_cls(im, labels=None, pred=None, names=None, nmax=25, verbose=False, f=Path("images.jpg")):
    """Displays a grid of images with optional labels and predictions, saving to a file."""
    from utils.augmentations import denormalize

    names = names or [f"class{i}" for i in range(120)]
    blocks = torch.chunk(
        denormalize(im.clone()).cpu().float(), len(im), dim=0
    )  # select batch index 0, block by channels
    n = min(len(blocks), nmax)  # number of plots
    m = min(8, round(n**0.5))  # 8 x 8 default
    fig, ax = plt.subplots(math.ceil(n / m), m)  # 8 rows x n/8 cols
    ax = ax.ravel() if m > 1 else [ax]
    # plt.subplots_adjust(wspace=0.05, hspace=0.05)
    for i in range(n):
        ax[i].imshow(blocks[i].squeeze().permute((1, 2, 0)).numpy().clip(0.0, 1.0))
        ax[i].axis("off")
        if labels is not None:
            s = f"true: {names[labels[i]]}" +"\n"+ (f"pred: {names[pred[i]]}" if pred is not None else "")
            ax[i].set_title(s, fontsize=8, verticalalignment="top")
    plt.savefig(f, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        LOGGER.info(f"Saving {f}")
        if labels is not None:
            LOGGER.info("True:     " + " ".join(f"{names[i]:3s}" for i in labels[:nmax]))
        if pred is not None:
            LOGGER.info("Predicted:" + " ".join(f"{names[i]:3s}" for i in pred[:nmax]))
    return f