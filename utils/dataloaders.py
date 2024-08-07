import torch
import os
import torchvision
from pathlib import Path
import sys
from utils.augmentations import classify_augmentations

FILE = Path('dataloaders.py').resolve()
ROOT = FILE.parents[1]  # dog-breed root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))
def dataSets(data_dir,folder,augment = True):
    augs = classify_augmentations(augment)
    return  torchvision.datasets.ImageFolder(
        os.path.join(data_dir, 'train_valid_test', folder),
        transform=augs)
def dataLoader(dataset,batch_size,shuffle = True,drop_last = True):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle = shuffle, drop_last=drop_last)