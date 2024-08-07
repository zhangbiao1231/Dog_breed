
import argparse
import csv
import os
import platform
import sys
import torch


from utils.downloads import read_csv_labels
from pathlib import Path
from utils.dataloaders import(
    dataSets,
    dataLoader,
)

FILE = Path('main.py').resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory#
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

if __name__ == '__main__':
    data_dir = ROOT / 'data/datasets'
    train_ds = dataSets(data_dir,'train')
    train_valid_ds = dataSets(data_dir,'train_valid')
    valid_ds =dataSets(data_dir,'valid')
    test_ds = dataSets(data_dir,'test')

    batch_size = 128
    train_iter = dataLoader(dataset=train_ds,batch_size=batch_size,
                            shuffle=True,drop_last = True)
    train_valid_iter = dataLoader(dataset=train_valid_ds,batch_size=batch_size,
                            shuffle=True,drop_last = True)
    valid_iter = dataLoader(dataset=valid_ds, batch_size=batch_size,
                            shuffle=False, drop_last=True)
    test_iter = dataLoader(dataset=test_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False)


