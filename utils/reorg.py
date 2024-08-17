import collections
from pathlib import Path
import os
import math
import argparse

import sys
import shutil

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # dog-breed root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def read_csv_labels(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()[1:]
    tokens = [l.rstrip().split(',') for l in lines]
    return dict(((name, label) for name, label in tokens))
def copyfile(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)
#@save
def reorg_train_valid(data_dir, labels, valid_ratio):
    n = collections.Counter(labels.values()).most_common()[-1][1]
    n_valid_per_label = max(1, math.floor(n * valid_ratio))
    label_count = {}
    for train_file in os.listdir(os.path.join(data_dir, 'train')):
        label = labels[train_file.split('.')[0]]
        fname = os.path.join(data_dir, 'train', train_file)
        copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                     'train_valid', label))
        if label not in label_count or label_count[label] < n_valid_per_label:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'valid', label))
            label_count[label] = label_count.get(label, 0) + 1
        else:
            copyfile(fname, os.path.join(data_dir, 'train_valid_test',
                                         'train', label))
    return n_valid_per_label
#@save
def reorg_test(data_dir):
    for test_file in os.listdir(os.path.join(data_dir, 'test')):
        copyfile(os.path.join(data_dir, 'test', test_file),
                 os.path.join(data_dir, 'train_valid_test', 'test',
                              'unknown'))
def reorg_dog_data(data_dir, valid_ratio):
    labels = read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    reorg_train_valid(data_dir, labels, valid_ratio)
    reorg_test(data_dir)
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=ROOT / "data/datasets/dog-breed-identification", help="dataset path")
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    #parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    opt = parser.parse_args()
    return opt
def main(opt):
    valid_ratio = opt.valid_ratio
    print(valid_ratio)
    data_dir = opt.data_dir
    reorg_dog_data(data_dir,valid_ratio)
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)

