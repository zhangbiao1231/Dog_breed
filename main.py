# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch
__file__ = 'main.py'
FILE = Path(__file__).resolve()  #__file__指的是当前文件,（即main.py）
ROOT = FILE.parents[0]  # dog-greed root directory 当前项目的父目录，
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    __file__ = 'main.py'

