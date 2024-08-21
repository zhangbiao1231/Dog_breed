# dog-breed 🐶, 1.0.0 license
"""General utils."""

import contextlib
import glob
import inspect
import logging
import logging.config
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
import urllib
from copy import deepcopy
from datetime import datetime
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from subprocess import check_output
from tarfile import is_tarfile
from typing import Optional
from zipfile import ZipFile, is_zipfile
import pkg_resources as pkg

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml

# Import 'ultralytics' package or install if missing
try:
    import ultralytics

    assert hasattr(ultralytics, "__version__")  # verify package is not directory
except (ImportError, AssertionError):
    os.system("pip install -U ultralytics")
    import ultralytics

from ultralytics.utils.checks import check_requirements

from utils import TryExcept
from utils.downloads import curl_download
# from utils.metrics import fitness
from utils.reorg import read_csv_labels

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
RANK = int(os.getenv("RANK", -1))

# Settings
NUM_THREADS = min(8, max(1, os.cpu_count() - 1))  # number of YOLOv5 multiprocessing threads
DATASETS_DIR = Path(os.getenv("dog-breed_DATASETS_DIR", ROOT / "data/datasets"))  # global datasets directory
AUTOINSTALL = str(os.getenv("dog-breed_AUTOINSTALL", True)).lower() == "true"  # global auto-install mode
VERBOSE = str(os.getenv("dog-breed_VERBOSE", True)).lower() == "true"  # global verbose mode
TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
FONT = "Arial.ttf"  # https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf

_,TEXT_LABELS = read_csv_labels(DATASETS_DIR / 'dog-breed-identification/labels.csv') #
TEXT_LABELS = sorted(TEXT_LABELS)

torch.set_printoptions(linewidth=320, precision=5, profile="long")
np.set_printoptions(linewidth=320, formatter={"float_kind": "{:11.5g}".format})  # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
os.environ["NUMEXPR_MAX_THREADS"] = str(NUM_THREADS)  # NumExpr max threads
os.environ["OMP_NUM_THREADS"] = "1" if platform.system() == "darwin" else str(NUM_THREADS)  # OpenMP (PyTorch and SciPy)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress verbose TF compiler warnings in Colab
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"  # suppress "NNPACK.cpp could not initialize NNPACK" warnings
os.environ["KINETO_LOG_LEVEL"] = "5"  # suppress verbose PyTorch profiler output when computing FLOPs
# os.environ["TENSORBOARD_PROXY_URL"]= os.environ["NB_PREFIX"]+"/proxy/6006/"

LOGGING_NAME = "dog-breed 🐶"

def set_logging(name=LOGGING_NAME, verbose=True):
    """Configures logging with specified verbosity; `name` sets the logger's name, `verbose` controls logging level."""
    rank = int(os.getenv("RANK", -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {name: {"format": "%(message)s"}},
            "handlers": {
                name: {
                    "class": "logging.StreamHandler",
                    "formatter": name,
                    "level": level,
                }
            },
            "loggers": {
                name: {
                    "level": level,
                    "handlers": [name],
                    "propagate": False,
                }
            },
        }
    )

set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

def colorstr(*input):
    """
    Colors a string using ANSI escape codes, e.g., colorstr('blue', 'hello world').

    See https://en.wikipedia.org/wiki/ANSI_escape_code.
    """
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m",
    }
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    """
    Generates an incremented file or directory path if it exists, with optional mkdir; args: path, exist_ok=False,
    sep="", mkdir=False.

    Example: runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc
    """
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def check_requirements():
    return None

class WorkingDirectory(contextlib.ContextDecorator):
    # Usage: @WorkingDirectory(dir) decorator or 'with WorkingDirectory(dir):' context manager
    def __init__(self, new_dir):
        """Initializes a context manager/decorator to temporarily change the working directory."""
        self.dir = new_dir  # new dir
        self.cwd = Path.cwd().resolve()  # current dir

    def __enter__(self):
        """Temporarily changes the working directory within a 'with' statement context."""
        os.chdir(self.dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores the original working directory upon exiting a 'with' statement context."""
        os.chdir(self.cwd)
def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Logs the arguments of the calling function, with options to include the filename and function name."""
    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix("")
    except ValueError:
        file = Path(file).stem
    s = (f"{file}: " if show_file else "") + (f"{func}: " if show_func else "")
    LOGGER.info(colorstr(s) + ", ".join(f"{k}={v}" for k, v in args.items()))


def init_seeds(seed=0, deterministic=False):
    """
    Initializes RNG seeds and sets deterministic options if specified.

    See https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for Multi-GPU, exception safe
    # torch.backends.cudnn.benchmark = True  # AutoBatch problem https://github.com/ultralytics/yolov5/issues/9287
    if deterministic :  # https://github.com/ultralytics/yolov5/pull/8213
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["PYTHONHASHSEED"] = str(seed)
def file_date(path=__file__):
    """Returns a human-readable file modification date in 'YYYY-M-D' format, given a file path."""
    t = datetime.fromtimestamp(Path(path).stat().st_mtime)
    return f"{t.year}-{t.month}-{t.day}"
    return result
def check_dataset(data, autodownload=True):
    """Validates and/or auto-downloads a dataset, returning its configuration as a dictionary."""

    # Download (optional)
    extract_dir = ""
    if isinstance(data, (str, Path)) and (is_zipfile(data) or is_tarfile(data)):
        download(data, dir=f"{DATASETS_DIR}/{Path(data).stem}", unzip=True, delete=False, curl=False, threads=1)
        data = next((DATASETS_DIR / Path(data).stem).rglob("*.yaml"))
        extract_dir, autodownload = data.parent, False

    # Read yaml (optional)
    if isinstance(data, (str, Path)):
        data = yaml_load(data)  # dictionary

    # Checks
    for k in "train", "val", "names":
        assert k in data, str(f"data.yaml '{k}:' field missing ❌")
    if isinstance(data["names"], (list, tuple)):  # old array format
        data["names"] = dict(enumerate(data["names"]))  # convert to dict
    assert all(isinstance(k, int) for k in data["names"].keys()), "data.yaml names keys must be integers, i.e. 2: car"
    data["nc"] = len(data["names"])

    # Resolve paths
    path = Path(extract_dir or data.get("path") or "")  # optional 'path' default to '.'
    if not path.is_absolute():
        path = (ROOT / path).resolve()
        data["path"] = path  # download scripts
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    # Parse yaml
    train, val, test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            LOGGER.info("\nDataset not found ⚠️, missing paths %s" % [str(x) for x in val if not x.exists()])
            if not s or not autodownload:
                raise Exception("Dataset not found ❌")
            t = time.time()
            if s.startswith("http") and s.endswith(".zip"):  # URL
                f = Path(s).name  # filename
                LOGGER.info(f"Downloading {s} to {f}...")
                torch.hub.download_url_to_file(s, f)
                Path(DATASETS_DIR).mkdir(parents=True, exist_ok=True)  # create root
                unzip_file(f, path=DATASETS_DIR)  # unzip
                Path(f).unlink()  # remove zip
                r = None  # success
            elif s.startswith("bash "):  # bash script
                LOGGER.info(f"Running {s} ...")
                r = subprocess.run(s, shell=True)
            else:  # python script
                r = exec(s, {"yaml": data})  # return None
            dt = f"({round(time.time() - t, 1)}s)"
            s = f"success ✅ {dt}, saved to {colorstr('bold', DATASETS_DIR)}" if r in (0, None) else f"failure {dt} ❌"
            LOGGER.info(f"Dataset download {s}")
    check_font("Arial.ttf" if is_ascii(data["names"]) else "Arial.Unicode.ttf", progress=True)  # download fonts
    return data  # dictionary
def check_version(current="0.0.0", minimum="0.0.0", name="version ", pinned=False, hard=False, verbose=False):
    """Checks if the current version meets the minimum required version, exits or warns based on parameters."""
    current, minimum = (pkg.parse_version(x) for x in (current, minimum))
    result = (current == minimum) if pinned else (current >= minimum)  # bool
    s = f"WARNING ⚠️ {name}{minimum} is required by YOLOv5, but {name}{current} is currently installed"  # string
    if hard:
        assert result, emojis(s)  # assert min requirements met
    if verbose and not result:
        LOGGER.warning(s)
    return result
class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0, device: torch.device = None):
        """Initializes a profiling context for YOLOv5 with optional timing threshold and device specification."""
        self.t = t
        self.device = device
        self.cuda = bool(device and str(device).startswith("cuda"))

    def __enter__(self):
        """Initializes timing at the start of a profiling context block for performance measurement."""
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        """Concludes timing, updating duration for profiling upon exiting a context block."""
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        """Measures and returns the current time, synchronizing CUDA operations if `cuda` is True."""
        if self.cuda:
            torch.cuda.synchronize(self.device)
        return time.time()
def yaml_load(file="data.yaml"):
    """Safely loads and returns the contents of a YAML file specified by `file` argument."""
    with open(file, errors="ignore") as f:
        return yaml.safe_load(f)

def yaml_save(file="data.yaml", data=None):
    """Safely saves `data` to a YAML file specified by `file`, converting `Path` objects to strings; `data` is a
    dictionary.
    """
    if data is None:
        data = {}
    with open(file, "w") as f:
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)
def unzip_file(file, path=None, exclude=(".DS_Store", "__MACOSX")):
    """Unzips `file` to `path` (default: file's parent), excluding filenames containing any in `exclude` (`.DS_Store`,
    `__MACOSX`).
    """
    if path is None:
        path = Path(file).parent  # default path
    with ZipFile(file) as zipObj:
        for f in zipObj.namelist():  # list all archived filenames in the zip
            if all(x not in f for x in exclude):
                zipObj.extract(f, path=path)
def download(url, dir=".", unzip=True, delete=True, curl=False, threads=1, retry=3):
    """Downloads and optionally unzips files concurrently, supporting retries and curl fallback."""

    def download_one(url, dir):
        """Downloads a single file from `url` to `dir`, with retry support and optional curl fallback."""
        success = True
        if os.path.isfile(url):
            f = Path(url)  # filename
        else:  # does not exist
            f = dir / Path(url).name
            LOGGER.info(f"Downloading {url} to {f}...")
            for i in range(retry + 1):
                if curl:
                    success = curl_download(url, f, silent=(threads > 1))
                else:
                    torch.hub.download_url_to_file(url, f, progress=threads == 1)  # torch download
                    success = f.is_file()
                if success:
                    break
                elif i < retry:
                    LOGGER.warning(f"⚠️ Download failure, retrying {i + 1}/{retry} {url}...")
                else:
                    LOGGER.warning(f"❌ Failed to download {url}...")

        if unzip and success and (f.suffix == ".gz" or is_zipfile(f) or is_tarfile(f)):
            LOGGER.info(f"Unzipping {f}...")
            if is_zipfile(f):
                unzip_file(f, dir)  # unzip
            elif is_tarfile(f):
                subprocess.run(["tar", "xf", f, "--directory", f.parent], check=True)  # unzip
            elif f.suffix == ".gz":
                subprocess.run(["tar", "xfz", f, "--directory", f.parent], check=True)  # unzip
            if delete:
                f.unlink()  # remove zip

    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)  # make directory
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))  # multithreaded
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)

def save_to_csv(valid_dir,csv,pred,TEXT_LABELS):
    ids = sorted(os.listdir(valid_dir))
    with open(csv, 'w') as f:
        f.write('id,' + ','.join(TEXT_LABELS) + '\n')
        for i, output in zip(ids, pred):
            f.write(i.split('.')[0] + ',' + ','.join(
                [str(num) for num in output]) + '\n')

def intersect_dicts(da, db, exclude=()):
    """Returns intersection of `da` and `db` dicts with matching keys and shapes, excluding `exclude` keys; uses `da`
    values.
    """
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

