
from models.get_Net import get_net
import torch
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory#
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(ROOT)
def load(cfg,args):
    clone = get_net(cfg)
    load_folder = os.path.join(cfg.OUTPUT_DIR, "train")
    if not os.path.exists(load_folder):
        os.makedirs(load_folder)
    path = os.path.join(load_folder,args.load_model_filename)
    clone.load_state_dict(torch.load(path))
    return clone



