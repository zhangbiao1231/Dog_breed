# dog-breed 🐶, 1.0.0 license
"""
Run dog-breed classification inference on images, etc.

Usage - sources:
    $ python classify/predict.py --weights yolov5s-cls.pt --source img.jpg               # image
Usage - formats:
    $ python classify/predict.py --weights yolov5s-cls.pt                 # PyTorch
"""

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.ioff()

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator

# from models.common import DetectMultiBackend
from utils.augmentations import classify_transforms
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
from utils.general import (
    LOGGER,
    TEXT_LABELS,
    Profile,
    check_file,
    # check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    print_args,
)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights = ROOT / 'runs/train-cls/resnet34-resume-test2/weights/last.pt',# model.pt path(s)
        source = ROOT / "data/images", # file/dir/URL/glob
        data = None,
        imgsz=(224, 224),  # inference size (height, width)
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        nosave=False,  # do not save images/videos
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / "runs/predict-cls",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=False,  # use FP16 half-precision inference
        vid_stride=1,  # video frame-rate stride
):
    """Conducts dog-breed classification inference on diverse input sources and saves results."""
    source = str(source)
    save_img = not nosave and not source.endswith(".txt")  # save inference images
    is_file = Path(source).suffix[1:] in IMG_FORMATS
    is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    if is_url and is_file:
        source = check_file(source) #download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / "labels" if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #load model
    device = select_device(device)
    model = nn.ModuleList()
    file = Path(str(weights).strip().replace("'", ""))
    ckpt = torch.load(file, map_location="cpu")  # load
    ckpt = (ckpt["model"]).to(device).float()  # FP32 model
    model.append(ckpt.eval())
    model = model[-1]

    # Dataloader
    dataset = LoadImages(source, img_size=imgsz, transforms=classify_transforms(imgsz[0]), vid_stride=vid_stride)

    # Run inference
    seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.Tensor(im).to(device)
            im = im.float()
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim (1,3,224,224)
                # im = im.unsqueeze(dim=0) #增加batch维度
        # Inference
        with dt[1]:
            results = model(im) #(1,120)

        # Post-process
        with dt[2]:
            pred = F.softmax(results,dim=1) # probabilities #(1,120)

        # Process predictions
        for i ,prob in enumerate(pred): #prob(120,)
            seen += 1
            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

            p= Path(p)  # to Path
            save_path = str(save_dir / p.name) # im.jpg
            txt_path = str(save_dir / "labels" /p.stem) +("" if dataset.mode == "image" else f"_{frame}") # im.txt

            s += "%gx%g " % im.shape[2:]
            annotator = Annotator(im0, example=str(TEXT_LABELS), pil=True)

            #Print results
            top5i = prob.argsort(0,descending=True)[:5].tolist() #(5,)
            s += f"{', '.join(f'{TEXT_LABELS[j]} {prob[j]:.2f}' for j in top5i)}, "

            # Write results
            text = "\n".join(f"{prob[j]:.2f} {TEXT_LABELS[j]}" for j in top5i)
            if save_img or view_img:  # Add bbox to image
                annotator.text([32, 32], text, txt_color=(255, 255, 255))
            if save_txt: # Write to file
                with open(f"{txt_path}.txt",'a') as f:
                    f.write(text + '\n')

            im0 = annotator.result()
            # Save results (image with detections)
            if save_img:
                if dataset.mode =="image":
                    cv2.imwrite(save_path,im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
    LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms Post-process per image at shape {(1, 3, *imgsz)}" % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / "labels"}" if save_txt else ""
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")

def parse_opt():
    """Parses command line arguments for YOLOv5 inference settings including model, source, device, and image size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / 'runs/train-cls/resnet34-resume-test2/weights/last.pt', help="model path(s)")
    parser.add_argument("--source", type=str, default=ROOT / "data/images", help="file/dir/URL/glob/screen/0(webcam)")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="(optional) dataset.yaml path")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[224], help="inference size h,w")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-txt", default = True,action="store_true", help="save results to *.txt")
    parser.add_argument("--nosave", action="store_true", help="do not save images/videos")
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--visualize", action="store_true", help="visualize features")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument("--project", default=ROOT / "runs/predict-cls", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument("--vid-stride", type=int, default=1, help="video frame-rate stride")
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    """Executes YOLOv5 model inference with options for ONNX DNN and video frame-rate stride adjustments."""
    # check_requirements(ROOT / "requirements.txt", exclude=("tensorboard", "thop"))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
