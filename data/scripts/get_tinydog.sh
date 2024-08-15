#!/bin/bash
# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
# Download COCO 2017 dataset http://cocodataset.org
# Example usage: bash data/scripts/get_coco.sh
# parent
# ├── yolov5
# └── datasets
#     └── coco  ← downloads here

# Download/unzip labels
d='../datasets' # unzip directory
mkdir -p $d && cd $d
url=http://d2l-data.s3-accelerate.amazonaws.com/
f='kaggle_dog_tiny.zip'
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f -# && unzip -q $f -d $d && rm $f &

wait