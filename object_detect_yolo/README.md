# Install package Ultrlytics for using yolo model
Follow this [link](https://github.com/ultralytics/ultralytics?)

## Download requirement data
1. Dataset: This data is labeled and saved as yolo dataset format. Download [dataset](https://drive.google.com/open?id=1kT1M0RgSd8x05RteP4DCHTKA0XYdguRk&usp=drive_copy) and extract data to folder **data/object_custome**
2. Pretrained model: Download model [yolo8m.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt) and save to folder **checkpoints**.

## Start train
1. Change path to dataset in file config `object_detect_yolo8m.yaml` to local path.
2. Change path of `model`, `config` in file train.py
3. Define number of epoches and batch > Run `python train.py`

