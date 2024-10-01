# Install package requirements
Create anaconda environment with `python >= 3.7`
## Ffmpeg
```bash
conda install conda-forge::ffmpeg
```
# Retiface-2
```bash
cd RetinaFace_tf2
pip install -r requirements.txt
```
Download 2 models in [link](https://github.com/peteryuX/retinaface-tf2?tab=readme-ov-file#models) to folder `RetinaFace_tf2/data`
# PaddleOCR
Follow guide line here: [Quick Start](https://paddlepaddle.github.io/PaddleOCR/en/ppocr/quick_start.html)
# Third party package
1. Pytorch: install torch 2.0.0 with CPU/GPU version [here](https://pytorch.org/get-started/previous-versions/#v200)
2. Tesseract ocr v3 [here](https://github.com/sirfz/tesserocr)
3. Scenedetect [here](https://www.scenedetect.com/download/)

# Run processes
## Video processing
Font recognition: Detect text informations in single video or folder contains videos > save data in json file
```bash
python font_recognition.py --video <path to video folder>
```
Speaker recognition: Detect speaker informations in single video or folder contains videos > save data in json file
```bash
python speaker_recognition.py --video <path to video folder>
```
## Image processing
Speaker recognition: Detect speaker informations in single image or folder contain images > save data in json file
```bash
python speaker_recognition_image.py --image <path to image folder>
```



