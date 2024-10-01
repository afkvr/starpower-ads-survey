# Install package requirements
Create anaconda environment with `python >= 3.7`
## Ffmpeg
```bash
conda install conda-forge::ffmpeg
```
# Package depences
```bash
pip install -r requirements.txt
```
# PaddleOCR
Follow guide line here: [Quick Start](https://paddlepaddle.github.io/PaddleOCR/en/ppocr/quick_start.html)
# Third party packages
1. Pytorch: install torch 2.3.1 with CPU/GPU version [here](https://pytorch.org/get-started/previous-versions/#v231)
2. Tesseract ocr v3 [here](https://github.com/sirfz/tesserocr)
3. Download [checkpoint](https://drive.google.com/open?id=1DgQlpJbpiww8g-z-B0kisxHH9emSxYzE&usp=drive_copy) for Font_classifier/checkpoints
4. Download [**big-lama**](https://drive.google.com/drive/folders/1wpY-upCo4GIW4wVPnlMh_ym779lLIG2A) model to folder Detextor/pretrained_models
5. Put .env contrains OPENAI key into root folder.
# Run processes

## Detect text and logo
```bash
python text_logo_detection.py --image <path to image folder>
```



