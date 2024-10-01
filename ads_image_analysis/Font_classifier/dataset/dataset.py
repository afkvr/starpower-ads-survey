from PIL import Image
import cv2 as cv
from torch.utils.data import Dataset
import torch
from .constant import LABEL2ID
from .transformation import inference_input

class HENetdataset(Dataset):
    def __init__(self, img_paths, transform = None):
        self.img_paths = img_paths
        self.transform = transform
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv.imread(img_path)
        processed_img = inference_input(img)
        input_img = Image.fromarray(cv.cvtColor(processed_img, cv.COLOR_BGR2RGB)) 

        res = self.transform(input_img) if self.transform else input_img

        label = LABEL2ID[img_path.split(",")[1][:-4]]
        return res, torch.as_tensor(label, dtype= torch.long)
    