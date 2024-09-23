from PIL import Image
from torch.utils.data import Dataset
import torch
from .constant import LABEL2ID

class CustomDataset(Dataset):
    def __init__(self, imgs, root, transform = None):
        self.imgs = imgs
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        set_ = img.split("_")[0]
        label = LABEL2ID[set_]

        input_img = Image.open(f"{self.root}/{set_}/{img}")

        tinput = self.transform(input_img) if self.transform else input_img

        return tinput, torch.as_tensor(label, dtype= torch.long)