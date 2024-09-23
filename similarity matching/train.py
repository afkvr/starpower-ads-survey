from dataset.dataset import CustomDataset
from dataset.transformations import augmenter, to_tensor
from model.lightning_wrapper import EncoderWrapper
from model.similarity_module import ContrastiveEmbedding, LightContrastiveEmbedding
from model.configs import load_config

import torch
from torch.utils.data import DataLoader 
import pytorch_lightning as L
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
    StochasticWeightAveraging
)

import os
import random

torch.set_float32_matmul_precision("high")
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_list(root, save_path, list_name): 
    imgs_list = []

    sets = os.listdir(root)

    for set_ in sets: 
        imgs = os.listdir(f"{root}/{set_}")
        for img in imgs: 
            imgs_list.append(img)

    with open(f"{save_path}/{list_name}", "w", encoding="utf-8") as f: 
        for img in imgs_list: 
            f.writelines(f"{img}\n")

def save_data(save_path, file_name, data_list):
    with open(f"{save_path}/{file_name}", "w", encoding="utf-8") as f: 
        for img in data_list: 
            f.writelines(f"{img}\n") 

if __name__=="__main__":

    config = load_config("./configs/train_cfg.yml")

    root = config['data']['root']
    data_cache = config['data']['data_cache']
    img_list = config['data']['imgs_list']

    os.makedirs(data_cache, exist_ok=True)
    create_list(root, data_cache, list_name=img_list)

    # Load data
    dataset_path = f"{data_cache}/{img_list}"
    with open(dataset_path, "r", encoding="utf-8") as f: 
        imgs = f.read().splitlines()

    # Shuffle and split data
    train_ratio = config['training']['train_ratio']
    validation_ratio = config['training']['validation_ratio']

    random.shuffle(imgs) 
    total_size = len(imgs)
    train_size = int(train_ratio*total_size)

    train_data = imgs[:train_size]
    validation_data  = imgs[train_size:]

    # save data split to cache
    if config['data']['save_data']:
        save_data(save_path=data_cache, file_name="train.txt", data_list=train_data)
        save_data(save_path=data_cache, file_name="val.txt", data_list=validation_data)

    # Torch dataset
    train_dataset =  CustomDataset(imgs  = train_data, root=root, transform=augmenter)
    validation_dataset =  CustomDataset(imgs  = validation_data, root=root, transform=to_tensor)

    # Data loader 
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']
    pwf = False
    pwt = True
    train_loader =  DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers= num_workers,  persistent_workers= pwt)
    validation_loader =  DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers= num_workers,  persistent_workers= pwt)

    # Define model
    model = EncoderWrapper(model=LightContrastiveEmbedding(embedding_size= config['training']['embedding_size'], req_grad=config['training']['require_backbone_grad']),
                        embedding_size= config['training']['embedding_size'],
                        margin=config['training']['margin'],
                        learning_rate=config['training']['learning_rate'],
                        )

    torch.cuda.empty_cache()

    # Define trainer 
    training_callbacks = [
            EarlyStopping(monitor="val_loss", mode="min"),
            StochasticWeightAveraging(swa_lrs=1e-2),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=config['checkpoint']['save_dir'],
                save_top_k=config['checkpoint']['k'],
                monitor="val_loss",
                filename="CE-{epoch:02d}-{val_loss:.4f}",
                save_last=True,
            ),
            ModelSummary(-1)
        ]

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=config['logging']['save_dir'])
    trainer = L.Trainer(max_epochs=config['training']['epochs'], callbacks=training_callbacks, log_every_n_steps=config['logging']['log_every_n_steps'], logger=tb_logger)

    # Train 
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=validation_loader,
        ckpt_path= None,
)
