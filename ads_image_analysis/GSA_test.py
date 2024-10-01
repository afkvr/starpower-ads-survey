
import os, sys
groundingdino_path = "/media/anlab/data/GroundingDINO-0.1.0-alpha2"
sys.path.append("groundingdino_path")

import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO

import groundingdino.datasets.transforms as T
from groundingdino.util.inference import load_model, load_image, predict, annotate
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import sam_model_registry, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


# diffusers
import PIL
import requests
import torch
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
from huggingface_hub import hf_hub_download
from rembg import remove, new_session

if __name__ == "__main__":
    DEVICE = "cpu"
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "/media/anlab/data/GroundingDINO-0.1.0-alpha2/weights/checkpoint0029_4scale_swin.pth"
    ckpt_config_filename = "/media/anlab/data/GroundingDINO-0.1.0-alpha2/groundingdino/config/GroundingDINO_SwinL.py"

    # groundingdino_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)

    groundingdino_model = load_model(f"{groundingdino_path}/groundingdino/config/GroundingDINO_SwinT_OGC.py", f"{groundingdino_path}/weights/groundingdino_swint_ogc.pth")

    sam = sam_model_registry["vit_h"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_vit_h_4b8939.pth")
    sam.to(device="cpu")
        # # mask_generator = SamAutomaticMaskGenerator(sam)
    sam_predictor = SamPredictor(sam)
    TEXT_PROMPT = "object. person."
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.15
    image_path = "/media/anlab/data/static_image_recognition/adimageset/"
    list_image = [os.path.join(image_path,image) for image in  os.listdir(image_path)]
    model_name_object = "isnet-general-use"
    # model_name_object = "sam"
    sesssion_object = new_session(model_name_object)
    # sys.exit()
    model_name_human = "u2net_human_seg"
    sesssion_human = new_session(model_name_human)
    padding_ratio = 0.2
    # image_path = "/media/anlab/data/static_image_recognition/adimageset/raid1.jpg"
    for filename in list_image:
        # count = 1
        # filename = "/media/anlab/data/static_image_recognition/adimageset/raid4.jpg"
        # filename = "/home/anlab/Pictures/test_1.png"
        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):
            print(filename)
    
            image_name = os.path.basename(filename).split(".")[0]
            
            image_source, image = load_image(filename)
            image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
            h,w = image_source.shape[:2]
            boxes, logits, phrases = predict(
                model=groundingdino_model, 
                image=image, 
                caption=TEXT_PROMPT, 
                box_threshold=BOX_TRESHOLD, 
                text_threshold=TEXT_TRESHOLD,
                device=DEVICE
            )
            boxes_np = boxes.numpy()
            logits_np =[round(tmp,4)  for tmp in logits.numpy()]
            print("boxes: ", boxes)
            print("logits: ", logits)
            print("phrases: ", phrases)
            for idx in range(len(phrases)):
                center_x,center_y,w_tmp,h_tmp = np.array(boxes_np[idx]*[w,h,w,h]).astype(float)
                # print(w_tmp,h_tmp)
                # print(center_x,center_y)
                w_tmp = int(w_tmp*(1+padding_ratio))
                h_tmp = int(h_tmp*(1+padding_ratio))
                # print(w_tmp,h_tmp)
                x_tl = max(int(center_x - w_tmp/2),0)
                y_tl = max(int(center_y - h_tmp/2),0)
                x_br = min(int(center_x + w_tmp/2),w)
                y_br = min(int(center_y + h_tmp/2),h)
                # w_tmp = int(w_tmp)
                # h_tmp = int(h_tmp)
                # print(x_tl, y_tl, x_br, y_br)
                # print(center_x,center_y,w_tmp,h_tmp)
                # print(x,y)
                
                crop_img = image_source[y_tl:y_br, x_tl:x_br]
                
                ## Check phrases and choose rembg session
                # print(phrases[idx])
                if phrases[idx] != "person":
                    crop_img_rmbg = remove(crop_img, sesssion_object)
                else:
                    crop_img_rmbg = remove(crop_img, sesssion_human)
                # rmbg_mask = crop_img_rmbg[:,:,3]
                # ret, rmbg_text = cv2.threshold(rmbg_mask,100,255,cv2.THRESH_BINARY)
                print(f"/tmp/{image_name}_{phrases[idx]}_{idx}.png")
                cv2.imwrite(f"/tmp/{image_name}_{phrases[idx]}_{idx}_{logits_np[idx]}_box.jpg", crop_img)
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            annotated_frame = annotated_frame[...,::-1] # BGR to RGB
            cv2.imwrite(f"/tmp/groundingdino_annotated_{image_name}.png", annotated_frame)
            # break