import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import re
import argparse
from rembg import remove
import matplotlib.pyplot as plt

from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
from rembg import remove, new_session
from OCR_processing import *
# from deepfont_recognize import *
import io
from utils import client
from omegaconf import OmegaConf

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "Detextor"))
from Detextor.lama_inpaint import inpaint_img_with_lama
from Detextor.lama.saicinpainting.training.trainers import load_checkpoint
from Detextor.utils import polygon_region, dilate_mask
import yaml
import base64
import shutil
from map_script_language import script_language_map
from GPT_category_test import compare_inpaint, category_object_GPT, category_text_logo_GPT, category_text_logo_GPT_2, category_group_GPT
from CTR_image import detect_CTR
import time

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def check_image_language(image_path):
    prompt = f"""There is a list of language and its symbol here: {json.dumps(script_language_map)}. Given a image below, I need you find all languages were used in image and return only a list of language symbols without adding any words. Example ["en", "japan"]. If image do not have any language, return 'Not detect'"""
    # print(type(prompt))
    promt_tmp = []
    previous_image = cv2.imread(image_path)
                
    _, buffer_bf = cv2.imencode(".png", previous_image)
    encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    promt_tmp.append({
                        "type": "text",
                        "text": prompt
                    })
    promt_tmp.append({
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                }
            })     
    # print(type(encoded_image_bf))   
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": promt_tmp
            },
    ]
    # print(type(promt_tmp))
    # # # Send the request to the OpenAI API
    try: 
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
    except Exception as e:
        time.sleep(1)
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
    # print(response_string)
    if response_string.choices[0].message.content == 'Not detect':
        message_contents = ["en"]
    else:
        message_contents = json.loads(response_string.choices[0].message.content)
    # print(message_contents)
    # print(type(message_contents))
    return message_contents

def draw_mask(mask,ax):
    color = np.array([1, 1, 1, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask#.reshape(h, w, 1)# * color.reshape(1, 1, -1)
    # print(mask_image.shape)
    cv2.imwrite("/tmp/mask.png",mask_image*255)
    ax.imshow(mask_image)

DEFAULT_WORDS = ["rectangle", "box", "screen", "label", "body", "main object", "Text"]

def visualMaskSam(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0],
                  sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


color_annotated = {
    "design": [0, 255, 0],      # Design element
    "text_design": [0, 0, 255],      # Text in design
    "text_normal": [255, 0, 0],      # Normal text
    "object_bound": [202, 39, 217],      # Object bound
    "logo": [50, 161, 240],      # Logo
}


def remove_overlap_box(box_data):
    iou_thresh = 0.6
    pop_index = []
    box_convert_list = [obj_tmp["box"] for obj_tmp in box_data]
    for idx in range(len(box_data)):
        if idx in pop_index:
            continue
        for idx_2 in range(idx+1, len(box_data)):
            iou_tmp = calculate_IoU_box(
                box_convert_list[idx], box_convert_list[idx_2])
            if iou_tmp >= iou_thresh:
                if box_data[idx]["score"] >= box_data[idx_2]["score"]: # or box_data[idx_2]["label-text"] in DEFAULT_WORDS:
                    pop_index.append(idx_2)
                else:   # box_data[idx]["label-text"] in DEFAULT_WORDS
                    pop_index.append(idx)
                    break       # Break loop if current box idx has lower score
            # elif checkInnerBox(box_convert_list[idx_2], box_convert_list[idx]):
            #     if  box_data[idx]["score"] < box_data[idx_2]["score"]:
            #         pop_index.append(idx)
            #         # pass
            # # elif checkInnerBox(box_convert_list[idx_2], box_convert_list[idx]) and box_data[idx_2]["score"] > box_data[idx]["score"]:
            #     pop_index.append(idx)
            #     break
            
    pop_index = np.unique(pop_index)
    fitler_data = [box_data[idx] for idx in range(len(box_data)) if idx not in pop_index]
    # print("Clean data groundingDiN", fitler_data, pop_index)
    return fitler_data, pop_index

def search_brand_product(list_text):
    example = {'brandA':['product1','product2']}
    if list_text == []:
        message_data = {
            "key": []
        }
        # inpainted_path = filename
    else:
        # Compare promt ChatGPT
        prompt = f"""
            There is a list of texts  {list_text}. I want you to find any brand that has the same name with a given text in the list and return all categories of products that brand has made or sold. Just give me a JSON object with the found brand names and their product categories, do not ads any words. Example "{example}"
            """
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": prompt
                },
        ]
        # # Send the request to the OpenAI API
        try: 
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=500,
                                                                n=1,
                                                                stop=None,
                                                                temperature=0.1
                                                                )
        except Exception as e:
            time.sleep(1)
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=500,
                                                                n=1,
                                                                stop=None,
                                                                temperature=0.1
                                                                )
        
        # print(response_string)
        message_contents = response_string.choices[0].message.content
        # print("search_brand_product: ",message_contents)
        message_contents = message_contents.replace("```json", "")
        message_contents = message_contents.replace("```", "")
        if len(message_contents) == 0:
            message_data = {"default":[]}
        message_data = json.loads(message_contents)
    
    return message_data

def inpainting_text(image, image_metadata, save_path):
    h, w = np.array(image.shape[:2]).astype(int)
    # print( h, w )
    if image_metadata["Elements"] != []:
        BoundingBoxes = []
        for image_metadata_tmp in image_metadata["Elements"]:
            if image_metadata_tmp["category"].lower() == "text":
                box = [np.array(tmp,dtype=np.float)*[w,h] for tmp in image_metadata_tmp["location-objective"]]
            else:
                
                box = [np.array(tmp,dtype=np.float)*[w,h] for tmp in image_metadata_tmp["location-objective"]]
                # print(box)
                rect1 = cv2.boundingRect(np.array(box, dtype=np.int32))
                x,y,w_,h_ = rect1
                box = [[x,y],[x+w_,y],[x+w_,y+h_],[x,y+h_]]
            BoundingBoxes.append(box)
        # print(BoundingBoxes)
        mask = np.zeros((image.shape[0],image.shape[1]), dtype=np.uint8)
        # sys.exit()
        for i, box in enumerate(BoundingBoxes): 
            mask = polygon_region(mask, box, offset=2)
        # Dilate mask
        dilate_kernel_size = 11
        mask = dilate_mask(mask, dilate_factor=dilate_kernel_size)
        inpainted_mask_p = f"{asset_img_folder}/{image_name}_text_mask.png"
        # cv2.imwrite(inpainted_mask_p, mask)
        image_impaint = image.copy()[:,:,:3] if image.shape[2] > 3 else image.copy()
        img_inpainted = inpaint_img_with_lama(image_impaint, mask, lama_model, predict_config, device=DEVICE)
        # impainted_image = dalle_inpainter.inpaint(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            #                                           text_boxes=text_boxes,
            #                                           prompt=DalleInpainter.DEFAULT_PROMPT)
        #Replace inpainted region into original image
        weave = np.where(mask[...,np.newaxis], img_inpainted,image_impaint)
        cv2.imwrite(save_path, weave)
    else:
        cv2.imwrite(save_path, image)
    # return

def detect_object_groundDiNO(inpainted_path, message_data):
    data_GSA = []
    image_source, image_dino = load_image(inpainted_path) #
    image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    
    annotated_frame = image_source.copy()
    h, w = image_source.shape[:2]
    groundingdino_res_tmp = [[],[],[]]
    for key in message_data:
        data_string = [tmp.lower() for tmp in message_data[key]]
        data_string += DEFAULT_WORDS
        TEXT_PROMPT = ", ".join(data_string)
        print(TEXT_PROMPT)
        
        boxes, logits, phrases = predict(model=groundingdino_model,
                                        image=image_dino,
                                        caption=TEXT_PROMPT,
                                        box_threshold=BOX_THRESHOLD,
                                        text_threshold=TEXT_THRESHOLD,
                                        device=DEVICE)
        # print("boxes: ", boxes)
        # print("logits: ", logits)
        # print("phrases: ", phrases)
        # annotated_frame = annotate(image_source=annotated_frame, boxes=boxes, logits=logits, phrases=phrases)
        
        boxes_np = boxes.numpy()
        logits_np = [round(tmp, 4) for tmp in logits.numpy()]
        for idx in range(len(boxes_np)):
            data_GSA.append({
                "label-text": phrases[idx].replace(" ", "_"),
                "box": convert_to_bbox(boxes_np[idx], w, h),
                "score": logits_np[idx],
            })
            groundingdino_res_tmp[0].append(boxes_np[idx])
            groundingdino_res_tmp[1].append(logits_np[idx])
            groundingdino_res_tmp[2].append(phrases[idx].replace(" ", "_"))
    # annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
    # cv2.imwrite(f"{asset_img_folder}/groundingdino_annotated_GPT_{image_name}.png", annotated_frame)
    
    if data_GSA == []:
        return data_GSA, None
    
    # Filter overlap box and choose main catergories label
    data_GSA_filter, pop_index = remove_overlap_box(data_GSA)
    tmp_label =  [[],[],[]]
    for idx in range(len(data_GSA)):
        if idx not in pop_index:
            tmp_label[0].append(groundingdino_res_tmp[0][idx])
            tmp_label[1].append(groundingdino_res_tmp[1][idx])
            tmp_label[2].append(groundingdino_res_tmp[2][idx])
    annotated_frame = annotate(
            image_source=annotated_frame, 
            boxes=torch.Tensor(tmp_label[0]), 
            logits=torch.Tensor(tmp_label[1]), 
            phrases=tmp_label[2])
    
    return data_GSA_filter, annotated_frame

def inpaint_lama_multiples(inpainted_text_path, data_GSA_filter, times=3):
    """
    """
    data_mask = []
    image =cv2.imread(inpainted_text_path)
    sam_predictor.set_image(cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB))
    list_coor = []
    for idx in range(len(data_GSA_filter)):
        # print(data_GSA_filter[idx])
        data_tmp = data_GSA_filter[idx]
        boxes_filt = torch.tensor([data_tmp["box"]])
        list_coor.append(data_tmp["box"])
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(DEVICE)
        mask_list, scores, logits_sam = sam_predictor.predict_torch(
                        point_coords=None,
                        point_labels=None,
                        boxes=transformed_boxes.to(DEVICE),
                        multimask_output=False,
                        hq_token_only=True
                    )
        try:
            mask_img = torch.zeros(mask_list.shape[-2:])
            mask_img[mask_list[0].cpu().numpy()[0] == True] = 255
            mask = mask_img.numpy().reshape(image.shape[0], image.shape[1]).astype(np.uint8)
            data_mask.append(mask)
            dilate_kernel_size = 5
            mask = dilate_mask(mask, dilate_factor=dilate_kernel_size)
            # Crop object and save final object image
            output_save_obj_final = f"""{object_img_folder}/{data_tmp["label-text"]}_{idx}_final.png"""
            crop_image = image[data_tmp["box"][1]:data_tmp["box"][3], data_tmp["box"][0]:data_tmp["box"][2]]
            mask_img_crop = mask[data_tmp["box"][1]:data_tmp["box"][3], data_tmp["box"][0]:data_tmp["box"][2]]
            rgba = np.dstack((crop_image, mask_img_crop))
            cv2.imwrite(output_save_obj_final, rgba)
            cv2.imwrite(f"""{object_img_folder}/{data_tmp["label-text"]}_{idx}_crop.png""", crop_image)
        except Exception as e:
            # pass
            crop_image = image[data_tmp["box"][1]:data_tmp["box"][3], data_tmp["box"][0]:data_tmp["box"][2]]
            cv2.imwrite(f"""{object_img_folder}/{data_tmp["label-text"]}_{idx}_final.png""", crop_image)

        # cv2.imwrite(f"""{asset_img_folder}/{image_name}_{idx}_maskSAM.png""", mask)
        
    lama_inpain_multi = []
    for idx in range(times):
        img_inpainted_lama = image.copy()#cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        for mask in data_mask:
            # # Dilate mask
            dilate_kernel_size = 25
            mask = dilate_mask(mask, dilate_factor=dilate_kernel_size)
            img_inpainted_lama = inpaint_img_with_lama(img_inpainted_lama, mask, lama_model, predict_config, device=DEVICE)
            # cv2.imwrite(f"{asset_img_folder}/{image_name}_inpainted_asset_{idx}.png", img_inpainted_lama)
            # cv2.imwrite(f"{asset_img_folder}/{image_name}_inpainted_asset_{idx}_mask.png", mask)
        lama_inpain_multi.append(img_inpainted_lama)
    return lama_inpain_multi, list_coor

def save_metadata(image_metadata):
    final_metadata = {
        "data":image_metadata
    }
    # print(final_metadata)
    path_final_output = f"{asset_img_folder}/{image_name}_metadata.json"
    with open(path_final_output ,'w') as fp:
        json.dump(final_metadata, fp, indent=4, cls=NpEncoder)

def main():
    # with open("/home/ubuntu/static_image_metadata/text_data.json", 'r') as f:
    #     text_data = json.load(f)
    # text_data = text_data["data"]
    with open("/home/ubuntu/static_image_metadata/list_image_gen_aug11.txt","r") as fp:
        list_image_name = fp.readlines()
    list_image_raw  = [path.replace("\n","") for path in list_image_name]
    
    global asset_img_folder, image_name, image_metadata, text_img_folder, object_img_folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="")
    args = parser.parse_args()
    image_input = args.image
    if os.path.isfile(image_input):
        list_image = [image_input]
    else:
        list_image =[]
        # list_image = [os.path.join(image_input, image)
        #               for image in os.listdir(image_input)]
        for path, subdirs, files in os.walk(image_input):
            for name in files:
                list_image.append(os.path.join(path, name))
    # num_threads = 1
    with open("/home/ubuntu/scan_debug.txt","r") as fp:
        list_image_name = fp.readlines()
    list_image = [path.replace("\n","") for path in list_image_name]
    for filename in list_image:
        # count = 1
        # filename = "781490520120333.mp4"
        # asset_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):  # and not os.p
            # try:
            with open(filename, 'rb') as f:
                check_chars = f.read()[-2:]
            # print(check_chars)
            # if check_chars != b'\xff\xd9':
            #     continue
            #     print('Not complete image')
            extension = filename.split(".")[-1]
            image_name = os.path.basename(filename).split(".")[0]
            
            print(filename)
            asset_img_folder = f"{os.path.dirname(filename)}/{os.path.basename(filename).split('.')[0]}"
            text_img_folder = os.path.join(asset_img_folder, "text_detection")
            object_img_folder = os.path.join(asset_img_folder, "object_detection")
            # asset_img_folder = f"/tmp/allbird_Aug05_mask_rmbg/{image_name}"
            # asset_img_folder = "/tmp"
            # text_inpainted_folder = f"{os.path.dirname(filename)}/inpainted"
            if not os.path.exists(asset_img_folder):
                os.mkdir(asset_img_folder, mode=0o777)
                os.mkdir(text_img_folder, mode=0o777)
                os.mkdir(object_img_folder, mode=0o777)
            # elif  :
            #     continue
            
            path_final_output = f"{asset_img_folder}/{image_name}_metadata.json"
            # if os.path.exists(path_final_output):
            #     continue
            #     pass
            image_metadata = {
                "Elements": [],
                "Group": {}
            }
            image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            # Check multiple language inside image
            # language_list = check_image_language(filename)
            # print("List language: ", language_list)
            language_list = ['en']
            # Detect raw text by Paddle OCR
            if language_list != []:
                result = detect_multiple_language(image, language_list, save_path_image=False)
        
            # # init_OCR_model("en")
            # # result = ocr.ocr(grayFrame, cls=True)
            print(result)
            if result != [[]]:
                track_design_elements(image, result, image_metadata, text_img_folder)
                list_polygon_box =  group_design_elements(image, image_metadata, image_name)
                inpainted_text_path = f"{asset_img_folder}/{image_name}_inpainted_text_lama.png"
                # inpainting_text(image, image_metadata, inpainted_text_path)
            else:
                inpainted_text_path = filename
            category_group_GPT(image, image_metadata, image_name, list_polygon_box)
            sys.exit()
            # print(inpainted_text_path)
            list_text = [tmp["text-content"] for tmp in image_metadata["Elements"] if tmp["category"] =="Text"]
            # print(list_text)
            
            # Search brand products name from OCR output
            message_data = search_brand_product(list_text)
            print("search_brand_product: ", message_data)
            
            # Find object using GroundDiNo model
            data_GSA_filter, annotated_frame = detect_object_groundDiNO(inpainted_text_path, message_data)
            # print(data_GSA_filter)
            ## Inpaint object with lama
            if data_GSA_filter != []:
                ## Category object using ChatGPT
                data_GSA_filter = category_object_GPT(inpainted_text_path, data_GSA_filter)
                
                image_metadata["Elements"] += data_GSA_filter
                
                cv2.imwrite(f"{asset_img_folder}/groundingdino_annotated_GPT_{image_name}_visualize.png", annotated_frame)
                # sys.exit()
                
                ## Inpaint object with lama
                lama_inpaint_imgs, list_coors = inpaint_lama_multiples(inpainted_text_path, data_GSA_filter)
                best_index = compare_inpaint(lama_inpaint_imgs[0],
                                            lama_inpaint_imgs[1],
                                            lama_inpaint_imgs[2],
                                            list_coors,
                                            []
                                            )
                print("Best lama index: ", best_index)
                try:
                    img_inpainted_lama_best = lama_inpaint_imgs[int(best_index)]
                except Exception:
                    img_inpainted_lama_best = lama_inpaint_imgs[0]
                
                cv2.imwrite(f"{asset_img_folder}/{image_name}_background.png", img_inpainted_lama_best)
            else:
                shutil.copy(inpainted_text_path, f"{asset_img_folder}/{image_name}_background.png")
            
            # Detect category logo Text
            count_logo = category_text_logo_GPT(image, image_metadata)
            if count_logo >1:
                image_path_index = [tmp for tmp in list_image_raw if image_name in tmp]
                # print(image_path_index)
                image_path_index = image_path_index[0]
                image_name_raw = os.path.basename(image_path_index).split(".")[0]
                path_json = os.path.join(os.path.dirname(image_path_index), image_name_raw,f"{image_name_raw}_metadata.json")
                with open(path_json, 'r') as fp:
                    data_raw = json.load(fp)["data"]
                # print()
                _ = category_text_logo_GPT_2(image, data_raw)
            # Detect CTR
            ctr_image = detect_CTR(image)
            image_metadata["Click-through-rate"] = ctr_image
            
            # Detect speaker
            
            
            ## Inpaint with dalle
            # img_inpainted_dalle = image_source.copy()
            # dalle_textbox = convert_object_box(data_GSA_filter)
            # # img_inpainted_dalle = dalle_inpainter.inpaint(image=Image.fromarray(img_inpainted_dalle),
            # #                                           text_boxes=dalle_textbox,
            # #                                           prompt="Inpainting regions and keep quality of image")
            # img_inpainted_dalle.save(f"{asset_img_folder}/{image_name}_inpainted_dalle.png")
        
            # #Save medata
            save_metadata(image_metadata)
            # except Exception as e: 
            #     print(filename)
            #     print("Error: ", e )
            #     continue

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#Load model GroundDiNo
groundingdino_path = "/home/ubuntu/GroundingDINO"
sys.path.append("groundingdino_path")
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filename= f"{groundingdino_path}/weights/groundingdino_swint_ogc.pth"
ckpt_config_filename = f"{groundingdino_path}/groundingdino/config/GroundingDINO_SwinT_OGC.py"
groundingdino_model = load_model(ckpt_config_filename, ckpt_filename)

# #Load model Lama
lama_ckpt = f"{PATH}/script/Detextor/pretrained_models/big-lama"
lama_config = f"{PATH}/script/Detextor/lama/configs/prediction/default.yaml"
predict_config = OmegaConf.load(lama_config)
predict_config.model.path = lama_ckpt
train_config_path = os.path.join(predict_config.model.path, 'config.yaml')

with open(train_config_path, 'r') as f:
    train_config = OmegaConf.create(yaml.safe_load(f))

train_config.training_model.predict_only = True
train_config.visualizer.kind = 'noop'

checkpoint_path = os.path.join(
    predict_config.model.path, 'models',
    predict_config.model.checkpoint
)
lama_model = load_checkpoint(
    train_config, checkpoint_path, strict=False, map_location='cpu')
lama_model.freeze()
# if not predict_config.get('refine', False):
lama_model.to(DEVICE)
    
## Init SAM-HQ model
sam = sam_model_registry["vit_l"](checkpoint=f"{PATH}/SAM_script/checkpoints/sam_hq_vit_l.pth")
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)
# dalle_inpainter = DalleInpainter()

BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.15
if __name__ == "__main__":
    main()