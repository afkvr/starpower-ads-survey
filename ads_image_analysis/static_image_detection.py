import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import os
import sys
import re
import argparse
from rembg import remove
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
import matplotlib.pyplot as plt


from groundingdino.util.inference import load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
from rembg import remove, new_session
from OCR_processing import *
from deepfont_recognize import *
from backgroundremover.bg import get_model, alpha_matting_cutout, naive_cutout
from backgroundremover.u2net import detect
import io
from utils import DalleInpainter, client


def draw_mask(mask, ax):
    color = np.array([1, 1, 1, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask  # .reshape(h, w, 1)# * color.reshape(1, 1, -1)
    print(mask_image.shape)
    cv2.imwrite("/tmp/mask.png", mask_image*255)
    ax.imshow(mask_image)


DEFAULT_WORDS = ["rectangle", "box", "screen",
                 "label", "body", "main object", "Text"]

bgrm_model_choices = ["u2net", "u2net_human_seg", "u2netp"]
groundingdino_path = "/media/anlab/data/GroundingDINO-0.1.0-alpha2"
sys.path.append("groundingdino_path")


def remove_bg(crop_img, out_img_path, model_name=bgrm_model_choices[0],
              alpha_matting=True,
              alpha_matting_foreground_threshold=240,
              alpha_matting_background_threshold=10,
              alpha_matting_erode_structure_size=3,
              alpha_matting_base_size=1000):
    model = get_model(model_name)
    img = crop_img

    mask = detect.predict(model, np.array(img)).convert("L")

    if alpha_matting:
        cutout = alpha_matting_cutout(
            img,
            mask,
            alpha_matting_foreground_threshold,
            alpha_matting_background_threshold,
            alpha_matting_erode_structure_size,
            alpha_matting_base_size,
        )
    else:
        cutout = naive_cutout(img, mask)

    bio = io.BytesIO()
    cutout.save(bio, "PNG")
    f = open(out_img_path, "wb")
    f.write(bio.getbuffer())
    f.close()


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
                # or box_data[idx_2]["input_text"] in DEFAULT_WORDS:
                if box_data[idx]["score"] >= box_data[idx_2]["score"]:
                    pop_index.append(idx_2)
                else:   # box_data[idx]["input_text"] in DEFAULT_WORDS
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
    fitler_data = [box_data[idx]
                   for idx in range(len(box_data)) if idx not in pop_index]

    return fitler_data, pop_index


def main():
    global text_img_folder, image_name, image_metadata
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="")
    args = parser.parse_args()
    image_input = args.image
    if os.path.isfile(image_input):
        list_image = [image_input]
    else:
        list_image = [os.path.join(image_input, image)
                      for image in os.listdir(image_input)]
    num_threads = 1
    for filename in list_image:
        # count = 1
        # filename = "781490520120333.mp4"
        # text_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):  # and not os.p

            with open(filename, 'rb') as f:
                check_chars = f.read()[-2:]
            # print(check_chars)
            # if check_chars != b'\xff\xd9':
            #     continue
            #     print('Not complete image')

            text_img_folder = f"{os.path.dirname(filename)}/{os.path.basename(filename).split('.')[0]}"
            if not os.path.exists(text_img_folder):
                os.mkdir(text_img_folder, mode=0o777)
            # else:
            #     continue
            # print(filename)
            image_metadata = {
                "Design-elements": [],
                "Text-normal": [],
                "Logo": [],
                "Object-bound": [],
                "Background": []
            }
            extension = filename.split(".")[-1]
            image_name = os.path.basename(filename).split(".")[0]
            print(image_name)
            image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # Detect raw text by Paddle OCR
            grayFrame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(grayFrame, cls=True)
            if result != [None]:
                track_design_elements(image, result, image_metadata)
            # sys.exit()
            list_text = [tmp["text"] for tmp in image_metadata["Text-normal"]]
            list_text += [tmp["text"]
                          for tmp in image_metadata["Design-elements"]]
            # # list_text = ['NEWLOW PRICE', 'Wool Runner', 'allbirds']
            # print(list_text)
            # if list_text == []:
            #     message_data = {
            #         "key": []
            #     }
            #     inpainted_path = filename
            # else:
            # transform textbox data to inpainting image
            # text_boxes = convert_text_box(image_metadata)
            # print(text_boxes)
            # Remove text from image using DallEInpainter
            # impainted_image = dalle_inpainter.inpaint(image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
            #                                           text_boxes=text_boxes,
            #                                           prompt=DalleInpainter.DEFAULT_PROMPT)
            # inpainted_path = f"{text_img_folder}/{image_name}_inpainted_3.png"
            # impainted_image.save(inpainted_path, quality=95)
            # del impainted_image
            # sys.exit()
            # # Compare promt ChatGPT
            # # promt_tmp = f"There is a list of texts {list_text}. I want you find any brand has the same name with given text in list and return all categories of product that brand had made or sale. Output format should be only a list of texts like given input list."
            # prompt = f"""
            #     There is a list of texts  {list_text}. I want you to find any brand that has the same name with a given text in the list and return all categories of products that brand has made or sold. Just give me a JSON object with the found brand names and their product categories, do not ads any words
            #     """
            # message = [
            #     {"role": "system", "content": "You are a helpful assistant."},
            #     {"role": "user",
            #      "content": prompt
            #      },
            # ]
            # # data_string = [
            # #     "Website Building",
            # #     "E-commerce",
            # #     "Marketing Tools",
            # #     "Domain Services",
            # #     "Analytics"
            # # ]

            # # # Send the request to the OpenAI API
            # # # parameters = { "model": "gpt-4o", "messages": message, "max_tokens": 500 }
            # response_string = client.chat.completions.create(model="gpt-4o",
            #                                                  messages=message,
            #                                                  max_tokens=100,
            #                                                  n=1,
            #                                                  stop=None,
            #                                                  temperature=0)
            # # print(response_string)
            # message_contents = response_string.choices[0].message.content
            # print(message_contents)

            # message_contents = message_contents.replace("```json", "")
            # message_contents = message_contents.replace("```", "")
            # message_data = json.loads(message_contents)
            # print(data)

            # Detect object using GSA and chat GPT text
            # message_data = {
            #     # "ROTATE STUDIO": ["Clothing", "Accessories", "Footwear"],
            #     # "SCHEDULE": ["Planners", "Calendars", "Stationery"]
            #     "allbirds": ["shoes", "apparel", "accessories"],
            #      "TREELOUNGER": ["Hunting Gear", "Tree Stands"]
            # }
            message_data = {
                "allbirds": [
                    "shoes",
                    "socks",
                    "apparel",
                    "accessories"
                ]}
            data_GSA = []
            image_source, image_dino = load_image(filename)
            image_source = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)

            annotated_frame = image_source.copy()
            h, w = image_source.shape[:2]
            groundingdino_res_tmp = [[], [], []]
            for key in message_data:
                data_string = [tmp.lower() for tmp in message_data[key]]
                # if data_string == [] and data_GSA != []:
                #     continue
                data_string += DEFAULT_WORDS
                TEXT_PROMPT = ", ".join(data_string)
                print("TEXT_PROMPT: ", TEXT_PROMPT)

                boxes, logits, phrases = predict(
                    model=groundingdino_model,
                    image=image_dino,
                    caption=TEXT_PROMPT,
                    box_threshold=BOX_TRESHOLD,
                    text_threshold=TEXT_TRESHOLD,
                    device=DEVICE
                )

                print("boxes: ", boxes)
                print("logits: ", logits)
                print("phrases: ", phrases)
                annotated_frame = annotate(
                    image_source=annotated_frame, boxes=boxes, logits=logits, phrases=phrases)

                boxes_np = boxes.numpy()
                logits_np = [round(tmp, 4) for tmp in logits.numpy()]
                for idx in range(len(boxes_np)):
                    data_GSA.append({
                        "input_text": phrases[idx].replace(" ", "_"),
                        "box": convert_to_bbox(boxes_np[idx], w, h),
                        "score": logits_np[idx],
                    })
                    groundingdino_res_tmp[0].append(boxes_np[idx])
                    groundingdino_res_tmp[1].append(logits_np[idx])
                    groundingdino_res_tmp[2].append(phrases[idx].replace(" ", "_"))
            # annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB
            # cv2.imwrite(
            #     f"{text_img_folder}/groundingdino_annotated_GPT_{image_name}.png", annotated_frame)
            cv2.imwrite(
                f"/tmp/groundingdino_annotated_GPT_{image_name}.png", annotated_frame)
            print(data_GSA)
            if data_GSA == []:
                continue
            # Filter overlap box and choose main catergories label
            data_GSA_filter, pop_index = remove_overlap_box(data_GSA)
            tmp_label = [[], [], []]
            for idx in range(len(data_GSA)):
                if idx not in pop_index:
                    tmp_label[0].append(groundingdino_res_tmp[0][idx])
                    tmp_label[1].append(groundingdino_res_tmp[1][idx])
                    tmp_label[2].append(groundingdino_res_tmp[2][idx])
            annotated_frame = annotate(
                image_source=image_source.copy(),
                boxes=torch.Tensor(tmp_label[0]),
                logits=torch.Tensor(tmp_label[1]),
                phrases=tmp_label[2])
            # cv2.imwrite(
            #     f"{text_img_folder}/groundingdino_annotated_GPT_{image_name}_clean.png", annotated_frame)
            cv2.imwrite(
                f"/tmp/groundingdino_annotated_GPT_{image_name}_clean.png", annotated_frame)

            print(data_GSA_filter)
            image_metadata["Object-bound"] = data_GSA_filter
            # sam_predictor.set_image(image_source)
            # Remove background for element
            value = 0  # 0 for background
            # boxes_filt = torch.tensor([data_tmp["box"]
            #                           for data_tmp in data_GSA_filter])
            # boxes_filt = np.array([data_tmp["box"] for data_tmp in data_GSA_filter])
            # print(boxes_filt)

            # transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            #     boxes_filt, image.shape[:2]).to(DEVICE)
            # print(transformed_boxes[0])
            # sys.exit()
            # mask_list, scores, logits_sam = sam_predictor.predict_torch(
            #     point_coords=None,
            #     point_labels=None,
            #     boxes=transformed_boxes.to(DEVICE),
            #     multimask_output=False,
            #     # multimask_output=True,
            #     return_logits=True,
            # )
            # sys.exit()
            # print(logits_sam.shape)
            # print(logits_sam[0].shape)
            # for idx in range(len(data_GSA_filter)):
            #     # mask_img = torch.zeros(image_source.shape[-2:])
            #     data_tmp = data_GSA_filter[idx]
            #     box = np.array(data_tmp["box"])
            #     # mask_input_tmp = logits_sam[idx][0,:,:]
            #     # print(transformed_boxes[idx].cpu().numpy())
            #     masks, _, _ = sam_predictor.predict(
            #         point_coords=None,
            #         point_labels=None,
            #         box=box[None,:],
            #         multimask_output=False,
            #     )
            #     h, w = masks[0].shape[-2:]
            #     # # cv2.imwrite(f"/tmp/mask_{idx}.png", mask_list_tmp[0])
            #     #
            #     masks = masks[0].reshape(h, w, 1)
            #     mask_img = masks*255
            #     plt.figure(figsize=(10, 10))
            #     # plt.imshow(image)
            #     draw_mask(masks, plt.gca())
            #     # show_box(box, plt.gca())
            #     plt.axis('off')
            #     plt.show()
            #     # mask_img = mask_img.numpy()
            #     mask_img_crop = mask_img[box[1]:box[3], box[0]:box[2]]
            #     output_save_crop = f"""{text_img_folder}/{data_tmp["input_text"]}_{idx}_final.png"""
            #     crop_image = image_source[box[1]:box[3], box[0]:box[2]]
            #     cv2.imwrite(
            #         f"""{text_img_folder}/{data_tmp["input_text"]}_{idx}_crop.png""", crop_image)
            #     cv2.imwrite(
            #         f"""{text_img_folder}/{data_tmp["input_text"]}_{idx}_mask.png""", mask_img_crop)
            #     rgba = np.dstack((crop_image, mask_img_crop))
            #     cv2.imwrite(output_save_crop, rgba)

            # ####Visualize image
            # # Object bound
            # # tmp_image = image.copy()
            # # # boolean indexing and assignment based on mask
            # # tmp_image[rmbg_text !=0] = color_annotated["object_bound"]
            # # image = cv2.addWeighted(tmp_image, 0.5, image, 0.5, 0)
            # if image_metadata["Design-elements"] != []:
            #     for obj_tmp  in image_metadata["Design-elements"]:
            #         print(obj_tmp)
            #         # cv2.rectangle(image,obj_tmp["location"][0], obj_tmp["location"][1] , color_annotated["design"], 2)
            #         # image_bg[obj_tmp["location"][0][1]:obj_tmp["location"][1][1],obj_tmp["location"][0][0]:obj_tmp["location"][1][0] ] =0
            #         cv2.rectangle(image,obj_tmp["text_location"][0], obj_tmp["text_location"][1] , color_annotated["text_design"], 2)
            #         cv2.putText(image,obj_tmp["text"], [obj_tmp["text_location"][0][0],obj_tmp["location"][0][1] -10],cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
            #         # image_bg[obj_tmp["text_location"][0][1]:obj_tmp["text_location"][1][1],obj_tmp["text_location"][0][0]:obj_tmp["text_location"][1][0] ] =0
            # if image_metadata["Text-normal"] != []:
            #     for obj_tmp  in image_metadata["Text-normal"]:
            #         print(obj_tmp)
            #         cv2.rectangle(image,obj_tmp["location"][0], obj_tmp["location"][1], color_annotated["text_normal"], 2)
            #         # image_bg[obj_tmp["location"][0][1]:obj_tmp["location"][1][1],obj_tmp["location"][0][0]:obj_tmp["location"][1][0] ] =0
            #         cv2.putText(image,obj_tmp["text"], [obj_tmp["location"][0][0],obj_tmp["location"][0][1] -10],cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
            # cv2.imwrite(f"{text_img_folder}/{image_name}_visualize.png", image)
            # bg_only = np.dstack((image, image_bg))
            # # print(bg_only.shape, image.shape, image_bg.shape)
            # cv2.imwrite(f"{text_img_folder}/{image_name}_background.png",bg_only)

            # #Save medata
            final_metadata = {
                "data": image_metadata
            }
            # print(final_metadata)
            # path_final_output = f"{text_img_folder}/{image_name}_metadata.json"
            # with open(path_final_output ,'w') as fp:
            #     json.dump(final_metadata, fp, indent=4, cls=NpEncoder)


DEVICE = "cpu"
ckpt_repo_id = "ShilongLiu/GroundingDINO"
ckpt_filenmae = "/media/anlab/data/GroundingDINO-0.1.0-alpha2/weights/checkpoint0029_4scale_swin.pth"
ckpt_config_filename = "/media/anlab/data/GroundingDINO-0.1.0-alpha2/groundingdino/config/GroundingDINO_SwinL.py"
groundingdino_model = load_model(f"{groundingdino_path}/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                 f"{groundingdino_path}/weights/groundingdino_swint_ogc.pth")
# sam = sam_model_registry["vit_h"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_hq_vit_h.pth")
# # sam = sam_model_registry["vit_l"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_vit_l_0b3195.pth")
# sam.to(device="cpu")
# sam_predictor = SamPredictor(sam)
# dalle_inpainter = DalleInpainter()
# # mask_generator = SamAutomaticMaskGenerator(sam)


BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.15
image_path = "/media/anlab/data/static_image_recognition/adimageset/"
list_image = [os.path.join(image_path, image)
              for image in os.listdir(image_path)]
# model_name_object = "isnet-general-use"
# # model_name_object = "sam"
# sesssion_object = new_session(model_name_object)
# # sys.exit()
# model_name_human = "u2net_human_seg"
# sesssion_human = new_session(model_name_human)

if __name__ == "__main__":
    main()
    # image_path = "/media/anlab/data/static_image_recognition/adimageset/"
    # list_image = [os.path.join(image_path,image) for image in  os.listdir(image_path)]
