from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import json

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def draw_mask(mask,ax):
    color = np.array([1, 1, 1, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)# * color.reshape(1, 1, -1)
    print(mask_image.shape)
    cv2.imwrite("/tmp/mask.png",mask_image*255)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

def main():
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
            print(filename)
            # with open(filename, 'rb') as f:
            #     check_chars = f.read()[-2:]
            text_img_folder = f"{os.path.dirname(filename)}/{os.path.basename(filename).split('.')[0]}"
            image_name = os.path.basename(filename).split(".")[0]
            inpainted_path = f"{text_img_folder}/{image_name}_inpainted.png"
            path_final_output = f"{text_img_folder}/{image_name}_metadata.json"
            with open(path_final_output ,'r') as fp:
                metadata = json.load(fp)
            metadata = metadata["data"]
            if os.path.exists(inpainted_path):
                image = cv2.imread(inpainted_path)
            else:
                image = cv2.imread(filename)
            h, w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            sam_predictor.set_image(image_rgb)
            boxes_filt = torch.tensor([data_tmp["box"] for data_tmp in metadata["Object-bound"]])
            transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to("cpu")
            mask_list, scores, logits_sam = sam_predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_boxes.to("cpu"),
                            multimask_output=False,
                        )

            # input_box = np.array([[281, 204, 539,460],
            #                         [ 36,  77, 228, 348],
            #                         [ 36,  75, 540, 460],
            #                         [ 35,  0, 598, 459],
            #                         [ 38,  0, 599, 124],
            #                         [ 35,  0, 598, 460],
            #                         [ 35, 78, 228, 350],
            #                         [ 35, 78, 228, 350],
            #                         [281, 202, 539, 459],
            #                         [  0, 489, 598, 599],
            #                         [ 35, 0, 598, 460]
            #                         ])
            background_mask = np.zeros([image.shape[0],image.shape[1],1], dtype = "uint8")
            for idx in range(len(metadata["Object-bound"])):
                data_tmp = metadata["Object-bound"][idx]
                box = np.array(data_tmp["box"])
                masks, scores, logits_sam = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box[None, :],
                    multimask_output=False,
                )
                mask_input = logits_sam[np.argmax(scores), :, :]
                masks, _, _ = sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    mask_input=mask_input[None, :, :],
                    box=box[None, :],
                    multimask_output=False,
                )
                # mask_img = torch.zeros(mask_list.shape[-2:])
                # mask_img[mask_list[idx].cpu().numpy()[0] == True] = 255
                # mask_img = mask_img.numpy().reshape(h, w, 1).astype(np.uint8)
                mask_img = masks[0].reshape(h, w, 1).astype(np.uint8)*255
                print(background_mask.shape, mask_img.shape , type(background_mask), type(mask_img))
                # mask_img = cv2.bitwise_not(mask_img)
                # print(background_mask)
                # print(mask_img)
                background_mask = cv2.bitwise_or(mask_img,background_mask)
                
                mask_img_crop = mask_img[box[1]:box[3], box[0]:box[2]]
                output_save_crop = f"""{text_img_folder}/{data_tmp["input_text"]}_{idx}_final.png"""
                crop_image = image[box[1]:box[3], box[0]:box[2]]
                cv2.imwrite(
                    f"""{text_img_folder}/{data_tmp["input_text"]}_{idx}_crop.png""", crop_image)
                cv2.imwrite(
                    f"""{text_img_folder}/{data_tmp["input_text"]}_{idx}_mask.png""", mask_img_crop)
                rgba = np.dstack((crop_image, mask_img_crop))
                cv2.imwrite(output_save_crop, rgba)
                

                # plt.figure(figsize=(10, 10))
                # # plt.imshow(image)
                # draw_mask(masks[0], plt.gca())
                # # plt.imshow(mask_img)
                # # show_box(box, plt.gca())
                # plt.axis('off')
                # plt.show()
            background_mask = cv2.bitwise_not(background_mask)
            output_save_bg = f"""{text_img_folder}/{image_name}_bg.png"""
            rgba = np.dstack((image, background_mask))
            cv2.imwrite(output_save_bg, rgba)
        # break
sam = sam_model_registry["vit_l"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_hq_vit_l.pth")
# sam = sam_model_registry["vit_l"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_vit_l_0b3195.pth")
sam.to(device="cpu")
sam_predictor = SamPredictor(sam)

if __name__ == "__main__":
    main()