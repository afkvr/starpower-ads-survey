from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

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


sam = sam_model_registry["vit_h"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_hq_vit_h.pth")
# sam = sam_model_registry["vit_l"](checkpoint="/media/anlab/data/segment-anything/checkpoints/sam_vit_l_0b3195.pth")
sam.to(device="cpu")
sam_predictor = SamPredictor(sam)

image = cv2.imread('/media/anlab/data/static_image_recognition/adimageset/allbirds6/allbirds6_inpainted.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

sam_predictor.set_image(image)
boxes_filt = torch.tensor([[281, 204, 539,460],
                        [ 36,  77, 228, 348],
                        [ 36,  75, 540, 460],
                        [ 35,  0, 598, 459],
                        [ 38,  0, 599, 124],
                        [ 35,  0, 598, 460],
                        [ 35, 78, 228, 350],
                        [ 35, 78, 228, 350],
                        [281, 202, 539, 459],
                        [  0, 489, 598, 599],
                        [ 35, 0, 598, 460]
                        ])
sam_predictor.set_image(image)
transformed_boxes = sam_predictor.transform.apply_boxes_torch(    boxes_filt, image.shape[:2]).to("cpu")
mask_list, scores, logits_sam = sam_predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes.to("cpu"),
                multimask_output=False,
                # multimask_output=True,
            )

input_box = np.array([[281, 204, 539,460],
                        [ 36,  77, 228, 348],
                        [ 36,  75, 540, 460],
                        [ 35,  0, 598, 459],
                        [ 38,  0, 599, 124],
                        [ 35,  0, 598, 460],
                        [ 35, 78, 228, 350],
                        [ 35, 78, 228, 350],
                        [281, 202, 539, 459],
                        [  0, 489, 598, 599],
                        [ 35, 0, 598, 460]
                        ])
for idx in range(len(input_box)):
    mask_img = torch.zeros(mask_list.shape[-2:])
    mask_img[mask_list[idx].cpu().numpy()[0] == True] = 255
    # masks, _, _ = sam_predictor.predict(
    #     point_coords=None,
    #     point_labels=None,
    #     box=box[None, :],
    #     multimask_output=False,
    # )

    plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # draw_mask(mask_img, plt.gca())
    plt.imshow(mask_img)
    # show_box(box, plt.gca())
    plt.axis('off')
    plt.show()
    # break