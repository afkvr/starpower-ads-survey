import cv2
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
import os
from paddleocr import PaddleOCR
import tesserocr
import torch
import skimage
from skimage.filters import threshold_multiotsu
from skimage.color import label2rgb
from skimage.segmentation import clear_border, expand_labels
from rembg import remove, new_session
from backgroundremover.bg import remove as bgrm
from utils import TextBox
from scipy.cluster.hierarchy import linkage, fcluster
from shapely.geometry import Polygon
from Font_classifier.inference import font_style_predict
import matplotlib as mpl
from collections import defaultdict
import pytesseract
from itertools import chain
from shapely.ops import unary_union
from collections import defaultdict
import easyocr
from shapely.geometry.base import BaseGeometry
from shapely.geometry import Point, Polygon, MultiPolygon, GeometryCollection



PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["TESSDATA_PREFIX"] = (
   f"{PATH}/tessdata/tessdata"
)

# Define the sector_map based on the image
sector_map = [
    ["upper left corner", "upper center edge", "upper right corner"],
    ["up left edge", "up center", "up right edge"],
    ["mid-up left edge", "mid-up center", "mid-up right edge"],
    ["mid-low left edge", "mid-low center", "mid-low right edge"],
    ["low left edge", "low center", "low right edge"],
    ["lower left corner", "lower center edge", "lower right corner"]
]

def uniqueList(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list


def pca_eigenvectors(pts: np.ndarray) -> np.ndarray:
    """
    Returns the principal axes of a set of points.
    Method is essentially running a PCA on the points.

    Parameters
    ----------
    pts : array_like
    """
    ca = np.cov(pts, y=None, rowvar=False, bias=True)
    val, vect = np.linalg.eig(ca)

    return np.transpose(vect)


def oriented_bounding_box(pts: np.ndarray) -> np.ndarray:
    """
    Returns the oriented bounding box width set of points.

    Based on [Create the Oriented Bounding-box (OBB) with Python and NumPy](https://stackoverflow.com/questions/32892932/create-the-oriented-bounding-box-obb-with-python-and-numpy).

    Parameters
    ----------
    pts : array_like
    """
    tvect = pca_eigenvectors(pts)
    rot_matrix = np.linalg.inv(tvect)

    rot_arr = np.dot(pts, rot_matrix)

    mina = np.min(rot_arr, axis=0)
    maxa = np.max(rot_arr, axis=0)
    diff = (maxa - mina) * 0.5

    center = mina + diff

    half_w, half_h = diff
    corners = np.array([
        center + [-half_w, -half_h],
        center + [half_w, -half_h],
        center + [half_w, half_h],
        center + [-half_w, half_h],
    ])

    return np.dot(corners, tvect)


# def polygon_region(image, bbox): 

#     pt1 = (int(bbox[0][0]), int(bbox[0][1]))
#     pt2 = (int(bbox[1][0]), int(bbox[1][1]))
#     pt3 = (int(bbox[2][0]), int(bbox[2][1]))
#     pt4 = (int(bbox[3][0]), int(bbox[3][1]))
    
#     polygon = [pt1, pt2, pt3, pt4]


#     mask = np.zeros((image.shape[0], image.shape[1]))
#     if not type(polygon) == type(np.array([])):
#         polygon = np.array(polygon)

#     cv2.fillConvexPoly(mask, polygon, 1)

#     b_img = image[:,:,0] * mask 
#     g_img = image[:,:,1] * mask 
#     r_img = image[:,:,2] * mask 

#     masked = np.zeros_like(image)
#     masked[:,:,0] = b_img
#     masked[:,:,1] = g_img
#     masked[:,:,2] = r_img

#     return masked, mask 


def polygon_region_image(image, bbox): 

    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
    pt2 = (int(bbox[1][0]), int(bbox[1][1]))
    pt3 = (int(bbox[2][0]), int(bbox[2][1]))
    pt4 = (int(bbox[3][0]), int(bbox[3][1]))
    
    polygon = [pt1, pt2, pt3, pt4]


    mask = np.zeros((image.shape[0], image.shape[1]))
    if not type(polygon) == type(np.array([])):
        polygon = np.array(polygon)

    cv2.fillConvexPoly(mask, polygon, 1)

    b_img = image[:,:,0] * mask 
    g_img = image[:,:,1] * mask 
    r_img = image[:,:,2] * mask 

    # masked = np.zeros_like(image)
    # masked[:,:,0] = b_img
    # masked[:,:,1] = g_img
    # masked[:,:,2] = r_img

    return image, mask 


def rotate_image(image, angle, center):
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def get_rotate_angle(mask):
    blob = np.transpose(np.nonzero(mask)).astype(float)

    mean, eigenvectors, eigenvalues = cv2.PCACompute2(blob, None)

    vectY = np.array(eigenvectors[0])
    vectX = np.array(eigenvectors[1])
    
    center = np.array([int(mean[0,1]), int(mean[0,0])], dtype=np.float64)

    basisX = np.array([1, 0], dtype=np.float32)
    #basisY = np.array([0, 1], dtype=np.float32)

    angle = np.arccos(np.dot(vectX, basisX))*180/np.pi

    if (vectX[1] < 0):
        return angle, center

    return -angle, center

def get_roi(image, bbox):
    pt1 = (int(bbox[0][0]), int(bbox[0][1]))
    pt2 = (int(bbox[1][0]), int(bbox[1][1]))
    pt3 = (int(bbox[2][0]), int(bbox[2][1]))
    pt4 = (int(bbox[3][0]), int(bbox[3][1]))


    polygon = [pt1, pt2, pt3, pt4]
    copy = np.copy(image)

    img, mask = polygon_region_image(copy, polygon)

    rotate_angle, center = get_rotate_angle(mask)

    if (rotate_angle>=-1.5 and rotate_angle<=1.5):
        return image[pt1[1]:pt3[1], pt1[0]:pt3[0]]

    rotated_mask = rotate_image(mask, rotate_angle, center)
    rotated_img  = rotate_image(img, rotate_angle, center)

    region = np.transpose(np.nonzero(rotated_mask))
    top_left = region[0] #+ np.array([1, 1], dtype=np.int64)
    bottom_right = region[len(region)-1] #- np.array([1, 1], dtype=np.int64)
    result = rotated_img[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    H,W = result.shape[:2]
    if H > W: 
        return cv2.rotate(result, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return result

def find_minimum_bounding_rectangle(polygons):
    # Create a shapely MultiPolygon from the list of polygons
    multi_poly = MultiPolygon(polygons)
    combined_poly = unary_union(multi_poly)
    min_bounding_rect = combined_poly.envelope
    
    return min_bounding_rect

def combine_box(image, list_polygon):
    h,w = image.shape[:2]
    threshold_dist = float(np.sqrt((h*0.01)**2 + (w*0.01)**2))
    print("threshold_dist: ", threshold_dist)
    list_polygon = [Polygon(shell=tmp) for tmp in list_polygon]
    list_pair = []
    pair_final = []
    num_polygons = len(list_polygon)
    
    distance_polygon = np.zeros((num_polygons, num_polygons))
    # Calculate distance matrix
    for i in range(num_polygons):
        for j in range(i+1,num_polygons):
            distance_polygon[i,j] = list_polygon[i].distance(list_polygon[j])
    # print("distance_polygon: ", distance_polygon)
    for i in range(num_polygons):
        pair_tmp = [i]
        for j in range(i + 1, num_polygons):
            if distance_polygon[i, j] <= threshold_dist:
                pair_tmp.append(j)
        pair_final.append(pair_tmp)
    pair_final = combine_pairs(pair_final)
    
    combined_box = []
    for i,list_index in enumerate(pair_final):
        list_polygon_tmp = [list_polygon[idx] for idx in list_index]
        # list_polygon_tmp = [np.array(list_polygon[idx].exterior.coords) for idx in list_index]
        # list_polygon_tmp = np.reshape(np.array(list_polygon_tmp), (-1, 2)).astype(np.int32)
        # rec_bounding = oriented_bounding_box(list_polygon_tmp)
        
        rec_bounding = find_minimum_bounding_rectangle(list_polygon_tmp)
        combined_box.append(rec_bounding.exterior.coords)
    
    return combined_box, pair_final

def detect_text_easyOCR(image, save_path_image=False):
    # results = []
    results = reader.readtext(image.copy())
    # image = image[:,:,:3]
    image_raw = image.copy()
    height, width = image.shape[:2]
    
    if results != []:
        list_box = [res[0] for res in results]
        combined_box, pair_final = combine_box(image, list_box)
    
    # print("Pair final: ", pair_final)
    # print("combined_box: ", combined_box)
    
    # padding_size_ratio = 0.125
    if save_path_image != False:
    # Draw box 
        for i,box_ in enumerate(combined_box):
            text = [ results[idx][1] for idx in pair_final[i]]
            text = " ". join(text)
            # box_ = np.array(box_.exterior.coords)
            # print("Box: ", box_)
            
            text_coor = np.array([box_[0][0], max(box_[0][1]-15,0)]).astype(np.int)
            boubox = np.reshape(np.array(box_), (-1, 1, 2)).astype(np.int32)
            # image = cv2.polylines(image.copy(), pts=[boubox], isClosed=True, color=(0,255,0), thickness=2)
            # image = cv2.putText(image,text , text_coor , cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2,cv2.LINE_AA) 
            # topleft = np.array(box_[0],dtype=np.int32)
            # bottomright = np.array(box_[2],dtype=np.int32)
            # padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
            # padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
            # padding_topleft = [
            #     max(0, topleft[0] - padding_X),
            #     max(0, topleft[1] - padding_Y),
            # ]
            # padding_bottomright = [
            #     min(width-1, bottomright[0] + padding_X),
            #     min(height-1, bottomright[1] + padding_Y),
            # ]
            # cv2.rectangle(image, np.array(padding_topleft,dtype=np.int32), np.array(padding_bottomright,dtype=np.int32), (255,255,0), 2)
            # crop = image_raw[
            # padding_topleft[1] : padding_bottomright[1],
            # padding_topleft[0] : padding_bottomright[0],                ] #
            # crop = get_roi(image_raw, box_)
            # cv2.imwrite(os.path.join(os.path.dirname(save_path_image),text+".png"), crop)
            # save_crop
            # # Save result
        # cv2.imwrite(save_path_image,image)
            # image.save(save_path_image)
    
    return [results]

def check_image(image,alpha_color=(255, 255, 255)):
    if image.shape[2] ==4:
        # image = image[:,:,:3] 
        B, G, R, A = cv2.split(image)
        alpha = A / 255
        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        image = cv2.merge((B, G, R))
    if isinstance(image, np.ndarray) and len(image.shape) == 2:
        print("Image is gray! Try to convert BRG")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


# Union-Find/Disjoint-Set class
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self.parent[root_v] = root_u
    def add(self, u):
        if u not in self.parent:
            self.parent[u] = u

def combine_pairs(pairs):
    uf = UnionFind()
    # Add all pairs to the union-find structure
    for pair in pairs:
        for num in pair:
            uf.add(num)
        first = pair[0]
        for num in pair[1:]:
            uf.union(first, num)
    # Collect components
    components = defaultdict(list)
    for num in uf.parent:
        root = uf.find(num)
        components[root].append(num)
    # Convert components to list of lists and sort each list
    return [sorted(component) for component in components.values()]



def get_sector_frame(width_frame,height_frame, sector_map, rectangle):
    # Define the sector_map dimensions
    rows, cols = 6, 3
    row_height = height_frame / rows
    col_width = width_frame / cols
    # Extract rectangle coordinates
    top_left, bottom_right = rectangle
    x1, y1 = top_left
    x2, y2 = bottom_right
    # Initialize an empty set to store intersecting regions
    intersecting_regions = set()
    # Iterate through the sector_map and check intersections
    for i in range(rows):
        for j in range(cols):
            region_top_left = (j * col_width, i * row_height)
            region_bottom_right = ((j + 1) * col_width, (i + 1) * row_height)
            if not (x2 < region_top_left[0] or x1 > region_bottom_right[0] or y2 < region_top_left[1] or y1 > region_bottom_right[1]):
                intersecting_regions.add(sector_map[i][j])
    return list(intersecting_regions)



def convert_to_bbox(box, w, h):
    center_x, center_y, width, height = np.array(box * [w, h, w, h]).astype(np.int)
    x_min = max(int(center_x - width / 2), 0)
    y_min = max(int(center_y - height / 2), 0)
    x_max = min(int(center_x + width / 2), w)
    y_max = min(int(center_y + height / 2), h)
    return [x_min, y_min, x_max, y_max]


def cal_interArea(a, b):  # returns None if rectangles don't intersect
    # print(a[0][0], b[0][0],a[0], b[0] )
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def calculate_IoU_box(boxA, boxB):
    interArea = cal_interArea(boxA, boxB)
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (
        (interArea / float(boxAArea + boxBArea - interArea))
        if (float(boxAArea + boxBArea - interArea) != 0)
        else (interArea / 0.0001)
    )

    # Return the intersection over union value
    return iou


def checkInnerBox(boxMajor, box_check, threshold=0.8):
    # Compute the area of intersection rectangle
    interArea = cal_interArea(boxMajor, box_check)
    # Compute the area of both the prediction and ground-truth rectangles
    box_checkArea = (box_check[2] - box_check[0]) * (box_check[3] - box_check[1])

    # iou = (interArea / float( box_checkArea - interArea)) if (float(box_checkArea - interArea) !=0) else (interArea /0.0001)
    iou = interArea / float(box_checkArea)
    # print("checkInnerBox: ", interArea,box_checkArea, iou)
    return iou >= threshold


def remove_inner_regions(image):
    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(
        image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    # Create an empty mask to draw the filtered contours
    filtered_mask = np.zeros_like(image)
    contours_mask = np.zeros_like(image)
    # Draw only the outer contours (those not inside another contour)
    for i in range(len(contours)):
        if (
            hierarchy[0][i][3] != -1 or cv2.contourArea(contours[i]) < 50
        ):  # If the contour has no parent or area is too small
            cv2.drawContours(
                filtered_mask, contours, i, 255, thickness=-1
            )  # Draw contour
            if cv2.contourArea(contours[i]) >= 50:
                cv2.drawContours(contours_mask, contours, i, 255, thickness=1)

    return image - filtered_mask + contours_mask


def detect_font_detail(crop_img, mask, text):
    # Detect font weight, size
    tesseract_model.SetImage(Image.fromarray(crop_img))
    if tesseract_model.Recognize():
        iterator = tesseract_model.GetIterator()
        iterator = iterator.WordFontAttributes()
        font_size = int(iterator["pointsize"]) if iterator["pointsize"] > 5 else 20
    else:
        font_size = 20
    # cv2.imwrite("/tmp/test_crop.png",crop_img)
    font_style, font_style_score = font_style_predict(crop_img)
    # detect font weight
    font_weight = ""

    # detect capitalize
    text_raw = text
    if text_raw.isupper():
        font_cap = "ALL CAPS"
    elif text_raw.istitle():
        font_cap = "Start Case"
    elif text_raw.islower():
        font_cap = "no caps"
    else:
        font_cap = "standard case"

    text_color_hex = ""
    ##Detect text color
    # if mask.sum() != 0:
    #     img_text = np.where(cv2.merge([mask, mask, mask]), crop_img, 0)
    #     img_text_pil = Image.fromarray(img_text)
    #     text_color = sorted(
    #         img_text_pil.getcolors(img_text_pil.size[0] * img_text_pil.size[1])
    #     )
    #     # cv2.imwrite(f"/tmp/text_{name}_only.png", img_text)
    #     if text_color[-2][0] <= mask.sum() * color_threshold:
    #         text_color_tmp = (
    #             int(img_text[:, :, 2].sum() / mask.sum()),
    #             int(img_text[:, :, 1].sum() / mask.sum()),
    #             int(img_text[:, :, 0].sum() / mask.sum()),
    #         )
    #         text_color_hex = hex = "#%02x%02x%02x" % text_color_tmp
    #     else:
    #         text_color_hex = hex = "#%02x%02x%02x" % text_color[-2][1]

    font_details = {
        "font-style": font_style,  # 3-g-i: Font style (closest match to font) (Link font returned by whatfontis)
        "font-size": font_size,  # 3-g-ii: Font size (in pt)
        "serif": True,  # 3-g-iii: True is Serif | False is Sans Serif
        "font-weight": font_weight,  # 3-g-iv: Font weight (bolded, unbolded)
        "cap-style": font_cap,  # 3-g-v: Capitalization style - output is 1 in 4 types: "standard case"/"Start Case"/"ALL CAPS"/"no caps"
        "font-color": text_color_hex,  # 3-g-vi: Font color (save as format HEX)
        "font-outline":{                                   # 3-g-vii: Font outline (yes or no). If no => result as None/null. 
                "color":"#001400",                             # If yes => output would has field "color"
            },
        "font-outline-weight":3
    }

    return font_details


def get_freq(array, exclude):
    count = np.bincount(array[array != exclude])
    if count.size == 0:
        return exclude
    else:
        return np.argmax(count)


def iou_mask(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = cv2.bitwise_or(mask1, mask2).sum()
    return intersection / union


def find_contour_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 30, 200)
    contours, hierarchy = cv2.findContours(
        edged, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered_mask = np.zeros_like(gray)
    # print(len(contours))
    for i in range(len(contours)):
        if (
            not cv2.contourArea(contours[i]) < 20
        ):  # If the contour has no parent or area is too small
            cv2.drawContours(
                filtered_mask, contours, i, 255, thickness=-1
            )  # Draw contour
    cv2.imwrite("/tmp/contours.jpg", filtered_mask)
    label_mask = skimage.measure.label(filtered_mask, background=0, connectivity=2)
    # label_mask[label_mask >0] = 255
    cv2.imwrite("/tmp/label_mask.jpg", label_mask)
    # labelrgb = label2rgb(label_mask)
    # print(labelrgb.shape)
    # cv2.imwrite("/tmp/label_rgb.jpg", labelrgb)
    return label_mask

def distance_between_polygons(poly1, poly2):
    # Convert the list of points to a numpy array
    poly1 = np.array(poly1, dtype=np.int32)
    poly2 = np.array(poly2, dtype=np.int32)
    rect1 = cv2.boundingRect(poly1)
    rect2 = cv2.boundingRect(poly2)
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    # print(rect1, rect2)
    # Calculate the vertical distance
    vertical_distance = y2 - (y1 + h1) if (y1 < y2) else y1 - (y2 + h2)
    horizontal_distance = x2 - (x1 + w1) if (x1 < x2) else x1 - (x2 + w2)

    # Ensure the distances are non-negative
    vertical_distance = max(0, vertical_distance)
    horizontal_distance = max(0, horizontal_distance)
    
    return vertical_distance, horizontal_distance

def combine_text(image, result, threshold= 0.01):
    list_text_raw = []
    list_text_final = []
    list_polygon = []
    height, width = image.shape[:2]
    padding_size_ratio = 0.1 # extend text box by 10% of minimum edge.
    
    #Scan text content and location
    for line in result[0]:
        # count +=1
        boxes = line[0]
        txts = line[1][0]
        score = round(float(line[1][1]), 5)
        if score < 0.5:  #Remove text has confidence score lower than 0.5
            continue
        
        #Conver polygon point to rectangle
        rect1 = cv2.boundingRect(np.array(boxes, dtype=np.int32))
        x,y,w_,h_ = rect1
        boxes = [[x,y],[x+w_,y],[x+w_,y+h_],[x,y+h_]]
        topleft = np.array(boxes[0]).astype(int)
        bottomright = np.array(boxes[2]).astype(int)
        # box_w = int(bottomright[0] - topleft[0])
        # box_h = int(bottomright[1] - topleft[1])
        
        padding_pixel = int(min(w_, h_)*padding_size_ratio)
        padding_topleft = [
            max(0, topleft[0] - padding_pixel),
            max(0, topleft[1] - padding_pixel),
        ]
        padding_bottomright = [
            min(width-1, bottomright[0] + padding_pixel),
            min(height-1, bottomright[1] + padding_pixel),
        ]
        list_text_raw.append({
            "text-content":txts,
            "box":[padding_topleft, padding_bottomright]
        })
        list_polygon.append([padding_topleft, padding_bottomright])
    num_polygons = len(list_polygon)
    #Group neighbor text
    if num_polygons >1:
        distances_vert = np.zeros((num_polygons, num_polygons))
        distances_horiz = np.zeros((num_polygons, num_polygons))
        # Calculate distance matrix
        for i in range(num_polygons):
            for j in range(i + 1, num_polygons):
                vert_dist, horiz_dist = distance_between_polygons(list_polygon[i], list_polygon[j])
                # rint(i,j, vert_dist, horiz_dist)
                distances_vert[i, j] = vert_dist / height
                distances_horiz[i, j] = horiz_dist / width
        pair_final = []
        # Calculate distance matrix
        for i in range(num_polygons):
            pair_tmp = [i]
            for j in range(i + 1, num_polygons):
                if distances_vert[i, j] <= threshold and  distances_horiz[i, j] <= threshold :
                    pair_tmp.append(j)
            pair_final.append(pair_tmp)
        
        #Combine pair
        list_polygon_box = {}
        pair_final = combine_pairs(pair_final)
        for i,list_index in enumerate(pair_final):
            element_tmp = {
                "text-content": "",
                "box":[]
            }
            #Combine box
            list_polygon_box[str(i)] = [[0,0]]
            text_content_tmp = []
            sector_tmp = []
            for idx in list_index:
                list_polygon_box[str(i)] = np.concatenate((list_polygon_box[str(i)],list_polygon[idx]))
                text_content_tmp.append(list_text_raw[idx]["text-content"])
            rect_tmp = cv2.boundingRect(np.array(list_polygon_box[str(i)][1:], dtype=np.int32))
            x1, y1, w1, h1 = rect_tmp
            start_p = [x1,y1]
            end_p = [x1+w1,y1+h1]
            # list_polygon_box[str(i)] = 
            crop_img = image[start_p[1]:end_p[1], start_p[0]:end_p[0]]
            #Combine metadata
            element_tmp["text-content"] = " ".join(text_content_tmp)
            element_tmp["box"] = [start_p, end_p]
            
            list_text_final.append(element_tmp)
    else:
        list_text_final = list_text_raw
    return list_text_final

def track_design_elements(image, result, image_metadata, text_img_folder):
    global color_threshold, outline_threshold, te
    height, width = image.shape[:2]
    color_threshold = 0.6
    outline_threshold = 0.5
    image_contours = find_contour_image(image)
    # cv2.imwrite(f"/tmp/contours.jpg",label2rgb(image_contours))
    list_combine_text = combine_text(image, result)
    for idx, text_data in enumerate(list_combine_text):
        txts = text_data["text-content"]
        topleft, bottomright = text_data["box"]
        boxes = [topleft,
                 [bottomright[0], topleft[1]],
                bottomright,
                 [topleft[0], bottomright[1]]]
        box_w = int(bottomright[0] - topleft[0])
        box_h = int(bottomright[1] - topleft[1])
        box_size = box_w*box_h
        block_coordinates = [[round(box_[0]/width,5), round(box_[1]/height,5)] for box_ in boxes]
        sectors_list = get_sector_frame(width, height,sector_map,[boxes[0],boxes[2]])
        print(topleft, bottomright)
        crop_img = image[
            topleft[1] : bottomright[1],
            topleft[0] : bottomright[0]
        ]
        # print(crop_img.shape, padding_topleft, padding_bottomright,padding_topleft[1] - padding_bottomright[1] )
        # Redetect OCR
        # result_tmp = ocr.ocr(cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY), cls=0)
        # if result_tmp != [None]:
        #     # print(result_tmp)
        #     txts = ""
        #     for line in result_tmp[0]:
        #         txts += line[1][0]
        ## Detect font-detail
        # cv2.imwrite(f"{text_img_folder}/{image_name}_{txts}.jpg", crop_img)
        crop_img_tmp = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        thresh_sauvola = threshold_multiotsu(crop_img_tmp, classes=3)
        regions = np.digitize(crop_img_tmp, bins=thresh_sauvola)
        # cv2.imwrite(f"/tmp/text_{name}_label.png", regions)
        crop_img_rmbg = remove(crop_img)
        rmbg_mask = crop_img_rmbg[:, :, 3]
        ret, rmbg_text = cv2.threshold(rmbg_mask, 100, 255, cv2.THRESH_BINARY)
        # cv2.imwrite(f"/tmp/{txts}_rmbg_mask.jpg", rmbg_text)
        rmbg_mask = rmbg_text
        rmbg_mask[rmbg_mask > 0] = 1

        ## Select highest index as text label > Remove edge-connected components > get text color
        mask = np.uint8(np.equal(regions, 2))
        inv_mask = 1 - mask
        iou_1 = iou_mask(rmbg_mask, clear_border(mask))
        iou_2 = iou_mask(rmbg_mask, clear_border(inv_mask))
        # print(iou_1, iou_2)

        if iou_2 > iou_1:
            mask = inv_mask
        # cv2.imwrite(f"/tmp/{txts}_inv_mask.png", inv_mask*255)
        mask = cv2.bitwise_and(clear_border(mask), rmbg_mask)
        # cv2.imwrite(f"/tmp/{txts}_mask.png", mask*255)

        ## Reduce noise and inner region
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
        # mask = remove_inner_regions(mask * 255)
        mask =  mask*255
        mask[mask < 200] = 0
        mask[mask != 0] = 1
        # cv2.imwrite(f"/tmp/{txts}_mask_1.png", mask*255)
        font_details = detect_font_detail(crop_img, mask, txts)
        # cv2.imwrite(f"/tmp/{txts}_mask_1_not.png", cv2.bitwise_not(mask*255))
        ## Detect design elements
        # Check mask boundary
        mask_args = np.where(mask != 0)  # , topleft)
        mask_args = [
            np.add(mask_args[0], topleft[0] - 1),
            np.add(mask_args[1], topleft[1] - 1),
        ]

        # Get boundary most color
        # print(image_contours.shape, mask.shape)
        contour_match = np.where(cv2.bitwise_not(mask*255), image_contours[topleft[1] : bottomright[1],topleft[0] : bottomright[0]], 0)#np.array(image_contours[mask_args])
        # print(contour_match.shape)
        # image_contours_tmp = np.zeros_like(image_contours)
        # image_contours_tmp[ padding_topleft[1] : padding_bottomright[1],padding_topleft[0] : padding_bottomright[0]] = cv2.bitwise_not(mask*255)
        # cv2.imwrite(f"/tmp/{txts}_contours.png", contour_match)
        contour_label_match = get_freq(contour_match, 0)
        # print(contour_label_match)
        if contour_label_match == 0:  # 0 is background
            image_metadata["Elements"].append(
                {
                    "category":"Text",
                    "text-content": txts,
                    # "location": [topleft.tolist(), bottomright.tolist()],
                    "font-details": font_details,
                    "size-objective":[box_w,box_h],                                # 3-c: Size (objective), rectangle -  Format [weight, height] of box text
                    "size-percentage-of-frame":round(box_w*box_h/(height*width),5),                              # 3-d: Size (% of video frame)- Fomula - box_text_area / frame_area
                    "location-objective":block_coordinates,     # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                    "location-sector":sectors_list
                }
            )
            text_normal = image[
                topleft[1] : bottomright[1], topleft[0] : bottomright[0]
            ]
            cv2.imwrite(f"{text_img_folder}/{txts}_text_normal_{idx}.png", text_normal)
            continue
        contour_label_match_mask = np.uint8(
            np.equal(image_contours, contour_label_match)
        )
        # print(type(contour_label_match_mask))
        # cv2.imwrite(f"/tmp/{txts}.png", contour_label_match_mask*255)
        # Check mask is small than crop text image
        if contour_label_match_mask.sum() < box_size:
            image_metadata["Elements"].append(
                {   
                    "category":"Text",
                    "text-content": txts,
                    # "location": [topleft.tolist(), bottomright.tolist()],
                    "font-details": font_details,
                    "size-objective":[box_w,box_h],                                # 3-c: Size (objective), rectangle -  Format [weight, height] of box text
                    "size-percentage-of-frame":round(box_w*box_h/(height*width),5),                              # 3-d: Size (% of video frame)- Fomula - box_text_area / frame_area
                    "location-objective":block_coordinates,     # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                    "location-sector":sectors_list
                }
            )
            text_normal = image[
                topleft[1] : bottomright[1], topleft[0] : bottomright[0]
            ]
            cv2.imwrite(f"{text_img_folder}/{txts}_text_normal_{idx}.png", text_normal)
            continue
        # print()
        # print(txts, "Design element!")
        mask_args = np.where(image_contours == contour_label_match)
        min_x = int(min(mask_args[1]))
        max_x = int(max(mask_args[1]))
        min_y = int(min(mask_args[0]))
        max_y = int(max(mask_args[0]))
        image_metadata["Elements"].append(
            {   
                "category":"Text",
                "text-content": txts,
                # "text-location": [topleft.tolist(), bottomright.tolist()],
                "font-detail": font_details,
                "size-objective":[box_w,box_h],                                # 3-c: Size (objective), rectangle -  Format [weight, height] of box text
                "size-percentage-of-frame":round(box_w*box_h/(height*width),5),                              # 3-d: Size (% of video frame)- Fomula - box_text_area / frame_area
                "location-objective":block_coordinates,     # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                "location-sector":sectors_list
            })
        
        
        image_metadata["Elements"].append( {   
                "category":"Design element",
                "location-objective": [[round(min_x/width, 5), round(min_y/height, 5)], [round(max_x/width, 5), round(max_y/height, 5)]],
                })
            
        text_in_design_img = image[
                topleft[1] : bottomright[1], topleft[0] : bottomright[0]
            ]
        cv2.imwrite(f"{text_img_folder}/{txts}_text_in_design_{idx}.png", text_in_design_img)
        continue
        ## Detect outlint color
        # if mask.sum() != 0:
        #     ## Enpand label mask for detect outline. > Find most color > get color mask > check connect border > compare with text boundary (threshold) > get color/weight
        #     text_boundary = np.uint8(expand_labels(mask,distance=3))

        #     text_boundary = text_boundary - mask
        #     # cv2.imwrite(f"/tmp/text_{name}_boundary.png", text_boundary*255)
        #     # print(text_boundary.shape, text_boundary.max(), mask.max())
        #     text_boundary_img = np.where(cv2.merge([text_boundary, text_boundary,text_boundary]), crop_img, 0)
        #     # cv2.imwrite(f"/tmp/text_{name}_boundary_color.png", text_boundary_img)
        #     text_boundary_img_pil = Image.fromarray(text_boundary_img)
        #     text_boundary_color = sorted(text_boundary_img_pil.getcolors(text_boundary_img_pil.size[0]*text_boundary_img_pil.size[1]))
        #     if text_boundary_color[-2][0] <= text_boundary.sum()*color_threshold:
        #         text_boundary_color = (int(text_boundary_img[:,:,2].sum()/text_boundary.sum()),
        #                     int(text_boundary_img[:,:,1].sum()/text_boundary.sum()),
        #                     int(text_boundary_img[:,:,0].sum()/text_boundary.sum()))
        #     else:
        #         text_boundary_color = text_boundary_color[-2][1]
        #     print("text_boundary_color: ", text_boundary_color)
        #     text_boundary_color_hex = hex = ('#%02x%02x%02x' % text_boundary_color)
        #     # Define the range of the background color
        #     tolerance = 50
        #     # Define the lower and upper bounds for the color range
        #     lower_bound = np.clip(np.array(text_boundary_color) - tolerance, 0, 255)
        #     upper_bound = np.clip(np.array(text_boundary_color) + tolerance, 0, 255)
        #     # print("lower_bound: ", lower_bound)
        #     # print("upper_bound: ", upper_bound)
        #     text_boundary_color_scan = cv2.inRange(crop_img, lower_bound, upper_bound)
        #     text_boundary_color_scan = clear_border(text_boundary_color_scan)
        #     # cv2.imwrite(f"/tmp/text_{name}_boundary_scan.png", text_boundary_color_scan)
        #     text_boundary_color_scan = cv2.bitwise_and(text_boundary_color_scan, rmbg_mask)
        #     cv2.imwrite(f"{text_img_folder}/{image_name}_{txts}_boundary_final.png", text_boundary_color_scan*255)

def group_design_elements_bk(image, image_metadata,threshold = 0.05):
    height, width = image.shape[:2]
    list_polygon = []
    group_tmp = {}
    print(image_metadata)
    for tmp in image_metadata["Elements"]:
        print(tmp["location-objective"])
        list_polygon.append([np.array(coor)*np.array([width,height]) for coor in tmp["location-objective"]])
    
    num_polygons = len(list_polygon)
    if num_polygons >1:
        distances = np.zeros((num_polygons, num_polygons))

        # Calculate distance matrix
        for i in range(num_polygons):
            for j in range(i + 1, num_polygons):
                vert_dist, horiz_dist = distance_between_polygons(list_polygon[i], list_polygon[j])
                print(vert_dist, horiz_dist)
                distances[i, j] = max(vert_dist / height, horiz_dist / width)
                distances[j, i] = distances[i, j]
        
        # print("distances: ", distances)
        # # Apply hierarchical clustering
        # Z = linkage(distances, 'single')
        # print("Z: ", Z)
        # # Form clusters with a threshold of 5% of the image dimensions
        # # threshold = 0.05
        # clusters = fcluster(Z, threshold, criterion='distance')
        group_index_tmp = {}
        group_num = 0
        # Calculate distance matrix
        for i in range(num_polygons):
            if not str(i) in group_index_tmp.keys():
                group_num +=1
                group_index_tmp[str(i)] = group_num
            for j in range(i + 1, num_polygons):
                if distances[i, j] <= threshold:
                    # print(i,j)
                    group_index_tmp[str(j)] = group_index_tmp[str(i)]
        clusters = list(map(str, clusters))
        print("Cluster: ", clusters)
        for idx in range(len(clusters)):
            print(clusters[idx], group_tmp.keys(), clusters[idx] in list(group_tmp.keys()))
            if not clusters[idx] in group_tmp.keys():
                group_tmp[clusters[idx]] = [image_metadata["Elements"][idx]]
            else:
                group_tmp[clusters[idx]].append(image_metadata["Elements"][idx])
    image_metadata["Group"] = group_tmp
    # return group_tmp
    print("Group", group_tmp)

def combine_processed_text(image, list_text_raw,text_img_folder_2, threshold = 0.01):
    if image.shape[2] ==4:
        image = image[:,:,:3]
    name = "viridis"
    cmap = mpl.colormaps[name]  
    colors = cmap.colors 
    height, width = image.shape[:2]
    list_polygon = []
    group_tmp = {}
    # print(image_metadata["Elements"])
    for i,tmp in enumerate( list_text_raw):
        list_polygon.append([np.array(coor)*np.array([width,height]) for coor in tmp["location-objective"]])
        # try: 
        #     print(i , " - ", tmp["text-content"], " - ",list_polygon[-1] )
        # except Exception as e:
        #     print(i , " - Object - ",list_polygon[-1] )
    num_polygons = len(list_polygon)
    list_text_final = []
    if num_polygons >1:
        distances_vert = np.zeros((num_polygons, num_polygons))
        distances_horiz = np.zeros((num_polygons, num_polygons))
        # Calculate distance matrix
        for i in range(num_polygons):
            for j in range(i + 1, num_polygons):
                vert_dist, horiz_dist = distance_between_polygons(list_polygon[i], list_polygon[j])
                # rint(i,j, vert_dist, horiz_dist)
                distances_vert[i, j] = vert_dist / height
                distances_horiz[i, j] = horiz_dist / width
        
        group_index_tmp = {}
        pair_final = []
        
        # Calculate distance matrix
        for i in range(num_polygons):
            pair_tmp = [i]
            for j in range(i + 1, num_polygons):
                # if not str(i) in group_index_tmp.keys() and not():
                #     group_index_tmp[str(i)] = group_num
                #     group_num +=1
                # print(i,j, distances_vert[i, j], distances_horiz[i, j])
                if distances_vert[i, j] <= threshold and  distances_horiz[i, j] <= threshold :
                    pair_tmp.append(j)
                    # print("Match: ", i,j)
                    # if str(j) in group_index_tmp.keys() and not  str(i) in group_index_tmp.keys():
                    #     group_index_tmp[str(i)] = group_index_tmp[str(j)]
                    # elif not str(j) in group_index_tmp.keys() and str(i) in group_index_tmp.keys():
                    #     group_index_tmp[str(j)] = group_index_tmp[str(i)]
            
            pair_final.append(pair_tmp)
        
        #Combine pair
        list_polygon_box = {}
        pair_final = combine_pairs(pair_final)
        for i,list_index in enumerate(pair_final):
            
            element_tmp = {
                "text-content": "",
                "font-details": "",
                "size-objective": [],
                "size-percentage-of-frame": 0,
                "location-objective": [],
                "location-sector": []
            }
            #Combine box
            list_polygon_box[str(i)] = [[0,0]]
            text_content_tmp = []
            sector_tmp = []
            for idx in list_index:
                group_index_tmp[str(idx)] = i+1
                list_polygon_box[str(i)] = np.concatenate((list_polygon_box[str(i)],list_polygon[idx]))
                if "text-content" in list_text_raw[idx].keys():
                    text_content_tmp.append(list_text_raw[idx]["text-content"])
                    # sector_tmp.append(list_text_raw[idx]["location-sector"])
            rect_tmp = cv2.boundingRect(np.array(list_polygon_box[str(i)][1:], dtype=np.int32))
            x1, y1, w1, h1 = rect_tmp
            start_p = [x1,y1]
            end_p = [x1+w1,y1+h1]
            list_polygon_box[str(i)] = [start_p, end_p]
            crop_img = image[start_p[1]:end_p[1], start_p[0]:end_p[0]]
            #Combine metadata
            element_tmp["text-content"] = " ".join(text_content_tmp)
            element_tmp["font-details"] = detect_font_detail(crop_img, None, element_tmp["text-content"])
            element_tmp["size-objective"] = [w1,h1]
            element_tmp["size-percentage-of-frame"] = round(w1*h1/(height*width),5)
            element_tmp["location-objective"] = [[round(start_p[0]/width,5), round(start_p[1]/height,5)],
                                                 [round(end_p[0]/width,5), round(start_p[1]/height,5)],
                                                 [round(end_p[0]/width,5), round(end_p[1]/height,5)],
                                                 [round(start_p[0]/width,5), round(end_p[1]/height,5)]]
            element_tmp["location-sector"] = get_sector_frame(width, height, sector_map, [start_p, end_p])
            #Save crop image
            cv2.imwrite(os.path.join(text_img_folder_2, element_tmp["text-content"].replace(" ","_")+".png"), crop_img)
            # print(os.path.join(text_img_folder_2, element_tmp["text-content"]+".png"))
            list_text_final.append(element_tmp)
        # print("group_index_tmp: ", group_index_tmp)
        # image_tmp = image.copy()#cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        # image_tmp_2 =image.copy()# cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        group_num = len(pair_final)
        
        
        
        ##Visual box
        # for idx in range(len(list_polygon)):
        #     group_index = group_index_tmp[str(idx)]-1
        #     # print(int(group_index*len(colors)/group_num))
        #     color_tmp = tuple([tmp*255 for tmp in colors[int(group_index*len(colors)/group_num)-1]])
        #     # print("list_polygon: ", list_polygon)
        #     rect_tmp = cv2.boundingRect(np.array(list_polygon[idx], dtype=np.int32))
        #     x1, y1, w1, h1 = rect_tmp
        #     start_p = [x1,y1]
        #     end_p = [x1+w1,y1+h1]
            # print(x1, y1, w1, h1)
            # start_p = list_polygon[idx][0].astype(int)
            # end_p = list_polygon[idx][2].astype(int) if len(list_polygon[idx]) == 4 else  list_polygon[idx][1].astype(int)
        #     cv2.rectangle(image_tmp, start_p, end_p, color_tmp, -1)
        #     cv2.rectangle(image_tmp, list_polygon_box[str(group_index)][0], list_polygon_box[str(group_index)][1], color_tmp, 3)
        #     reversed_color = tuple([255-tmp for tmp in color_tmp])
        #     # print(type((2, 24, 219)))
        #     cv2.putText(image_tmp, f"{group_index_tmp[str(idx)]}", [int((start_p[0]+end_p[0])/2), int((start_p[1]+end_p[1])/2)+10], cv2.FONT_HERSHEY_SIMPLEX,0.7,reversed_color,2,cv2.LINE_AA) 
        # cv2.putText(image_tmp, f"{round(threshold*100)}%", [10, 30], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 
        # cv2.imwrite(f"/tmp/{image_name}_grouping.png", image_tmp[:,:,:3])
        
        
        # for idx in range(len(list_polygon)):
        #     group_index = group_index_tmp[str(idx)]
        #     if not group_index in group_tmp.keys():
        #         group_tmp[group_index] = [list_text_raw[idx]]
        #     else:
        #         group_tmp[group_index].append(list_text_raw[idx])
    else: 
        list_text_final = list_text_raw
    # image_metadata["Group"] = group_tmp
    return list_text_final


def group_design_elements(image, image_metadata, threshold = 0.05):
    name = "viridis"
    cmap = mpl.colormaps[name]  
    colors = cmap.colors 
    height, width = image.shape[:2]
    list_polygon = []
    group_tmp = {}
    # print(image_metadata["Elements"])
    for i,tmp in enumerate( image_metadata["Elements"]):
        list_polygon.append([np.array(coor)*np.array([width,height]) for coor in tmp["location-objective"]])
    num_polygons = len(list_polygon)
    if num_polygons >1:
        distances_vert = np.zeros((num_polygons, num_polygons))
        distances_horiz = np.zeros((num_polygons, num_polygons))
        # Calculate distance matrix
        for i in range(num_polygons):
            for j in range(i + 1, num_polygons):
                vert_dist, horiz_dist = distance_between_polygons(list_polygon[i], list_polygon[j])
                # rint(i,j, vert_dist, horiz_dist)
                distances_vert[i, j] = vert_dist / height
                distances_horiz[i, j] = horiz_dist / width
        
        group_index_tmp = {}
        # group_num = 1
        pair_final = []
        # Calculate distance matrix
        for i in range(num_polygons):
            pair_tmp = [i]
            for j in range(i + 1, num_polygons):
                if distances_vert[i, j] <= threshold and  distances_horiz[i, j] <= threshold :
                    pair_tmp.append(j)
            
            pair_final.append(pair_tmp)
        
        #Combine pair
        list_polygon_box = {}
        pair_final = combine_pairs(pair_final)
        for i,list_index in enumerate(pair_final):
            list_polygon_box[str(i)] = [[0,0]]
            for idx in list_index:
                group_index_tmp[str(idx)] = i+1
                list_polygon_box[str(i)] = np.concatenate((list_polygon_box[str(i)],list_polygon[idx]))
            rect_tmp = cv2.boundingRect(np.array(list_polygon_box[str(i)][1:], dtype=np.int32))
            x1, y1, w1, h1 = rect_tmp
            start_p = [x1,y1]
            end_p = [x1+w1,y1+h1]
            list_polygon_box[str(i)] = [start_p, end_p]
        # print("group_index_tmp: ", group_index_tmp)
        image_tmp = image.copy()#cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        # image_tmp_2 =image.copy()# cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        group_num = len(pair_final)
        # for idx in range(len(list_polygon)):
        #     group_index = group_index_tmp[str(idx)]-1
        #     # print(int(group_index*len(colors)/group_num))
        #     color_tmp = tuple([tmp*255 for tmp in colors[int(group_index*len(colors)/group_num)-1]])
        #     # print("list_polygon: ", list_polygon)
        #     rect_tmp = cv2.boundingRect(np.array(list_polygon[idx], dtype=np.int32))
        #     x1, y1, w1, h1 = rect_tmp
        #     start_p = [x1,y1]
        #     end_p = [x1+w1,y1+h1]
        #     # print(x1, y1, w1, h1)
        #     # start_p = list_polygon[idx][0].astype(int)
        #     # end_p = list_polygon[idx][2].astype(int) if len(list_polygon[idx]) == 4 else  list_polygon[idx][1].astype(int)
        #     cv2.rectangle(image_tmp, start_p, end_p, color_tmp, -1)
        #     cv2.rectangle(image_tmp, list_polygon_box[str(group_index)][0], list_polygon_box[str(group_index)][1], color_tmp, 3)
        #     reversed_color = tuple([255-tmp for tmp in color_tmp])
        #     # print(type((2, 24, 219)))
        #     cv2.putText(image_tmp, f"{group_index_tmp[str(idx)]}", [int((start_p[0]+end_p[0])/2), int((start_p[1]+end_p[1])/2)+10], cv2.FONT_HERSHEY_SIMPLEX,0.7,reversed_color,2,cv2.LINE_AA) 
        # cv2.putText(image_tmp, f"{round(threshold*100)}%", [10, 30], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 
        # cv2.imwrite(f"/tmp/{image_name}_grouping.png", image_tmp[:,:,:3])
        for idx in range(len(list_polygon)):
            group_index = group_index_tmp[str(idx)]
            if not group_index in group_tmp.keys():
                group_tmp[group_index] = [image_metadata["Elements"][idx]]
            else:
                group_tmp[group_index].append(image_metadata["Elements"][idx])
    
    image_metadata["Group"] = group_tmp
    # return list_polygon_box
    # print("Group", group_tmp)

def combine_images(columns, space, images):
    rows = len(images) // columns
    if len(images) % columns:
        rows += 1
    width_max = max([Image.fromarray(image).width for image in images])
    height_max = max([Image.fromarray(image).height for image in images])
    background_width = width_max*columns + (space*columns)-space
    background_height = height_max*rows + (space*rows)-space
    background = Image.new('RGB', (background_width, background_height), (0, 0, 0))
    x = 0
    y = 0
    for i, image in enumerate(images):
        img = Image.fromarray(image)
        x_offset = int((width_max-img.width)/2)
        y_offset = int((height_max-img.height)/2)
        background.paste(img, (x+x_offset, y+y_offset))
        x += width_max + space
        if (i+1) % columns == 0:
            y += height_max + space
            x = 0
    # background.save('/tmp/image_vis.png')
    return background

def group_design_elements_visual(image, image_metadata, threshold=0.2):
    name = "viridis"
    cmap = mpl.colormaps[name]  
    colors = cmap.colors 
    # print(len(colors))
    # return
    
    height, width = image.shape[:2]
    list_polygon = []
    group_tmp = {}
    for tmp in image_metadata["Elements"]:
        # print(tmp)
        tmp_coor = [np.array(coor)*np.array([width,height]) for coor in tmp["location-objective"]]
        if tmp["category"] == "Text":
            tmp_coor = [tmp_coor[0],tmp_coor[2]]
        list_polygon.append(tmp_coor)
    num_polygons = len(list_polygon)
    distances = np.zeros((num_polygons, num_polygons))
    # Calculate distance matrix
    for i in range(num_polygons):
        for j in range(i + 1, num_polygons):
            vert_dist, horiz_dist = distance_between_polygons(list_polygon[i], list_polygon[j])
            # print(list_polygon[i], list_polygon[j],vert_dist, horiz_dist)
            # dist_tmp = max(vert_dist / height, horiz_dist / width)
            distances[i, j] = max(vert_dist / height, horiz_dist / width)
            # distances[j, i] = distances[i, j]
            # if (dist_tmp) or ():
                
    # print(distances)
    # Apply hierarchical clustering
    
    
    # Form clusters with a threshold of 20% of the image dimensions
    list_threshold = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2]
    image_draw_tmp = []
    for threshold in list_threshold:
        group_tmp = {}
        group_num = 0
        # Calculate distance matrix
        for i in range(num_polygons):
            if not str(i) in group_tmp.keys():
                group_num +=1
                group_tmp[str(i)] = group_num
            for j in range(i + 1, num_polygons):
                if distances[i, j] <= threshold:
                    # print(i,j)
                    group_tmp[str(j)] = group_tmp[str(i)]
        
        # print("Num group: ", group_num)
        # print(group_tmp)
        # colors = cmr.get_sub_map('viridis', 0.2, 0.8, N=group_num)
        # distances_tmp = distances.copy()
        # # distances_tmp[distances_tmp > threshold] = 0
        # print(distances_tmp)/
        # Z = linkage(distances_tmp, 'single')
        # print(height*threshold)
        image_tmp = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        image_tmp_2 = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        # clusters = fcluster(Z.copy(), 1.0, criterion='distance')
        # print(Z)
        # clusters = list(map(str, clusters))
        # print(clusters)
        
        for idx in range(len(list_polygon)):
            group_index = group_tmp[str(idx)]-1
            # print(int(group_index*len(colors)/group_num))
            color_tmp = tuple([tmp*255 for tmp in colors[int(group_index*len(colors)/group_num)-1]])
            # print(color_tmp)
            start_p = list_polygon[idx][0].astype(int)
            end_p = list_polygon[idx][1].astype(int)# if len(list_polygon[idx]) == 4 else  list_polygon[idx][1].astype(int)
            cv2.rectangle(image_tmp, start_p, end_p, color_tmp, -1)
            reversed_color = tuple([255-tmp for tmp in color_tmp])
            # print(type((2, 24, 219)))
            cv2.putText(image_tmp, f"{group_tmp[str(idx)]}", [int((start_p[0]+end_p[0])/2), int((start_p[1]+end_p[1])/2)+10], cv2.FONT_HERSHEY_SIMPLEX,0.7,reversed_color,2,cv2.LINE_AA) 
        #     cv2.putText(image_tmp_2, f"{group_tmp[str(idx)]}", [int((start_p[0]+end_p[0])/2), int((start_p[1]+end_p[1])/2)+10], cv2.FONT_HERSHEY_SIMPLEX,0.7,reversed_color,2,cv2.LINE_AA) 
        # image_tmp = cv2.addWeighted(image_tmp_2,0.5,image_tmp, 0.5,0)
        cv2.putText(image_tmp, f"{round(threshold*100)}%", [10, 30], cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 
        # cv2.imwrite(f"/tmp/image_{list_threshold.index(threshold)}.png",image_tmp)
        image_draw_tmp.append(image_tmp)
        # break
            # print(clusters[idx], group_tmp.keys(), clusters[idx] in list(group_tmp.keys()))
    #         if not clusters[idx] in group_tmp.keys():
    #             group_tmp[clusters[idx]] = [image_metadata["Elements"][idx]]
    #         else:
    #             group_tmp[clusters[idx]].append(image_metadata["Elements"][idx])
    # image_metadata["Group"] = group_tmp
    visual_img = combine_images(3,5,image_draw_tmp)
    return visual_img

def convert_text_box(image_metadata, width, height):
    text_boxes = [
        TextBox(
            int(tmp["location-objective"][0][0]*width),
            int(tmp["location-objective"][0][1]*height),
            int((tmp["location-objective"][2][0] - tmp["location-objective"][0][0])*width),
            int((tmp["location-objective"][2][1] - tmp["location-objective"][0][1])*height),
            tmp["text-content"]
        )
        for tmp in image_metadata["Elements"] if tmp["category"] == "Text"
    ]
    # text_boxes += [
    #     TextBox(
    #         int(tmp["location"][0][0]),
    #         int(tmp["location"][0][1]),
    #         int(tmp["location"][1][0] - tmp["location"][0][0]),
    #         int(tmp["location"][1][1] - tmp["location"][0][1]),
    #         tmp["text"]
    #     ) for tmp in image_metadata["Elements"]]

    return text_boxes

def convert_object_box(boxes):
    text_boxes = [
        TextBox(
            int(tmp["box"][0]),
            int(tmp["box"][1]),
            int(tmp["box"][2]),
            int(tmp["box"][3]),
            tmp["label-text"]
        )
        for tmp in boxes
    ]
    return text_boxes

def detect_multiple_language(image, language_list, save_path_image=False):
    results = []
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for language in language_list:
        try:
            init_OCR_model(language)
        except Exception as e:
            print(e)
            continue
        result_tmp = ocr.ocr(gray_img, cls=True)
        # print(result_tmp)
        if result_tmp != [None]:
            results += result_tmp[0]
    if save_path_image != False:
    # Draw box 
        if results != []:
            # image = Image.open(filename).convert("RGB")
            image = Image.fromarray(image).convert("RGB")
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("/home/ubuntu/Downloads/arial.ttf", size=20)
            for res in results:
                for line in res:
                    box = [tuple(point) for point in line[0]]
                    # Finding the bounding box
                    box = [(min(point[0] for point in box), min(point[1] for point in box)),
                        (max(point[0] for point in box), max(point[1] for point in box))]
                    txt = line[1][0]
                    draw.rectangle(box, outline="red", width=2)  # Draw rectangle
                    draw.text((box[0][0], box[0][1] - 25), txt, fill="blue", font=font)  # Draw text above the box

            # Save result
            image.save(save_path_image)
    
    return [results]

def init_OCR_model(language):
    global ocr
    if torch.cuda.is_available():
        ocr = PaddleOCR(
            use_angle_cls=True,
            det=True,
            rec=False,
            bin=True,
            lang=language,
            show_log=False,
            use_gpu=True,
            draw_img_save_dir="/tmp"
        )  # need to run only once to download
    else:
        ocr = PaddleOCR(
            use_angle_cls=True,
            det=True,
            rec=False,
            bin=True,
            lang=language,
            show_log=False,
            use_gpu=False,
            draw_img_save_dir="/tmp"
        )
# ocr = PaddleOCR(
#             use_angle_cls=True,
#             det=True,
#             rec=False,
#             bin=True,
#             lang="en",
#             show_log=False,
#             use_gpu=False,
#             draw_img_save_dir="/tmp"
#         )
tesseract_model = tesserocr.PyTessBaseAPI(path=os.environ["TESSDATA_PREFIX"])
tesseract_model.SetPageSegMode(2)
