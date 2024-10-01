import cv2
import numpy as np
import json
import subprocess
import ffmpeg
from PIL import Image
import os
import re
from tqdm import tqdm
from difflib import SequenceMatcher
# import matplotlib.pyplot as plt
import pandas as pd
from scenedetect import SceneManager,StatsManager,  open_video, ContentDetector, split_video_ffmpeg, AdaptiveDetector, SceneDetector
import math
import tesserocr
# import tesseract
import base64
import requests
import time
import argparse
import torch
from detect_scene_change.section_detect import get_content_val, detect_scene_final, ffmpeg_encoder
from detect_text_OCR.main import detectTextOCR
from paddleocr import PaddleOCR

PATH = os.path.dirname(os.path.abspath(__file__))

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ["TESSDATA_PREFIX"]=f"{PATH}/detect_text_OCR/tessdata/"

# Define the sector_map based on the image
sector_map = [
    ["upper left corner", "upper center edge", "upper right corner"],
    ["up left edge", "up center", "up right edge"],
    ["mid-up left edge", "mid-up center", "mid-up right edge"],
    ["mid-low left edge", "mid-low center", "mid-low right edge"],
    ["low left edge", "low center", "low right edge"],
    ["lower left corner", "lower center edge", "lower right corner"]
]

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

def uniqueList(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    return unique_list

def similar_difflib(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_closest_font_name(folder_path, title,extension =None):
    # Normalize the title to make the comparison case-insensitive and to replace common synonyms
    normalized_title = title.lower().replace("semibold", "semi bold").replace("extrabold", "extra bold")
    
    # Tokenize the normalized title to catch key terms like 'Bold', 'Italic', etc.
    title_tokens = set(normalized_title.split())

    # List all files in the given directory
    font_files = os.listdir(folder_path)
    if extension is not None:
        font_files = [tmp for tmp in font_files if tmp.endswith(tuple([".otf",".ttf"]))]
    best_match = None
    highest_score = 0

    # Search for the best match based on the presence of key tokens and overall similarity
    for font in font_files:
        # if font.endswith(".otf"):
        normalized_font_name = font.lower().replace("-", " ").replace("_", " ").replace(".otf", "")
        font_tokens = set(normalized_font_name.split())
        
        # Check for direct token matches (e.g., both have "bold" or "italic")
        style_match_score = len(title_tokens.intersection(font_tokens))
        
        # Use SequenceMatcher to find the closest match in terms of text similarity
        similarity_score = SequenceMatcher(None, normalized_title, normalized_font_name).ratio()
        
        # Calculate a weighted score to prioritize style matches
        combined_score = similarity_score# + style_match_score * 0.5
        # print(similarity_score,font_tokens, title_tokens)
        if combined_score > highest_score:
            highest_score = combined_score
            best_match = font

    return best_match

def calculate_iou(boxA, boxB):
    # print(boxA, boxB)
    # if boxA == boxB:
    #     return 1
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[2][0], boxB[2][0])
    yB = min(boxA[2][1], boxB[2][1])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2][0] - boxA[0][0]) * (boxA[2][1] - boxA[0][1])
    boxBArea = (boxB[2][0] - boxB[0][0]) * (boxB[2][1] - boxB[0][1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (interArea / float(boxAArea + boxBArea - interArea)) if (float(boxAArea + boxBArea - interArea) !=0) else (interArea /0.0001)

    # Return the intersection over union value
    return iou
    

def find_font_path(title,path_saved_fonts_folder):
    font_folder_path = find_closest_font_name(path_saved_fonts_folder,title)
    # print("Font folder: ", font_folder_path)
    font_path = find_closest_font_name(os.path.join(path_saved_fonts_folder, font_folder_path),title,".otf")
    # print("Font file matching: ", font_path)
    return os.path.join(path_saved_fonts_folder,font_folder_path,font_path)

def get_font_text(image):
    _, buffer_bf = cv2.imencode(".jpg", image)
    encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
    # URL of the API endpoint
    url = 'https://www.whatfontis.com/api2/'

    # Data for the form
    form_data = {
        'urlimagebase64': encoded_image_bf, 
        'API_KEY': 'b747403f6ecb8fd7e703bb9cef496fb551bc74dc7b2e0190b1d8d04549ce4970',
        'IMAGEBASE64': '1', 
        'NOTTEXTBOXSDETECTION': '1',
        'urlimage': '',  
        'limit': '1'
    }
    try:
        # Make the POST request with the form data
        response = requests.post(url, data=form_data)
        # print(response)
        text = response.text # Print the response text
    
        # print(type(text))
        # print(text)
        if text == "No chars found":
            return "Not detected"
        else:
            text = json.loads(text)
            return text[0]["url"]     
    except Exception as e:
        print("Get error response when get font text: ", e)
        print(text)
        return "Not detected"

def check_textblocks_in_section(text_object, start_frame, end_frame,frame_area):
    textContentList = []
    active_block = False
    textblocks_frame = []
    textblocks_area = []
    section_range = set(list(range(start_frame,end_frame+1)))
    for text_obj in list(text_object.keys()):
        textblock_frame = set(text_object[text_obj]["frame"])
        frames_text_in_section = list(section_range.intersection(textblock_frame))
        textblocks_frame += frames_text_in_section
        if len(frames_text_in_section) >0:
            textContentList.append(text_object[text_obj]["text_origin"])
        if any(np.array(text_object[text_obj]["frame"]) < start_frame):
            active_block = True
        textblock_area = [calculate_area(box) for box in text_object[text_obj]["box"]]
        textblocks_area += textblock_area
    # print(len(np.unique(textblocks_frame)), (end_frame-start_frame+1), start_frame, end_frame)
    ratio_text_duration = round(len(np.unique(textblocks_frame))/(end_frame-start_frame+1),5)
    ratio_text_area = round(np.sum(np.array(textblocks_area))/(frame_area*(end_frame-start_frame+1)),5)
    
    num_text_blocks = len(uniqueList(textContentList))
    
    return num_text_blocks,active_block,ratio_text_duration,ratio_text_area

def detect_font_infos(list_frames, text_obj,tesseract_model,width,height,largest_box):
    padding_size_ratio =  0.2
    frame_index = text_obj["frame"][int(len(text_obj["frame"])/2)]
    topleft = np.array(largest_box[0]).astype(int)
    bottomright = np.array(largest_box[2]).astype(int)
    padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
    padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
    
    padding_topleft = [max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y)]
    padding_bottomright = [min(width, bottomright[0] + padding_X), min(height, bottomright[1] + padding_Y)]
    # frame_index = int( (text_obj_tmp["start_frame"] + text_obj_tmp["end_frame"])/2)
    # print(frame_index, padding_X, padding_Y)
    # print(topleft, bottomright)
    # print(padding_topleft[1],padding_bottomright[1],padding_topleft[0],padding_bottomright[0])
    crop_img = list_frames[frame_index][padding_topleft[1]:padding_bottomright[1],padding_topleft[0]:padding_bottomright[0]]
    
    #Detect font weight, size
    tesseract_model.SetImage(Image.fromarray(crop_img))
    if tesseract_model.Recognize():
        iterator = tesseract_model.GetIterator()
        iterator = iterator.WordFontAttributes()
        # print("Pointsize: ",iterator["pointsize"])
        font_size = int(iterator["pointsize"]) if iterator["pointsize"] > 5 else 20
        font_bold = "bold" if iterator["bold"] else "unbold"
    else:
        font_size = 20
        font_bold = "unbold"
    # cv2.imwrite("/tmp/test_crop.png",crop_img)
    font_style = "Not detected"#get_font_text(crop_img)
    
    #Detect font color
    crop_img = Image.fromarray(crop_img)
    colors = sorted(crop_img.getcolors(crop_img.size[0]*crop_img.size[1]))
    colors_hex = hex = ('#%02x%02x%02x' % colors[-2][1])
    
    
    # detect capitalize
    text_raw = text_obj["text_origin"]
    if text_raw.isupper():
        font_cap = 'ALL CAPS'
    elif text_raw.istitle():
        font_cap = 'Start Case'
    elif text_raw.islower():
        font_cap = 'no caps'
    else:
        font_cap = 'standard case'
    
    font_infos = {
        "font_size":font_size,
        "font_bold":font_bold,
        "font_style":font_style,
        "font_cap":font_cap,
        "font_color": colors_hex
    }
    # print(font_infos)
    return font_infos

textchange_frame_min = 1

def calculate_area(box):
    # Extract coordinates
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]
    # Return area
    return (x3-x1) * (y3-y1)

def get_largest_box(boxes):
    # Calculate areas of all boxes
    areas = [(calculate_area(box), box) for box in boxes]
    # Find the box with the maximum area
    largest_box = max(areas, key=lambda x: x[0])[1]
    
    return largest_box

def main():
    tesseract_model = tesserocr.PyTessBaseAPI(path=os.environ["TESSDATA_PREFIX"])
    tesseract_model.SetPageSegMode(2)

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default="")
    args = parser.parse_args()
    video_input = args.video
    if os.path.isfile(video_input):
        list_video = [video_input]
    else: 
        list_video = [os.path.join(video_input,video) for video in  os.listdir(video_input)]
        
    num_threads = 1
    
    for filename in list_video:
        count = 1
        text_object = {}
        # filename = "781490520120333.mp4"
        path_folder = f"{os.path.dirname(filename)}"
        if filename.lower().endswith(('.mp4')):# and not os.path.isfile(f"""{path_folder}/{os.path.basename(filename).split(".")[0]}_font_recognition_results.json"""):
            filename = filename.split(".")[0]
            video_name = os.path.basename(filename).split(".")[0]

            print(f"{path_folder}/{video_name}.mp4")
            video_path = f"{path_folder}/{video_name}.mp4"#"/home/anlab/Downloads/advertisments_video/test_April_03/shop/728635192798056.mp4"
            cap = cv2.VideoCapture(video_path)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video_duration = round(total_frame/fps,3)
            if video_duration >= 120:
                cap.release()
                continue
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
            frame_temp = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_temp.append(frame)
                
            cap.release()
            ### Detect scene/section by pyscenedetect and text on screen
            pyscene_score_diff = get_content_val(video_path,f'{path_folder}/{video_name}_stats.csv')
            # Detect text using PaddleOCR and normalized results
            text_object,diff_text_score = detectTextOCR(frame_temp ,text_object, total_frame,ocr)
            path_final_output = f"{path_folder}/{video_name}_text_object.json"
            with open(path_final_output ,'w') as fp:
                json.dump(text_object, fp, indent=4)
            # print(len(pyscene_score_diff), len(diff_text_score))
            scene_data = detect_scene_final(total_frame,pyscene_score_diff,diff_text_score,fps)
            # print(scene_data)
            scene_data.to_csv(f"{path_folder}/{video_name}section_detect.csv",sep=';', float_format='%.5f')
            ## Detect font for each text object
            
            frame_area = width*height
            
            ## Define textblocks-data
            print("Get Text blocks data")
            textblocks_data = []
            # for text_obj in list(text_object.keys()):
            list_keys = list(text_object.keys())
            for idx in tqdm(range(len(list_keys)), desc="Processing textblocks-data"):
                text_obj = list_keys[idx]
                start_time = round(min(text_object[text_obj]["frame"])/fps,3)
                end_time = round(max(text_object[text_obj]["frame"])/fps,3)
                text_origin = text_object[text_obj]["text_origin"].replace("\n","").split(" ")
                list_word_tmp = [ text for text in text_origin if text != ""]
                wpm = round(len(list_word_tmp)/(end_time-start_time)/60,3)
                largest_box = get_largest_box(text_object[text_obj]["box"])
                box_w = largest_box[1][0] - largest_box[0][0]
                box_h = largest_box[2][1] - largest_box[0][1]
                sectors_list = get_sector_frame(width, height,sector_map,
                                                [largest_box[0],largest_box[2]]
                                                )
                font_info = detect_font_infos(frame_temp,text_object[text_obj],tesseract_model,width,height, largest_box)
                block_coordinates = [[round(box_[0]/width,5), round(box_[1]/height,5)] for box_ in largest_box]
                # break
                data_tmp ={
                    "block-name":f"""block-{text_obj}""",
                    "timing-info":{
                        "start-time":start_time,                                  # 3-a-i Time the text is first shown (second)
                        "end-time":end_time,                                    # 3-a-ii Time the text is last shown (second)
                        "duration":end_time-start_time,                                    # 3-a-iii Duration time from entry to exit (second)
                        "percentage-duration-text-on-screen":round((end_time-start_time+round(1/fps,3))/video_duration,5)                          # 3-a-iv Duration of text as % of video length
                    },
                    "text-info":{
                        "text-content":text_object[text_obj]["text_origin"],                            # 3-b-i Text capitalized and punctuated as shown
                        "wpm":wpm                                          # 3-a-ii WPM (number of words shown divided by duration time from entry to exit)
                    },
                    "bsize-objective":[box_w,box_h],                                # 3-c: Size (objective), rectangle -  Format [weight, height] of box text
                    "size-percentage-of-frame":round(box_w*box_h/frame_area,5),                              # 3-d: Size (% of video frame)- Fomula - box_text_area / frame_area
                    "location-objective":block_coordinates,     # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                    "location-sector":sectors_list,                            # 3-f: Location (section of video frame) (map to be included)
                    "font-details":{
                        "font-style":font_info["font_style"],  # 3-g-i: Font style (closest match to font) (Link font returned by whatfontis)
                        "font-size":font_info["font_size"],                                    # 3-g-ii: Font size (in pt)
                        "serif":True,                                      # 3-g-iii: True is Serif | False is Sans Serif
                        "font-weight":font_info["font_bold"],                              # 3-g-iv: Font weight (bolded, unbolded)
                        "cap-style":font_info["font_cap"],                       # 3-g-v: Capitalization style - output is 1 in 4 types: "standard case"/"Start Case"/"ALL CAPS"/"no caps"
                        "font-color":font_info["font_color"],                            # 3-g-vi: Font color (save as format HEX)
                        "font-outline":{                                   # 3-g-vii: Font outline (yes or no). If no => result as None/null. 
                            "color":"#001400",                             # If yes => output would has field "color"
                        },
                        "font-outline-weight":3                            # 3-g-viii: font outline weight. If "font-outline" is None, font-outline-weight equal 0.
                    },
                    "text-natural":True    
                }
            
                textblocks_data.append(data_tmp)
            # print(textblocks_data)
            
            # break
            ## Define sections-data
            sections_data =[]
            # for idx in range(len(scene_data["scene-name"])):
            for idx in tqdm(range(len(scene_data["scene-name"])), desc="Processing Section-data"):
                start_time = scene_data["start_time"][idx]
                end_time = scene_data["end_time"][idx]
                if start_time == end_time:
                    continue
                # print(idx ,f"""{int(scene_data["start_frame"][idx])} - {int(scene_data["end_frame"][idx])}""")
                num_text_blocks, active_block,ratio_text_duration,ratio_text_area = check_textblocks_in_section(text_object,
                                                                                                                int(scene_data["start_frame"][idx]),
                                                                                                                int(scene_data["end_frame"][idx]),
                                                                                                                frame_area)
                data_tmp = {
                    "section-name":scene_data["scene-name"][idx],
                    "start-time":start_time,
                    "end-time":end_time,
                    "duration":round(end_time-start_time,3),
                    "amount-of-text-blocks":num_text_blocks,
                    "block-still-active":active_block,
                    "percentage-duration-text-on-screen":ratio_text_duration,
                    "percentage-frame-text-cover-cross-duration":ratio_text_area
                }
                sections_data.append(data_tmp)
            ## Define video-infos
            num_text_blocks, _,ratio_text_duration_all,ratio_text_area_all = check_textblocks_in_section(text_object,1,total_frame, frame_area)
            video_infos = {                                                # 1: For each video
                "video-name":video_name,
                "video-resolution":[width,height],                                    # 1-a: Video resolution in format [weight, height]
                "video-duration":video_duration,                                            # 1-b: Video Duration (seconds)
                "amount-of-text-blocks":num_text_blocks,                                       # 1-c Amount of Text Blocks in Video
                "percentage-duration-text-on-screen":ratio_text_duration_all,                                 # 1-d % of video with any text on screen (by duration)
                "percentage-frame-text-cover-cross-duration":ratio_text_area_all                                      # 1-e: % of video covered by text (% of video frame across duration)| fomula = (all_box_text_area x text_duration) / (frame_area x video_duration)
            }
            
            output_font_reg = {
                "data":[
                    {
                        "video-info":video_infos,
                        "sections-data":sections_data,
                        "textblocks-data":textblocks_data
                    }
                ]
            }
            
            ##Export data to output json      
            path_final_output = f"{path_folder}/{video_name}_font_recognition_results.json"
            with open(path_final_output ,'w') as fp:
                json.dump(output_font_reg, fp, indent=4)
            #     font_detect = json.load(fp)
            # font_final = []
        break

if torch.cuda.is_available():
    ocr = PaddleOCR(use_angle_cls=True, lang="en",show_log=False,  use_gpu = True)  # need to run only once to download
else:
    ocr = PaddleOCR(use_angle_cls=True, lang="en",show_log=False,  use_gpu = False)
if __name__ == "__main__":
    main()
    # crop = cv2.imread("/tmp/test_crop.png")
    # get_font_text(crop)