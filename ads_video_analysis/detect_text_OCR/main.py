from .tracking import track_text_object_2, tracking, track_merged_text_object
from .merge_box import *
from .visual import *
import json
import os
import cv2
from tqdm import tqdm
# from paddleocr import PaddleOCR, draw_ocr
import torch

def remove_low_score(result, threshold):
    ind2rem = []                        #Store index of box to remove
    ans = []                            #Store chosen boxes
    
    if len(result) != 0:
        result = result[0]
        for i in range(len(result)):
            if result[i][1][1] < threshold:
                ind2rem.append(i)
        for i in range(len(result)):
            if i not in ind2rem:
                ans.append(result[i])
    return ans

def remove_low_score_from_raw(result, threshold=0.9):
    new_dict = {}
    for key in result.keys():
        frame_res = result[key]
        frame_res = remove_low_score(frame_res,threshold)

        new_dict[key] = frame_res
    return new_dict

def pipeline(video_dir):
    file_name = os.listdir(video_dir)
    file_name = [i.split('.')[0] for i in file_name]
    for vid_name in file_name:
        #Get numframe
        video = cv2.VideoCapture(video_dir + f'/{vid_name}.mp4')
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()

        with open(f"/home/kientran/Code/Work/OCR/pipeline/raw_results/{vid_name}.json", 'r') as file:
            result = json.load(file)
        # ##Remove low score
        remove_low_score_from_raw(result,vid_name)
        ##Tracking
        with open(f"/home/kientran/Code/Work/OCR/pipeline/remove_low_score/{vid_name}.json", 'r') as file:
            myDict = json.load(file)
        test = tracking(myDict)
        with open(f"/home/kientran/Code/Work/OCR/pipeline/tracking_results/{vid_name}.json", 'w') as file:
            json.dump(test, file)
        ##Merge boxes
        do_merge(vid_name, num_frame = num_frames)
        #Visual
        visualize(vid_name)
        print(vid_name)
        break

def detectTextOCR(list_frames ,text_object, total_frame,ocr):
    ## Load Paddler OCR
    for idx_frame in tqdm(range(1,total_frame+1), desc="Processing text"):
        frame = list_frames[idx_frame-1]
        if  frame is None:
            break
        #Detect raw text by Paddle OCR
        grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        result = ocr.ocr(grayFrame, cls=True)
        if result != [None]:
            text_object[idx_frame] = result
        else:
            text_object[idx_frame] = []

    with open(f"/tmp/text_objects.json", 'w') as file:
        json.dump(text_object, file)
    # Remove results have score lower than 0.9
    text_object_filter = remove_low_score_from_raw(text_object,0.8)
    with open(f"/tmp/text_object_filter.json", 'w') as file:
        json.dump(text_object_filter, file, indent=4)
    # Tracking each box through frames
    tracked_text_object,diff_text_score = track_text_object_2(text_object_filter)
    with open(f"/tmp/tracked_text_object.json", 'w') as file:
        json.dump(tracked_text_object, file, indent=4)
    # Merge small to final box
    merged_text_object = do_merge_2(tracked_text_object)
    with open(f"/tmp/merged_text_object.json", 'w') as file:
        json.dump(merged_text_object, file, indent=4)
    merged_text_object = track_merged_text_object(merged_text_object)
    with open(f"/tmp/merged_text_object_tracked.json", 'w') as file:
        json.dump(merged_text_object, file, indent=4)
        
    return merged_text_object,diff_text_score

if __name__ == "__main__":

    with open(f"/tmp/text_objects.json", 'r') as file:
        # json.dump(text_object, file)
        text_object = json.load(file)
    # pipeline("/home/kientran/Code/Work/OCR/Video")
    text_object_filter = remove_low_score_from_raw(text_object,0.9)
    with open(f"/tmp/text_object_filter.json", 'w') as file:
        json.dump(text_object_filter, file, indent=4)
    # Tracking each box through frames
    tracked_text_object,diff_text_score = track_text_object_2(text_object_filter)
    print(len(diff_text_score))
    with open(f"/tmp/tracked_text_object.json", 'w') as file:
        json.dump(tracked_text_object, file, indent=4)
    # Merge small to final box
    merged_text_object = do_merge_2(tracked_text_object, 551)
    with open(f"/tmp/merged_text_object.json", 'w') as file:
        json.dump(merged_text_object, file, indent=4)
    merged_text_object = track_merged_text_object(merged_text_object)
    with open(f"/tmp/merged_text_object_tracked.json", 'w') as file:
        json.dump(merged_text_object, file, indent=4)