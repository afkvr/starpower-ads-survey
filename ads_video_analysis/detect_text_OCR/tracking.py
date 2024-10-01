
import json
from difflib import SequenceMatcher
import re
import numpy as np

def similar_difflib(a, b):
    return SequenceMatcher(None, a, b).ratio()

# def normaline_text(text):
#     return text.lower()
def normalize_text(l_string):
    # convert to lower case
    lower_string = l_string.lower()
    # remove all punctuation except words and space
    pattern = "[^a-zA-Z\d$/@.' ]"
    no_punc_string = re.sub(pattern,'', lower_string) 
    # remove white spaces
    no_wspace_string = no_punc_string.strip()
    # convert string to list of words
    lst_string = [no_wspace_string][0].split()
    # print(lst_string)
    
    # remove stopwords
    no_stpwords_string= "".join(lst_string)
    # for i in lst_string:
    #     if not i in stop_words:
    #         no_stpwords_string += [i]
    # # removing last space
    # no_stpwords_string = no_stpwords_string[:-1]
    return no_stpwords_string

def cal_interArea(a, b):  # returns None if rectangles don't intersect
    # print(a[0][0], b[0][0],a[1][0], b[1][0] )
    dx = min(a[2][0], b[2][0]) - max(a[0][0], b[0][0])
    dy = min(a[2][1], b[2][1]) - max(a[0][1], b[0][1]) 
    if (dx>=0) and (dy>=0):
        return dx*dy
    else:
        return 0

def checkInnerBox(boxMajor, box_check,threshold=0.8):
    # Compute the area of intersection rectangle
    interArea = cal_interArea(boxMajor, box_check)
    # Compute the area of both the prediction and ground-truth rectangles
    box_checkArea = (box_check[2][0] - box_check[0][0]) * (box_check[2][1] - box_check[0][1])
    
    iou = (interArea / float( box_checkArea - interArea)) if (float(box_checkArea - interArea) !=0) else (interArea /0.0001)
    # print(interArea,box_checkArea, iou)
    return iou >= threshold


def checkInnerText(textMajor,text_check,text_diff_threshold=0.8, char_index_thresh= 3):
    idx_tmp = [0]
    innerScore = []
    for char in text_check:
        innerScore.append(char in textMajor[idx_tmp[-1]:])
        if char in textMajor[idx_tmp[-1]:]:
            idx_tmp += [textMajor[idx_tmp[-1]:].index(char) +1 + idx_tmp[-1]]
    idx_tmp = idx_tmp[1:] 
    idx_dist = np.array(idx_tmp[1:]) - np.array(idx_tmp[:-1])
    innerScore = sum(innerScore)/len(innerScore)
    
    #Expect char distance index not bigger than 2 letters
    result_checking = (innerScore  >= text_diff_threshold) and not any(idx_dist > char_index_thresh)
    return result_checking 

def calculate_IoU_score(boxA, boxB):
    interArea = cal_interArea(boxA, boxB)
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2][0] - boxA[0][0]) * (boxA[2][1] - boxA[0][1])
    boxBArea = (boxB[2][0] - boxB[0][0]) * (boxB[2][1] - boxB[0][1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (interArea / float(boxAArea + boxBArea - interArea)) if (float(boxAArea + boxBArea - interArea) !=0) else (interArea /0.0001)

    # Return the intersection over union value
    return iou

def check_diff_txt(txt1, txt2, threshold = 0.8):
    return similar_difflib(txt1, txt2) >= threshold

def check_IoU_score(coor1, coor2, threshold = 0.7):
    return calculate_IoU_score(coor1, coor2) >= threshold

def tracking(result: list):                        #Result: [frame[box, (text, score)]]
    
    text_info_dict = {}

    last_key = -1
    with open('/home/kientran/Code/Work/OCR/pipeline/log/log.txt', 'w') as file:
        for frame_count in range(len(result)):
            str_frame_count = str(frame_count)
            if result.get(str_frame_count, []) != []:
                boxes = [line[0] for line in result[str_frame_count]]
                txts = [line[1][0] for line in result[str_frame_count]]
                normalized_txts = [normalize_text(txt) for txt in txts]

                # temp_obj = []

                for box_id in range(len(result[str_frame_count])):
                    box_coor = boxes[box_id]
                    box_normalized_txt = normalized_txts[box_id]
                    isInDict = False
                    for key in text_info_dict.keys():
                        diff = similar_difflib(text_info_dict[key]['text_normalized'], box_normalized_txt)
                        iou_score = calculate_IoU_score(text_info_dict[key]['box'], box_coor)
                        file.write(f"{text_info_dict[key]['text_origin']} vs {txts[box_id]}: {diff}  {iou_score} \n")
                        if check_diff_txt(text_info_dict[key]['text_normalized'], box_normalized_txt) and check_IoU_score(text_info_dict[key]['box'], box_coor):
                            isInDict = True
                            text_info_dict[key]['frame'] = text_info_dict[key]['frame'] + [frame_count]
                            break
                    if not isInDict:
                        temp_obj = {
                                    "text_origin": txts[box_id],
                                    "text_normalized": box_normalized_txt,
                                    "box": box_coor,
                                    'frame': [frame_count]
                                }
                        text_info_dict[last_key+1] = temp_obj
                        last_key += 1
            
    #Check text duration
    tmp = []
    for key in text_info_dict.keys():
        if len(text_info_dict[key]['frame']) < 5:
            tmp.append(key)
    for key in tmp:
        text_info_dict.pop(key)
    return text_info_dict

def find_diffscore_text(total_frame, text_object):
    data = text_object
    difflib_scores= []
    for idx in range(1,total_frame):
        previous_str = data[idx-1] if data[idx-1] != [] else ""
        current_str = data[idx] if data[idx] != [] else ""
        difflib_score = similar_difflib(previous_str, current_str)
        difflib_scores.append(difflib_score)
    return difflib_scores

def track_text_object(result):                        #Result: [frame[box, (text, score)]]
    
    text_info_dict = {}

    last_key = -1
    list_text_frames = []
    frame_keys = list(result.keys())
    for frame_key in frame_keys:
        if result.get(frame_key, []) != []:
            boxes = [line[0] for line in result[frame_key]]
            txts = [line[1][0] for line in result[frame_key]]
            normalized_txts = [normalize_text(txt) for txt in txts]
            list_text_frames.append("".join(normalized_txts))

            for box_id in range(len(result[frame_key])):
                box_coor = boxes[box_id]
                box_normalized_txt = normalized_txts[box_id]
                isInDict = False
                for key in text_info_dict.keys():
                    if check_diff_txt(text_info_dict[key]['text_normalized'], box_normalized_txt) and check_IoU_score(text_info_dict[key]['box'], box_coor):
                        isInDict = True
                        text_info_dict[key]['frame'] = text_info_dict[key]['frame'] + [frame_key]
                        break
                if not isInDict:
                    temp_obj = {
                                "text_origin": txts[box_id],
                                "text_normalized": box_normalized_txt,
                                "box": box_coor,
                                'frame': [frame_key]
                            }
                    text_info_dict[last_key+1] = temp_obj
                    last_key += 1
        else:
            list_text_frames.append("")
    diff_text_score = find_diffscore_text(len(result),list_text_frames)
    tmp = []
    for key in text_info_dict.keys():
        if len(text_info_dict[key]['frame']) < 5:
            tmp.append(key)
    for key in tmp:
        text_info_dict.pop(key)
    return text_info_dict,diff_text_score

def track_text_object_2(result):                        #Result: [frame[box, (text, score)]]
    
    text_info_dict = {}

    
    list_text_frames = []
    frame_keys = list(result.keys())
    for frame_key in frame_keys:
        if result.get(frame_key, []) != []:
            boxes = [line[0] for line in result[frame_key]]
            txts = [line[1][0] for line in result[frame_key]]
            normalized_txts = [normalize_text(txt) for txt in txts]
            list_text_frames.append("".join(normalized_txts))
            temp_obj = {}
            last_key = 0
            for box_id in range(len(result[frame_key])):
                box_coor = boxes[box_id]
                box_normalized_txt = normalized_txts[box_id]
                temp_obj[last_key] = {
                            "text_origin": txts[box_id],
                            "text_normalized": box_normalized_txt,
                            "box": box_coor,
                        }
                last_key += 1
            text_info_dict[frame_key] = temp_obj
        else:
            text_info_dict[frame_key] = {}
            list_text_frames.append("")
    diff_text_score = find_diffscore_text(len(result),list_text_frames)
  
    return text_info_dict,diff_text_score

thresh_numframe = 3
def track_merged_text_object(result):                        #Result: [frame[box, (text, score)]]
    text_info_dict = {}

    last_key = 0
    frame_keys = list(result.keys())
    for frame_key in frame_keys:
        merged_obj = result[frame_key]
        
        if merged_obj != {}:
            text_obj_keys = list(merged_obj.keys())
            for text_obj_key in text_obj_keys:
                text_obj_tmp = merged_obj[text_obj_key]
                boxes = text_obj_tmp["box"]
                txts = text_obj_tmp["text_origin"]
                normalized_txts = text_obj_tmp["text_normalized"]
                # temp_obj = []
                isInDict = False
                for key in text_info_dict.keys():
                    # print(check_diff_txt(text_info_dict[key]['text_normalized'], normalized_txts),check_IoU_score(text_info_dict[key]['box'][-1], boxes))
                    # print(text_info_dict[key]['text_normalized'], normalized_txts,checkInnerText(text_info_dict[key]['text_normalized'], normalized_txts),  checkInnerBox(text_info_dict[key]['box'][-1], boxes))
                    if check_diff_txt(text_info_dict[key]['text_normalized'], normalized_txts) and \
                        check_IoU_score(text_info_dict[key]['box'][-1], boxes) and \
                        (frame_key - text_info_dict[key]['frame'][-1] <= thresh_numframe ):  # Check box continues
                        isInDict = True
                        text_info_dict[key]['frame'] = text_info_dict[key]['frame'] + [frame_key]
                        text_info_dict[key]['box'] = text_info_dict[key]['box'] + [boxes]
                        break
                    elif checkInnerText(text_info_dict[key]['text_normalized'], normalized_txts) and \
                        checkInnerBox(text_info_dict[key]['box'][-1], boxes) and \
                        (frame_key - text_info_dict[key]['frame'][-1] <= thresh_numframe ):
                        isInDict = True
                        text_info_dict[key]['frame'] = text_info_dict[key]['frame'] + [frame_key]
                        text_info_dict[key]['box'] = text_info_dict[key]['box'] + [boxes]
                if not isInDict:
                    temp_obj = {
                                "text_origin": txts,
                                "text_normalized": normalized_txts,
                                "box": [boxes],
                                'frame': [frame_key]
                            }
                    text_info_dict[last_key] = temp_obj
                    last_key += 1
        # print(frame_key,merged_obj.keys() ,text_info_dict )
    #Check text duration
    tmp = []
    for key in text_info_dict.keys():
        text_info_dict[key]['frame'] = np.unique(np.array(text_info_dict[key]['frame'])).tolist()
        if len(text_info_dict[key]['frame']) < 5:
            tmp.append(key)
    for key in tmp:
        text_info_dict.pop(key)
    return text_info_dict

if __name__ == "__main__":
    with open("/home/kientran/Code/Work/OCR/pipeline/remove_low_score/404759268832213.json", 'r') as file:
        myDict = json.load(file)
    test = tracking(myDict)
    with open("/home/kientran/Code/Work/OCR/pipeline/tracking_results/404759268832213.json", 'w') as file:
        json.dump(test, file)


                


