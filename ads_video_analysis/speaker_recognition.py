import base64
import requests
import cv2
import time
import os
import sys
import numpy as np
import ffmpeg
import subprocess
from PIL import Image
import shutil
import time
import pandas as pd
from tqdm import tqdm
import json
import argparse
import mediapipe as mp
import torch
import matplotlib as mpl
from RetinaFace_tf2.src.retinafacetf2.retinaface import RetinaFace
from RetinaFace_tf2.detect_gender_age import detect_gender_age, most_frequent,detect_gender_age_DeepFace
import math
import tensorflow as tf
from detect_scene_change.section_detect import get_content_val, detect_scene_final, ffmpeg_encoder
from detect_text_OCR.main import detectTextOCR
from deepface import DeepFace
from scripts.speaking_detection import lip_movement_detection, landmark_points_61_67

def sig(x):
 return 1/(1 + np.exp(-x))

gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)
PATH = os.path.dirname(os.path.abspath(__file__))

FACEMESH_pose_estimation = [34,264,168,33, 263]

# Define the sector_map based on the image
sector_map = [
    ["upper left corner",   "upper center edge", "upper right corner"   ],
    ["up left edge",        "up center",         "up right edge"        ],
    ["mid-up left edge",    "mid-up center",     "mid-up right edge"    ],
    ["mid-low left edge",   "mid-low center",    "mid-low right edge"   ],
    ["low left edge",       "low center",        "low right edge"       ],
    ["lower left corner",   "lower center edge", "lower right corner"   ]
]

def get_sector_frame(width_frame,height_frame, sector_map, rectangle):
    """_summary_

    Args:
        width_frame (int):
        height_frame (int):
        sector_map (list): Map defines sectors in a frame
        rectangle (list): Coordinates of box with format [[xA, yA],[xB, yB]] | A is top-left, B is bottom-right corner

    Returns:
        list: list sectors map box land in.
    """
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

class KalmanTracking(object):
    # init kalman filter object
    def __init__(self, point):
        deltatime = 1/30 # 30fps
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0]], np.float32)

        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1]], np.float32)

        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], np.float32) * deltatime # 0.03
        # self.kalman.measurementNoiseCov = np.array([[1, 0],
        #                                             [0, 1]], np.float32) *0.1
        self.measurement = np.array((point[0], point[1]), np.float32)

    def getpoint(self, kp):
        self.kalman.correct(kp-self.measurement)
        # get new kalman filter prediction
        prediction = self.kalman.predict()
        prediction[0][0] = prediction[0][0] +  self.measurement[0]
        prediction[1][0] = prediction[1][0] +  self.measurement[1]

        return prediction

def binaryMaskIOU_(mask1, mask2):
    mask1_area = torch.count_nonzero(mask1)
    mask2_area = torch.count_nonzero(mask2)
    # print("mask1_area ", mask1_area)
    # print("mask2_area ", mask2_area)
    intersection = torch.count_nonzero(torch.logical_and(mask1, mask2))
    iou = intersection / (mask1_area + mask2_area - intersection)
    return iou.numpy()

def calculate_iou(boxA, boxB):
    # print(boxA, boxB)
    # if boxA == boxB:
    #     return 1
    # Determine the (x, y)-coordinates of the intersection rectangle
    boxA = [[boxA[0],boxA[1]],[boxA[2],boxA[3]]]
    boxB = [[boxB[0],boxB[1]],[boxB[2],boxB[3]]]
    xA = max(boxA[0][0], boxB[0][0])
    yA = max(boxA[0][1], boxB[0][1])
    xB = min(boxA[1][0], boxB[1][0])
    yB = min(boxA[1][1], boxB[1][1])
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[1][0] - boxA[0][0]) * (boxA[1][1] - boxA[0][1])
    boxBArea = (boxB[1][0] - boxB[0][0]) * (boxB[1][1] - boxB[0][1])
    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = (interArea / float(boxAArea + boxBArea - interArea)) if (float(boxAArea + boxBArea - interArea) !=0) else (interArea /0.0001)
    # Return the intersection over union value
    return iou

class KalmanArray(object):
    def __init__(self):
        self.kflist = []
        # self.oldmask = None
        # self.resetVal = 1
        self.w = 0
        self.h = 0

    def noneArray(self):
        return len(self.kflist) == 0

    def setpoints(self, points, w=1920, h=1080):
        for value in points:
            intpoint = np.array([np.float32(value[0]), np.float32(value[1])], np.float32)
            self.kflist.append(KalmanTracking(intpoint))

        self.w = w
        self.h = h
        # self.oldmask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)


        # print('setpoints ', self.w)


    def getpoints(self, kps):
        # print('old ', kps[:3])
        # print("KPS:",len(kps),'\n', kps)
        # print("Kflist",len(self.kflist))
        orginmask = np.zeros((self.h,self.w),dtype=np.float32)
        orginmask = cv2.fillConvexPoly(orginmask, np.array(kps[:-18], np.int32), 1) #kps[:-18]
        kps_o = kps.copy()
        for i in range(len(kps)):
            # print(i)
            # kps[i] = kflist[i]
            intpoint = np.array([np.float32(kps[i][0]), np.float32(kps[i][1])], np.float32)
            tmp = self.kflist[i].getpoint(intpoint)
            kps[i] = (tmp[0][0], tmp[1][0])

        newmask = np.zeros((self.h,self.w),dtype=np.float32)
        newmask = cv2.fillConvexPoly(newmask, np.array(kps[:-18], np.int32), 1)
        # cv2.imwrite('orginmask.jpg' , orginmask*255)
        val = binaryMaskIOU_(torch.from_numpy(orginmask), torch.from_numpy(newmask))
        # print('binaryMaskIOU_ ', val)

        # distance = spatial.distance.cosine(orgindata, newdata)
        # print(distance)
        if val < 0.9:
            del self.kflist[:]
            # self.oldmask = None
            self.setpoints(kps_o,self.w, self.h)
            return kps_o

        # self.olddata = newdata
        # print('new ', kps[:3])
        return kps

def cross_point(line1, line2):
    x1 = line1[0]
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    b1 = y1 * 1.0 - x1 * k1 * 1.0
    if (x4 - x3) == 0:
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)
        b2 = y3 * 1.0 - x3 * k2 * 1.0
    if k2 == None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]

def point_line(point,line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    x3 = point[0]
    y3 = point[1]

    k1 = (y2 - y1)*1.0 /(x2 -x1)
    b1 = y1 *1.0 - x1 *k1 *1.0
    k2 = -1.0/k1
    b2 = y3 *1.0 -x3 * k2 *1.0
    x = (b2 - b1) * 1.0 /(k1 - k2)
    y = k1 * x *1.0 +b1 *1.0
    return [x,y]

def point_point(point_1,point_2):
    x1 = point_1[0]
    y1 = point_1[1]
    x2 = point_2[0]
    y2 = point_2[1]
    # distance = ((x1-x2)**2 +(y1-y2)**2)**0.5
    distance = math.sqrt((x1-x2) ** 2 + (y1-y2) ** 2)
    # distance = math.hypot(x2-x2, y1-y2)
    # if distance == 0:
    #     distance = distance + 0.1
    return distance

def facePose(point1, point31, point51, point60, point72):
    crossover51 = point_line(point51, [point1[0], point1[1], point31[0], point31[1]])
    yaw_mean = point_point(point1, point31) / 2
    yaw_right = point_point(point1, crossover51)
    yaw = (yaw_mean - yaw_right) / yaw_mean
    if math.isnan(yaw):
        return None, None, None
    yaw = int(yaw * 71.58 + 0.7037)

    #pitch
    pitch_dis = point_point(point51, crossover51)
    if point51[1] < crossover51[1]:
        pitch_dis = -pitch_dis
    if math.isnan(pitch_dis):
        return None, None, None
    pitch = int(1.497 * pitch_dis + 18.97)

    #roll
    roll_tan = abs(point60[1] - point72[1]) / abs(point60[0] - point72[0])
    roll = math.atan(roll_tan)
    roll = math.degrees(roll)
    if math.isnan(roll):
        return None, None, None
    if point60[1] >point72[1]:
        roll = -roll
    roll = int(roll)

    return yaw, pitch, roll

def filter_straight_face(list_img, face_data, facemesh, fps):
    """_summary_
    Filter each speaking segment by yaw, pitch and roll from 5 points of Retinaface output
    Args:
        list_img (_type_): _description_
        face_data (_type_): _description_
        facemesh (_type_): _description_
        fps (_type_): _description_

    Returns:
        _type_: _description_
    """
    h,w = list_img[0].shape[:2]
    min_time = 0.5 * fps # Set minimum time second to detect a face is legal
    yaw_threshold = 40  # Mean legal yaw in [-50 , 50]
    pitch_threshold = 45    
    straight_threshold = 0.1   # Mean how many percent face look straight in time is consider as straight face
    face_keys = list(face_data.keys())
    # number_face_list = [False]*len(face_keys)
    face_data_new = {}
    face_data_nonfilter = {}
    for face_key in face_keys:
        face_data_tmp = face_data[face_key]
        frame_keys = list(face_data_tmp.keys())
        if len(frame_keys) < min_time:
            continue
        list_face_angles = [False]*len(frame_keys)
        face_new_tmp = {}
        for frame_key in frame_keys:
            box_face_tmp = [int(coor) for coor in face_data_tmp[frame_key]["box"][:-1]] + [face_data_tmp[frame_key]["box"][-1]]  # Format [xA, yA, xB, yB] with A is top-left corner, B is bottom-right
            xA, yA, xB, yB = [max(box_face_tmp[0],0),max(box_face_tmp[1],0),min(box_face_tmp[2],w),min(box_face_tmp[3],h)]
            crop_img = list_img[int(frame_key)][yA:yB, xA:xB]
            results = facemesh.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                face_new_tmp[frame_key] ={
                                            "box": box_face_tmp,
                                            "yaw_pitch_roll":[None,None,None],
                                            "points": None,
                                            "lippoints":None
                                        }
                continue
            face_landmarks = results.multi_face_landmarks[0]
            bbox_w = xB-xA
            bbox_h = yB-yA
            
            # Extract Face pose keypoints for transforming face to straight position
            posePoint = []
            for i in range(len(FACEMESH_pose_estimation)):
                idx = FACEMESH_pose_estimation[i]
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                realx = x * bbox_w 
                realy = y * bbox_h
                posePoint.append((realx, realy))
            
            # Extract Inner lip keypoints 61,62,63,65,66,67 for detecting SPEAKING
            lippoints = []
            for i in range(len(landmark_points_61_67)):
                idx = landmark_points_61_67[i]
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y
                realx = x * 256     #Resize x-coordinate to normalized facebox shape [256,320]
                realy = y * 320     #Resize y-coordinate to normalized facebox shape [256,320]
                lippoints.append((realx, realy))
             #Cal yaw, pitch and roll distance for face
            yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
            
            if abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold:
                list_face_angles[frame_keys.index(frame_key)] = True
            face_new_tmp[frame_key] ={
                                            "box": box_face_tmp,
                                            "yaw_pitch_roll":[yaw,pitch,roll],
                                            "points": face_data_tmp[frame_key]["points"],
                                            "lippoints":lippoints
                                        }
        face_data_nonfilter[face_key]=face_new_tmp
        if float(np.sum(list_face_angles)/len(frame_keys)) > straight_threshold:
            face_data_new[face_key]=face_new_tmp
            # number_face_list[face_keys.index(face_key)] = True
                    
    return face_data_new,face_data_nonfilter

def get_face_data(frame_temp, detector,fps):
    duration_threshold = fps*0.3
    face_data = {}
    for idx_frame in tqdm(range(len(frame_temp)), desc="Detecting face"):
        frame = frame_temp[idx_frame]
        faces, keypoints = detector.detect(frame, 0.5)
        if len(faces) == 0:# not results.multi_face_landmarks:
            # count += 1 
            continue
        # Matching face box with face_data
        face_idx = 0
        for face in faces:
            list_face_data_key = list(face_data.keys()) 
            # print(face, info)
            IoU_highest = 0
            match_face_key = None
            match_frame_key = None
            for face_key_tmp in list_face_data_key:
                face_data_tmp = face_data[face_key_tmp]
                # print(face_data_tmp[list(face_data_tmp.keys())[-1]])
                face_box_latest = face_data_tmp[list(face_data_tmp.keys())[-1]]["box"]
                iou = calculate_iou(face[:-1], face_box_latest)
                # match_frame_key = int(list(face_data_tmp.keys())[-1])
                if iou >IoU_highest:
                    IoU_highest = iou
                    match_face_key = face_key_tmp
                    match_frame_key = int(list(face_data_tmp.keys())[-1])
            if IoU_highest >= 0.7 and abs(match_frame_key-idx_frame) <= 3:  #If box intersection is  more than 0.7 and frame_idc different is less than 3
                # print("match")
                face_data[match_face_key][str(idx_frame)] = {"box":face.tolist(), "points":keypoints[face_idx].tolist()}
            else:
                # print("Not match")
                new_key = "face_"+str(len(list_face_data_key)+1)
                face_data[new_key] = {str(idx_frame):{"box":face.tolist(), "points":keypoints[face_idx].tolist()}}
            face_idx +=1
    #Remove face has duration less duration_threshold
    face_data_final = {}
    for face_key in list(face_data.keys()):
        if not len(face_data[face_key]) < duration_threshold:
            face_data_final[face_key] = face_data[face_key]
    
    return face_data_final

def calculate_distance( list_ypr, target):
    if target[-1] is None:
        return sig(float(abs(list_ypr[0] - target[0])*math.pi/180)) + sig(float(abs(list_ypr[1] - target[1])*math.pi/180))
    else: 
        return sig(float(abs(list_ypr[0] - target[0])*math.pi/180)) + sig(float(abs(list_ypr[1] - target[1])*math.pi/180)) + sig(float(abs(list_ypr[2] - target[2])*math.pi/180))
    #     return math.sqrt((list_ypr[0] - target[0]) ** 2 + (list_ypr[1] - target[1]) ** 2)
    # else: 
    #     return math.sqrt((list_ypr[0] - target[0]) ** 2 + (list_ypr[1] - target[1]) ** 2 + (list_ypr[2] - target[2]) ** 2)

def calculate_area(box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    x3, y3 = box[2]
    x4, y4 = box[3]
    return (x3-x1) * (y3-y1)

def most_straight_face(face_data,top_n=5):
    target_yaw = 0
    target_pitch = 10
    target_roll = 0
    distances = []
    distances_for_agegender = []
    count = 0
    for frame, attributes in face_data.items():
        if count == 0:
            best_frame = int(frame)
            closest_frames = [int(frame)]
        yaw, pitch, roll = attributes["yaw_pitch_roll"]
        if yaw is None or pitch is None:
            continue
        distance_for_agegender = calculate_distance(attributes["yaw_pitch_roll"],  [target_yaw,target_pitch,target_roll])
        distances_for_agegender.append((frame, distance_for_agegender,attributes["box"][-1]))
        count += 1
    # Sort by distance
    distances_for_agegender.sort(key=lambda x: x[1])
    # Get the best frame (the one with the minimum distance) and Get top N closest frames
    # print(distances_for_agegender[:top_n])
    if len(distances_for_agegender) >0:
        best_frame = int(distances_for_agegender[0][0]) 
        closest_frames = [int(frame) for frame, _,_ in distances_for_agegender[:top_n]]
    return closest_frames, best_frame

def verify_face(best_face_crop,speaker_info,idx):
    # cv2.imwrite(f"/tmp/best_face_crop_{idx}.png",best_face_crop)
    if len(speaker_info) == 0:
        return f"speaker-{len(speaker_info)+1}"
    match_Id = None
    for id in list(speaker_info.keys()):
        face_crop = speaker_info[id]["face_img"]
        # cv2.imwrite(f"/tmp/face_crop_{id}.png",face_crop)
        check_face = None
        resp_obj = DeepFace.verify(best_face_crop, face_crop, detector_backend="skip")
        # print(resp_obj)
        check_face = resp_obj["verified"]
        
        if check_face:
            match_Id = id
            break
    if match_Id is None:
        match_Id = f"speaker-{len(speaker_info)+1}"
    
    return match_Id

def find_padding(box):
    padding_ratio = 0.15 #extend box 15% of bigger edge
    box_size_max = max(box[2]-box[0], box[3] - box[1])
    padding_X = (box_size_max - (box[2]-box[0]))/2 + box_size_max*padding_ratio
    padding_Y = (box_size_max - (box[3] - box[1]))/2 + box_size_max*padding_ratio
    return int(padding_X), int(padding_Y)

def combineFace_detectGenderAge(straight_face_data, frame_temp, skip_data = False):
    """
    Verify speakerId for each speaking segment and detect gender, age.
    """
    padding = 20
    speaker_info = {}       #Format element {"speakerID":{"gender":gender, "age": age, "face_img":face_img}}
    speaker_data = []       #Format element []"speakerID":Id, "gender":gender, "age": age, "box":{box}}]
    #Scan each face_data
    face_keys = list(straight_face_data.keys())
    for face_key in face_keys:
        #Get 5 frames have face look like straight best
        closest_frames_keys, best_frame_keys = most_straight_face(straight_face_data[face_key])
        ## Verify face and get speakerID
        best_face_box = straight_face_data[face_key][str(best_frame_keys)]["box"]
        padding_x, padding_y = find_padding(best_face_box)
        best_face_box = [max(0,best_face_box[0]-padding_x),
                                max(0,best_face_box[1]-padding_y) , 
                                min(best_face_box[2]+padding_x, frame_temp[0].shape[1]-1) ,
                                min(best_face_box[3]+padding_y,frame_temp[0].shape[0]-1)]
        best_face_crop = frame_temp[best_frame_keys][best_face_box[1]:best_face_box[3],best_face_box[0]:best_face_box[2]]
        # cv2.imwrite(f"""/tmp/{video_name}_best_face_{best_frame_keys}.png""", best_face_crop)
        speakerID_tmp = verify_face(best_face_crop,speaker_info, face_key)
        #### speakerID_tmp = "speaker-" + str(len(speaker_info.keys())+1)
        if not skip_data:
            #Detect gender, age
            gender_list = []
            age_list = []
            for key in closest_frames_keys:
            # key  = best_frame_keys
                face_box_tmp = straight_face_data[face_key][str(key)]["box"]
                padding_x, padding_y = find_padding(face_box_tmp)
                # print(padding_x,padding_y, face_box_tmp, straight_face_data[face_key][str(key)]["points"])
                face_box_tmp = [max(0,face_box_tmp[0]-padding_x),
                                max(0,face_box_tmp[1]-padding_y) , 
                                min(face_box_tmp[2]+padding_x, frame_temp[0].shape[1]-1) ,
                                min(face_box_tmp[3]+padding_y,frame_temp[0].shape[0]-1)]
                face_crop_tmp = frame_temp[key][face_box_tmp[1]:face_box_tmp[3],face_box_tmp[0]:face_box_tmp[2]]
                gender_tmp,age_tmp = detect_gender_age(face_crop_tmp, genderNet, ageNet)
                # gender_tmp,age_tmp = detect_gender_age_DeepFace(face_crop_tmp)
                gender_list.append(gender_tmp)
                age_list.append(age_tmp)
                # print(f"""/tmp/closest_frames_{key}.png""")
            # print(gender_list, age_list)
            # gender_list = most_frequent(gender_list)
            # age_list = most_frequent(age_list)
            ## Get gender,age if face was detected
            if not speakerID_tmp in list(speaker_info.keys()):
                speaker_info[speakerID_tmp] = {
                    "gender":gender_list,
                    "age": age_list,
                    "face_img":best_face_crop
                }
            else:
                speaker_info[speakerID_tmp]["gender"] = speaker_info[speakerID_tmp]["gender"] + gender_list
                speaker_info[speakerID_tmp]["age"] = speaker_info[speakerID_tmp]["age"] + age_list
            speaker_data.append({
                "speakerID":speakerID_tmp,
                "frames_infos":straight_face_data[face_key]
            })
        
        else:
            if not speakerID_tmp in list(speaker_info.keys()):
                ## Get gender,age if face was detected
                # print()
                speaker_data.append({
                    "speakerID":speakerID_tmp
                })
                speaker_info[speakerID_tmp] = {
                    "face_img":best_face_crop
                }
    if not skip_data:               
        for speakerId in list(speaker_info.keys()):
            speaker_info[speakerId]["gender"] = most_frequent(speaker_info[speakerId]["gender"])
            speaker_info[speakerId]["age"] = most_frequent(speaker_info[speakerId]["age"])
        for idx in range(len(speaker_data)):
            speaker_data[idx]["gender"] = speaker_info[speaker_data[idx]["speakerID"]]["gender"]
            speaker_data[idx]["age"] = speaker_info[speaker_data[idx]["speakerID"]]["age"]
        
    return speaker_data

def get_speaker_block_info(width,height, speaker_frame_infos):
    """ Return
    1: Size of objective (biggest box as represent), 
    2: Size box of frame(%) (biggest box as represent),
    3: Coordinates of biggest box as represent

    Args:
        width (int): 
        height (int): 
        speaker_frame_infos (Dict): Speaker data of frames with format:
        {"frame_number": {
                "box": [317,
                    225,
                    598,
                    581,
                    0.9997963309288025],
                "yaw_pitch_roll": [1,-67,0],
                "points":,
                "lippoints": []
                ]
            },
    Returns:
        list: [1,2,3]
    """
    frame_keys = list(speaker_frame_infos.keys())
    widths = []
    heights = []
    block_union = speaker_frame_infos[frame_keys[0]]["box"]
    box_size_highest = 0
    for frame_key in frame_keys:
        xA, yA, xB, yB ,_= speaker_frame_infos[frame_key]["box"]
        size = (xB - xA)*(yB - yA)
        if size > box_size_highest:
            block_union = [xA, yA, xB, yB ]
            box_size_highest = size
    avg_width = block_union[2] - block_union[0]
    avg_height = block_union[3] - block_union[1]
    avg_ratio_size = round((avg_width / width)*(avg_height / height),5)
    block_union = [[block_union[0],block_union[1]],
                   [block_union[2],block_union[1]],
                   [block_union[2],block_union[3]],
                   [block_union[0],block_union[3]]]
    block_union = [[round(box_[0]/width,5), round(box_[1]/height,5)] for box_ in block_union ]
    return [[avg_width,avg_height], avg_ratio_size,block_union]

def check_speaker_in_section(speakers_data,
                            start_time,
                            end_time,
                            frame_area,
                            video_duration):
    list_speaker_tmp = []
    ratio_allspeakers_areas = None
    speakers_area = 0
    speakerId_list = []
    for speaker_data in speakers_data:
        # Calculate the intersection
        intersect_start = max(start_time, speaker_data["timing-info"]["start-time"])
        intersect_end = min(end_time, speaker_data["timing-info"]["end-time"])
        # Check if there is an actual intersection
        if intersect_start < intersect_end:
            section_speaker_tmp =speaker_data["speaker-info"].copy()
            section_speaker_tmp.pop("speaking")
            section_speaker_tmp["percentage-duration-speaker-cover"] = round((intersect_end - intersect_start)/(end_time-start_time),5)
            section_speaker_tmp["percentage-frame-speaker-cover-cross-duration"] = round(section_speaker_tmp["percentage-duration-speaker-cover"]*speaker_data["size-percentage-of-frame"],5)
            list_speaker_tmp.append(section_speaker_tmp)
            speakers_area += section_speaker_tmp["percentage-frame-speaker-cover-cross-duration"]
            speakerId_list.append(speaker_data["speaker-info"]["speakerID"])
    
    num_speaker = len(uniqueList(speakerId_list))
    
    return num_speaker, list_speaker_tmp,speakers_area

def main():
    # Define args params
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default="", help="Full path to input video")
    parser.add_argument('--audio', default="", help="Full path to input audio")
    # parser.add_argument('--scene_path', default="")
    args = parser.parse_args()

    # Load model mediapipe face
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_faces=50)
    # trackkpVideoFace = KalmanArray()

    #Check GPU and initialize retinaface model
    if torch.cuda.is_available():
        detector = RetinaFace(True, 0.4)
    else:
        detector = RetinaFace(False, 0.4)
    video_input = args.video
    
    if os.path.isfile(video_input):
        list_video = [video_input]
    else: 
        list_video = [os.path.join(video_input,video) for video in  os.listdir(video_input)]
    video_count = 0
    for video in list_video:
        
        path_folder = f"{os.path.dirname(video)}"
        if not video.lower().endswith(('.mp4')) or not os.path.isfile(f"""{path_folder}/{os.path.basename(video).split(".")[0]}_text_object.json""") \
            or os.path.isfile(f"""{path_folder}/{os.path.basename(video).split(".")[0]}_speaker_recognition_results.json"""):#.json"""):
        # if not video.lower().endswith(('.mp4')):
            continue
        video_name = os.path.basename(video).split(".")[0]
        # video_name = "1425726988318872"
        video_path = f"{path_folder}/{video_name}.mp4"
        
        # Load input video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = round(total_frame/fps,3)
        
        if video_duration >= 120:
            continue
        face_data = {}
        frame_temp = []
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_temp.append(frame)
            
        cap.release()
        ### Detect scene/section by pyscenedetect and text on screen
        pyscene_score_diff = get_content_val(video_path,f'{path_folder}/{video_name}_stats.csv')
        diff_text_score = [1]*len(pyscene_score_diff) # detect scene without text on screen
        # speaker_data,diff_text_score = detectTextOCR(frame_temp ,speaker_data, total_frame)     # Detect text using PaddleOCR and normalized results
        scene_data = detect_scene_final(total_frame,pyscene_score_diff,diff_text_score,fps)
        scene_data.to_csv(f"{path_folder}/{video_name}_section_detect.csv",sep=';', float_format='%.5f')      # Save predict scen to file
        
        # Detect face using RetinaFace model
        face_data = get_face_data(frame_temp, detector,fps)
        # # # print(face_data)
        path_final_output = f"{path_folder}/{video_name}_face_data.json"
        with open(path_final_output ,'w') as fp:
            json.dump(face_data, fp, indent=4)
        #     # face_data= json.load(fp)
        ## Filter straight face by face angle(yaw and pitch)
        straight_face_data,face_data_total = filter_straight_face(frame_temp,face_data,face_mesh_256, fps)
        path_final_output = f"{path_folder}/{video_name}_straight_face_data.json"
        with open(path_final_output ,'w') as fp:
            json.dump(straight_face_data, fp, indent=4)
            # straight_face_data = json.load(fp)
        face_data_total = straight_face_data
        # Combine the same faces and detect gender / age.
        speaker_data = combineFace_detectGenderAge(straight_face_data, frame_temp)
        combined_face_data_total = combineFace_detectGenderAge(face_data_total, frame_temp, skip_data=True)
        numberOfPeople = len(combined_face_data_total)
        
        path_final_output = f"{path_folder}/{video_name}_speaker_data.json"
        with open(path_final_output ,'w') as fp:
            json.dump(speaker_data, fp, indent=4)
        frame_area = width*height
        
        ## Define speakers-data
        speakers_data = []
        for idx in tqdm(range(len(speaker_data)), desc="Processing Speakers-data"):
            speaker_obj =  speaker_data[idx]
            speaker_frame_infos = speaker_obj["frames_infos"]
            frame_timetamp = [int(frame_idx) for frame_idx in list(speaker_frame_infos.keys())]
            start_time = round(min(frame_timetamp)/fps,3)
            end_time = round((max(frame_timetamp)+1)/fps,3)
            speaker_duration = round(end_time - start_time,3)
            block_size, block_size_ratio, face_coor = get_speaker_block_info(width,height, speaker_frame_infos)
            sectors_list = get_sector_frame(1, 1,sector_map,[face_coor[0],face_coor[2]])
            lippoints_list = [speaker_frame_infos[frame_idx]["lippoints"] for frame_idx in list(speaker_frame_infos.keys()) if speaker_frame_infos[frame_idx]["lippoints"] != None]
            
            speaking = lip_movement_detection(lippoints_list, "continuous")
            
            data_tmp ={
                    "speaker-info":{
                        "speakerID":speaker_obj["speakerID"],                              # 3-a-i: Speaker ID (if the same speaker is shown multiple times, should be the same ID)
                        "gender":speaker_obj["gender"],                            # 3-a-ii: Gender (Male/Female)
                        "age-range":speaker_obj["age"],                               # 3-a-iii: Est. age range
                        "speaking":speaking                                    # 3-a-iv: Speaking? True is speaking, False is non-speaking
                    },
                    "timing-info":{
                        "start-time":start_time,                                  # Time the speaker is first shown
                        "end-time":end_time,                                   # Time the speaker is last shown
                        "duration":speaker_duration,                                   # Duration time from entry to exit
                        "percentage-duration-speaker-cover":round(speaker_duration/video_duration,5)                      # Duration of speaker on video as % of video length
                    },
                    "size-objective":block_size,                                # Size (objective), rectangle
                    "size-percentage-of-frame":block_size_ratio,                              # Size (% of video frame)
                    "location-objective":face_coor,      # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                    "location-sector":sectors_list,                             # 3-f: Location (section of video frame) (map to be included)
                    "speaker-show":"shoulderup"                            # 3-g: What is shown of speaker? => 1 in 3 options "shoulderup/waistup/fullbody"
                }
            speakers_data.append(data_tmp)
        ## Define sections-data
        sections_data =[]
        for idx in tqdm(range(len(scene_data["scene-name"])), desc="Processing Section-data"):
            start_time = scene_data["start_time"][idx]
            end_time = scene_data["end_time"][idx]
            if start_time == end_time:
                continue
            num_speaker, list_speaker_tmp,ratio_allspeakers_areas = check_speaker_in_section(speakers_data,
                                                                                            int(start_time),
                                                                                            int(end_time),
                                                                                            frame_area,round(end_time-start_time,3))
            data_tmp = {
                    "section-name":scene_data["scene-name"][idx],                                
                    "duration":round(end_time-start_time,3),                                        # 2-a: Section Duration (second)
                    "amount-of-speakers":num_speaker,                                      # 2-b: Amount of Speakers in Section 
                    "speaker-infos":list_speaker_tmp,
                    "percentage-frame-allspeakers-cover-cross-duration":ratio_allspeakers_areas                          # 2-c: % of Section covered by speakers (% of video frame across duration)
                }
            sections_data.append(data_tmp)
        ## Define video-infos
        num_speaker, list_speaker_tmp,ratio_allspeakers_areas = check_speaker_in_section(speakers_data,0,video_duration,frame_area,video_duration)
        
        
        video_infos = {                                                # 1: For each video
                "video-name":video_name,                                         
                "video-resolution":[width,height],                                    # 1-a: Video resolution in format [weight, height]
                "video-duration":video_duration,                                            # 1-b: Video Duration (seconds)
                "amount-of-speakers":num_speaker,                                          # 1-c: Amount of Speakers in Video
                "speaker-infos":list_speaker_tmp,
                "percentage-frame-allspeakers-cover-cross-duration":ratio_allspeakers_areas,                                 # 1-d: % of video covered by speakers (% of video frame across duration)
                "numberOfPeople":numberOfPeople
            }
        
        output_speaker_reg = {
            "data":[
                {
                    "video-info":video_infos,
                    "sections-data":sections_data,
                    "speakers-data":speakers_data
                }
            ]
        }
        
        # ##Export data to output json      
        path_final_output = f"{path_folder}/{video_name}_speaker_recognition_results.json"
        with open(path_final_output ,'w') as fp:
            json.dump(output_speaker_reg, fp, indent=4)
            # font_detect = json.load(fp)
        video_count +=1
        if video_count ==20:
            break
        

ageProto=f"{PATH}/RetinaFace_tf2/gender-age-data/age_deploy.prototxt"
ageModel=f"{PATH}/RetinaFace_tf2/gender-age-data/age_net.caffemodel"
genderProto=f"{PATH}/RetinaFace_tf2/gender-age-data/gender_deploy.prototxt"
genderModel=f"{PATH}/RetinaFace_tf2/gender-age-data/gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

# faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

if __name__ == "__main__":
    main()
    # resp_obj = DeepFace.verify("/tmp/best_face_crop.png", "/tmp/best_face_crop_face_3.png", detector_backend="skip")
    # print(resp_obj)
    # date = int(time.time())
    # print(date)