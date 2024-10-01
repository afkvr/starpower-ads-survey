import cv2
import time
import os
import sys
import numpy as np
from PIL import Image
import time
import pandas as pd
from tqdm import tqdm
import json
import argparse
import mediapipe as mp
import torch
import matplotlib as mpl
import math
import tensorflow as tf
from deepface import DeepFace
from .src.retinafacetf2.retinaface import RetinaFace
landmark_points_61_67 = [82,13,312,317,14,87]
PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# from memory_profiler import profile


def sig(x):
 return 1/(1 + np.exp(-x))

gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.2)
config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
session = tf.compat.v1.Session(config=config)

FACEMESH_pose_estimation = [34,264,168,33, 263]

## Define body part COCO
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                   "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                   "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
                   "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

# POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
#                 ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
#                 ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
#                 ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
#                 ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

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
    straight_threshold = 0.3   # Mean how many percent face look straight in time is consider as straight face
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

def get_face_data(frame_temp,fps):
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
    return closest_frames, best_frame, distances_for_agegender[0][1]

def verify_face(best_face_crop,speaker_info,idx):
    # cv2.imwrite(f"/tmp/best_face_crop_{idx}.png",best_face_crop)
    if len(speaker_info) == 0:
        return f"speaker-{len(speaker_info)+1}"
    match_Id = None
    for id in list(speaker_info.keys()):
        face_crop = speaker_info[id]["face_img"]
        # cv2.imwrite(f"/tmp/face_crop_{id}.png",face_crop)
        check_face = None
        resp_obj = DeepFace.verify(best_face_crop, face_crop, detector_backend="skip",threshold=0.4)
        print(resp_obj)
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

def checkPointsInBox(box,points):
    x1, y1, x2, y2 = box
    total_points = len(points)
    inside_count = sum(1 for x, y in points if (x1 <= x <= x2) and (y1 <= y <= y2))
    percentage_inside = float(inside_count / total_points)
    return percentage_inside

def find_number_in_multiple_lists(multiple_list, X):
    result = []
    for i, sublist in enumerate(multiple_list):
        if X in sublist:
            result.append((i, sublist.index(X)))
    return result

# @profile
def get_pose_Mediapipe(list_img, speaker_data, conf_thresh = 0.6):
    """
    Detect body pose using mediapipe. Cause model detect only 1 person, so we make loop that do these steps:
    1 - Extract body pose and segmentation mask
    2 - Get waist points, fullbody and head.
    3 - Matching speakerID with pose by checkin headpoint in speaker face box
    4 - Remove human in image by segmentation mask and forward loop
    Args:
        list_img (_type_): list images section contain face
        speaker_data (_type_): data speaker total
        pose_256 (_type_): model mediapipe Pose estimation
        conf_thresh (float, optional): Threshold to check visibility of list score. Defaults to 0.5.

    Returns:
        res: Body part showed
    """
    
    mp_pose = mp.solutions.pose
    pose_256 = mp_pose.Pose(min_detection_confidence=0.2,enable_segmentation=True)
    
    def check_mse(img1, img2, thresh=0.01):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff**2)
        mse = err/(float(h*w))
        return mse <= thresh
    
    speaker_frames = []
    speaker_box = []
    for idx in range(len(speaker_data)):
        
        list_frame = []
        box_tmp = []
        for key in list(speaker_data[idx]["frames_infos"].keys()):
            list_frame.append(int(key))
            box_tmp.append(speaker_data[idx]["frames_infos"][key]["box"])
        speaker_box.append(box_tmp)
        speaker_frames.append(list_frame)
        speaker_data[idx]["speaker-show"] = { "Full_body":[],
                                                "Waist_up":[],
                                                "Shoulders_up":[]
                                                }
    white_frame =  np.zeros((100,100,3), np.uint8) + 255
    for idx_frame in tqdm(range(len(list_img)), desc="Detecting pose"):
        check_exist_speaker =  find_number_in_multiple_lists(speaker_frames,idx_frame)
        if check_exist_speaker == []:
            continue
        # print("check_exist_speaker : ", check_exist_speaker)
        frame = list_img[idx_frame]
        vis_list = []
        pose_list = []
        count = 0
        previouse_mask = np.zeros(list_img[0].shape[:2],np.uint8)
        while True:
            count +=1
            if count ==20:      #Prevent mediapipe pose infinity loop. Max detect people should under 20
                break
            ### Extract body part using mediapipe Pose
            # print("Retrying ", count)
            pose_256.process(white_frame)
            results= pose_256.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not results.pose_landmarks:
                # print("No human detected ", count)
                break
            seg_mask = np.array(results.segmentation_mask).astype(np.uint8)*255
            
            if check_mse(seg_mask,previouse_mask):
                # print("Diff mask low ", count)
                break
            previouse_mask = seg_mask
            
            pose_landmarks = results.pose_landmarks
            vis_tmp = []
            pose_tmp = []
            height, width = frame.shape[:2]
            # print(pose_landmarks)
            for idx in range(33):
                x = int(pose_landmarks.landmark[idx].x * width)
                y = int(pose_landmarks.landmark[idx].y * height)
                vis = pose_landmarks.landmark[idx].visibility
                vis_tmp.append(vis)
                pose_tmp.append([x,y])
            # vis_list.append(vis_tmp)
            # pose_list.append(pose_tmp)
            
            # Create a 3-channel binary mask
            three_channel_mask = np.array(cv2.merge([seg_mask, seg_mask, seg_mask]).astype(np.uint8))
            frame = cv2.add(frame, three_channel_mask)
            # cv2.imwrite("/tmp/mask.jpg", three_channel_mask)
            del results
            face_pose_list = pose_tmp[:11]
            # print("face_pose_list : ", len(face_pose_list), face_pose_list )
            ## Matching data to speaker
            best_match_point_score = 0
            best_match_speaker_idx = None
            # cv2.polylines(frame, [np.array(face_pose_list,np.int32 )],True, (0,255,0), 1)

            for check_exist_speaker_tmp in  check_exist_speaker:
                speaker_idx, box_idx = check_exist_speaker_tmp
                # print("box: ", speaker_box[speaker_idx][box_idx][:-1])
                score = checkPointsInBox(speaker_box[speaker_idx][box_idx][:-1], face_pose_list)
                if score > best_match_point_score:
                    # print("Update: ",best_match_point_score, score )
                    best_match_speaker_idx = speaker_idx
                # print(f"""SCORE: {score} - {idx_frame}""")
            #     cv2.rectangle(frame,[speaker_box[speaker_idx][box_idx][0],speaker_box[speaker_idx][box_idx][1]],[speaker_box[speaker_idx][box_idx][2],speaker_box[speaker_idx][box_idx][3]],(0,0,255),2)
            # cv2.imwrite("/tmp/draw_face.png", frame)
            if best_match_speaker_idx == None:
                # print("No matching face")
                continue
            # print("Best match point: ",best_match_point_score, best_match_speaker_idx )
            # print(speaker_data[best_match_speaker_idx]["speakerID"] )
            # print("Full_body: ", np.mean(vis_tmp[fullbody_points[0]:fullbody_points[-1] +1]),vis_tmp[fullbody_points[0]:fullbody_points[-1] +1] )
            # print("Waist_up: ", np.mean(vis_tmp[waist_points[0]:waist_points[-1] +1]), vis_tmp[waist_points[0]:waist_points[-1] +1])
            if np.mean(vis_tmp[fullbody_points[0]:fullbody_points[-1] +1]) >= conf_thresh:
                speaker_data[best_match_speaker_idx]["speaker-show"]["Full_body"].append(idx_frame)
            if np.mean(vis_tmp[waist_points[0]:waist_points[-1] +1]) >= conf_thresh:
                speaker_data[best_match_speaker_idx]["speaker-show"]["Waist_up"].append(idx_frame)
            # if np.mean(vis_tmp[waist_points[0]:waist_points[-1] +1]) >= conf_thresh:
            speaker_data[best_match_speaker_idx]["speaker-show"]["Shoulders_up"].append(idx_frame)
            
            
    # Normalized speaker-show to percentage
    for idx in range(len(speaker_data)):
        # print("Full_body: ", len(uniqueList(speaker_data[idx]["speaker-show"]["Full_body"])),len(speaker_frames[idx]), uniqueList(speaker_data[idx]["speaker-show"]["Full_body"]),speaker_frames[idx] )
        # print("Waist_up: ", len(uniqueList(speaker_data[idx]["speaker-show"]["Waist_up"])),len(speaker_frames[idx]), uniqueList(speaker_data[idx]["speaker-show"]["Waist_up"]),speaker_frames[idx] )
        # print("Shoulders_up: ", len(uniqueList(speaker_data[idx]["speaker-show"]["Shoulders_up"])),len(speaker_frames[idx]), uniqueList(speaker_data[idx]["speaker-show"]["Shoulders_up"]),speaker_frames[idx] )
        speaker_data[idx]["speaker-show"]["Full_body"] = round(len(uniqueList(speaker_data[idx]["speaker-show"]["Full_body"]))/len(speaker_frames[idx]), 5)
        speaker_data[idx]["speaker-show"]["Waist_up"] = round(len(uniqueList(speaker_data[idx]["speaker-show"]["Waist_up"]))/len(speaker_frames[idx]), 5)
        speaker_data[idx]["speaker-show"]["Shoulders_up"] = round(len(uniqueList(speaker_data[idx]["speaker-show"]["Shoulders_up"]))/len(speaker_frames[idx]), 5)    
    
    del mp_pose, pose_256, speaker_frames, speaker_box
    
    return speaker_data

#Check GPU and initialize retinaface model
# if torch.cuda.is_available():
#     detector = RetinaFace(True, 0.4)
# else:
#     detector = RetinaFace(False, 0.4)

# poseProto = f"{PATH}/models/openpose/openpose_pose_coco.prototxt"
# protoModel = f"{PATH}/models/openpose/openpose_pose_coco.caffemodel"
# poseNet = cv2.dnn.readNet(poseProto, protoModel)
# poseSize = 368
# poseScale = 0.003922

waist_points = [23,24]
fullbody_points = [25,26,27,28]
# shoulder_points = [25,26,27,28]
head_points = [0, 10]
# mp_pose = mp.solutions.pose
# pose_256 = mp_pose.Pose(min_detection_confidence=0.5, enable_segmentation=True)
# if __name__ == "__main__":
#     mp_pose = mp.solutions.pose
#     pose_256 = mp_pose.Pose(min_detection_confidence=0.5, enable_segmentation=True)
    # image = cv2.imread("/home/anlab/Pictures/man-woman.jpg")
    # image = cv2.imread("/tmp/Frame_1_af.png")
    # get_pose_COCO([image], None)
    # get_pose_Mediapipe([image], None, pose_256)
    # video_path = "/media/anlab/data/ads_video/ads_for_ha/shop/382347757934654.mp4"
    # speaker_data_path = "/media/anlab/data/ads_video/ads_for_ha/shop/382347757934654_speaker_data_2.json"
    # with open(speaker_data_path ,'r') as fp:
    #     speaker_data_input = json.load(fp)
        
    # # Load input video
    # cap = cv2.VideoCapture(video_path)
    # total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frame_temp = []
    # while True:
    #     ret, frame = cap.read()
    #     if not ret or frame is None:
    #         break
    #     frame_temp.append(frame)
    # cap.release()
    # print("frame_temp: ", len(frame_temp), total_frame)
    # speaker_data = get_pose_Mediapipe(frame_temp, speaker_data_input, pose_256)
    # path_final_output = f"/tmp/382347757934654_speaker_data_test.json"
    # with open(path_final_output ,'w') as fp:
    #     json.dump(speaker_data, fp, indent=4)
    
    # print(image)
#     main()
    # resp_obj = DeepFace.verify("/tmp/best_face_crop.png", "/tmp/best_face_crop_face_3.png", detector_backend="skip")
    # print(resp_obj)
    # date = int(time.time())
    # print(date)
    