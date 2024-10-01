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
# from moviepy.editor import VideoFileClip
import shutil
import time
import pandas as pd
from tqdm import tqdm
import json
import argparse
import mediapipe as mp
import torch
import matplotlib as mpl
from src.retinafacetf2.retinaface import RetinaFace
import math
FACEMESH_pose_estimation = [34,264,168,33, 263]

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


def ffmpeg_encoder(outfile, fps, width, height,bitrate):
    LOGURU_FFMPEG_LOGLEVELS = {
        "trace": "trace",
        "debug": "debug",
        "info": "info",
        "success": "info",
        "warning": "warning",
        "error": "error",
        "critical": "fatal",
        }
    # if torch.cuda.is_available():
    #     frames = ffmpeg.input(
    #     "pipe:0",
    #     format="rawvideo",
    #     pix_fmt="rgb24",
    #     vsync="1",
    #     s='{}x{}'.format(width, height),
    #     r=fps,
    #     hwaccel="cuda",
    #     hwaccel_device="0",
    #     # hwaccel_output_format="cuda",
    #     thread_queue_size=1,
    #     )
    #     encoder_ = subprocess.Popen(
    #     ffmpeg.compile(
    #         ffmpeg.output(
    #             frames,
    #             outfile,
    #             pix_fmt="yuv420p",
    #             # vcodec="libx264",
    #             vcodec="h264_nvenc",
    #             acodec="copy",
    #             r=fps,
    #             cq=17,
    #             maxrate=bitrate,
    #             minrate= bitrate,
    #             bufsize= "8M",
    #             # rc="vbr",
    #             vsync="1",
    #             # async=4,
    #         )
    #         .global_args("-hide_banner")
    #         .global_args("-nostats")
    #         .global_args(
    #             "-loglevel",
    #             LOGURU_FFMPEG_LOGLEVELS.get(
    #                 os.environ.get("LOGURU_LEVEL", "INFO").lower()
    #             ),
    #         ),
    #         overwrite_output=True,
    #     ),
    #     stdin=subprocess.PIPE,
    #     # stdout=subprocess.DEVNULL,
    #     # stderr=subprocess.DEVNULL,
    #     )
    # else:
    frames = ffmpeg.input(
    "pipe:0",
    format="rawvideo",
    pix_fmt="rgb24",
    vsync="1",
    s='{}x{}'.format(width, height),
    r=fps,
    thread_queue_size=1,
    )
    encoder_ = subprocess.Popen(
    ffmpeg.compile(
        ffmpeg.output(
            frames,
            outfile,
            pix_fmt="yuv420p",
            # vcodec="libx264",
            vcodec="libx264",
            acodec="copy",
            r=fps,
            crf=17,
            maxrate=bitrate,
            minrate= bitrate,
            vsync="1",
            # async=4,
        )
        .global_args("-hide_banner")
        .global_args("-nostats")
        .global_args(
            "-loglevel",
            LOGURU_FFMPEG_LOGLEVELS.get(
                os.environ.get("LOGURU_LEVEL", "INFO").lower()
            ),
        ),
        overwrite_output=True,
        cmd="/home/anlab/anaconda3/envs/chatgpt/bin/ffmpeg"
    ),
    stdin=subprocess.PIPE,
    # stdout=subprocess.DEVNULL,
    # stderr=subprocess.DEVNULL,
    )   
    return encoder_

def write_frame(images,encoder_video):
    image_draw = cv2.cvtColor(images,cv2.COLOR_RGB2BGR)
    imageout = Image.fromarray(np.uint8(image_draw))
    encoder_video.stdin.write(imageout.tobytes())

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


def visual_face_box(list_img , speaker_data, encode_video):
    name = "tab20"
    cmap = mpl.colormaps[name]  
    colors = cmap.colors 
    count = 0
    
    
    # Get list frame_index
    face_keys = list(speaker_data.keys())
    
    for frame in (list_img):
        for face_key in face_keys:
            face_data_tmp = speaker_data[face_key]
            if str(count) in list(face_data_tmp.keys()):
                faceIdx = face_keys.index(face_key)
                # listpoint = []
                color_tmp = [tmp*255 for tmp in colors[faceIdx]]
                face_box = face_data_tmp[str(count)]
                cv2.rectangle(frame,[int(face_box[0]), int(face_box[1])],[int(face_box[2]), int(face_box[3])],color_tmp,2)
                cv2.putText(frame, f"{face_key}", [int(face_box[0]), int(face_box[1])-15],cv2.FONT_HERSHEY_SIMPLEX,1,color_tmp,2,cv2.LINE_AA)
        # return faceIdx
        count +=1
        write_frame(frame, encode_video)
    encode_video.stdin.flush()
    encode_video.stdin.close()
    # return image

def filter_face(list_img, face_data, facemesh, fps):
    h,w = list_img[0].shape[:2]
    min_time = 0.5 * fps # Set minimum time second to detect a face is legal
    yaw_threshold = 50  # Mean legal yaw in [-50 , 50]
    pitch_threshold = 50    
    straight_threshold = 0.1   # Mean how many percent face look straight in time is consider as straight face
    face_keys = list(face_data.keys())
    number_face_list = [False]*len(face_keys)
    
    for face_key in face_keys:
        face_data_tmp = face_data[face_key]
        frame_keys = list(face_data_tmp.keys())
        if len(frame_keys) < min_time:
            continue
        list_face_angles = [False]*len(frame_keys)
        
        for frame_key in frame_keys:
            box_face_tmp = [int(coor) for coor in face_data_tmp[frame_key]]   # Format [xA, yA, xB, yB] with A is top-left corner, B is bottom-right
            xA, yA, xB, yB = [max(box_face_tmp[0],0),max(box_face_tmp[1],0),min(box_face_tmp[2],w),min(box_face_tmp[3],h)]
            crop_img = list_img[int(frame_key)][yA:yB, xA:xB]
            results = facemesh.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                continue
            face_landmarks = results.multi_face_landmarks[0]
            bbox_w = xB-xA
            bbox_h = yB-yA
            
            # Extract Face pose keypoints
            posePoint = []
            for i in range(len(FACEMESH_pose_estimation)):
                idx = FACEMESH_pose_estimation[i]
                x = face_landmarks.landmark[idx].x
                y = face_landmarks.landmark[idx].y

                realx = x * bbox_w 
                realy = y * bbox_h
                posePoint.append((realx, realy))
            
             #Cal yaw, pitch and roll distance for face
            yaw, pitch, roll = facePose(posePoint[0], posePoint[1], posePoint[2], posePoint[3], posePoint[4])
            
            if abs(yaw) <= yaw_threshold and abs(pitch) <= pitch_threshold:
                list_face_angles[frame_keys.index(frame_key)] = True
            # print(yaw, pitch, roll)
        if float(np.sum(list_face_angles)/len(frame_keys)) > straight_threshold:
            number_face_list[face_keys.index(face_key)] = True
                    
    return number_face_list

def main():
    # Define args params
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default="")
    # parser.add_argument('--scene_path', default="")
    args = parser.parse_args()
    # Load input scene timestamp
    
    # Load model mediapipe face
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_faces=50)
    trackkpVideoFace = KalmanArray()
    
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
        
    for video in list_video:
        
        if not video.lower().endswith(('.mp4')):
            continue
        video_name = os.path.basename(video).split(".")[0]
        path_video_folder = f"{os.path.dirname(video)}/{video_name}"
        # try: 
        #     os.makedirs(path_video_folder)
        #     # os.makedirs(path_video_scene)
        #     # os.makedirs(path_video_image)
        # except Exception as e:
        #     print("Error make folder: ", e)
        #     pass
        
        # Load input video
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        face_data = {}
        frame_temp = []
        count  = 0 

        for _ in tqdm(range(total_frame), desc="Loading video"):
            ret, frame = cap.read()
            
            # if count <=100:
            #     count += 1 
            #     continue
            if not ret or frame is None :#or count == 50:
                break
            frame_temp.append(frame)
            faces, _ = detector.detect(frame, 0.5)
            # results = face_mesh_256.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(faces) == 0:# not results.multi_face_landmarks:
                # print("Frame: ", count, " No face detected")
                # write_frame(frame, encode_video)
                count += 1 
                continue
            
            # Matching face box with face_data
            for face in faces:
                
                list_face_data_key = list(face_data.keys()) 
                # print(face)
                IoU_highest = 0
                match_key = None
                for face_key_tmp in list_face_data_key:
                    face_data_tmp = face_data[face_key_tmp]
                    face_box_latest = face_data_tmp[list(face_data_tmp.keys())[-1]]
                    iou = calculate_iou(face[:-1], face_box_latest)
                    if iou >IoU_highest:
                        IoU_highest = iou
                        match_key = face_key_tmp
                if IoU_highest > 0.7:
                    face_data[match_key][str(count)] = face[:-1]
                else:
                    new_key = "face_"+str(len(list_face_data_key)+1)
                    face_data[new_key] = {str(count):face[:-1]}
                # print(face_data)
            # write_frame(frame, encode_video)
            count += 1 
        # print(face_data)
        cap.release()
        
        #Save detect to json
        df_raw = pd.DataFrame(face_data)
        path_raw_face  =f"{path_video_folder}_face_raw_data.json"
        df_raw.to_json(path_raw_face,indent=4)
        speaker_check = filter_face(frame_temp,face_data,face_mesh_256, fps)
        face_keys = list(face_data.keys())
        speaker_data = {face_keys[idx]:face_data[face_keys[idx]] for idx in range(len(speaker_check)) if speaker_check[idx] is True}
        
        df_speaker = pd.DataFrame(speaker_data)
        path_save_speaker  =f"{path_video_folder}_speaker_data.json"
        df_speaker.to_json(path_save_speaker,indent=4)
        print(video_name, " - ", np.sum(speaker_check))
        encode_video = ffmpeg_encoder(path_video_folder+"_visualbox_face.mp4",fps, width,height,"5M")
        visual_face_box(frame_temp, speaker_data, encode_video)
        # encode_video.stdin.flush()
        # encode_video.stdin.close()
        # Define face angle info variable
        
        # Process frame to detect keypoint and face angle: yaw, pitch and roll
        # return

if __name__ == "__main__":
    main()
    
    