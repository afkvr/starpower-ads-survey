import cv2
import os
import sys
# from utils import client
import json
import base64
import argparse
import numpy as np
from RetinaFace_tf2.src.retinafacetf2.retinaface import RetinaFace
from RetinaFace_tf2.detect_gender_age import detect_gender_age, most_frequent,detect_gender_age_MiVOLO
from RetinaFace_tf2.detect_speaker import *

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



def get_face_data(frame_image, detector,facemesh):
    yaw_threshold = 40  # Mean legal yaw in [-50 , 50]
    pitch_threshold = 45 
    face_data = {}
    face_data_mesh = {}
    frame = frame_image
    h,w = frame_image.shape[:2]
    faces, keypoints = detector.detect(frame, 0.5)
    
    if len(faces) == 0:# not results.multi_face_landmarks:
        return None
    # Matching face box with face_data
    face_idx = 0
    for face in faces:
        list_face_data_key = list(face_data.keys()) 
        new_key = "face_"+str(len(list_face_data_key)+1)
        face_data[new_key] = {"box":face.tolist(), "points":keypoints[face_idx].tolist()}
        face_idx +=1
    for frame_key in face_data.keys():
            box_face_tmp = [int(coor) for coor in face_data[frame_key]["box"][:-1]] + [face_data[frame_key]["box"][-1]]  # Format [xA, yA, xB, yB] with A is top-left corner, B is bottom-right
            xA, yA, xB, yB = [max(box_face_tmp[0],0),max(box_face_tmp[1],0),min(box_face_tmp[2],w),min(box_face_tmp[3],h)]
            crop_img = frame[yA:yB, xA:xB]
            results = facemesh.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                # face_data_mesh[frame_key] ={
                #                             "box": box_face_tmp,
                #                             "yaw_pitch_roll":[None,None,None],
                #                             "points": None,
                #                             "lippoints":None
                #                         }
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
                face_data_mesh[frame_key] ={
                                            "box": box_face_tmp,
                                            "yaw_pitch_roll":[yaw,pitch,roll],
                                            "points": face_data[frame_key]["points"],
                                            "lippoints":lippoints
                                        }
    return face_data_mesh

def combineFace_detectGenderAge(straight_face_data, frame_image, skip_data = False):
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
        # closest_frames_keys, best_frame_keys = most_straight_face(straight_face_data[face_key])
        ## Verify face and get speakerID
        best_face_box = straight_face_data[face_key]["box"]
        padding_x, padding_y = find_padding(best_face_box)
        best_face_box = [max(0,best_face_box[0]-padding_x),
                                max(0,best_face_box[1]-padding_y) , 
                                min(best_face_box[2]+padding_x, frame_image.shape[1]-1) ,
                                min(best_face_box[3]+padding_y,frame_image.shape[0]-1)]
        best_face_crop = frame_image[best_face_box[1]:best_face_box[3],best_face_box[0]:best_face_box[2]]
        # cv2.imwrite(f"""/tmp/best_face_{best_frame_keys}.png""", best_face_crop)
        # print(face_key, closest_frames_keys, best_frame_keys)
        speakerID_tmp = verify_face(best_face_crop,speaker_info, face_key)
        #### speakerID_tmp = "speaker-" + str(len(speaker_info.keys())+1)
        if not skip_data:
            #Detect gender, age
            gender_list = []
            age_list = []
            # key  = best_frame_keys
            face_box_tmp = straight_face_data[face_key]["box"]
            padding_x, padding_y = find_padding(face_box_tmp)
            face_box_tmp = [max(0,face_box_tmp[0]-padding_x),
                            max(0,face_box_tmp[1]-padding_y) , 
                            min(face_box_tmp[2]+padding_x, frame_image.shape[1]-1) ,
                            min(face_box_tmp[3]+padding_y,frame_image.shape[0]-1)]
            face_crop_tmp = frame_image[face_box_tmp[1]:face_box_tmp[3],face_box_tmp[0]:face_box_tmp[2]]
            cv2.imwrite(f"/tmp/face_crop_{face_key}.png",face_crop_tmp)
            gender_tmp,age_tmp = detect_gender_age(face_crop_tmp, genderNet, ageNet)
            # gender_tmp,age_tmp = detect_gender_age_DeepFace(face_crop_tmp)
            gender_list.append(gender_tmp)
            age_list.append(age_tmp)
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
                "frames_infos":{"0":straight_face_data[face_key]}
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



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

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
 # Load model mediapipe face
mp_face_mesh = mp.solutions.face_mesh
face_mesh_256 = mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2, max_num_faces=50)
# trackkpVideoFace = KalmanArray()

#Check GPU and initialize retinaface model
if torch.cuda.is_available():
    detector = RetinaFace(True, 0.4)
else:
    detector = RetinaFace(False, 0.4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="")
    args = parser.parse_args()
    image_input = args.image
    if os.path.isfile(image_input):
        list_image = [image_input]
    else:
        list_image =[]
        # list_image = [os.path.join(image_input, image)
        #               for image in os.listdir(image_input)]
        for path, subdirs, files in os.walk(image_input):
            for name in files:
                list_image.append(os.path.join(path, name))
    num_threads = 1
    for filename in list_image:
        # count = 1
        # filename = "781490520120333.mp4"
        # text_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):  # and not os.p
            print(filename)
            speakers_data = []
            frame_image = cv2.imread(filename)
            file_basename = os.path.basename(filename).split('.')[0]
            file_dir = os.path.dirname(filename)
            metadata_path = os.path.join(file_dir,f"{file_basename}/{file_basename}_metadata.json")
            if os.path.exists(metadata_path):
                print(f"File {filename} does not have metadata.json")
                continue
            face_data_mesh = get_face_data(frame_image,detector, face_mesh_256)
            print(face_data_mesh)
            if face_data_mesh is not None:
                speaker_data = combineFace_detectGenderAge(face_data_mesh, frame_image)
                print(speaker_data)
                speaker_data = get_pose_Mediapipe([frame_image], speaker_data)
                
                ## Export final data speaker
                for idx in tqdm(range(len(speaker_data)), desc="Processing Speakers-data"):
                    speaker_obj =  speaker_data[idx]
                    speaker_frame_infos = speaker_obj["frames_infos"]
                    frame_timetamp = [int(frame_idx) for frame_idx in list(speaker_frame_infos.keys())]
                    # start_time = round(min(frame_timetamp)/fps,3)
                    # end_time = round((max(frame_timetamp)+1)/fps,3)
                    # speaker_duration = round(end_time - start_time,3)
                    block_size, block_size_ratio, face_coor = get_speaker_block_info(frame_image.shape[1], frame_image.shape[0], speaker_frame_infos)
                    sectors_list = get_sector_frame(1, 1,sector_map,[face_coor[0],face_coor[2]])
                    lippoints_list = [speaker_frame_infos[frame_idx]["lippoints"] for frame_idx in list(speaker_frame_infos.keys()) if speaker_frame_infos[frame_idx]["lippoints"] != None]
                    
                    speaking = False#lip_movement_detection(lippoints_list, "continuous")
                    
                    data_tmp ={
                            "speaker-info":{
                                "speakerID":speaker_obj["speakerID"],                              # 3-a-i: Speaker ID (if the same speaker is shown multiple times, should be the same ID)
                                "gender":speaker_obj["gender"],                            # 3-a-ii: Gender (Male/Female)
                                "age-range":speaker_obj["age"],                               # 3-a-iii: Est. age range
                                "speaking":speaking                                    # 3-a-iv: Speaking? True is speaking, False is non-speaking
                            },
                            "size-objective":block_size,                                # Size (objective), rectangle
                            "size-percentage-of-frame":block_size_ratio,                              # Size (% of video frame)
                            "location-objective":face_coor,      # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                            "location-sector":sectors_list,                             # 3-f: Location (section of video frame) (map to be included)
                            "speaker-show":speaker_obj["speaker-show"]                  # 3-g: What is shown of speaker? => 1 in 3 options "shoulderup/waistup/fullbody"
                        }
                    speakers_data.append(data_tmp)
                # print(speakers_data)
            with open(metadata_path ,'r') as fp:
                final_metadata = json.load(fp)
                
            final_metadata["Speaker-detection"] = speakers_data
            # print(final_metadata)
            with open(metadata_path ,'w') as fp:
                json.dump(final_metadata, fp, indent=4, cls=NpEncoder)
            # break
            # message_contents = message_contents.replace("```json", "")
            # message_contents = message_contents.replace("```", "")
            # message_data = json.loads(message_contents)