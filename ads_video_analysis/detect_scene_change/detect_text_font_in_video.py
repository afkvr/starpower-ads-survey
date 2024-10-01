import cv2
import numpy as np
import json
import subprocess
import ffmpeg
from PIL import Image, ImageFont, ImageDraw  
import os
from paddleocr import PaddleOCR, draw_ocr
import re
from tqdm import tqdm
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import pandas as pd
from scenedetect import SceneManager,StatsManager,  open_video, ContentDetector, split_video_ffmpeg, AdaptiveDetector, SceneDetector
import math
# import tesserocr
import base64
import requests
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

os.environ["TESSDATA_PREFIX"]="/home/anlab/anaconda3/envs/chatgpt/share/tessdata/"

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

def get_scene_change_frame(video_path,stat_save_file):
    video = open_video(video_path)
    # video_manager = VideoManager([video_path])
    scene_manager = SceneManager(stats_manager=StatsManager())

    # Initialize ContentDetector with a save_images parameter set to False.
    content_detector = AdaptiveDetector()
    
    # Add the detector to the scene manager.
    scene_manager.add_detector(content_detector)
    scene_manager.detect_scenes(video=video)
    scene_manager.stats_manager.save_to_csv(csv_file=stat_save_file)
    # df = pd.read_csv(stat_save_file)
    # content_vals = df['content_val']
    content_vals = scene_manager.get_scene_list()
    scene_change_frame = [] 
    for idx in range(len(content_vals)):
        scene_change_frame.append(content_vals[idx][0].get_frames() )
    scene_change_frame.append(content_vals[-1][1].get_frames() )
    # return np.array(content_vals)
    return scene_change_frame

def get_content_val(video_path,stat_save_file):
    video = open_video(video_path)
    # video_manager = VideoManager([video_path])
    scene_manager = SceneManager(stats_manager=StatsManager())

    # Initialize ContentDetector with a save_images parameter set to False.
    content_detector = AdaptiveDetector()
    
    # Add the detector to the scene manager.
    scene_manager.add_detector(content_detector)
    scene_manager.detect_scenes(video=video)
    scene_manager.stats_manager.save_to_csv(csv_file=stat_save_file)
    df = pd.read_csv(stat_save_file)
    content_vals = df['content_val']
    # content_vals = scene_manager.get_con()
    return np.array(content_vals)
    # return scene_change_frame

def normalize_texts(l_string):
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
    
def filter_text(data):
    diffscore_thrashold = 0.9
    IoU_box_threshold = 0.7
    
    text_info = {}
    for idx in  data:
        obj_tmp = data.get(idx)
        
        for text_info_tmp in obj_tmp:
                
            text_info_key_tmp = list(text_info.keys())
            joined_text = "".join(text_info_tmp["text"])
            check_diffscore_text = np.array([similar_difflib(joined_text, str(text_idx)) for  text_idx in text_info.keys()])
            
            if np.any(check_diffscore_text >= diffscore_thrashold):# and np.any(check_IoU_box >= 0.7):   # Check text different score more than threshold 0.9 and IoU of text box  mor than 0.7
                # print(np.argwhere(check_diffscore_text>=0.9), check_diffscore_text)
                index_diffscore = np.argwhere(check_diffscore_text>=diffscore_thrashold)[0][0]
                text_key = text_info_key_tmp[index_diffscore]
                
                #Check box exist
                check_IoU_box = np.array([calculate_iou(text_info_tmp["box"],text_info[text_key][idx_tmp]["box"]) for  idx_tmp in range(len(text_info[text_key]))])
                
                if np.any(check_IoU_box >= IoU_box_threshold):
                    index_IoU_box  = np.argwhere(check_IoU_box>=IoU_box_threshold)[0][0]
                    text_info[text_key][index_IoU_box]["frame"].append(int(idx))
                else:
                    text_info[text_key].append({
                    "box": text_info_tmp["box"],
                    "frame":[int(idx)] })
            else: 
                text_info[joined_text] = [
                    {
                        "box": text_info_tmp["box"],
                        "frame":[int(idx)] }
                    ]
    return text_info

def find_font_path(title,path_saved_fonts_folder):
    
    font_folder_path = find_closest_font_name(path_saved_fonts_folder,title)
    # print("Font folder: ", font_folder_path)
    font_path = find_closest_font_name(os.path.join(path_saved_fonts_folder, font_folder_path),title,".otf")
    # print("Font file matching: ", font_path)
    return os.path.join(path_saved_fonts_folder,font_folder_path,font_path)

def detectTextOCR(cap , ocr,text_object, total_frame):
    # encode_video = ffmpeg_encoder(f"{path_folder}/{filename}_PaddledrawText.mp4",30, width,height,"5M")
    for _ in tqdm(range(total_frame), desc="Processing text"):
        ret, frame = cap.read()
        # if count <91:
        #     count += 1
        #     continue
        if not ret or frame is None:# or count ==10:
            break
        # try: 
        # frame = cv2.resize(frame,(540,960))
        #Detect by Paddle OCR
        img_tmp = "/tmp/Image_tmp.png"
        cv2.imwrite(img_tmp,frame)
        result = ocr.ocr(img_tmp, cls=True)
        # print(f"Frame num: {count} - {result}")
        image = Image.open(img_tmp).convert('RGB')
        if result != [None]:
            for idx in range(len(result)):
                res = result[idx]
                boxes = [line[0] for line in res]
                txts = [line[1][0] for line in res]
                scores = [line[1][1] for line in res]
                # for box in boxes:
                # # im_show = draw_ocr(image, boxes, txts, scores, font_path='/home/anlab/Downloads/fonts/Roboto.ttf')
                #     cv2.rectangle(frame,np.array(box[0]).astype(int),np.array(box[2]).astype(int),(0,255,0),2)
                normalized_texts = []
                for text_tmp in txts:
                
                    if float(scores[txts.index(text_tmp)]) >= 0.9:
                        box_index = txts.index(text_tmp)
                        normalized_text_tmp = normalize_texts(text_tmp)
                        # normalized_texts += normalized_text_tmp
                        normalized_texts.append({"text":normalized_text_tmp, "box":boxes[box_index]})
                        # cv2.rectangle(frame,np.array(boxes[box_index][0]).astype(int),np.array(boxes[box_index][2]).astype(int),(0,255,0),2)
                # print(f"Frame num: {count} - {txts} - {normalized_texts} - {scores}")
                text_object[count] = normalized_texts
            
        else:
            text_object[count] = []
        count +=1
    cap.release()
    # encode_video.stdin.flush()
    # encode_video.stdin.close()
    return text_object

def preprocessText(text_object, path_video_folder, filename, fps ):
    data = text_object
    text_info = filter_text(data)
    ## Preprocessing text info
        # print(text_info.keys())
        # if data[obj].text
    path_json = f"{path_video_folder}/{filename}_textframe_preprocessed.json"
    with open(path_json ,'w') as fp:
        json.dump(text_info, fp, indent=4)
        # text_info = json.load(fp)
        # break
    text_scene_duration_theshold = int(fps/2)
    # scene_change_frame = [0, 129, 222, 287, 376, 445]

    #Extend scene change timestamp to check text change corresponding
    extend_frame = 10
    # extended_scene_change_frame = []
    # # extended_scene_change_frame = [extended_scene_change_frame + list[]]
    # for timestamp in scene_change_frame:
    #     extended_scene_change_frame += list(range(np.max([1,timestamp-extend_frame]),np.min([total_frame,timestamp+extend_frame ])))
    # # print("extended_scene_change_frame: ", extended_scene_change_frame)
    # extended_scene_change_frame = set(extended_scene_change_frame)
    text_scene_final = []
    for idx in text_info:
        text_info_tmp = text_info.get(idx)
        # print(idx,text_info_tmp)
        for idx_tmp in range(len(text_info_tmp)):
            frame_tmp =set(text_info_tmp[idx_tmp]["frame"])
            if len(frame_tmp) > text_scene_duration_theshold :# and len(extended_scene_change_frame.intersection(frame_tmp) )>0:
                text_scene_final.append({
                    "text":idx,
                    "box":text_info_tmp[idx_tmp]["box"],
                    "start_frame":text_info_tmp[idx_tmp]["frame"][0],
                    "end_frame":text_info_tmp[idx_tmp]["frame"][-1],
                    "timestamp":[round(text_info_tmp[idx_tmp]["frame"][0]/fps,3),round(text_info_tmp[idx_tmp]["frame"][-1]/fps,3)]
                })
    print(text_scene_final)
    return  text_scene_final
    
def visualize(text_scene_final,video_path,path_video_folder,filename, total_frame,  fps, width,height ):
    # Visualize text box
    encode_video = ffmpeg_encoder(f"{path_video_folder}/{filename}_visual_text.mp4",fps, width,height,"5M")
    # fr_count = 1
    cap = cv2.VideoCapture(video_path)
    for fr_count in tqdm(range(1,total_frame+1), desc="Visualzing text box"):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        for text in text_scene_final:
            if fr_count >= text["start_frame"] and fr_count <= text["end_frame"]:
                # print(fr_count,text["start_frame"] , text["start_frame"] )
                cv2.rectangle(frame,np.array(text["box"][0]).astype(int),np.array(text["box"][2]).astype(int).astype(int),(0,255,0),2)
        cv2.putText(frame, f"Frame: {fr_count}", (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
        write_frame(frame,encode_video)
    cap.release()
    encode_video.stdin.flush()
    encode_video.stdin.close()

textchange_frame_min = 1

if __name__ == "__main__":
    ## Load model EasyOCR
    # reader = easyocr.Reader(['en'],gpu=False,show_log=False)
    # ## Load Paddler OCR
    # ocr = PaddleOCR(use_angle_cls=True, lang="en",show_log=False)  # need to run only once to download
    # tesseract_model = tesserocr.PyTessBaseAPI()
    # tesseract_model.SetPageSegMode(2)
    # text = reader.readtext(img)
    # print(text)

    path_folder = "/home/anlab/Downloads/advertisments_video/detect_text_area"
    path_font_folder ="/home/anlab/Downloads/advertisments_video/detect_text_area/font"
    # filename = "test.mp4"
    num_threads = 1
    padding_size_ratio =  0.1
    list_video = [
                "230954393378831.mp4",
                    # "781490520120333.mp4",
                "279573828566031.mp4"
                # "728635192798056.mp4"
                ]
    for filename in list_video:#os.listdir(path_folder):
        count = 1
        text_object = {}
        # filename = "781490520120333.mp4"
        if filename.lower().endswith(('.mp4')) :#and not filename.lower().endswith(+('230954393378831.mp4')):
            # print(filename)
            # filename = "test.mp4"
            filename = filename.split(".")[0]
            path_video_folder = f"{path_folder}/{filename}"
            path_video_scene = path_video_folder + "/scene"
            path_video_image = path_video_folder + "/image"
            try: 
                os.makedirs(path_video_folder)
                os.makedirs(path_video_scene)
                os.makedirs(path_video_image)
            except Exception as e:
                print("Error make folder: ", e)
                pass
            video_path = f"{path_folder}/{filename}.mp4"#"/home/anlab/Downloads/advertisments_video/test_April_03/shop/728635192798056.mp4"
            cap = cv2.VideoCapture(video_path)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
            text_scan_frame = int(fps*1.0)

            frame_temp = []
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_temp.append(frame)
            cap.release()

            # Detect text using PaddleOCR and normalized results
            # text_object = detectTextOCR(cap , ocr,text_object, total_frame)

            # path_json = f"{path_video_folder}/{filename}_textframe.json"
            # with open(path_json ,'r') as fp:
            #     # json.dump(text_object, fp, indent=4)
            #     data = json.load(fp)
            
            # # Preprocessing OCR results
            
            path_json = f"{path_video_folder}/{filename}_detected_font.json"
            with open(path_json ,'r') as fp:
                # json.dump(text_scene_final, fp, indent=4)
                font_detect = json.load(fp)
            # font_final = []
            path_json = f"{path_video_folder}/{filename}_detected_fontsize.json"
            with open(path_json ,'r') as fp:
                # json.dump(font_final, fp, indent=4)
                fontsize_detect = json.load(fp)
            text_obj = []
            font_detected = []
            font_render = []
            fontsize_detected = []
            fontsize_render = []
            print("len(font_detect): ", len(font_detect))
            for idx in range(len(font_detect)):
                
                #Get data font
                font_obj_tmp = font_detect[idx]
                fontsize_obj_tmp = fontsize_detect[idx]
                # print(font_obj_tmp)
                font_tmp = font_obj_tmp["font"]
                fontsize_tmp = fontsize_obj_tmp["font"]["pointsize"]
                if font_tmp == "No chars found":
                    print("No font info of text: ", font_obj_tmp["text"])
                    continue
                if fontsize_tmp < 5:
                    print("fontsize is 0 ", font_obj_tmp["text"])
                    fontsize_tmp = 20
                    
                ## Gen text from font and font_size
                frame_index = int( (font_obj_tmp["start_frame"] + font_obj_tmp["end_frame"])/2)
                ## Get box text and save to image
                topleft = np.array(font_obj_tmp["box"][0]).astype(int)
                bottomright = np.array(font_obj_tmp["box"][2]).astype(int)
                padding_X = int((bottomright[0] - topleft[0]) * padding_size_ratio)
                padding_Y = int((bottomright[1] - topleft[1]) * padding_size_ratio)
                
                padding_topleft = [max(0, topleft[0] - padding_X), max(0, topleft[1]- padding_Y)]
                padding_bottomright = [min(width, bottomright[0] + padding_X), min(height, bottomright[1] + padding_Y)]
                # frame_index = int( (text_obj_tmp["start_frame"] + text_obj_tmp["end_frame"])/2)
                # print(frame_index, padding_X, padding_Y)
                # print(topleft, bottomright)
                # print(padding_topleft[1],padding_bottomright[1],padding_topleft[0],padding_bottomright[0])
                text_img_crop = frame_temp[frame_index][padding_topleft[1]:padding_bottomright[1],padding_topleft[0]:padding_bottomright[0]]
                
                img_tmp = f'{path_video_image}/{font_obj_tmp["text"]}.png'
                img_tmp_gen_text = f'{path_video_image}/{font_obj_tmp["text"]}_text_gen.png'
                count = 0
                while True:
                    if os.path.isfile(img_tmp):
                        img_tmp = f'{path_video_image}/{font_obj_tmp["text"]}_{count}.png'
                        img_tmp_gen_text = f'{path_video_image}/{font_obj_tmp["text"]}_{count}_text_gen.png'
                        count +=1
                        continue
                    else:
                        break
                
                # img_ = cv2.imread(img_tmp)
                # cv2.imwrite(img_tmp,text_img_crop)

                # Find font path
                
                font_path_tmp = find_font_path(font_tmp[0]["title"], path_font_folder)
                # Initialize font - cal new withd, height for image
            
                # # print("font_path_tmp: ",font_path_tmp)
                # font = ImageFont.truetype(font_path_tmp, fontsize_tmp)  
                # h, w, _ = text_img_crop.shape
                
                # left, top, right, bottom = font.getbbox(font_obj_tmp["text"])
                # w = max(w, right-left)
                # h = max(h, bottom-top)
                # image = Image.new('RGB', (w,h+50),color='white')
                # draw = ImageDraw.Draw(image)
                # # drawing text size 
                # draw.text((10, 10), font_obj_tmp["text"], fill ="black",font = font, align ="left")  
                # image.save(img_tmp_gen_text)
                text_obj.append(font_obj_tmp["text"])
                font_detected.append(font_tmp[0]["title"])
                font_render.append(font_path_tmp.split("/")[-1])
                fontsize_detected.append(fontsize_obj_tmp["font"]["pointsize"])
                fontsize_render.append(fontsize_tmp)
            dp = pd.DataFrame(data={
                "text_obj: ": text_obj,
                "font_detected": font_detected,
                "font_render": font_render,
                "fontsize_detected": fontsize_detected,
                "fontsize_render": fontsize_render
            })
            # zipped_lists = zip(text_obj, font_detected,font_render,fontsize_detected,fontsize_render)
            # sorted_lists = sorted(zipped_lists)
            # text_obj, font_detected,font_render,fontsize_detected,fontsize_render = zip(*sorted_lists)
            dp.to_csv(f"{path_video_folder}/{filename}.csv")
            print(filename)
            print("text_obj: ", text_obj)
            print("font_detected: ", font_detected)
            print("font_render: ", font_render)
            print("fontsize_detected: ", fontsize_detected)
            print("fontsize_render: ", fontsize_render)
            # break       
                
                
                # break
                # _, buffer_bf = cv2.imencode(".jpg", img_)
                # encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
                # # URL of the API endpoint
                # url = 'https://www.whatfontis.com/api2/'

                # # Data for the form
                # form_data = {
                #     'urlimagebase64': encoded_image_bf, 
                #     'API_KEY': 'b747403f6ecb8fd7e703bb9cef496fb551bc74dc7b2e0190b1d8d04549ce4970',
                #     'IMAGEBASE64': '1', 
                #     'NOTTEXTBOXSDETECTION': '1',
                #     'urlimage': '',  
                #     'limit': '5'
                # }

                # # Make the POST request with the form data
                # response = requests.post(url, data=form_data)
                # print(response.text) # Print the response text
                

        # path_json = f"{path_video_folder}/{filename}_detected_fontsize.json"
        # with open(path_json ,'w') as fp:
        #     json.dump(font_final, fp, indent=4)
        # break
