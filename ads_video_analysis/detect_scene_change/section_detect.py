import cv2, base64, time
import os
import sys
import numpy as np
import ffmpeg
import subprocess
from PIL import Image
from scenedetect import SceneManager,StatsManager,  open_video, ContentDetector, AdaptiveDetector, SceneDetector
import time
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
# import re
from tensorflow.keras.models import load_model
# import json

PATH = os.path.dirname(os.path.abspath(__file__))

def similar_difflib(a, b):
    return SequenceMatcher(None, a, b).ratio()

def sig(x):
 return 1/(1 + np.exp(-x))

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

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


def find_scenes_content(video_path, threshold=27.0):
    video = open_video(video_path)
    scene_manager = SceneDetector()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold,min_scene_len=20))
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    return scene_manager.get_metrics()

def find_scenes_adaptive(video_path, min_content_val=15.0):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        AdaptiveDetector(min_content_val=min_content_val))
    # Detect all scenes in video from current position to end.
    scene_manager.detect_scenes(video)
    # `get_scene_list` returns a list of start/end timecode pairs
    # for each scene that was found.
    return scene_manager.get_scene_list()

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

    return np.array(content_vals)

def detect_scene_final(total_frame, pyscene_score_diff,difflib_scores,fps):
    pyscene_scan_frame = fps*0.1
    dict_tmp = {    
            "Frame_number":list(range(2,total_frame+1)),
            "pyscene_score":pyscene_score_diff,
            "text_score": difflib_scores
                }
    df = pd.DataFrame(dict_tmp)
    # df.to_csv(f'{path_folder}/{filename}_scores.csv',sep=';', float_format='%.5f')
    
    df_test = df
    model = load_model(f'{PATH}/models/scene_change_normalized_4.h5', compile=False)
    predictions = model.predict([df_test["pyscene_score"],df_test["text_score"]])
    output_neural_A = predictions[0] # Output of Neural A (pyscene)
    output_neural_B = predictions[1]  # Output of Neural B (text)
    output_neural_C = predictions[2] # Output of Neural C (combined
    dict_tmp_test = pd.DataFrame({
            'frame_number': df_test["Frame_number"],
            'pyscene_score': df_test["pyscene_score"],
            'text_score': df_test["text_score"],
            'output_neural_A':output_neural_A.reshape(-1),
            'output_neural_B':output_neural_B.reshape(-1),
            'output_neural_C':output_neural_C.reshape(-1),
            'output_neural_A_normalized':np.array([sig(idx) for idx in predictions[0]  ]).reshape(-1),
            'output_neural_B_normalized':np.array([sig(idx) for idx in predictions[1]  ] ).reshape(-1),
            'output_neural_C_normalized':np.array([sig(idx) for idx in predictions[2]  ]  ).reshape(-1),
            "predict":output_neural_C.round().reshape(-1)
        })
    dict_tmp_test.to_csv(f'/tmp/predict.csv',sep=';', float_format='%.5f')
    scene_change_timestamp = np.array(list(filter(lambda x: dict_tmp_test["predict"][x] == 1.0, range(len(dict_tmp_test["predict"])))))+1
    # Normalize timestamp
    scene_timestamp = [0]
    transition_timestmap = []
    neighbor_list = []
    for idx in scene_change_timestamp:
        if idx - scene_timestamp[-1] >= pyscene_scan_frame:
            if len(neighbor_list) >0:
                transition_timestmap.append(scene_timestamp[-1])
                neighbor_list = []
            scene_timestamp.append(idx)
        else:
            neighbor_list.append(idx)
    dict_scene_name= [f"""section-{idx}""" for idx in range(1,len(scene_timestamp)+1)]
    dict_start_frame = scene_timestamp
    dict_end_frame = np.concatenate([np.array(scene_timestamp[1:]),[total_frame]]).tolist()
    dict_start_time = [round(i/fps,3) for i in dict_start_frame]
    dict_end_time = [round(i/fps,3) for i in dict_end_frame]
    
    # print(scene_timestamp, transition_timestmap)
# # Save to CSV files
    dict_tmp = {"scene-name":dict_scene_name,
                "start_frame":dict_start_frame,
                "end_frame":dict_end_frame,
                "start_time":dict_start_time,
                "end_time":dict_end_time}
    # print(dict_tmp)
    df = pd.DataFrame(dict_tmp)
    return df
        

def find_diffscore_text(total_frame, text_object):
    data = text_object
    difflib_scores= []
    for idx in range(2,total_frame):
        previous_str = "".join(data[idx-1]) if data[idx-1] != [] else ""
        current_str = "".join(data[idx]) if data[idx] != [] else ""
        difflib_score = similar_difflib(previous_str, current_str)
        difflib_scores.append(difflib_score)
    
    return difflib_scores

count = 0
previous_frame = None
threshold_array = 3.2
duration_transition = 1
blur_threshold = 10
ssim_min_diff = 0.1
blur_min_diff = 5
blur_frame_min = 4
transition_frame_min = 4



list_scene_final = []
if __name__ == "__main__":
    sig(1)
    # print(os.path.dirname(os.path.abspath(__file__)))
    # print(len(PATH))
    # break
    detect_scene_final(6,list(range(5)),list(range(5)),5)
