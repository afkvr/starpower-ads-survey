import cv2
import os
import sys
from utils import client
import json
import base64
import argparse
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def detect_CTR(image):
    _, buffer_bf = cv2.imencode(".jpg", image)
    encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
    prompt = f"""There is an advertisement image. I want you to predict Click-through-rate of this image, just return only value and do not add any words in response."""
    promt_tmp =[
        {
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{encoded_image_bf}"
            }
        },
        {
            "type": "text",
            "text": prompt
        }
    ]
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": promt_tmp
            }
    ]
    # print(type(message))
    # Send the request to the OpenAI API
    response_string = client.chat.completions.create(model="gpt-4o",
                                                        messages=message,
                                                        max_tokens=300,
                                                        # n=1,
                                                        # stop=None,
                                                        # temperature=0
                                                        )
    # print(response_string)
    message_contents = response_string.choices[0].message.content
    message_contents = float(message_contents.replace("%",""))
    
    return message_contents

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default="")
    args = parser.parse_args()
    image_input = args.image
    if os.path.isfile(image_input):
        list_image = [image_input]
    else:
        list_image = [os.path.join(image_input, image)
                      for image in os.listdir(image_input)]
    num_threads = 1
    for filename in list_image:
        # count = 1
        # filename = "781490520120333.mp4"
        # text_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):  # and not os.p
            print(filename)
            file_basename = os.path.basename(filename).split('.')[0]
            metadata_path = f"/home/anlab/Downloads/allbirds_July172024_model_4M_21_XL/{file_basename}/{file_basename}_metadata.json"
            previous_image = cv2.imread(filename)
            _, buffer_bf = cv2.imencode(".jpg", previous_image)
            encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
            prompt = f"""There is an advertisement image. I want you to predict Click-through-rate of this image, just return only value and do not add any words in response."""
            promt_tmp =[
                {
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",
                    "content": promt_tmp
                    }
            ]
            # print(type(message))
            # Send the request to the OpenAI API
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=300,
                                                                # n=1,
                                                                # stop=None,
                                                                # temperature=0
                                                                )
            # print(response_string)
            message_contents = response_string.choices[0].message.content
            print(message_contents)
            # with open(metadata_path ,'r') as fp:
            #     final_metadata = json.load(fp)
                
            # final_metadata["Click-through-rate"] = message_contents.replace("%","")
            # # print(final_metadata)
            # with open(metadata_path ,'w') as fp:
            #     json.dump(final_metadata, fp, indent=4, cls=NpEncoder)
            # break
            # message_contents = message_contents.replace("```json", "")
            # message_contents = message_contents.replace("```", "")
            # message_data = json.loads(message_contents)