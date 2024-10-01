import io
import math
import openai
from PIL import Image, ImageDraw
import requests
from dataclasses import dataclass
import os
import sys
from openai import OpenAI
from dotenv import load_dotenv
import json
import numpy as np
load_dotenv()
import cv2
from difflib import SequenceMatcher

# Initialise Open AI client
client = OpenAI(api_key=os.environ["OpenAI_KEY"])


@dataclass
class TextBox:
    # (x, y) is the top left corner of a rectangle; the origin of the coordinate system is the top-left of the image.
    # x denotes the vertical axis, y denotes the horizontal axis (to match the traditional indexing in a matrix).
    x: int
    y: int
    h: int
    w: int
    text: str = None

class Inpainter:
    """Interface for in-painting models."""

    # TODO(julia): Run some experiments to determine the best prompt.

    # DEFAULT_PROMPT = "plain background and keep quality of image"
    DEFAULT_PROMPT = "Remove, fill texts with surround color for each transparent regions in mask and keep quality of image"
    def inpaint(self, image: str, text_boxes, prompt: str):
        pass


class DalleInpainter(Inpainter):
    """In-painting model that calls the DALL-E API."""

    def __init__(self):
        print()

    @staticmethod
    def _make_mask(text_boxes, height: int, width: int) -> bytes:
        """Returns an .png where the text boxes are transparent."""
        mask = Image.new("RGBA", (width, height), (0, 0, 0, 1))  # fully opaque
        mask_draw = ImageDraw.Draw(mask)
        for text_box in text_boxes:
            mask_draw.rectangle(
                xy=(
                    text_box.x,
                    text_box.y,
                    text_box.x + text_box.h,
                    text_box.y + text_box.w,
                ),
                fill=(0, 0, 0, 0),
            )  # fully transparent
        # Convert mask to bytes.
        # bytes_arr = io.BytesIO()
        # mask.save(bytes_arr, format="PNG")
        # return bytes_arr.getvalue()
        return mask

    @staticmethod
    def resize_to_square(image, ismask=False):
        # Get the original image size
        original_width, original_height = image.size
        new_size = max(original_width, original_height)
        new_image = (
            Image.new("RGB", (new_size, new_size), (255, 255, 255))
            if not ismask
            else Image.new("RGBA", (new_size, new_size), (0, 0, 0, 1))
        )
        paste_x = (new_size - original_width) // 2
        paste_y = (new_size - original_height) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image, paste_x, paste_y, original_width, original_height, new_size

    @staticmethod
    def crop_back_image(
        square_image, paste_x, paste_y, original_width, original_height, new_size
    ):
        square_image = square_image.resize(
            (new_size, new_size), Image.ANTIALIAS
        )
        # Calculate the bounding box of the original image
        left = paste_x
        upper = paste_y
        right = paste_x + original_width
        lower = paste_y + original_height
        cropped_image = square_image.crop((left, upper, right, lower))

        return cropped_image

    def inpaint(self, image: str, text_boxes, prompt: str):
        # image = Image.open(in_image_path)  # open the image to inspect its size
        print("Inpainting with DALL-E 2!!!!")
        square_image, paste_x, paste_y, original_width, original_height, new_size = (
            self.resize_to_square(image)
        )
        mask = self._make_mask(text_boxes, image.height, image.width)
        mask.save("/tmp/dallE_mask.png")
        resized_mask, _, _, _, _,_ = self.resize_to_square(mask, True)
        
        buf = io.BytesIO()
        square_image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        buf_mask = io.BytesIO()
        resized_mask.save(buf_mask, format="PNG")
        byte_mask = buf_mask.getvalue()

        response = client.images.edit(
            model="dall-e-2",
            image=byte_im,
            mask=byte_mask,
            prompt=prompt,
            n=1,
            size="1024x1024",  # f"{image.height}x{image.width}",
        )
        print(response)
        url = response.data[0].url
        out_image_data = requests.get(url).content
        out_image = Image.open(io.BytesIO(out_image_data))
        out_image = self.crop_back_image(
            out_image, paste_x, paste_y, original_width, original_height, new_size
        )
        # out_image.save(out_image_path)
        return out_image

class Point2D:
    x=0
    y=0
    def __init__(self,point):
        self.x=point[0]
        self.y=point[1]

class Vector2D:
    x=0
    y=0
    def __init__(self,x,y):
        self.x=x
        self.y=y

def my_scalar_product(u,v):
    # TODO
    return (u.x * v.x) + (u.y * v.y)

def my_norm(v):
    # TODO
    return math.sqrt(v.x**2 + v.y**2)

def angle(A,B,C):
    # TODO
    u = Vector2D(A.x-B.x, A.y-B.y)
    v =  Vector2D(C.x-B.x, C.y-B.y)

    scalarProduct = my_scalar_product(u, v)
    tmp = my_norm(u) * my_norm(v)
    return math.acos(scalarProduct / tmp)

def get_polygon_angle(list_polygon):
    list_polygon_tmp = [Point2D(np.array(tmp, dtype=np.int32)) for tmp in list_polygon]
    center = np.array([(list_polygon_tmp[0].x+list_polygon_tmp[2].x)/2, (list_polygon_tmp[0].y+list_polygon_tmp[2].y)/2], dtype=float)
    if list_polygon_tmp[0].y > list_polygon_tmp[1].y:
        point_3rd= Point2D([list_polygon[1][0], list_polygon[0][1]])
        rotarted_angle = -1 * angle(point_3rd, list_polygon_tmp[0], list_polygon_tmp[1])
    else:
        point_3rd= Point2D([list_polygon[1][0], list_polygon[0][1]])
        rotarted_angle = angle(list_polygon_tmp[1], list_polygon_tmp[0], point_3rd)
    
    rotarted_angle = float(rotarted_angle/(2*math.pi))*360
    
    # print("Angle: ", rotarted_angle, center)
    
    return rotarted_angle, center

def check_image_language(image_path):
    prompt = f"""There is a list of language and its symbol here: {json.dumps(script_language_map_easyOCR)}. Given a image below, I need you find all languages were used in image and return only a list of language symbols without adding any words. Example ["en", "japan"]. If image do not have any language, return 'Not detect'"""
    # print(type(prompt))
    promt_tmp = []
    previous_image = cv2.imread(image_path)
                
    _, buffer_bf = cv2.imencode(".png", previous_image)
    encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    promt_tmp.append({
                        "type": "text",
                        "text": prompt
                    })
    promt_tmp.append({
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                }
            })     
    # print(type(encoded_image_bf))   
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": promt_tmp
            },
    ]
    # print(type(promt_tmp))
    # # # Send the request to the OpenAI API
    try: 
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
    except Exception as e:
        time.sleep(1)
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
    print("Check language", response_string.choices[0].message.content)
    try: 
        if response_string.choices[0].message.content == 'Not detect':
            message_contents = ["en"]
        else:
            message_contents = extract_list_from_string(response_string.choices[0].message.content)
    except Exception as e:
        message_contents = ["en"]
        pass
    # print(message_contents)
    # print(type(message_contents))
    return message_contents

def similar_difflib(a, b):
    return SequenceMatcher(None, a, b).ratio()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def combine_pairs(pairs):
    uf = UnionFind()
    # Add all pairs to the union-find structure
    for pair in pairs:
        for num in pair:
            uf.add(num)
        first = pair[0]
        for num in pair[1:]:
            uf.union(first, num)
    # Collect components
    components = defaultdict(list)
    for num in uf.parent:
        root = uf.find(num)
        components[root].append(num)
    # Convert components to list of lists and sort each list
    return [sorted(component) for component in components.values()]

def check_image(image,alpha_color=(255, 255, 255)):
    if image.shape[2] ==4:
        # image = image[:,:,:3] 
        B, G, R, A = cv2.split(image)
        alpha = A / 255
        R = (alpha_color[0] * (1 - alpha) + R * alpha).astype(np.uint8)
        G = (alpha_color[1] * (1 - alpha) + G * alpha).astype(np.uint8)
        B = (alpha_color[2] * (1 - alpha) + B * alpha).astype(np.uint8)

        image = cv2.merge((B, G, R))
    if isinstance(image, np.ndarray) and len(image.shape) == 2:
        print("Image is gray! Try to convert BRG")
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def resize_and_compress_image(image, max_size=(100, 100), quality=95):
    # Read the image
    # image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Only resize if the image is larger than the max_size
    if width < max_size[0] or height < max_size[1]:
        # Calculate the scaling factor while maintaining aspect ratio
        scaling_factor = max(max_size[0] / width, max_size[1] / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        print("New size: ", new_size)
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized_image = image
    
    # Compress the image
    _, buffer_bf = cv2.imencode(".jpg", resized_image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    
    return buffer_bf