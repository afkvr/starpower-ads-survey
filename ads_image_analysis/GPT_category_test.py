from scipy.cluster.hierarchy import linkage, fcluster
from shapely.geometry import Polygon
import numpy as np
import argparse
import json
import os
from openai import OpenAI
from dotenv import load_dotenv
import cv2
import base64
import sys
from utils import client
import time 


def calculate_distances(polygon1, polygon2):
    # Create shapely polygon objects
    poly1 = Polygon(polygon1)
    poly2 = Polygon(polygon2)
    
    # Calculate the bounds of the polygons
    minx1, miny1, maxx1, maxy1 = poly1.bounds
    minx2, miny2, maxx2, maxy2 = poly2.bounds

    # Calculate the vertical distance
    if maxy1 < miny2:
        vertical_distance = miny2 - maxy1
    elif maxy2 < miny1:
        vertical_distance = miny1 - maxy2
    else:
        vertical_distance = 0  # Polygons overlap vertically

    # Calculate the horizontal distance
    if maxx1 < minx2:
        horizontal_distance = minx2 - maxx1
    elif maxx2 < minx1:
        horizontal_distance = minx1 - maxx2
    else:
        horizontal_distance = 0  # Polygons overlap horizontally

    return vertical_distance, horizontal_distance

def cluster_polygons(polygons, image_width, image_height):
    num_polygons = len(polygons)
    distances = np.zeros((num_polygons, num_polygons))

    # Calculate distance matrix
    for i in range(num_polygons):
        for j in range(i + 1, num_polygons):
            vert_dist, horiz_dist = calculate_distances(polygons[i], polygons[j])
            distances[i, j] = max(vert_dist / image_height, horiz_dist / image_width)
            distances[j, i] = distances[i, j]

    # Apply hierarchical clustering
    Z = linkage(distances, 'single')

    # Form clusters with a threshold of 20% of the image dimensions
    threshold = 0.2
    clusters = fcluster(Z, threshold, criterion='distance')

    return clusters

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def category_object_GPT(inpainted_text_path,data_GSA_filter):
    """A"""
    # prompt = f"""I give you a list of {len(list_label)} images and 5 category: ["Design element", "Text", "Background image", "Product image", "Logo"]. 
    # Predict which category for each image in 5 given category. Return only list category for {len(list_label)} corresponding images without add any words. Example output: ["Text","Logo"]"""
    
    # prompt = f"""I give you a list of {len(data_GSA_filter)} images and 5 category: ["Design element", "Text", "Background image", "Product image", "Logo"]. 
    # Predict which category for each image in 5 given category with some condition: 
    #  1 - The background image should occupy at least 50% size of the whole image.
    #  2 - A logo may be a stylish text. Expect only 1 logo in an image.
    # # Return only list category for {len(data_GSA_filter)} corresponding images without add any words. Example output: ["Text","Logo"]"""
    
    prompt = f"""I give you 2 images and 5 category: ["Design element", "Text", "Background image", "Product image", "Logo"]. The first image is a full adsvertisement image, the second image is cropped image of object. 
    Predict which category for second image in 5 given category with some condition: 
     1 - The background image should occupy at least 50% size of the whole image.
     2 - A logo may be a stylish text or stylist shape.
    # Return only 1 category for second images without add any words. Example output: 'Text'"""
    
    # Convert ads image to 
    inpainted_text_img = cv2.imread(inpainted_text_path)
    _, buffer_bf = cv2.imencode(".jpg", inpainted_text_img)
    ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
    # print(data_GSA_filter)
    for idx in range(len(data_GSA_filter)):
        prompt_tmp =[]
        box = data_GSA_filter[idx]["box"]
        # print(idx, box)
        crop_img = inpainted_text_img[box[1]:box[3],
                                      box[0]:box[2]]
        _, buffer_bf = cv2.imencode(".jpg", crop_img)
        encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{ad_img_bf}"
                        }
                    })
        
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                        }
                    })
        prompt_tmp.append({
                "type": "text",
                "text": prompt
            })
        # URL = "https://www.allbirds.com/products/mens-wool-runners-natural-white"
        # prompt_tmp.append({
        #         "type": "text",
        #         "text": f"Here is a product URL for you to improve category results more accurate: {URL}."
        #     })
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": prompt_tmp
                }
        ]
        ## Send the request to the OpenAI API
        try: 
            response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
        except Exception as e:
            time.sleep(1)
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=600)
        # print(response_string)
        message_contents = response_string.choices[0].message.content
        print("category_object_GPT: ",message_contents, "- Position: ", box)
        data_GSA_filter[idx]["category"] = message_contents
        # print(key , " => ", message_contents)
    return data_GSA_filter

def compare_inpaint(dalle_inpaint, lama_inpaint, firefly_inpaint,list_coors, list_name):
    # "Display the cropped images using plt.subplots.
    # After displaying the cropped images, "
    # prompt = f"""Hey ChatGPT - I am going to be giving you three images with name is {list_name} that have used AI image models to fill in a part of the image. The changed areas in each image are in list with format [topleft_x,topleft_y,bottom_right_x, bottomright_y] here: {list_coors}. Examine all three images, and then I want you to describe each of the changed portions of each image in one sentence each. Verify that your descriptions are for the correct image name. Then on a new line tell me which of the three images is filled in best - we want a plain image with no text (ignore text mark "Adobe Firefly" in image name "_firefly.png"). Use the file name for that image.
    # If all of them have text in the changed area, also mention that"""
    
    prompt = f"""Hey ChatGPT - I am going to be giving you three images with name is {list_name} that have used AI image models to fill in a part of the image. The changed areas in each image are in list with format [topleft_x,topleft_y,bottom_right_x, bottomright_y] here: {list_coors}.Return just only Ordinal of best image is 0, 1 or 2 and not add any words or warning. Example: '1' or '2'""" # Examine all three images and 
    
    prompt_tmp = []
    _, buffer_bf = cv2.imencode(".png", dalle_inpaint)
    dalle_inpaint_bf = base64.b64encode(buffer_bf).decode("utf-8")
    _, buffer_bf = cv2.imencode(".png", lama_inpaint)
    lama_inpaint_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{dalle_inpaint_bf}"
                        }
                    })
    prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{lama_inpaint_bf}"
                        }
                    })
    
    if firefly_inpaint is not None:
        _, buffer_bf = cv2.imencode(".png", firefly_inpaint)
        firefly_inpaint_bf = base64.b64encode(buffer_bf).decode("utf-8")
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{firefly_inpaint_bf}"
                        }
                    })
    prompt_tmp.append({
                "type": "text",
                "text": prompt
            })
    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": prompt_tmp
                }
        ]
    ## Send the request to the OpenAI API
    try: 
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
    except Exception as e:
        time.sleep(1)
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=600)
    print("compare_inpaint: ",response_string)
    message_contents = response_string.choices[0].message.content
    
    # print(message_contents)
    return message_contents

def category_text_logo_GPT_2(image, image_metadata):
    #get List elements have category "Text"/ "Logo"
    list_elements_text = {}
    for i, element in enumerate(image_metadata["Elements"]):
        if element["category"].lower() == "text" or "logo" in element["category"].lower():
            list_elements_text[str(i)] = element
    
    _, buffer_bf = cv2.imencode(".jpg", image)
    ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    prompt = f"""I'm going to send you some images, the first one is a full adsvertisement image, and then a list  cropped images of {len(list_elements_text)} element I want you to identify.
    Given the images below, can you tell me if the cropped image contains a logo, wordmark logo, or just text. I expected image has maximum 1 logo, so if there are more than 1 crop image was detected as 'logo', try to find best results. Give me a list has {len(list_elements_text)} results for each  crop image without add any words. Example output: ["text","logo"]"""
    prompt_tmp =[{
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{ad_img_bf}"
                        }
                    }
                ]
    h, w = np.array(image.shape[:2]).astype(int)
    list_box_tmp = []
    for tmp in list_elements_text.keys():
        # print(list_elements_text[tmp])
        if  "location-objective" in  list_elements_text[tmp].keys(): #list_elements_text[tmp]["category"].lower() == "text" and
            
            box = list_elements_text[tmp]["location-objective"]
            box = [np.array(tmp,dtype=float)*[w,h] for tmp in box]
            rect1 = cv2.boundingRect(np.array(box, dtype=np.int32))
            x,y,w_,h_ = rect1
            # boxes_tmp = [[x,y],[x+w_,y],[x+w_,y+h_],[x,y+h_]]
            topleft = [x,y]
            bottom_right = [x+w_,y+h_]
            # topleft = np.array([box[0][0] * image.shape[1],box[0][1] * image.shape[0]], dtype=np.int)# *[int(image.shape[1]), int[image.shape[0]]]
            # bottom_right = np.array([box[2][0] * image.shape[1],box[2][1] * image.shape[0]], dtype=np.int)# *[int(image.shape[1]), int[image.shape[0]]]
        else:
            box = list_elements_text[tmp]["box"]
            topleft = [box[0],box[1]]
            bottom_right = [box[2],box[3]]
        list_box_tmp.append(box)
    #     crop_img = image[topleft[1]:bottom_right[1],topleft[0]:bottom_right[0]]
    #     _, buffer_bf = cv2.imencode(".jpg", crop_img)
    #     encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
    #     prompt_tmp.append({
    #                     "type": "image_url",
    #                     "image_url": {
    #                     "url": f"data:image/jpeg;base64,{encoded_image_bf}"
    #                     }
    #                 })
    
    # prompt_tmp.append({
    #         "type": "text",
    #         "text": prompt
    #     })
    # # URL = "https://www.allbirds.com/products/mens-wool-runners-natural-white"
    # # prompt_tmp.append({
    # #         "type": "text",
    # #         "text": f"Here is a product URL for you to improve category results more accurate: {URL}."
    # #     })
    # message = [
    #     {"role": "system", "content": "You are a helpful assistant."},
    #     {"role": "user",
    #         "content": prompt_tmp
    #         }
    # ]
    # ## Send the request to the OpenAI API (Retry 2 times)
    # try: 
    #     response_string = client.chat.completions.create(model="gpt-4o",
    #                                                         messages=message,
    #                                                         max_tokens=1000)
    # except Exception as e:
    #     print(e)
    #     time.sleep(1)
    #     response_string = client.chat.completions.create(model="gpt-4o",
    #                                                         messages=message,
    #                                                         max_tokens=1000)
    
    # # print("category_text_logo_GPT: ",response_string)
    # message_contents = response_string.choices[0].message.content
    # try:
    #     message_contents = json.loads(message_contents)
    # except Exception as e:
    #     return image_metadata
    print("There are more than 1 detected logo in image => try GPT label text&logo second time: ")
    count = 0
    # print(image_metadata["Elements"])
    for i, element in enumerate(image_metadata["Elements"]):
        if str(i) in list_elements_text.keys():
            # print(image_metadata["Elements"][i])
            if "text-content" in image_metadata["Elements"][i].keys():
                print(image_metadata["Elements"][i]["text-content"], " => ", image_metadata["Elements"][i]["category"], " - Location: ", list_box_tmp[count]) 
            else:
                print("Object => ", image_metadata["Elements"][i]["category"], " - Location: ", list_box_tmp[count])
            # image_metadata["Elements"][i]["category"] = message_contents[count]
            count +=1
    
    return image_metadata

def category_text_logo_GPT_1(image, image_metadata):
    #get List elements have category "Text"/ "Logo"
    list_elements_text = {}
    for i, element in enumerate(image_metadata["Elements"]):
        if element["category"].lower() == "text" or "logo" in element["category"].lower():
            list_elements_text[str(i)] = element
    
    _, buffer_bf = cv2.imencode(".jpg", image)
    ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    prompt = f"""I'm going to send you some images, the first one is a full adsvertisement image, and then a list  cropped images of {len(list_elements_text)} element I want you to identify.
    Given the images below, can you tell me if the cropped image contains a logo, wordmark logo, or just text. I expected image has maximum 1 logo, so if there are more than 1 crop image was detected as 'logo', try to find best results. Give me a list has {len(list_elements_text)} results for each  crop image without add any words. Example output: ["text","logo"]"""
    prompt_tmp =[{
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{ad_img_bf}"
                        }
                    }
                ]
    h, w = np.array(image.shape[:2]).astype(int)
    list_box_tmp = []
    for tmp in list_elements_text.keys():
        # print(list_elements_text[tmp])
        if list_elements_text[tmp]["category"].lower() == "text" and "location-objective" in  list_elements_text[tmp].keys():
            
            box = list_elements_text[tmp]["location-objective"]
            box = [np.array(tmp,dtype=float)*[w,h] for tmp in box]
            rect1 = cv2.boundingRect(np.array(box, dtype=np.int32))
            x,y,w_,h_ = rect1
            # boxes_tmp = [[x,y],[x+w_,y],[x+w_,y+h_],[x,y+h_]]
            topleft = [x,y]
            bottom_right = [x+w_,y+h_]
            # topleft = np.array([box[0][0] * image.shape[1],box[0][1] * image.shape[0]], dtype=np.int)# *[int(image.shape[1]), int[image.shape[0]]]
            # bottom_right = np.array([box[2][0] * image.shape[1],box[2][1] * image.shape[0]], dtype=np.int)# *[int(image.shape[1]), int[image.shape[0]]]
        else:
            box = list_elements_text[tmp]["box"]
            topleft = [box[0],box[1]]
            bottom_right = [box[2],box[3]]
        list_box_tmp.append(box)
        crop_img = image[topleft[1]:bottom_right[1],topleft[0]:bottom_right[0]]
        _, buffer_bf = cv2.imencode(".jpg", crop_img)
        encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                        }
                    })
    
    prompt_tmp.append({
            "type": "text",
            "text": prompt
        })
    # URL = "https://www.allbirds.com/products/mens-wool-runners-natural-white"
    # prompt_tmp.append({
    #         "type": "text",
    #         "text": f"Here is a product URL for you to improve category results more accurate: {URL}."
    #     })
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": prompt_tmp
            }
    ]
    ## Send the request to the OpenAI API (Retry 2 times)
    try: 
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=1000)
    except Exception as e:
        print(e)
        time.sleep(1)
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=1000)
    
    # print("category_text_logo_GPT: ",response_string)
    message_contents = response_string.choices[0].message.content
    try:
        message_contents = json.loads(message_contents)
    except Exception as e:
        return image_metadata
    print("There are more than 1 detected logo in image => try GPT label text&logo second time: ")
    count = 0
    
    for i, element in enumerate(image_metadata["Elements"]):
        if str(i) in list_elements_text.keys():
            if "text-content" in image_metadata["Elements"][i].keys():
                print(image_metadata["Elements"][i]["text-content"], " => ", message_contents[count], " - Location: ", list_box_tmp[count]) 
            else:
                print("Object => ", message_contents[count], " - Location: ", list_box_tmp[count])
            image_metadata["Elements"][i]["category"] = message_contents[count]
            count +=1
    
    return image_metadata

def category_text_logo_GPT(image, image_metadata):
    #get List elements have category "Text"/ "Logo"
    list_elements_text = {}
    for i, element in enumerate(image_metadata["Elements"]):
        # if element["category"].lower() == "text" or "logo" in element["category"].lower():
            list_elements_text[str(i)] = element
    
    _, buffer_bf = cv2.imencode(".jpg", image)
    ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    #  prompt = f"""I'm going to send you some images, the first one is a full adsvertisement image, and then a list  cropped images of {len(list_elements_text)} element I want you to identify.
    #Given the images below, can you tell me if the cropped image contains a logo, wordmark logo, or just text. I expected image has maximum 1 logo, so if there are more than 1 crop image was detected as 'logo', try to find best results. Give me a list has {len(list_elements_text)} results for each  crop image without add any words. Example output: ["text","logo"]"""
    prompt = f"""I'm going to send you two images, one of a full images adsvertisement, and then a cropped image of the element I want you to identify.
    Given the images below, can you tell me if the cropped image contains a logo, wordmark logo, or just text. Give me the justification and then on a new line, just the word "text", "wordmark logo" or "logo" """
    h, w = np.array(image.shape[:2]).astype(int)
    list_box_tmp = []
    print("GPT label text & logo first time: ")
    # print(list_elements_text)
    count_logo = 0
    list_crop = {}
    # list_category = []
    for idx, tmp in enumerate(list(list_elements_text.keys())):
        # print(list_elements_text[tmp])
        # if list_elements_text[tmp]["category"].lower() == "text" and "location-objective" in  list_elements_text[tmp].keys():
            
        box = list_elements_text[tmp]["location-objective"]
        box = [np.array(tmp,dtype=float)*[w,h] for tmp in box]
        rect1 = cv2.boundingRect(np.array(box, dtype=np.int32))
        x,y,w_,h_ = rect1
        # boxes_tmp = [[x,y],[x+w_,y],[x+w_,y+h_],[x,y+h_]]
        topleft = [x,y]
        bottom_right = [x+w_,y+h_]
            # topleft = np.array([box[0][0] * image.shape[1],box[0][1] * image.shape[0]], dtype=np.int)# *[int(image.shape[1]), int[image.shape[0]]]
            # bottom_right = np.array([box[2][0] * image.shape[1],box[2][1] * image.shape[0]], dtype=np.int)# *[int(image.shape[1]), int[image.shape[0]]]
        # else:
        #     box = list_elements_text[tmp]["box"]
        #     topleft = [box[0],box[1]]
        #     bottom_right = [box[2],box[3]]
        
        prompt_tmp =[{
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{ad_img_bf}"
                        }
                    }
                ]            
        # print(box)
        crop_img = image[topleft[1]:bottom_right[1],topleft[0]:bottom_right[0]]
        # list_crop[str(idx)] = crop_img
        _, buffer_bf = cv2.imencode(".jpg", crop_img)
        encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                        }
                    })
    
        prompt_tmp.append({
                "type": "text",
                "text": prompt
            })
        # URL = "https://www.allbirds.com/products/mens-wool-runners-natural-white"
        # prompt_tmp.append({
        #         "type": "text",
        #         "text": f"Here is a product URL for you to improve category results more accurate: {URL}."
        #     })
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": prompt_tmp
                }
        ]
        ## Send the request to the OpenAI API (Retry 2 times)
        try: 
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=1000)
        except Exception as e:
            print(e)
            time.sleep(1)
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=1000)
        
        # print(response_string)
        
        message_contents = response_string.choices[0].message.content
        # print(message_contents)
        # try:
        #     message_contents = json.loads(message_contents)
        # except Exception as e:
        #     return image_metadata
        if "text-content" in list_elements_text[tmp]:
            print(list_elements_text[tmp]["text-content"], f" - Location: {box} =>")
            print("GPT response: ", "**", message_contents,"**")
            print("")
        else:
            print(f"Object - Location: {box} =>")
            print("GPT response: ", "**", message_contents,"**")
            print("")
        
        if message_contents.split("\n")[-1] == "logo":
            count_logo +=1
            list_crop[str(idx)] = crop_img
        
        # print("GPT label text: ")
        
        # ptin,message_contents)
        # count = 0
        
        # for i, element in enumerate(image_metadata["Elements"]):
        #     if str(i) in list_elements_text.keys():
        image_metadata["Elements"][idx]["category"] = message_contents.split("\n")[-1]
        #         count +=1
        
        # return image_metadata
    return count_logo, list_crop

def category_group_GPT(image, image_metadata, image_name, list_polygon_box):
    #get List elements have category "Text"/ "Logo"
    list_elements_text = {}
    # for i, element in image_metadata["Group"].keys():
    #     if element["category"].lower() == "text" or "logo" in element["category"].lower():
    #         list_elements_text[str(i)] = element
    
    _, buffer_bf = cv2.imencode(".jpg", image)
    ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    #  prompt = f"""I'm going to send you some images, the first one is a full adsvertisement image, and then a list  cropped images of {len(list_elements_text)} element I want you to identify.
    #Given the images below, can you tell me if the cropped image contains a logo, wordmark logo, or just text. I expected image has maximum 1 logo, so if there are more than 1 crop image was detected as 'logo', try to find best results. Give me a list has {len(list_elements_text)} results for each  crop image without add any words. Example output: ["text","logo"]"""
    prompt = f"""I'm going to send you two images, one of a full images adsvertisement, and then a cropped image of the element I want you to identify.
    Given the images below, can you tell me if the cropped image contains a logo, wordmark logo, or just text. Give me the justification and then on a new line, just the word "text", "wordmark logo" or "logo" """
    h, w = np.array(image.shape[:2]).astype(int)
    list_box_tmp = []
    print("GPT label text & logo first time: ")
    # print(list_elements_text)
    count_logo = 0
    list_key = list(image_metadata["Group"].keys())
    for i,tmp in enumerate(list_key):#list_elements_text.keys():
        # print(list_elements_text[tmp])
        group_tmp = image_metadata["Group"][tmp]
        topleft, bottom_right = list_polygon_box[str(i)]
        
        prompt_tmp =[{
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{ad_img_bf}"
                        }
                    }
                ]            
        # print(box)
        crop_img = image[topleft[1]:bottom_right[1],topleft[0]:bottom_right[0]]
        _, buffer_bf = cv2.imencode(".jpg", crop_img)
        encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                        }
                    })
    
        prompt_tmp.append({
                "type": "text",
                "text": prompt
            })
        message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
                "content": prompt_tmp
                }
        ]
        ## Send the request to the OpenAI API (Retry 2 times)
        try: 
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=1000)
        except Exception as e:
            print(e)
            time.sleep(1)
            response_string = client.chat.completions.create(model="gpt-4o",
                                                                messages=message,
                                                                max_tokens=1000)
        message_contents = response_string.choices[0].message.content
        print(f"Group {tmp} =>")
        print("GPT response: ", "**", message_contents,"**")
        print("")
        
        if "logo" in message_contents.split("\n")[-1]:
            count_logo +=1
    return count_logo

def category_logo(image, list_crop, image_metadata):
    _, buffer_bf = cv2.imencode(".jpg", image)
    ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
    
    prompt = f"""I'm going to send you some images, one of a full images adsvertisement, and then a list {len(list_crop)} cropped image of the element I want you to identify.
    Given the images below, Fcan you tell me if the cropped image contains a logo, wordmark logo, or just text. I expected image has maximum 1 logo, so if there are more than 1 crop image was detected as 'logo', try to find best results. Give me a list has {len(list_crop)} results for each  crop image without add any words. Example output: ["text","logo"] """
    prompt_tmp =[{
                    "type": "image_url",
                    "image_url": {
                    "url": f"data:image/jpeg;base64,{ad_img_bf}"
                    }
                }
                ]
    h, w = np.array(image.shape[:2]).astype(int)
    list_box_tmp = []
    for tmp in list_crop.keys():
        # print(list_elements_text[tmp])
        _, buffer_bf = cv2.imencode(".jpg", list_crop[tmp])
        encoded_image_bf = base64.b64encode(buffer_bf).decode("utf-8")
        prompt_tmp.append({
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image_bf}"
                        }
                    })
    
    prompt_tmp.append({
            "type": "text",
            "text": prompt
        })
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
            "content": prompt_tmp
            }
    ]
    ## Send the request to the OpenAI API (Retry 2 times)
    try: 
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=1000)
    except Exception as e:
        print(e)
        time.sleep(1)
        response_string = client.chat.completions.create(model="gpt-4o",
                                                            messages=message,
                                                            max_tokens=1000)
    
    # print("category_text_logo_GPT: ",response_string)
    message_contents = response_string.choices[0].message.content
    print("Try to detect logo 2 time: ", message_contents)
    try:
        message_contents = json.loads(message_contents)
    except Exception as e:
        return 
    count = 0
    for i, element in enumerate(image_metadata["Elements"]):
        if str(i) in list_crop.keys():
            image_metadata["Elements"][i]["category"] = message_contents[count]
            count +=1

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
    
    path_inpain = "/home/anlab/Downloads/Allbird_Aug082024"
    
    for filename in list_image:
        # count = 1
        # filename = "781490520120333.mp4"
        # text_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):  # and not os.p
            print(filename)
            image_name = os.path.basename(filename).split(".")[0]
            
            path_object = f"/home/anlab/Downloads/allbirds_July172024_model_4M_21_XL/{image_name}/{image_name}_metadata.json"
            list_coors = []
            with open(path_object ,'r') as fp:
                # json.dump(final_metadata, fp, indent=4, cls=NpEncoder)
                ojbect_metadata = json.load(fp)
            if ojbect_metadata["Object-bound"] != [] :
                for tmp in ojbect_metadata["Object-bound"]: 
                    list_coors.append(tmp["box"])
            
            dalle_inpaint_p = os.path.join(path_inpain,"dalle",image_name+"_inpainted",f"{image_name}_inpainted_inpainted_dalle.png")
            lama_inpaint_p = os.path.join(path_inpain,"lama",image_name,f"{image_name}_inpainted_lama.png")
            firefly_inpaint_p = os.path.join(path_inpain,"firefly",f"{image_name}_inpaint_firefly.png")
            list_name = [os.path.basename(dalle_inpaint_p), os.path.basename(lama_inpaint_p), os.path.basename(firefly_inpaint_p)]
            print(dalle_inpaint_p)
            print(firefly_inpaint_p)
            # sys.exit()
            
            dalle_inpaint_img = cv2.imread(dalle_inpaint_p)
            lama_inpaint_img = cv2.imread(lama_inpaint_p)
            firefly_inpaint_img = cv2.imread(firefly_inpaint_p) if os.path.exists(firefly_inpaint_p) else None
            response = compare_inpaint(dalle_inpaint_img, lama_inpaint_img, firefly_inpaint_img, list_coors, list_name)
            # path_elements = f"/tmp/allbirds_Jul29/{image_name}_metadata.json"
            # 
            
            # with open(path_elements ,'r') as fp:
            #     # json.dump(final_metadata, fp, indent=4, cls=NpEncoder)
            #     final_metadata = json.load(fp)
            
            
            # if ojbect_metadata["Object-bound"] != [] :
            #     for tmp in ojbect_metadata["Object-bound"]:
            #         tmp["category"] = "Not detected"
            #         final_metadata["data"]["Elements"].append(tmp)
            
            # final_metadata["data"]["Click-through-rate"] = ojbect_metadata["Click-through-rate"]
            # final_metadata["data"]["Speaker-detection"] = ojbect_metadata["Speaker-detection"]
            
            # path_final_output = f"/tmp/{image_name}_metadata.json"
            # with open(path_final_output ,'w') as fp:
            #     json.dump(final_metadata, fp, indent=4, cls=NpEncoder)
            # ad_img = cv2.imread(filename)
            # _, buffer_bf = cv2.imencode(".png", ad_img)
            # ad_img_bf = base64.b64encode(buffer_bf).decode("utf-8")
            # path_text = f"/home/anlab/Downloads/allbirds_July172024_model_4M_21_XL/{image_name}/text_detection"
            # path_object = f"/home/anlab/Downloads/allbirds_July172024_model_4M_21_XL/{image_name}/object_detection"
            