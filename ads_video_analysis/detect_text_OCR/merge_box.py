import json
import numpy as np

def get_frame_list(obj_info_dict, video_length):
    # destinationFrame = []
    # for frame_count in range(video_length):
    #     tmp = [True if frame_count in obj_info_dict[key]['frame'] else False for key in obj_info_dict.keys()]
    #     destinationFrame.append(tmp)
    # print(destinationFrame)
    frameDict = {}
    for frame_count in range(video_length):
        frameDict[frame_count] = {}
        for key in obj_info_dict.keys():
            
            if frame_count in obj_info_dict[key]['frame']:
                frameDict[frame_count][key] = obj_info_dict[key]
    return frameDict


def sort_boxes(box_dict):
    """
    Sorts the given dictionary of boxes from top to bottom and left to right.

    Parameters:
    box_dict (dict): Dictionary containing the boxes.

    Returns:
    list: A sorted list of tuples (key, value) based on top-to-bottom and left-to-right order.
    """
    # Convert the dictionary to a list of items (key, value)
    items = list(box_dict.items())
    
    # Define the sort key function
    def sort_key(item):
        key, value = item
        # Use the top-left corner of the box as the sort key (y-coordinate first, then x-coordinate)
        box = value['box']
        top_left_corner = box[0]  # Assuming the first point is the top-left corner
        x, y = top_left_corner
        return (y, x)
    
    # Sort the list of items based on the sort key function
    sorted_items = sorted(items, key=sort_key)
    
    return dict(sorted_items)


def check_same_line(box1, box2, threshold = 5):
    return abs(box1[0][1]-box2[0][1]) <= threshold

def check_height(height1, height2):
    return abs(height1 - height2) < 15
def merge_boxes(box_dict):
    # Initialize a new dictionary to store the merged boxes
    merged_dict = {}
    # Keep track of which keys have been merged to avoid duplicate processing
    merged_keys = set()
    # List of dictionary keys to iterate over
    box_keys = list(box_dict.keys())

    # Iterate through the keys of the box dictionary
    for i in range(len(box_keys)):
        key1 = box_keys[i]
        # Skip if this key has already been merged
        if key1 in merged_keys:
            continue
        # Get the first box
        box1 = box_dict[key1]
        # Initialize a variable to hold the merged box data
        merged_box = {
            'text_origin': box1['text_origin'],
            'text_normalized': box1['text_normalized'],
            'box': box1['box'],
            'box_height': box1['box_height']
            # 'frame': box1['frame']
        }
        # Iterate through the remaining boxes to compare
        for j in range(i + 1, len(box_keys)):
            key2 = box_keys[j]

            # Skip if this key has already been merged
            if key2 in merged_keys:
                continue

            # Get the second box
            box2 = box_dict[key2]

            # if not check_height(merged_box['box_height'], box2['box_height']):
            #     continue
            # Check if the two boxes can be merged using the is_merge function
            
            if is_merge(merged_box, box2):
                # Combine the text origin and normalized text
                if check_same_line(merged_box['box'], box2['box']):
                    merged_box['text_origin'] += " " + box2['text_origin']
                else:
                    merged_box['text_origin'] += " \n " + box2['text_origin']
                merged_box['text_normalized'] += box2['text_normalized']
                # Update the bounding box to include both boxes
                merged_box['box'] = merge_bounding_boxes(merged_box['box'], box2['box'])

                # Combine the frames
                # merged_box['frame'] = list(set(merged_box['frame'] + box2['frame']))

                # Mark the second box as merged
                merged_keys.add(key2)

        # Add the merged box to the new dictionary
        merged_dict[key1] = merged_box

    return merged_dict


def merge_bounding_boxes(box1, box2):
    # Extract the x and y coordinates from each bounding box
    x_coords = [point[0] for point in box1 + box2]
    y_coords = [point[1] for point in box1 + box2]

    # Find the minimum and maximum x and y coordinates to form a new bounding box
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)

    # Define the merged box using the new coordinates
    merged_box = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y]
    ]

    return merged_box


def box_distance(box1, box2):
    y_dist = min(abs(box1[0][1] - box2[3][1]), abs(box1[3][1] - box2[0][1]))
    x_dist = min(abs(box1[0][0] - box2[1][0]), abs(box1[1][0] - box2[0][0]))
    edge_dist = min(x_dist, y_dist)
    return edge_dist
# Example `is_merge` function that checks if boxes are close to each other
# def is_merge(box1, box2, threshold = 200.0):
#     # Define a tolerance for merging (e.g., distance between boxes or other criteria)
    
#     # Calculate the distance between the centers of the two boxes
#     if True:
#         center1 = [(box1['box'][0][0] + box1['box'][2][0]) / 2, (box1['box'][0][1] + box1['box'][2][1]) / 2]
#         center2 = [(box2['box'][0][0] + box2['box'][2][0]) / 2, (box2['box'][0][1] + box2['box'][2][1]) / 2]

#         distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    # return distance <= threshold

def is_merge(box1, box2, threshold=30.0):
    text1 = box1['text_origin']
    text2 = box2['text_origin']
    box1 = box1['box']
    box2 = box2['box']
    # Calculate the horizontal and vertical distances between the boxes
    horizontal_dist = min(abs(box1[0][0] - box2[1][0]), abs(box2[0][0] - box1[1][0]))
    vertical_dist = min(abs(box1[0][1] - box2[3][1]), abs(box2[0][1] - box1[3][1]))
    
    # print(f"Vertical dist: {vertical_dist}, Horizontal dist: {horizontal_dist}")
    # Calculate the widths and heights of the boxes
    center_1 = np.array([(box1[0][0]+box1[1][0])/2,(box1[0][1]+box1[3][1])/2])
    center_2 = np.array([(box2[0][0]+box2[1][0])/2,(box2[0][1]+box2[3][1])/2])
    box1_width = abs(box1[0][0] - box1[1][0])
    box1_height = abs(box1[0][1] - box1[3][1])
    box2_width = abs(box2[0][0] - box2[1][0])
    box2_height = abs(box2[0][1] - box2[3][1])
    
    with open('/tmp/log_dist.txt', 'a') as file:
        file.writelines(f"{text1} - {text2}: Vertical dist: {vertical_dist} - {vertical_dist <= threshold} - {abs(center_1[0] - center_2[0]) < max(box1_width, box2_width)/2} , Horizontal dist: {horizontal_dist} - {horizontal_dist <= threshold} - {abs(center_1[1] - center_2[1]) < max(box1_height, box2_height)/2}\n")
    
    # Check if the boxes are close enough horizontally or vertically
    # if horizontal_dist <= threshold and abs(box1[0][1] - box2[0][1]) < min(box1_height, box2_height)/2:
    #     return True
    # elif vertical_dist <= threshold and abs(box1[0][0] - box2[0][0]) < min(box1_width, box2_width)/2:
    if horizontal_dist <= threshold and abs(center_1[1] - center_2[1]) < max(box1_height, box2_height)/2:
        return True
    elif vertical_dist <= threshold and abs(center_1[0] - center_2[0]) < max(box1_width, box2_width)/2:
        return True
    else:
        return False

def do_merge(vid_name, num_frame = 1500):
    with open(f"/home/kientran/Code/Work/OCR/pipeline/tracking_results/{vid_name}.json", 'r') as file:
        obj_info = json.load(file)
    
    frameDict = get_frame_list(obj_info, num_frame)

    res = {}
    for frame_count in frameDict.keys():
        frame = frameDict[frame_count]
        frame = sort_boxes(frame)
        for key in frame.keys():
            tmp_box = frame[key]['box']
            frame[key]['box_height'] = abs(tmp_box[0][1] - tmp_box[3][1])
        merge_ = merge_boxes(frame)
        res[frame_count] = merge_
    with open(f"/home/kientran/Code/Work/OCR/pipeline/merge/{vid_name}.json", 'w') as file:
        json.dump(res, file)

# def do_merge_2(text_objects,total_frame):
#     frameDict = get_frame_list(text_objects,total_frame)
#     res = {}
#     for frame_count in frameDict.keys():
#         frame = frameDict[frame_count]
#         frame = sort_boxes(frame)
#         for key in frame.keys():
#             tmp_box = frame[key]['box']
#             frame[key]['box_height'] = abs(tmp_box[0][1] - tmp_box[3][1])
#         merge_ = merge_boxes(frame)
#         res[frame_count] = merge_

#     return res
def do_merge_2(text_objects):
    # frameDict = get_frame_list(text_objects)
    res = {}
    for frame_key in text_objects.keys():
        textObj_list = text_objects[frame_key]
        textObj_list = sort_boxes(textObj_list)
        for key in textObj_list.keys():
            tmp_box = textObj_list[key]['box']
            textObj_list[key]['box_height'] = abs(tmp_box[0][1] - tmp_box[3][1])
        merge_ = merge_boxes(textObj_list)
        res[frame_key] = merge_

    return res



if __name__ == "__main__":
    with open("/home/kientran/Code/Work/OCR/pipeline/tracking_results/404759268832213.json", 'r') as file:
        obj_info = json.load(file)
    
    frameDict = get_frame_list(obj_info, 1500)

    res = {}
    for frame_count in frameDict.keys():
        frame = frameDict[frame_count]
        frame = sort_boxes(frame)
        for key in frame.keys():
            tmp_box = frame[key]['box']
            frame[key]['box_height'] = abs(tmp_box[0][1] - tmp_box[3][1])
        merge_ = merge_boxes(frame, is_merge)
        res[frame_count] = merge_
    with open("test.json", 'w') as file:
        json.dump(res, file)
    