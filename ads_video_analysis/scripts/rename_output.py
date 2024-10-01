import json
import os 
import shutil
import copy
# Define the field name mapping
field_name_mapping_speaker = {
    "video-infos": "video-info",
    "resolution": "video-resolution",
    "duration": "video-duration",
    "num-speakers": "amount-of-speakers",
    "ratio-speaker-duration": "percentage-duration-speaker-cover",
    "ratio-speaker-area": "percentage-frame-speaker-cover-cross-duration",
    "ratio-allspeakers-areas":"percentage-frame-allspeakers-cover-cross-duration",
    "block-size":"size-objective",
    "ratio-block-frame":"size-percentage-of-frame",
    "block-coordinates":"location-objective",
    "block-sectors":"location-sector",
    
    # Add more field mappings as needed
}
field_name_mapping_text = {
    # "video-infos": "video-info",
    "resolution": "video-resolution",
    "duration": "video-duration",
    # "num-text-blocks": "amount-of-text-blocks",
    # "ratio-text-duration": "percentage-duration-text-on-screen",
    # "ratio-text-area": "percentage-frame-text-cover-cross-duration",
    # "block-size":"size-objective",
    # "ratio-block-frame":"size-percentage-of-frame",
    # "block-coordinates":"location-objective",
    # "block-sectors":"location-sector",
    
    # Add more field mappings as needed
}

# def rename_fields(data, field_mapping):
#     if isinstance(data, dict):
#         return {field_mapping.get(k, k): rename_fields(v, field_mapping) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [rename_fields(item, field_mapping) for item in data]
#     else:
#         return data

def rename_fields(data, field_mapping, target_key=None):
    if isinstance(data, dict):
        new_data = {}
        for k, v in data.items():
            if k == target_key:
                # Apply field mapping only to the target_key
                new_data[k] = {field_mapping.get(inner_k, inner_k): rename_fields(inner_v, field_mapping) for inner_k, inner_v in v.items()}
            else:
                new_data[k] = rename_fields(v, field_mapping, target_key)
        return new_data
    elif isinstance(data, list):
        return [rename_fields(item, field_mapping, target_key) for item in data]
    else:
        return data


folder_input = "/home/anlab/Downloads/Ads_video_detect_results-20240520T102122Z-001/Ads_video_detect_results/Food/font_recognition"
folder_output = "/home/anlab/Downloads/Ads_video_detect_results-20240520T102122Z-001/Ads_video_detect_results_rename/Food/speaker_recognition"
folder_output2 = "/home/anlab/Downloads/Ads_video_detect_results-20240520T102122Z-001/results/Food/font_recognition"
folder_output3 = "/home/anlab/Downloads/Ads_video_detect_results-20240520T102122Z-001/results_/Food/font_recognition"
for file in os.listdir(folder_output2):
    # if not file.lower().endswith(('.json')):
    #     continue
    # id = file.split("_")[0]
    # speaker_path = f"""{folder_output}/{id}_speaker_recognition_results.json"""
    # if os.path.isfile(speaker_path):
    #     speaker_path_save = f"""{folder_output2}/{id}_speaker_recognition_results.json"""
    #     shutil.copyfile(speaker_path, speaker_path_save)
    # Read the JSON file
    # print(file)
    with open(os.path.join(folder_output2,file), 'r') as json_file:
        json_data = json.load(json_file)

    # Update the JSON field names
    updated_json_data = rename_fields(json_data, field_name_mapping_text,target_key="video-info")
    # Rename field in video-infos
    # updated_json_data = copy.deepcopy(json_data)
    # for key in list(field_name_mapping_text.keys()):
        
    #     updated_json_data["data"][0]["video-info"][field_name_mapping_text[key]] = updated_json_data["data"][0]["video-info"][key]
    #     updated_json_data["data"][0]["video-info"].pop(key)
    # Update number
    # updated_json_data["data"][0]["video-info"]["amount-of-speakers"] = len(updated_json_data["data"][0]["speakers-data"])
    # for idx in range(len(updated_json_data["data"][0]["sections-data"])):
    #     updated_json_data["data"][0]["sections-data"][idx]["amount-of-speakers"] = len(updated_json_data["data"][0]["sections-data"][idx]["speaker-infos"])
    #     # updated_json_data["data"][0]["sections-data"]
    # # Write the updated JSON to a new file
    new_path = f"""{folder_output3}/{file.split("/")[-1]}"""
    
    with open(new_path, 'w') as updated_json_file:
        json.dump(updated_json_data, updated_json_file, indent=4)

print("Field names updated successfully!")
