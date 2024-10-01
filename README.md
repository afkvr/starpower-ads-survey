# **Summary**
## **1. Folder "ads_video_analysis"**
This folder contains code for detecting text and speaker in video. Detail informations in [document](https://github.com/afkvr/starpower-ads-survey/blob/master/ads_video_analysis/Pipeline%20-%20Product%20Requirements%20-%20Starpower%20AI%20Ads%20Product%20Data%20Creation%20.docx).

## **2. Folder "ads_image_analysis"**
This folder contains code for detecting text, logo and product in static images. Detail informations in [document](https://github.com/afkvr/starpower-ads-survey/blob/master/ads_image_analysis/Ad%20Image%20Evaluation%20Metadata%20Requirements_.docx).
1. Here are current pipeline for detecting text and logo:
    1. Detect all languages, logo name in image using ChatGPT.
    2. Detect bounding box of text using **easyOCR**.
    3. Detect text content in each box from step 2 using paddleOCR to get all text are the same logo name (Wordmark logo).
    4. Use ChatGPT to evaluate all remain text from step to get "picture logo", "wordmark logo" or "text".
    5. Inpainting all texts in image using lama model.
    6. Detect logo using GroundingDiNo model with 2 inputs:
        - Input image: Inpainted image from step 5.
        - Input text: logo name from step 1.
    7. Use ChatGPT to evaluate and filter detected logos from step 6.
    8. Combine detected logos and wordmark logo nearby.

2. Here are current pipeline for detecting product and background:
    1. Detext all text in images using paddlOCR.
    2. Use ChatGPT to find all brands and product categories fron detected texts in step1.
    3. Inpainting texts by lama model.
    4. Detect all objects/products by groundingDINO model with 2 inputs:
        - Input image: output step 3.
        - Input text: product categories from step 2.
    5. Crop and remove background for each object/product from step4 by SAM-HQ model.
    6. Inpainting object boxe/mask by lama model to get background image.
## **3. Folder "similarity matching"**
This folder contain code train Siamese model for extracting logo images to features.
The latest dataset contains 1820 logo images for 128 brands. Here are [dataset](https://drive.google.com/open?id=14Qs5BJby9w0qLl6pbXUHnQ8DHGMHbpkE&usp=drive_copy)

## **4. Folder "similarity matching inference pipeline"**
This folder contains code for inferencing logo matching (Expected replace step 1.7 in ads_image_analysis):
1. Extract a logo dataset and save features into Database.
2. Extract input image and return top 3 most similar logos.

## **5. Folder "object_detect_yolo"**
This folder contains code for training object detection model based on Yolov8/Yolov11 (Expected replace for step 2.4 in ads_image_analysis)

## **6. Folder "autoit_script"**
This folder contain code for inpainting text/object in image with Firefly function(Generative fill/ content-aware fill) in Photoshop (Expected replace for all inpainting step above). 