from paddleocr import PaddleOCR,draw_ocr
import cv2
import json


##Remove low score 
def remove_low_score(result, threshold):
    ind2rem = []                        #Store index of box to remove
    ans = []                            #Store chosen boxes
    if len(result) != 0:
        for i in range(len(result)):
            if result[i][1][1] < threshold:
                ind2rem.append(i)
        for i in range(len(result)):
            if i not in ind2rem:
                ans.append(result[i])
    return ans
    
def get_ocr(video_path,thresh_hold = 0.9, using_gpu = False, is_show = False, write_res = False):

    cap = cv2.VideoCapture(video_path)

    ocr = PaddleOCR(use_angle_cls=False, lang="en", use_gpu = using_gpu)
    
    count_frame = 0

    raw = {}

    while cap.isOpened():
        
        ret, frame = cap.read()
        
        if ret:
            grayFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            result = ocr.ocr(grayFrame, cls=True)

            if result == [None]:
                count_frame += 1
                continue

            result = result[0]
            #Remove low score
            # result = remove_low_score(result, 0.9)
            raw[count_frame] = result

            boxes = [line[0] for line in result]
            txts = [line[1][0].strip() for line in result]
            score = [line[1][1] for line in result]

            if is_show:
                RGBimg = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                im_show = draw_ocr(RGBimg, boxes ,font_path='/home/kientran/Code/Work/OCR/font/latin.ttf')
                im_show = cv2.cvtColor(im_show, cv2.COLOR_RGB2BGR)
                cv2.imshow("Image", im_show)
            
            count_frame += 1

            # if count_frame > 10:
            #     break
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
     
    cap.release()
    cv2.destroyAllWindows()
    if write_res:
        video_name = video_path.split('/')[-1]
        video_name = video_name.split('.')[0]
        with open(f"/home/kientran/Code/Work/OCR/pipeline/raw_results/{video_name}.json", 'w') as file:
        # with open(f"/home/kientran/Code/Work/OCR/pipeline/raw_results/a.json", 'w') as file:
            json.dump(raw, file)
    return result
def remove_low_score_from_raw(result, video_name):
    new_dict = {}
    for key in result.keys():
        frame_res = result[key]
        frame_res = remove_low_score(frame_res, 0.9)

        new_dict[key] = frame_res
    with open(f"/home/kientran/Code/Work/OCR/pipeline/remove_low_score/{video_name}.json", 'w') as file:
        # with open(f"/home/kientran/Code/Work/OCR/pipeline/raw_results/a.json", 'w') as file:
            json.dump(new_dict, file)

if __name__ == "__main__":
    # get_ocr("/home/kientran/Code/Work/OCR/Video/279573828566031.mp4", write_res=False)
    with open("/home/kientran/Code/Work/OCR/pipeline/raw_results/404759268832213.json", 'r') as file:
        result = json.load(file)
    remove_low_score_from_raw(result,"404759268832213")
