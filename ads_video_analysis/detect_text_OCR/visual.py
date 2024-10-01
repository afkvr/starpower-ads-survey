import cv2
import json
import numpy as np



def visualize_raw(raw_path, video_path):
    with open(raw_path, 'r') as file:
        txt_inf = json.load(file)
    

    cap = cv2.VideoCapture(video_path)
    vidFrame = []
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            # frame = cv2.resize(frame, (595,595))
            width, height = int(cap.get(3)), int(cap.get(4))
            
            vidFrame.append(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    for key in txt_inf.keys():
        for i in range(len(txt_inf[key])):
            box = txt_inf[key][i][0]
            
        
    
            pts = np.array(box, dtype = np.int32)


            
            vidFrame[int(key)] = cv2.polylines(vidFrame[int(key)], [pts], 
                          True, (0,0,255), 2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_vid.mp4', fourcc, 30.0, (width,height))

    for i in range(len(vidFrame)):
        out.write(vidFrame[i])
    out.release()



def visualize(vid_path):
    with open(f"/home/kientran/Code/Work/OCR/pipeline/merge/{vid_path}.json", 'r') as file:
        txt_inf = json.load(file)
    cap = cv2.VideoCapture(f"/home/kientran/Code/Work/OCR/Video/{vid_path}.mp4")
    vidFrame = []
    while cap.isOpened():
        ret, frame = cap.read()
        
        if ret:
            width, height = int(cap.get(3)), int(cap.get(4))
            
            vidFrame.append(frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # for key in txt_inf.keys():
    #     box = txt_inf[key]['box']
    #     frame = txt_inf[key]['frame']
    #     if len(frame) < 4:
    #         continue
    
    #     pts = np.array(box, dtype = np.int32)


    #     for i in frame:
    #         vidFrame[i] = cv2.polylines(vidFrame[i], [pts], 
    #                       True, (0,0,255), 2)

    for frame_count in txt_inf.keys():
        for boxid in txt_inf[frame_count]:
            box = txt_inf[frame_count][boxid]['box']
            pts = np.array(box, dtype = np.int32)
            vidFrame[int(frame_count)] = cv2.polylines(vidFrame[int(frame_count)], [pts], 
                        True, (0,0,255), 2)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'/home/kientran/Code/Work/OCR/pipeline/Visualize video/{vid_path}.mp4', fourcc, 30.0, (width,height))

    for i in range(len(vidFrame)):
        out.write(vidFrame[i])
    out.release()
    # if __name__ =="__main__":
        # visualize_raw("/home/kientran/Code/Work/OCR/pipeline/raw_results/279573828566031.json", "/home/kientran/Code/Work/OCR/Video/279573828566031.mp4")