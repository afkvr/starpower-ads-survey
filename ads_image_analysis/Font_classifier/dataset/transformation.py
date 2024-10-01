from torchvision import transforms as T
import numpy as np
import cv2 as cv 
to_tensor = T.Compose([
    T.ToTensor(),
    T.Resize((224, 224)),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

augmenter = T.Compose([
    T.RandAugment(),
    to_tensor
])

custom1 = T.Compose([

])

def right_padding(img, size):
    top    = 0
    bottom = 0
    left   = 0
    right  = size

    border = [255,255,255]

    new_img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, None, border)
    return new_img 

def inference_input(img):
    
    h, w = img.shape[:2]
    block = w/h 
        #Branching ([-inf, 0], [0, 1.5), [1.5, 2] [2, +inf])
    log_ratio = np.log2(block)
    
    if (log_ratio>=2):


        n_grids = int(np.round(log_ratio)) 
        grid_dim = h #square grid

        # Padding to make the width of image the smallest square number (with offset) that's larger then multiple of height and total grid
        padding_offset = 10
        if (grid_dim*n_grids*n_grids > w):
            img = right_padding(img, size= grid_dim*n_grids*n_grids - w + padding_offset)

        new_img = np.zeros((grid_dim*n_grids, grid_dim*n_grids, 3), dtype=np.uint8)
        new_w, _ = new_img.shape[:2] # new_h = new_w 
        for line in range(n_grids):
            new_img[(grid_dim*line):(grid_dim*(line+1)), :, :] = img[:,new_w*line:new_w*(line+1),:]

        #Padding 
        top    = int(0.1*h)
        bottom = top
        left   = top
        right  = top

        border = [255,255,255]

        new_img = cv.copyMakeBorder(new_img, top, bottom, left, right, cv.BORDER_CONSTANT, None, border)

    else: 
        new_img = img 
    
    return new_img

# Debug 
if __name__ == "__main__":

    img_path = "F:/tfont/a- bekerchief_34712,Fraunces-300italic.jpg"
    img = cv.imread(img_path)

    new_img = inference_input(img) 

    cv.imshow("original", img)
    cv.imshow("mod", new_img)
    cv.waitKey(0)

