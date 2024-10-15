from PIL import Image, ImageEnhance
import numpy as np
import random
import os
import argparse

# Rotation and Flipping
def rotate(image):
    rotated_images = [image.rotate(angle) for angle in [5, -5]]
    return rotated_images

# Noise Injection (Gaussian Noise)
def add_gaussian_noise(image):
    np_img = np.array(image)
    mean = 0
    stddev = 25
    noise = np.random.normal(mean, stddev, np_img.shape)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

# Color Adjustments 
def adjust_colors(image):
    enhancers = [
        ImageEnhance.Brightness(image),
        ImageEnhance.Contrast(image),
    ]
    
    # Random adjustments
    brightness_img = enhancers[0].enhance(random.uniform(0.7, 1.3))  # Brightness range
    contrast_img = enhancers[1].enhance(random.uniform(0.7, 1.3))    # Contrast range

    return [brightness_img, contrast_img]

def parse_argument(): 
    parser = argparse.ArgumentParser(
        description="Data augmentation script."
    )

    parser.add_argument(
        "-r", "--root", type=str, nargs="?", help="Dataset path."
    )


if __name__ == "__main__":
    
    args = parse_argument()
    root = args.root

    sets = os.listdir(root)

    for set_ in sets: 
        imgs = os.listdir(f"{root}/{set_}")
        s_ = 0
        for img in imgs: 
            pimg = Image.open(f"{root}/{set_}/{img}")
            
            rotated_imgs = rotate(pimg)
            noisy_img = add_gaussian_noise(pimg)
            color_adjusted_imgs = adjust_colors(pimg)

            img_name = set_

            for i, rotated_img in enumerate(rotated_imgs): 
                rotated_img.save(f"{root}/{set_}/{img_name}_m{s_}{i+1}.jpg")

            for i, color_adjusted_img in enumerate(color_adjusted_imgs): 
                rotated_img.save(f"{root}/{set_}/{img_name}_m{s_}{i+3}.jpg")

            noisy_img.save(f"{root}/{set_}/{img_name}_m{s_}5.jpg")

            s_ += 1



