from ultralytics import YOLO
import numpy as np
from PIL import Image
import os
# Load a model
# model = YOLO("/home/ubuntu/object_detect_yolo/checkpoints/yolov8m.pt")
model = YOLO("/home/ubuntu/object_detect_yolo/output/train4/weights/best.pt")

path_test = "/home/ubuntu/object_detect_yolo/data/object_custom/test/images"
list_image_test = [os.path.join(path_test, tmp) for tmp in os.listdir(path_test)]
# # Evaluate model performance on the validation set
# results = model.val(data="./config/object_detect_yolo8m.yaml")
results = model(list_image_test)
# ## Test model with test dataset

# # print("Class indices with average precision:", results.ap_class_index)
# # print("Average precision for all classes:", results.box.all_ap)
# # print("Average precision:", results.box.ap)
# # print("Average precision at IoU=0.50:", results.box.ap50)
# # print("Class indices for average precision:", results.box.ap_class_index)
# # print("Class-specific results:", results.box.class_result)
# print("F1 score:", results.box.f1)
# # print("F1 score curve:", results.box.f1_curve)
# # print("Overall fitness score:", results.box.fitness)
# print("Mean average precision:", results.box.map)
# print("Mean average precision at IoU=0.50:", results.box.map50)
# print("Mean average precision at IoU=0.75:", results.box.map75)
# # print("Mean average precision for different IoU thresholds:", results.box.maps)
# # print("Mean results for different metrics:", results.box.mean_results)
# print("Mean precision:", results.box.mp)
# print("Mean recall:", results.box.mr)
# print("Precision:", np.mean(results.box.p))
# # print("Precision curve:", results.box.p_curve)
# # print("Precision values:", results.box.prec_values)
# # print("Specific precision metrics:", results.box.px)
# print("Recall:", results.box.r)
# # print("Recall curve:", results.box.r_curve)


# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

    # Show results to screen (in supported environments)
    # r.show()

    # Save results to disk
    r.save(filename=f"/home/ubuntu/object_detect_yolo/output_test/{i}.jpg")