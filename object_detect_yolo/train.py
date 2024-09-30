from ultralytics import YOLO

# Load a model
model = YOLO("/home/ubuntu/object_detect_yolo/checkpoints/yolov8m.pt")

# Train the model
train_results = model.train(
    data="./config/object_detect_yolo8m.yaml",  # path to dataset YAML
    epochs=50,  # number of training epochs
    imgsz=640,  # training image size
    device=0,  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    project="./output",
    batch=16
)

# Evaluate model performance on the validation set
metrics = model.val()

# # Perform object detection on an image
# results = model("path/to/image.jpg")
# results[0].show()

# # Export the model to ONNX format
# path = model.export(format="onnx")  # return path to exported model
