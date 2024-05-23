from ultralytics import YOLO

import torch
print(torch.cuda.is_available())
print(torch.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')
model = YOLO('yolov8.yaml')
# model = YOLO('my_yolov8_CBAM.yaml')

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
# results = model.train(data='coco128.yaml', epochs=3)
results = model.train(data='my_data.yaml', epochs=20)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')

# Export the model to ONNX format
success = model.export(format='onnx')

#/root/miniconda3/envs/myconda/bin/python