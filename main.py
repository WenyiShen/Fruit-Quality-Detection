from ultralytics import YOLO

model = (YOLO("ultralytics/cfg/models/v8/my_yolov8_CBAM.yaml"))
model.train(**{'cfg':'ultralytics/cfg/default.yaml'})