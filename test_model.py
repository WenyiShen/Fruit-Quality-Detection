from ultralytics import RTDETR, YOLO
from ultralytics.utils import ASSETS, DEFAULT_CFG, LINUX, MACOS, ONLINE, ROOT, SETTINGS, WINDOWS


CFG = 'my_yolov8_CBAM.yaml'
SOURCE = ASSETS / 'bus.jpg'

def test_model_forward():
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment