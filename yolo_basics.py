import cv2
from ultralytics import YOLO
# import time

model = YOLO('yolov8m.pt')
result = model("D:\\trees.jpg",show = True)
cv2.waitKey(0)