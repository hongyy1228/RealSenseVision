from ultralytics import YOLO
from PIL import Image
import cv2
import torch
import numpy as np

model = YOLO("yolov8n-seg.pt")
with open('YOLOClass.txt') as f:
    class_list = f.read().splitlines()

im1 = cv2.imread('image.jpg')
results = model(source=im1, save = True)

colors = np.random.randint(0, 255, (90, 3))

results = results
value = results[0].cpu()

boxes = value.boxes.numpy()
masks = value.masks.numpy() # Masks object for segmenation masks outputs

#results = model.predict(source="0")

detection_count = boxes.shape[0]


detection_threshold = 0.5

for i in range(detection_count):
    cls_id = int(boxes.cls[i])
    box = boxes[i,:]
    mask = masks.masks[i,:,:]
    class_name = class_list[cls_id]
    pred_prob = box.conf
    color = colors[cls_id]
    x = int(box.xyxy[0,0])
    y = int(box.xyxy[0, 1])
    x1 = int(box.xyxy[0, 2])
    y1 = int(box.xyxy[0, 3])
    cv2.rectangle(im1, (x, y), (x1, y1), (int(color[0]), int(color[1]), int(color[2])), 2)
    cv2.putText(im1, class_name.capitalize(), (x,y-10), 0, 0.8,(int(color[0]), int(color[1]), int(color[2])),2)
    cv2.putText(im1, str(pred_prob.min()), (x, y - 30), 0, 0.8, (int(color[0]), int(color[1]), int(color[2])), 2)









cv2.imshow('window_name',bgr_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()



