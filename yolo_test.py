from ultralytics import YOLO
from PIL import Image
import cv2
import torch

model = YOLO("yolov8n-seg.pt")

im1 = cv2.imread('image.jpg')
results = model(source=im1)

value = results[0]

boxes = value.boxes
masks = value.masks # Masks object for segmenation masks outputs
probs = value.probs # Class probabilities for classification outputs
#results = model.predict(source="0")



