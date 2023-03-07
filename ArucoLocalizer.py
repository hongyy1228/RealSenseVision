import argparse
import imutils
import cv2
import sys
import pyrealsense2 as rs2
import time

from yolo_mask_rcnn import *
from streamingFC import *


yolo_mask = YOLO_mask()
rs = StreamingCamera()




while True:
    now = time.time()
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame, depth_intrin = rs.get_frame()


    obj_boxes, masks = yolo_mask.yolo_detect_objects_mask(bgr_frame, depth_frame, depth_intrin)

    # Draw object mask
    # bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    bgr_frame, x_diff, y_diff, dis_diff = yolo_mask.yolo_draw_object_mask(bgr_frame, depth_frame,depth_intrin)
    #cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Bgr frame", bgr_frame)

    print(x_diff)
    print(y_diff)
    print(dis_diff)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elapsed = time.time() - now

    while elapsed < 1:
        elapsed = time.time() - now



rs.release()
cv2.destroyAllWindows()