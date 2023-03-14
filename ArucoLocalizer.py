import argparse
import cv2
import sys
import pyrealsense2 as rs2
import time
import socket
import json



from yolo_mask_rcnn import *
from streamingFC import *


def client_program(arr):
    host = socket.gethostname()  # as both code is running on same pc
    port = 5000  # socket server port number

    client_socket = socket.socket()  # instantiate
    client_socket.connect((host, port))  # connect to the server

    data_string = json.dumps(arr)
    client_socket.send(data_string.encode())

yolo_mask = YOLO_mask()
rs = StreamingCamera()

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 65432  # The port used by the server



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

    send_value = [x_diff+20, y_diff, -dis_diff]
    client_program(send_value)

    print(send_value)

    # with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    #     s.connect((HOST, PORT))
    #     s.send(send_value)
    #     data = s.recv(1024)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elapsed = time.time() - now

    while elapsed < 1:
        elapsed = time.time() - now



rs.release()
cv2.destroyAllWindows()