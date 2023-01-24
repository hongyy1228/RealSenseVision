import cv2
from streamingFC import *
from mask_rcnn import *
from yolo_mask_rcnn import *

# Load Realsense camera
rs = StreamingCamera()
mrcnn = MaskRCNN()
yolo_mask = YOLO_mask()

while True:
    # Get frame in real time from Realsense camera
    ret, bgr_frame, depth_frame = rs.get_frame()

    # Get object mask
    # boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame) #cv2 rcnn method
    obj_boxes, masks = yolo_mask.yolo_detect_objects_mask(bgr_frame)

    # Draw object mask
    #bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    bgr_frame = yolo_mask.yolo_draw_object_mask(bgr_frame,depth_frame)

    # Show depth info of the objects
    #mrcnn.draw_object_info(bgr_frame, depth_frame)
    #yolo_mask.yolo_draw_object_info(bgr_frame, depth_frame)

    #cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Bgr frame", bgr_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

rs.release()
cv2.destroyAllWindows()