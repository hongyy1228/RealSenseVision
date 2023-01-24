import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class YOLO_mask:
    def __init__(self):
        # Loading YOLOv8
        self.model = YOLO("yolov8n-seg.pt")
        self.object_tracker = DeepSort(max_age=5,
                                  n_init=2,
                                  nms_max_overlap=1.0,
                                  max_cosine_distance=0.3,
                                  nn_budget=None,
                                  override_track_class=None,
                                  embedder="mobilenet",
                                  half=True,
                                  bgr=True,
                                  embedder_gpu=True,
                                  embedder_model_name=None,
                                  embedder_wts=None,
                                  polygon=False,
                                  today=None)

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.class_list = []
        with open('YOLOClass.txt') as f:
            self.class_list = f.read().splitlines()



        # Distances
        self.distances = []


    def yolo_detect_objects_mask(self, bgr_frame):

        results = self.model(source=bgr_frame)
        value = results[0].cpu()

        self.obj_boxes = value.boxes.numpy()
        self.masks = value.masks.numpy()
        self.detection = []
        tmp = self.obj_boxes.boxes
        detection_count = tmp.shape[0]
        for i in range(detection_count):
            box = self.obj_boxes[i, :]
            pred_prob = box.conf
            if pred_prob < self.detection_threshold:
                continue
            cls_id = int(self.obj_boxes.cls[i])
            class_name = self.class_list[cls_id]
            x = int(box.xyxy[0, 0])
            y = int(box.xyxy[0, 1])
            x1 = int(box.xyxy[0, 2])
            y1 = int(box.xyxy[0, 3])
            self.detection.append(([x, y, x1, y1], pred_prob, class_name))




        return self.obj_boxes, self.masks

    def yolo_draw_object_mask(self, bgr_frame, depth_frame):
        tmp = self.obj_boxes.boxes
        detection_count = tmp.shape[0]
        # loop through the detection
        for i in range(detection_count):
            cls_id = int(self.obj_boxes.cls[i])
            box = self.obj_boxes[i, :]
            mask = self.masks.masks[i, :, :]
            class_name = self.class_list[cls_id]
            pred_prob = box.conf
            color = self.colors[cls_id]
            x = int(box.xyxy[0, 0])
            y = int(box.xyxy[0, 1])
            x1 = int(box.xyxy[0, 2])
            y1 = int(box.xyxy[0, 3])

            if pred_prob < self.detection_threshold:
                continue

            tracks = self.object_tracker.update_tracks(self.detection,
                                                  frame=bgr_frame)  # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class )

            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()

                cx = (x + x1) // 2
                cy = (y + y1) // 2
                depth_mm = depth_frame[cy, cx]

                bbox = ltrb

                cv2.rectangle(bgr_frame, (x, y), (x1, y1), (int(color[0]), int(color[1]), int(color[2])), 2)
                cv2.putText(bgr_frame, "ID: " + str(track_id) + class_name.capitalize(), (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (int(color[0]), int(color[1]), int(color[2])), 2)
                cv2.putText(bgr_frame, str(pred_prob.min()), (x, y - 30), 0, 0.8,
                            (int(color[0]), int(color[1]), int(color[2])),
                            2)
                cv2.putText(bgr_frame, "{} cm".format(depth_mm / 10), (x, y - 50), 0, 1.0,
                            (int(color[0]), int(color[1]), int(color[2])), 2)





        return bgr_frame





    def yolo_draw_object_info(self, bgr_frame, depth_frame):
        # loop through the detection
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center

            depth_mm = depth_frame[cy, cx]

            cv2.line(bgr_frame, (cx, y), (cx, y2), color, 1)
            cv2.line(bgr_frame, (x, cy), (x2, cy), color, 1)

            class_name = self.classes[int(class_id)]
            cv2.rectangle(bgr_frame, (x, y), (x + 250, y + 70), color, -1)
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 25), 0, 0.8, (255, 255, 255), 2)
            cv2.putText(bgr_frame, "{} cm".format(depth_mm / 10), (x + 5, y + 60), 0, 1.0, (255, 255, 255), 2)
            cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)




        return bgr_frame
