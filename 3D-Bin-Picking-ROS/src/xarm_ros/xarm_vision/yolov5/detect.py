import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.augmentations import Albumentations, augment_hsv, copy_paste, letterbox, mixup, random_perspective
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import cv2
import sys
import rospy
import math
import random
import threading
import numpy as np
import moveit_commander
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

if sys.version_info < (3, 0):
    PY3 = False
    import Queue as queue
else:
    PY3 = True
    import queue


class GazeboCamera(object):
    def __init__(self, topic_name='/camera/image_raw/compressed'):
        self._frame_que = queue.Queue(10)
        self._bridge = CvBridge()
        self._img_sub = rospy.Subscriber(topic_name, CompressedImage, self._img_callback)

    def _img_callback(self, data):
        if self._frame_que.full():
            self._frame_que.get()
        self._frame_que.put(self._bridge.compressed_imgmsg_to_cv2(data))
    
    def get_frame(self):
        if self._frame_que.empty():
            return None
        return self._frame_que.get()


def run(
        weights=ROOT / 'best.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    cudnn.benchmark = True  # set True to speed up constant image size inference
    bs=1
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    rospy.init_node('color_recognition_node', anonymous=False)
    dof = rospy.get_param('/xarm/DOF', default=6)
    rate = rospy.Rate(10.0)
    cam = GazeboCamera(topic_name='/camera/image_raw/compressed')
    while not rospy.is_shutdown():
        rate.sleep()
        fram0 = cam.get_frame()
        if fram0 is None:
            continue
        fram=[]
        fram.append(fram0)
        frame0=fram.copy()
        # frame = letterbox(fram,imgsz, stride=stride, auto=True)[0] 
        frame = [letterbox(x, imgsz, stride=stride, auto=True)[0] for x in fram]


        frame = np.stack(frame, 0)
        # Convert
        print(frame.shape)
        frame = frame[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        frame = np.ascontiguousarray(frame)
        dt, seen = [0.0, 0.0, 0.0], 0
        t1 = time_sync()
        frame = torch.from_numpy(frame).to(device)
        frame = frame.half() if model.fp16 else frame.float()  # uint8 to fp16/32
        frame /= 255  # 0 - 255 to 0.0 - 1.0
        if len(frame.shape) == 3:
            frame = frame[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model(frame, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3
        print(pred)
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(frame0[0].shape)[[1, 0, 1, 0]]
            annotator = Annotator(frame0[0], line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(frame.shape[2:], det[:, :4], frame0[0].shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                    print(xywh)
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
            
            im0 = annotator.result()
            view_img=True
            if view_img:
                cv2.imshow("kych", im0)
                print("Result : ",im0.shape)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    rospy.signal_shutdown('key to exit')

if __name__ == "__main__":
    run()






