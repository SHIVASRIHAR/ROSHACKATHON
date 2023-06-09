#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import cv2
import rospy
import time
import math
import yaml
import random
import threading
import numpy as np
import moveit_commander
import argparse

from pathlib import Path
from unittest import result

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

import torch
import torch.backends.cudnn as cudnn
if sys.version_info < (3, 0):
    PY3 = False
    import Queue as queue
else:
    PY3 = True
    import queue



class GripperCtrl(object):
    def __init__(self):
        self._commander = moveit_commander.move_group.MoveGroupCommander('xarm_gripper')
        self._init()

    def _init(self):
        self._commander.set_max_acceleration_scaling_factor(1.0)
        self._commander.set_max_velocity_scaling_factor(1.0)
    
    def open(self, wait=True):
        try:
            self._commander.set_named_target('open')
            ret = self._commander.go(wait=wait)
            print('gripper_open, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] gripper open exception, {}'.format(e))
        return False

    def close(self, wait=True):
        try:
            self._commander.set_named_target('close')
            ret = self._commander.go(wait=wait)
            print('gripper_close, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] gripper close exception, {}'.format(e))
        return False

@torch.no_grad()
def run(
        weights='/home/oem/Downloads/best.pt',  # model.pt path(s)
        source=4 , # file/dir/URL/glob, 0 for webcam
        data='/home/oem/Downloads/data.yaml',  # dataset.yaml path
        imgsz=(640, 480),  # inference size (height, width)
        conf_thres=0.80,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results
        save_txt=True,  # save results to *.txt
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
    j=0
    k=0
    results=[]
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    # while j<100:
    #     j+=1
    #     print(j)
    for path, im, im0s, vid_cap, s in dataset:
        if(j<20):
            j+=1
            print(j)
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred) :  # per image
                print('b')
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            
                            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                            results_cls=[]
                            results_cls.append(xywh)
                            results_cls.append(int(cls.item())+1)
                            results.append(results_cls)

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()
                

                if view_img:
                    
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(100)  # 1 millisecond
        else:
            break
    return results
                 

            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    return results

class XArmCtrl(object):
    def __init__(self, dof):
        self._commander = moveit_commander.move_group.MoveGroupCommander('xarm{}'.format(dof))
        self._init()
    
    def _init(self):
        # self._commander.set_max_acceleration_scaling_factor(0.5)
        # self._commander.set_max_velocity_scaling_factor(0.5)
        pass

    def set_joints(self, angles, wait=True):
        try:
            joint_target = self._commander.get_current_joint_values()
            for i in range(joint_target):
                if i >= len(angles):
                    break
                if angles[i] is not None:
                    joint_target[i] = math.radians(angles[i])
            print('set_joints, joints={}'.format(joint_target))
            self._commander.set_joint_value_target(joint_target)
            ret = self._commander.go(wait=wait)
            print('move to finish, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] set_joints exception, ex={}'.format(e))
    
    def set_joint(self, angle, inx=-1, wait=True):
        try:
            joint_target = self._commander.get_current_joint_values()
            joint_target[inx] = math.radians(angle)
            print('set_joints, joints={}'.format(joint_target))
            self._commander.set_joint_value_target(joint_target)
            ret = self._commander.go(wait=wait)
            print('move to finish, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] set_joint exception, ex={}'.format(e))
        return False

    def moveto(self, x=None, y=None, z=None, ox=None, oy=None, oz=None, relative=False, wait=True):
        if x == 0 and y == 0 and z == 0 and ox == 0 and oy == 0 and oz == 0 and relative:
            return True
        try:
            pose_target = self._commander.get_current_pose().pose
            if relative:
                pose_target.position.x += x / 1000.0 if x is not None else 0
                pose_target.position.y += y / 1000.0 if y is not None else 0
                pose_target.position.z += z / 1000.0 if z is not None else 0
                pose_target.orientation.x += ox if ox is not None else 0
                pose_target.orientation.y += oy if oy is not None else 0
                pose_target.orientation.z += oz if oz is not None else 0
            else:
                pose_target.position.x = x / 1000.0 if x is not None else pose_target.position.x
                pose_target.position.y = y / 1000.0 if y is not None else pose_target.position.y
                pose_target.position.z = z / 1000.0 if z is not None else pose_target.position.z
                pose_target.orientation.x = ox if ox is not None else pose_target.orientation.x
                pose_target.orientation.y = oy if oy is not None else pose_target.orientation.y
                pose_target.orientation.z = oz if oz is not None else pose_target.orientation.z
            print('move to position=[{:.2f}, {:.2f}, {:.2f}], orientation=[{:.6f}, {:.6f}, {:.6f}]'.format(
                pose_target.position.x * 1000.0, pose_target.position.y * 1000.0, pose_target.position.z * 1000.0,
                pose_target.orientation.x, pose_target.orientation.y, pose_target.orientation.z
            ))
            self._commander.set_pose_target(pose_target)
            ret = self._commander.go(wait=wait)
            print('move to finish, ret={}'.format(ret))
            return ret
        except Exception as e:
            print('[Ex] moveto exception: {}'.format(e))
        return False


class MotionThread(threading.Thread):
    def __init__(self, que, **kwargs):
        if PY3:
            super().__init__()
        else:
            super(MotionThread, self).__init__()
        self.que = que
        self.daemon = True
        self.in_motion = True
        dof = kwargs.get('dof', 6)
        self._xarm_ctrl = XArmCtrl(dof)
        self._gripper_ctrl = GripperCtrl()
        self._offset_z = -172
        self._grab_z = kwargs.get('grab_z', 195) + self._offset_z
        self._safe_z = kwargs.get('safe_z', 300) + self._offset_z
        self._iden_z = kwargs.get('iden_z', 200) + self._offset_z
        self._only_check_xyz = kwargs.get('only_check_xyz', True)
        self._detection_point = kwargs.get('detection_point', [370, 0, 600, 180, 0, 0])
        self._fixed_point = kwargs.get('fixed_point', [420, 35, self._iden_z])
        self._params = self._read_params_from_yaml()
    
    def _check_detection_point(self):
        for i in range(6):
            if i >= 3 and self._only_check_xyz:
                return True
            if self._detection_point[i] != self._params['DP'][i]:
                print('DP1={}, DP2={}'.format(self._params['DP'], self._detection_point))
                return False
        return True
    
    def _read_params_from_yaml(self, path=os.path.join(os.path.expanduser('~'), '.ros', 'xarm_vision', 'color_recognition.yaml')):
        params = {
            'DP': [370, 0, 600, 180, 0, 0],
            'FP': [[420, 35], [326.5, 231.5]],
            'params': [[(0.9090909090909091, 0.0009410878976096362), (0.9192200557103064, 2.8529486521114114e-05)], [(0.9392265193370166, -0.00022946441150392823), (0.9510086455331412, -0.00015799585418878643)], [(0.8602150537634409, -0.00039801737594607), (0.9038461538461539, 0.00021413081114573658)], [(0.9579085370131666, 0.00015027274444733406), (0.9133663184308918, 0.00017367528238027713)]]
        }
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                LU = data.get('LU')
                LD = data.get('LD')
                RU = data.get('RU')
                RD = data.get('RD')
                DP = data.get('DP')
                FP = data.get('FP')
                new_params = {
                    'DP': [DP['x'], DP['y'], DP['z'], DP['rx'], DP['ry'], DP['rz']],
                    'FP': [[FP['x'], FP['y']], [FP['cx'], FP['cy']]],
                    'params': [
                        [(LU['xp_base'], LU['xp_step']), (LU['yp_base'], LU['yp_step'])],
                        [(LD['xp_base'], LD['xp_step']), (LD['yp_base'], LD['yp_step'])],
                        [(RU['xp_base'], RU['xp_step']), (RU['yp_base'], RU['yp_step'])],
                        [(RD['xp_base'], RD['xp_step']), (RD['yp_base'], RD['yp_step'])],
                    ]
                }
                params = new_params
        except Exception as e:
            print('read parameters from yaml failed, {}'.format(e))
        return params
        

    def _write_params_to_yaml(self, path=os.path.join(os.path.expanduser('~'), '.ros', 'xarm_vision', 'color_recognition.yaml')):
        try:
            data = {
                'LU': {
                    'xp_base': self._params['params'][0][0][0], 'xp_step': self._params['params'][0][0][1],
                    'yp_base': self._params['params'][0][1][0], 'yp_step': self._params['params'][0][1][1]
                },
                'LD': {
                    'xp_base': self._params['params'][1][0][0], 'xp_step': self._params['params'][1][0][1],
                    'yp_base': self._params['params'][1][1][0], 'yp_step': self._params['params'][1][1][1]
                },
                'RU': {
                    'xp_base': self._params['params'][2][0][0], 'xp_step': self._params['params'][2][0][1],
                    'yp_base': self._params['params'][2][1][0], 'yp_step': self._params['params'][2][1][1]
                },
                'RD': {
                    'xp_base': self._params['params'][3][0][0], 'xp_step': self._params['params'][3][0][1],
                    'yp_base': self._params['params'][3][1][0], 'yp_step': self._params['params'][3][1][1]
                },
                'DP': {
                    'x': self._params['DP'][0],
                    'y': self._params['DP'][1],
                    'z': self._params['DP'][2],
                    'rx': self._params['DP'][3],
                    'ry': self._params['DP'][4],
                    'rz': self._params['DP'][5],
                },
                'FP': {
                    'x': self._params['FP'][0][0],
                    'y': self._params['FP'][0][1],
                    'cx': self._params['FP'][1][0],
                    'cy': self._params['FP'][1][1],
                }
            }
            dir_path = os.path.abspath(os.path.dirname(path))
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
                print('write parameters to {} success'.format(path))
                return True
        except Exception as e:
            print('write parameters to yaml failed: {}'.format(e))
        return False

    def _rect_to_move_params(self, rect):
        xp, yp = self._get_xp_yp(rect)
        return int((self._params['FP'][1][1] - rect[0][1]) * xp + self._params['FP'][0][0]), int((self._params['FP'][1][0] - rect[0][0]) * yp + self._params['FP'][0][1]), rect[2] - 90

    def _get_xp_yp(self, rect):
        inx = 0
        if rect[0][0] <= self._params['FP'][1][0] and rect[0][1] <= self._params['FP'][1][1]:
            inx = 0
        elif rect[0][0] <= self._params['FP'][1][0] and rect[0][1] > self._params['FP'][1][1]:
            inx = 1
        elif rect[0][0] >= self._params['FP'][1][0] and rect[0][1] <= self._params['FP'][1][1]:
            inx = 2
        else:
            inx = 3
        xp = self._params['params'][inx][0][0] + self._params['params'][inx][0][1]
        yp = self._params['params'][inx][1][0] + self._params['params'][inx][1][1]
        return xp, yp
    
    def _motion_init(self):
        pass
    
    def _gripper_init(self):
        pass

    def _move_to_detection_point(self):
        pose = [self._detection_point[0], self._detection_point[1], self._detection_point[2] + self._offset_z, 1, 0, 0]
        ret = self._xarm_ctrl.moveto(*pose)
        self._xarm_ctrl.set_joint(0)
        return ret

    def _parameters_identification(self):
        print('Please make sure that the current camera screen does not recognize any color blocks.')
        input('Press to start parameters identification >>> ')

        def iden_point(pose):
            rect = None
            while rect is None:
                self.in_motion = True
                if not self._xarm_ctrl.moveto(z=self._safe_z):
                    continue
                if not self._xarm_ctrl.moveto(x=pose[0], y=pose[1], z=self._safe_z):
                    continue
                if not self._xarm_ctrl.moveto(*pose):
                    continue
                print('Please place the color block to be recognized in the middle of the gripper.')
                input('Press to continue >>> ')
                if not self._xarm_ctrl.moveto(z=self._safe_z):
                    continue
                while not self._move_to_detection_point():
                    time.sleep(1)
                time.sleep(1)
                self.in_motion = False
                try:
                    item = self.que.get(timeout=3)
                except:
                    self.in_motion = True
                    print('No color block is recognized.')
                    input('Press to continue >>> ')
                else:
                    self.in_motion = True
                    if len(item) == 1:
                        rect = item[0]
                        break
                    else:
                        print('More than one color block is recognized, please remove the extra color block.')
                        input('Press to continue >>> ')
            return rect

        rects = []
        tmp_poses = [
            [500, 200, self._iden_z],  # Left-Up
            [250, 200, self._iden_z],  # Left-Down
            [500, -200, self._iden_z],  # Right-Up
            [250, -200, self._iden_z],  # Right-Down
        ]
        poses = [
            self._fixed_point,  # Fixed Point
        ]
        for p in tmp_poses:
            poses.append([(poses[0][0] + p[0]) / 2.0, (poses[0][1] + p[1]) / 2.0, p[2]])
            poses.append(p)

        for p in poses:
            pose = p + [1, 0, 0]
            rect = iden_point(pose)
            print('pose={}, rect={}'.format(pose, rect))
            rects.append(rect)
        
        params = []

        for i in range(1, len(rects), 2):
            xp1 = abs(poses[0][0] - poses[i][0]) / float(abs(rects[0][0][1] - rects[i][0][1]))
            yp1 = abs(poses[0][1] - poses[i][1]) / float(abs(rects[0][0][0] - rects[i][0][0]))
            xp = abs(poses[0][0] - poses[i+1][0]) / float(abs(rects[0][0][1] - rects[i+1][0][1]))
            yp = abs(poses[0][1] - poses[i+1][1]) / float(abs(rects[0][0][0] - rects[i+1][0][0]))

            print('[{}] Xp1: {}, Yp1: {}, Xp: {}, Yp: {}'.format(i, xp1, yp1, xp, yp))
            params.append([
                (xp, (xp1 - xp) / float(abs(rects[i][0][1] - rects[i-1][0][1]))),
                (yp, (yp1 - yp) / float(abs(rects[i][0][0] - rects[i-1][0][0]))),
            ])

        self._params = {
            'DP': self._detection_point,
            'FP': [[poses[0][0], poses[0][1]], [rects[0][0][0], rects[0][0][1]]],
            'params': params
        }
        print('*' * 60)
        print(self._params)
        print('*' * 60)

        print('End of parameter identification')
        tmp = input('Whether to save the parameters in the configuration file? Y/N (N) >>> ')
        if tmp.upper() == 'Y':
            self._write_params_to_yaml()

    def run(self):
        self._move_to_detection_point()
        self._gripper_ctrl.open()

        if not self._check_detection_point():
            print('This parameter does not meet the current detection position')
            tmp = input('Re-identify the coordinate conversion parameters? Y/N (Y) >>> ')
            if tmp.upper() != 'N':
                self._parameters_identification()
        else:
            tmp = input('Re-identify the coordinate conversion parameters? Y/N (N) >>> ')
            if tmp.upper() == 'Y':
                self._parameters_identification()

        moved = True
        grabbed = True
        
        while True:
            if grabbed:
                if not self._gripper_ctrl.open():
                    continue
                grabbed = False
            if moved:
                if not self._move_to_detection_point():
                    continue
                moved = False
            input('Press to recognition >>> ')
            self.in_motion = False
            rects = self.que.get()
            self.in_motion = True
            rect = rects[random.randint(0, 100) % len(rects)]
            x, y, angle = self._rect_to_move_params(rect)
            print('target: x={:.2f}mm, y={:.2f}mm, anlge={:.2f}'.format(x, y, angle))
            moved = True
            ret = self._xarm_ctrl.set_joint(angle)
            if not ret:
                continue
            ret = self._xarm_ctrl.moveto(z=self._safe_z)
            if not ret:
                continue
            ret = self._xarm_ctrl.moveto(x=x, y=y, z=self._safe_z, relative=False)
            if not ret:
                continue
            ret = self._xarm_ctrl.moveto(x=x, y=y, z=self._grab_z, relative=False)
            if not ret:
                continue
            self._gripper_ctrl.close()
            grabbed = True
            ret = self._xarm_ctrl.moveto(z=self._safe_z, relative=False)
            if not ret:
                continue
            ret = self._xarm_ctrl.moveto(z=self._grab_z, relative=False)
            if not ret:
                continue
            ret = self._gripper_ctrl.open()
            grabbed = not ret
            self._xarm_ctrl.moveto(z=self._safe_z, relative=False)


class V4L2Camera(object):
    def __init__(self, index=0):
        self._cap = cv2.VideoCapture(index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 10.0)

    def get_frame(self):
        success, frame = self._cap.read()
        if success:
            return frame
        return None

def get_recognition_rect():
    
    khch=run()

    # rects.append(0)    
    return khch


if __name__ == '__main__':
    rospy.init_node('color_recognition_node', anonymous=False)
    dof = rospy.get_param('/xarm/DOF')

    rate = rospy.Rate(10.0)
    motion_que = queue.Queue(1)
    motion = MotionThread(motion_que, dof=dof, grab_z=195, safe_z=300, iden_z=200)
    motion.start()
    cnts = 0
    rects= get_recognition_rect()
    print(rects)
    final_rect=rects[0][0]
    final_rect.append(0)
    finalrects=[]
    finalrects.append(final_rect)
    print(finalrects)
    while not rospy.is_shutdown():
        
        rate.sleep()
        
        # cnts += 1
        # if cnts < 50:
        #     continue
        # if len(rects) == 0:
        #     continue
        if motion.in_motion or motion_que.qsize() != 0:
            continue
        motion_que.put(finalrects)
        