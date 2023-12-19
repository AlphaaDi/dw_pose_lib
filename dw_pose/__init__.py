# https://github.com/IDEA-Research/DWPose
# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2
from munch import Munch
import numpy as np
import torch
from controlnet_aux.util import HWC3, resize_image
from PIL import Image
from pathlib import Path

from . import util
from .wholebody import Wholebody


def draw_pose(pose, H, W):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


class DWposeDetector:
    def __init__(self, dir_path):
        self.dir_path = Path(dir_path)

    def to(self, device):
        self.pose_estimation = Wholebody(self.dir_path, device)
        return self

    def __call__(self, input_image, detect_resolution=512, image_resolution=512, output_type="pil", **kwargs):
        input_image = cv2.cvtColor(np.array(input_image, dtype=np.uint8), cv2.COLOR_RGB2BGR)

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)
        H, W, C = input_image.shape
        with torch.no_grad():
            candidate, subset = self.pose_estimation(input_image)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands = np.vstack([hands, candidate[:,113:]])

            bodies = dict(candidate=body, subset=score)
            pose = dict(bodies=bodies, hands=hands, faces=faces)

            detected_map = draw_pose(pose, H, W)
            detected_map = HWC3(detected_map)

            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            if output_type == "pil":
                detected_map = Image.fromarray(detected_map)

            return detected_map, pose

        
class DWposeDetectorInference:
    def __init__(self, dir_path, device):
        self.dwprocessor = DWposeDetector(dir_path)
        self.dwprocessor.to(device)

    def to(self, device):
        self.dwprocessor.to(device)
        return self
        
    def procces_np_image(self, img):
        with torch.no_grad():
            input_image = HWC3(img)
            detected_map, pose = self.dwprocessor(resize_image(input_image, 1024))
            eps = 0.01

        H, W, C = input_image.shape

        bodies = pose['bodies']
        faces = pose['faces']
        hands = pose['hands']
        candidate = bodies['candidate']
        subset = bodies['subset']

        body_dots = []
        for i in range(18):
            for n in range(len(subset)):
                index = int(subset[n][i])
                if index == -1:
                    body_dots.append(None)
                    continue
                x, y = candidate[index][0:2]
                x = int(x * W)
                y = int(y * H)
                body_dots.append((y,x))

        hands_dots = []
        for peaks in hands:
            peaks = np.array(peaks)
            for keyponit in peaks:
                x, y = keyponit
                x = int(x * W)
                y = int(y * H)

                if x > eps and y > eps:
                    hands_dots.append((y,x))
                else:
                    hands_dots.append(None)

        faces_dots = []
        for lmks in faces:
            lmks = np.array(lmks)
            for lmk in lmks:
                x, y = lmk
                x = int(x * W)
                y = int(y * H)
                if x > eps and y > eps:
                    faces_dots.append((y,x))
                else:
                    faces_dots.append(None)

        points = Munch(
            face_center = body_dots[0],
            up_center = body_dots[1],
            up_left = body_dots[2],
            lefter_elbow = body_dots[3],
            lefter_hand = body_dots[4],
            up_right = body_dots[5],
            righter_elbow = body_dots[6],
            righter_hand = body_dots[7],
            bottom_left = body_dots[8],
            lefter_knee = body_dots[9],
            lefter_feet = body_dots[10],
            bottom_right = body_dots[11],
            righter_knee = body_dots[12],
            righter_feet = body_dots[13],
            lefter_eye = body_dots[14],
            righter_eye = body_dots[15],
            lefter_ear = body_dots[16],
            righter_ear = body_dots[17],
            faces_dots = faces_dots,
            hands_dots = hands_dots,
        )
        return points