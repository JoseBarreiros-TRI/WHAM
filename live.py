import os
import sys
import time
import colorsys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict

import threading
import queue
import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger

from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
import pdb
from lib.vis.renderer import Renderer

def prepare_cfg():
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    return cfg

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        success=0
        while success == 0:
            success, frame = self.cap.read()

        self.im_shape=frame.shape
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
    def read(self):
        return 1,self.q.get()

    def isOpened(self):
        return self.cap.isOpened()

class WHAMController:
    def __init__(self,camera_port):

        self.cfg = prepare_cfg()
        self.network = build_network(self.cfg, build_body_model(self.cfg.DEVICE, self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN))
        self.network.eval()
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())
        self.slam = None

        # For webcam input:
        self.cap = cv2.VideoCapture(camera_port)
        self.fps = int(self.cap.get(5))
        self.width, self.height = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.setup_renderer(self.network.smpl)
        time.sleep(1)

    def setup_renderer(self, smpl):
        # to torch tensor
        tt = lambda x: torch.from_numpy(x).float().to(self.cfg.DEVICE)
        # create renderer with cliff focal length estimation
        focal_length = (self.width ** 2 + self.height ** 2) ** 0.5
        self.renderer = Renderer(self.width, self.height, focal_length, self.cfg.DEVICE, smpl.faces)

        # build default camera
        self.default_R, self.default_T = torch.eye(3), torch.zeros(3)

    def render_image(self, img, results):
        img = img[..., ::-1].copy()
        # render onto the input video
        self.renderer.create_camera(self.default_R, self.default_T)
        for _id, val in results.items():
            # render onto the image
            frame_i2 = np.where(val['frame_id'] == frame_i)[0]
            if len(frame_i2) == 0: continue
            frame_i2 = frame_i2[0]
            # pdb.set_trace()
            img = self.renderer.render_mesh(torch.from_numpy(val['verts_cam'][frame_i2]).to(cfg.DEVICE), img)
        return img

    def process_frame(self):
        # with self.pose as pose:
        if self.cap.isOpened():
            for i in range(100):
                success, image = self.cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('raw', cv2.flip(image, 1))
            cv2.waitKey(1)
            pdb.set_trace()

            # 2D detection and tracking
            self.detector.track(image, None)

            # SLAM
            if self.slam is not None:
                self.slam.track()

            tracking_results = self.detector.process(self.fps)

            if self.slam is not None:
                slam_results = self.slam.process()
            else:
                slam_results = np.zeros((1, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion

            tracking_results = self.extractor.run_on_image(image, tracking_results)

            # Build dataset
            dataset = CustomDataset(self.cfg, tracking_results, slam_results, self.width, self.height, self.fps)

            # run WHAM
            results = defaultdict(dict)
            for batch in dataset:
                if batch is None: break

                _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch

                # inference
                pred = self.network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)

                # Store results
                results[_id]['poses_body'] = pred['poses_body'].cpu().squeeze(0).numpy()
                results[_id]['poses_root_cam'] = pred['poses_root_cam'].cpu().squeeze(0).numpy()
                results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
                results[_id]['verts_cam'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
                results[_id]['poses_root_world'] = pred['poses_root_world'].cpu().squeeze(0).numpy()
                results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
                results[_id]['frame_id'] = frame_id

            # pdb.set_trace()

            render_img = self.render_image(image, results)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('WHAM results', cv2.flip(render_img, 1))
            cv2.waitKey(1)
            #pdb.set_trace()

if __name__ == '__main__':
    controller = WHAMController(0)
    for i in range(100):
        controller.process_frame()