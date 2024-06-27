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
    def __init__(self, name, video_batch_size=1):
        self.cap = cv2.VideoCapture(name)
        success=0
        while success == 0:
            success, frame = self.cap.read()

        self.video_batch_size = video_batch_size
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
        return 1, [self.q.get() for _ in range(self.video_batch_size)]

    def isOpened(self):
        return self.cap.isOpened()

class WHAMController:
    def __init__(self, camera_port, video_batch_size):

        self.cfg = prepare_cfg()
        self.network = build_network(self.cfg, build_body_model(self.cfg.DEVICE, self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN))
        self.network.eval()
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())

        # For webcam input:
        self.cap = VideoCapture(camera_port, video_batch_size)

        # self.width, self.height, _ = self.cap.im_shape
        self.height, self.width, _ = self.cap.im_shape
        self.fps = 30
        self.setup_renderer(self.network.smpl)

    def setup_renderer(self, smpl):
        # create renderer with cliff focal length estimation
        focal_length = (self.width ** 2 + self.height ** 2) ** 0.5
        self.renderer = Renderer(self.width, self.height, focal_length, self.cfg.DEVICE, smpl.faces)

        # build default camera
        self.default_R, self.default_T = torch.eye(3), torch.zeros(3)

    @torch.no_grad()
    def preprocessing(self, video_batch):
        length = len(video_batch)

        # t0 = time.time()
        for img in video_batch:
            # 2D detection and tracking
            self.detector.track(img, self.fps, length)
        # print("detector track: ", time.time()-t0)
        # t0 = time.time()
        tracking_results = self.detector.process(self.fps)
        # print("detector process: ", time.time()-t0)

        slam_results = np.zeros((length, 7))
        slam_results[:, 3] = 1.0    # Unit quaternion

        # Extract image features
        # TODO: Merge this into the previous while loop with an online bbox smoothing.
        # t0 = time.time()
        tracking_results = self.extractor.run_on_video_batch(video_batch, tracking_results)
        # print("extractor: ", time.time()-t0)
        return tracking_results, slam_results

    @torch.no_grad()
    def wham_inference(self, tracking_results, slam_results):
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

        return results

    def render_video_batch(self, video_batch, results):
        frame_i = 0
        # run rendering
        for org_img in video_batch:
            img = org_img[..., ::-1].copy()

            # render onto the input video
            self.renderer.create_camera(self.default_R, self.default_T)
            for _id, val in results.items():
                # render onto the image
                frame_i2 = np.where(val['frame_id'] == frame_i)[0]
                if len(frame_i2) == 0: continue
                frame_i2 = frame_i2[0]

                img = self.renderer.render_mesh(torch.from_numpy(val['verts_cam'][frame_i2]).to(self.cfg.DEVICE), img)
            cv2.imshow('render', img)
            cv2.waitKey(1)
            frame_i += 1

    def init_tracking(self):
        if self.cap.isOpened():
            success, image_batch = self.cap.read()
            self.detector.initialize_tracking()
            self.tracking_results, self.slam_results = self.preprocessing(image_batch)

    def process_video_batch(self):
        # with self.pose as pose:
        if self.cap.isOpened():
            success, image_batch = self.cap.read()
            # for image in image_batch:
            #     cv2.imshow('raw', image)
            #     cv2.waitKey(1)
            t0 = time.time()
            self.detector.initialize_tracking()
            tracking_results, slam_results = self.preprocessing(image_batch)
            print("detector: ", time.time()-t0)
            if tracking_results == {}:
                print("empty tracking_results")

            t0 = time.time()
            results = self.wham_inference(tracking_results, slam_results)
            print("wham: ", time.time()-t0)
            if results == {}:
                print("empty results")
            # # pdb.set_trace()
            t0 = time.time()
            self.render_video_batch(image_batch, results)
            print("render: ", time.time()-t0)
            # pdb.set_trace()

if __name__ == '__main__':
    controller = WHAMController(0, 40)
    # controller.init_tracking()
    for i in range(100):
        controller.process_video_batch()