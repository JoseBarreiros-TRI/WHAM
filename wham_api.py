import os
import sys
import time
import colorsys
import argparse
import os.path as osp
from glob import glob
from collections import defaultdict
import datetime
import cv2
import torch
import joblib
import imageio
import numpy as np
from smplx import SMPL
from loguru import logger
from lib.utils.transforms import matrix_to_axis_angle
from configs.config import get_cfg_defaults
from lib.data._custom import CustomDataset
from lib.models import build_network, build_body_model
from lib.models.preproc.detector import DetectionModel
from lib.models.preproc.extractor import FeatureExtractor
import pdb

try:
    from lib.models.preproc.slam import SLAMModel
    _run_global = False
except:
    logger.info('DPVO is not properly installed. Only estimate in local coordinates !')
    _run_global = False

def prepare_cfg():
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/yamls/demo.yaml')
    return cfg

def load_video(video):
    cap = cv2.VideoCapture(video)
    assert cap.isOpened(), f'Faild to load video file {video}'
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    return cap, fps, length, width, height


class WHAM_API(object):
    def __init__(self):
        self.cfg = prepare_cfg()
        self.network = build_network(self.cfg, build_body_model(self.cfg.DEVICE, self.cfg.TRAIN.BATCH_SIZE * self.cfg.DATASET.SEQLEN))
        self.network.eval()
        self.detector = DetectionModel(self.cfg.DEVICE.lower())
        self.extractor = FeatureExtractor(self.cfg.DEVICE.lower())
        self.slam = None
        # pdb.set_trace()

    @torch.no_grad()
    def preprocessing(self, video, cap, fps, length, output_dir):
        if not (osp.exists(osp.join(output_dir, 'tracking_results.pth')) and
                osp.exists(osp.join(output_dir, 'slam_results.pth'))):
            while (cap.isOpened()):
                flag, img = cap.read()
                if not flag: break

                # 2D detection and tracking
                self.detector.track(img, fps, length)

                # SLAM
                if self.slam is not None:
                    self.slam.track()

            tracking_results = self.detector.process(fps)

            if self.slam is not None:
                slam_results = self.slam.process()
            else:
                slam_results = np.zeros((length, 7))
                slam_results[:, 3] = 1.0    # Unit quaternion

            # Extract image features
            # TODO: Merge this into the previous while loop with an online bbox smoothing.
            tracking_results = self.extractor.run(video, tracking_results)
            # Save the processed data
            joblib.dump(tracking_results, osp.join(output_dir, 'tracking_results.pth'))
            joblib.dump(slam_results, osp.join(output_dir, 'slam_results.pth'))
            logger.info(f'Save processed data at {output_dir}')

        # If the processed data already exists, load the processed data
        else:
            tracking_results = joblib.load(osp.join(output_dir, 'tracking_results.pth'))
            slam_results = joblib.load(osp.join(output_dir, 'slam_results.pth'))
            logger.info(f'Already processed data exists at {output_dir} ! Load the data .')

        return tracking_results, slam_results

    @torch.no_grad()
    def wham_inference(self, tracking_results, slam_results, width, height, fps, output_dir):
        # Build dataset
        dataset = CustomDataset(self.cfg, tracking_results, slam_results, width, height, fps)

        # run WHAM
        results = defaultdict(dict)
        for batch in dataset:
            if batch is None: break

            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch

            # inference
            pred = self.network(x, inits, features, mask=mask, init_root=init_root, cam_angvel=cam_angvel, return_y_up=True, **kwargs)
            # pdb.set_trace()
            # Store results
            results[_id]['poses_body'] = pred['poses_body'].cpu().squeeze(0).numpy()
            results[_id]['pose_body_aa'] = matrix_to_axis_angle(pred['poses_body']).cpu().numpy()


            results[_id]['poses_root_cam'] = pred['poses_root_cam'].cpu().squeeze(0).numpy()
            results[_id]['poses_root_cam_aa'] = matrix_to_axis_angle(pred['poses_root_cam']).cpu().numpy().reshape(-1, 3)

            results[_id]['betas'] = pred['betas'].cpu().squeeze(0).numpy()
            results[_id]['verts_cam'] = (pred['verts_cam'] + pred['trans_cam'].unsqueeze(1)).cpu().numpy()
            results[_id]['poses_root_world'] = pred['poses_root_world'].cpu().squeeze(0).numpy()
            results[_id]['poses_root_world_aa'] = matrix_to_axis_angle(pred['poses_root_world']).cpu().numpy().reshape(-1, 3)

            results[_id]['trans_world'] = pred['trans_world'].cpu().squeeze(0).numpy()
            results[_id]['frame_id'] = frame_id
            # pdb.set_trace()
        joblib.dump(results, osp.join(output_dir, 'wham_results.pth'))
        return results

    @torch.no_grad()
    def __call__(self, video, output_dir='output/demo', calib=None, run_global=True, visualize=False):
        # load video information
        cap, fps, length, width, height = load_video(video)
        os.makedirs(output_dir, exist_ok=True)

        # Whether or not estimating motion in global coordinates
        run_global = run_global and _run_global
        if run_global: self.slam = SLAMModel(video, output_dir, width, height, calib)

        # preprocessing to get detection, tracking, slam results and image features from video input
        # t0 = time.time()
        tracking_results, slam_results = self.preprocessing(video, cap, fps, length, output_dir)
        # tf = time.time()-t0
        # print(f"preprocessing took {tf}[sec]; {tf/length} spf")

        # WHAM forward inference to get the results
        # t0 = time.time()
        results = self.wham_inference(tracking_results, slam_results, width, height, fps, output_dir)
        # tf = time.time()-t0
        # print(f"inference took {tf}[sec]; {tf/length} spf")

        # Visualize
        if visualize:
            from lib.vis.run_vis import run_vis_on_demo
            run_vis_on_demo(self.cfg, video, results, output_dir, self.network.smpl, vis_global=run_global)

        return results, tracking_results, slam_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video_path",
        help="path to the logs directory.",
        required=True,
        )
    parser.add_argument(
        "--output_path",
        help="path to the logs directory.",
        )
    args = parser.parse_args()

    if not args.output_path:
        output_path = (os.path.dirname(args.video_path) +"/"+
                       os.path.splitext(os.path.basename(args.video_path))[0]+
                    #    datetime.now().strftime("%Y%m%d_%H_%M") +
                       "/")
    else:
        output_path = args.output_path
    # pdb.set_trace()
    wham_model = WHAM_API()
    results, tracking_results, slam_results = wham_model(
        args.video_path,
        run_global=False,
        visualize=True,
        output_dir= output_path,
        )
