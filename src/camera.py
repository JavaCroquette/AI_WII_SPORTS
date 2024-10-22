import numpy as np
from threading import Thread
import sys
import random
import screeninfo
import posenet
import argparse
import time
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class camera(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, args, sess, model_cfg, model_outputs):
        self.ListPoint = []
        self.arret = True
        self.frame_count = 0
        self.cap = cv2.VideoCapture(args.cam_id)
        self.cap.set(3, args.cam_width)
        self.cap.set(4, args.cam_height)
        self.model_cfg = model_cfg
        self.model_outputs = model_outputs
        self.output_stride = model_cfg['output_stride']
        self.args = args
        self.sess = sess
        self.hotpoints = ['leftWrist', 'rightWrist', 'leftShoulder',
                          'rightShoulder', 'leftKnee', 'rightKnee']
        Thread.__init__(self)

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        while self.arret == True:
            input_image, display_image, output_scale = posenet.read_cap(
                self.cap, scale_factor=self.args.scale_factor, output_stride=self.output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run(
                self.model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=self.output_stride,
                max_pose_detections=1,
                min_pose_score=0.2)

            self.Add(pose_scores, keypoint_coords)

    def Add(self, pose_scores, keypoint_coords):
        self.ListPoint.append([sum(pose_scores), keypoint_coords])

    def stopthread(self):
        self.arret = False
