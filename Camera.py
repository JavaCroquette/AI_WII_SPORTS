import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import posenet
import screeninfo
import random
import sys
from threading import Thread

class camera(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self,args,sess,model_cfg, model_outputs):
        self.List = []
        self.frame_count = 0
        Thread.__init__(self)
        self.cap = cv2.VideoCapture(args.cam_id)
        self.cap.set(3, args.cam_width)
        self.cap.set(4, args.cam_height)
        self.model_cfg = model_cfg;
        self.model_outputs = model_outputs
        self.output_stride = model_cfg['output_stride']
        self.args = args;
        self.sess = sess

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        while True:
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
                min_pose_score=0)

            keypoint_coords *= output_scale

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0, min_part_score=0)
            overlay_image = cv2.flip(overlay_image, 1)

            self.List.append(overlay_image.copy())
            self.frame_count += 1
            
    def stopthread(self):
        self.arret=True
