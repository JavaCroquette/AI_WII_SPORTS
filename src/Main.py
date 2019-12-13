import numpy as np
from Exercice import exercice
import posenet
import argparse
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#==============================================================================#
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)  # > à 640
parser.add_argument('--cam_width', type=int, default=640)  # > à 480
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=1)  # laissé à 1
parser.add_argument('--file', type=str, default=None,
                    help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
#==============================================================================#


def main():
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    Exercice1 = exercice(sess, model_cfg, model_outputs, args,
                         './Exercice1/Exercice1.mp4', './Exercice1/video.npy')
    Exercice1.start()


if __name__ == "__main__":
    main()
