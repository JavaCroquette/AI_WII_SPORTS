import numpy as np
from Video import video
from Camera import camera
import os
import random
import screeninfo
import posenet
import argparse
import time
import cv2
import sharedmem
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# coucou
Afficher = True
Cordonner = False

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=320)
parser.add_argument('--cam_height', type=int, default=240)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None,
                    help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

MOT_DOUX = ["TES NULL",
            "ENTRAINE TOI PLUS",
            "TU PEUX MIEUX FAIRE",
            "ARRET TOUT DE SUITE",
            "IL FAUT MIEUX ARRETER",
            "RECOMMENCE",
            "HAA C'EST JEOF",
            "MERCI, MERCI, MAIS NON MERCI"]
rand = 0
dir_path = os.path.dirname(os.path.realpath(__file__))
video_path = dir_path+'/Exercice1/video.avi'
positions_path = dir_path+'/Exercice1/video.npy'


def exercise(display, resolution):
    with tf.Session() as sess:
        start = time.time()

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        Camera_thread = camera(args, sess, model_cfg, model_outputs)
        # Camera_thread = video(
        #    MOT_DOUX, './Exercice1/video.avi', './Exercice1/video.npy')
        Video_thread = video(
            MOT_DOUX, video_path, positions_path)

        Camera_thread.start()
        Video_thread.start()

        Video = None
        Camera = None
        VideoPoint = None
        CameraPoint = None
        rand = 0
        check = False
        sum = 0
        while True:
            if Camera_thread.arret == True or Video_thread.arret == True:
                Video_thread.c = False
                Camera_thread.c = False
                break
            if len(Camera_thread.List) != 0:
                Camera = Camera_thread.List[0]
                del Camera_thread.List[0]
            if len(Camera_thread.ListPoint) != 0:
                CameraPoint = Camera_thread.ListPoint[0]
                del Camera_thread.ListPoint[0]
            if len(Video_thread.List) != 0:
                Video = Video_thread.List[0]
                del Video_thread.List[0]
                check = True
                if Video_thread.frame_count % 100 == 0:
                    rand = random.randint(0, len(MOT_DOUX)-1)
            if len(Video_thread.ListPoint) != 0:
                VideoPoint = Video_thread.ListPoint[0]
                del Video_thread.ListPoint[0]
            if Video is not None and Camera is not None and CameraPoint is not None and VideoPoint is not None and check:
                widthV = Video.shape[1]
                heightV = Video.shape[0]  # keep original height
                dim = (widthV, heightV)
                widthC = Camera.shape[1]
                heightC = Camera.shape[0]
                # resize image
                Camera = cv2.resize(Camera, dim, interpolation=cv2.INTER_AREA)
                # Video[0:Camera.shape[0], Video.shape[1] - Camera.shape[1]:Video.shape[1]] = Camera
                Video = Video + Camera
                # Video = Camera
                sum = 0
                for p in range(0, len(CameraPoint)):
                    sum = sum + \
                        abs(CameraPoint[p][1][0]/heightC -
                            VideoPoint[p][1][0]/heightV)
                    sum = sum + \
                        abs(CameraPoint[p][1][1]/widthC -
                            VideoPoint[p][1][1]/widthV)
                    print(
                        str(abs(CameraPoint[p][1][0]/heightC - VideoPoint[p][1][0]/heightV)), end=" : ")
                    print(
                        str(abs(CameraPoint[p][1][1]/widthC - VideoPoint[p][1][1]/widthV)))
                print("===============")
                check = False
                """cv2.putText(Video, str(round(sum, 2)), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)

                cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(
                    'Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Video', Video)"""
                display = cv2.resize(
                    Video, (resolution[1], resolution[0]))
                print('Time running: %d...' % (time.time() - start))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                Video_thread.c = False
                Camera_thread.c = False
                break
