import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import posenet
import screeninfo
import random
from threading import Thread

class video(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self,MOT_DOUX,cheminVideo,cheminNpy):
        self.List = []
        self.frame_count = 0
        self.MOT_DOUX = MOT_DOUX
        self.cheminVideo = cheminVideo
        self.cheminNpy = cheminNpy
        Thread.__init__(self)

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        cap = cv2.VideoCapture(self.cheminVideo)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                self.List.append(frame.copy())
                self.frame_count += 1
                time.sleep(.025)
            else:
                break
        self.run()

    def stopthread(self):
        self.arret=True
