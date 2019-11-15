import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import posenet
import screeninfo
import random
from threading import Thread
import numpy as np

class video(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self,MOT_DOUX,cheminVideo,cheminNpy):
        self.c = True;
        self.List = []
        self.ListPoint = []
        self.frame_count = 0
        self.MOT_DOUX = MOT_DOUX
        self.cheminVideo = cheminVideo
        self.npy = np.load(cheminNpy,allow_pickle=True)
        Thread.__init__(self)

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        cap = cv2.VideoCapture(self.cheminVideo)

        if (cap.isOpened()== False):
            print("Error opening video stream or file")

        while(cap.isOpened() and self.c == True):
            ret, frame = cap.read()
            if ret == True:
                self.List.append(frame.copy())
                self.ListPoint.append(self.npy[self.frame_count])
                self.frame_count += 1
                time.sleep(.025)
            else:
                break
        if self.c:
            self.run()
        else:
            self.stopthread()

    def stopthread(self):
        self.arret=True
