import numpy as np
from threading import Thread
import random
import screeninfo
import posenet
import argparse
import time
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class video(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, MOT_DOUX, cheminVideo, cheminNpy):
        self.arret = True
        self.List = []
        self.ListPoint = []
        self.Taille = None
        self.frame_count = 0
        self.MOT_DOUX = MOT_DOUX
        self.cheminVideo = cheminVideo
        self.npy = np.load(cheminNpy, allow_pickle=True)
        Thread.__init__(self)

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        cap = cv2.VideoCapture(self.cheminVideo)
        self.Taille = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        while(cap.isOpened() and self.arret == True):
            ret, frame = cap.read()
            if ret == True:
                self.List.append(frame.copy())
                self.ListPoint.append(self.npy[self.frame_count])
            else:
                break
        self.stopthread()

    def stopthread(self):
        self.arret = False
