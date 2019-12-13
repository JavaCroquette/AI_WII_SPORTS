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
        self.ListPoint = []
        self.frame_count = 0
        self.MOT_DOUX = MOT_DOUX
        self.cheminVideo = cheminVideo
        self.cap = cv2.VideoCapture(self.cheminVideo)
        self.npy = np.load(cheminNpy, allow_pickle=True)
        self.Taille = len(self.npy)
        Thread.__init__(self)

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""


        if (self.cap.isOpened() == False):
            print("Error opening video stream or file")
        i = 0

        while(self.cap.isOpened() and self.arret == True):
            ret, frame = self.cap.read()
            if ret == True:
                self.Add(frame,self.npy[i])
                i += 1
            else:
                break
        self.stopthread()

    def Add(self,frame,tableau):
        self.ListPoint.append([frame.copy(),tableau])

    def stopthread(self):
        self.arret = False
