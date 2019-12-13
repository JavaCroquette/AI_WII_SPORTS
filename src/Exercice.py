import numpy as np
from Video import video
from Camera import camera
from threading import Thread
import utile
import screeninfo
import posenet
import argparse
import cv2
import math
import time
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

MIN = 0.01

MOT_DOUX = ["Tres Bien", "Bien", "Assez bien", "Courage"]

IMAGE = [cv2.imread('img/verygood.png'),
         cv2.imread('img/good.png'),
         cv2.imread('img/bad.png'),
         cv2.imread('img/courage.png')]


class exercice(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, sess, model_cfg, model_outputs, args, cheminV, cheminN):
        self.arret = True
        self.Camera_thread = camera(args, sess, model_cfg, model_outputs)
        self.Video_thread = video(MOT_DOUX, cheminV, cheminN)
        self.VideoPoint = None
        self.CameraPoint = None
        self.Camera = None
        self.data = None
        self.Video = None
        self.check = False
        self.i = len(IMAGE)-1
        self.graph = []
        self.fig = plt.figure(figsize=(8, 2), dpi=80)
        self.listSum = []
        self.sum = 0
        Thread.__init__(self)

    def AddCamera(self):
        if len(self.Camera_thread.ListPoint) != 0:
            self.Camera = self.Camera_thread.ListPoint[0][0]
            if self.Camera > MIN:
                self.CameraPoint = utile.Patron(
                    self.Camera_thread.ListPoint[0][1][0], False)
                self.Camera_thread.frame_count += 1
            del self.Camera_thread.ListPoint[0]
            self.check = True

    def AddVideo(self):
        self.Video = self.Video_thread.ListPoint[0][0]
        self.VideoPoint = self.Video_thread.ListPoint[0][1]
        del self.Video_thread.ListPoint[0]
        self.VideoPoint = utile.Patron(self.VideoPoint[:, 1], True)
        self.Video_thread.frame_count += 1

    def AddData(self):
        plt.cla()
        plt.ylim(0, 1)
        canvas = FigureCanvas(self.fig)
        plt.plot(range(0, len(self.listSum)), self.listSum)
        canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        self.Camera_thread.start()
        self.Video_thread.start()

        while self.Video_thread.frame_count < 1:
            if len(self.Camera_thread.ListPoint) != 0:
                self.AddVideo()

        cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            'Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for i in range(0, 90):
            Video = self.Video.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            if i < 30:
                text = "3"
            elif i < 60:
                text = "2"
            else:
                text = "1"

            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1+i/3 % 10, 24)[0]

            # get coords based on boundary
            textX = int((Video.shape[1] - textsize[0]) / 2)
            textY = int((Video.shape[0] + textsize[1]) / 2)

            # add text centered on image
            cv2.putText(Video, text, (textX, textY), font, 1+i/3 %
                        10, (0, 191, 255), 24)
            cv2.putText(Video, text, (textX, textY), font, 1+i/3 %
                        10, (0, 255, 255), 12)
            time.sleep(0.01)
            cv2.imshow('Video', Video)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.stopthread()
                break

        while self.arret:
            self.AddCamera()
            if len(self.Video_thread.ListPoint) != 0 and self.Camera is not None:
                if self.Camera > MIN:
                    self.AddVideo()
            if self.Video is not None:  # On check si il y en a deux
                if self.check:
                    print("Camera : " + str(self.Camera_thread.frame_count) +
                          " -- Video : " + str(self.Video_thread.frame_count), end="\r")
                    if self.Camera > MIN:
                        Comparatif = [5, 6, 7, 8, 9,
                                      10, 11, 12, 13, 14, 15, 16]
                        for p in Comparatif:
                            # Normalisation
                            self.sum += (utile.Distance(self.CameraPoint[p], self.VideoPoint[p])/(
                                sqrt(2)*len(Comparatif)))
                        if (self.Camera_thread.frame_count) % 10 == 0:
                            self.sum = self.sum / 10
                            self.listSum.append(self.sum)
                            if self.sum < 0.225:
                                self.i = 0
                            elif self.sum < 0.275:
                                self.i = 1
                            elif self.sum < 0.30:
                                self.i = 2
                            else:
                                self.i = 3
                            self.sum = 0
                        self.check = False
                else:
                    print("Camera : " + str(self.Camera_thread.frame_count) +
                          " == Video : " + str(self.Video_thread.frame_count), end="\r")
    #==============================================================================#
                if (self.Camera_thread.frame_count) % 10 == 0:
                    self.data = self.AddData()

                if self.data is not None:
                    self.Video[0:self.data.shape[0], self.Video.shape[1] -
                               self.data.shape[1]:self.Video.shape[1]] = self.data
                    cv2.putText(self.Video, str(
                        MOT_DOUX[self.i]), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                    cv2.putText(self.Video, str("Marge d'erreurs:")+str(round(self.listSum[len(
                        self.listSum)-1], 2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                    img = IMAGE[self.i]
                    self.Video[250:img.shape[0]+250, img.shape[1] +
                               100 - img.shape[1]:img.shape[1]+100] = img
    #==============================================================================#
                self.Video[self.Video.shape[0] -
                           500:self.Video.shape[0], 0:500] = [0, 0, 0]
                self.Video = utile.draw(
                    self.VideoPoint, self.Video, [0, 255, 255], True)
                if self.Camera > 0:
                    self.Video = utile.draw(
                        self.CameraPoint, self.Video, [255, 255, 0], False)
                else:  # Si la personne sort du cadre de la caméra
                    self.Video[:, :] = [100, 100, 100]
                    cv2.putText(self.Video, str("REVENEZ DEVANT LA CAMERA"),
                                (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)
    #==============================================================================#
                cv2.imshow('Video', self.Video)
    #==============================================================================#
            if self.Video_thread.Taille is not None:
                if self.Video_thread.Taille == self.Video_thread.frame_count:
                    self.stopthread()
                    break
    #==============================================================================#
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.stopthread()
                break

        b = 0
        for i in self.listSum:
            b = b + i
        if self.Video_thread.frame_count > 10:
            print(" !=! "+str(b / (int(self.Video_thread.frame_count/10))
                              * 100)+str(" %")+" !=! ")

        cv2.destroyWindow('Video')

    def stopthread(self):
        self.Video_thread.stopthread()
        self.Camera_thread.stopthread()
        self.arret = False
