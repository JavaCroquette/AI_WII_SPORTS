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
from PIL import ImageFont, ImageDraw, Image
import screeninfo
tf.disable_v2_behavior()

import win32api


MIN = 0.01

MOT_DOUX = ["Tres Bien", "Bien", "Assez bien", "Courage"]

IMAGE = [cv2.imread('img/verygood.png'),
         cv2.imread('img/good.png'),
         cv2.imread('img/bad.png'),
         cv2.imread('img/courage.png')]


class exercice(Thread):

    """Thread chargé simplement d'afficher une lettre dans la console."""

    def __init__(self, sess, model_cfg, model_outputs, args, cheminV, cheminN):
        screen = screeninfo.get_monitors()[0]
        self.width, self.height = screen.width, screen.height
        print(self.width)
        print(self.height)
        self.clique = False
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
        self.score = 0
        self.totalscore = []
        self.erreurpourcent = 0
        self.counttresbien = 0
        self.countbien = 0
        self.countpasmal = 0
        self.countcourage = 0
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
        self.Video = self.Video_thread.ListPoint[self.Video_thread.frame_count][0]
        if self.Video_thread.frame_count > 0:
            self.VideoPoint = self.Video_thread.ListPoint[self.Video_thread.frame_count-1][1]
            self.VideoPoint = utile.Patron(self.VideoPoint[:, 1], True)
        self.Video_thread.frame_count += 1

    def AddData(self,listSum,ymax):
        plt.cla()
        plt.ylim(0, ymax)
        plt.xlim(0,len(self.listSum)-1)
        canvas = FigureCanvas(self.fig)
        plt.plot(range(0, len(listSum)), listSum)
        canvas.draw()
        data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        return data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def on_click(self,event, x, y, p1, p2):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clique = True

    def Debut(self):
        for i in range(0, 90):
            Video = self.Video.copy()
            font = cv2.FONT_HERSHEY_DUPLEX
            if i < 30:
                text = "3"
            elif i < 60:
                text = "2"
            else:
                text = "1"

            #fontpath = "SuperMario256.ttf"
            #img_pil = Image.fromarray(Video)
            #draw = ImageDraw.Draw(img_pil)

            #font = ImageFont.truetype(fontpath, int(32*(i%30)/3)+(i%30))
            # get boundary of this text
            #textsize = draw.textsize(text,font)[0]
            # get coords based on boundary
            #textX = int((Video.shape[1] - textsize) / 2)
            #textY = int((Video.shape[0] - textsize) / 2)
            # add text centered on image
            #draw.text((textX, textY),  text, font = font, fill = (0, 165, 255, (i%30)*3))

            #font = ImageFont.truetype(fontpath, int(32*(i%30)/3))
            # get boundary of this text
            #textsize = draw.textsize(text,font)[0]
            # get coords based on boundary
            #textX = int((Video.shape[1] - textsize) / 2)
            #textY = int((Video.shape[0] - textsize) / 2)
            # add text centered on 8
            #draw.text((textX, textY),  text, font = font, fill = (0, 215, 255, (i%30)*3))
            #Video = np.array(img_pil)



            # get boundary of this text
            textsize = cv2.getTextSize(text, font, 1+i/3 % 10, 24)[0]

            # get coords based on boundary
            textX = int((Video.shape[1] - textsize[0]) / 2)
            textY = int((Video.shape[0] + textsize[1]) / 2)

            # add text centered on image
            cv2.putText(Video, text, (textX, textY), font, 1+i/3 %10, (0, 191, 255), 24)
            cv2.putText(Video, text, (textX, textY), font, 1+i/3 %10, (0, 255, 255), 12)
            cv2.imshow('Video', Video)
            if hasattr(self, 'height'):
                self.height, self.width = Video.shape[:2]
            if (cv2.waitKey(25) & 0xFF == ord('q')) or self.clique == True:
                self.stopthread()
                break

    def reset(self):
        self.Camera_thread.ListPoint[:] = []
        self.check = False
        self.Camera_thread.frame_count = 0

    def run(self):
        """Code à exécuter pendant l'exécution du thread."""
        self.Camera_thread.start()
        self.Video_thread.start()

        while self.Video_thread.frame_count < 1:
            if len(self.Camera_thread.ListPoint) != 0:
                self.AddVideo()

        cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Video', self.on_click)

        self.Debut()
        self.reset()

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
                            if self.sum < 0.3:
                                self.i = 0
                                self.score = 200
                                colortext = (0,252,124)
                                self.counttresbien += 1
                            elif self.sum < 0.35:
                                self.i = 1
                                self.score = 100
                                colortext = (47,255,173)
                                self.countbien += 1
                            elif self.sum < 0.4:
                                self.i = 2
                                self.score = 50
                                colortext = (0,215,255)
                                self.countpasmal +=1
                            else:
                                self.i = 3
                                self.score = 0
                                colortext = (0,140,255)
                                self.countcourage +=1
                            self.sum = 0
                            self.totalscore.append(self.score)
                        self.check = False
                else:
                    print("Camera : " + str(self.Camera_thread.frame_count) +
                          " == Video : " + str(self.Video_thread.frame_count), end="\r")
#==============================================================================#
                if (self.Camera_thread.frame_count) % 10 == 0:
                    self.data = self.AddData(self.listSum,1)

                if self.data is not None and len(self.listSum) != 0:
                    self.Video[0:self.data.shape[0], self.Video.shape[1] -
                               self.data.shape[1]:self.Video.shape[1]] = self.data
                    cv2.putText(self.Video, str(
                        MOT_DOUX[self.i]), (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                    cv2.putText(self.Video, str("Marge d'erreurs:")+str(round(self.listSum[len(
                        self.listSum)-1], 2)), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                    img = IMAGE[self.i]
                    self.Video[250:img.shape[0]+250, img.shape[1]+100 -img.shape[1]:img.shape[1]+100] = img
#==============================================================================#
                    cv2.putText(self.Video, str("+")+str(self.score), (400, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, colortext, 4)
                    cv2.putText(self.Video, str(sum(self.totalscore)), (1700, 250),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 4)
#==============================================================================#
                self.Video[self.Video.shape[0] -
                           500:self.Video.shape[0], 0:500] = [0, 0, 0]
                if self.VideoPoint is not None:
                    self.Video = utile.draw(
                        self.VideoPoint, self.Video, [0, 255, 255], True)

                if self.Camera is not None:
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
                    self.Video_thread.stopthread()
                    self.Camera_thread.stopthread()
                    break
    #==============================================================================#
            if (cv2.waitKey(25) & 0xFF == ord('q')) or self.clique == True:
                self.stopthread()
                break

        b = 0
        for i in self.listSum:
            b = b + i
        if self.Video_thread.frame_count > 10:
            self.erreurpourcent = (b / (int(self.Video_thread.frame_count/10))* 100)

        self.fin()
        self.stopthread()
        cv2.destroyWindow('Video')

    def fin(self):
        self.Video[:, :] = [200, 200, 200]
        if len(self.totalscore) == 0:
            self.totalscore.append(0)
        i = 0
        self.fig.patch.set_facecolor('#c8c8c8')
        self.fig.patch.set_alpha(0.5)
        score = 0
        while True:
            Image = self.Video.copy()
            if i < len(self.totalscore):
                score += self.totalscore[i]
                i += 1
            cv2.putText(Image, str("Score :")+str(score),(50, 700), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)
            self.data = self.AddData(self.totalscore[0:i],250)
            self.Video[0:self.data.shape[0], self.Video.shape[1] - self.data.shape[1]:self.Video.shape[1]] = self.data
            ##Note :
            if self.erreurpourcent < 10 :
                note = str("S")
                colornote = (0,215,255)
            elif self.erreurpourcent < 15:
                note = str("A")
                colornote = (0,69,255)
            elif self.erreurpourcent < 20:
                note= str("B")
                colornote = (0,140,255)
            elif self.erreurpourcent < 30:
                note = str("C")
                colornote = (0,255,255)
            elif self.erreurpourcent < 50:
                note = str("D")
                colornote = (50,205,50)
            elif self.erreurpourcent < 70:
                note = str("E")
                colornote = (50,205,154)
            elif self.erreurpourcent < 100:
                note = str("F")
                colornote = (209,206,0)
            cv2.putText(Image, str("Note : "),(1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 12)
            cv2.putText(Image, note,(1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 20, (0,0,0), 25)
            cv2.putText(Image, note,(1410, 900), cv2.FONT_HERSHEY_SIMPLEX, 20, colornote, 25)

            if i < self.countcourage:
                cv2.putText(Image, str("Courage : ")+str(i),(50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,140,255), 4)
                cv2.putText(Image, str("Bien : ")+str(0),(1050, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (47,255,173), 4)
                cv2.putText(Image, str("Pas mal : ")+str(0),(500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,215,255), 4)
                cv2.putText(Image, str("Tres bien : ")+str(0),(1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,252,124), 4)
            else:
                cv2.putText(Image, str("Courage : ")+str(self.countcourage),(50, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,140,255), 4)
                if i-self.countcourage < self.countpasmal:
                    cv2.putText(Image, str("Pas mal : ")+str(i-self.countcourage),(500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,215,255), 4)
                    cv2.putText(Image, str("Bien : ")+str(0),(1050, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (47,255,173), 6)
                    cv2.putText(Image, str("Tres bien : ")+str(0),(1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,252,124), 4)
                else:
                    cv2.putText(Image, str("Pas mal : ")+str(self.countpasmal),(500, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,215,255), 4)
                    if i-self.countpasmal-self.countcourage < self.countbien:
                        cv2.putText(Image, str("Bien : ")+str(i-self.countpasmal-self.countcourage),(1050, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (47,255,173), 4)
                        cv2.putText(Image, str("Tres bien : ")+str(0),(1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,252,124), 4)
                    else:
                        cv2.putText(Image, str("Bien : ")+str(self.countbien),(1050, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (47,255,173), 4)
                        if i-self.countbien-self.countcourage-self.countpasmal < self.counttresbien:
                            cv2.putText(Image, str("Tres Bien : ")+str(i-self.countbien-self.countcourage-self.countpasmal),(1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,252,124), 4)
                        else:
                            cv2.putText(Image, str("Tres Bien : ")+str(self.countbien),(1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,252,124), 4)
            time.sleep(.05)

            #cv2.putText(Image, str("Bien : ")+str(self.countbien),(950, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (47,255,173), 6)
            #cv2.putText(Image, str("Pas mal : ")+str(self.countpasmal),(500, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,215,255), 6)
            #cv2.putText(Image, str("Courage : ")+str(self.countcourage),(50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,140,255), 6)
            #cv2.putText(Image, str("Tres bien : ")+str(i),(1400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,252,124), 6)
            cv2.putText(Image, str("Cliquez pour retourner au menu"),(50, 1000), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)
            cv2.imshow('Video', Image)

            if (cv2.waitKey(25) & 0xFF == ord('q')) or self.clique == True:
                self.stopthread()
                break

    def stopthread(self):
        self.Video_thread.stopthread()
        self.Camera_thread.stopthread()
        self.arret = False
        self.clique = True
