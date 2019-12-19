import numpy as np
from video import video
from camera import camera
import random
import screeninfo
import posenet
import argparse
import cv2
import math
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
################################################################################
# Liste des Indice :
#
# 0 - nose
# 1 - left eye
# 2 - right eye
# 3 - left ear
# 4 - right ear
# 5 - left shoulder
# 6 - right shoulder
# 7 - left elbow
# 8 - right elbow
# 9 - left wrist
# 10 - right wrist
# 11 - left hip
# 12 - right hip
# 13 - left knee
# 14 - right knee
# 15 - left ankle
# 16 - right ankle
################################################################################

Comparatif = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

################################################################################
#       Desine les point d'un corps humain détecté par la camré                #
################################################################################


def draw(Point, Video, color, postion):
    cv_keypoints = []
    Liste = []
    for y in Comparatif:
        if postion:
            cv_keypoints.append(cv2.KeyPoint(
                Point[y][1]*250, Point[y][0]*500+Video.shape[0]-500, 5))
            Liste.append(
                [int(Point[y][1]*250), int(Point[y][0]*500+Video.shape[0]-500)])
        else:
            cv_keypoints.append(cv2.KeyPoint(
                250+Point[y][1]*250, Point[y][0]*500+Video.shape[0]-500, 5))
            Liste.append([int(250+Point[y][1]*250),
                          int(Point[y][0]*500+Video.shape[0]-500)])
    draw_squeleton(Video, Liste, color)
    Video = cv2.drawKeypoints(Video, cv_keypoints, outImage=np.array([]), color=(
        color[0], color[1], color[2]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return Video

################################################################################
#               Permet de designer le squelette d'une personne                 #
################################################################################


def draw_squeleton(image, points, color):
    i = len(points) - 17
    Liste = [[5+i, 6+i], [5+i, 7+i], [7+i, 9+i], [6+i, 8+i], [8+i, 10+i], [5+i, 11+i],
             [6+i, 12+i], [11+i, 13+i], [13+i, 15+i], [12+i, 14+i], [14+i, 16+i],
             [11+i, 12+i]]
    for i in Liste:
        cv2.line(image, (points[i[0]][0], points[i[0]][1]),
                 (points[i[1]][0], points[i[1]][1]), color, 2)

################################################################################
#       Definie les 4 point du rectangle d'une personne et normalise           #
################################################################################


def Patron(Liste, Video):
    if not len(Liste) == 0:
        # TODO si quelqu'un arrive a faire sans les liste x)
        X = []
        Y = []
        for i in Comparatif:
            X.append(Liste[i][0])
            Y.append(Liste[i][1])
        Xmax = max(X)
        Ymax = max(Y)
        Xmin = min(X)
        Ymin = min(Y)
        A = [Xmin, Ymax]
        B = [Xmax, Ymax]
        C = [Xmin, Ymin]
        D = [Xmax, Ymin]
        AB = Distance(A, B)
        AC = Distance(A, C)
        for i in range(0, len(Liste)):
            Liste[i] -= C
            Liste[i][0] = Liste[i][0]/(AB)
            if Video:
                Liste[i][1] = Liste[i][1]/(AC)
            else:
                Liste[i][1] = 1-(Liste[i][1]/(AC))
        return Liste


################################################################################
#                                      Carré                                   #
################################################################################
def Sqr(a):
    return a*a

################################################################################
#                               Distance de 2 points                           #
################################################################################


def Distance(A, B):
    return sqrt(Sqr(B[1]-A[1])+Sqr(B[0]-A[0]))
