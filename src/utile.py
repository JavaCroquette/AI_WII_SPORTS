import numpy as np
from Video import video
from Camera import camera
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

def draw(Point,Video,color,postion):
    cv_keypoints = []
    Liste = []
    for y in range(0, len(Point)):
        if postion:
            cv_keypoints.append(cv2.KeyPoint( Point[y][1]*250, Point[y][0]*500+Video.shape[0]-500, 5))
            Liste.append( [int(Point[y][1]*250), int(Point[y][0]*500+Video.shape[0]-500)])
        else:
            cv_keypoints.append(cv2.KeyPoint( 250+Point[y][1]*250, Point[y][0]*500+Video.shape[0]-500, 5))
            Liste.append( [int(250+Point[y][1]*250), int(Point[y][0]*500+Video.shape[0]-500)])
    draw_squeleton(Video, Liste, color)
    Video = cv2.drawKeypoints(Video, cv_keypoints, outImage=np.array([]), color=(color[0],color[1],color[2]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return Video

def draw_squeleton(image, keypoints, color):
    middle = (int((keypoints[6][0] - keypoints[5][0]) / 2 + keypoints[5][0]),
              int((keypoints[6][1] - keypoints[5][1]) / 2 + keypoints[5][1]))
    cv2.line(image, (keypoints[5][0], keypoints[5][1]),
             (keypoints[6][0], keypoints[6][1]), color, 2)
    cv2.line(image, (keypoints[5][0], keypoints[5][1]),
             (keypoints[7][0], keypoints[7][1]), color, 2)
    cv2.line(image, (keypoints[7][0], keypoints[7][1]),
             (keypoints[9][0], keypoints[9][1]), color, 2)
    cv2.line(image, (keypoints[6][0], keypoints[6][1]),
             (keypoints[8][0], keypoints[8][1]), color, 2)
    cv2.line(image, (keypoints[8][0], keypoints[8][1]),
             (keypoints[10][0], keypoints[10][1]), color, 2)
    cv2.line(image, (keypoints[5][0], keypoints[5][1]),
             (keypoints[11][0], keypoints[11][1]), color, 2)
    cv2.line(image, (keypoints[6][0], keypoints[6][1]),
             (keypoints[12][0], keypoints[12][1]), color, 2)
    cv2.line(image, (keypoints[11][0], keypoints[11][1]),
             (keypoints[13][0], keypoints[13][1]), color, 2)
    cv2.line(image, (keypoints[13][0], keypoints[13][1]),
             (keypoints[15][0], keypoints[15][1]), color, 2)
    cv2.line(image, (keypoints[12][0], keypoints[12][1]),
             (keypoints[14][0], keypoints[14][1]), color, 2)
    cv2.line(image, (keypoints[14][0], keypoints[14][1]),
             (keypoints[16][0], keypoints[16][1]), color, 2)
    cv2.line(image, (keypoints[11][0], keypoints[11][1]),
             (keypoints[12][0], keypoints[12][1]), color, 2)
    cv2.line(image, middle, (keypoints[0][0], keypoints[0][1]), color, 2)
    cv2.line(image, (keypoints[0][0], keypoints[0][1]),
             (keypoints[1][0], keypoints[1][1]), color, 2)
    cv2.line(image, (keypoints[0][0], keypoints[0][1]),
             (keypoints[2][0], keypoints[2][1]), color, 2)
    cv2.line(image, (keypoints[1][0], keypoints[1][1]),
             (keypoints[3][0], keypoints[3][1]), color, 2)
    cv2.line(image, (keypoints[2][0], keypoints[2][1]),
             (keypoints[4][0], keypoints[4][1]), color, 2)

def Patron(Liste, Video):
    if not len(Liste) == 0:
        # TODO si quelqu'un arrive a faire sans les liste x)
        X = []
        Y = []
        for i in range(0, len(Liste)):
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
        AB = Distance(A,B)
        AC = Distance(A,C)
        for i in range(0, len(Liste)):
            Liste[i] -= C
            Liste[i][0] = Liste[i][0]/(AB)
            if Video:
                Liste[i][1] = Liste[i][1]/(AC)
            else:
                Liste[i][1] = 1-(Liste[i][1]/(AC))
        return Liste

def Sqr(a):
    return a*a

def Distance(A,B):
    return sqrt(Sqr(B[1]-A[1])+Sqr(B[0]-A[0]))
