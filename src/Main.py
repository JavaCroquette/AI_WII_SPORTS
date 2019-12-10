import numpy as np
from Video import video
from Camera import camera
import random
import screeninfo
import posenet
import argparse
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)# ne peux pas être inférieur à 640
parser.add_argument('--cam_height', type=int, default=480)# ne peux pas être inférieur à 480
parser.add_argument('--scale_factor', type=float, default=1)# laissé à 1
parser.add_argument('--file', type=str, default=None,help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

MOT_DOUX = ["Tres Bien",
            "Bien",
            "Assez bien",
            "Courage"]
good = cv2.imread('img/good.png')
bad = cv2.imread('img/bad.png')
verygood = cv2.imread('img/verygood.png')
courage = cv2.imread('img/courage.png')
IMAGE = [verygood,good,bad,courage]

def Centremassev1(a,b,c,d):
    p = [a,b,c,d]
    n = len(p)
    x = 0
    y = 0
    for i in p:
        x = i[0]+x
        y = i[1]+y
    xG = x/n
    yG = y/n
    return [xG,yG]

def Patron(Liste,Video):
    if not len(Liste) == 0:
#TODO si quelqu'un arrive a faire sans les liste x)
        X = []
        Y = []
        for i in range(0,len(Liste)):
            X.append(Liste[i][0])
            Y.append(Liste[i][1])
        Xmax = max(X)
        Ymax = max(Y)
        Xmin = min(X)
        Ymin = min(Y)
        A = [Xmin,Ymax]
        B = [Xmax,Ymax]
        C = [Xmin,Ymin]
        D = [Xmax,Ymin]
        AB = math.sqrt((B[0] - A[0]) * (B[0] - A[0]) + (B[1] - A[1]) * (B[1] - A[1]))
        AC = math.sqrt((C[0] - A[0]) * (C[0] - A[0]) + (C[1] - A[1]) * (C[1] - A[1]))
        milieu = Centremassev1(A,B,C,D)
        for i in range(0,len(Liste)):
            Liste[i] -= milieu
            Liste[i][0] = Liste[i][0]/(1.05*AB)+0.5
            if Video:
                Liste[i][1] = Liste[i][1]/(2.1*AC)+0.5
            else:
                Liste[i][1] = 1-(Liste[i][1]/(2.1*AC)+0.5)
        return Liste

def draw_squeleton(image,keypoints,color):
    middle = (int((keypoints[6][0] - keypoints[5][0])/ 2 + keypoints[5][0]), int((keypoints[6][1] - keypoints[5][1])/ 2 + keypoints[5][1]) )
    cv2.line(image, (keypoints[5][0],keypoints[5][1]), (keypoints[6][0],keypoints[6][1]),color,2)
    cv2.line(image, (keypoints[5][0],keypoints[5][1]), (keypoints[7][0],keypoints[7][1]), color, 2)
    cv2.line(image, (keypoints[7][0],keypoints[7][1]), (keypoints[9][0],keypoints[9][1]), color, 2)
    cv2.line(image, (keypoints[6][0],keypoints[6][1]), (keypoints[8][0],keypoints[8][1]), color, 2)
    cv2.line(image, (keypoints[8][0], keypoints[8][1]), (keypoints[10][0], keypoints[10][1]), color, 2)
    cv2.line(image, (keypoints[5][0], keypoints[5][1]), (keypoints[11][0], keypoints[11][1]), color, 2)
    cv2.line(image, (keypoints[6][0], keypoints[6][1]), (keypoints[12][0], keypoints[12][1]), color, 2)
    cv2.line(image, (keypoints[11][0], keypoints[11][1]), (keypoints[13][0], keypoints[13][1]), color, 2)
    cv2.line(image, (keypoints[13][0], keypoints[13][1]), (keypoints[15][0], keypoints[15][1]), color, 2)
    cv2.line(image, (keypoints[12][0], keypoints[12][1]), (keypoints[14][0], keypoints[14][1]), color, 2)
    cv2.line(image, (keypoints[14][0], keypoints[14][1]), (keypoints[16][0], keypoints[16][1]), color, 2)
    cv2.line(image, (keypoints[11][0], keypoints[11][1]), (keypoints[12][0], keypoints[12][1]), color, 2)
    cv2.line(image, middle, (keypoints[0][0], keypoints[0][1]), color, 2)
    cv2.line(image, (keypoints[0][0], keypoints[0][1]), (keypoints[1][0], keypoints[1][1]), color, 2)
    cv2.line(image, (keypoints[0][0], keypoints[0][1]), (keypoints[2][0], keypoints[2][1]), color, 2)
    cv2.line(image, (keypoints[1][0], keypoints[1][1]), (keypoints[3][0], keypoints[3][1]), color, 2)
    cv2.line(image, (keypoints[2][0], keypoints[2][1]), (keypoints[4][0], keypoints[4][1]), color, 2)

def main():
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    Camera_thread = camera(args, sess, model_cfg, model_outputs)
    Video_thread = video(MOT_DOUX, './Exercice1/video2.mp4', './Exercice1/video.npy')

    Camera = None
    data = None
    Video = None
    VideoPoint = None
    CameraPoint = None
    rand = 0
    check = False
    i = 4
    graph = []
    fig = plt.figure(figsize=(8,2), dpi=80)
    listSum = []
    Camera_thread.start()
    Video_thread.start()

    while True:
        if len(Camera_thread.ListPoint) != 0:
            CameraPoint = Camera_thread.ListPoint[0][0]
            Camera = Camera_thread.ListPoint[0][1:4]
            del Camera_thread.ListPoint[0]
            pose_scores = Camera[0].copy()
            keypoint_scores = Camera[1].copy()
            keypoint_coords = Camera[2][0].copy()
            keypoint_coords = Patron(keypoint_coords,False)
            Camera_thread.frame_count += 1
            check = True

        if len(Video_thread.ListPoint) != 0 and Camera is not None:
            Video = Video_thread.ListPoint[0][0]
            VideoPoint = Video_thread.ListPoint[0][1]
            del Video_thread.ListPoint[0]
            VideoPoint = Patron(VideoPoint[:,1],True)
            Video_thread.frame_count += 1

        if Video is not None and Camera is not None and CameraPoint is not None and VideoPoint is not None:
            if check:
                print("Camera : " + str(Camera_thread.frame_count)+" -- Video : " + str(Video_thread.frame_count))
                widthV = Video.shape[1]
                heightV = Video.shape[0]
                widthC = args.cam_width
                heightC = args.cam_height
                sum = 0
                for p in range(0, len(CameraPoint)):
                    if keypoint_coords[p][0] is not None:
                        sum = sum + abs(keypoint_coords[p][0] - VideoPoint[p][0])
                    else:
                        sum = sum + 5
                    if keypoint_coords[p][1] is not None:
                        sum = sum + abs(keypoint_coords[p][1] - VideoPoint[p][1])
                    else:
                        sum = sum + 5
                listSum.append(sum)
                check = False

                if sum < 0.25:
                    i = 0
                elif sum < 0.5:
                    i = 1
                elif sum < 0.75:
                    i = 2
                else:
                    i = 3
            else:
                print("Camera : " + str(Camera_thread.frame_count)+" == Video : " + str(Video_thread.frame_count))
            cv2.putText(Video, str(MOT_DOUX[i]), (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            cv2.putText(Video, str(round(sum,2)), (100, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            if (Camera_thread.frame_count-1)%5 == 0:
                plt.cla()
                plt.ylim(0,5)
                canvas = FigureCanvas(fig)
                plt.plot(range(0,Camera_thread.frame_count), listSum)
                canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            Video[0:data.shape[0], Video.shape[1] - data.shape[1]:Video.shape[1]] = data


            img = IMAGE[i]
#==============================================================================#
            Video[250:img.shape[0]+250, img.shape[1]+100 -img.shape[1]:img.shape[1]+100] = img
#==============================================================================#
            Video[Video.shape[0]-500:Video.shape[0],0:500] = [0,0,0]
#==============================================================================#
            cv_keypoints = []
            Liste = []
            for y in range(0,len(VideoPoint)):
                cv_keypoints.append(cv2.KeyPoint(VideoPoint[y][1]*500, VideoPoint[y][0]*500+Video.shape[0]-500, 10))
                Liste.append([int(VideoPoint[y][1]*500),int(VideoPoint[y][0]*500)+Video.shape[0]-500])
            Video = cv2.drawKeypoints(Video, cv_keypoints, outImage=np.array([]), color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            draw_squeleton(Video,Liste,[0, 255, 255])
#==============================================================================#
            cv_keypoints = []
            Liste = []
            for y in range(0,len(keypoint_coords)):
                cv_keypoints.append(cv2.KeyPoint(keypoint_coords[y][1]*500, keypoint_coords[y][0]*500+Video.shape[0]-500, 10))
                Liste.append([int(keypoint_coords[y][1]*500), int(keypoint_coords[y][0]*500+Video.shape[0]-500)])
            Video = cv2.drawKeypoints(Video, cv_keypoints, outImage=np.array([]), color=(255, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            draw_squeleton(Video,Liste,[255, 255, 0])
#==============================================================================#

            cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Video', Video)

        if Video_thread.Taille is not None:
            if Video_thread.Taille == Video_thread.frame_count:
                Video_thread.stopthread()
                Camera_thread.stopthread()
                break

        if cv2.waitKey(25) & 0xFF == ord('q'):
            Video_thread.stopthread()
            Camera_thread.stopthread()
            break

    print("C'EST FINIIIIIIIII ===============")
    b = 0
    for i in listSum:
        b = b + i
    print(b)
    print(Video_thread.frame_count*5)
    print(" !=! "+str(b / (Video_thread.frame_count*4)*100)+str(" %")+" !=! ")

if __name__ == "__main__":
    main()
