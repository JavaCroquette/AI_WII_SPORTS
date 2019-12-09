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
parser.add_argument('--cam_id', type=int, default=1)
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

def AffichePoint(Liste):
    if not len(Liste) == 0:
        EpauleA = Liste[5][1]
        EpauleB = Liste[6][1]
        poigneE = Liste[7][1]
        poigneF = Liste[8][1]
        GenouC = Liste[11][1]
        GenouD = Liste[12][1]
        AB = math.sqrt((EpauleB[0] - EpauleA[0]) * (EpauleB[0] - EpauleA[0]) + (EpauleB[1] - EpauleA[1]) * (EpauleB[1] - EpauleA[1]))
        AC = math.sqrt((GenouC[0] - EpauleA[0]) * (GenouC[0] - EpauleA[0]) + (GenouC[1] - EpauleA[1]) * (GenouC[1] - EpauleA[1]))
        milieu = Centremassev1(EpauleA,EpauleB,GenouC,GenouD)
        for i in range(0,len(Liste)):
            Liste[i,1] -= milieu
            Liste[i][1][0] = Liste[i][1][0]/(8*AB)+0.5
            Liste[i][1][1] = Liste[i][1][1]/(4*AC)+0.5
        print(Liste[:,1])
        return Liste[:,1]

def AffichePointCamera(Liste):
    print(Liste[0])
    if not len(Liste) == 0:
        EpauleA = Liste[5]
        EpauleB = Liste[6]
        poigneE = Liste[7]
        poigneF = Liste[8]
        GenouC = Liste[11]
        GenouD = Liste[12]
        AB = math.sqrt((EpauleB[0] - EpauleA[0]) * (EpauleB[0] - EpauleA[0]) + (EpauleB[1] - EpauleA[1]) * (EpauleB[1] - EpauleA[1]))
        AC = math.sqrt((GenouC[0] - EpauleA[0]) * (GenouC[0] - EpauleA[0]) + (GenouC[1] - EpauleA[1]) * (GenouC[1] - EpauleA[1]))
        milieu = Centremassev1(EpauleA,EpauleB,GenouC,GenouD)
        for i in range(0,len(Liste)):
            Liste[i] -= milieu
            Liste[i][0] = Liste[i][0]/(8*AB)+0.5
            Liste[i][1] = 1-(Liste[i][1]/(4*AC)+0.5)
        print(Liste)
        return Liste

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
            Camera_thread.frame_count += 1
            check = True

        if len(Video_thread.ListPoint) != 0 and Camera is not None:
            Video = Video_thread.ListPoint[0][0]
            VideoPoint = Video_thread.ListPoint[0][1]
            del Video_thread.ListPoint[0]
            print(VideoPoint)
            VideoPoint = AffichePoint(VideoPoint)
            Video_thread.frame_count += 1
            print(VideoPoint)

        if Video is not None and Camera is not None and CameraPoint is not None and VideoPoint is not None:
            if check:
                print("Camera : " + str(Camera_thread.frame_count)+" -- Video : " + str(Video_thread.frame_count))
                widthV = Video.shape[1]
                heightV = Video.shape[0]
                widthC = args.cam_width
                heightC = args.cam_height
                pose_scores = Camera[0].copy()
                keypoint_scores = Camera[1].copy()
                keypoint_coords = Camera[2][0].copy()
                print(keypoint_coords)
                keypoint_coords = AffichePointCamera(keypoint_coords)
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

                if sum < 0.5:
                    i = 0
                elif sum < 1:
                    i = 1
                elif sum < 3.5:
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
            for y in range(0,len(VideoPoint)):
                cv_keypoints.append(cv2.KeyPoint(VideoPoint[y][1]*500, VideoPoint[y][0]*500+Video.shape[0]-500, 10))
            Video = cv2.drawKeypoints(Video, cv_keypoints, outImage=np.array([]), color=(0, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#==============================================================================#
            cv_keypoints = []
            for y in range(0,len(keypoint_coords)):
                cv_keypoints.append(cv2.KeyPoint(keypoint_coords[y][1]*500, keypoint_coords[y][0]*500+Video.shape[0]-500, 10))
            Video = cv2.drawKeypoints(Video, cv_keypoints, outImage=np.array([]), color=(255, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
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
