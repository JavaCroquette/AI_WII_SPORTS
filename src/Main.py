import numpy as np
from Video import video
from Camera import camera
import utile
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

#==============================================================================#
MIN = 0.01
#==============================================================================#
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)# > à 640
parser.add_argument('--cam_width', type=int, default=640)# > à 480
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=1)  # laissé à 1
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()
#==============================================================================#
MOT_DOUX = ["Tres Bien", "Bien", "Assez bien", "Courage"]
#==============================================================================#
IMAGE = [cv2.imread('img/verygood.png'),
         cv2.imread('img/good.png'),
         cv2.imread('img/bad.png'),
         cv2.imread('img/courage.png')]
#==============================================================================#
#=============================== MAIN DU PROGRAMME ============================#
#==============================================================================#
def main():
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    Camera_thread = camera(args, sess, model_cfg, model_outputs)
    Video_thread = video(MOT_DOUX, './Exercice1/Exercice1.mp4', './Exercice1/video.npy')
#==============================================================================#
    Camera = None
    data = None
    Video = None
    VideoPoint = None
    CameraPoint = None
    rand = 0
    check = False
    i = 3
    graph = []
    fig = plt.figure(figsize=(8, 2), dpi=80)
    listSum = []
    sum = 0
#==============================================================================#
    Camera_thread.start()
    Video_thread.start()
#==============================================================================#
    while True:
        if len(Camera_thread.ListPoint) != 0:#On check si on a une donnée camera
            Camera = Camera_thread.ListPoint[0][0]
            CameraPoint = utile.Patron(Camera_thread.ListPoint[0][1][0], False)
            del Camera_thread.ListPoint[0]
            if Camera > MIN:
                Camera_thread.frame_count += 1
            check = True

        if len(Video_thread.ListPoint) != 0 and Camera is not None:# " " " video
            if Camera > MIN:
                Video = Video_thread.ListPoint[0][0]
                VideoPoint = Video_thread.ListPoint[0][1]
                del Video_thread.ListPoint[0]
                VideoPoint = utile.Patron(VideoPoint[:, 1], True)
                Video_thread.frame_count += 1

        if Video is not None:#On check si il y en a deux
            if check:
                print("Camera : " + str(Camera_thread.frame_count)+" -- Video : " + str(Video_thread.frame_count))
                if Camera > MIN:
                    for p in range(0, len(CameraPoint)):
                        sum = sum + (utile.Distance(CameraPoint[p],VideoPoint[p])/(sqrt(2)*17))#Normalisation
                else:
                    sum = sum + 1
                if (Camera_thread.frame_count)%10 == 0:
                    sum = sum / 10
                    listSum.append(sum)
                    if sum-0.1 < 0.1:
                        i = 0
                    elif sum-0.1 < 0.2:
                        i = 1
                    elif sum-0.1 < 0.3:
                        i = 2
                    else:
                        i = 3
                    sum = 0
                check = False
            else:
                print("Camera : " + str(Camera_thread.frame_count) +
                      " == Video : " + str(Video_thread.frame_count))
#==============================================================================#
            if (Camera_thread.frame_count)%10 == 0:
                plt.cla()
                plt.ylim(0,1)
                canvas = FigureCanvas(fig)
                plt.plot(range(0, len(listSum)), listSum)
                canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            if data is not None:
                Video[0:data.shape[0], Video.shape[1] - data.shape[1]:Video.shape[1]] = data
                cv2.putText(Video, str(MOT_DOUX[i]), (50, 200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                cv2.putText(Video, str("Marge d'erreurs:")+str(round(listSum[len(listSum)-1],2)), (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
#==============================================================================#
                img = IMAGE[i]
                Video[250:img.shape[0]+250, img.shape[1]+100 -img.shape[1]:img.shape[1]+100] = img
#==============================================================================#
            Video[Video.shape[0]-500:Video.shape[0], 0:500] = [0, 0, 0]
#==============================================================================#
            Video = utile.draw(VideoPoint,Video,[0, 255, 255],True)
            if Camera > 0:
                Video = utile.draw(CameraPoint,Video,[255, 255, 0],False)
            else:#Si la personne sort du cadre de la caméra
                Video[:,:] = [100,100,100]
                cv2.putText(Video, str("REVENEZ DEVANT LA CAMERA"), (50, 500),cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 12)
#==============================================================================#
            cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Video', Video)
#==============================================================================#
        if Video_thread.Taille is not None:
            if Video_thread.Taille == Video_thread.frame_count:
                Video_thread.stopthread()
                Camera_thread.stopthread()
                break
#==============================================================================#
        if cv2.waitKey(25) & 0xFF == ord('q'):
            Video_thread.stopthread()
            Camera_thread.stopthread()
            break
#==============================================================================#
    print("============= C'EST FINIIIIIIIII ===============")
    b = 0
    for i in listSum:
        b = b + i
    print(" !=! "+str(b / (int(Video_thread.frame_count/10))*100)+str(" %")+" !=! ")


if __name__ == "__main__":
    main()
