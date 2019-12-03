import numpy as np
from Video import video
from Camera import camera
import random
import screeninfo
import posenet
import argparse
import cv2
import matplotlib.pyplot as plt
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

def main():
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    Camera_thread = camera(args, sess, model_cfg, model_outputs)
    Video_thread = video(MOT_DOUX, './Exercice1/video.avi', './Exercice1/video.npy')

    Camera = None
    data = None
    Video = None
    VideoPoint = None
    CameraPoint = None
    rand = 0
    check = False
    sum = 0
    i = 0
    graph = []
    fig = plt.figure()
    plt.ylim(0,10)

    Camera_thread.start()
    Video_thread.start()

    while True:
        if len(Camera_thread.List) != 0:
            Camera = Camera_thread.List[0]
            del Camera_thread.List[0]
        if len(Camera_thread.ListPoint) != 0:
            CameraPoint = Camera_thread.ListPoint[0]
            del Camera_thread.ListPoint[0]

        if len(Video_thread.List) != 0:
            Video = Video_thread.List[0]
            del Video_thread.List[0]
            check = True
        if len(Video_thread.ListPoint) != 0:
            VideoPoint = Video_thread.ListPoint[0]
            del Video_thread.ListPoint[0]

        if Video is not None and Camera is not None and CameraPoint is not None and VideoPoint is not None and check:
            widthV = Video.shape[1]
            heightV = Video.shape[0]
            widthC = args.cam_width
            heightC = args.cam_height
            # resize image
            pose_scores = Camera[0].copy()
            keypoint_scores = Camera[1].copy()
            keypoint_coords = Camera[2].copy()
            keypoint_coords[0,:,0] = keypoint_coords[0,:,0]/heightC*heightV
            keypoint_coords[0,:,1] = (-keypoint_coords[0,:,1]+widthC)/widthC*widthV

            Video = posenet.draw_skel_and_kp(
                Video, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.01, min_part_score=0.01)

            sum = 0
            for p in range(0, len(CameraPoint)):
                sum = sum + abs(CameraPoint[p][1][0]/heightC - VideoPoint[p][1][0]/heightV)
                sum = sum + abs(CameraPoint[p][1][1]/widthC - VideoPoint[p][1][1]/widthV)
            check = False
            i = 0
            if sum < 1:
                i = 0
            elif sum < 2:
                i = 1
            elif sum < 3:
                i = 2
            else:
                i = 3
            cv2.putText(Video, str(MOT_DOUX[i]), (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
            if Camera_thread.frame_count%2 == 0:
                fig.canvas.draw()
                plt.plot(Camera_thread.frame_count, sum, "or")
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            if data is not None:
                Video[0:data.shape[0], Video.shape[1] - data.shape[1]:Video.shape[1]] = data
            cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(
                'Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow('Video', Video)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            Video_thread.stopthread()
            Camera_thread.stopthread()
            break

if __name__ == "__main__":
    main()
