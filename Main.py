import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import posenet
import screeninfo
import random
from Camera import camera
from Video import video

Afficher = True;
Cordonner = False;

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=320)
parser.add_argument('--cam_height', type=int, default=240)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

MOT_DOUX = ["TES NULL",
            "ENTRAINE TOI PLUS",
            "TU PEUX MIEUX FAIRE",
            "ARRET TOUT DE SUITE",
            "IL FAUT MIEUX ARRETER",
            "RECOMMENCE",
            "HAA C'EST JEOF",
            "MERCI, MERCI, MAIS NON MERCI"]
rand = 0
def main():
    with tf.Session() as sess:
        start = time.time()

        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        Camera_thread = camera(args,sess,model_cfg, model_outputs);
        Video_thread = video(MOT_DOUX);

        Camera_thread.start();
        Video_thread.start();

        Video = None;
        Camera = None;
        rand = 0;
        while True:
            #print(Camera_thread.frame_count)
            #print(Video_thread.frame_count)
            if len(Camera_thread.List) != 0:
                Camera = Camera_thread.List[0]
                del Camera_thread.List[0]
            if len(Video_thread.List) != 0:
                Video = Video_thread.List[0]
                del Video_thread.List[0]
                if Video_thread.frame_count%100 == 0:
                    rand = random.randint(0,len(MOT_DOUX)-1)
                    print(rand)
            if not Video is None and not Camera is None:
                Video[0:Camera.shape[0],Video.shape[1]-Camera.shape[1]:Video.shape[1]] = Camera
                cv2.putText(Video, MOT_DOUX[rand], (100, 100),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
                cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty('Video',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
                cv2.imshow('Video',Video)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    Camera_thread.stopthread()
                    Video_thread.stopthread()
                    break

if __name__ == "__main__":
    main()
