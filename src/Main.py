import multiprocessing as mp
from Exercice import exercice
import posenet
import argparse
import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from os import listdir, stat
from os.path import isfile, join, dirname, realpath
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
mp.set_start_method('spawn', True)
#==============================================================================#
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)  # > à 640
parser.add_argument('--cam_width', type=int, default=640)  # > à 480
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=1)  # laissé à 1
args = parser.parse_args()
#==============================================================================#


path = dirname(realpath(__file__)).replace('src', 'videos')


def start_session(video, model):
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    Exercice1 = exercice(sess, model_cfg, model_outputs, args,
                         video, model)
    Exercice1.start()


class Window:
    def __init__(self):
        self.window = Tk()
        self.window.attributes("-fullscreen", True)
        # self.window.geometry("270x360")
        self.window.title("Basic Fat")
        self.window_video_list()
        self.window.mainloop()
        self.video_process.terminate()

    def window_video_list(self):
        files = self.list_video_files()
        pannel = PanedWindow(self.window)
        pannel.pack(expand=True, padx=20)
        for f in files:
            b = Button(text=f, padx=40, pady=50,
                       command=lambda: self.start_video_process(f))
            """vidcap = cv2.VideoCapture(join(path, f))
            success, image = vidcap.read()
            if success == False:
                image = cv2.resize(image, (200, 200))
                img = Image.fromarray(image)
                img = ImageTk.PhotoImage(image=img)
                b.config(image=img)"""
            pannel.add(b)

    def start_video_process(self, video):
        videoFile = join(path, video)
        modelFile = videoFile.replace('mp4', 'npy')
        self.video_process = mp.Process(
            target=start_session, args=(videoFile, modelFile))
        self.video_process.start()
        # self.video_process.join(timeout=1)

    def list_video_files(self):
        # return list with video files names who have a .npy associated
        return [f for f in listdir(path) if isfile(
            join(path, f)) and f.lower().endswith('.mp4') and isfile(join(path, f.replace('mp4', 'npy')))]


if __name__ == "__main__":
    window = Window()
