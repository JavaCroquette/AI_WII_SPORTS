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


path = dirname(realpath(__file__)).replace('/src', '/videos')


def start_session(video, model):
    sess = tf.Session()
    model_cfg, model_outputs = posenet.load_model(args.model, sess)
    Exercice1 = exercice(sess, model_cfg, model_outputs, args,
                         video, model)
    Exercice1.start()


class Window:
    def __init__(self):
        self.window = Tk()
        self.window.geometry("270x360")
        self.window.title("Basic Fat")
        self.window_video_list()

        launchButton = Button(self.window, text="Begin",
                              command=self.start_video_process)
        launchButton.pack(side=BOTTOM)
        self.window.mainloop()
        self.video_process.terminate()

    def window_video_list(self):
        files = self.list_video_files()
        self.tree = ttk.Treeview(self.window)
        self.tree.column("#0", minwidth=270)
        self.tree.heading("#0", text="Name", anchor=W)
        for f in files:
            self.tree.insert("", "end", f, text=f)
        self.tree.pack(side=TOP, fill=X)

    def start_video_process(self):
        videoFile = join(path, self.tree.selection()[0])
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
