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
from PIL import Image, ImageTk
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

directoryPath = dirname(realpath(__file__))
videoPath = directoryPath.replace('src', 'videos')

BUTTON_WIDTH = 160
BUTTON_HEIGHT = 100
QUIT_BUTTON_WIDTH = 50
QUIT_BUTTON_HEIGHT = 50


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
        #self.window.attributes('-alpha', 0.5)
        self.window.title("Basic Fat")
        # self.build_window()
        self.window_video_list()
        self.window.mainloop()
        self.video_process.terminate()

    def build_window(self):
        img = Image.open(join(directoryPath, 'img/fondmenu.jpg'))
        background_image = ImageTk.PhotoImage(img)
        background_label = Label(self.window, image=background_image)
        background_label.place(x=0, y=0, relwidth=1, relheight=1)
        background_label.image = background_image
        # background_label.pack()

    def window_video_list(self):
        frame = Frame(self.window)
        files = self.list_video_files()
        s, redCross = self.resize_image(
            QUIT_BUTTON_HEIGHT, QUIT_BUTTON_WIDTH, image=join(directoryPath, "img/red_cross.png"))
        quitButton = Button(frame, text="Quit",
                            height=QUIT_BUTTON_HEIGHT, width=QUIT_BUTTON_WIDTH, command=self.close_window)
        if s:
            quitButton.config(image=redCross)
            quitButton.image = redCross
        quitButton.pack(side=RIGHT)
        frame.pack(fill=BOTH)
        pannel = PanedWindow(self.window)
        pannel.pack(expand=True, padx=20)
        for f in files:
            b = Button(text=f, height=BUTTON_HEIGHT, width=BUTTON_WIDTH,
                       command=lambda k=f: self.start_video_process(k))
            success, img = self.resize_image(
                BUTTON_HEIGHT, BUTTON_WIDTH, video=join(videoPath, f))
            if success:
                b.config(image=img)
                b.image = img
            pannel.add(b)

    def resize_image(self, height, width, image=None, video=None):
        if image is not None:
            img = Image.open(image)
        elif video is not None:
            vidcap = cv2.VideoCapture(video)
            success, image = vidcap.read()
            if success == True:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(image)
            else:
                return (False, None)
        else:
            return (False, None)
        img = img.resize((width, height), Image.ANTIALIAS)
        return (True, ImageTk.PhotoImage(img))

    def start_video_process(self, video):
        videoFile = join(videoPath, video)
        modelFile = videoFile.replace('mp4', 'npy')
        self.video_process = mp.Process(
            target=start_session, args=(videoFile, modelFile))
        self.video_process.start()
        self.video_process.join()

    def list_video_files(self):
        # return list with video files names who have a .npy associated
        return [f for f in listdir(videoPath) if isfile(
            join(videoPath, f)) and f.lower().endswith('.mp4') and isfile(join(videoPath, f.replace('mp4', 'npy')))]

    def close_window(self):
        self.window.destroy()


if __name__ == "__main__":
    window = Window()
