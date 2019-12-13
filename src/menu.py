from termcolor import colored
import sharedmem
import numpy as np
import PIL
from PIL import ImageTk
from ctypes import c_uint8
from Main import exercise
import tkinter
import multiprocessing as mp
mp.set_start_method('spawn', True)

resolution = (480, 720)


#videoArray = sharedmem.empty((resolution+(3,)), dtype='uint8')


class Window:
    def __init__(self):
        self.sharedVideoArray = mp.RawArray(
            c_uint8, resolution[0]*resolution[1]*3)
        self.npVideoArray = np.frombuffer(
            self.sharedVideoArray, dtype='uint8').reshape(resolution+(3,))
        self.window = tkinter.Tk()
        self.window.geometry("1220x960")
        self.window.title("Basic Fat")

        self.Pannels = tkinter.PanedWindow(
            self.window, orient=tkinter.VERTICAL)
        self.Pannels.pack(side=tkinter.TOP, expand=tkinter.Y,
                          fill=tkinter.BOTH)

        self.VideoPanel = tkinter.LabelFrame(self.Pannels, text="Image")
        self.Video = tkinter.Canvas(self.VideoPanel)
        self.Video.pack(padx=5, pady=5)
        self.Pannels.add(self.VideoPanel)
        self.start_video_process()
        # self.window.after(20, self.updateImage)
        # self.window.after(2000, self.console_log, "test after fenetre")
        self.window.mainloop()
        self.video_process.terminate()

    def updateImage(self):
        # img = np.asarray(videoArray)
        self.npVideoArray = np.frombuffer(
            self.sharedVideoArray, dtype='uint8').reshape(resolution+(3,))
        print(colored("Image updated", 'green'))
        img = PIL.Image.fromarray(self.npVideoArray)
        img = ImageTk.PhotoImage(img)
        self.Video.configure(image=img)
        self.Video.image = img
        self.window.after(20, self.updateImage)

    def console_log(self, message):
        print(colored(message, 'green'))
        self.window.after(2000, self.console_log, "test after fenetre")

    def start_video_process(self):
        self.video_process = mp.Process(
            target=exercise, args=(self.sharedVideoArray, resolution))
        self.video_process.start()
        # self.video_process.join(timeout=1)


if __name__ == "__main__":
    window = Window()
