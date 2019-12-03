from termcolor import colored
import sharedmem
import numpy as np
import PIL
from PIL import ImageTk
from ctypes import c_int
from multiprocessing import Process
from Main import exercise
from tkinter import *
import multiprocessing
multiprocessing.set_start_method('spawn', True)

resolution = (480, 720)


def main():
    videoArray = sharedmem.empty(
        resolution[0]*resolution[1]*3, dtype='uint8')

    fenetre = Tk()
    fenetre.geometry("1220x960")
    fenetre.title("Basic Fat")

    Pannels = PanedWindow(fenetre, orient=VERTICAL)
    Pannels.pack(side=TOP, expand=Y, fill=BOTH)

    VideoPanel = LabelFrame(Pannels, text="Image")
    Video = Label(VideoPanel)
    Video.pack(padx=5, pady=5)

    Pannels.add(VideoPanel)

    p = Process(target=exercise, args=(videoArray, resolution))
    p.start()
    # p.join(timeout=1)
    fenetre.after(20, updateImage, Video, videoArray)
    fenetre.mainloop()


def updateImage(display, videoArray):
    img = np.asarray(videoArray).reshape(resolution+(3,))
    img = PIL.Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    display.configure(image=img)
    display.image = img


def toNumpyArray(mp_array):
    return np.frombuffer(mp_array.get_obj())


if __name__ == "__main__":
    main()
