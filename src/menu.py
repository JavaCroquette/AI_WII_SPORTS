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


class Menu:
    def __init__(self):
        self.resolution = (480, 720)


def main():
    videoArray = sharedmem.empty(
        (resolution+(3,)), dtype='uint8')

    fenetre = Tk()
    fenetre.geometry("1220x960")
    fenetre.title("Basic Fat")

    Pannels = PanedWindow(fenetre, orient=VERTICAL)
    Pannels.pack(side=TOP, expand=Y, fill=BOTH)

    VideoPanel = LabelFrame(Pannels, text="Image")
    Video = Label(VideoPanel)
    Video.pack(padx=5, pady=5)

    Pannels.add(VideoPanel)
    fenetre.after(20, updateImage, Video, videoArray)
    fenetre.after(1, print(colored("Test fenetre after", 'red')))

    p = Process(target=exercise, args=(videoArray, resolution))
    p.start()
    p.join(timeout=1)
    fenetre.mainloop()
    p.terminate()


def updateImage(display, videoArray):
    #img = np.asarray(videoArray)
    print(colored("Image updated", 'green'))
    img = PIL.Image.fromarray(videoArray)
    img = ImageTk.PhotoImage(img)
    display.configure(image=img)
    display.image = img


if __name__ == "__main__":
    main()
