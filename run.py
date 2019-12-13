from termcolor import colored
import sharedmem
import numpy as np
import PIL
from PIL import ImageTk
from ctypes import c_uint8
# from src.Main import main
from tkinter import *
from tkinter import ttk
from os import listdir, stat
from os.path import isfile, join
import multiprocessing as mp
mp.set_start_method('spawn', True)

resolution = (480, 720)

path = './videos'


class Window:
    def __init__(self):
        self.sharedVideoArray = mp.RawArray(
            c_uint8, resolution[0]*resolution[1]*3)
        self.npVideoArray = np.frombuffer(
            self.sharedVideoArray, dtype='uint8').reshape(resolution+(3,))
        self.window = Tk()
        self.window.geometry("1220x960")
        self.window.title("Basic Fat")

        """files = self.list_video_files()
        pannels = np.zeros(len(files))
        for i in range(0, len(files)):
            p = PanedWindow(
                self.window, orient=VERTICAL)
            p.pack(side=TOP, expand=Y,
                   fill=BOTH, pady=2, padx=2)
            p.add(Label(
                pannels, text=files[i].replace('.mp4', ''), anchor=CENTER))
            pannels[i] = p"""
        self.window_video_list()
        # self.start_video_process()
        # self.window.after(20, self.updateImage)
        # self.window.after(2000, self.console_log, "test after fenetre", 2000)
        self.window.mainloop()
        # self.video_process.terminate()

    def window_video_list(self):
        files = self.list_video_files()
        tree = ttk.Treeview(self.window)
        tree.column("#0", minwidth=270, width=270, stretch=NO)
        tree.heading("#0", text="Name", anchor=W)
        for f in files:
            tree.insert("", "end", f, text=f)
        tree.pack(side=TOP, fill=X)

    def console_log_loop(self, message, loop):
        print(colored(message, 'green'))
        self.window.after(loop, self.console_log_loop,
                          message, loop)

    """def start_video_process(self):
        self.video_process = mp.Process(
            target=main, args=(self.sharedVideoArray, resolution))
        self.video_process.start()
        # self.video_process.join(timeout=1)"""

    def list_video_files(self):
        # return list with video files names who have a .npy associated
        return [f for f in listdir(path) if isfile(
            join(path, f)) and f.lower().endswith('.mp4') and isfile(join(path, f.replace('mp4', 'npy')))]


if __name__ == "__main__":
    window = Window()
