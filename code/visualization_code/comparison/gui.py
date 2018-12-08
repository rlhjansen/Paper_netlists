""" Interface for the Cornellsearch implementation of elasticsearch
interface example basis: https://stackoverflow.com/questions/34276663/tkinter-gui-layout-using-frames-and-grid
Bryan Oakley
"""

# Use Tkinter for python 2, tkinter for python 3

try:
    from tkinter import *   # Python 3.x
except:
    from Tkinter import *   # Python 2.x

from PIL import Image, ImageTk
import os

from IDparser import IDparser

p = IDparser()

root = Tk()
root.title('Model Definition')
root.geometry('{}x{}'.format(1260, 350))

# create all of the main containers

class Resviewer:
    def __init__(self, root):
        self.root = root
        top_frame = Frame(self.root, bg='cyan', width=450, height=50, pady=3)
        center = Frame(self.root, bg='gray2', width=50, height=40, padx=3, pady=3)
        btm_frame = Frame(self.root, bg='white', width=450, height=45, pady=3)
        btm_frame2 = Frame(self.root, bg='lavender', width=450, height=60, pady=3)

        # layout all of the main containers
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        top_frame.grid(row=0, sticky="ew")
        center.grid(row=1, sticky="nsew")
        btm_frame.grid(row=3, sticky="ew")
        btm_frame2.grid(row=4, sticky="ew")

        # create the widgets for the top frame
        model_label = Label(top_frame, text='Model Dimensions')
        width_label = Label(top_frame, text='Width:')
        length_label = Label(top_frame, text='Length:')
        entry_W = Entry(top_frame, background="pink")
        entry_L = Entry(top_frame, background="orange")

        # layout the widgets in the top frame
        model_label.grid(row=0, columnspan=3)
        width_label.grid(row=1, column=0)
        length_label.grid(row=1, column=2)
        entry_W.grid(row=1, column=1)
        entry_L.grid(row=1, column=3)

        # create the center widgets
        center.grid_rowconfigure(0, weight=1)
        center.grid_columnconfigure(1, weight=1)

        ctr_left = Frame(center, bg='blue', width=100, height=190)
        ctr_mid = Frame(center, bg='yellow', width=250, height=190, padx=3, pady=3)
        ctr_right = Frame(center, bg='green', width=100, height=190, padx=3, pady=3)

        ctr_left.grid(row=0, column=0, sticky="ns")
        ctr_mid.grid(row=0, column=1, sticky="nsew")
        ctr_right.grid(row=0, column=2, sticky="ns")
        Tagviewer(ctr_mid, self)

class Tagviewer:
    def __init__(self, frame, mainframe):
        self.mainframe = mainframe
        self.parser = IDparser()
        self.parser.make_commondict(["Sexp", "SA", "T40"], "")
        self.tagframe = frame

        self.activation_list = [make_active_dict(self.parser.tagsdict, self.parser.tagsdict)]
        for elem in self.parser.subset_results_files:
            self.activation_list.append(make_active_dict(self.parser.tagsdict, elem))


        for i, k in enumerate(self.parser.tagsdict.keys()):
            c = "gray" if self.activation_list[0][k] else "red"
            tag_label = Button(self.tagframe, text=k, bg=c)
            tag_label.grid(row=0, rowspan=2, column=i)
        empty_row = Label(self.tagframe, text="", bg=self.tagframe.cget('bg'))
        empty_row.grid(row=2, rowspan=1,columnspan=5)
        for j, selection in enumerate(self.parser.subset_results_files):
            for i, k in enumerate(self.parser.tagsdict.keys()):
                c = "gray" if self.activation_list[j+1][k] else "red"
                tag_label = Button(self.tagframe, text=k, bg=c)
                tag_label.grid(row=j+3, rowspan=2, column=i)


def make_active_dict(complete_d, subset_d):
    subs_k = [k for k in subset_d.keys()]
    active_dict = {k:k in subs_k for k in complete_d.keys()}
    return active_dict


if __name__=="__main__":
    r = Resviewer(root)
    r.root.mainloop()
