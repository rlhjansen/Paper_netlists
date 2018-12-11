
import random
import matplotlib.pyplot as plt
from IDparser import IDparser

import sys

def lprint(iterable, join=""):
    for elem in iterable:
        if join:
            join.join(elem)
        else:
            print(elem)

def listing_header():
    return ["index", "type", "restrictiontags"]

class resfile_reader:
    def __init__(self):
        self.idp = IDparser()
        self.listings = []


    def asklistings(self):
        print("\t".join(listing_header()))
        lprint(self.listings, join="\t")
        inp = input()
        parsed = inp.split(" ")
        if parsed[0] == "add":
            pass
        elif parsed[0] == "del":
            pass
        elif parsed[0] == "edit":
            pass
        elif parsed[0] == "quit":
            raise KeyboardInterupt
        else:
            self.print_instructions()
            self.asklistings()

    @staticmethod
    def print_instructions():
        print("select an option from [add,edit,del,quit]")
        print("example input:")
        print("add *type*,*tags*\nor\nedit *index* *type*,*tags*")


    def plot_all_best_with_tags(self, tags):
        cmnd = self.idp.set_commondict(tags)
        if "PPA" in tags or "SELA" in tags:
            endtag = "all_scores.txt"
            files = [elem[1] for elem in cmnd[tags[0]] if elem[1][-len(endtag):] == endtag]
            for file in files:
                self.plot_going_best_generational(file)
        elif set(["SA", "HC"]).intersection(set(tags)):
            endtag = "used_scores.txt"
            files = [elem[1] for elem in cmnd[tags[0]] if elem[1][-len(endtag):] == endtag]
            for file in files:
                self.plot_single_best_iterative(file)
        else:
            raise ValueError("call without algorithm")
        plt.show()

    def plot_first_generational_with_tags(self, tags):
        cmnd = self.idp.get_commondict(tags)
        endtag = "all_scores.txt"
        files = [elem[1] for elem in cmnd[tags[0]] if elem[1][-len(endtag):] == endtag]
        self.plot_single_generational(files[0])
        plt.show()

    @staticmethod
    def plot_single_iterative(file):
        data = [[i, int(l.split(',')[0])] for i, l in enumerate(open(file, 'r').readlines())]
        connections = [d[1] for d in data]
        index = [d[0] for d in data]
        plt.plot(index, connections)

    @staticmethod
    def plot_single_best_iterative(file):
        data = [[i, int(l.split(',')[0])] for i, l in enumerate(open(file, 'r').readlines())]
        connections = [d[1] for d in data]
        best = 0
        bc = []
        for c in connections:
            if c > best:
                best = c
            bc.append(best)
        index = [d[0] for d in data]
        plt.plot(index, bc)


    @staticmethod
    def plot_going_best_generational(file):
        data = [float(l.split(',')[0]) for i, l in enumerate(open(file, 'r').readlines()) if l[0] != "-"]

        connections = [d for d in data if d != "-"]
        bc = []
        best = 0
        for c in connections:
            if  c > best:
                best = c
            bc.append(best)

        index = [i for i in range(len(bc))]

        generation_index_correction = 0
        generation_indices = []
        for i, elem in enumerate(data):
            if elem == '-':
                generation_indices.append(i - generation_index_correction)
                generation_index_correction += 1
        c_choice = random.choice(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        plt.plot(index, bc, c=c_choice)
        # for xvl in generation_indices:
        #     plt.axvline(xvl, c=c_choice)



if __name__ == '__main__':
    rfr = resfile_reader()
    tags = ["SELA", "C100"]
    while tags:
        rfr.plot_all_best_with_tags(tags)
        rfr.idp.show_commondict()
        tags = input().split(" ")
