
import random
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from IDparser import IDparser


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


    def make_files_d(self, tags):
        cmnd = self.idp.set_commondict(tags)
        endtag = "scores.txt"
        files = [elem[1] for elem in cmnd[tags[0]] if elem[1][-len(endtag):] == endtag]
        file_score_dict = {}
        for file in files:
            if "RC" in tags:
                raise ValueError("wrong function called, instead call 'get_rand_distribution(tags)'")
            else:
                if "PPA" in tags or "SELA" in tags:
                    file_score_dict[file] = extract_file_generational(file)
                elif "HC" in tags or "SA" in tags:
                    file_score_dict[file] = extract_file_iterative(file)
        best_file = sorted(files, key=lambda file : -file_score_dict[file]["max_val"])
        file_score_dict["best"] = file_score_dict[best_file]
        return file_score_dict



    @staticmethod
    def extract_file_iterative(file):
        maximum = []
        data = [float(line) for line in open(file, 'r').readlines()]

        gen_info = {"min":[], "max":[], "mean":[]}
        for gen in data:
            gen_info["min"].append(min(gen))
            gen_info["max"].append(max(gen))
            gen_info["mean"].append(sum(gen)/len(gen))
        gen_info["max_val"] = max(gen_info["max"])
        return gen_info


    @staticmethod
    def extract_file_generational(file):
        minimum = []
        maximum = []
        mean = []
        data = [[]]
        last_split = False
        data_index = 0
        generation_lines = []
        for i, line in enumerate(open(file, 'r').readlines()):
            if l[0] == '-':
                if not last_split:   #Temporary fix for (fixed) mistake in saving implementation, remove after initial results
                    data.append([])
                    data_index += 1
                    generation_lines.append(i - data_index)
                last_split = True    #Temporary fix for (fixed) mistake in saving implementation, remove after initial results
            else:
                last_split = False   #Temporary fix for (fixed) mistake in saving implementation, remove after initial results
                data[data_index].append(float(line))
        gen_info = {"min":[], "max":[], "mean":[]}
        for i, gen in enumerate(data):
            gen_info["min"].append(min(gen))
            gen_info["max"].append(max(gen))
            gen_info["mean"].append(sum(gen)/len(gen))
            gen_info["i"].append(generation_lines[i])
        gen_info["max_val"] = max(gen_info["max"])
        df = pd.DataFrame(data=gen_info)
        return df


if __name__ == '__main__':
    rfr = resfile_reader()
    selatags = ["SELA", "C100"]
    sela_df = rfr.make_files_d(selatags)
    df = sela_df["best"]
    sns.tsplot((df['max']-df['mean']), df['i'],color=color)
    sns.tsplot((df['min']-df['mean']), df['i'],color=color)
    plt.fill_between(df['i'], (df['max']-df['mean']), (df['min']-df['mean']), color=color, alpha='0.5')
    plt.show()
