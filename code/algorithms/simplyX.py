from random import random

from ..classes.grid import file_to_grid
from ..alghelper import swap_two_X_times
from ..alghelper import combine_score
from .optimizer import Optimizer
import os

import matplotlib.pyplot as plt


class SIMPLY(Optimizer):
    def __init__(self, c, cX, n, nX, x, y, tag, iters=200, chance_on_same=.5, **kwargs):
        """Initializes instance to collect sample solutions

        :param grid_num: Assigned number of circuit
        :param net_num: Filename of the netlist
        :param x,y,g: attributs of grid for filename generation
        :param iters: number of iterations (of climbing)
        :param chance_on_same:acceptance chance when same score is reached in
            new order
        """
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('simple')
        self.circuit = file_to_grid(self.circuit_path, None, max_g=True)
        self.circuit.read_nets(self.netlist_path)
        self.circuit.disconnect()


    def run_algorithm(self, interval=100, _plot=False):
        self.circuit.connect()
        self.iter = 0
        ords = [self.circuit.get_random_net_order() for i in range(self.iters)]
        data = [self.circuit.solve_order(ords[i], reset=True)[:2] for i in range(self.iters)]
        combined_data = [data[i] + ords[i] for i in range(len(data))]

        self.add_iter_batch(combined_data)
        self.save_all_data(plot=_plot)
        data = sorted(self.all_data, key=lambda x : [-x[1], x[2]])
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        if _plot:
            print(xs)
            print(ys)
            plt.scatter(xs, ys)
            plt.xlim(min(xs)-1, self.n)
            plt.xlabel("nets placed")
            plt.ylabel("total_length")
            plt.title("goals = " + str(self.n) + " nets, sample of " + str(self.iters))
            fname = 'C'+str(self.c)+"_"+str(self.cX) + "N"+str(self.n)+"_"+str(self.nX)
            savefile = "./count" + str(self.iters) + "/" + fname + "scatter.png"
            print(savefile)
            if not os.path.exists(os.path.dirname(savefile)):
                os.makedirs(os.path.dirname(savefile))
            plt.savefig(savefile)
            plt.clf()
        return
