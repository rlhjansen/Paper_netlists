""" Program to acquire data, config near the bottom (ln. 650~ish)

"""

from random import random
from math import exp, log, tanh, ceil
import multiprocessing as mp

import sys

from .optimizer import Optimizer
from ..classes.grid import file_to_grid
from ..alghelper import \
    combine_score, \
    swap_two_X_times



class PPASELA(Optimizer):
    """PPA is the Plant Propagation solver class.

    for info check the original paper by Salhi and Frage
    """
    def __init__(self, c,cX, n, nX, x, y, tag, iters=10000, workercount=None, **kwargs):
        """

        :param subdir: Subdirectory name where files are located
        :param grid_num: Assigned number of circuit
        :param net_num: Filename of the netlist
        :param x,y,g: attributs of grid for filename generation
        :param max_generations: Maximum amount of generations

        :param elitism: amount best plants being added to the next generation
            0 means all are added
        :param pop_cut: X best of a generation who will have offspring
        :param max_runners: maximum amount of child solution sets produced by
            one parent.
        :param combined: fitness based on combined score (integer & decimal part)
        :param multi_objective: fitness based on [X, Y] percentages for
            multi optimization X for connections, Y for length
        :param connect_only: fitness based on number of connections only
        :param workercount: # of workers paralel processes
        """
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('sela', **kwargs)
        self.workercount = workercount
        self.pop_cut = kwargs.get("pop_cut")
        self.max_distance = kwargs.get("distance")
        self.best_ordering = kwargs.get("ordering")
        self.arbitrary = kwargs.get("arbitrary")
        self.best_percent = kwargs.get("best_percent")
        self.tot_nets = n
        circuit_count = self.calc_neccesary_circuits()
        pool = mp.Pool(processes=self.workercount)
        self.circuits = pool.map(self.make_circuit, [_ for _ in range(circuit_count)])
        pool.close()

        self.pop = [self.circuits[0].get_random_net_order() for i in range(self.pop_cut)]
        self.pop_score = [0]*self.pop_cut

    def calc_neccesary_circuits(self):
        fp = sum([ceil(self.arbitrary/(i+1)) for i in range(ceil(self.pop_cut*self.best_percent))])
        sp = self.pop_cut - ceil(self.pop_cut*self.best_percent)
        return fp+sp

    def run_algorithm(self):
        """ See paper Selamoglu & Salhi:

        The Plant Propagation Algorithm for Discrete Optimisation:
            The Case of the Travelling Salesman Problem
        """
        self.initial_score_Selamoglu()
        print("Initial scoring done")
        done = len(self.pop)
        while self.iters - done > 0:
            next_ords = []
            popsize = ceil(len(self.pop))
            best_plants = ceil(popsize * self.best_percent)
            for j in range(best_plants):
                new = [(swap_two_X_times(self.pop[j], 1), j) for _ in range(ceil(self.arbitrary / (j + 1)))]
                next_ords.extend(new)
            next_ords.extend([(swap_two_X_times(self.pop[j], self.max_distance), j) for j in range(best_plants, popsize, 1)])

            pool = mp.Pool(processes=self.workercount)
            data_clo = pool.map(multi_selamoglu, [(self.circuits[ind], next_ords[ind]) for ind in range(len(next_ords))])
            pool.close()
            done += len(data_clo)
            self.add_iter_batch(data_clo)

            for inst in data_clo:
                (cur_conn, cur_len, cur_order, index) = inst
                self.current_score = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
                if runner_eval > self.pop_score[index]:
                    self.pop[index], self.pop_score[index] = cur_order, runner_eval

            zipped = zip(self.pop, self.pop_score)
            resorted = list(zip(*sorted(zipped, key=lambda x: -x[1])))
            self.pop, self.pop_score = list(resorted[0]), list(resorted[1])

            print("generation done")
        self.save(all_scores=True, all_results=True)



    def initial_score_Selamoglu(self):
        pool = mp.Pool(processes=self.workercount)
        data_clo = pool.map(multi_selamoglu,
                            [(self.circuits[ind], (self.pop[ind], ind)) for ind in
                             range(len(self.pop))])
        pool.close()
        scores, qslist = self.sort_data(data_clo)

        for inst in qslist:
            (cur_conn, cur_len, cur_order, index) = inst
            runner_eval = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
            self.pop[index], self.pop_score[index] = cur_order, runner_eval
        self.add_iter_batch(data_clo)



    def sort_data(self, data_clo):
        """ sorts data to either a minimum or maximum scoring measure,
        :param data_clo: list of instances with template:
            connections, length, order (and index if applicable)
        :return:
        """
        scores = []
        for inst in data_clo:
            scores.append(combine_score(inst[0], inst[1], self.best_ordering, self.n))
        qslist = [x for _, x in sorted(zip(scores, data_clo), key=lambda x: -x[0])]
        return scores, qslist


def multi_run(gps):
    gps[0].connect()
    satisfies = gps[0].solve_order(gps[1])

    cur_conn, cur_len = satisfies[:2]
    plant_data = (cur_conn, cur_len, tuple(gps[1]))
    return plant_data

def multi_selamoglu(gpis):
    gpis[0].connect()
    satisfies = gpis[0].solve_order(gpis[1][0])
    # print(gpis[1][0])
    cur_conn, cur_len = satisfies[:2]
    plant_data = (cur_conn, cur_len, tuple(gpis[1][0]), gpis[1][1])
    return plant_data
