""" Program to acquire data, config near the bottom (ln. 650~ish)

"""

from random import random
from math import exp, log, tanh, ceil
import multiprocessing as mp
#import pathos.multiprocessing as mp

import sys

from .optimizer import Optimizer
from ..classes.grid import file_to_grid
from ..alghelper import combine_score, \
    swap_two_X_times


class PPA(Optimizer):
    """PPA is the Plant Propagation solver class.

    for info check the original paper by Salhi and Frage
    """
    def __init__(self, c, cX, n, nX, x, y, tag, iters=10000, workercount=None, **kwargs):
        """

        :param pop_cut: X best of a generation who will have offspring
        :param max_runners: maximum amount of child solution sets produced by
        :param workercount: # of workers paralel processes
        """
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('ppa', **kwargs)
        self.workercount = workercount
        self.pop_cut = kwargs.get("pop_cut")
        self.max_runners = kwargs.get("runners")
        self.max_distance = kwargs.get("distance")
        self.best_ordering = kwargs.get("ordering")

        pool = mp.Pool(processes=self.workercount)
        self.circuits = pool.map(self.make_circuit, [_ for _ in range(self.pop_cut*self.max_runners)])
        pool.close()

        self.pop = [self.circuits[0].get_random_net_order() for i in range(self.pop_cut)]
        self.tot_nets = n
        self.ofv = combine_score
        self.init_pop()

    def init_pop(self):
        pool = mp.Pool(processes=self.workercount)
        data_clo = pool.map(multi_run, [(self.circuits[ind], self.pop[ind]) for ind in range(len(self.pop))])
        scores, qslist = self.sort_data(data_clo)
        qslen = len(qslist)
        scores = tuple([self.ofv(qslist[i][0], qslist[i][1], self.best_ordering, self.tot_nets) for i in range(qslen)])
        orderlist = [qslist[i][2] for i in range(qslen)]
        zipped = zip(scores, orderlist)
        new_ranking = sorted(zipped, key=lambda x: -x[0])
        tot = [plant for plant in zip(*new_ranking)]
        print("tot0", tot[0], end="\n\n")
        print("tot1", tot[1])
        self.last_pop = self.pop
        self.last_scores = tot[0]
        self.add_iter_batch(data_clo)


    def fitnesses_to_newpop(self, scores, orders):
        """ runners & distances from original paper by Salhi & Fraga

        original mapping fitness:
            0.5 * (tanh(4*score - 2) + 1)

        original distance function:
            2*(1-mrank)*(random()-0.5)
            --> maps greater ranks closer to 0 and randomizes +/-

        applied distance function:
            (1-mrank)*random()
            -->  maps greater ranks closer to 0

        :param orders: best placement orders of last generation
        :returns: new population plants to be considered
        """
        newpop = []
        mapped_ranks = [0.5 * (tanh(4*score - 2) + 1) for score in scores]
        runners = []
        for mrank in mapped_ranks:
            value = ceil(self.max_runners*mrank*random())
            runners.append(value)

        for i, runner_count in enumerate(runners):
            cur_swaps = []
            for _ in range(runner_count):
                distance = (1-mapped_ranks[i])*random()*self.max_runners
                cur_swaps.append(ceil(distance))
                new_ord = swap_two_X_times(orders[i], ceil(distance))
                newpop.append(new_ord)
        return newpop

    def combine_old_new(self, nscores, norderlist):
        """ Combines new & old results, sorts them and returns them

        :param nscores: scores from the latest
        :param norderlist: orders from the latest generation

        :variable plant: order of combined net-orders and scores, decreasing

        :return: newest values, combined & sorted with the best (see elitism)
            of last generation.
        """
        #Todo make adaptible with max & min scoring, see lambda in selamoglu

        print("last_pop", len(self.last_pop))
        print("norderlist", len(norderlist))

        scores = list(self.last_scores[:])
        scores.extend(nscores)
        orders = list(self.last_pop[:])
        orders.extend(norderlist)
        zipped = zip(scores, orders)
        new_ranking = sorted(zipped, key=lambda x: -x[0])
        return [plant for plant in zip(*new_ranking)]

    def populate(self, qslist):
        """ Sets a new population for the following iteration.

        :param qslist: elements are tuple(nets solved, solution length,
                                          net order)
        :variable ranks:
        *   fitness/ranking function should be the following type:

            original:
            f(z) = (zMax - z)/(zMax-zMin), with min being objective

            The the goal of the function is to minimize z

            Implemented:
            f(z) = (z - zMin)/(zMax-zMin)
            the score z made of two parts:
              - An integer part: the connections made
              - A decimal part: (10000-length)/10000

            better (in our case, higher) values are closer to 1.

            The the goal of the function is to maximize z which should yield
            similar goal results

        """

        qslen = len(qslist)
        scores = tuple([self.ofv(qslist[i][0], qslist[i][1], self.best_ordering, self.n) for i in range(qslen)])
        orderlist = tuple([qslist[i][2] for i in range(qslen)])
        tot = self.combine_old_new(scores, orderlist)

        plant_scores = tot[0][:self.pop_cut]
        plant_orders = tot[1][:self.pop_cut]

        self.last_pop = tot[1]
        self.last_scores = tuple(tot[0])

        zMax = max(plant_scores)
        zMin = min(plant_scores)
        if zMax == zMin:
            fitnesses = [0.5 for _ in plant_scores]
        else:
            fitnesses = [(score-zMin)/(zMax-zMin) for score in plant_scores]
        self.pop = self.fitnesses_to_newpop(fitnesses, plant_orders)


    def run_algorithm(self):
        done = 0
        while self.iters - done > 0:
            print("self.iters - done =", self.iters - done)
            pool = mp.Pool(processes=self.workercount)
            data_clo = pool.map(multi_run, [(self.circuits[ind], self.pop[ind]) for ind in range(len(self.pop))])
            pool.close()
            self.add_iter_batch(data_clo)
            scores, qslist = self.sort_data(data_clo)
            self.populate(qslist[:self.pop_cut])
            done += len(data_clo)
        self.save(all_scores=True, all_results=True)


    def sort_data(self, data_clo):
        """ sorts data to either a minimum or maximum scoring measure,
        :param data_clo: list of instances with template:
            connections, length, order (and index if applicable)
        :return:
        """
        scores = []
        for inst in data_clo:
            scores.append(self.ofv(inst[0], inst[1], self.best_ordering, self.n))

        qslist = [x for _, x in sorted(zip(scores, data_clo), key=lambda x: -x[0])]
        return scores, qslist

def multi_run(gps):
    gps[0].connect()
    gps[0].reset_nets()
    satisfies = gps[0].solve_order(gps[1])
    cur_conn, cur_len = satisfies[:2]
    plant_data = (cur_conn, cur_len, tuple(gps[1]))
    return plant_data
