""" Program to acquire data, config near the bottom (ln. 650~ish)

"""

from random import random
from math import exp, log, ceil
#import pathos.multiprocessing as mp

import sys

from ..classes.grid import file_to_grid
from ..alghelper import swap_two_X_times
from ..alghelper import combine_score
from .optimizer import Optimizer


class SA(Optimizer):
    def __init__(self,c, cX, n, nX, x, y, tag, iters=10000, chance_on_same=.5, ask=True, **kwargs):
        """Initializes the Simulated Annealing solver

        :param subdir: subdirectory of circuit & netfiles
        :param grid_num: circuit number
        :param net_num: net number
        :param x: x parameter of circuit
        :param y: y parameter of circuit
        :param tot_gates: total gates of a circuit
        :param iterations: total number of climbing iterations
        :param function_params: parameters to select how to anneal
            see determine anneal
        :param additions:
        """
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('sap', **kwargs)
        self.circuit = file_to_grid(self.circuit_path, None)
        self.circuit.read_nets(self.netlist_path)
        self.swaps = kwargs.get("swaps")
        self.best_ordering = kwargs.get("ordering")
        self.determine_anneal(**kwargs)

        self.current = self.circuit.get_random_net_order()
        cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
        self.cur_conn, self.cur_len = cur_conn, cur_len
        self.current_score = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
        self.current = self.circuit.get_random_net_order()
        self.used_score = self.current_score
        self.used = self.current[:]
        self.used_conn, self.used_len = self.cur_conn, self.cur_len
        self.circuit.reset_nets()
        self.add_iter()
        self.circuit.disconnect()



    def run_algorithm(self, interval=100):
        self.circuit.connect()
        for i in range(1, self.iters):
            if not i % interval:
                print("completed " + str(i//interval) + " batches of ", interval)
            self.current = swap_two_X_times(self.current, self.swaps)
            cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
            self.circuit.reset_nets()
            self.cur_conn, self.cur_len = cur_conn, cur_len
            self.current_score = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
            self.check_anneal(i)
            self.add_iter()
        print("finished gathering simulated annealing")
        self.save(used_scores=True, used_results=True, all_scores=True, all_results=True)



    ################################################################
    # Anneal Checks                                                #
    ################################################################

    def check_anneal(self, i):
        if self.anneal(i):
            print("change sol order :: equal")
            self.used = self.current
            self.used_score = self.current_score
            self.used_conn, self.used_len = self.cur_conn, self.cur_len


    ################################################################
    # Anneal Checks - Comparisons and temperatures                 #
    ################################################################

    def linear_anneal(self, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        chance = exp((-(self.current_score - self.used_score))/(self.T-(i)*self.T_step))
        if random() < chance:
            return True
        else:
            return False

    def logarithmic_anneal(self, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        self.T = self.T_start/log(1+i)
        chance = exp((-(self.current_score - self.used_score))/self.T)
        print("current solution =", self.cur_len, "\nbest solution =", self.used_len)
        if random() < chance:
            return True
        else:
            return False

    def exp_anneal(self, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        Temp = self.exp_temp(i + 1)
        if exp(-(self.current_score - self.used_score) / Temp) > random():
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def geman_anneal(self, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        T = self.cv / log(i+self.dv)
        if exp(-(self.current_score - self.used_score) / T) > random():
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def exp_temp(self, i):
        """
        :param i: iteration number i
        :return: temperature for annealing at current iteration
        """
        return self.st*(self.et/self.st)**(i/self.iters)

    def determine_anneal(self, **kwargs):
        """
        for geman see Nourani and Andresen 'A comparison of simulated annealing cooling schedules' p.3

        :param function: per index:
            0 'length', 'connections', 'all'
            1 (anneal function type) 'linear', 'log', 'exp', 'geman'
            2+ algorithms specific
        """

        schema = kwargs.get("schema", None)
        start_temp = kwargs.get("start_temp", None)
        end_temp = kwargs.get("end_temp", None)

        if schema == 'linear':
            self.anneal = self.linear_anneal
            self.T = start_temp
            self.T_step = start_temp/(self.iters+1)
        elif schema == 'log':
            self.T_start = start_temp # Starting Temperature
            self.anneal = self.logarithmic_anneal
        elif schema == 'exp':
            self.st = start_temp # Starting Temperature
            self.et = end_temp # End Temperature
            self.anneal = self.exp_anneal
        elif schema == 'geman':
            self.anneal = self.geman_anneal
            self.cv = kwargs.get("cv", None) # largest barrier, arbitrarily chosen
            self.dv = kwargs.get("dv", None) # arbitrarily chosen


if __name__ == '__main__':
    SA()
