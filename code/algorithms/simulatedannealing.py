""" Program to acquire data, config near the bottom (ln. 650~ish)

"""

from random import random
from math import exp, log, ceil
#import pathos.multiprocessing as mp

import sys

from ..classes.grid import file_to_grid
from ..alghelper import swap_two_X_times
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
        self.set_saveloc('sa', **kwargs)
        self.circuit = file_to_grid(self.circuit_path, None)
        self.circuit.read_nets(self.netlist_path)
        self.current = self.circuit.get_random_net_order()
        self.sol_len = 5000
        self.swaps = kwargs.get("swaps")
        self.all_connected = False
        self.sol_len = 5000000
        self.sol_conn = 0
        self.tot_nets = n
        self.sol_conn = 0
        self.determine_anneal(**kwargs)
        self.annealFunc = None

    def run_algorithm(self, interval=100):
        cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
        self.current_score = [cur_conn, cur_len]
        self.used_score = self.current_score[:]
        self.used = self.current[:]
        self.circuit.reset_nets()
        self.add_iter()
        self.check_anneal(cur_conn, cur_len, 0)
        for i in range(1, self.iters):
            if not i % interval:
                print("completed " + str(i//interval) + " batches of ", interval)
            self.current = swap_two_X_times(self.current, self.swaps)
            cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
            self.circuit.reset_nets()
            self.check_anneal(cur_conn, cur_len, i)
            self.add_iter()
        print("finished gathering simulated annealing")
        self.save(used_scores=True, used_results=True, all_scores=True, all_results=True)



    ################################################################
    # Anneal Checks                                                #
    ################################################################

    def check_anneal(self, cur_conn, cur_len, i):
        if not self.all_connected:
            if cur_conn > self.used_score[0]:
                self.used = self.current
                self.sol_conn = cur_conn
                self.used_score = [cur_conn, cur_len]
                self.sol_len = cur_len
            elif cur_conn <= self.used_score[0]:
                if self.anneal_conn(cur_conn, i):
                    print("change sol order :: equal")
                    self.used = self.current
                    self.sol_conn = cur_conn
                    self.used_score = [cur_conn, cur_len]
                else:
                    print("unchanged anneal")
        else:
            if cur_conn == self.tot_nets:
                if self.sol_len > cur_len:
                    self.used, self.sol_len = self.current, cur_len
                    self.used_score = [cur_conn, cur_len]
                elif self.anneal_len(cur_len, i):
                    print("enter anneal len")
                    self.used, self.sol_len = self.current, cur_len
                    self.used_score = [cur_conn, cur_len]
                else:
                    print("unchanged anneal len")

    def check_len_anneal(self, cur_conn, cur_len, i):
        if self.sol_len > cur_len:
            self.sol_conn, self.used, self.sol_len = cur_conn, self.current, cur_len
        elif self.anneal_len(cur_len, i):
            print("enter anneal len")
            self.sol_conn, self.used, self.sol_len = cur_conn, self.current, cur_len
        else:
            print("unchanged anneal len")

    def check_conn_anneal(self, cur_conn, cur_len, i):
        if cur_conn > self.sol_conn:
            self.sol_conn, self.used, self.sol_len = cur_conn, self.current, cur_len
        elif self.anneal_len(cur_len, i):
            print("enter anneal conn")
            self.sol_conn, self.used, self.sol_len = cur_conn, self.current, cur_len
        else:
            print("unchanged anneal conn")

    ################################################################
    # Anneal Checks - Comparisons and temperatures                 #
    ################################################################

    def linear_anneal_len(self, cur_len, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        chance = exp((-(cur_len - self.sol_len))/(self.T-(i)*self.T_step))
        if random() < chance:
            return True
        else:
            return False

    def linear_anneal_conn(self, cur_conn, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        chance = exp((-(cur_conn - self.sol_conn))/(self.T-(i)*self.T_step))
        if random() < chance:
            return True
        else:
            return False

    def logarithmic_anneal_len(self, cur_len, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        self.T = self.T_start/log(1+i)
        chance = exp((-(cur_len - self.sol_len))/self.T)
        print("current solution =", cur_len, "\nbest solution =", self.sol_len)
        if random() < chance:
            return True
        else:
            return False

    def logarithmic_anneal_conn(self, cur_conn, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """

        self.T = self.T_start/log(1+i)
        chance = exp(-(self.sol_conn - cur_conn)/self.T)
        if random() < chance:
            return True
        else:
            return False

    def exp_anneal_conn(self, difference, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """

        T = self.exp_temp(i + 1)
        if exp(difference / T) > random():
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def exp_anneal_len(self, difference, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        T = self.exp_temp(i+1)
        if exp(difference / T) > random():
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def geman_anneal_len(self, difference, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        T = self.c / log(i+self.d)
        if exp(difference / T) > random():
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def geman_anneal_conn(self, difference, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        T = self.c / log(i+self.d)
        if exp(difference / T) > random():
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
        self.annealFunc = self.check_anneal

        schema = kwargs.get("schema", None)
        start_temp = kwargs.get("start_temp", None)
        end_temp = kwargs.get("end_temp", None)

        if schema == 'linear':
            self.anneal_conn = self.linear_anneal_conn
            self.anneal_len = self.linear_anneal_len
            self.T = start_temp
            self.T_step = start_temp/(self.iters+1)
        elif schema == 'log':
            self.T_start = start_temp # Starting Temperature
            self.anneal_conn = self.logarithmic_anneal_conn
            self.anneal_len = self.logarithmic_anneal_len
        elif schema == 'exp':
            self.st = start_temp # Starting Temperature
            self.et = end_temp # End Temperature
            self.anneal_conn = self.exp_anneal_conn
            self.anneal_len = self.exp_anneal_len
        elif schema == 'geman':
            self.anneal_len = self.geman_anneal_len
            self.anneal_conn = self.geman_anneal_conn
            self.c = kwargs.get("cv", None) # largest barrier, arbitrarily chosen
            self.d = kwargs.get("dv", None) # arbitrarily chosen


if __name__ == '__main__':
    SA()
