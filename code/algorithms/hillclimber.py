from random import random
import sys

from ..classes.grid import file_to_grid
from ..alghelper import swap_two
from .optimizer import Optimizer


class HC(Optimizer):
    def __init__(self, c, cX, n, nX, x, y, tag, iters=10000, chance_on_same=.5, **kwargs):
        """Initializes the HillClimber solver

        :param subdir: Subdirectory name where files are located
        :param grid_num: Assigned number of circuit
        :param net_num: Filename of the netlist
        :param x,y,g: attributs of grid for filename generation
        :param consec_swaps:
        :param iterations: number of iterations (of climbing)
        :param additions: specifics added to the algorithm, (for documentation
            purposes & changing what is done during runtime)
        :param chance_on_same:acceptance chance when same score is reached in
            new order
        """
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('hc')
        self.circuit = file_to_grid(self.circuit_path, None)
        self.circuit.read_nets(self.netlist_path)
        self.current = self.circuit.get_random_net_order()
        self.sol_len = 5000000
        self.tot_nets = n
        assert "swaps" in kwargs
        self.swaps = kwargs.get("swaps")
        self.sol_conn = 0
        self.all_connected = False
        self.chance = chance_on_same


    def check_climb(self, cur_conn, cur_len):
        self.current_score = [cur_conn, cur_len]
        if cur_conn > self.sol_conn:
            self.used = self.current
            self.sol_conn = cur_conn
            self.used_score = [cur_conn, cur_len]
        elif cur_conn == self.sol_conn:
            if cur_len < self.sol_len:
                self.used, self.sol_len = self.current, cur_len
                self.used_score = [cur_conn, cur_len]

            elif cur_len == self.sol_len:
                if self.chance < random():
                    print("changed :: equal")
                    self.used = self.current
                    self.sol_conn = cur_conn
                    self.current_score = [cur_conn, cur_len]



    def run_algorithm(self, interval=100):
        cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
        self.current_score = [cur_conn, cur_len]
        self.used_score = self.current_score[:]
        self.used = self.current[:]
        self.circuit.reset_nets()
        self.add_iter()
        for i in range(self.iters-1):
            if not i % interval:
                print("completed " + str(i//interval) + " batches of ", interval)
            self.current = swap_two(self.used)
            cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
            self.current_score = [cur_conn, cur_len]
            self.check_climb(cur_conn, cur_len)
            self.circuit.reset_nets()
            self.add_iter()
        #self.G.solve_order(self.used)
        #print(self.G)
        print("finished gathering hillclimber")
        self.save(used_scores=True, used_results=True, all_scores=True, all_results=True)
