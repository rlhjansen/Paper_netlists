from random import random

from ..classes.grid import file_to_grid
from ..alghelper import swap_two_X_times
from ..alghelper import combine_score
from .optimizer import Optimizer


class HC(Optimizer):
    def __init__(self, c, cX, n, nX, x, y, tag, iters=10000, chance_on_same=.5, **kwargs):
        """Initializes the HillClimber solver

        :param grid_num: Assigned number of circuit
        :param net_num: Filename of the netlist
        :param x,y,g: attributs of grid for filename generation
        :param iters: number of iterations (of climbing)
        :param chance_on_same:acceptance chance when same score is reached in
            new order
        """
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('hc', swaps=kwargs.get("swaps"))
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
        self.circuit.disconnect()
        self.best_ordering = kwargs.get("ordering")



    def check_climb(self):
        if self.current_score > self.used_score:
            self.used_score = self.current_score
            self.used = self.current
            self.used_conn, self.used_len = self.cur_conn, self.cur_len
        elif self.current_score == self.used_score:
            if self.chance < random():
                self.used_score = self.current_score
                self.used = self.current
                self.used_conn, self.used_len = self.cur_conn, self.cur_len



    def run_algorithm(self, interval=100):
        self.circuit.connect()
        cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
        self.current_score = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
        self.cur_conn, self.cur_len = cur_conn, cur_len

        self.used_score = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
        self.used_conn, self.used_len = self.cur_conn, self.cur_len
        self.used = self.current[:]
        self.circuit.reset_nets()
        self.add_iter()
        for i in range(self.iters-1):
            if not i % interval:
                print("completed " + str(i//interval) + " batches of ", self.iters/interval)
            self.current = swap_two_X_times(self.used, self.swaps)
            cur_conn, cur_len = self.circuit.solve_order(self.current)[:2]
            self.cur_conn, self.cur_len = cur_conn, cur_len
            self.current_score = combine_score(cur_conn, cur_len, self.best_ordering, self.n)
            self.check_climb()
            self.circuit.reset_nets()
            self.add_iter()
        print("finished gathering hillclimber")
        self.save(used_scores=True, used_results=True, all_scores=True, all_results=True)
