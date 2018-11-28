""" Program to acquire data, config near the bottom (ln. 650~ish)
"""

from random import random
import multiprocessing as mp
import sys

from ..classes.grid import file_to_grid
from .optimizer import Optimizer



class RC(Optimizer):
    """
    RC or Random Collector Takes in a Grid & Netlist.
    Its goal is to apply the netlist, in random order, to the grid.
    while it does this it collects data on the number of connections, as well
    as their total length
    """
    def __init__(self, c, cX, n, nX, x, y, tag, iters=10000, batch_size=20, **kwargs):
        Optimizer.__init__(self, c, cX, n, nX, x, y, iters, tag, **kwargs)
        self.set_saveloc('rc')

        self.batches = int(float(iters)/float(batch_size))
        print(self.batches)
        self.circuits = [self.make_circuit() for _ in range(batch_size)]
        print(self.batches)
        self.tot_nets = n

    def run_algorithm(self):
        for _ in range(self.batches):
            pool = mp.Pool(processes=None)
            batch_results = pool.map(multi_run, self.circuits)
            print(batch_results)
            pool.close()
            self.add_iter_batch(batch_results)
            print("finished batch")
        self.save(all_scores=True, all_results=True)


def multi_run(circuit):
    print("starting new order...")
    circuit.connect()
    circuit.reset_nets()
    order = circuit.get_random_net_order()
    satisfies = circuit.solve_order(order)
    circuit.disconnect()

    cur_conn, cur_len = satisfies[:2]
    return cur_conn, cur_len, order
