from random import random
from math import exp, log

from Gridfile import SXHC
from independent_functions import swap_up_to_x_elems, \
    get_name_circuitfile, get_name_netfile, quicksort, print_start_iter, \
    print_final_state

class HC:
    def __init__(self, grid_file, subdir, net_file, consec_swaps, iterations, chance_on_same=.5):
        G, cur_ords, tot_nets = SXHC(grid_file, subdir, net_file, consec_swaps)
        self.name = "HC"
        self.G = G
        self.iterations = iterations
        self.cur_ords = cur_ords
        self.sol_len = 5000000
        self.sol_ord = cur_ords[0]
        self.tot_nets = tot_nets
        self.swaps = consec_swaps
        self.nets_solved = 0
        self.all_connected = False
        self.chance = chance_on_same

    def check_climb(self, cur_conn, cur_ord, cur_len):
        if not self.all_connected:
            if cur_conn > self.nets_solved:
                self.sol_ord = cur_ord
                self.nets_solved = cur_conn
                print("changed :: greater")
                if cur_conn == self.tot_nets:
                    self.sol_len = cur_len
                    self.all_connected = True
                    print("all_connected")
            elif cur_conn == self.nets_solved:
                print("enter chance conn")
                if self.chance < random():
                    print("changed :: equal")
                    sol_ord = cur_ord
                    nets_solved = cur_conn
                else:
                    print("unchanged")
        else:
            if cur_conn == self.tot_nets:
                if cur_len < self.sol_len:
                    self.sol_ord, self.sol_len = cur_ord, cur_len
                elif cur_len == self.sol_len:
                    print("enter chance len")
                    if self.chance < random():
                        print("changed :: equal")
                        sol_ord = cur_ord
                        nets_solved = cur_conn
                    else:
                        print("unchanged")

    def run_algorithm(self):
        for i in range(self.iterations):
            print_start_iter(self.name, self.swaps, i)
            if self.sol_ord:
                cur_ords = swap_up_to_x_elems(self.sol_ord, self.swaps)

            data_lco = []  # conn, len, order
            for cur_ord in cur_ords:
                cur_len, cur_conn = self.G.solve_order(cur_ord)
                data_lco.append((cur_conn, cur_len, tuple(cur_ord)))
                self.G.reset_nets()
            qslist = quicksort(data_lco)
            cur_conn, cur_len, tcur_ord = qslist[0]
            cur_ord = list(tcur_ord)
            print("checking choice", cur_conn, cur_len)
            print("current best =", self.nets_solved, self.sol_len)

            self.check_climb(cur_conn, cur_ord, cur_len)
        self.G.solve_order(self.sol_ord)
        print_final_state(self.G, self.sol_ord, self.sol_len, \
                          self.nets_solved, self.tot_nets)


class SA:
    def __init__(self, grid_file, subdir, net_file, consec_swaps, iterations, function_params):
        G, cur_orders, all_nets = SXHC(grid_file, subdir, net_file, consec_swaps)
        self.G = G
        self.name = "SA"
        self.sol_len = 5000000
        self.nets_solved = 0
        self.sol_ord = cur_orders[0]
        self.tot_nets = all_nets
        self.func = function_params[0]
        self.swaps = consec_swaps
        self.nets_solved = 0
        self.all_connected = False
        self.iterations = iterations
        self.determine_anneal(function_params)



    def run_algorithm(self):
        for i in range(self.iterations):
            print_start_iter(self.name, self.swaps, i)
            if self.sol_ord:
                cur_ords = swap_up_to_x_elems(self.sol_ord, self.swaps)

            data_lco = []  # conn, len, order
            for cur_ord in cur_ords:
                cur_len, cur_conn = self.G.solve_order(cur_ord)
                data_lco.append((cur_conn, cur_len, tuple(cur_ord)))
                self.G.reset_nets()
            qslist = quicksort(data_lco)
            cur_conn, cur_len, tcur_ord = qslist[0]
            cur_ord = list(tcur_ord)

            self.check_anneal(cur_conn, cur_ord, cur_len)
        self.G.solve_order(self.sol_ord)
        print_final_state(self.G, self.sol_ord, self.sol_len, \
                          self.nets_solved, self.tot_nets)


    def check_anneal(self, cur_conn, cur_ord, cur_len):
        if not self.all_connected:
            print(cur_conn)
            if cur_conn > self.nets_solved:
                self.sol_ord = cur_ord
                self.nets_solved = cur_conn
                print("change sol order :: greater")
                if cur_conn == self.tot_nets:
                    self.sol_len = cur_len
                    self.all_connected = True
                    print("all_connected")
            elif cur_conn <= self.nets_solved:
                print("enter anneal conn")
                if self.anneal_conn(cur_conn):
                    print("change sol order :: equal")
                    self.sol_ord = cur_ord
                    self.nets_solved = cur_conn
                else:
                    print("unchanged anneal")
        else:
            if cur_conn == self.tot_nets:
                if self.sol_len > cur_len:
                    self.sol_ord, self.sol_len = cur_ord, cur_len
                elif self.anneal_len(cur_len):
                    print("enter anneal len")
                    self.sol_ord, self.sol_len = cur_ord, cur_len
                else:
                    print("enter anneal len")

    def linear_anneal_len(self, cur_len):
        chance = exp((-(cur_len - self.sol_len))/self.T)
        self.T -= self.T_step
        if random() < chance:
            return True
        else:
            return False

    def linear_anneal_conn(self, cur_conn):
        chance = exp(-(self.nets_solved - cur_conn)/self.T)
        self.T -= self.T_step
        if random() < chance:
            return True
        else:
            return False

    def logarithmic_anneal_len(self, cur_len):
        self.T = self.T_start/log(1+self.time)
        self.time += self.timestep
        chance = exp((-(cur_len - self.sol_len))/self.T)
        print("current solution =", cur_len, "\nbest solution =", self.sol_len)
        if random() < chance:
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def logarithmic_anneal_conn(self, cur_conn):
        self.time += self.timestep
        self.T = self.T_start/log(1+self.time)
        chance = exp(-(self.nets_solved - cur_conn)/self.T)
        print("current solved =", cur_conn, "\nbest solved =", self.nets_solved, "\nall =", self.tot_nets)
        if random() < chance:
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False



    def determine_anneal(self, function):
        self.functype = function[0]
        if self.functype == 'linear':
            self.anneal_conn = self.linear_anneal_conn
            self.anneal_len = self.linear_anneal_len
            self.T = function[1]
            self.T_step = function[1]/self.iterations
        elif self.functype == 'log':
            self.T_start = function[1]
            self.time = 0
            self.timestep = function[2]
            self.anneal_conn = self.logarithmic_anneal_conn
            self.anneal_len = self.logarithmic_anneal_len



SUBDIR = "DAAL_circuits"
GRIDFILE = get_name_circuitfile(0, 18, 13, 25)
NETFILE = get_name_netfile(0, 0)
ITER = 2000
CONSEQ_SWAP = 5


#ANNEAL_FUNC = ["log", 10, 1]
ANNEAL_FUNC = ["linear", 100]

hc = HC(GRIDFILE, SUBDIR, NETFILE, CONSEQ_SWAP, ITER)
hc.run_algorithm()

#sa = SA(GRIDFILE, SUBDIR, NETFILE, CONSEQ_SWAP, ITER, ANNEAL_FUNC)
#sa.run_algorithm()