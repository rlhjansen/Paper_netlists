from random import random
from math import exp, log

from Gridfile import SXHC, SRC
from independent_functions import swap_up_to_x_elems, \
    get_name_circuitfile, get_name_netfile, quicksort, print_start_iter, \
    print_final_state, create_data_directory, write_connections_length_ord, print_start_iter

class HC:
    def __init__(self, subdir, grid_num, net_num, x, y, tot_gates,
                 consec_swaps, iterations, additions, chance_on_same=.5):
        net_file = get_name_netfile(grid_num, net_num)
        grid_file = get_name_circuitfile(grid_num, x, y, tot_gates)
        G, cur_ords, tot_nets = SXHC(grid_file, subdir, net_file, consec_swaps)
        self.name = "HC"
        self.gn = grid_num
        self.nn = net_num
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
        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              tot_gates, net_num, additions)

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
            print_start_iter(self.gn, self.nn, self.name, self.swaps, i)
            if self.sol_ord:
                cur_ords = swap_up_to_x_elems(self.sol_ord, self.swaps)

            data_lco = []  # conn, len, order
            for i, cur_ord in enumerate(cur_ords):
                cur_conn, cur_len = self.G.solve_order(cur_ord)
                data_lco.append((cur_conn, cur_len, tuple(cur_ord)))
                self.G.reset_nets()
                write_connections_length_ord(self.savefile, [[(cur_conn, cur_len), [i+1], [i[1:] for i in cur_ord]]])

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
    def __init__(self, subdir, grid_num, net_num, x, y, tot_gates,
                 consec_swaps, iterations, function_params, additions):
        net_file = get_name_netfile(grid_num, net_num)
        grid_file = get_name_circuitfile(grid_num, x, y, tot_gates)
        G, cur_orders, all_nets = SXHC(grid_file, subdir, net_file, consec_swaps)

        self.all_connected = False
        self.iterations = iterations
        self.G = G
        self.name = "SA"
        self.sol_len = 5000000
        self.nets_solved = 0
        self.sol_ord = cur_orders[0]
        self.tot_nets = all_nets
        self.func = function_params[0]
        self.swaps = consec_swaps
        self.nets_solved = 0
        self.determine_anneal(function_params)
        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              tot_gates, net_num, additions)



    def run_algorithm(self):
        for i in range(self.iterations):
            print_start_iter(1, 3, self.name, self.swaps, i)
            if self.sol_ord:
                cur_ords = swap_up_to_x_elems(self.sol_ord, self.swaps)

            data_lco = []  # conn, len, order
            for j, cur_ord in enumerate(cur_ords):
                cur_len, cur_conn = self.G.solve_order(cur_ord)
                data_lco.append((cur_conn, cur_len, tuple(cur_ord)))
                self.G.reset_nets()
                write_connections_length_ord(self.savefile, [[(cur_conn, cur_len), [j+1], [i[1:] for i in cur_ord]]])
                self.check_anneal(cur_conn, cur_ord, cur_len, i)

        self.G.solve_order(self.sol_ord)
        print_final_state(self.G, self.sol_ord, self.sol_len, \
                          self.nets_solved, self.tot_nets)


    def check_anneal(self, cur_conn, cur_ord, cur_len, i):
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
                if self.anneal_conn(cur_conn, i):
                    print("change sol order :: equal")
                    self.sol_ord = cur_ord
                    self.nets_solved = cur_conn
                else:
                    print("unchanged anneal")
        else:
            if cur_conn == self.tot_nets:
                if self.sol_len > cur_len:
                    self.sol_ord, self.sol_len = cur_ord, cur_len
                elif self.anneal_len(cur_len, i):
                    print("enter anneal len")
                    self.sol_ord, self.sol_len = cur_ord, cur_len
                else:
                    print("enter anneal len")

    def linear_anneal_len(self, cur_len, i):
        chance = exp((-(cur_len - self.sol_len))/(self.T-(i)*self.T_step))
        self.T -= self.T_step
        if random() < chance:
            return True
        else:
            return False

    def linear_anneal_conn(self, cur_conn, i):
        print('self.T', self.T)
        print('i', i)
        print('self.T_step', self.T_step)
        chance = exp((-(cur_conn - self.nets_solved))/(self.T-(i)*self.T_step))
        if random() < chance:
            return True
        else:
            return False

    def logarithmic_anneal_len(self, cur_len, i):
        self.T = self.T_start/log(1+i)
        chance = exp((-(cur_len - self.sol_len))/self.T)
        print("current solution =", cur_len, "\nbest solution =", self.sol_len)
        if random() < chance:
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def logarithmic_anneal_conn(self, cur_conn, i):
        self.T = self.T_start/log(1+i)
        chance = exp(-(self.nets_solved - cur_conn)/self.T)
        print("current solved =", cur_conn, "\nbest solved =", self.nets_solved, "\nall =", self.tot_nets)
        if random() < chance:
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False

    def exp_anneal_conn(self, difference, i):
        T = self.exp_temp(i + 1)
        if exp(difference / T) > random():
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
            return False


    def exp_anneal_len(self, difference, i):
        T = self.exp_temp(i+1)

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
        return self.st*(self.et/self.st)**(i/self.iterations)

    def determine_anneal(self, function):
        self.functype = function[0]
        if self.functype == 'linear':
            self.anneal_conn = self.linear_anneal_conn
            self.anneal_len = self.linear_anneal_len
            self.T = function[1]
            self.T_step = function[1]/(self.iterations+1)
        elif self.functype == 'log':
            self.T_start = function[1]
            self.time = 0
            self.timestep = function[2]
            self.anneal_conn = self.logarithmic_anneal_conn
            self.anneal_len = self.logarithmic_anneal_len
        elif self.functype == 'exp':
            self.st = function[1]
            self.et = function[2]
            self.anneal_conn = self.exp_anneal_conn
            self.anneal_len = self.exp_anneal_len





class RC:
    """
    RC or Random Collector Takes in a Grid & Netlist.
    Its goal is to apply the netlist, in random order, to the grid.
    while it does this it collects data on the number of connections, as well
    as their total length
    """
    def __init__(self, main_subdir, grid_num, list_num,  x, y, tot_gates,
                 batch_size, batches, additions):
        net_file = get_name_netfile(grid_num, list_num)
        grid_file = get_name_circuitfile(grid_num, x, y, tot_gates)
        G, cur_orders, all_nets = SRC(grid_file, main_subdir, net_file)
        self.savefile = create_data_directory(main_subdir, grid_num, x, y,
                                              tot_gates, list_num, additions)
        self.G = G
        self.tot_nets = all_nets
        self.batches = batches
        self.batches_size = batch_size


    def run_algorithm(self):
        for i in range(self.batches):
            res = [self.G.get_results(self.G.get_random_net_order()) for _ in
                   range(self.batches_size)]
            write_connections_length_ord(self.savefile, res)




"""
SUBDIR = "circuit_map_100"
GRIDNUM = 0
X = 30
Y = 30
G = 100
BATCH_SIZE = 10
BATCHES = 2
ADDITIONS = ['vannilla', 'A-star']
while True:
    for i in range(10):
        NETLISTNUM = i
        rc = RC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, BATCH_SIZE, BATCHES,
                ADDITIONS)
        rc.run_algorithm()
    GRIDNUM += 1
"""

"""
SUBDIR = "circuit_map_100"
GRIDNUM = 0
X = 30
Y = 30
G = 100
CONSEC_SWAPS = 2
ITERATIONS = 20
ADDITIONS = ['vannilla', 'A-star', 'Hill-Climber']
while True:
    for i in range(10):
        NETLISTNUM = i
        # grid_num, subdir, net_num, x, y, tot_gates, consec_swaps, iterations, additions
        hc = HC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, CONSEC_SWAPS, ITERATIONS,
                ADDITIONS)
        hc.run_algorithm()
    GRIDNUM += 1
"""

SUBDIR = "circuit_map_100"
GRIDNUM = 0
X = 30
Y = 30
G = 100
CONSEC_SWAPS = 2
ITERATIONS = 20
FUNC_PARAMS = ['linear', 100]
ADDITIONS = ['vannilla', 'A-star', 'Simulated-Annealing']
while True:
    for i in range(10):
        NETLISTNUM = i
        # grid_num, subdir, net_num, x, y, tot_gates, consec_swaps, iterations, additions
        sa = SA(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, CONSEC_SWAPS, ITERATIONS,
                FUNC_PARAMS, ADDITIONS)
        sa.run_algorithm()
    GRIDNUM += 1


