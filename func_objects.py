from random import random
from math import exp, log, tanh, ceil

from Gridfile import SXHC, SPPA, SRC
from independent_functions import swap_up_to_x_elems, swap_two_elems, \
    get_name_circuitfile, get_name_netfile, quicksort, print_start_iter, \
    print_final_state, create_data_directory, write_connections_length_ord, write_connections_length



#Todo call save_prompt only once


class HC:
    def __init__(self, subdir, grid_num, net_num, x, y, tot_gates,
                 consec_swaps, iterations, additions, chance_on_same=.5):
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
        self.sol_conn = 0
        self.all_connected = False
        self.chance = chance_on_same
        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              tot_gates, net_num, additions)

    def check_climb(self, cur_conn, cur_ord, cur_len):
        if not self.all_connected:
            if cur_conn > self.sol_conn:
                self.sol_ord = cur_ord
                self.sol_conn = cur_conn
                print("changed :: greater")
                if cur_conn == self.tot_nets:
                    self.sol_len = cur_len
                    self.all_connected = True
                    print("all_connected")
            elif cur_conn == self.sol_conn:
                print("enter chance conn")
                if self.chance < random():
                    print("changed :: equal")
                    self.sol_ord = cur_ord
                    self.sol_conn = cur_conn
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
                        self.sol_ord = cur_ord
                        self.sol_conn = cur_conn
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
            print("current best =", self.sol_conn, self.sol_len)

            self.check_climb(cur_conn, cur_ord, cur_len)
        self.G.solve_order(self.sol_ord)
        print_final_state(self.G, self.sol_ord, self.sol_len, \
                          self.sol_conn, self.tot_nets)


class SA:
    def __init__(self, subdir, grid_num, net_num, x, y, tot_gates,
                 consec_swaps, iterations, function_params, additions):
        """Initializes the Simulated Annealing solver

        :param subdir: subdirectory of circuit & netfiles
        :param grid_num: circuit number
        :param net_num: net number
        :param x: x parameter of circuit
        :param y: y parameter of circuit
        :param tot_gates: total gates of a circuit
        :param consec_swaps: consecutive swaps per step
        :param iterations: total number of climbing iterations
        :param function_params: parameters to select how to anneal
            see determine anneal
        :param additions:
        """
        net_file = get_name_netfile(grid_num, net_num)
        grid_file = get_name_circuitfile(grid_num, x, y, tot_gates)
        G, cur_orders, all_nets = SXHC(grid_file, subdir, net_file, consec_swaps)

        self.all_connected = False
        self.iterations = iterations
        self.G = G
        self.name = "SA"
        self.sol_len = 5000000
        self.sol_conn = 0
        self.sol_ord = cur_orders[0]
        self.tot_nets = all_nets
        self.func = function_params[0]
        self.swaps = consec_swaps
        self.sol_conn = 0
        self.determine_anneal(function_params)
        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              tot_gates, net_num, additions)
        self.annealFunc = None



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
                write_connections_length_ord(self.savefile, [[(cur_conn, cur_len), cur_ord]])
                self.check_anneal(cur_conn, cur_ord, cur_len, i)

        self.G.solve_order(self.sol_ord)
        print_final_state(self.G, self.sol_ord, self.sol_len, \
                          self.sol_conn, self.tot_nets)

    ################################################################
    # Anneal Checks                                                #
    ################################################################

    def check_all_anneal(self, cur_conn, cur_ord, cur_len, i):
        if not self.all_connected:
            print(cur_conn)
            if cur_conn > self.sol_conn:
                self.sol_ord = cur_ord
                self.sol_conn = cur_conn
                print("change sol order :: greater")
                if cur_conn == self.tot_nets:
                    self.sol_len = cur_len
                    self.all_connected = True
                    print("all_connected")
            elif cur_conn <= self.sol_conn:
                print("enter anneal conn")
                if self.anneal_conn(cur_conn, i):
                    print("change sol order :: equal")
                    self.sol_ord = cur_ord
                    self.sol_conn = cur_conn
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
                    print("unchanged anneal len")



    def check_len_anneal(self, cur_conn, cur_ord, cur_len, i):
        if self.sol_len > cur_len:
            self.sol_conn, self.sol_ord, self.sol_len = cur_conn, cur_ord, cur_len
        elif self.anneal_len(cur_len, i):
            print("enter anneal len")
            self.sol_conn, self.sol_ord, self.sol_len = cur_conn, cur_ord, cur_len
        else:
            print("unchanged anneal len")


    def check_conn_anneal(self, cur_conn, cur_ord, cur_len, i):
        if cur_conn > self.sol_conn:
            self.sol_conn, self.sol_ord, self.sol_len = cur_conn, cur_ord, cur_len
        elif self.anneal_len(cur_len, i):
            print("enter anneal conn")
            self.sol_conn, self.sol_ord, self.sol_len = cur_conn, cur_ord, cur_len
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
        self.T -= self.T_step
        if random() < chance:
            return True
        else:
            return False

    def linear_anneal_conn(self, cur_conn, i):
        """
        :param i: iteration number i
        :returns: whether or not to accept an outcome
        """
        print('self.T', self.T)
        print('i', i)
        print('self.T_step', self.T_step)
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
            print("returned True, changing")
            return True
        else:
            print("returned False, not changing")
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
        return self.st*(self.et/self.st)**(i/self.iterations)

    def determine_anneal(self, function):
        """
        for geman see Nourani and Andresen 'A comparison of simulated annealing cooling schedules' p.3

        :param function: per index:
            0 'length', 'connections', 'all'
            1 (anneal function type) 'linear', 'log', 'exp', 'geman'
            2+ algorithms specific
        """
        anneal_on = function[0]
        if anneal_on == 'all':
             self.annealFunc = self.check_all_anneal
        elif anneal_on == 'length':
            self.annealFunc = self.check_len_anneal
        elif anneal_on == 'connections':
            self.annealFunc = self.check_conn_anneal

        functype = function[1]
        if functype == 'linear':
            self.anneal_conn = self.linear_anneal_conn
            self.anneal_len = self.linear_anneal_len
            self.T = function[2]
            self.T_step = function[2]/(self.iterations+1)
            self.T = function[2]
            self.T_step = function[2] / (self.iterations + 1)
        elif functype == 'log':
            self.T_start = function[2] #Temperature start
            self.anneal_conn = self.logarithmic_anneal_conn
            self.anneal_len = self.logarithmic_anneal_len
        elif functype == 'exp':
            self.st = function[2] #start temperature
            self.et = function[3] #end temperature
            self.anneal_conn = self.exp_anneal_conn
            self.anneal_len = self.exp_anneal_len
        elif functype == 'geman':
            self.anneal_len = self.geman_anneal_len
            self.anneal_conn = self.geman_anneal_conn
            self.c = function[2] #largest barrier, arbitrarily chosen
            self.d = function[3] #arbitrarily chosen





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
            write_connections_length(self.savefile, res)


class PPA:
    """PPA is the Plant Propagation solver class.

    for info check the original paper by Salhi and Frage
    """
    # Todo implement a last_generation attribute, to save computation time
    # Todo save (maximum) generation (or count)
    def __init__(self, subdir, grid_num, net_num, x, y, g, max_generations,
                elitism=0, pop_cut=20, max_runners=4):
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
        """
        gridfile = get_name_circuitfile(grid_num, x, y, g)
        netfile = get_name_netfile(grid_num, net_num)
        self.gn = grid_num
        self.nn = net_num
        self.elitism = elitism
        self.gens = max_generations
        self.pop_cut = pop_cut
        self.max_runners = max_runners

        G, initial_pop, all_nets = SPPA(gridfile, subdir, netfile, pop_cut)
        self.pop = initial_pop
        self.tot_nets = all_nets
        self.G = G
        if elitism:
            FUNC_PARAMS = ["pop" + str(pop_cut), "max_runners" + str(max_runners), \
                           "elite" + str(elitism)]
        else:
            FUNC_PARAMS = ["pop" + str(pop_cut), "max_runners" + str(max_runners), \
                           "all_elite"]
        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              g, net_num, FUNC_PARAMS)

    def ranks_to_newpop(self, scores, orders):
        """ runners & distances from original paper by Salhi & Fraga

        scoring function yet not yet justified

        :param orders: best placement orders of last generaion
        :returns: new population plants to be considered
        """
        newpop = []
        mapped_ranks = [0.5 * (tanh(4*score - 2) + 1) for score in scores]
        runners = []
        for mrank in mapped_ranks:
            value = ceil(self.max_runners*mrank*random())
            runners.append(value)
            print("mrank", mrank)
            print("value", value, "\n")

        distances = [2*(mrank)*(random()-0.5) for mrank in mapped_ranks]

        if True:
            print("distances", distances)
            print("mapped ranks", mapped_ranks)
            print("runners", runners)
            print("len orders", len(orders))
            print("len runners", len(runners))

        swaplist = []
        for i, rcount in enumerate(runners):
            rbase = 1.0
            cur_swaps = []
            for _ in range(rcount):
                rbase += distances[i]*self.max_runners
                print(rbase)
                cur_swaps.append(ceil(rbase))
                new_ords = swap_up_to_x_elems(orders[i], ceil(rbase))
                newpop.append(new_ords[0])
                rbase = rbase % 1
            swaplist.append(cur_swaps)
        print("swaplist", swaplist)
        return newpop



    def populate(self, qslist):
        """ Sets a new population for the following iteration.

        :param qslist: elements are tuple(nets solved, solution length,
                                          net order)
        :variable ranks: fitness/ranking function should be the following type:
            f(z) = (Zmax - z)/(Zmax-Zmin), with Zmax & min being objective
            maximum & minimum functions
        """
        new_pop = []
        if self.elitism:
            for i in range(self.elitism):
                new_pop.append(qslist[i][2])
        else:
            for i, _ in enumerate(qslist):
                print(_)
                new_pop.append(qslist[i][2])

        orderlist = [qslist[i][2] for i in range(len((qslist)))]
        qslen = len(qslist)

        ranks = [(i+0.5)/qslen for i in range(qslen)]  # maps linearly to linspace between (0.0, 1.0)
        created_pop = self.ranks_to_newpop(ranks, orderlist)
        new_pop.extend(created_pop)

        self.sol_ord = list(qslist[0][2])
        self.sol_conn = qslist[0][0]
        self.sol_len = qslist[0][1]

        self.pop = new_pop


    def run_algorithm(self):
        """main loop, tries "plants"/orders per generation.

        """
        for i in range(self.gens):
            print_start_iter(self.gn, self.nn, "Plant Propagation", 2, i+1)
            data_clo = []  # conn, len, order
            for i, cur_ord in enumerate(self.pop):
                satisfies = self.G.solve_order(cur_ord)
                cur_conn, cur_len = satisfies
                self.pop[i] = cur_ord
                data_clo.append((cur_conn, cur_len, tuple(cur_ord)))
                self.G.reset_nets()
                write_connections_length_ord(self.savefile, [
                    [(cur_conn, cur_len), cur_ord]])
            qslist = quicksort(data_clo)
            self.sol_len = qslist[1]
            self.populate(qslist[:self.pop_cut])

            self.G.solve_order(self.pop[0])
            print(self.G)
            print("Current best path =\t", self.sol_ord, "\nCurrent best "
                                                         "length =\t",
                  self.sol_len)
            self.G.reset_nets()

        self.G.solve_order(self.pop[0])
        print(self.G)
        print("Final Path =\t", self.pop[0], "\nFinal Length =\t", \
              self.sol_len)


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

"""
SUBDIR = "circuit_map_100"
GRIDNUM = 0
X = 30
Y = 30
G = 100
CONSEC_SWAPS = 2
ITERATIONS = 20
FUNC_PARAMS = ['linear', 100]
ADDITIONS = ['vannilla', 'A-star', 'Simulated-Annealing']  #for simulated annealing
"""

GRIDNUM = 0
NETLISTNUM = 0
X = 30
Y = 30
G = 100
NETL_LEN = 100
CONSEC_SWAPS = 2
ITERATIONS = 1500
GENERATIONS = 20
ELITISM = 0
SUBDIR = "circuit_map_" + str(NETL_LEN)


#Todo SA possible with different values to anneal con & len
ANN_FUNC_PARAMS_LEN = ["length", "geman", 700, 10, None, None]
ANN_FUNC_PARAMS_CONN = ["connections", "geman", None, None, 100, 1]
ANN_FUNC_PARAMS_BOTH = ["all", "geman", 700, 10, 100, 1]

RC_ADDITION = ["random collector", "A-star, " + str(NETL_LEN) + "_length NL"]
HC_ADDITION = ["Hillclimber", "A-star,, " +str(NETL_LEN) + "_length NL"]
SA_LEN_ADDITION = ["Simulated Annealing Len", "A-star, " +str(NETL_LEN) + "_length NL"]
SA_CONN_ADDITION = ["Simulated Annealing Con", "A-star, " +str(NETL_LEN) + "_length NL"]
SA_ALL_ADDITION = ["Simulated Annealing ConLen", "A-star, " +str(NETL_LEN) + "_length NL"]


while True:
    if False:
        rc = RC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, 30, NETL_LEN, RC_ADDITION)
        rc.run_algorithm()
    if False:
        hc = HC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, 30, NETL_LEN, HC_ADDITION)
        hc.run_algorithm()
    if False:  #SA length
        sa = SA(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, CONSEC_SWAPS, ITERATIONS,
                ANN_FUNC_PARAMS_BOTH, SA_LEN_ADDITION)
        sa.run_algorithm()
    if False:  #SA connections
        sa = SA(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, CONSEC_SWAPS, ITERATIONS,
                ANN_FUNC_PARAMS_BOTH, SA_CONN_ADDITION)
        sa.run_algorithm()
    if False:  #SA all
        sa = SA(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, CONSEC_SWAPS, ITERATIONS,
                ANN_FUNC_PARAMS_BOTH, SA_ALL_ADDITION)
        sa.run_algorithm()
    if True:   # PPA standard
        ppa = PPA(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, GENERATIONS,
                elitism=0, pop_cut=20, max_runners=5)
        ppa.run_algorithm()
    NETLISTNUM += 1
    if NETLISTNUM > 9:
        break



