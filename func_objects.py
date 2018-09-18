""" Program to acquire data, config near the bottom (ln. 650~ish)

"""

from random import random
from math import exp, log, tanh, ceil
import multiprocessing as mp
#import pathos.multiprocessing as mp

import sys

#sys.setrecursionlimit(9000)

from Gridfile import SXHC, SPPA, SRC
from independent_functions import \
    create_data_directory, \
    combine_score, \
    get_name_circuitfile, \
    get_name_netfile, \
    multi_objective_score, \
    print_start_iter, \
    print_final_state, \
    quicksort, \
    score_connect_only, \
    split_score, \
    swap_two_X_times, \
    write_connections_length, \
    write_connections_length_ord, \
    writebar

class HC:
    def __init__(self, subdir, grid_num, net_num, x, y, tot_gates,
                 swaps, iterations, additions, chance_on_same=.5, ask=True):
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
        G, cur_ords, tot_nets = SXHC(grid_file, subdir, net_file, swaps)
        self.name = "HC"
        self.gn = grid_num
        self.nn = net_num
        self.G = G
        self.iterations = iterations
        self.cur_ords = cur_ords
        self.sol_len = 5000
        self.sol_ord = cur_ords[0]
        self.tot_nets = tot_nets
        self.swaps = swaps
        self.sol_conn = 0
        self.all_connected = False
        self.chance = chance_on_same
        additions += [str(swaps) + "swaps",
                      str(iterations) + "iterations"]
        self.savefile = create_data_directory(subdir, grid_num, x, y, tot_gates, net_num, additions, ask=ask)

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
            print_start_iter(self.gn, self.nn, self.name, i)
            if self.sol_ord:
                cur_ords = swap_two_X_times(self.sol_ord, 1) #HARD CODED

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
        additions += [str(consec_swaps) + "swaps",
                      str(iterations) + "iterations"] + function_params

        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              tot_gates, net_num, additions)
        self.annealFunc = None

    def run_algorithm(self):
        for i in range(self.iterations):
            print_start_iter(1, 3, self.name, self.swaps, i)
            if self.sol_ord:
                cur_ords = swap_two_X_times(self.sol_ord, self.swaps)

            data_lco = []  # conn, len, order
            for j, cur_ord in enumerate(cur_ords):
                cur_len, cur_conn = self.G.solve_order(cur_ord)
                data_lco.append((cur_conn, cur_len, tuple(cur_ord)))
                self.G.reset_nets()
                write_connections_length_ord(self.savefile, [[(cur_conn, cur_len), cur_ord]])
                self.check_anneal(cur_conn, cur_ord, cur_len, i)

        self.G.solve_order(self.sol_ord)
        print_final_state(self.G, self.sol_ord, self.sol_len,
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
            self.T_start = function[2] # Starting Temperature
            self.anneal_conn = self.logarithmic_anneal_conn
            self.anneal_len = self.logarithmic_anneal_len
        elif functype == 'exp':
            self.st = function[2] # Starting Temperature
            self.et = function[3] # End Temperature
            self.anneal_conn = self.exp_anneal_conn
            self.anneal_len = self.exp_anneal_len
        elif functype == 'geman':
            self.anneal_len = self.geman_anneal_len
            self.anneal_conn = self.geman_anneal_conn
            self.c = function[2] # largest barrier, arbitrarily chosen
            self.d = function[3] # arbitrarily chosen





class RC:
    """
    RC or Random Collector Takes in a Grid & Netlist.
    Its goal is to apply the netlist, in random order, to the grid.
    while it does this it collects data on the number of connections, as well
    as their total length
    """
    def __init__(self, main_subdir, grid_num, list_num,  x, y, tot_gates,
                 additions, batches, batch_size=100, ask=True):
        net_file = get_name_netfile(grid_num, list_num)
        grid_file = get_name_circuitfile(grid_num, x, y, tot_gates)
        G, cur_orders, all_nets = SRC(grid_file, main_subdir, net_file)
        self.savefile = create_data_directory(main_subdir, grid_num, x, y,
                                              tot_gates, list_num, additions, ask=ask)
        self.G = G
        self.tot_nets = all_nets
        self.batches = batches
        self.batches_size = batch_size


    def run_algorithm(self):
        for i in range(self.batches):
            res = [self.G.get_results(self.G.get_random_net_order()) for _ in
                   range(self.batches_size)]
            write_connections_length(self.savefile, res)
            writebar(self.savefile, "Batch " + str(i) + " out of " + str(self.batches))


class PPA:
    """PPA is the Plant Propagation solver class.

    for info check the original paper by Salhi and Frage
    """
    def __init__(self, subdir, grid_num, net_num, x, y, g, max_generations, version_specs=[], height=7, elitism=0, pop_cut=20, max_runners=5, max_distance=5, objective_function_value=False, multi_objective_distribution=False, ref_pop=None, workercount=None, ask=True, solvertype="Astar", algver="Selamoglu", arbitrary=24, best_percent=0.2, best_ordering="max"):
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
        :param multi_objective: fitness based on [0.X, 0.Y] percentages for
            multi optimization X for connections, Y for length
        :param connect_only: fitness based on number of connections only
        :param workercount: so far this is only for multiple grid creation.
            multiple grids need be created as pythons deepcopy is unable to
            handle highly linked objects (the grid) due to recursiondepth/implementation

            NOTE - Nothing concerning multiprocessing has yet been done.
        """
        gridfile = get_name_circuitfile(grid_num, x, y, g)
        netfile = get_name_netfile(grid_num, net_num)
        self.workercount = workercount
        self.gn = grid_num
        self.nn = net_num
        self.elitism = elitism
        self.gens = max_generations
        self.pop_cut = pop_cut
        self.pop_score = [0]*pop_cut  # Only relevant for Selamoglu.
        self.max_runners = max_runners
        self.max_distance = max_distance

        Gs, initial_pop, all_nets = SPPA(gridfile, subdir, netfile, height, pop_cut,
                                        solvertype, gridcopies=200, ref_pop=ref_pop)

        self.pop = initial_pop
        self.tot_nets = all_nets
        self.Gs = Gs
        self.last_pop = tuple((),)
        self.last_scores = tuple((),)
        self.solvertype = solvertype
        self.best_ordering = best_ordering
        self.objective_function_value = objective_function_value
        if algver == "Selamoglu":
            self.run = self.run_selamoglu
            self.arbitrary = arbitrary
            self.best_percent = best_percent
        elif algver == "Standard":
            self.run = self.run_algorithm
        else:
            print("no sufficient algorithms version input\n Use algver='Selamoglu' or 'Standard'")
        self.algver = algver


        func_params, dir_specs = self.param_to_specs()

        if multi_objective_distribution:
            func_params.append("multi_objective_distribution" + str('+'.join(multi_objective_distribution)))
            dir_specs.append("mod" + str('+'.join(multi_objective_distribution)))

        if objective_function_value:
            if objective_function_value == "combined":
                self.ofv = combine_score
            elif objective_function_value == "multi_objective":
                if multi_objective_distribution:
                    self.ofv = multi_objective_score
                    self.mod = multi_objective_distribution
                else:
                    print("no distribution for multi objective function specified input i.e.\n, multi_objective_distribution=[0.5, 0.5]")
                    exit()
            elif objective_function_value == "connect_only":
                self.ofv = score_connect_only

        else:
            print("no optimization method selected (combined, connect_only, multi_objective)")
            exit()


        self.savefile = create_data_directory(subdir, grid_num, x, y,
                                              g, net_num, version_specs, dir_specs, func_params, ask=ask)


    def param_to_specs(self):
        FUNC_PARAMS = ["PPA",
                       self.solvertype,
                       "pop" + str(self.pop_cut),
                       "max_runners" + str(self.max_runners),
                       "max_distance" + str(self.max_distance),
                       "elitism" + str(self.elitism),
                       "generations" + str(self.gens),
                       "objectiveFunction" + str(self.objective_function_value)]

        dir_specs = ["PPA",
                      self.solvertype[:2],
                      "p" + str(self.pop_cut),
                      "mr" + str(self.max_runners),
                      "md" + str(self.max_distance),
                      "e" + str(self.elitism),
                      "g" + str(self.gens),
                      "ofv_" + str(self.objective_function_value)]
        if self.algver == "Selamoglu":
            dir_specs.extend(["vSela",
                              "a"+str(self.arbitrary),
                              "bp"+str(self.best_percent)])

            FUNC_PARAMS.extend(["AlgVersionSelamoglu",
                               "arbitrary" + str(self.arbitrary),
                                "best_percent" + str(self.best_percent)])
        else:
            dir_specs.append("vStdPPA")
            FUNC_PARAMS.append("AlgVersionStandard")


        print("dir specs", dir_specs)
        print("FUNC PARAMS", FUNC_PARAMS)
        return FUNC_PARAMS, dir_specs


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

        :param orders: best placement orders of last generaion
        :returns: new population plants to be considered
        """
        newpop = []
        mapped_ranks = [0.5 * (tanh(4*score - 2) + 1) for score in scores]
        runners = []
        for mrank in mapped_ranks:
            value = ceil(self.max_runners*mrank*random())
            runners.append(value)

        swap_list = []
        for i, runner_count in enumerate(runners):
            cur_swaps = []
            for _ in range(runner_count):
                distance = (1-mapped_ranks[i])*random()*self.max_runners
                cur_swaps.append(ceil(distance))
                new_ord = swap_two_X_times(orders[i], ceil(distance))
                newpop.append(new_ord)
            swap_list.append(cur_swaps)
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

        scores = self.last_scores + nscores
        orders = self.last_pop + norderlist
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

        *  Elitism: amount of the most recent selection that is retained
            in the new generation

        """

        qslen = len(qslist)
        scores = tuple([self.ofv(qslist[i][0], qslist[i][1], scoring=self.best_ordering, total_nets=self.tot_nets) for i in range(qslen)])
        orderlist = tuple([qslist[i][2] for i in range(qslen)])

        tot = self.combine_old_new(scores, orderlist)
        plant_scores = tot[0][:self.pop_cut]
        plant_orders = tot[1][:self.pop_cut]
        best_con, best_len = split_score(plant_scores[0])


        self.last_pop = tot[1]
        self.last_scores = tuple(tot[0])

        zMax = max(plant_scores)
        zMin = min(plant_scores)
        if zMax == zMin:
            fitnesses = [0.5 for _ in plant_scores]
        else:
            fitnesses = [(score-zMin)/(zMax-zMin) for score in plant_scores]

        self.pop = self.fitnesses_to_newpop(fitnesses, plant_orders)

        self.sol_ord = plant_orders[0]
        self.sol_conn = best_con
        self.sol_len = best_len


    def run_selamoglu(self):
        """ See paper Selamoglu & Salhi:

        The Plant Propagation Algorithm for Discrete Optimisation:
            The Case of the Travelling Salesman Problem


        :param best_percent: best X percent get close runners
            proportional to arbitrary
        :param arbitrairy: dependency for close runners
        Optional:
        :return:

        (Note, looks like multiple paralel hillclimbers)
        """
        #Todo customize saving to view how plants propagate
        #Todo Distance function implementation

        self.initial_score_Selamoglu()
        print("Initial scoring done")
        for i in range(self.gens):
            next_ords = []
            popsize = ceil(len(self.pop))
            best_plants = ceil(popsize * self.best_percent)
            print("Best plants", best_plants)
            for j in range(best_plants):
                print("making", self.arbitrary/(j+1))
                next_ords.extend([(swap_two_X_times(self.pop[j], 1), j) for _ in range(ceil(self.arbitrary/(j+1)))])
                print("best plant", j, " produces",  len(next_ords[j]))
                print(next_ords[j])

            next_ords.extend([(swap_two_X_times(self.pop[j], self.max_distance), j) for j in range(best_plants, popsize, 1)])



            print_start_iter(self.gn, self.nn, "Plant Propagation Selamoglu", i + 1)

            pool = mp.Pool(processes=self.workercount)
            data_clo = pool.map(multi_selamoglu, [(self.Gs[ind], next_ords[ind]) for ind in range(len(next_ords))])
            pool.close()

            writebar(self.savefile, "generation", str(i + 1))
            write_connections_length(self.savefile, data_clo)

            for inst in data_clo:

                (cur_conn, cur_len, cur_order, index) = inst
                print("comparing for offspring of", index)
                runner_eval = combine_score(cur_conn, cur_len, scoring=self.best_ordering, total_nets=self.tot_nets)
                if runner_eval > self.pop_score[index]:
                    self.pop[index], self.pop_score[index] = cur_order, runner_eval
            zipped = zip(self.pop, self.pop_score)

            if self.best_ordering == "max":
                resorted = list(zip(*sorted(zipped, key=lambda x: -x[1])))
            if self.best_ordering == "min":
                resorted = list(zip(*sorted(zipped, key=lambda x: x[1])))

            self.pop, self.pop_score = list(resorted[0]), list(resorted[1])


    def initial_score_Selamoglu(self):
        pool = mp.Pool(processes=self.workercount)
        data_clo = pool.map(multi_selamoglu,
                            [(self.Gs[ind], (self.pop[ind], ind)) for ind in
                             range(len(self.pop))])
        pool.close()
        scores = []
        for inst in data_clo:
            (cur_conn, cur_len, cur_order, index) = inst
            print("comparing for offspring of", index)
            scores.append(combine_score(cur_conn, cur_len,
                                        scoring=self.best_ordering,
                                        total_nets=self.tot_nets))

        print(scores)
        if self.best_ordering == "max":
            qslist = [x for _,x in sorted(zip(scores, data_clo), key=lambda x: -x[0])]
        if self.best_ordering == "min":
            print("enter min")
            qslist = [x for _,x in sorted(zip(scores, data_clo), key=lambda x:x[0])]

        print("savefile before writing", self.savefile)
        writebar(self.savefile, "generation", "0")
        write_connections_length(self.savefile, data_clo)
        print("initial scoring")
        for inst in qslist:
            print(inst)

        for inst in qslist:
            (cur_conn, cur_len, cur_order, index) = inst
            runner_eval = combine_score(self.tot_nets, cur_conn, cur_len, scoring=self.best_ordering, total_nets=self.tot_nets)
            self.pop[index], self.pop_score[index] = cur_order, runner_eval

    def run_algorithm(self):
        """main loop, generates evaluates & saves "plants" per generation.

        """
        #Todo customize option to save amount of swaps with kwarg
        #Todo tournament based fitness? - not for now
        #Todo Distance function implementation (swap, reverse, other)
        #todo remove quicksort function in favor of lambda style function like in selamoglu


        for i in range(self.gens):
            print_start_iter(self.gn, self.nn, "Plant Propagation", i+1)
            data_clo = []  # conn, len, order

            pool = mp.Pool(processes=self.workercount)
            data_clo = pool.map(multi_run, [(self.Gs[ind], self.pop[ind]) for ind in range(len(self.pop))])
            pool.close()
            print("savefile before writing", self.savefile)
            writebar(self.savefile, "generation", str(i))
            write_connections_length(self.savefile, data_clo)
            qslist = quicksort(data_clo)
            self.sol_len = qslist[1]
            self.populate(qslist[:self.pop_cut])

            #self.Gs[0].solve_order(self.last_pop[0], _print=True)
            #print("Current best path =\t", self.sol_ord,"\nCurrent best length =\t", self.sol_len)

        self.Gs[0].solve_order(self.last_pop[0], _print=True)
        print("Final Path =\t", self.last_pop[0], "\nFinal Length =\t",
              self.sol_len)

def multi_run(gps):
    gps[0].connect()
    satisfies = gps[0].solve(gps[1])

    cur_conn, cur_len, tot_tries = satisfies[:3]
    plant_data = (cur_conn, cur_len, tuple(gps[1]))
    return plant_data

def multi_selamoglu(gpis):
    gpis[0].connect()
    satisfies = gpis[0].solve(gpis[1][0])

    cur_conn, cur_len, tot_tries = satisfies[:3]
    plant_data = (cur_conn, cur_len, tuple(gpis[1][0]), gpis[1][1])
    return plant_data


###############################################################################
###  Config
###############################################################################

#General
ASK = False


"""
# created
GRIDNUMS = [0]
NETLISTNUMS = [1]
Xs = [30]
Ys = [30]
Gs = [100]
NETL_LEN = 100

RC_ADDITION = ["random collector", "TEST_A-star, " + str(NETL_LEN) + "_length NL"]
HC_ADDITION = ["Hillclimber", "A-star, " +str(NETL_LEN) + "_length NL"]
SA_LEN_ADDITION = ["Simulated Annealing Len", "A-star, " +str(NETL_LEN) + "_length NL"]
SA_CONN_ADDITION = ["Simulated Annealing Con", "A-star, " +str(NETL_LEN) + "_length NL"]
SA_ALL_ADDITION = ["Simulated Annealing ConLen", "A-star, " +str(NETL_LEN) + "_length NL"]
"""

# Heuristics case
GRIDNUMS = [1, 2]
#GRIDNUMS = [2]

NETLIST_NUMS = [1,2,3]
#NETLIST_NUMS = [3]

Xs = [18, 18]
#Xs = [18]

Ys = [13, 17]
#Ys = [17]

Gs = [25, 50]
#Gs = [50]

#RC
BATCHES = 100

#HC/SA
CONSEC_SWAPS = 1
ITERATIONS = 5000

#PPA
GENERATIONS = 120

ELITISM = 30
POP_CUT = 30

MAX_RUNNERS = 5
MAX_DISTANCE = MAX_RUNNERS

#SUBDIR_OLD = "circuit_map_git_swap_always_2"  # <-- bad data
#SUBDIR_NEW = "circuit_map_git_swap_two_X_times"
SUBDIR_EXP = "Experimental_map"
SUBDIR_HEUR = "heuristics_baseline_circuits"

SOLVER = "elevator"

#Todo SA possible with different values to anneal con & len
ANN_FUNC_PARAMS_LEN = ["length", "geman", 700, 10, None, None]
ANN_FUNC_PARAMS_CONN = ["connections", "geman", None, None, 100, 1]
ANN_FUNC_PARAMS_BOTH = ["all", "geman", 700, 10, 100, 1]


if __name__ == '__main__':
    for j in range(10):
        for ig, k in enumerate(GRIDNUMS):
            for i, NETLIST_NUM in enumerate(NETLIST_NUMS):
                if True:   # PPA standard
                    ppa = PPA(SUBDIR_HEUR, GRIDNUMS[ig], NETLIST_NUM, Xs[ig], Ys[ig], Gs[ig], GENERATIONS,
                              version_specs='RV_test.' +str(j) + '_', elitism=ELITISM,
                              pop_cut=POP_CUT, max_runners=MAX_RUNNERS, max_distance=MAX_DISTANCE,
                              objective_function_value="combined", ask=ASK, height=7,
                              workercount=None, solvertype="A_star", algver="Selamoglu", best_ordering="max")
                    ppa.run()
