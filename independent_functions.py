import os
import functools
import operator

import numpy as np

from shutil import copy
from random import randint
from math import sqrt, floor

# for plotting
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html



###############################################################################
# Sorting                                                                     #
###############################################################################


def swap_two_elems(net_path, prev_swap_ind=-1):
    """ swaps two random elements of a list in place

    :param net_path: net order for a solver function
    :param prev_swap_ind: index of last swapped for a series of swaps
        -1 means two random elements are swapped
        any other number is the index of the first element to be swapped,
            the second is random
    :return: same net order, but with two elements swapped
    """

    npath = list(net_path)[:]
    end_index = len(net_path)-1
    if prev_swap_ind == -1:
        i1 = randint(0, end_index)
    else:
        i1 = prev_swap_ind
    i2 = randint(0, end_index)
    while i2 == i1:
        i2 = randint(0, end_index)
    npath[i1], npath[i2] = npath[i2], npath[i1]
    return npath, i2

def swap_two_X_times(net_path, X):
    npath = list(net_path)[:]
    end_index = len(net_path) - 1
    swaps = []
    while True:
        swaps = [randint(0, end_index) for _ in range(X*2)]
        set_swaps = set(swaps)
        if len(swaps) == len(set_swaps):
            break
        #print("false swaplist", swaps)
    #print("true swaps", swaps)
    for i in range(X):
        npath[swaps[2*i]], npath[swaps[2*i+1]] = npath[swaps[2*i+1]], npath[swaps[2*i]]
    return npath

def swap_two_inplace(net_path):
    """ swaps two random elements of a list in place

    :param net_path: net order for a solver function
    :return: same net order, but with two elements swapped
    """
    end_index = len(net_path) - 1
    i1 = randint(0, end_index)
    i2 = randint(0, end_index)
    while i2 == i1:
        i2 = randint(0, end_index)
    net_path[i1], net_path[i2] = net_path[i2], net_path[i1]
    return net_path


def swap_up_to_x_elems(net_path, x):
    """

    :param net_path: specific netlist order
    :param x: maximum times of two element swap
    :return: list of netlist orders with [2, 3, 4, ... 2+x] elements swapped each
    """
    new_paths = []
    curnew = net_path[:]
    new_order, swapped_with = swap_two_elems(curnew, prev_swap_ind=-1)
    new_paths.append(new_order)
    curnew = new_paths[-1]
    for i in range(x-2):
        new_order, swapped_with = swap_two_elems(curnew, prev_swap_ind=swapped_with)
        new_paths.append(new_order)
        curnew = new_paths[-1]
    return new_paths


def quicksort(arr):
    """
    based on implementation from https://brilliant.org/wiki/quick-sort/
    :param arr:
    :return:
    """
    less = []
    pivotList = []
    more = []
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]
        for i in arr:
            if i[0] > pivot[0]:
                less.append(i)
            elif i[0] < pivot[0]:
                more.append(i)
            else:
                if i[1] < pivot[1]:
                    less.append(i)
                elif i[1] > pivot[1]:
                    more.append(i)
                else:
                    pivotList.append(i)
        less = quicksort(less)
        more = quicksort(more)
    return less + pivotList + more


def combine_score(n_nets, connections, length, *args):
    """ combines cnnections & length components into a single score

    ex: 45 connections, 400  wire length --> 45.9600
        46 connections, 1680 wire length --> 46.8320

    :param n_nets: number of nets given by netlist.
    :param connections: number of connected nets
    :param length: tot length of nets on the circuit
    :return:
    """
    frac_part = float(10000-length)/10000.
    return float(connections)+frac_part


def score_connect_only(n_nets, connections, *args):
    """ Returns objective function evaluation when basing only on number of

    connections

    :param n_nets: number of nets given by netlist.
    :param connections: number of connected nets
    :param args: smoother function handling for other scoring functions,
        arbitrary
    :return: float of number of connected nets
    """
    return float(connections)

def multi_objective_score(n_nets, connections, length, cp, lp):
    """ calculates score for multi objective function evaluation score

    :param connections: number of connected nets
    :param length: tot length of nets on the circuit
    :param cp: connection part
    :param lp: length part
    :return:
    """
    raise(NotImplementedError)

def split_score(combination):
    """ splits score into connections & length component

    ex: 45.9600 --> 45 connections, 400 wire length
        46.832  --> 46 connections, 1680 wire length

    :param combination:  45.9600 from above
    :return: (connected_nets, total_netlength)
    """
    connections = floor(combination)
    length = -10000. * (combination - float(connections)) + 10000
    return connections, length


def order_from_float(float_indeces, nets):
    """ converts list of float space to index space for net order for

    a more accurate ppa implementation

    :param float_indeces: floating point numbers for index representation
    :param nets: nets in a pre-defined order
    :return: net order according to floating point index
    """
    zipped = zip(float_indeces, nets)
    ordered = sorted(zipped, key=lambda x: x[0])
    return [combined[1] for combined in zip(*ordered)]


###############################################################################
#  Filename Generating                                                        #
###############################################################################


def create_fpath(subdir, outf):
    print(outf)
    print(subdir)
    rel_path = os.path.join(subdir, outf)
    script_dir = os.path.dirname(__file__)
    dir_check_path = os.path.join(script_dir, subdir)
    if not os.path.exists(dir_check_path):
        os.mkdir(dir_check_path)
    fpath = os.path.join(script_dir, rel_path)
    return fpath


def get_name_netfile(gridnum, listnum):
    return "C" + str(gridnum) + "_netlist_" + str(listnum) + ".csv"


def get_name_circuitfile(gridnum, x, y, tot_gates):
    return "Gateplatform_" + str(gridnum) + "_" + str(x) + "x" + str(
        y) + "g" + str(tot_gates) + ".csv"


def create_data_directory(main_subdir, gridnum, x, y, tot_gates, listnum, version_specs, dir_specs, Func_specs, ask=True):
    gridfn = get_name_circuitfile(gridnum, x, y, tot_gates)
    netfn = get_name_netfile(gridnum, listnum)
    print(Func_specs)
    grid_net_subdir = gridfn[:-4] + "_" + netfn[:-4] + ' - ' + version_specs + ' - ' + '_'.join(dir_specs)
    rel_path = os.path.join(main_subdir, grid_net_subdir)
    script_dir = os.path.dirname(__file__)
    dir_check_path = os.path.join(script_dir, rel_path)
    if ask:
        ans = input("current additions are as follows:" + str(Func_specs) + "does this match what you are trying to record? (Y/n)")
        while True:
            if ans == 'Y':
                break
            elif ans == 'n':
                print('program terminated')
                exit(0)
            else:
                print('invalid answer, did you perhaps not enter the capital "Y"?')
                ans = input("try input again (Y/n)")
    if not os.path.exists(dir_check_path):
        os.mkdir(dir_check_path)
        copy(create_fpath(main_subdir,gridfn), os.path.join(dir_check_path, gridfn))
        copy(create_fpath(main_subdir,netfn), os.path.join(dir_check_path, netfn))
        lwrite_specs(dir_check_path, Func_specs)
    else:
        ans = input("Continuing will append pre-recorded data\nContinue? (Y/n)")
        while True:
            if ans == 'Y':
                break
            elif ans == 'n':
                print('program terminated')
                exit(0)
            else:
                print('invalid answer, did you perhaps not enter the capital "Y"?')
                ans = input("Continuing will append pre-recorded data\nContinue? (Y/n)")
    return os.path.join(dir_check_path, 'data ' + '_'.join(Func_specs) + '.tsv')


def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)]


def get_res_subdirs(a_dir):
    forbidden = ['.py', 'txt', 'lsx', '.md', 'png']
    _list = [name for name in os.listdir(a_dir) if (name[-3:] not in forbidden)]
    return _list


###############################################################################
#  Circuit Generating                                                        #
###############################################################################

def gates_from_lol(lol):
    """
    :param lol: list of lists with input for each position on the ground
    floor of the grid. ex: [['g0', '0', 'g1],['0','g2','g3']]
    :return: tuple of all gate coordinates for a circuit and the gate numbers
    """
    gate_coords = []
    gates = []
    for x in range(len(lol)):
        for y in range(len(lol[0])):
            gate = lol[x][y]
            if gate[0] == 'g':
                gate_coords.append((x, y))
                gates.append(gate)
    return gate_coords, gates


def read_grid(fpath):
    """
    reads a grid configuration fom the file at the file path
    :return: list of lists
    """
    base = []
    with open(fpath, 'r') as fin:
        for line in fin:
            base.append(line[:-1].split(','))  # [:-1] so no '\n')
    otherbase = []
    for i in range(len(base)):
        otherbase.append(base[:][i])
    return base


def prodsum(iterable):
    """
    :param iterable: list of numbers
    :return: returns the product of all numbers i.e [5,6,2] returns 5*6*2
    """
    return functools.reduce(operator.mul, iterable, 1)


def manhattan(loc1, loc2):
    """
    :param loc1: tuple, coordinate
    :param loc2: tuple, coordinate
    :return: manhattan distance between the two coordinates
    """
    manh_d = sum([abs(loc1[i] - loc2[i]) for i in range(len(loc1))])
    return manh_d


def euclidian(loc1, loc2):
    """
    :param loc1: tuple, coordinate
    :param loc2: tuple, coordinate
    :return: euclidian distance between the two coordinates
    """
    eucl_d = sqrt(sum([abs(loc1[i] - loc2[i])**2 for i in range(len(loc1))]))
    return eucl_d


def count_to_pos(count, params):
    """
    :param count: count is the number of the node being made
    :param params: parameters of the grid
    :return: returns a set of new coordinates to be placed in the griddict
    """
    base = [0]*len(params)
    for i in range(len(params)):
        base[i] = count // prodsum(params[:i]) % params[i]
    return base


def params_inp(params):
    """ return all tuples for a grid of certain size,

    params = (10,10) creates tuples for positions (0, 0), (1, 0), ..., (9,9)

    :return: tuple for every node in a grid with params
    """
    base = [0]*len(params)
    count = 0
    tot = prodsum(params)
    return tuple([tuple(count_to_pos(c, params)) for c in range(tot)])


def neighbours(coords):
    """
    :param - tuple: tuple of coordinates of a point in the grid:
    :return: neighbouring nodes in the grid
    """
    rl = []
    for i in range(len(coords)):
        temp1 = list(coords)
        temp2 = list(coords)
        temp1[i] -= 1
        temp2[i] += 1
        rl.extend((tuple(temp1), tuple(temp2)))
        rl[-2], rl[0] = rl[0], rl[-2]
    return tuple(rl)


###############################################################################
#  funcs for printing & writing                                               #
###############################################################################


def transform_print(val, advanced_heuristics):
    if advanced_heuristics:
        if val == '0':
            return '___'
        elif val[0] == 'n':
            return val[1:].zfill(3)
        elif val[0] == 'g':
            return val[1:].zfill(3)
        else:
            raise NotImplementedError
    else:
        if val == '0':
            return '__'
        elif val[0] == 'n':
            return val[1:].zfill(2)
        elif val[0] == 'g':
            return 'GA'
        else:
            raise NotImplementedError


def lprint(some_list):
    for elem in some_list:
        print(elem)


def print_start_iter(gridnum, netnum, algorithm, iteration):
    print("############################")
    print("Algorithm", algorithm)
    print("Grid", gridnum, "\nnet", netnum)
    print("Starting iteration", iteration)
    print("############################")


def print_final_state(grid, best_order, best_len, \
                      nets_solved, tot_nets):
    print(grid)
    print("Final Path =\t", best_order, )
    print("Final Length =\t", best_len)
    print("All connected =\t", nets_solved, "/", tot_nets)


def write_connections_length_ord(filename, con_len_list):
    w_str = '\n'.join(['\t'.join([','.join([str(l) for l in m]) for m in n])
                       for n in con_len_list])
    with open(filename, 'a') as f:
        f.write(w_str)
        f.write('\n')

def write_connections_gen(filename, gen_lst):

    with open(filename, 'a') as f:
        f.write()

def writebar(filename, *extra):
    """ appends a breakline to a file, representing the end of a generation

    :param filename: file to append to
    :param extra: extra information at the breakline
    """
    with open(filename, 'a') as f:
        f.write('#### ' + ' '.join([*extra]) + '\n')


def write_connections_length(filename, con_len_list):
    """ appends result data to file, first connections then total length,

    separated by a comma (',')

    :param filename:
    :param con_len_list:
    """
    w_str = '\n'.join(['\t'.join([','.join([str(l) for l in m]) for m in n[:-1]]) for n in con_len_list])
    with open(filename, 'a') as f:
        f.write(w_str)
        f.write('\n')

def lwrite_specs(subdir, additions):
    with open(os.path.join(subdir, "verbose_specs.txt"), 'w') as wf:
        for spec in additions:
            wf.write(spec+'\n')
