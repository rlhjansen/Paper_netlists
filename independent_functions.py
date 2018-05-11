import os
import functools
import operator

from shutil import copy

from random import randint
from math import sqrt, floor

# for plotting
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



###### Plotting ########################
# circuit
########################################
shapes_string = "- -- -. :"
SHAPES = shapes_string.split(' ')
COLOURS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

col_len = len(COLOURS)

def get_markers(n):
    return [SHAPES[i%4] + COLOURS[i%col_len] for i in range(n)]


def paths_to_plotlines(paths):
    """ transforms paths format from the grid to a series of plottable points

    :param paths: list of tuples of tuples
        each outer tuple represents a path for how a certain net is laid
        each inner tuple represents a specific (x,y,z) location on the circuit
    :return xpp, ypp, zpp: multiple list of lists where the inner list is a
        series of points to be plotted for a single netlist (on x,y,z axis
        respectively)
    """
    xpp = []
    ypp = []
    zpp = []

    for path in paths:
        print(path)
        pxs = [spot[0] for spot in path]
        pys = [spot[1] for spot in path]
        pzs = [spot[2] for spot in path]

        xpp.append(pxs)
        ypp.append(pys)
        zpp.append(pzs)
    return xpp, ypp, zpp


def remove_empty_paths(paths, order):
    clean_paths = []
    clean_order = []
    for i, path in enumerate(paths):
        print(path)
        if len(path)-1:
            print("added")
            clean_paths.append(path)
            clean_order.append(order[i])
        else:
            print("not added")
    print(len(clean_order))
    return clean_paths, clean_order


def plot_circuit(paths, order, gates, gate_tags):
    """ Plots the complete circuit in 3D, filled by way of the paths

    This is a temporary function as a prelude to further visualisation

    :param paths: List of tuples of tuples
        Each outer tuple represents a path for how a certain net is laid
        Each inner tuple represents a specific (x,y,z) location on the circuit
    :param order: Instance of an order of netlists
    :param gates: List of tuples, each representing a gate on the circuit
    :param gate_tags: The names of the gates (i.e. g1, g2, g3) in the same
        order as in the gates list
    """
    #TODO add gates to plot
    #TODO add algorithm specifics to plot-title
    #TODO add max dimensions of plot (to resemble circuit)
    original_len = len(order)
    c_paths, c_order = remove_empty_paths(paths, order)
    xpp, ypp, zpp = paths_to_plotlines(c_paths)
    xgs, ygs, zgs = split_gates(gates)
    markers = get_markers(len(xpp))
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    plotcount = len(xpp)
    ax.set_title(create_plot_title(original_len, plotcount))

    for i in range(plotcount):
        ax.plot(xpp[i], ypp[i], zpp[i], markers[i], label=c_order[i])

    ax.scatter3D(xgs, ygs, zgs)
    fig.savefig("temp_plus_gates.png")


def create_plot_title(original_len, placed):
    return "netlist_placement for " + str(placed) +" out of" + str(original_len) + \
           "nets"


def split_gates(gates):
    print("gates", gates)
    for gate in gates:
        print(gate)
    xgs = [gate[0] for gate in gates]
    ygs = [gate[1] for gate in gates]
    zgs = [gate[2] for gate in gates]
    return xgs, ygs, zgs


###### Plotting ########################
# results
########################################

def file_vars(fname):
    connections = []
    lengths = []
    with open(fname, 'r') as f:
        for line in f:
            line = f.readline()
            line = line.split("\t")[0]
            line = line.split(",")
            connections.append(line[0])
            lengths.append(line[1])
    return connections, lengths

def plot_values(series, labels):
    fig = plt.figure()
    plot_len = len(series[0])
    xs = [i for i in range(plot_len)]
    for i, values in enumerate(series):
        plt.plot(xs, values, label=labels[i])
    #Todo make appropraite per results
    fig.savefig(",".join(labels)+".png")



###### Sorting ########################
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
    # print("", net_path)
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

def combine_score(connections, length):
    frac_part = float(10000-length)/10000.
    return float(connections)+frac_part

def split_score(combination):
    print("combination", combination)
    connections = floor(combination)
    print("connections", connections)
    length = -10000. * (combination - float(connections)) + 10000
    print("length", length)
    return connections, length

###### Filename Generating ############
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

def create_data_directory(main_subdir, gridnum, x, y, tot_gates, listnum, additions, ask=True):
    gridfn = get_name_circuitfile(gridnum, x, y, tot_gates)
    netfn = get_name_netfile(gridnum, listnum)
    new_subdir = '_'.join(additions) + '_' + gridfn[:-4] + "_" + netfn
    rel_path = os.path.join(main_subdir, new_subdir)
    script_dir = os.path.dirname(__file__)
    dir_check_path = os.path.join(script_dir, rel_path)
    if ask:
        ans = input("current additions are as follows:" + str(additions) + "does this match what you are trying to record? (Y/n)")
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
    return os.path.join(dir_check_path, 'data ' + '_'.join(additions) + '.tsv')

def get_subdirs(a_dir):
    return [name for name in os.listdir(a_dir)]

def get_res_subdirs(a_dir):
    forbidden = ['.py', 'txt', 'lsx', '.md', 'png']
    _list = [name for name in os.listdir(a_dir) if (name[-3:] not in forbidden)]
    return _list


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
    """
    :param - params: parameters of a grid to be created
    :yield: node for a grid with size of parameters.
    params = (10,10) yields (0, 0), (1, 0), ..., (9,9)
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



###### funcs for printing ############
def transform_print(val, Advanced_heuristics):
    if Advanced_heuristics:
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

def printbar(n):
    print('#'*n)
    print('#'*n)


def write_connections_length_ord(filename, con_len_list):
    print(con_len_list)
    w_str = '\n'.join(['\t'.join([','.join([str(l) for l in m]) for m in n]) for n in con_len_list])
    with open(filename, 'a') as f:
        f.write(w_str)
        f.write('\n')

def writebar(filename, *extra):
    with open(filename, 'a') as f:
        f.write('#### ' + ' '.join([*extra]) + '\n')



def write_connections_length(filename, con_len_list):
    print(con_len_list)
    w_str = '\n'.join(['\t'.join([','.join([str(l) for l in m]) for m in n[:-1]]) for n in con_len_list])
    with open(filename, 'a') as f:
        f.write(w_str)
        f.write('\n')

