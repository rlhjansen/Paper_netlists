import os
import functools
import operator
from shutil import copy

from random import randint
from math import sqrt


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



###### Sorting ########################
def swap_two_elems(net_path):
    npath = list(net_path)[:]
    end_index = len(net_path)-1
    i1 = randint(0, end_index)
    i2 = randint(0, end_index)
    while i2 == i1:
        i2 = randint(0, end_index)
    npath[i1], npath[i2] = npath[i2], npath[i1]
    return npath

def swap_two_inplace(net_path):
    end_index = len(net_path) - 1
    i1 = randint(0, end_index)
    i2 = randint(0, end_index)
    while i2 == i1:
        i2 = randint(0, end_index)
    net_path[i1], net_path[i2] = net_path[i2], net_path[i1]
    return net_path

def swap_up_to_x_elems(net_path, x):
    new_paths = []
    # print("", net_path)
    curnew = net_path[:]
    for i in range(x):
        new_paths.append(swap_two_elems(curnew))
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

def quicksort_DAAL(arr):
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
            if i[0] < pivot[0]:
                less.append(i)
            elif i[0] > pivot[0]:
                more.append(i)
            else:
                if i[1] < pivot[1]:
                    less.append(i)
                elif i[1] > pivot[1]:
                    more.append(i)
                else:
                    pivotList.append(i)
        less = quicksort_DAAL(less)
        more = quicksort_DAAL(more)
        return less + pivotList + more


###### Filename Generating ############
def get_name_netfile(gridnum, listnum):
    return "C" + str(gridnum) + "_netlist_" + str(listnum) + ".csv"

def get_name_circuitfile(gridnum, x, y, tot_gates):
    return "Gateplatform_" + str(gridnum) + "_" + str(x) + "x" + str(
        y) + "g" + str(tot_gates) + ".csv"

def create_data_directory(main_subdir, gridnum, x, y, tot_gates, listnum, additions):
    gridfn = get_name_circuitfile(gridnum, x, y, tot_gates)
    netfn = get_name_netfile(gridnum, listnum)
    new_subdir = '_'.join(additions) + '_' + gridfn[:-4] + "_" + netfn
    rel_path = os.path.join(main_subdir, new_subdir)
    script_dir = os.path.dirname(__file__)
    dir_check_path = os.path.join(script_dir, rel_path)
    if not os.path.exists(dir_check_path):
        os.mkdir(dir_check_path)
        copy(create_fpath(main_subdir,gridfn), os.path.join(dir_check_path, gridfn))
        copy(create_fpath(main_subdir,netfn), os.path.join(dir_check_path, netfn))
    else:
        ans = input("Continuing will overwrite pre-recorded data\nContinue? (Y/n)")
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


# calculate the manhattan distance between two points
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


def print_start_iter(gridnum, netnum, algorithm, swaps, iteration):
    print("############################")
    print("Grid", gridnum, "net", netnum)
    print(algorithm, swaps, "swaps")
    print("Starting iteration", iteration)
    print("############################")


def print_final_state(grid, best_order, best_len, \
                      nets_solved, tot_nets):
    print(grid)
    print("Final Path =\t", best_order, )
    print("Final Length =\t", best_len)
    print("All connected =\t", nets_solved, "/", tot_nets)

def write_connections_length_ord(filename, con_len_list):
    print(con_len_list)
    w_str = '\n'.join(['\t'.join([','.join([str(l) for l in m]) for m in n]) for n in con_len_list])
    with open(filename, 'a') as f:
        f.write(w_str)
        f.write('\n')

