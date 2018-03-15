import os
import functools
import operator

from random import randint


def create_fpath(subdir, outf):
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
    print("", net_path)
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




###### Filename Generating ############
def get_name_netfile(gridnum, listnum):
    return "C" + str(gridnum) + "_netlist_" + str(listnum) + ".csv"

def get_name_circuitfile(gridnum, x, y, tot_gates):
    return "Gateplatform_" + str(gridnum) + "_" + str(x) + "x" + str(
        y) + "g" + str(tot_gates) + ".csv"






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
        temp1[i] += 1
        temp2[i] -= 1
        rl.extend((tuple(temp1), tuple(temp2)))
    return tuple(rl)

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
