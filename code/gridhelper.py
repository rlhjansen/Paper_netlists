import functools
import operator

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


def transform_print(val):
    if val == '0':
        return '___'
    elif val[0] == 'n':
        return val[1:].zfill(3)
    elif val[0] == 'g':
        return val[1:].zfill(3)
    else:
        raise ValueError("incorrect node value")
