import numpy as np
import functools
import operator
from random import randint
import os



# TODO write function so grid is generated from csv file
# TODO write function to make netlist on a given grid
# TODO generate nets function, add_net has already been written, still has a TODO aspect
def create_fpath(subdir, outf):
    rel_path = os.path.join(subdir, outf)
    script_dir = os.path.dirname(__file__)
    dir_check_path = os.path.join(script_dir, subdir)
    if not os.path.exists(dir_check_path):
        os.mkdir(dir_check_path)
    fpath = os.path.join(script_dir, rel_path)
    return fpath


def prodsum(iterable):
    return functools.reduce(operator.mul, iterable, 1)


# calculate the manhattan distance between two points
def manhattan(loc1, loc2):
    return sum([abs(loc1[i] - loc2[i]) for i in range(len(loc1))])


def params_inp(params):
    """
    :param - params: parameters of a grid to be created
    :yield: node for a grid with size of parameters.
    params = (10,10) yields (0, 0), (1, 0), ..., (9,9)
    """
    base = [0]*len(params)
    count = 0
    tot = prodsum(params)
    yield tuple(base)
    while True:
        if count == tot:
            break
        count += 1
        for i in range(len(params)):
            base[i] = count // prodsum(params[:i]) % params[i]
        yield tuple(base)


def neighbours_nodes(coords):
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


class Grid:
    def __init__(self, size_params, gates, nets, _print=False, empty=False):

        # initialize the grid basics
        self.params = size_params  # parameters of the grid
        self.gate_number = gates  # number of total gates
        self.net_number = nets

        self.gate_coords = {}  # key:val gX:tuple(gateloc)
        self.coord_gate = {}
        self.gate_net = {}  #key:val gX:set(nA, nB, nC...)
        self.net_gate = {}  # key:val nX:tuple(g1, g2)
        self.nets = {}  # Live nets
        self.connections = {}  # key:val coord_tuple:tuple(neighbour_tuples)
        self.griddict = {n:Node() for n in params_inp(size_params)}
        self.connect()

        # Place gates on the grid
        if empty:
            pass
        else:
            if type(gates) == int:
                print("starting to generate gates")
                self.generate_gates(gates)
                print("done generating gates")
            elif type(gates) == tuple:
                print("### placing read gates ###")
                # Todo write function to place gates from a tuple of tuples
                self.place_premade_gates(gates)
            if type(nets) == int:
                self.generate_nets(nets)

        # for printing everything, smaller values don't overflow the terminal output
        if _print:
            print(self.gate_coords.keys())
            print(self.gate_coords.values())
            print(self.connections.keys())
            print(self.connections.values())


    ### Gate functions ###
    def write_grid(self, fname):
        grid = self.to_base()
        with open(fname, 'w') as fout:
            for row in grid:
                fout.write(",".join(row) + "\n")
        fout.close()

    def read_grid(fpath):
        base = []
        with open(fpath, 'r') as fin:
            for line in fin:
                base.append(line[:-1].split(','))  # [:-1] so no '\n')
        otherbase = []
        for i in range(len(base)):
            otherbase.append(base[:][i])
        return base

    def gates_from_base(base):
        gate_coords = []
        gates = []
        for x in range(len(base)):
            for y in range(len(base[0])):
                gate = base[x][y]
                if gate[0] == 'g':
                    gate_coords.append((x, y))
                    gates.append(gate)
        return gate_coords, gates

    def to_dicts(fpath, nets):
        base = Grid.read_grid(fpath)
        xlen = len(base[0])
        ylen = len(base)
        gate_coords, gates = Grid.gates_from_base(base)
        Newgrid = Grid([xlen, ylen], (gate_coords, gates), nets)
        return Newgrid

    def to_base(self):
        x = self.params[0]
        y = self.params[1]
        list = [str(self.griddict[i]) for i in params_inp(self.params)]
        newbase = [[list[j * x + i] for i in range(x)] for j in
                   range(y)]
        return newbase

    # connects nodes in the grid to it's neighbouring nodes
    def connect(self):
        """
        adds the connections each node has into the connection dictionary
        """
        print("connecting...")
        for key in self.griddict.keys():
            self.connections[key] = tuple([pn for pn in neighbours(key) if self.griddict.get(pn, False)])
        print("done connecting")

    def rand_loc(self):
        x_pos = randint(1, self.params[0]-2)
        y_pos = randint(1, self.params[1]-2)
        gate_pos = tuple([x_pos, y_pos])
        lencheck = [x for x in self.connections.get(gate_pos, []) if self.coord_gate.get(x, False)]
        check1 = len(lencheck) > 3
        check2 = (self.griddict.get(gate_pos, False)) != "0"
        while check1 or check2:
            print("in rand_loc")
            x_pos = randint(1, self.params[0]-2)
            y_pos = randint(1, self.params[1]-2)
            gate_pos = tuple([x_pos, y_pos])
            # checks
            lencheck = [x for x in self.connections.get(gate_pos, []) if
                        self.coord_gate.get(x, False)]
            check1 = len(lencheck) > 3
            check2 = (self.griddict.get(gate_pos, False)) != "0"

        return gate_pos


    def generate_gates(self, num):
        """
        places num gates on the grid
        """
        for i in range(num):
            rescoords = self.rand_loc()
            # print("next gate")
            self.add_gate(rescoords, "g"+str(i))


    # adds a gate to the grid
    def add_gate(self, coords, gate_string):
        """
        places the gate inside the grid ditionary
        places the gatestring inside the gatecoord dictionary
        """
        self.griddict[coords] = gate_string
        self.gate_coords[gate_string] = coords
        self.coord_gate[coords] = gate_string
        self.gate_net[gate_string] = set()

    # add goal-connection to the grid
    # this connection is an unrealised version of it, only the begin- & endpoint
    def add_net(self, gate1, gate2, n_str):
        self.net_gate[n_str] = (self.gate_coords[gate1], self.gate_coords[gate2])
        self.gate_net[gate1].add(n_str)
        self.gate_net[gate2].add(n_str)
        self.nets[n_str] = (gate1, gate2)


    def place_premade_gates(self, gate_pairs):
        gatecoords, gates = gate_pairs
        for n, val in enumerate(gatecoords):
            self.add_gate(val[::-1], gates[n])



    def generate_nets(self, num):
        AG = list(self.gate_coords.keys())
        GN = len(AG)-1
        print(num)
        for i in range(num):
            g1, g2, net = AG[randint(0,GN)], AG[randint(0,GN)], 'n'+str(i)
            g1nets = self.gate_net.get(g1)
            g2nets = self.gate_net.get(g2)
            common = (g1nets & g2nets)

            while common:
                g1, g2 = AG[randint(0, GN)], AG[randint(0, GN)]
                g1nets = self.gate_net.get(g1)
                g2nets = self.gate_net.get(g2)
                common = g1nets & g2nets
            self.add_net(g1, g2, net)

    def remove_nets(self):
        for key in self.gate_net.keys():
            self.gate_net[key] = set()
        self.net_gate = {}  # key:val nX:tuple(g1, g2)
        self.net_number = 0
        self.nets = {}

    def write_nets(self, subdir, gridnum, listnum):
        outfile = "C" + str(gridnum) + "_netlist_" + str(listnum) + ".csv"
        writepath = create_fpath(subdir, outfile)
        with open(writepath, 'w') as out:
            for netk in self.nets.keys():
                g1, g2 = self.nets.get(netk)
                out.write(','.join([netk,g1,g2])+'\n')

    def read_nets(self, subdir, filename):
        readpath = create_fpath(subdir, filename)
        nets = []
        with open(readpath, 'r') as inf:
            for line in inf:
                nets.append(line[:-1].split(','))
        for line in nets:
            net, g1, g2 = line
            self.add_net(g1, g2, net)

    def print_nets(self):
        _list = []
        for key in self.net_gate.keys():
            _list.append([key, self.net_gate[key]])
        print(_list)


def get_name_netfile(gridnum, listnum):
    return "C" + str(gridnum) + "_netlist_" + str(listnum) + ".csv"

def get_name_circuitfile(gridnum, x, y, tot_gates):
    return "Gateplatform_" + str(gridnum) + "_" + str(x) + "x" + str(
        y) + "g" + str(tot_gates) + ".csv"



# Paper, making = True



"""
if __name__ == "__main__":
    x = 10
    y = 20
    tot_gates = 30
    tot_nets = 50
    fname = "Gateplatform_" + str(x) + "x" + str(y) + "g" + str(
        tot_gates) + "v" + str(0) + ".csv"
    subdir = "circuit_map"
    fpath = create_fpath(subdir, fname)
    newgrid = Grid([x, y], tot_gates, tot_nets)
    print(newgrid.to_base())
    newgrid.write_grid(fpath)
    checkgrid = Grid.to_dicts(fpath, tot_nets)
    print(checkgrid.to_base())
"""