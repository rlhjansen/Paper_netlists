
from random import randint, shuffle
import queue as Q

from ..gridhelper import \
    neighbours, \
    transform_print, \
    params_inp, \
    manhattan, \
    read_grid, \
    gates_from_lol

from .node import Node


class Grid:
    def __init__(self, size_params, solver="A_star", height=7,verbose=False, empty=False, AH=False, gates=None, max_g=False):

        # initialize the grid basics
        self.max_g = max_g
        self.AH = AH
        self.params = size_params + [height]  # parameters of the grid
        self.platform_params = size_params
        self.gate_coords = {}  # key:val gX:tuple(gate_loc)
        self.coord_gate = {}
        self.gate_net = {} # key:val gX:set(nA, nB, nC...)
        self.net_gate = {}  # key:val nX:tuple(g1, g2)
        self.nets = {}  # Live nets
        self.connections = {}  # key:val coord_tuple:tuple(neighbour_tuples)
        self.wire_locs = set()
        self.griddict = {n:Node(n, '0') for n in params_inp(self.params)}
        self.connect()
        self.allowed_height = 0

        #elevator
        self.wire_loc_dict = dict()
        self.unsolved_nets = set()
        self.net_paths = {}
        self.height = height
        self.solving = False
        if gates:
            self.place_premade_gates(gates_from_lol(gates))


    def set_solver(self, solver):
        if solver == "A_star":
            self.solve = self.solve_order
        elif solver == "elevator":
            self.solve = self.solve_order_ele
        else:
            raise NotImplementedError("no solver with the name '" + solver +
                                      "' implemented\nTry using 'A_star' or 'elevator'")

    ### Gate functions ###
    def write_grid(self, fname):
        """
        writes current gate configuration to an out-file
        :param fname: filename to save to
        """
        grid = self.to_base()
        with open(fname, 'w+') as fout:
            for row in grid:
                fout.write(",".join(row) + "\n")
        fout.close()

    def to_base(self):
        """
        :return: a list of lists of the "ground floor" of the grid
        """
        x = self.platform_params[0]
        y = self.platform_params[1]
        list = [str(self.griddict[i+(0,)].get_value()) for i in params_inp(self.platform_params)]
        newbase = [[list[j * x + i] for i in range(x)] for j in
                   range(y)]
        return newbase

    # connects nodes in the grid to it's neighbouring nodes
    def connect(self):
        """
        adds the connections each node has into the connection dictionary
        """
        for key in self.griddict.keys():
            neighbour_nodes = tuple([self.griddict.get(pn) for pn in neighbours(key) if self.griddict.get(pn, False)])
            # neigbour_coords = tuple([pn for pn in neighbours(key) if self.griddict.get(pn, False)]) // testing with coords
            self.griddict[key].connect(neighbour_nodes)

    def disconnect(self):
        for key in self.griddict.keys():
            neighbour_nodes = tuple([self.griddict.get(pn) for pn in neighbours(key) if self.griddict.get(pn, False)])
            # neigbour_coords = tuple([pn for pn in neighbours(key) if self.griddict.get(pn, False)]) // testing with coords
            self.griddict[key].disconnect()

    def rand_loc(self):
        """
        :return: random empty location on the "ground floor" of the grid
        """
        x_pos = randint(1, self.params[0]-2)
        y_pos = randint(1, self.params[1]-2)
        z_pos = 0
        gate_pos = tuple([x_pos, y_pos, z_pos])
        lencheck = [x for x in self.connections.get(gate_pos, []) if self.coord_gate.get(x, False)]
        check1 = len(lencheck) > 3
        check2 = self.griddict[gate_pos].get_value() != "0"
        while check1 or check2:
            x_pos = randint(1, self.params[0]-2)
            y_pos = randint(1, self.params[1]-2)
            gate_pos = tuple([x_pos, y_pos, z_pos])
            # checks
            lencheck = [x for x in self.connections.get(gate_pos, []) if
                        self.coord_gate.get(x, False)]
            check1 = len(lencheck) > 3
            check2 = self.griddict[gate_pos].get_value() != "0"

        return gate_pos


    def generate_gates(self, num):
        """
        places num gates on the grid
        """
        self.wipe_gates()
        for i in range(num):
            rescoords = self.rand_loc()
            self.add_gate(rescoords, "g"+str(i))


    def wipe_gates(self):
        """
        remove all gates from circuit
        """
        self.griddict = {n:Node(n, '0') for n in params_inp(self.params)}
        self.connect()

        self.gate_coords = {}  # key:val gX:tuple(gate_loc)
        self.coord_gate = {}
        self.gate_net = {}


    # adds a gate to the grid
    def add_gate(self, coords, gate_string):
        """
        places the gate inside the grid ditionary
        places the gatestring inside the gatecoord dictionary
        """
        self.griddict[coords].set_value(gate_string)
        self.griddict[coords].add_base_outgoing()
        self.gate_coords[gate_string] = coords
        self.coord_gate[coords] = gate_string
        self.gate_net[gate_string] = set()


    def place_premade_gates(self, gate_pairs):
        gatecoords, gates = gate_pairs
        for n, val in enumerate(gatecoords):
            self.add_gate(val[::-1] + (0,), gates[n])

    def get_gate_coords(self):
        kv = [[k , v] for k, v in self.gate_coords.items()]
        k = [kv[i][0] for i in range(len(kv))]
        v = [kv[i][1] for i in range(len(kv))]
        return k, v


    ### Net functions ###

    # add goal-connection to the grid
    # this connection is an unrealised version of it, only the begin- & endpoint
    def add_net(self, gate1, gate2, n_str):
        self.net_gate[n_str] = (gate1, gate2)
        self.griddict[self.gate_coords[gate1]].add_net(n_str)
        self.griddict[self.gate_coords[gate2]].add_net(n_str)
        self.gate_net[gate1].add(n_str)
        self.gate_net[gate2].add(n_str)
        self.nets[n_str] = (gate1, gate2)

        # elevator
        self.wire_loc_dict[n_str] = set()


    def generate_nets(self, num):
        AG = list(self.gate_coords.keys())
        GN = len(AG)-1
        for i in range(num):
            g1, g2, net = AG[randint(0,GN)], AG[randint(0,GN)], 'n'+str(i)
            g1nets = self.gate_net.get(g1, set())
            g2nets = self.gate_net.get(g2, set())
            common = (g1nets & g2nets)
            roomleft1 = self.griddict.get(self.gate_coords.get(g1)).has_room()
            roomleft2 = self.griddict.get(self.gate_coords.get(g2)).has_room()
            no_room_left = not (roomleft1 and roomleft2)
            while (common or no_room_left) or (g1==g2):
                g1, g2 = AG[randint(0, GN)], AG[randint(0, GN)]
                if g1 == g2:
                    continue
                g1nets = self.gate_net.get(g1)
                g2nets = self.gate_net.get(g2)
                common = g1nets & g2nets
                roomleft1 = self.griddict.get(self.gate_coords.get(g1)).has_room()
                roomleft2 = self.griddict.get(self.gate_coords.get(g2)).has_room()
                no_room_left = not (roomleft1 and roomleft2)
            self.add_net(g1, g2, net)


    def get_random_net_order(self):
        key_list = list(self.net_gate.keys())
        shuffle(key_list)
        return key_list[:]

    def get_net_ordered_manhattan(self):
        values = list(self.net_gate.values())
        nets = list(self.net_gate.keys())
        manh_vals = [manhattan(self.gate_coords[values[i][0]],
                               self.gate_coords[values[i][1]]) for i in range(len(nets))]
        to_sort = [(manh_vals[i], nets[i]) for i in range(len(nets))]
        s = sorted(to_sort)
        s_nets = [n[1] for n in s]
        s_manh = [n[0] for n in s]
        return s_nets



    def write_nets(self, abspath):
        with open(abspath, 'w+') as out:
            for netk in self.nets.keys():
                g1, g2 = self.nets.get(netk)
                out.write(','.join([netk,g1,g2])+'\n')

    def read_nets(self, abspath):
        nets = []
        with open(abspath, 'r') as inf:
            for line in inf:
                nets.append(line[:-1].split(','))
        for line in nets:
            net, g1, g2 = line
            self.add_net(g1, g2, net)

    def print_nets(self):
        for key in self.net_gate.keys():
            print([key, self.net_gate[key]])


    ###########################
    #####   Reset Block   #####
    ###########################
    def wipe_nets(self):
        """
        remove the netlist from class
        """
        for key in self.coord_gate.keys():
            self.griddict[key].remove_out_nets()
        for key in self.gate_net.keys():
            self.gate_net[key] = set()
        self.net_gate = {}
        self.reset_nets()
        self.nets = {}


    def reset_nets(self):
        """
        retains netlist connections but resets their placement
        """
        for spot in self.wire_locs:
            self.griddict[spot].remove_net()
        self.wire_locs = set()
        self.allowed_height = 0

    def reset_nets_ele(self):
        """
        retains netlist connections but resets their placement
        """
        for spot in self.wire_locs:
            self.griddict[spot].remove_net()
        for coord in self.coord_gate.keys():
            self.griddict[coord].reset_cur_outgoing()
        self.wire_locs = set()


    ###########################
    ###### printing block #####
    ###########################
    def __str__(self):
        complete = []
        pars = self.params
        for z in range(pars[2]):
            complete.append("### Layer" + str(z + 1) + "###")
            for y in range(pars[1]):
                vals = [self.griddict[(x,y,z)].get_value() for x in range(pars[0])]
                transformed_vals = [transform_print(val) for val in vals]
                complete.append(" ".join(transformed_vals))
        return "\n".join(complete)

    def extract_route(self, path_dict, end_loc):
        path = ((),)
        get_loc = path_dict.get(end_loc)[0]
        while path_dict.get(get_loc)[0] != get_loc:

            path = path + (get_loc,)
            get_loc = path_dict.get(get_loc)[0]
        print(path[::-1])
        return path[::-1]

    def A_star(self, net):
        """ finds a path for a net with A-star algorithm, quits searching early if the end-gate is closed off by its immediate neighbourse.

        :param net: gate-pair (gX, gY)
        :return: path, length if path founde, else false, false
        """

        q = Q.PriorityQueue()
        count = 0
        end_loc = self.gate_coords.get(self.net_gate.get(net)[1])
        if self.griddict.get(end_loc).is_blocked_in():
            return False, False, count
        start_loc = self.gate_coords.get(self.net_gate.get(net)[0])
        if self.griddict.get(start_loc).is_blocked_in():
            return False, False, count

        path = ((start_loc),)
        manh_d = manhattan(path[-1], end_loc)
        q.put((manh_d, 0, start_loc))
        visited = dict()
        visited[start_loc] = [start_loc, 0]
        while not q.empty():
            count += 1
            k = q.get()
            _, steps, current = k
            for neighbour in self.griddict.get(current).get_neighbours():
                n_coord = neighbour.get_coord()

                if neighbour.is_occupied():
                    if n_coord == end_loc:
                        visited[n_coord] = [current, steps]
                        return self.extract_route(visited, n_coord), \
                               visited.get(end_loc)[1], count
                    else:
                        continue
                if n_coord in visited:
                    if visited.get(n_coord)[1] > steps:
                        visited[n_coord] = [current, steps]
                        q.put((manhattan(n_coord, end_loc) + steps + 1, steps + 1,
                          n_coord))
                else:
                    visited[n_coord] = [current, steps]
                    q.put((manhattan(n_coord, end_loc) + steps + 1, steps + 1,
                           n_coord))
        return False, False, count


    def A_star_max_g(self, net):
        """ finds a path for a net with A-star algorithm, quits searching early if the end-gate is closed off by its immediate neighbourse.

        :param net: gate-pair (gX, gY)
        :return: path, length if path founde, else false, false
        """

        q = Q.PriorityQueue()
        count = 0
        end_loc = self.gate_coords.get(self.net_gate.get(net)[1])
        if self.griddict.get(end_loc).is_blocked_in():
            return False, False, count
        start_loc = self.gate_coords.get(self.net_gate.get(net)[0])
        if self.griddict.get(start_loc).is_blocked_in():
            return False, False, count

        path = ((start_loc),)
        manh_d = manhattan(path[-1], end_loc)
        q.put((manh_d, 0, start_loc))
        visited = dict()
        visited[start_loc] = [start_loc, 0]
        print("start_loc", start_loc)
        while not q.empty():
            count += 1
            k = q.get()
            _, steps, current = k
            for neighbour in self.griddict.get(current).get_neighbours():
                n_coord = neighbour.get_coord()

                if neighbour.is_occupied():
                    if n_coord == end_loc:
                        visited[n_coord] = [current, steps]
                        print("end_loc", end_loc)
                        return self.extract_route(visited, n_coord), \
                               visited.get(end_loc)[1], count
                    else:
                        continue
                if n_coord in visited:
                    if visited.get(n_coord)[1] > steps:
                        visited[n_coord] = [current, steps]
                        q.put((manhattan(n_coord, end_loc) - steps + 1, steps - 1,
                          n_coord))
                else:
                    visited[n_coord] = [current, steps]
                    q.put((manhattan(n_coord, end_loc) + steps + 1, steps + 1,
                           n_coord))
        return False, False, count



    def solve_order(self, net_order, reset=False):
        self.solving = True
        tot_length = 0
        solved = 0
        nets_solved = []
        tries = 0
        for net in net_order:
            if self.max_g:
                path, length, ntries = self.A_star_max_g(net)
            else:
                path, length, ntries = self.A_star(net)
            tries += ntries
            if path:
                self.place(net, path)
                solved += 1
                tot_length += length
                nets_solved.append(net)
        self.solving = False
        if reset:
            self.reset_nets()
        return [solved, tot_length, tries]


    def solve_order_paths(self, net_order, reset=False):
        self.solving = True
        tot_length = 0
        solved = 0
        nets_solved = []
        tries = 0
        paths = []
        for net in net_order:
            if self.max_g:
                print("entering max g")
                path, length, ntries = self.A_star_max_g(net)
            else:
                path, length, ntries = self.A_star(net)
            tries += ntries
            if path:
                self.place(net, path)
                solved += 1
                tot_length += length
                nets_solved.append(net)
            paths.append(path)
        self.solving = False
        if reset:
            self.reset_nets()
        return [solved, tot_length, tries, paths]


    def solve_order_ele(self, net_order):
        """ Solver with elevator

        :param net_order: Order to lay search for solutions per net.
        :return highest: Highest point where a net lies.
        :return tot_length: Total length of the path of all nets.
        :return new_order: Net order if could initially be placed, otherwise
        the new order is returned.
        """
        self.solving = True
        tot_length = 0
        solved = 0
        new_order = net_order[:]
        tries = 0
        h = 0
        self.reset_nets_ele()

        unplaced = set(net_order[:])
        for net in new_order:
            if self.pre_place_net(net):
                unplaced.remove(net)
            else:
                self.unsolved_nets.remove(net)

        while self.unsolved_nets:
            for net in new_order:
                if net not in self.unsolved_nets:
                    continue
                check = self.a_star_ele(net, h)
                if not check[0]:
                    tries += check[1]
                    continue
                path, length, count = check
                tot_length += length
                tries += count
                if path:
                    Err = self.place(net, path)
                    solved += 1
                    self.unsolved_nets.remove(net)
            h += 1
            extra_length, valid = self.elevate_unsolved(h)
            tot_length += extra_length
            if not valid:
                self.solving = False
                return len(net_order) - len(unplaced), tot_length, tries
        self.solving = False
        return len(net_order) - len(unplaced), tot_length, tries


    def pre_place_net(self, net):
        start_loc = self.gate_coords.get(self.net_gate.get(net)[0])
        end_loc = self.gate_coords.get(self.net_gate.get(net)[1])
        self.unsolved_nets.add(net)
        start = False
        end = False
        temp_net_path = []
        for n in self.griddict.get(start_loc).get_neighbours():
            if not n.is_occupied():
                free_coord = n.get_coord()
                manh_d = manhattan(free_coord, end_loc)
                temp_net_path = [manh_d+1, 1, (start_loc,free_coord,),]
                self.griddict[free_coord].set_value(net)
                self.wire_loc_dict[net].add(free_coord)
                start = True
                break
        # end
        for n in self.griddict.get(end_loc).get_neighbours():
            if not n.is_occupied():
                free_end_coord = n.get_coord()
                temp_net_path.append((free_end_coord,))
                self.griddict[free_end_coord].set_value(net)
                self.wire_loc_dict[net].add(free_end_coord)
                end = True
                break
        if start and end:
            self.net_paths[net] = tuple(temp_net_path)
            return True
        else:
            return False

    def reset_nets_ele(self):
        """
        retains netlist connections but resets their placement
        """
        for netk in self.wire_loc_dict.keys():
            self.reset_single_net(netk)
        for coord in self.coord_gate.keys():
            self.griddict[coord].reset_cur_outgoing()
        for spot in self.wire_locs:
            self.griddict[spot].remove_net()
            pass

        self.wire_locs = set()
        self.unsolved = set()

    def reset_single_net(self, netk):
        for coord in self.wire_loc_dict.get(netk):
            self.griddict[coord].remove_net()
        self.wire_loc_dict[netk] = set()

    def elevate_net(self, net, height):
        net_path = list(self.net_paths.get(net))
        manh_steps, steps, coord_path, end_path = net_path

        tlcoord_path, tncoord = self.elevate_path_to_H(coord_path, height)

        tlend_path, tnend = self.elevate_path_to_H(end_path, height)
        self.net_paths[net] = tuple((
            manh_steps + 1, steps + 1, tlcoord_path, tlend_path))
        if len(self.net_paths[net]) == 1:
            exit(-1)
        self.wire_loc_dict[net].add(tncoord)
        self.griddict[tncoord].set_value(net)
        self.wire_loc_dict[net].add(tnend)
        self.griddict[tnend].set_value(net)

    def elevate_path_to_H(self, coord_path, height):
        lcoordpath = list(coord_path)

        ncoord = list(lcoordpath[-1])
        ncoord[2] = height
        tncoord = tuple(ncoord)
        lcoordpath.append(tncoord)
        tlcoord_path = tuple(lcoordpath)
        return tlcoord_path, tncoord

    def elevate_unsolved(self, height):
        if height >= self.height:
            return 1000, False
        for net in self.unsolved_nets:
            self.elevate_net(net, height)
        return len(self.unsolved_nets)*2, True

    def a_star_ele(self, net, h):
        q = Q.PriorityQueue()
        count = 0
        visited = dict()

        path_tuples = self.net_paths.get(net, False)
        if not path_tuples:
            return False, False, count

        _, steps, path, end_path = path_tuples

        start_loc = path[-1]
        end_loc = end_path[-1]
        manh_d = manhattan(path[-1], end_loc)
        q.put((manh_d, 0, start_loc))
        visited[start_loc] = [start_loc, 0]

        while not q.empty():
            count += 1
            _, steps, current = q.get()

            for neighbour in self.griddict.get(current).get_neighbours():
                n_coord = neighbour.get_coord()

                if neighbour.is_occupied():
                    if n_coord == end_loc:
                        visited[end_loc] = [current, steps]
                        return self.extract_route(visited, end_loc), \
                               visited.get(end_loc)[1], count
                    else:
                        continue

                if n_coord[2] <= h:
                    if n_coord in visited:
                        if visited.get(n_coord)[1] > steps:
                            visited[n_coord] = [current, steps]
                            q.put((manhattan(n_coord, end_loc) + steps + 1,
                                   steps + 1,
                                   n_coord))
                    else:
                        visited[n_coord] = [current, steps]
                        q.put((manhattan(n_coord, end_loc) + steps + 1,
                               steps + 1,
                               n_coord))
        return False, False, count

    def get_solution_placement(self, net_order, max_g=False):
        paths = []
        for net in net_order:
            if max_g:
                path, length = self.A_star_max_g(net)[:2]
            else:
                path, length = self.A_star(net)[:2]
            if path:
                paths.append(path)
                Err = self.place(net, path)
            else:
                paths.append( ((),))
        self.reset_nets()
        return paths

    def place(self, net, path):
        for spot in path[:-1]:
            if self.griddict[spot].set_value(net):
                self.wire_locs.add(spot)
            else:
                raise ValueError("invalid placement")
        return False


def file_to_grid(fpath, nets, max_g=True, height=7):
    """
    :param nets: either a netlist or a number of nets
    :return: a new Grid
    """
    base = read_grid(fpath)
    xlen = len(base[0])
    ylen = len(base)
    Newgrid = Grid([xlen, ylen], gates=base, height=height, max_g=max_g)
    return Newgrid
