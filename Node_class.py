
class Node:
    def __init__(self, coord, value):
        self.value = value
        self.coord = coord
        self.neighbours = []
        self.gate = False
        self.net = False
        self.neighbour_num = 0
        self.set_value(value)
        self.out_nets = set()

    def set_value(self, value):
        if self.is_occupied():
            return False
        self.value = value
        if value[0] == 'g':
            self.gate = True
        elif value[0] == 'n':
            self.net = True


    def get_value(self):
        return self.value

    def get_neighbours(self):
        return self.neighbours

    def get_coord(self):
        return self.coord()

    def is_occupied(self):
        return self.gate or self.net

    def is_gate(self):
        return self.gate

    def get_adjecent_occupied(self):
        count = 0
        for adj in self.neighbours:
            if adj.is_occupied():
                count += 1
        return count

    def has_room(self):
        #Todo add a check for outgoing netlists / gate_class
        count = self.get_adjecent_occupied() + len(self.out_nets)
        print([i.get_value() for i in self.get_neighbours()], self.out_nets, count)
        if count < self.neighbour_num:
            print("has room")
            return True
        else:
            print("has NO room")
            return False

    def add_net(self, net):
        self.out_nets.add(net)


    def connect(self, neighbours):
        self.neighbours = neighbours
        self.neighbour_num = len(neighbours)


    ##########################
    #     Removing stuff     #
    ##########################

    def remove_out_nets(self):
        self.out_nets = set()

    def set_null(self):
        self.set_value("0")
        self.remove_out_nets()
        self.net = False









