
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
        self.base_outgoing_nets = 0
        self.cur_outgoing = 0


    def set_value(self, value):
        if self.is_occupied():
            return False

        self.value = value
        if value[0] == 'g':
            self.gate = True
        elif value[0] == 'n':
            self.net = True
        return True


    def add_base_outgoing(self):
        self.base_outgoing_nets += 1


    def get_value(self):
        """
        :return: string "0", "gX", or "nY"
        """
        return self.value


    def get_neighbours(self):
        """
        :return: list of neighbouring node objects
        """
        return self.neighbours

    def get_coord(self):
        """
        :return: coordinate at which th node lies
        """
        return self.coord

    def is_occupied(self):
        """
        :return: True if node is in use by a net or gate
        """
        return self.gate or self.net

    def is_gate(self):
        """
        :return: True if node is a gate
        """
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
        """
        :param net: adds net to the set of nets allowed at the gate
        :return:
        """
        if self.is_gate():
            self.out_nets.add(net)
            self.base_outgoing_nets += 1
        else:
            print("a net should not be added here")


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

    def remove_net(self):
        if self.is_gate():
            print("WRONG")
        else:
            self.value = "0"
            self.net = False


    ##################### DAAL ################
    def reset_cur_outgoing(self):
        self.cur_outgoing = 0

    def incr_outgoing(self):
        self.cur_outgoing += 1


    def check_necessity(self):
        necesary = False
        for n in self.neighbours:
            if n.is_gate():
                if n.needs_space() >= n.has_space():
                    necesary = True
        return necesary

    def needs_space(self):
        return self.base_outgoing_nets - self.cur_outgoing

    def has_space(self):
        count = 0
        for n in self.neighbours:
            if not n.is_occupied():
                count += 1
        return count





