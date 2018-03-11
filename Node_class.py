
class Node:
    def __init__(self, coord, value):
        self.coord = coord
        self.value = value
        self.neighbours = []
        if value[0] == 'g':
            self.gate = True
        else:
            self.gate = False

    def get_connections(self):
        return self.neighbours

    def connect(self, neighbours):
        self.neighbours = neighbours





