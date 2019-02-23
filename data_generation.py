from code.classes.grid import Grid, file_to_grid

import os

def create_net_datapath(gen, c, n, cX, x, y, official=False):
    abspath = create_circuit_datapath(gen, c, cX, x, y, official=official)
    abspath = os.path.join(abspath, "N"+str(n))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    abspath = os.path.join(abspath, "N"+str(n)+"_"+str(len(os.listdir(abspath)))+".csv")
    open(abspath, "a").close()
    print(abspath)
    return abspath


def create_circuit_datapath(gen, c, cX, x, y, official=False):
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
    if official:
        abspath = os.path.join(abspath, "official_reference")
    abspath = os.path.join(abspath, "generated")
    abspath = os.path.join(abspath, "x"+str(x)+"y"+str(y))

    if not os.path.exists(abspath):
        os.makedirs(abspath)
    abspath = os.path.join(abspath, "g"+str(gen))
    abspath = os.path.join(abspath, 'C'+str(c))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    abspath = os.path.join(abspath, 'C'+str(c)+"_"+str(cX))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    print(abspath)
    return abspath

def gen_from_established(c, cX, x, y, netlistlen, netcount):
    circuit_path = create_circuit_datapath(0, c, cX, x, y, official=True) + ".csv"
    circuit = file_to_grid(circuit_path, None)
    for _ in range(netcount):
        netlistpath = create_net_datapath(0, c, netlistlen, cX, x, y, official=True)
        circuit.generate_nets(netlistlen)
        circuit.write_nets(netlistpath)
        circuit.wipe_nets()


def main(x, y):
    tot_gates = [100]
    netlistlen = [i+11 for i in range(60)]
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
    abspath = os.path.join(abspath, "generated")
    abspath = os.path.join(abspath, "x"+str(x)+"y"+str(y))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    gen = len(os.listdir(abspath))
    for c in tot_gates:
        for cX in range(1):
            newgrid = Grid([x, y], AH=True)
            newgrid.generate_gates(c)
            circuit_path = create_circuit_datapath(gen, c, cX, x, y) + ".csv"
            newgrid.write_grid(circuit_path)
            for n in netlistlen:
                for _ in range(10):
                    netlistpath = create_net_datapath(gen, c, n, cX, x, y)
                    newgrid.generate_nets(n)
                    newgrid.write_nets(netlistpath)
                    newgrid.wipe_nets()


if __name__ == '__main__':
    netlens = [i+10 for i in range(61)]
    for n in netlens:
        gen_from_established(100, 0, 20, 20, n, 10)
        gen_from_established(100, 0, 30, 30, n, 10)
        gen_from_established(100, 0, 40, 40, n, 10)
        gen_from_established(100, 0, 50, 50, n, 10)
        gen_from_established(100, 0, 60, 60, n, 10)
        gen_from_established(100, 0, 70, 70, n, 10)
        gen_from_established(100, 0, 80, 80, n, 10)
