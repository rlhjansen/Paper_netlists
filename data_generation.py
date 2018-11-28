from code.classes.grid import Grid
from code.alghelper import get_name_circuitfile, create_fpath

import os

def create_net_datapath(gen, c, n, cX, x, y):
    abspath = create_circuit_datapath(gen, c, cX, x, y)
    abspath = os.path.join(abspath, "N"+str(n))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    abspath = os.path.join(abspath, "N"+str(n)+"_"+str(len(os.listdir(abspath)))+".csv")
    open(abspath, "a").close()
    print(abspath)
    return abspath


def create_circuit_datapath(gen, c, cX, x, y):
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
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

def main(x, y):
    tot_gates = [100]
    tot_nets = [80,100,120]
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
    abspath = os.path.join(abspath, "generated")
    abspath = os.path.join(abspath, "x"+str(x)+"y"+str(y))
    if not os.path.exists(abspath):
        os.makedirs(abspath)
    gen = len(os.listdir(abspath))
    for c in tot_gates:
        for cX in range(5):
            #c_datapath = create_circuit_path(g)
            newgrid = Grid([x, y], AH=True)
            newgrid.generate_gates(c)
            circuit_path = create_circuit_datapath(gen, c, cX, x, y) + ".csv"
            newgrid.write_grid(circuit_path)
            for n in tot_nets:
                for _ in range(10):
                    netlistpath = create_net_datapath(gen, c, n, cX, x, y)
                    newgrid.generate_nets(n)
                    newgrid.write_nets(netlistpath)
                    #print(newgrid)
                    #print(newgrid.gate_coords)
                    newgrid.wipe_nets()


if __name__ == '__main__':
    main(30,30)
