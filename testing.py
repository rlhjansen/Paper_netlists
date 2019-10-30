

import os
import code.algorithms.simplyX as simple


def count_slash(str):
    return sum([c=='/' for c in str])

for ind, elem in enumerate(os.walk('./results')):
    if count_slash(elem[0]) == 10:                      # these give datafiles
        try:
            datafile = os.path.join(elem[0], elem[2][0])
            print(datafile)
            with open(datafile, 'r') as data:
                print(data.readline().split(';')[-1][:-1])
            print(datafile.split('/'))
            break
        except:
            pass



def get_datafiles(size, ncount, nlist):
    abspath = os.path.abspath(__file__)
    abspath = os.path.dirname(abspath)
    abspath = os.path.join(abspath, "data")
    if self.generated:
        abspath = os.path.join(abspath, "official_reference")
        abspath = os.path.join(abspath, "generated")
        abspath = os.path.join(abspath, "x"+str(size)+"y"+str(self.y))
        abspath = os.path.join(abspath, "g0")
    else:
        abspath = os.path.join(abspath, "baseline")
    abspath = os.path.join(abspath, 'C'+str(self.c))
    abspath = os.path.join(abspath, 'C'+str(self.c)+"_"+str(self.cX))
    self.circuit_path = abspath+".csv"
    abspath = os.path.join(abspath, "N"+str(self.n))
    abspath = os.path.join(abspath, "N"+str(self.n)+"_"+str(self.nX)+".csv")
    self.netlist_path = abspath




def check_7(order, size, netcount, netlist, iters):
    """toplayers is a dictionary saving the toplayers of each routing"""
    toplayers = {}
    tag='test-okt'
    start_add=10
    lens = [i+start_add for i in range(81)]
    for n in lens:
        pool = mp.Pool(mp.cpu_count()-1)
        Simples = simple_generator(100, 0, n, 20, size, size, tag=tag, iters=iters)
        grid = file_to_grid(self.circuit_path, None, max_g=max_g)
        self.circuit.read_nets(self.netlist_path)

        simple.SIMPLY(100, 0, n, net_num, x, y, tag, iters=iters)
        pool.map(meh, Simples)
        pool.close()

    simple_obj.circuit.connect()
    circuit = simple_obj.circuit
    if not ord:
        ord = circuit.get_random_net_order()
    g_coords, paths, _ = get_circuit_basics(circuit, ord)
