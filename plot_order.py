

import func_objects as fo
from independent_functions import plot_circuit
from Gridfile import Grid, SRC



SUBDIR = "circuit_map_100"
GRIDNUM = 0
NETLISTNUM = 0
X = 30
Y = 30
G = 100
NETL_LEN = 100

#self, main_subdir, grid_num, list_num,  x, y, tot_gates, additions, batches, batch_size=100, ask=True):
rc = fo.RC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G,["test_vis"],1,batch_size=1, ask=False)
order = rc.G.get_random_net_order()
paths = rc.G.get_solution_placement(order)
g_names, g_coords = rc.G.get_gate_coords()
print("plotting")
plot_circuit(paths, order, g_coords, g_names)