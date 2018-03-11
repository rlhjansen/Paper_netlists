from netlist import Grid
from netlist import get_name_circuitfile
from netlist import create_fpath


xys = [[10,20], [20,30], [30, 40]]
tot_gate_list = [20,30,40]
tot_net_list = [[20, 30, 40, 50,60], [30, 40, 50, 60, 70], [40,50,60,70,80], [50,60,70,80,90]]
subdir = "circuit_map"

# DAAL, making = False

if True:
    for i in range(3):
        x = xys[i][0]
        y = xys[i][1]
        tot_gates = tot_gate_list[i]
        tot_nets = tot_net_list[0][0]
        print(i)
        circ_fname = get_name_circuitfile(i, x, y, tot_gates)
        fpath = create_fpath(subdir, circ_fname)
        newgrid = Grid([x, y], tot_gates, tot_nets)
        newgrid.write_grid(fpath)
        checkgrid = Grid.to_dicts(fpath, tot_nets)
        for j in range(len(tot_net_list)):
            newgrid.remove_nets()
            tot_nets = tot_net_list[i][j]
            newgrid.generate_nets(tot_nets)
            newgrid.write_nets(subdir, i, j)
            newgrid.print_nets()
            print("#################################")
