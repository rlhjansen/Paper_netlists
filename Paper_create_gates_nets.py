from netlist import Grid
from netlist import get_name_circuitfile
from netlist import create_fpath


xys = [[50,50], [20,30], [30, 40], [32, 32]]
tot_gate_list = [400, 30, 35, 40]
tot_net_list = [[20, 30, 40, 30, 41], [30, 40, 50, 60, 65], [40,50,60,70,80], [50,60,70,80,85]]
subdir = "circuit_map"

# DAAL, making = False

if True:
    for i in range(len(tot_gate_list)):
        x = xys[i][0]
        y = xys[i][1]
        tot_gates = tot_gate_list[i]
        tot_nets = tot_net_list[0][0]
        circ_fname = get_name_circuitfile(i, x, y, tot_gates)
        fpath = create_fpath(subdir, circ_fname)
        newgrid = Grid([x, y], tot_gates, tot_nets, AH=True)
        print(newgrid)
        newgrid.write_grid(fpath)
        newgrid.nets_to_null()
        break
        for j in range(len(tot_net_list[0])):
            print("enter net loop", i, j, tot_net_list[i][j])
            tot_nets = tot_net_list[i][j]
            newgrid.generate_nets(tot_nets)
            newgrid.write_nets(subdir, i, j)
            newgrid.print_nets()
            newgrid.nets_to_null()
            print(newgrid)
            print(tot_net_list[i][j])

