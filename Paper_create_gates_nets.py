from Gridfile import Grid
from independent_functions import get_name_circuitfile, create_fpath


nets_per_grid = 10
gridfiles = 20
xys = [[30,30] for _ in range(gridfiles)]
tot_gate_list = [100]*gridfiles
tot_net_list = [[120 for _ in range(nets_per_grid)] for _ in range(gridfiles)]
subdir = "circuit_map_120"


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
        for j in range(len(tot_net_list[0])):
            print("enter net loop", i, j, tot_net_list[i][j])
            tot_nets = tot_net_list[i][j]
            newgrid.generate_nets(tot_nets)
            newgrid.write_nets(subdir, i, j)
            newgrid.print_nets()
            newgrid.nets_to_null()
            print(newgrid)
            print(tot_net_list[i][j])

