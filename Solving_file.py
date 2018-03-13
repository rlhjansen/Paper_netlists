from random import randint
from netlist import Grid, get_name_netfile, get_name_circuitfile, create_fpath

def swap_two_elems(path):
    end_index = len(path)-1
    print(path[-1])
    print(path[end_index])
    i1 = randint(0, end_index)
    i2 = randint(0, end_index)
    while i2 == i1:
        i2 = randint(0, end_index)
    path[i1], path[i2] = path[i2], path[i1]
    return path


def simple_hillclimber(subdir, gridfile, netfile, iterations):

    gridfile = create_fpath(SUBDIR, gridfile)


    G = Grid.file_to_Grid(gridfile, None)
    G.read_nets(subdir, netfile)
    cur_order = G.get_random_net_order()

    sol_length = 500000
    sol_order = []
    nets_to_solve = len(cur_order)
    best_nets_solved = 0
    all_connected = False

    for i in range(iterations):

        if sol_order:
            cur_order = swap_two_elems(sol_order)

        cur_sol_len, solved = G.simple_solve_order(cur_order)
        print("current solution length", cur_sol_len)

        if not all_connected:
            if solved >= best_nets_solved:
                sol_order = cur_order
                if solved == nets_to_solve:
                    sol_length = cur_sol_len
                    all_connected = True
        else:
            if cur_sol_len < sol_length:
                sol_order = cur_order
        if cur_sol_len:
            print("path", cur_order, "yields a solution")
            if cur_sol_len < sol_length:
                sol_order, sol_length = cur_order, cur_sol_len

        print(G)
        G.reset_nets()
        print("path on try", i, "=\t", sol_order, "\nlen =\t", sol_length)
    print("Final Path =\t", sol_order, "\nFinal Length =\t", sol_length)


SUBDIR = "DAAL_circuits"
gridfile = get_name_circuitfile(0, 18, 13, 25)
netfile = get_name_netfile(0, 2)

simple_hillclimber(SUBDIR, gridfile, netfile, 3)


