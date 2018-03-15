from random import randint, random
from netlist import Grid, get_name_netfile, get_name_circuitfile, create_fpath
from independent_functions import swap_two_elems, swap_up_to_X_elems

def check_anneal(cur_len, cur_conn, cur_ord, sol_len, tot_nets, nets_solved, sol_ord, func):
    if not nets_solved == tot_nets:
        print(cur_conn)
        if cur_conn > nets_solved:
            sol_ord = cur_ord
            nets_solved = cur_conn
            print("change sol order :: greater")
            if cur_conn == tot_nets:
                sol_len = cur_len
                all_connected = True
                print("all_connected")
        elif cur_conn == nets_solved:
            print("enter chance mode")
            if chance < random():
                print("change sol order :: equal")
                sol_ord = cur_ord
                nets_solved = cur_conn


def simulatead_anneal(subdir, gridfile, netfile, iterations, chance, _print=False):

    G, cur_ord, sol_ord, sol_len, nets_solved, tot_nets, all_connected = SSHC(gridfile, subdir, netfile)

    for i in range(iterations):

        if sol_ord:
            cur_ord = swap_two_elems(sol_ord)

        cur_len, cur_conn = G.solve_order(cur_ord)

        sol_len, sol_ord, nets_solved = check_climb(cur_len, cur_conn, cur_ord, sol_len, tot_nets, nets_solved, sol_ord, chance)

        if _print:
            print(G)
        G.reset_nets()
        print("iteration =", i, "best solved =", nets_solved)
        print("connected paths =", cur_conn, "/", tot_nets)
    G.solve_order(sol_ord)
    print(G)
    print("Final Path =\t", sol_ord, "\nFinal Length =\t", sol_len, "\nAll connected =", all_connected, nets_solved)





SUBDIR = "DAAL_circuits"
gridfile = get_name_circuitfile(0, 18, 13, 25)
netfile = get_name_netfile(0, 1)
simulatead_anneal(SUBDIR, gridfile, netfile, 1000, True, _print=False)
