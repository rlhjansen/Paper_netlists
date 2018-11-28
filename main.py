# algorithms
import code.algorithms.hillclimber as hc
# import code.algorithms.func_objects as fo
import code.algorithms.simulatedannealing as sa
import code.algorithms.random as rc
import code.algorithms.ppa as ppa
import code.algorithms.ppasela as sela
#
# data generation
import data_generation as dg

import sys


def ppa_cycle(c, cX, n, nX, x, y, tag, generated):
    pop_cuts = [20,25,30,35,40,45]
    runners = [3,4,5,6,7,8]
    distances = [3,4,5,6,7]
    for p in pop_cuts:
        for r in runners:
            for d in distances:
                PPA = ppa.PPA(c, cX, n, nX, x, y, tag, iters=10000, generated=generated, pop_cut=p, runners=r, distance=d, ordering="max")
                PPA.run_algorithm()

def sela_cycle(c, cX, n, nX, x, y, tag, generated):
    pop_cuts = [20,25,30,35,40,45]
    arbitrarys = [4,6,8,12,18,27]
    distances = [3,4,5,6,7]
    best_percent = [0.05, 0.1, 0.15, 0.2]
    for p in pop_cuts:
        for a in arbitrarys:
            for d in distances:
                for bp in best_percent:
                    PS = sela.PPASELA(c, cX, n, nX, x, y, tag, iters=10000, generated=generated, pop_cut=p, arbitrary=a, distance=d, ordering="max")
                    PS.run_algorithm()

def sa_cycle(c, cX, n, nX, x, y, tag, generated):
    start_temps = [100, 70, 40]
    end_temps = [20, 5]
    cvars = [100,70,40]
    dvars = [10,2]
    swapss = [1,2,3]
    for s in swapss:
        for st in start_temps:
            for et in end_temps:
                SA = sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=10000, schema="exp", start_temp=st, end_temp=et, swaps=s)
                SA.run_algorithm()
            SA = sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=10000, schema="log", start_temp=st, swaps=s)
            SA = sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=10000, schema="linear", start_temp=st, swaps=s)
        for cv in cvars:
            for dv in dvars:
                SA = sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=10000, schema="geman", cv=cv, dv=dv, swaps=s)
                SA.run_algorithm()

def hc_cycle(c, cX, n, nX, x, y, tag, generated):
    swapss = [1,2,3]
    for s in swapss:
        HC = hc.HC(c, cX, n, nX, x, y, tag, iters=10000, generated=generated, swaps=s)


def run_generated(tag, generated=True):
    gates = [100]
    circuits = 5
    netlist = [100]
    netlists = 5
    x = 30
    y = 30
    while True:
        for c in gates:
            for cX in range(circuits):
                for n in netlist:
                    for nX in range(netlists):
                        sa_cycle(c, cX, n, nX, x, y, tag, generated)
                        ppa_cycle(c, cX, n, nX, x, y, tag, generated)
                        sela_cycle(c, cX, n, nX, x, y, tag, generated)
                        hc_cycle(c, cX, n, nX, x, y, tag, generated)



def run_baseline(tag, generated=False):
    gates = [25,50]
    circuits = 3
    netlist = [[30,40,50], [50,60,70]]
    netlists = 1
    x = 30
    y = 30
    while True:
        for i, c in enumerate(gates):
            for cX in range(circuits):
                for n in netlist[i]:
                    for nX in range(netlists):
                        sa_cycle(c, cX, n, nX, x, y, tag, generated)
                        ppa_cycle(c, cX, n, nX, x, y, tag, generated)
                        sela_cycle(c, cX, n, nX, x, y, tag, generated)
                        hc_cycle(c, cX, n, nX, x, y, tag, generated)



if __name__ == '__main__':
    if "gen" in sys.argv:
        if 'x' in sys.argv:
            x = sys.argv[sys.argv.index('x')+1]
            y = sys.argv[sys.argv.index('y')+1]
        else:
            x = 30
            y = 30
        dg.main(x, y)
    elif "optimize" in sys.argv:
        if "tag" not in sys.argv:
            print("needs tag for who is running\n ex: tag *name initials*")
        else:
            g = not "baseline" in sys.argv
            if g:
                run_generated(sys.argv[sys.argv.index('tag')+1], generated=g)
            else:
                print("running baseline")
                run_baseline(sys.argv[sys.argv.index('tag')+1], generated=g)
    else:
        print("ah, no command for once, computers need free time too you know")
