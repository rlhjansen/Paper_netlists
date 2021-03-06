# algorithms
import code.algorithms.hillclimber as hc
# import code.algorithms.func_objects as fo
import code.algorithms.simulatedannealing as sa
import code.algorithms.random_collector as rc
import code.algorithms.ppa as ppa
import code.algorithms.ppasela as sela
import code.algorithms.simplyX as simple
#
# data generation
import data_generation as dg

# paralellizing HC & SA
import multiprocessing as mp

# command line handling
import sys
import datetime


def ppa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ord):
    pop_cuts = [30]
    runners = [3,5,7]
    distances = [7]
    for p in pop_cuts:
        for r in runners:
            for d in distances:
                PPA = ppa.PPA(c, cX, n, nX, x, y, tag, iters=iters, generated=generated, pop_cut=p, runners=r, distance=d, ordering=ord)
                PPA.run_algorithm()

def sela_cycle(c, cX, n, nX, x, y, tag, generated, iters, ord):
    pop_cuts = [30]
    arbitrarys = [7,14,28]
    distances = [7]
    best_percent = [0.05, 0.1, 0.15, 0.2]
    for p in pop_cuts:
        for a in arbitrarys:
            for d in distances:
                for bp in best_percent:
                    PS = sela.PPASELA(c, cX, n, nX, x, y, tag, iters=iters, generated=generated, pop_cut=p, arbitrary=a, distance=d, ordering=ord, best_percent=bp)
                    PS.run_algorithm()

def sa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ord):
    start_temps = [100, 70, 40]
    end_temps = [20, 5]
    cvars = [100,70,40]
    dvars = [10,2]
    swapss = [1,2,3]
    SAs = []
    for s in swapss:
        for st in start_temps:
            for et in end_temps:
                SAs.append(sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=iters, schema="exp", start_temp=st, end_temp=et, swaps=s, ordering=ord))
            SAs.append(sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=iters, schema="log", start_temp=st, swaps=s, ordering=ord))
            SAs.append(sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=iters, schema="linear", start_temp=st, swaps=s, ordering=ord))
        for cv in cvars:
            for dv in dvars:
                SAs.append(sa.SA(c, cX, n, nX, x, y, tag, generated=generated, iters=iters, schema="geman", cv=cv, dv=dv, swaps=s, ordering=ord))
    pool = mp.Pool(processes=None)
    pool.map(multi_run_pure_iterative, SAs)
    pool.close()

def hc_cycle(c, cX, n, nX, x, y, tag, generated, iters, ord):
    swapss = [1,2,3,4,5,6,7]
    pool = mp.Pool(processes=None)
    HCs = [hc.HC(c, cX, n, nX, x, y, tag, iters=iters, generated=generated, swaps=s, ordering=ord) for s in swapss]
    pool.map(multi_run_pure_iterative, HCs)
    pool.close()

def rc_cycle(c, cX, n, nX, x, y, tag, generated, iters):
    RC = rc.RC(c, cX, n, nX, x, y, tag, iters=iters, generated=generated, swaps=s, ordering=ord)
    RC.run_algorithm()

def multi_run_pure_iterative(alg_object):
    alg_object.run_algorithm()


def run_generated(tag, generated=True, spec=None):
    gates = [100]
    circuits = 1
    netlist = [80, 100, 60]
    netlists = 3
    x = 30
    y = 30
    iters = 10000
    ordering = "percdiv"
    while True:
        for c in gates:
            for cX in range(circuits):
                for n in netlist:
                    for nX in range(netlists):
                        if spec:
                            if "sa" in spec:
                                sa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "hc" in spec:
                                hc_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "sela" in spec:
                                sela_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "ppa" in spec:
                                ppa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "rc" in spec:
                                rc_cycle(c, cX, n, nX, x, y, tag, generated, iters)
                        else:
                            #sa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            hc_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            sela_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            ppa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)



def run_baseline(tag, generated=False, spec=None):
    gates = [25,50]
    circuits = 3
    netlist = [[30,40,50], [50,60,70]]
    netlists = 1
    x = 30
    y = 30
    iters = 10000
    ordering = "percdiv"
    while True:
        for i, c in enumerate(gates):
            for cX in range(circuits):
                for n in netlist[i]:
                    for nX in range(netlists):
                        if spec:
                            if "sa" in spec:
                                sa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "hc" in spec:
                                hc_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "sela" in spec:
                                sela_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "ppa" in spec:
                                ppa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            if "rc" in spec:
                                rc_cycle(c, cX, n, nX, x, y, tag, generated, iters)
                        else:
                            sa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            hc_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            sela_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)
                            ppa_cycle(c, cX, n, nX, x, y, tag, generated, iters, ordering)

def parse_args(args):
        if "gen" in sys.argv:
            if 'x' in sys.argv:
                x = sys.argv[sys.argv.index('x')+1]
                y = sys.argv[sys.argv.index('y')+1]
            else:
                x = 30
                y = 30
            dg.main(x, y)
        elif "optimize" in sys.argv:
            while "tag" not in sys.argv:
                print("needs tag for who is running\n ex: tag *name initials*")
            else:
                if "spec" in sys.argv:
                    inp = input("type which algs you want to try out\n(ppa, sela, sa, hc, rc)")
                    inp = inp.split(" ")
                else:
                    inp = None
                if "baseline" in sys.argv:
                    run_baseline(sys.argv[sys.argv.index('tag')+1], spec=inp)
                else:
                    run_generated(sys.argv[sys.argv.index('tag')+1], spec=inp)
        else:
            print("invalid argument\neither gen or optimize")


def simple_generator(c, cX, n, puzzels, x, y, tag="TESTRUN", iters=None):
    for nx in range(puzzels):
        net_num = nx
        cX = 0
        c = 100
        s = simple.SIMPLY(c, cX, n, net_num, x, y, tag, iters=iters)
        yield s

def meh(alg_obj):
    alg_obj.run_algorithm()

def gatherSimple(size, iters=1, start_add=10, end_iter=91, tag="TESTRUN"):
    print("gathering for chip of size:", size, "x", size)
    print("gatherin for netlists of length", start_add, "up to", end_iter-1+start_add)
    print("getting", iters, "orders per netlist")

    lens = [i+start_add for i in range(end_iter-start_add)]
    for n in lens:
        pool = mp.Pool(mp.cpu_count()-1)
        Simples = simple_generator(100, 0, n, 20, size, size, tag=tag, iters=iters)
        pool.map(meh, Simples)
        pool.close()


if __name__ == '__main__':
    #parse_args(sys.argv)
    starttime = datetime.datetime.now()
    for chipsize in [20, 30, 40, 50, 60, 70, 80, 90, 100]:
        gatherSimple(chipsize, iters=200, tag='test-okt')
        print("time elapsed:\t", datetime.datetime.now() - starttime)
