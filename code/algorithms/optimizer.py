import os
import matplotlib.pyplot as plt

from ..classes.grid import file_to_grid
from ..alghelper import combine_score


class Optimizer:
    def __init__(self, c, cX, n, nX, x, y, iters, tag, generated=True, solutiondict=False, **kwargs):
        self.c = c      # circuit with c gates
        self.cX = cX    # Xth circuit with c gates
        self.n = n      # netlist with n nets
        self.nX = nX    # Xth netlist with n nets
        self.x = x
        self.y = y
        self.tag = tag
        self.generated = kwargs.get("generated", generated) #either test circuits or generated circuits
        self.iters = iters
        self.used_results = []
        self.used_cl = []
        self.used_scores = []
        self.all_results = []
        self.all_scores = []
        self.all_cl = []
        self.all_data = []
        self.setup_load_paths()
        self.no_save = kwargs.get("no_save", False)
        if solutiondict:
            self.solutiondict = solutiondict

    def make_circuit(self, *args):
        print("making circuit")
        G = file_to_grid(self.circuit_path, None)
        G.read_nets(self.netlist_path)
        G.disconnect()
        return G

    def add_iter(self, _print=False):
        self.used_scores.append(self.used_score)
        self.used_results.append(self.used)
        self.used_cl.append([self.used_len, self.used_conn])
        self.all_results.append(self.current)
        self.all_scores.append(self.current_score)
        self.all_cl.append([self.cur_len, self.cur_conn])
        if _print:
            print(self.used_scores)
            #print(self.used_results)
            print(self.all_scores)
            #print(self.all_results)

    def add_iter_batch(self, batch_results):
        iter = str(self.iter)
        self.all_data.extend([[iter] + br for br in batch_results])


    def add_iter_batch_solutiondict(self):
        for key in sorted([k for k in self.solutiondict.keys()]):
            valuelist = self.solutiondict.get(key)

        self.used_results.extend(orders)
        self.used_scores.extend([combine_score(score[0], score[1], scoring=self.best_ordering, total_nets=self.n) for score in scores])
        self.used_cl.extend(scores)
        self.used_results.append("----")
        self.used_scores.append("----")
        self.used_cl.append("----")


    def setup_load_paths(self):
        abspath = os.path.abspath(__file__)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.join(abspath, "data")
        if self.generated:
            abspath = os.path.join(abspath, "official_reference")
            abspath = os.path.join(abspath, "generated")
            abspath = os.path.join(abspath, "x"+str(self.x)+"y"+str(self.y))
            abspath = os.path.join(abspath, "g0")
        else:
            abspath = os.path.join(abspath, "baseline")
        abspath = os.path.join(abspath, 'C'+str(self.c))
        abspath = os.path.join(abspath, 'C'+str(self.c)+"_"+str(self.cX))
        self.circuit_path = abspath+".csv"
        abspath = os.path.join(abspath, "N"+str(self.n))
        abspath = os.path.join(abspath, "N"+str(self.n)+"_"+str(self.nX)+".csv")
        self.netlist_path = abspath


    def set_saveloc(self, algtype, **kwargs):
        if self.no_save:
            return
        abspath = os.path.abspath(__file__)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.dirname(abspath)
        abspath = os.path.join("results")
        if self.generated:
            abspath = os.path.join(abspath, "generated")
            abspath = os.path.join(abspath, "x"+str(self.x)+"y"+str(self.y))
        else:
            abspath = os.path.join(abspath, "baseline")

        abspath = os.path.join(abspath, "ITER" + str(self.iters))
        if algtype == 'hc':
            abspath = self.saveloc_hc(abspath, **kwargs)
        if algtype == 'sa':
            abspath = self.saveloc_sa(abspath, **kwargs)
        if algtype == 'ppa':
            abspath = self.saveloc_ppa(abspath, **kwargs)
        if algtype == 'sela':
            abspath = self.saveloc_sela(abspath, **kwargs)
        if algtype == 'rc':
            abspath = self.saveloc_rc(abspath)
        if algtype == 'simple':
            abspath = self.saveloc_simple(abspath)

        abspath = os.path.join(abspath, 'C'+str(self.c))
        if not os.path.exists(abspath):
            os.makedirs(abspath)
        abspath = os.path.join(abspath, 'C'+str(self.c)+"_"+str(self.cX))
        if not os.path.exists(abspath):
            os.makedirs(abspath)
        abspath = os.path.join(abspath, "N"+str(self.n))
        abspath = os.path.join(abspath, "N"+str(self.n)+"_"+str(self.nX))
        if not os.path.exists(abspath):
            os.makedirs(abspath)

        prevdata = len(os.listdir(abspath))
        abspath = os.path.join(abspath, "t"+self.tag+str(prevdata))
        if not os.path.exists(abspath):
            os.makedirs(abspath)
        self.savedir = abspath


    @staticmethod
    def saveloc_hc(abspath, **kwargs):
        abspath = os.path.join(abspath, "HC")
        assert "move" in kwargs
        return os.path.join(abspath, "MOVE" + str(kwargs.get("move")))


    @staticmethod
    def saveloc_rc(abspath):
        return os.path.join(abspath, "RC")

    @staticmethod
    def saveloc_simple(abspath):
        return os.path.join(abspath, "Simple")

    @staticmethod
    def saveloc_sa(abspath, **kwargs):
        abspath = os.path.join(abspath, "SA")

        assert "schema" in kwargs
        abspath = os.path.join(abspath, "S" + kwargs.get("schema"))
        if "start_temp" in kwargs:
            abspath = os.path.join(abspath, "T" + str(kwargs.get("start_temp")))
        if "end_temp" in kwargs:
            abspath = os.path.join(abspath, "ET" + str(kwargs.get("end_temp")))

        if "cv" in kwargs:
            abspath = os.path.join(abspath, "cv" + str(kwargs.get("cv")))
        if "dv" in kwargs:
            abspath = os.path.join(abspath, "dv" + str(kwargs.get("dv")))

        assert "swaps" in kwargs
        return os.path.join(abspath, "SW" + str(kwargs.get("swaps")))

        return abspath

    @staticmethod
    def saveloc_ppa(abspath, **kwargs):
        abspath = os.path.join(abspath, "PPA")

        assert "pop_cut" in kwargs
        abspath = os.path.join(abspath, "P"+str(kwargs.get("pop_cut")))  #popsize
        assert "runners" in kwargs
        abspath = os.path.join(abspath, "R"+str(kwargs.get("runners")))  #max runners
        assert "distance" in kwargs
        abspath = os.path.join(abspath, "D"+str(kwargs.get("distance")))  #max distance
        assert "ordering" in kwargs
        abspath = os.path.join(abspath, "O"+str(kwargs.get("ordering")))  #max distance

        return abspath

    @staticmethod
    def saveloc_sela(abspath, **kwargs):
        abspath = os.path.join(abspath, "SELA")

        assert "pop_cut" in kwargs
        abspath = os.path.join(abspath, "P"+str(kwargs.get("pop_cut")))  #popsize
        assert "best_percent" in kwargs
        abspath = os.path.join(abspath, "BP"+str(kwargs.get("best_percent")))  #percentage best
        assert "arbitrary" in kwargs
        abspath = os.path.join(abspath, "A"+str(kwargs.get("arbitrary")))  #arbitrary
        assert "distance" in kwargs
        abspath = os.path.join(abspath, "D"+str(kwargs.get("distance")))  #max distance
        assert "ordering" in kwargs
        abspath = os.path.join(abspath, "O"+str(kwargs.get("ordering")))  #max distance
        return abspath

    def save(self, used_scores=False, used_results=False, all_scores=False, all_results=False):
        if used_scores:
            self.save_used_scores()
            self.save_used_cl()
            print("saved used scores")
        if used_results:
            self.save_used_results()
            print("saved used")
        if all_scores:
            self.save_all_scores()
            self.save_all_cl()
            print("saved all scores")
        if all_results:
            self.save_all_results()
            print("saved all")


    def save_used_scores(self):
        savefile = os.path.join(self.savedir, 'used_scores.txt')
        with open(savefile, 'w') as sf:
            sf.write("\n".join([str(score) for score in self.used_scores]))

    def save_used_results(self):
        savefile = os.path.join(self.savedir, 'used_results.txt')
        with open(savefile, 'w+') as sf:
            sf.write("\n".join([",".join([net for net in line]) for line in self.used_results]))

    def save_used_cl(self):
        savefile = os.path.join(self.savedir, 'used_cl.txt')
        with open(savefile, 'w+') as sf:
            sf.write("\n".join([",".join([str(sp) for sp in score]) for score in self.used_cl]))

    def save_all_cl(self):
        savefile = os.path.join(self.savedir, 'all_cl.txt')
        with open(savefile, 'w+') as sf:
            sf.write("\n".join([",".join([str(sp) for sp in score]) for score in self.all_cl]))

    def save_all_scores(self):
        savefile = os.path.join(self.savedir, 'all_scores.txt')
        with open(savefile, 'w+') as sf:
            sf.write("\n".join([str(score) for score in self.all_scores]))

    def save_all_results(self):
        savefile = os.path.join(self.savedir, 'all_results.txt')
        with open(savefile, 'w+') as sf:
            sf.write("\n".join([",".join([net for net in line]) for line in self.all_results]))

    def save_all_data(self, plot=False):
        if plot:
            self.save_scatter_data()
        self.save_data = [";".join([d[0], str(d[1]), str(d[2]), ",".join(d[3:])]) for d in self.all_data]
        savefile = os.path.join(self.savedir, "all_data.csv")
        with open(savefile, "w+") as sf:
            for line in self.save_data:
                sf.write(line + "\n")

    def save_scatter_data(self):
        data = sorted(self.all_data, key=lambda x : [-x[1], x[2]])
        xs = [d[1] for d in data]
        ys = [d[2] for d in data]
        plt.scatter(xs, ys)
        plt.xlim(min(xs), self.n)
        plt.xlabel("nets placed")
        plt.ylabel("total_length")
        plt.title("goals = " + str(self.n) + " nets, sample of " + str(self.iters))
        savefile = os.path.join(self.savedir, "scatter.png")
        if not os.path.exists(os.path.dirname(savefile)):
            os.makedirs(os.path.dirname(savefile))
        plt.savefig(savefile)
        plt.clf()
