
import matplotlib.pyplot as plt
import os
from testingmodule import lprint

path = os.path.curdir
path = os.path.join(path, "results")
path = os.path.join(path, "generated")
path = os.path.join(path, "x30y30")
path = os.path.join(path, "ITER200")

destinationpath = os.path.join(os.path.curdir, "Refscatter200")


def get_files():
    return [os.path.join(fdata[0], fdata[2][1]) for fdata in os.walk(path) if 'all_data.csv' in fdata[2]]

def reorder_by_netlength(files, lengths, iters, chipsize):
    """ Groups files by length and gathers individual filenames
    """
    netlendict = {k:{"filenames":[], "Ns":[], "Cs":[]} for k in lengths}
    f_c_start = 54
    f_start = f_c_start-6 + len(str(iters))+2*len(str(chipsize))
    for f in files:
        k = int(f[f_start:f[f_start:].index(os.sep) + f_start])
        strk = str(k) + os.sep
        nl_num_S = f_start+len(strk)*2+1
        i = f[nl_num_S:f[nl_num_S:].index(os.sep)+nl_num_S]
        #
        # print(f)
        # print(f[f_start:f[f_start:].index(os.sep) + f_start])
        # print(i)
        # print(f[f_c_start:f_c_start+1])
        fcheck = f[48+2*len(str(chipsize))+len(str(iters)):48+2*len(str(chipsize))+len(strk)+len(str(iters))]
        try:
            netlendict[k]["Ns"].append("N" + str(k) + "_" + str(i))
            netlendict[k]["Cs"].append("C100" + "_" + f[f_c_start:f_c_start+1])
            netlendict[k]["filenames"].append(f)
        except:
            lprint(netlendict.keys())
            raise ValueError("lel")
    return netlendict


def get_scatterpoints(f):
    """ Gets the scatterpoints from a file
    """
    readfile = open(f, 'r')
    xs = []
    ys = []
    for line in readfile.readlines():
        data = line.split(";")
        xs.append(int(data[0]))
        ys.append(int(data[1]))
    readfile.close()
    return xs, ys

def get_netlen_scatterpoints(files, netlendict):
    """ Groups by netlist length and collects scatterpoints per length category
    """
    netlen_pointdict = {k:{'xs':[], 'ys':[]} for k in [20,30,40,50,60,70,80,90]}
    for k in netlendict:
        strk = str(k)
        for i, f in enumerate(netlendict[k]["filenames"]):
            if strk in f[36:]:
                xs, ys = get_scatterpoints(f)
                netlen_pointdict[k]['xs'].extend(xs)
                netlen_pointdict[k]['ys'].extend(ys)
    return netlen_pointdict




def make_distr_ref_plots(basepoint_dict, netlendict):
    for k in netlendict:
        bxs = basepoint_dict[k]['xs']
        bxsmin = min(bxs)
        bys = basepoint_dict[k]['ys']
        for i, fname in enumerate(netlendict[k]["filenames"]):
            xs, ys = get_scatterpoints(fname)
            plt.scatter(bxs, bys, label="total distribution")
            plt.scatter(xs, ys, label="current file distribution")
            plt.xlim(bxsmin, k)
            plt.xlabel("nets placed")
            plt.ylabel("total_length")
            new_fname = os.path.join(destinationpath, netlendict[k]["Ns"][i] + netlendict[k]["Cs"][i] + ".png")
            # print(fname)
            # print(new_fname)
            if not os.path.exists(os.path.dirname(new_fname)):
                os.makedirs(os.path.dirname(new_fname))
            plt.savefig(new_fname)
            plt.clf()





if __name__ == '__main__':
    files = get_files()
    netlendict = reorder_by_netlength(files, [20, 30, 40, 50, 60, 70])
    basepoint_dict = get_netlen_scatterpoints(files)

    make_distr_ref_plots(basepoint_dict, netlendict)
