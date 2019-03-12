from refscatter import reorder_by_netlength
from statistics import mean
from collections import OrderedDict, Counter

from testingmodule import lprint


import matplotlib.pyplot as plt
import os

from matplotlib.pyplot import figure
figure(num=None, figsize=(15, 30), dpi=160, facecolor='w', edgecolor='k')




def get_files(xsize, ysize, iters):
    path = os.path.curdir
    path = os.path.join(path, "results")
    path = os.path.join(path, "generated")
    path = os.path.join(path, "x" + str(xsize) + "y" + str(ysize))
    path = os.path.join(path, "ITER" + str(iters))
    path = os.path.join(path, "Simple")
    path = os.path.join(path, "C100")
    path = os.path.join(path, "C100_0")

    return [os.path.join(fdata[0], fdata[2][fdata[2].index('all_data.csv')]) for fdata in os.walk(path) if 'all_data.csv' in fdata[2]]


def get_first_placed_count(f):
    """ Gets the first scatterpoint from a file
    """
    readfile = open(f, 'r')
    line = readfile.readlines()[0]
    data = line.split(";")
    firstcount = int(data[1])
    readfile.close()
    return firstcount

def get_min_placed_count(f):
    """ Gets the best scatterpoint from a file
    """
    readfile = open(f, 'r')
    best_count = 5000
    for line in readfile.readlines():
        data = line.split(";")
        count = int(data[1])
        if count < best_count:
            best_count = count
    readfile.close()
    return best_count


def get_max_placed_count(f):
    """ Gets the best scatterpoint from a file
    """
    readfile = open(f, 'r')
    best_count = 0
    for line in readfile.readlines():
        data = line.split(";")
        count = int(data[1])
        if count > best_count:
            best_count = count
    readfile.close()
    return best_count

def get_mean_placed_count(f, k):
    """ Gets the scatterpoints from a file
    """
    readfile = open(f, 'r')
    place_counts = []
    for line in readfile.readlines():
        data = line.split(";")
        place_counts.append(1 if int(data[1]) == k else 0)
    readfile.close()
    return mean(place_counts)


def make_netlen_scatterpoints_placepercent(files, netlengths, netlendict, squaresize):
    """ Groups by netlist length and collects scatterpoints per length category
    """
    netlen_countdict = {k:{'f':[], 'bc':[], 'mc':[], 'minc':[]} for k in netlengths}
    for k in netlendict:
        strk = str(k)
        for i, f in enumerate(netlendict[k]["filenames"]):
            if strk in f[36:]:
                firstcount = get_first_placed_count(f)
                best_count = get_max_placed_count(f)
                mean_count = get_mean_placed_count(f, k)
                min_count = get_min_placed_count(f)
                netlen_countdict[k]['f'].append(firstcount/k)
                netlen_countdict[k]['bc'].append(best_count/k)
                netlen_countdict[k]['mc'].append(mean_count/k)
                netlen_countdict[k]['minc'].append(min_count/k)
    return netlen_countdict


def make_distr_percent(netlen_countdict):
    alpha_val = None
    first_col = 'b'
    mean_col = 'g'
    best_col = 'r'
    worst_col = 'y'
    destinationpath = os.path.join(os.path.curdir, "NonAlphaCompareScatter200")
    destinationpath = os.path.join(destinationpath, str(squaresize))

    for k in netlen_countdict:
        plt.scatter([k for i in range(len(netlen_countdict[k]['f']))], netlen_countdict[k]['f'], c=first_col, alpha=alpha_val)
        plt.scatter([k for i in range(len(netlen_countdict[k]['mc']))], netlen_countdict[k]['mc'], c=mean_col, alpha=alpha_val)
        plt.scatter([k for i in range(len(netlen_countdict[k]['bc']))], netlen_countdict[k]['bc'], c=best_col, alpha=alpha_val)
        plt.scatter([k for i in range(len(netlen_countdict[k]['minc']))], netlen_countdict[k]['minc'], c=worst_col, alpha=alpha_val)
    plt.plot([], [], c=first_col, label="first", alpha=alpha_val)
    plt.plot([], [], c=mean_col, label="mean", alpha=alpha_val)
    plt.plot([], [], c=best_col, label="best", alpha=alpha_val)
    plt.plot([], [], c=worst_col, label="worst", alpha=alpha_val)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "all_percentages.png")
    print(new_fname)
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    plt.savefig(new_fname)
    plt.clf()

    for k in netlen_countdict:
        plt.scatter([k for i in range(len(netlen_countdict[k]['f']))], netlen_countdict[k]['f'], c=first_col, alpha=alpha_val)
    plt.plot([], [], c=first_col, label="first", alpha=alpha_val)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "first_percentages.png")
    print(new_fname)
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    plt.savefig(new_fname)
    plt.clf()

    for k in netlen_countdict:
        plt.scatter([k for i in range(len(netlen_countdict[k]['mc']))], netlen_countdict[k]['mc'], c=mean_col, alpha=alpha_val)
        plt.scatter([k for i in range(len(netlen_countdict[k]['f']))], netlen_countdict[k]['f'], c=first_col, alpha=alpha_val)
    plt.plot([], [], c=first_col, label="first", alpha=alpha_val)
    plt.plot([], [], c=mean_col, label="mean", alpha=alpha_val)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "mean_first_percentages.png")
    print(new_fname)
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    plt.savefig(new_fname)
    plt.clf()

    first_col = 'b'
    mean_col = 'g'
    best_col = 'r'
    worst_col = 'y'
    for k in netlen_countdict:
        plt.scatter([k for i in range(len(netlen_countdict[k]['f']))], netlen_countdict[k]['f'], c=first_col, alpha=alpha_val)
        plt.scatter([k for i in range(len(netlen_countdict[k]['bc']))], netlen_countdict[k]['bc'], c=best_col, alpha=alpha_val)
    plt.plot([], [], c=first_col, label="first", alpha=alpha_val)
    plt.plot([], [], c=best_col, label="best", alpha=alpha_val)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "first_best_percentages.png")
    print(new_fname)
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    plt.savefig(new_fname)
    plt.clf()


    first_col = 'b'
    mean_col = 'g'
    best_col = 'r'
    worst_col = 'y'
    for k in netlen_countdict:
        plt.scatter([k for i in range(len(netlen_countdict[k]['mc']))], netlen_countdict[k]['mc'], c=mean_col, alpha=alpha_val)
        plt.scatter([k for i in range(len(netlen_countdict[k]['bc']))], netlen_countdict[k]['bc'], c=best_col, alpha=alpha_val)
    plt.plot([], [], c=mean_col, label="mean", alpha=alpha_val)
    plt.plot([], [], c=best_col, label="best", alpha=alpha_val)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)

    new_fname = os.path.join(destinationpath, "mean_best_percentages.png")
    print(new_fname)
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    plt.savefig(new_fname)
    plt.clf()



def make_distr_percent_means(netlen_countdict):
    first_col = 'b'
    mean_col = 'g'
    best_col = 'r'
    worst_col = 'y'
    for k in netlen_countdict:
        plt.scatter(k, mean(netlen_countdict[k]['f']), c=first_col)
        plt.scatter(k, mean(netlen_countdict[k]['mc']), c=mean_col)
        plt.scatter(k, mean(netlen_countdict[k]['bc']), c=best_col)
        plt.scatter(k, mean(netlen_countdict[k]['minc']), c=worst_col)
    plt.plot([], [], c=first_col, label="first")
    plt.plot([], [], c=mean_col, label="mean")
    plt.plot([], [], c=best_col, label="best")
    plt.plot([], [], c=worst_col, label="worst")
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    plt.show()


def make_distr_ref_plots(netlendict, netlen_countdict):
    destinationpath = os.path.join(os.path.curdir, "NonAlphaCompareScatter200")
    destinationpath = os.path.join(destinationpath, str(squaresize))

    for k in netlendict:
        bxs = netlen_countdict[k]['xs']
        bxsmin = min(bxs)
        bys = netlen_countdict[k]['ys']
        for i, fname in enumerate(netlendict[k]["filenames"]):
            plt.scatter(bxs, bys, label="total distribution")
            plt.scatter(xs, ys, label="current file distribution")
            plt.xlim(bxsmin, k)
            plt.xlabel("nets placed")
            plt.ylabel("total_length")
            new_fname = os.path.join(destinationpath, netlendict[k]["Ns"][i] + netlendict[k]["Cs"][i] + ".png")
            print(fname)
            print(new_fname)
            if not os.path.exists(os.path.dirname(new_fname)):
                os.makedirs(os.path.dirname(new_fname))
            plt.savefig(new_fname)
            plt.clf()

def results_to_percents(results, p):
    fCounter = Counter(results)
    percdict = OrderedDict()
    for key in fCounter:
        percdict[key] = {"placed_val":key, "density":round(fCounter[key]/len(results), 3)}
    return percdict

def plottext(xs, ys, ss, p, **kwargs):
    if len(xs) == len(ys) and len(ss) == len(ys):
        for i in range(len(xs)):
            print("in loop", xs[i], ys[i], ss[i], kwargs.get("color"), kwargs.get("backgroundcolor"))
            plt.text(xs[i]-0.5, ys[i]-0.005, ss[i], color=kwargs.get("color"), fontsize=kwargs.get("fontsize", 8))
    print(len(xs), len(ys), len(ss), p, kwargs.get("color"), kwargs.get("backgroundcolor"))


def make_single_percent_plot(argkey, color, k, netlen_countdict, dont_make=False):
    vals = results_to_percents(netlen_countdict[k][argkey], argkey)
    if not dont_make:
        plottext([k for _ in vals.keys()], [key for key in vals.keys()], [vals[key]["density"] for key in vals.keys()], argkey, color=color, fontsize=20)
    plt.scatter([k for i in range(len(netlen_countdict[k][argkey]))], netlen_countdict[k][argkey], c=color, alpha=0)
    return mean(netlen_countdict[k][argkey])


def make_percent_placement_plots(netlengths, netlen_countdict, squaresize, directsave=True):
    alpha_val = 0
    first_col = 'blue'
    mean_col = 'green'
    best_col = 'red'
    worst_col = 'magenta'
    destinationpath = os.path.join(os.path.curdir, "PercentageCompareScatter200")
    destinationpath = os.path.join(destinationpath, str(squaresize))

    for k in netlen_countdict:
        print(k, netlen_countdict[k])
        make_single_percent_plot('f', first_col, k, netlen_countdict)
        make_single_percent_plot('mc', mean_col, k, netlen_countdict)
        make_single_percent_plot('bc', best_col, k, netlen_countdict)
        make_single_percent_plot('minc', worst_col, k, netlen_countdict)

    plt.plot([], [], c=first_col, label="first")
    plt.plot([], [], c=mean_col, label="mean")
    plt.plot([], [], c=best_col, label="best")
    plt.plot([], [], c=worst_col, label="worst")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "all_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)

    plt.savefig(new_fname)
    plt.clf()

    for k in netlen_countdict:
        make_single_percent_plot('f', first_col, k, netlen_countdict)
        make_single_percent_plot('mc', mean_col, k, netlen_countdict)

    plt.plot([], [], c=first_col, label="first")
    plt.plot([], [], c=mean_col, label="mean")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "first_mean_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()

    for k in netlen_countdict:
        make_single_percent_plot('f', first_col, k, netlen_countdict)
        make_single_percent_plot('bc', best_col, k, netlen_countdict)

    plt.plot([], [], c=first_col, label="first")
    plt.plot([], [], c=best_col, label="best")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "first_best_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()

    for k in netlen_countdict:
        make_single_percent_plot('mc', mean_col, k, netlen_countdict)
        make_single_percent_plot('bc', best_col, k, netlen_countdict)

    plt.plot([], [], c=mean_col, label="mean")
    plt.plot([], [], c=best_col, label="best")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "mean_best_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()

    fm = []
    for k in netlen_countdict:
        mean_at_k = make_single_percent_plot('f', first_col, k, netlen_countdict)
        fm.append(mean_at_k)
    plt.plot([k for k in netlen_countdict], fm, c=first_col)
    plt.plot([], [], c=first_col, label="first")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "first_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()

    bcm = []
    for k in netlen_countdict:
        mean_at_k = make_single_percent_plot('bc', best_col, k, netlen_countdict)
        bcm.append(mean_at_k)
    plt.plot([k for k in netlen_countdict], bcm, c=best_col)
    plt.plot([], [], c=best_col, label="best")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "best_percentages_mean.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()


    bcm = []
    bcmnp = []
    for k in netlen_countdict:
        mean_at_k = make_single_percent_plot('bc', best_col, k, netlen_countdict)
        bcm.append(mean_at_k)
        bcmnp.append(int(round(mean_at_k*k, 0)))
    # YEAH
    plottext([k for k in netlengths], [1.1 for _ in netlengths], bcmnp, "bc", color=best_col, fontsize=20)
    plt.plot([k for k in netlen_countdict], bcm, c=best_col)
    plt.plot([], [], c=best_col, label="best")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "toptext_best.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()


    bcm = []
    for k in netlen_countdict:
        mean_at_k = make_single_percent_plot('bc', best_col, k, netlen_countdict)
    plt.plot([], [], c=best_col, label="best")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "best_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()

    bcm = []
    fm = []
    for k in netlen_countdict:
        mean_at_kb = make_single_percent_plot('bc', best_col, k, netlen_countdict)
        mean_at_kf = make_single_percent_plot('f', first_col, k, netlen_countdict)
        bcm.append(mean_at_kb)
        fm.append(mean_at_kf)
    plt.plot([k for k in netlen_countdict], bcm, c=best_col)
    plt.plot([k for k in netlen_countdict], fm, c=first_col)
    plt.plot([], [], c=first_col, label="first")
    plt.plot([], [], c=best_col, label="best")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "first_best_mean_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()


    mcm = []
    fm = []
    for k in netlen_countdict:
        mean_at_kf = make_single_percent_plot('f', first_col, k, netlen_countdict, dont_make=True)
        mean_at_k = make_single_percent_plot('mc', mean_col, k, netlen_countdict, dont_make=True)
        fm.append(mean_at_kf)
        mcm.append(mean_at_k)
    plt.plot([k for k in netlen_countdict], fm, c=first_col)
    plt.plot([k for k in netlen_countdict], mcm, c=mean_col)
    plt.plot([], [], c=mean_col, label="mean")
    plt.plot([], [], c=first_col, label="first")
    plt.ylim(0.4, 1.05)
    plt.tick_params(labelsize=20)
    plt.xticks(netlengths)
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% nets sucesfully placed")
    plt.legend(fontsize = 30)
    new_fname = os.path.join(destinationpath, "mean_percentages.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    print(new_fname)
    plt.savefig(new_fname)
    plt.clf()

def make_netlen_solvabilitydict(files, netlengths, netlendict, chipsize, iters):
    netlen_solvabilitydict = {k:{'f':[], 'bc':[], 'mc':[], 'minc':[]} for k in netlengths}
    for k in netlendict:
        strk = str(k) + os.sep

        for i, f in enumerate(netlendict[k]["filenames"]):
            fcheck = f[48+2*len(str(chipsize))+len(str(iters)):48+2*len(str(chipsize))+len(strk)+len(str(iters))]
            if strk in fcheck:
                firstcount = get_first_placed_count(f)
                best_count = get_max_placed_count(f)
                mean_count = get_mean_placed_count(f, k)
                min_count = get_min_placed_count(f)
                netlen_solvabilitydict[k]['f'].append(1 if firstcount == k else 0)
                netlen_solvabilitydict[k]['mc'].append(mean_count)
                netlen_solvabilitydict[k]['bc'].append(1 if best_count == k else 0)
    return netlen_solvabilitydict


def make_solvability_plots(netlengths, solvabilitydict, chipsize, directsave=True, first=True, best=False):
    first_col = 'blue'
    destinationpath = os.path.join(os.path.curdir, "solvabilityplots")
    destinationpath = os.path.join(destinationpath, str(chipsize))

    solvability_scores_first = []
    solvability_scores_best = []
    ks = []
    for k in solvabilitydict:
        ysf = solvabilitydict[k]['f']
        ysb = solvabilitydict[k]['bc']
        ks.append(k)
        solvability_scores_first.append(mean(ysf))
        solvability_scores_best.append(mean(ysb))
        print(k, mean(ysf), ysf)
    if first:
        plt.plot(ks, solvability_scores_first, label="solvability unoptimized" + str(chipsize) + "x" + str(chipsize))
    if best:
        plt.plot(ks, solvability_scores_best, label="solvability based on best of 200" + str(chipsize) + "x" + str(chipsize))
    if directsave:
        add_solvability_labels()
        new_fname = os.path.join(destinationpath, "first_solvability" + str(chipsize) + "x" + str(chipsize) + ".png")
        if not os.path.exists(os.path.dirname(new_fname)):
            os.makedirs(os.path.dirname(new_fname))
        print(new_fname)
        plt.savefig(new_fname)
        plt.clf()


def add_solvability_labels():
    plt.xlabel("nets in tried netlist")
    plt.ylabel("% netlists fully placed")
    plt.legend()


def gather_data_chipsize(chipsize, netlengths, iters):
    files = get_files(chipsize, chipsize, iters)
    # lprint(files)
    netlendict = reorder_by_netlength(files, netlengths, iters, chipsize)
    # lprint([netlendict[k]["filenames"] for k in netlendict])
    # input()
    netlen_countdict = make_netlen_scatterpoints_placepercent(files, netlengths, netlendict, chipsize)
    netlen_solvability_dict = make_netlen_solvabilitydict(files, netlengths, netlendict, chipsize, iters)
    return files, netlendict, netlen_countdict, netlen_solvability_dict


def save_solvability_all():
    add_solvability_labels()
    destinationpath = os.path.join(os.path.curdir, "solvabilityplots")
    destinationpath = os.path.join(destinationpath, "everything")

    new_fname = os.path.join(destinationpath, "first_solvability.png")
    if not os.path.exists(os.path.dirname(new_fname)):
        os.makedirs(os.path.dirname(new_fname))
    plt.savefig(new_fname)
    plt.clf()


def make_solvability_comparison(sizes, iters):
    figure(num=None, figsize=(5, 5), dpi=160, facecolor='w', edgecolor='k')

    netlendicts_persize = []
    netlen_countdicts_persize = []
    netlen_solvabilitydicts_persize = []
    netlengths = [i+10 for i in range(81)]

    for chipsize in sizes:
        files, netlendict, netlen_countdict, netlen_solvabilitydict = gather_data_chipsize(chipsize, netlengths, iters)
        netlendicts_persize.append(netlendict)
        netlen_countdicts_persize.append(netlen_countdict)
        netlen_solvabilitydicts_persize.append(netlen_solvabilitydict)
    for i, chipsize in enumerate(sizes):
        make_solvability_plots(netlengths, netlen_solvabilitydicts_persize[i], chipsize, directsave=False)
    save_solvability_all()
    for i, chipsize in enumerate(sizes):
        make_solvability_plots(netlengths, netlen_solvabilitydicts_persize[i], chipsize)


def solvability_header_gen(chipsizes, best_of_N):
    for cs in chipsizes:
        random_solvability = ["solvability by random {}x{}".format(str(cs), str(cs))]
        mean_solvability = ["solvability of mean {}x{}".format(str(cs), str(cs))]
        best_solvability = ["solvability best of " +str(best_of_N) + " {}x{}".format(str(cs), str(cs))]
        yield random_solvability
        yield mean_solvability
        yield best_solvability


def make_solvability_csvs(chipsizes, best_of_N):
    netlendicts_persize = []
    netlen_countdicts_persize = []
    netlen_solvabilitydicts_persize = []
    netlengths = [i+10 for i in range(81)]
    csv_data_walk = [["netlist length"]] + [elem for elem in solvability_header_gen(chipsizes, best_of_N)]
    dw_len = len(csv_data_walk)
    csv_data_walk[0].extend([str(nl) for nl in netlengths])
    for i, chipsize in enumerate(chipsizes):
        print("getting solvability for chipsize", chipsize)
        j = i*3
        files, netlendict, netlen_countdict, netlen_solvabilitydict = gather_data_chipsize(chipsize, netlengths, best_of_N)
        print("arbitrary")
        lprint([str(mean(netlen_solvabilitydict[n]['f'])) for n in netlengths][:5])
        print("mean")
        lprint([str(mean(netlen_solvabilitydict[n]['mc'])) for n in netlengths][:5])
        print("bests")
        lprint([str(mean(netlen_solvabilitydict[n]['bc'])) for n in netlengths][:5])
        input()
        csv_data_walk[j+1].extend([str(mean(netlen_solvabilitydict[n]['f'])) for n in netlengths])
        csv_data_walk[j+2].extend([str(mean(netlen_solvabilitydict[n]['mc'])) for n in netlengths])
        csv_data_walk[j+3].extend([str(mean(netlen_solvabilitydict[n]['bc'])) for n in netlengths])
    with open("compare_solvability_best_of_"+str(best_of_N)+".csv", "w+") as inf:
        for i, netlength in enumerate(csv_data_walk[0]):
            line = ",".join([csv_data_walk[j][i] for j in range(dw_len)]) + "\n"
            inf.write(line)



chipsizes = [20, 30, 40, 50, 60, 70, 80, 90, 100]
make_solvability_comparison(chipsizes, 200)
# make_solvability_csvs(chipsizes, 1)
make_solvability_csvs(chipsizes, 200)
