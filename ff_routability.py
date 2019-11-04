# routable ratio
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp
import numpy as np

initial_col = '#1b9e77'
fit_col = '#1b9e77'
best_fit = '#d95f02'
best_col = '#d95f02'
mean_col = 'b'
worst_col = 'magenta'

def mean(list):
    return sum(list)/len(list)


def written_math(expression):
    def add_expr(func):
        func.math_expr = expression
        return func
    return add_expr


def named_function(name):
    def naming(func):
        func.name = name
        return func
    return naming


def format_chipsize(chipsize):
    return str(chipsize)+"x"+str(chipsize)+"x7"


def initial_solv_str(chipsize):
    return "routability by arb {}x{}".format(str(chipsize), str(chipsize))


def best_solv_str(chipsize):
    return "routability best of {}x{}".format(str(chipsize), str(chipsize))


def worst_solv_str(chipsize):
    return "routability worst of {}x{}".format(str(chipsize), str(chipsize))


def mean_solv_str(chipsize):
    return "routability of mean {}x{}".format(chipsize, chipsize)


@named_function("unbalanced sigmoid")
@written_math("e^{-e^{(2 \cdot (nl - shift) * slope)}}")
def expfunc(nl, const1, const2):
    return np.exp(-np.exp(2*(nl-const1)*const2))

@named_function("balanced sigmoid")
@written_math("1 - (1 /(1+ np.exp((nl-shift)*slope)))")
def regular_logistic(nl, shift, slope):
    return 1 - (1 /(1+ np.exp(-(nl-shift)*slope)))


def lstr(iterable):
    return [str(elem) for elem in iterable]


# 3x3 window location checker, used to assign labels/axis values
def determine_3x3_y(elem_n):
    return not elem_n % 3


def determine_3x3_solv(elem_n):
    return elem_n == 3


def determine_3x3_nl(elem_n):
    return elem_n == 7


def determine_3x3_x(elem_n):
    return elem_n // 6


def save_ab(chipsizes, param_func, N):
    fitfunc = eval(param_func)
    datafile = "compare_routability_best_of_"+str(N)+".csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    param_csv = open("params_" + param_func + ".csv", "w+")
    param_csv.write(", arbitrary netlists,,mean routability,,optimized netlist order,,worst order,\n")
    param_csv.write("chipsize (XxZ),initial_a,initial_b,mean_a,mean_b,best_a,best_b,worst_a,worst_b\n")
    for j, chipsize in enumerate(chipsizes):
        y_arb = df[initial_solv_str(chipsize)]
        popt, pcov = ABNLfit(nl, y_arb, fit_col, fitfunc, plot=False)

        y_mean = df[mean_solv_str(chipsize)]
        poptm, pcov = ABNLfit(nl, y_mean, fit_col, fitfunc,plot=False)

        y_best = df[best_solv_str(chipsize)]
        poptb, pcov = ABNLfit(nl, y_best, fit_col,fitfunc, plot=False)

        y_worst = df[worst_solv_str(chipsize)]
        poptw, pcov = ABNLfit(nl, y_worst, fit_col, fitfunc,plot=False)
        param_csv.write(",".join([str(chipsize)]+lstr(popt)+lstr(poptm)+lstr(poptb)+lstr(poptw))+"\n")


def conditional_label(boolean_value, label):
    if boolean_value:
        return label
    else:
        return None

def gen_filename_window(param_func, types, scatter, fitted):
    plot_savefile = param_func + "/" + "_".join(types)
    if scatter:
        plot_savefile += "_s"
    if fitted:
        plot_savefile += "_f"
    plot_savefile += "_3x3.png"
    return plot_savefile

def plot_shift_slope(chipsizes, types, param_func, title, scatter=True, fitted=True, legend=False):
    fitfunc = eval(param_func)
    plot_savefile = gen_filename_window(param_func, types, scatter, fitted)

    datafile = "compare_routability_best_of_"+str(N)+".csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())
    ab_df = load_ab(param_func)

    params_r = []
    params_b = []
    params_m = []
    params_w = []

    _best = "best" in types
    _mean = "mean" in types
    _arb = "initial" in types
    _worst = "worst" in types
    fig=plt.figure(figsize=(12,7))
    fig.suptitle('effect of permutation on expected solavability') # or plt.suptitle('Main title')
    legend_loc = 7

    for j, cs in enumerate(chipsizes):
        print((1+j+j//4)//4, (j+j//3)%4)
        ax = plt.subplot2grid((3,3), (j//3, j%3))
        ax.set_title(format_chipsize(cs))
        if not determine_3x3_x(j):
            ax.set_xticks([])
        else:
            ax.set_xticks([10,20,30,40,50,60,70,80,90])
        if determine_3x3_nl(j):
            ax.set_xlabel("netlist length")
        if not determine_3x3_y(j):
            ax.set_yticks([])
        if determine_3x3_solv(j):
            ax.set_ylabel("Ratio of routable netlists")

        labelwindow = j==legend_loc
        if labelwindow and legend and fitted and scatter:
            print("adding empty labels")
            if scatter:
                plt.scatter([],[],c="#000000", s=6, label="discovered")
            if fitted:
                plt.plot([],[],c="#000000", linestyle="--", label="model expectation")

        if _arb:
            y_arb = df[initial_solv_str(cs)]
            popta = ab_df["initial_a"][j], ab_df["initial_b"][j]
            if scatter:
                plt.scatter(nl, y_arb, c=initial_col, s=6, label=conditional_label(labelwindow, "initial sequence"))
            if fitted:
                ABNL_plot(nl, popta, fitfunc, c=fit_col, label=conditional_label(labelwindow, "expected routability of initial sequence"))

        if _mean:
            y_mean = df[mean_solv_str(cs)]
            poptm = ab_df["mean_a"][j], ab_df["mean_b"][j]
            if scatter:
                plt.scatter(nl, y_mean, c=mean_col, s=6, label=conditional_label(labelwindow, "average sequence routability"))
            if fitted:
                ABNL_plot(nl, poptm, fitfunc, c=fit_col, label=conditional_label(labelwindow, "expected average sequence routability"))

        if _best:
            y_best = df[best_solv_str(cs)]
            poptb = ab_df["best_a"][j], ab_df["best_b"][j]
            if scatter:
                plt.scatter(nl, y_best, c=best_col, s=6, label=conditional_label(labelwindow, "after permutation"))
            if fitted:
                ABNL_plot(nl, poptb, fitfunc, c=best_fit, label=conditional_label(labelwindow, "expected routability after permutation"))


        if labelwindow and legend:
            # Put a legend to the right of the current axis
            lgd = plt.legend(bbox_to_anchor=(1, 1.0))
            plt.legend(loc='upper center',
             bbox_to_anchor=(0.5, -0.25),fancybox=False, shadow=False, ncol=3)
    plt.suptitle(title)
    if legend:
        # tight bounding box
        plt.savefig(plot_savefile, bbox_extra_artists=(lgd,))
    else:
        plt.savefig(plot_savefile)
    #plt.show()


def plot_fits(types_of_fit, suptitle, param_func, N, cs="all", scatter=False):
    plt.clf()
    fitfunc = eval(param_func)
    bbox_tuple = (1.05, -0.3, 1.0, 1.0)
    ab_df = load_ab(param_func)
    datafile = "compare_routability_best_of_"+str(N)+".csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    type_count = len(types_of_fit)
    if cs == 'all':
        sizes = ab_df.index.values.tolist()
        print(sizes)
    else:
        sizes = [cs//10-2]
    plt.figure(figsize=(7,7))
    for i, t in enumerate(types_of_fit):
        fitted_vals = [fitfunc(nl, ab_df[t+'_a'][size], ab_df[t+'_b'][size]) for size in sizes]

        p = plt.subplot(type_count+2,1,i+1)
        plt.setp(p.get_xticklabels(), visible=False)
        for j, fitted_m in enumerate(fitted_vals):
            p.plot(nl, fitted_m, label=t+" fit")
            if scatter:
                y = df[eval(t+"_solv_str")((2+sizes[j])*10)]
                p.scatter(nl, y, c=initial_col, label=t, alpha=0.4)
        box = p.get_position()
        p.set_position([box.x0, box.y0 + box.height*0.1,
                     box.width * 0.7, box.height])

        p.set_title(types_of_fit[i])
        p.set_ylabel("routable ratio")
        p.legend(loc="upper left", bbox_to_anchor=bbox_tuple)

    # plt.suptitle("expected routability for differently sized meshs\n\n")
    p = plt.subplot(type_count+2,1,3)
    fitted_vals = [1 - (1 - fitfunc(nl, ab_df['mean'+'_a'][size], ab_df['mean'+'_b'][size]))**N for size in sizes]

    for j, fitted_m in enumerate(fitted_vals):
        p.plot(nl, fitted_m, label="expected best based on average fit over "+str(N))
        if scatter:
            y = df[eval(t+"_solv_str")((2+sizes[j])*10)]
            p.scatter(nl, y, c=initial_col, label=t, alpha=0.4)
    box = p.get_position()
    p.set_position([box.x0, box.y0 + box.height*0.1,
                 box.width * 0.7, box.height])

    p.set_title(types_of_fit[i])
    p.set_ylabel("routable ratio")
    p.legend(loc="upper left", bbox_to_anchor=bbox_tuple)

    p = plt.subplot(type_count+2,1,4)
    fitted_vals_m = [fitfunc(nl, ab_df['mean'+'_a'][size], ab_df['mean'+'_b'][size]) for size in sizes]
    fitted_vals_b = [fitfunc(nl, ab_df['best'+'_a'][size], ab_df['best'+'_b'][size]) for size in sizes]

    for j in range(len(fitted_vals_m)):
        p.plot(nl, (fitted_vals_b[j] - fitted_vals_m[j]), label="improvement by permutation")
    box = p.get_position()
    p.set_position([box.x0, box.y0 + box.height*0.1,
                 box.width * 0.7, box.height])

    p.set_title(types_of_fit[i])
    p.set_ylabel("routable ratio")
    p.legend(loc="upper left", bbox_to_anchor=bbox_tuple)

    plt.suptitle(suptitle)
    p.set_xlabel("netlist length")

    plt.savefig("expected_chipsize_compare.png")
    plt.show()



def compare_expected_best(param_func, N):
    fitfunc = eval(param_func)
    ["best", "mean"]
    ab_df = load_ab(param_func)
    datafile = "compare_routability_best_of_" + str(N) + ".csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    sizes = ab_df.index.values.tolist()
    plt.figure(figsize=(7,7))
    fitted_vals = [fitfunc(nl, ab_df["mean"+'_a'][size], ab_df["mean"+'_b'][size]) for size in sizes]
    fitted_bests = [fitfunc(nl, ab_df["best"+'_a'][size], ab_df["best"+'_b'][size]) for size in sizes]
    expected_bests = [[1.0 - (1.0-val)**N for val in fv] for fv in fitted_vals]
    p = plt.subplot(3,1,1)
    plt.setp(p.get_xticklabels(), visible=True)
    for fitted_m in fitted_vals:
        p.plot(nl, fitted_m, c='b')
    p = plt.subplot(3,1,2)
    plt.setp(p.get_xticklabels(), visible=True)
    for fitted_be in expected_bests:
        p.plot(nl, fitted_be, c='r')
    p = plt.subplot(3,1,3)
    plt.setp(p.get_xticklabels(), visible=True)
    for fitted_b in fitted_bests:
        p.plot(nl, fitted_b, c='g')

    p.set_title("mean")
    p.set_ylabel("routable ratio")
    p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))

    # plt.suptitle("expected routability for differently sized meshs\n\n")
    plt.suptitle("compare expected best vs discovered best")
    p.set_xlabel("netlist length")

    plt.savefig("expected_chipsize_compare.png")
    #plt.show()


def plot_fits_dif(t1, t2, param_func, suptitle, N):
    fitfunc = eval(param_func)
    ab_df = load_ab(param_func)
    datafile = "compare_routability_best_of_"+str(N)+".csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = df.index.values.tolist()

    sizes = ab_df.index.values.tolist()
    plt.figure(figsize=(7,7))
    fit_difs = [fitfunc(nl, ab_df[t1+'_a'][size], ab_df[t1+'_b'][size]) - expfunc(nl, ab_df[t2+'_a'][size], ab_df[t2+'_b'][size]) for size in sizes]
    fit_difs_maxs = [np.argmax(fd) for fd in fit_difs]
    for dif in fit_difs:
        p = plt.subplot(111)
        plt.setp(p.get_xticklabels(), visible=False)
        p.plot(nl, dif)

        p.set_ylabel("routable ratio")
        p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))
    p.scatter([nl[v] for v in fit_difs_maxs], [fit_difs[i][v] for i, v in enumerate(fit_difs_maxs)])
    p.scatter([nl[v] for v in fit_difs_maxs], [1 for i, v in enumerate(fit_difs_maxs)])

    # plt.suptitle("expected routability for differently sized meshs\n\n")
    plt.suptitle(suptitle)
    p.set_xlabel("netlist length")

    plt.savefig("expected_chipsize_compare.png")
    #plt.show()


def scatter_routability(types_of_scatter, cs, N, end=False, title=None, savename=None, print_options=True):
    if end and not savename:
        raise ValueError(cannot)
    _, df, _, nl = get_plot_necessities(param_func)
    plt.figure(figsize=(7,7))
    type_count = len(types_of_scatter) + 1
    for i, t in enumerate(types_of_scatter):

        p = plt.subplot(type_count,1,i + 1)
        y_arb = df[eval(t+"_solv_str(cs)")]
        p.scatter(nl, y_arb, c=initial_col, label=t, alpha=0.4)
        p.set_ylabel("routable ratio")
        p.set_title("expected routability based on " + t)
        p.legend()

    p = plt.subplot(type_count,1, 3)
    y_arb = 1 - (1 - df[eval("mean_solv_str(cs)")])**N
    p.scatter(nl, y_arb, c=initial_col, label="average based expected best", alpha=0.4)
    p.set_title("expected best based on average")
    p.set_ylabel("routable ratio")
    plt.subplots_adjust(hspace=0.8)

    if end:
        if title:
            plt.title("routability of netlists on a "+format_chipsize(cs) + " mesh")
        p.set_xlabel("netlist length")
        plt.legend()
        plt.savefig(savename)
        # plt.savefig("initial_sequence_"+format_chipsize(cs)+".png")
        #plt.show()


def example_legend_gen(t):
    """ returns title for example plot

    :param t: type of data - mean, arb, worst, best
    return: string
    """
    if t == "initial":
        return "initial sequence"
    elif t == "best":
        return "after permutation"

def example_scatter(param_func, types_of_scatter, cs, legend=True):
    _, df, _, nl = get_plot_necessities(param_func)
    plt.figure(figsize=(7,7))
    type_count = len(types_of_scatter) + 1
    for i, t in enumerate(types_of_scatter):

        y_arb = df[eval(t+"_solv_str(cs)")]
        plt.scatter(nl, 100*y_arb, c=eval(t+"_col"), label=example_legend_gen(t), alpha=0.4)
    plt.ylabel("Ratio of routable netlists")
    plt.xlabel("netlist length")
    # plt.title("routability for increasing netlist length")
    if legend:
        plt.legend()
    save_spec = "both" if (len(types_of_scatter)-1) else types_of_scatter[0]
    savestring = param_func + "/" + "ex_scatter_" + save_spec
    plt.savefig(savestring)



def fit_ab(mesh_metric, param_func, _arb=False, _mean=False, _best=False, _worst=False):
    print("\n" + mesh_metric)

    plt.clf()
    df = load_ab(param_func)
    if mesh_metric == "area":
        mesh_metric_col = getdfcol(df,0)**2
    elif mesh_metric == "edge size":
        mesh_metric_col = getdfcol(df,0)
    elif mesh_metric == "volume":
        mesh_metric_col = getdfcol(df,0)**2 * 8
    interp_mesh = np.array([i for i in range(min(mesh_metric_col), max(mesh_metric_col), 1)])
    initial_shift = getdfcol(df,1)
    best_shift = getdfcol(df,5)
    mean_shift = getdfcol(df,3)
    worst_shift = getdfcol(df,7)

    plt.scatter([],[],c="#000000", s=6, label="discovered")
    plt.plot([],[],c="#000000", linestyle='--', label="model expectation")
    print("shift")
    # for t in types:
    #
    if _arb:
        plt.scatter(mesh_metric_col, initial_shift, c=initial_col, label='discovered shift arbitrary')
        aapopt, pcov = curve_fit(logfunc, mesh_metric_col, initial_shift, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(mesh_metric_col)], [40000, 0.9, max(mesh_metric_col)]))
        print("shift arbitrary", aapopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *aapopt), c=fit_col, label="expected \n standard sequence", linestyle="--")

    if _best:
        plt.scatter(mesh_metric_col, best_shift, c=best_col, label='discovered shift best')
        abpopt, pcov = curve_fit(logfunc, mesh_metric_col, best_shift, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(mesh_metric_col)], [40000, 0.9, max(mesh_metric_col)]))
        print("shift best", abpopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *abpopt), c=best_fit, label="expected \n best sequence", linestyle="--")

    if _mean:
        plt.scatter(mesh_metric_col, mean_shift, c=mean_col, label='discovered shift mean')
        ampopt, pcov = curve_fit(logfunc, mesh_metric_col, mean_shift, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(mesh_metric_col)], [40000, 0.9, max(mesh_metric_col)]))
        print("shift mean", ampopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *ampopt), c=fit_col, label="expected \n average sequence", linestyle='--')

    if _worst:
        plt.scatter(mesh_metric_col, worst_shift, c=worst_col, label='discovered shift worst')
        awpopt, pcov = curve_fit(logfunc, mesh_metric_col, worst_shift, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(mesh_metric_col)], [40000, 0.9, max(mesh_metric_col)]))
        print("shift worst", awpopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *awpopt), c=fit_col, label="expected \n worst sequence", linestyle='--')

    plt.xlabel("chipsize")
    plt.title("shift of routability function w.r.t chipsize " + mesh_metric)
    plt.legend()
    print(param_func + "/shift_param_chipsize_" + mesh_metric + ".png")
    plt.savefig(param_func + "/shift_param_chipsize_" + mesh_metric + ".png")
    plt.show()
    plt.clf()

    initial_slope = getdfcol(df,2)
    best_slope = getdfcol(df,6)
    mean_slope = getdfcol(df,4)
    worst_slope = getdfcol(df,8)
    plt.scatter([],[],c="#000000", s=6, label="discovered")
    plt.plot([],[],c="#000000", linestyle="--", label="model expectation")
    print("slope")
    if _arb:
        plt.scatter(mesh_metric_col, initial_slope, c=initial_col, label='discovered standard sequence')
        bapopt, pcov = curve_fit(logfunc, mesh_metric_col, initial_slope, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 0.5, 10000]))
        print("slope arbitrary", bapopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *bapopt), c=fit_col, label="expected slope arbitrary", linestyle="--")

    if _best:
        plt.scatter(mesh_metric_col, best_slope, c=best_col, label='discovered best sequence')
        bbpopt, pcov = curve_fit(logfunc, mesh_metric_col, best_slope, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 0.5, 10000]))
        print("slope best", bbpopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *bbpopt), c=best_fit,  label="expected slope best", linestyle="--")

    if _mean:
        plt.scatter(mesh_metric_col, mean_slope, c=mean_col, label='discovered mean sequence')
        bapopt, pcov = curve_fit(logfunc, mesh_metric_col, mean_slope, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("slope arbitrary", bapopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *bapopt), c=fit_col, label="expected slope arbitrary", linestyle="--")

    if _worst:
        plt.scatter(mesh_metric_col, worst_slope, c=worst_col, label='discovered worst sequence')
        bbpopt, pcov = curve_fit(logfunc, mesh_metric_col, worst_slope, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("slope best", bbpopt)
        plt.plot(interp_mesh, logfunc(interp_mesh, *bbpopt), c=fit_col,  label="expected slope worst", linestyle='--')

    plt.xlabel("chipsize")
    plt.title("slope of routability function w.r.t " + mesh_metric)

    plt.legend()
    plt.savefig(param_func +"/slope_param_chipsize_" + mesh_metric + ".png")
    plt.show()




def ABNLfit(nl, routability, c, fitfunc, label=None, plot=False):
    popt, pcov = curve_fit(fitfunc, nl, routability, p0=(40, 0.05), bounds=([-100, -2], [200, 1]))
    popt, pcov = curve_fit(fitfunc, nl, routability, p0=(0.05, 0.05), bounds=([-1e6, 1e-2], [1e6, 1]))
    return popt, pcov

def ABNL_plot(nl, popts, fitfunc, c, label=None):
    if label:
        plt.plot(nl, fitfunc(nl, *popts), c=c, linestyle='--', label=label)
    else:
        plt.plot(nl, fitfunc(nl, *popts), c=c, linestyle='--')



def skiplist(iterable, start, interval=1):
    for i, elem in enumerate(iterable):
        if start <= i:
            if not (i-start) % (interval+1):
                yield elem


def lprint(iterable):
    for elem in iterable:
        print(elem)

def load_ab(param_func, _print=False):
    df = pd.read_csv("params_" + param_func + ".csv", header=1)
    if _print:
        print(df.head())
        lprint(df.columns)
    return df

@written_math("-const1 * natural log( const2 * (value - const3))")
def logfunc(value, const1, const2, const3):
    # return  -const1*np.log(const2* (value-const3))
    return  -const1 * np.log( const2 * (value - const3))



def getdfcol(df, n):
    """ get n'th column values of dataframe
    """
    return df[df.columns[n]]


def plot_residuals(types_of_scatter, mesh_size, param_func, end=False, title=None, print_options=True):

    fitfunc, df, ab_df, nl = get_plot_necessities(param_func)
    type_count = len(types_of_scatter) + 1
    cor_cs = int(mesh_size/10-2)
    fig, axs = plt.subplots(2, 3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for i, t in enumerate(types_of_scatter):
        fitted_val = np.array(fitfunc(nl, ab_df[t+'_a'][cor_cs], ab_df[t+'_b'][cor_cs]))
        y = df[eval(t+"_solv_str")(mesh_size)]

        axs[i,0].scatter(nl, y, c=eval(t+"_col"), label=t, alpha=0.4)
        axs[i,0].plot(nl, fitted_val, c=eval(t+"_col"), label=t, alpha=0.4)
        axs[i,0].set_ylabel("routable proportion")
        axs[i,0].set_title("routability for " + str(t) + " on size " + str(mesh_size))
        axs[i,0].legend()

        residual = (fitted_val - y)
        axs[i,1].scatter(nl, residual, label=t, alpha=0.4, c=eval(t+"_col"))
        axs[i,1].set_ylabel("% deviation from fit")
        axs[i,1].set_title("residual when fitting for " + str(t) + " on size " + str(mesh_size))
        axs[i,1].legend()

        rel_residual = [elem for elem in residual if abs(elem) > 0.01]
        rel_count = len(rel_residual)
        print(rel_count)
        axs[i,2].hist(rel_residual, label=t, color=eval(t+"_col"), bins=max([10, int(rel_count/3)]))
        axs[i,2].set_ylabel("occurances")
        axs[i,2].set_title("distribution of " + str(rel_count) + "relevant resduals for " + str(t) + " on size " + str(mesh_size) + \
            "total residual " + str(sum([abs(e) for e in residual])) + \
              "\naverage deviation" + str(round(sum([abs(e) for e in residual])/len(residual),2)) + \
              "\nvariance " + str(sum([(e - mean(residual))**2 for e in residual])/len(residual)) + \
              "\nlikelyhood " + str(log_likelyhood_model(param_func, t, cs)))
        axs[i,2].legend()
    fig.suptitle("analysed fit for " + fitfunc.name + ": " + fitfunc.math_expr + "\n")
    plt.savefig(param_func + "/" + param_func + " residual_comparison_"+str(cs)+".png")


def get_plot_necessities(param_func, N):
    fitfunc = eval(param_func)
    df = pd.read_csv("compare_routability_best_of_"+str(N)+".csv")
    ab_df = load_ab(param_func)
    nl = [10 + e for e in df.index.values.tolist()]
    return fitfunc, df, ab_df, nl


def log_likelyhood_model(param_func, modeltype, mesh_size):
    """ calculates the likelyhood of a model given the basic function type for a

    given parametrization function, mesh size and modeltype.

    :param param_func: string:
        expfunc - unbalanced sigmoid
        regular_logistic - balanced sigmoid
    :param modeltype: string:
        best
        mean
        worst
        arb - arbitrary

    :returns: float, likelyhood of model
    """
    fitfunc, df, ab_df, nl = get_plot_necessities(param_func)
    cor_mesh_size = int(mesh_size/10-2)
    fitted_val = np.array(fitfunc(nl, ab_df[modeltype+'_a'][cor_mesh_size], ab_df[modeltype+'_b'][cor_mesh_size]))
    data = df[eval(modeltype+"_solv_str(cs)")]
    likelyhood_for_n = fitted_val**data*(1-fitted_val)**(1-data)
    tot_log_likelyhood = np.sum(np.log(likelyhood_for_n))
    return tot_log_likelyhood




if __name__ == '__main__':
    N = 200
    param_func = "regular_logistic"
    save_ab([(i+2)*10 for i in range(9)], param_func, N)
    # param_func = "expfunc"
    # save_ab([(i+2)*10 for i in range(9)], param_func, 10)
    _arb = True
    _mean = False
    _best = True
    _worst = False
    fitfunc_names = ["regular_logistic"]
    cs = 80
    chipsizes = [(i+2)*10 for i in range(9)]

    for param_func in fitfunc_names:
        # example_scatter(param_func, ["initial", "best"], 60, legend=True)
        # example_scatter(param_func, ["initial"], 60, legend=False)
        # example_scatter(param_func, ["best"], 60, legend=False)
        # scatter_routability(["initial", "best"], cs, end=True, savename="k.png")
        # plot_fits(["initial", "best"],"" ,param_func, 200, cs=cs, scatter=True)
        # plot_fits_dif("best", "initial", param_func, "difference best and mean")
        fit_ab("area", param_func, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
        # fit_ab("edge size", param_func, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
        # fit_ab("volume", param_func, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
        # compare_expected_best(param_func, N)
        # plot_shift_slope(chipsizes, ["initial"], param_func, "")
        # plot_shift_slope(chipsizes, ["initial"], param_func, "", fitted=False)
        # plot_shift_slope(chipsizes, ["initial"], param_func, "", scatter=False)
        # plot_shift_slope(chipsizes, ["initial", "best"], param_func, "", scatter=False)
        # plot_shift_slope(chipsizes, ["initial", "best"], param_func, "", fitted=False, legend=True)
        # plot_shift_slope(chipsizes, ["best"], param_func, "")
        # plot_shift_slope(chipsizes, ["best"], param_func, "", fitted=False)
        # plot_shift_slope(chipsizes, ["best"], param_func, "", scatter=False)
        # for cs in chipsizes:
        #     plot_residuals(["mean", "best"], cs, param_func, end=True)
