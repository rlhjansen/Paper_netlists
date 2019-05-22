
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp
import numpy as np

arb_col = 'g'
fit_col = 'g'
best_fit = 'r'
best_col = 'orange'
mean_col = 'b'
worst_col = 'magenta'

def mean(list):
    return sum(list)/len(list)

def plot_percent_wrapper(x,y, plot_on=plt, **kwargs):
    plot_on.plot(x, y*100, **kwargs)

def scatter_percent_wrapper(x, y, plot_on=plt, **kwargs):
    plot_on.scatter(x, y*100, **kwargs)


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

def format_meshsize(meshsize):
    return str(meshsize)+"x"+str(meshsize)

def arb_solv_str(meshsize):
    return "routability by arb {}x{}".format(str(meshsize), str(meshsize))

def best_solv_str(meshsize):
    return "routability best of {}x{}".format(str(meshsize), str(meshsize))

def worst_solv_str(meshsize):
    return "routability worst of {}x{}".format(str(meshsize), str(meshsize))

def mean_solv_str(meshsize):
    return "routability of mean {}x{}".format(meshsize, meshsize)

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

def determine_3x3_y(elem_n):
    return not elem_n % 3

def determine_3x3_solv(elem_n):
    return elem_n == 3

def determine_3x3_nl(elem_n):
    return elem_n == 7

def determine_3x3_x(elem_n):
    return elem_n // 6



def save_ab(meshsizes, param_func):
    fitfunc = eval(param_func)
    datafile = "compare_routability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    param_csv = open("params_" + param_func + ".csv", "w+")
    param_csv.write(", arbitrary netlists,,mean routability,,optimized netlist order,,worst order,\n")
    param_csv.write("meshsize (XxZ),arb_a,arb_b,mean_a,mean_b,best_a,best_b,worst_a,worst_b\n")
    for j, meshsize in enumerate(meshsizes):

        y_arb = df[arb_solv_str(meshsize)]
        popt, pcov = ABNLfit(nl, y_arb, fit_col, fitfunc, plot=False)

        y_mean = df[mean_solv_str(meshsize)]
        poptm, pcov = ABNLfit(nl, y_mean, fit_col, fitfunc,plot=False)

        y_best = df[best_solv_str(meshsize)]
        poptb, pcov = ABNLfit(nl, y_best, fit_col,fitfunc, plot=False)

        y_worst = df[worst_solv_str(meshsize)]
        poptw, pcov = ABNLfit(nl, y_worst, fit_col, fitfunc,plot=False)
        param_csv.write(",".join([str(meshsize)]+lstr(popt)+lstr(poptm)+lstr(poptb)+lstr(poptw))+"\n")


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

def plot_alpha_beta(meshsizes, types, param_func, title, scatter=True, fitted=True):
    fitfunc = eval(param_func)
    plot_savefile = gen_filename_window(param_func, types, scatter, fitted)

    datafile = "compare_routability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())
    ab_df = load_ab(param_func)

    params_r = []
    params_b = []
    params_m = []
    params_w = []

    _best = "best" in types
    _mean = "mean" in types
    _arb = "arb" in types
    _worst = "worst" in types
    fig=plt.figure(figsize=(12,7))
    fig.suptitle('effect of permutation on predicted solavability') # or plt.suptitle('Main title')
    legend_loc = 5
    for j, cs in enumerate(meshsizes):
        print((1+j+j//4)//4, (j+j//3)%4)
        ax = plt.subplot2grid((3, 4), ((1+j+j//4)//4, (j+j//3)%4))
        ax.set_title(format_meshsize(cs))
        y_arb = df[arb_solv_str(cs)]
        if not determine_3x3_x(j):
            ax.set_xticks([])
        else:
            ax.set_xticks([10,50,90])
        if determine_3x3_nl(j):
            ax.set_xlabel("netlist length")
        if not determine_3x3_y(j):
            ax.set_yticks([])
        if determine_3x3_solv(j):
            ax.set_ylabel("routability %")

        labelwindow = j==legend_loc

        if _arb:
            y_arb = df[arb_solv_str(cs)]
            popta = ab_df["arb_a"][j], ab_df["arb_b"][j]
            if scatter:
                plotscatter(nl, y_arb, c=arb_col, s=6, label=conditional_label(labelwindow, "routability of arbitrary sequence"))
            if fitted:
                ABNL_plot(nl, popta, fitfunc, c=fit_col, label=conditional_label(labelwindow, "predicted routability of arbitrary sequence"))

        if _mean:
            y_mean = df[mean_solv_str(cs)]
            poptm = ab_df["mean_a"][j], ab_df["mean_b"][j]
            if scatter:
                plotscatter(nl, y_mean, c=mean_col, s=6, label=conditional_label(labelwindow, "average sequence routability"))
            if fitted:
                ABNL_plot(nl, poptm, fitfunc, c=fit_col, label=conditional_label(labelwindow, "predicted average sequence routability"))

        if _best:
            y_best = df[best_solv_str(cs)]
            poptb = ab_df["best_a"][j], ab_df["best_b"][j]
            if scatter:
                plotscatter(nl, y_best, c=best_col, s=6, label=conditional_label(labelwindow, "routability after permutation"))
            if fitted:
                ABNL_plot(nl, poptb, fitfunc, c=best_fit, label=conditional_label(labelwindow, "predicted routability after permutation"))


        if labelwindow:
            # Put a legend to the right of the current axis
            lgd = plt.legend(bbox_to_anchor=(1, 1.0))
    # plt.suptitle("routability for different meshsizes")
    plt.suptitle(title)
    plt.savefig(plot_savefile, bbox_extra_artists=(lgd,))
    plt.show()


def plot_fits(types_of_fit, suptitle, param_func, cs="all", scatter=False):
    fitfunc = eval(param_func)
    bbox_tuple = (1.05, -0.3, 1.0, 1.0)
    ab_df = load_ab(param_func)
    datafile = "compare_routability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    type_count = len(types_of_fit)
    if cs == 'all':
        sizes = ab_df.index.values.tolist()
        print(sizes)
    else:
        sizes = [cs//10-2]
    plt.figure(figsize=(7,7))
    # print(nl)
    # input()
    for i, t in enumerate(types_of_fit):
        fitted_vals = [fitfunc(nl, ab_df[t+'_a'][size], ab_df[t+'_b'][size]) for size in sizes]

        p = plt.subplot(type_count+2,1,i+1)
        plt.setp(p.get_xticklabels(), visible=False)
        for j, fitted_m in enumerate(fitted_vals):
            p.plot(nl, fitted_m*100, label=t+" fit")
            if scatter:
                y = df[eval(t+"_solv_str")((2+sizes[j])*10)]
                p.scatter(nl, y*100, c=arb_col, label=t, alpha=0.4)
        box = p.get_position()
        p.set_position([box.x0, box.y0 + box.height*0.1,
                     box.width * 0.7, box.height])

        p.set_title(types_of_fit[i])
        p.set_ylabel("routability %")
        p.legend(loc="upper left", bbox_to_anchor=bbox_tuple)

    # plt.suptitle("predicted routability for differently sized chips\n\n")
    p = plt.subplot(type_count+2,1,3)
    fitted_vals = [1 - (1 - fitfunc(nl, ab_df['mean'+'_a'][size], ab_df['mean'+'_b'][size]))**200 for size in sizes]

    for j, fitted_m in enumerate(fitted_vals):
        p.plot(nl, fitted_m*100, label="expected best based on average fit over 200")
        if scatter:
            y = df[eval(t+"_solv_str")((2+sizes[j])*10)]
            p.scatter(nl, y, c=arb_col, label=t, alpha=0.4)
    box = p.get_position()
    p.set_position([box.x0, box.y0 + box.height*0.1,
                 box.width * 0.7, box.height])

    p.set_title(types_of_fit[i])
    p.set_ylabel("routability %")
    p.legend(loc="upper left", bbox_to_anchor=bbox_tuple)

    p = plt.subplot(type_count+2,1,4)
    fitted_vals_m = [fitfunc(nl, ab_df['mean'+'_a'][size], ab_df['mean'+'_b'][size]) for size in sizes]
    fitted_vals_b = [fitfunc(nl, ab_df['best'+'_a'][size], ab_df['best'+'_b'][size]) for size in sizes]

    for j in range(len(fitted_vals_m)):
        p.plot(nl, (fitted_vals_b[j] - fitted_vals_m[j])*100, label="improvement by permutation")
    box = p.get_position()
    p.set_position([box.x0, box.y0 + box.height*0.1,
                 box.width * 0.7, box.height])

    p.set_title(types_of_fit[i])
    p.set_ylabel("routability %")
    p.legend(loc="upper left", bbox_to_anchor=bbox_tuple)

    plt.suptitle(suptitle)
    p.set_xlabel("netlist length")

    plt.savefig("predicted_meshsize_compare.png")
    plt.show()



def compare_expected_best(param_func):
    fitfunc = eval(param_func)
    ["best", "mean"]
    ab_df = load_ab(param_func)
    datafile = "compare_routability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    sizes = ab_df.index.values.tolist()
    plt.figure(figsize=(7,7))
    fitted_vals = [fitfunc(nl, ab_df["mean"+'_a'][size], ab_df["mean"+'_b'][size]) for size in sizes]
    fitted_bests = [fitfunc(nl, ab_df["best"+'_a'][size], ab_df["best"+'_b'][size]) for size in sizes]
    expected_bests = [[1.0 - (1.0-val)**200 for val in fv] for fv in fitted_vals]
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
    p.set_ylabel("routability %")
    p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))

    # plt.suptitle("predicted routability for differently sized chips\n\n")
    plt.suptitle("compare expected best vs real best")
    p.set_xlabel("netlist length")

    plt.savefig("predicted_meshsize_compare.png")
    plt.show()


def plot_fits_dif(t1, t2, param_func, suptitle):
    fitfunc = eval(param_func)
    ab_df = load_ab(param_func)
    datafile = "compare_routability_best_of_200.csv"
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

        p.set_ylabel("routability %")
        p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))
    p.scatter([nl[v] for v in fit_difs_maxs], [fit_difs[i][v] for i, v in enumerate(fit_difs_maxs)])
    p.scatter([nl[v] for v in fit_difs_maxs], [1 for i, v in enumerate(fit_difs_maxs)])

    # plt.suptitle("predicted routability for differently sized chips\n\n")
    plt.suptitle(suptitle)
    p.set_xlabel("netlist length")

    plt.savefig("predicted_meshsize_compare.png")
    plt.show()


def scatter_routability(types_of_scatter, cs, end=False, title=None, savename=None, print_options=True):
    if end and not savename:
        raise ValueError(cannot)
    _, df, _, nl = get_plot_necessities(param_func)
    plt.figure(figsize=(7,7))
    type_count = len(types_of_scatter) + 1
    for i, t in enumerate(types_of_scatter):

        p = plt.subplot(type_count,1,i + 1)
        y_arb = df[eval(t+"_solv_str")(cs)]
        p.scatter(nl, y_arb*100, c=arb_col, label=t, alpha=0.4)
        p.set_ylabel("routable proportion")
        p.set_title("expected routability based on " + t)
        p.legend()

    p = plt.subplot(type_count,1, 3)
    y_arb = 1 - (1 - df[eval("mean_solv_str")(cs)])**200
    p.scatter(nl, y_arb*100, c=arb_col, label="average based expected best", alpha=0.4)
    p.set_title("expected best based on average")
    p.set_ylabel("routability %")
    plt.subplots_adjust(hspace=0.8)

    if end:
        if title:
            plt.title("routability of arbitrary netlists on a "+format_meshsize(cs) + " chip")
        p.set_xlabel("netlist length")
        plt.legend()
        plt.savefig(savename)
        # plt.savefig("arbitrary_sequence_"+format_meshsize(cs)+".png")
        plt.show()


def fit_ab(meshsizes, param_func, _arb=False, _mean=False, _best=False, _worst=False):
    df = load_ab(param_func)
    if meshsizes == "area":
        meshsize_col = getdfcol(df,0)**2
    elif meshsizes == "edge_size":
        meshsize_col = getdfcol(df,0)
    interp_chip = [i for i in range(min(meshsize_col), max(meshsize_col), 1)]
    arbitrary_alpha = getdfcol(df,1)
    best_alpha = getdfcol(df,5)
    mean_alpha = getdfcol(df,3)
    worst_alpha = getdfcol(df,7)
    p = plt.subplot(111)

    if _arb:
        plt.scatter(meshsize_col, arbitrary_alpha, c=arb_col, label='real alpha arbitrary')
        aapopt, pcov = curve_fit(logfunc, meshsize_col, arbitrary_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(meshsize_col)], [40000, 0.9, max(meshsize_col)]))
        print("alpha arbitrary", aapopt)
        p.plot(interp_chip, logfunc(interp_chip, *aapopt), c=fit_col, label="predicted \n arbitrary sequence", linestyle="--")

    if _best:
        plt.scatter(meshsize_col, best_alpha, c=best_col, label='real alpha best')
        abpopt, pcov = curve_fit(logfunc, meshsize_col, best_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(meshsize_col)], [40000, 0.9, max(meshsize_col)]))
        print("alpha best", abpopt)
        p.plot(interp_chip, logfunc(interp_chip, *abpopt), c=best_fit, label="predicted \n best sequence", linestyle="--")

    if _mean:
        plt.scatter(meshsize_col, mean_alpha, c=mean_col, label='real alpha mean')
        ampopt, pcov = curve_fit(logfunc, meshsize_col, mean_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(meshsize_col)], [40000, 0.9, max(meshsize_col)]))
        print("alpha mean", ampopt)
        p.plot(interp_chip, logfunc(interp_chip, *ampopt), c=fit_col, label="predicted \n average sequence", linestyle='--')

    if _worst:
        pl.tscatter(meshsize_col, worst_alpha, c=worst_col, label='real alpha worst')
        awpopt, pcov = curve_fit(logfunc, meshsize_col, worst_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(meshsize_col)], [40000, 0.9, max(meshsize_col)]))
        print("alpha worst", awpopt)
        p.plot(interp_chip, logfunc(interp_chip, *awpopt), c=fit_col, label="predicted \n worst sequence", linestyle='--')

    p.set_xlabel("meshsize")
    plt.title("parametrization of alpha for average and best sequence routability by "+" ".join(meshsizes.split("_")))
    plt.legend()
    plt.savefig("alpha_param_meshsize_"+meshsizes+".png")
    plt.show()
    plt.clf()

    arbitrary_beta = getdfcol(df,2)
    best_beta = getdfcol(df,6)
    mean_beta = getdfcol(df,4)
    worst_beta = getdfcol(df,8)
    p = plt.subplot(111)
    if _arb:
        plt.scatter(meshsize_col, arbitrary_beta, c=arb_col, label='real arbitrary sequence')
        bapopt, pcov = curve_fit(logfunc, meshsize_col, arbitrary_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -40000], [300, 0.5, 10000]))
        print("beta arbitrary", bapopt)
        p.plot(interp_chip, logfunc(interp_chip, *bapopt), c=fit_col, label="predicted beta arbitrary", linestyle="--")

    if _best:
        plt.scatter(meshsize_col, best_beta, c=best_col, label='real best sequence')
        bbpopt, pcov = curve_fit(logfunc, meshsize_col, best_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 0.5, 10000]))
        print("beta best", bbpopt)
        p.plot(interp_chip, logfunc(interp_chip, *bbpopt), c=best_fit,  label="predicted beta best", linestyle="--")

    if _mean:
        plt.scatter(meshsize_col, mean_beta, c=mean_col, label='real mean sequence')
        bapopt, pcov = curve_fit(logfunc, meshsize_col, mean_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("beta arbitrary", bapopt)
        p.plot(interp_chip, logfunc(interp_chip, *bapopt), c=fit_col, label="predicted beta arbitrary", linestyle="--")

    if _worst:
        plt.scatter(meshsize_col, worst_beta, c=worst_col, label='real worst sequence')
        bbpopt, pcov = curve_fit(logfunc, meshsize_col, worst_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("beta best", bbpopt)
        p.plot(interp_chip, logfunc(interp_chip, *bbpopt), c=fit_col,  label="predicted beta worst", linestyle='--')

    p.set_xlabel("meshsize")
    plt.title("parametrization of slope for arbitrary and best sequence routability by " + meshsizes)
    plt.legend()
    plt.savefig("beta_param_meshsize_" + meshsizes + ".png")
    plt.show()


def plotscatter(nl, routability, c, label=None, alpha=0.4, s=30):
    scatter_percent_wrapper(nl, routability, c=c, label=label, alpha=alpha, s=s)


def ABNLfit(nl, routability, c, fitfunc, label=None, plot=False):
    popt, pcov = curve_fit(fitfunc, nl, routability, p0=(40, 0.05), bounds=([-100, -2], [200, 1]))
    # popt, pcov = curve_fit(fitfunc, nl, routability, p0=(0.05, 0.05), bounds=([-1e6, 1e-5], [1e6, 1]))
    return popt, pcov

def ABNL_plot(nl, popts, fitfunc, c, label=None, perc=True):
    m = 100 if perc else 1
    if label:
        plt.plot(nl, fitfunc(nl, *popts)*m, c=c, linestyle='--', label=label)
    else:
        plt.plot(nl, fitfunc(nl, *popts)*m, c=c, linestyle='--')



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
    return  -const1*np.log(const2* (value-const3))


def getdfcol(df, n):
    """ get n'th column values of dataframe
    """
    return df[df.columns[n]]


def plot_residuals(types_of_scatter, mesh_size, param_func, end=False, title=None, savename=None, print_options=True):
    if end and not savename:
        raise ValueError("cannot")

    fitfunc, df, ab_df, nl = get_plot_necessities(param_func)
    type_count = len(types_of_scatter) + 1
    cor_cs = int(mesh_size/10-2)
    fig, axs = plt.subplots(2, 3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    for i, t in enumerate(types_of_scatter):
        fitted_val = np.array(fitfunc(nl, ab_df[t+'_a'][cor_cs], ab_df[t+'_b'][cor_cs]))
        y = df[eval(t+"_solv_str")(mesh_size)]

        axs[i,0].scatter(nl, y*100, c=eval(t+"_col"), label=t, alpha=0.4)
        axs[i,0].plot(nl, fitted_val, c=eval(t+"_col"), label=t, alpha=0.4)
        axs[i,0].set_ylabel("% routability")
        axs[i,0].set_title("routability for " + str(t) + " on size " + str(mesh_size))
        axs[i,0].legend()

        residual = (fitted_val - y)*100
        axs[i,1].scatter(nl, residual, label=t, alpha=0.4, c=eval(t+"_col"))
        axs[i,1].set_ylabel("% deviation from fit")
        axs[i,1].set_title("residual when fitting for " + str(t) + " on size " + str(mesh_size))
        axs[i,1].legend()

        rel_residual = [elem for elem in residual if abs(elem) > 0.1]
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


def get_plot_necessities(param_func):
    fitfunc = eval(param_func)
    df = pd.read_csv("compare_routability_best_of_200.csv")
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
    data = df[eval(modeltype+"_solv_str")(cs)]
    likelyhood_for_n = fitted_val**data*(1-fitted_val)**(1-data)
    tot_log_likelyhood = np.sum(np.log(likelyhood_for_n))
    return tot_log_likelyhood





if __name__ == '__main__':
    # param_func = "regular_logistic"
    # save_ab([(i+2)*10 for i in range(9)], param_func)
    # param_func = "expfunc"
    # save_ab([(i+2)*10 for i in range(9)], param_func)
    _arb = True
    _mean = False
    _best = True
    _worst = False
    fitfunc_names = ["regular_logistic", "expfunc"]
    cs = 80
    meshsizes = [(i+2)*10 for i in range(9)]
    for param_func in fitfunc_names:
        scatter_routability(["arb", "best"], cs, end=True, savename="k.png")
        plot_fits(["arb", "best"],"" ,param_func, cs=cs, scatter=True)
        plot_fits_dif("best", "arb", param_func, "difference best and mean")
        fit_ab("area", param_func, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
        fit_ab("edge_size", param_func, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
        compare_expected_best(param_func)
        plot_alpha_beta(meshsizes, ["arb"], param_func, "")
        plot_alpha_beta(meshsizes, ["arb", "best"], param_func, "", scatter=False)
        for cs in meshsizes:
            plot_residuals(["mean", "best"], cs, param_func, end=True, savename="k.png")
