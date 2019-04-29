
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


def written_math(expression):
    def add_expr(func):
        func.math_expr = expression
        return func
    return add_expr

def format_chipsize(chipsize):
    return str(chipsize)+"x"+str(chipsize)

def arb_solv_str(chipsize):
    return "solvability by arb {}x{}".format(str(chipsize), str(chipsize))

def best_solv_str(chipsize):
    return "solvability best of {}x{}".format(str(chipsize), str(chipsize))

def worst_solv_str(chipsize):
    return "solvability worst of {}x{}".format(str(chipsize), str(chipsize))

def mean_solv_str(chipsize):
    return "solvability of mean {}x{}".format(chipsize, chipsize)

@written_math("e^{-e^{(2 \cdot (nl - shift) * slope)}}")
def expfunc(nl, const1, const2):
    return np.exp(-np.exp(2*(nl-const1)*const2))

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



def save_ab(chipsizes, fitfunc):
    datafile = "compare_solvability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    param_csv = open("params.csv", "w+")
    param_csv.write(", arbitrary netlists,,mean solvability,,optimized netlist order,,worst order,\n")
    param_csv.write("chipsize (XxZ),arb_a,arb_b,mean_a,mean_b,best_a,best_b,worst_a,worst_b\n")
    for j, chipsize in enumerate(chipsizes):

        y_arb = df[arb_solv_str(chipsize)]
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


def plot_alpha_beta(chipsizes, title, plot_savefile):
    datafile = "compare_solvability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())
    ab_df = load_ab()

    params_r = []
    params_b = []
    params_m = []
    params_w = []

    _best = False
    _mean = False
    _arb = True
    _worst = False
    fig=plt.figure(figsize=(7,7))
    fig.suptitle('effect of permutation on predicted solavability') # or plt.suptitle('Main title')
    legend_loc = 5
    for j, cs in enumerate(chipsizes):
        print((1+j+j//4)//4, (j+j//3)%4)
        ax = plt.subplot2grid((3, 4), ((1+j+j//4)//4, (j+j//3)%4))
        ax.set_title(format_chipsize(cs))
        y_arb = df[arb_solv_str(chipsize)]
        if not determine_3x3_x(j):
            ax.set_xticks([])
        else:
            ax.set_xticks([10,50,90])
        if determine_3x3_nl(j):
            ax.set_xlabel("netlist length")
        if not determine_3x3_y(j):
            ax.set_yticks([])
        if determine_3x3_solv(j):
            ax.set_ylabel("solvability %")

        labelwindow = j==legend_loc

        if _arb:
            y_arb = df[arb_solv_str(chipsize)]
            popta = ab_df[arb_solv_str(chipsize)]
            plotscatter(nl, y_arb, c=arb_col, s=6, label=conditional_label(labelwindow, "arbitrary case"))
            ABNL_plot(nl, popta, c=fit_col, label=conditional_label(labelwindow, "predicted arbitrary \n case"))

        if _mean:
            y_mean = df[mean_solv_str(chipsize)]
            poptm = ab_df[mean_solv_str(chipsize)]
            plotscatter(nl, y_mean, c=mean_col, s=6, label=conditional_label(labelwindow, "permutated average \n case"))
            ABNL_plot(nl, y_mean, c=fit_col, label=conditional_label(labelwindow, "predicted average \n case"))

        if _best:
            y_best = df[best_solv_str(chipsize)]
            poptb = ab_df[best_solv_str(chipsize)]
            plotscatter(nl, y_best, c=best_col, s=6, label=conditional_label(labelwindow, "permutated best case"))
            ABNL_plot(nl, poptb, c=best_fit, label=conditional_label(labelwindow, "predicted permutated \n best case"))

        if _worst:
            y_worst = df[worst_solv_str(chipsize)]
            poptw = ab_df[worst_solv_str(chipsize)]
            plotscatter(nl, y_worst, c=worst_col, s=6, label=conditional_label(labelwindow, "permutated worst case"))
            ABNL_plot(nl, poptw, c=fit_col, plot=plotfit, label=conditional_label(labelwindow, "predicted permutated \n worst case"))

        if labelwindow:
            # Put a legend to the right of the current axis
            lgd = plt.legend(bbox_to_anchor=(1, 1.0))
    # plt.suptitle("solvability for different chipsizes")
    plt.suptitle(title)
    plt.savefig(plot_savefile, bbox_extra_artists=(lgd,))
    plt.show()


def plot_fits(types_of_fit, suptitle, fitfunc, scatter=False):
    ab_df = load_ab()
    datafile = "compare_solvability_best_of_200.csv"
    df = pd.read_csv(datafile, index_col="netlist length")
    nl = np.array(df.index.values.tolist())

    type_count = len(types_of_fit)
    sizes = ab_df.index.values.tolist()
    plt.figure(figsize=(7,7))
    # print(nl)
    # input()
    for i, t in enumerate(types_of_fit):
        fitted_vals = [fitfunc(nl, ab_df[t+'_a'][size], ab_df[t+'_b'][size]) for size in sizes]

        p = plt.subplot(type_count,1,i+1)
        plt.setp(p.get_xticklabels(), visible=False)
        for j, fitted_m in enumerate(fitted_vals):
            plt.plot(nl, fitted_m)
            if scatter:
                y = df[eval(t+"_solv_str")((2+sizes[j])*10)]
                plt.scatter(nl, y, c=arb_col, label=t, alpha=0.4)
        box = p.get_position()
        p.set_position([box.x0, box.y0 + box.height*0.1,
                     box.width * 0.7, box.height])
        p.set_ylabel("solvability %")
        p.set_title(types_of_fit[i])
        p.set_ylabel("routability %")
        p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))

    # plt.suptitle("predicted solvability for differently sized chips\n\n")
    plt.suptitle(suptitle)
    p.set_xlabel("netlist length")

    plt.savefig("predicted_chipsize_compare.png")
    plt.show()



def compare_expected_best(fitfunc):
    ["best", "mean"]
    ab_df = load_ab()
    datafile = "compare_solvability_best_of_200.csv"
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
    p.set_ylabel("solvability %")
    p.set_title("mean")
    p.set_ylabel("routability %")
    p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))

    # plt.suptitle("predicted solvability for differently sized chips\n\n")
    plt.suptitle("compare expected best vs real best")
    p.set_xlabel("netlist length")

    plt.savefig("predicted_chipsize_compare.png")
    plt.show()


def plot_fits_dif(t1, t2, fitfunc, suptitle):
    ab_df = load_ab()
    datafile = "compare_solvability_best_of_200.csv"
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
        p.set_ylabel("solvability %")
        p.set_ylabel("routability %")
        p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))
    p.scatter([nl[v] for v in fit_difs_maxs], [fit_difs[i][v] for i, v in enumerate(fit_difs_maxs)])
    p.scatter([nl[v] for v in fit_difs_maxs], [1 for i, v in enumerate(fit_difs_maxs)])

    # plt.suptitle("predicted solvability for differently sized chips\n\n")
    plt.suptitle(suptitle)
    p.set_xlabel("netlist length")

    plt.savefig("predicted_chipsize_compare.png")
    plt.show()


def scatter_routability(types_of_scatter, cs, end=False, title=None, savename=None, print_options=True):

    if end and not savename:
        raise ValueError(cannot)
    datafile = "compare_solvability_best_of_200.csv"
    df = pd.read_csv(datafile)
    nl = df.index.values.tolist()
    plt.figure(figsize=(7,7))
    type_count = len(types_of_scatter) + 1
    for i, t in enumerate(types_of_scatter):

        p = plt.subplot(type_count,1,i + 1)
        y_arb = df[eval(t+"_solv_str")(cs)]
        plt.scatter(nl, y_arb, c=arb_col, label=t, alpha=0.4)
        p.set_ylabel("routability %")

    p = plt.subplot(type_count,1, 3)
    y_arb = 1 - (1 - df[eval("mean_solv_str")(cs)])**200
    plt.scatter(nl, y_arb, c=arb_col, label=t, alpha=0.4)
    p.set_ylabel("routability %")


    if end:
        if title:
            plt.title("solvability of arbitrary netlists on a "+format_chipsize(cs) + " chip")
        p.set_xlabel("netlist length")
        plt.legend()
        plt.savefig(savename)
        # plt.savefig("arbitrary_case_"+format_chipsize(cs)+".png")
        plt.show()


def fit_ab(chipsizes, _arb=False, _mean=False, _best=False, _worst=False):
    df = load_ab()
    if chipsizes == "area":
        chipsize_col = getdfcol(df,0)**2
    elif chipsizes == "edge_size":
        chipsize_col = getdfcol(df,0)
    interp_chip = [i for i in range(min(chipsize_col), max(chipsize_col), 1)]
    arbitrary_alpha = getdfcol(df,1)
    best_alpha = getdfcol(df,5)
    mean_alpha = getdfcol(df,3)
    worst_alpha = getdfcol(df,7)
    p = plt.subplot(111)

    if _arb:
        plotscatter(chipsize_col, arbitrary_alpha, c=arb_col, label='real alpha arbitrary')
        aapopt, pcov = curve_fit(logfunc, chipsize_col, arbitrary_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(chipsize_col)], [40000, 0.9, max(chipsize_col)]))
        print("alpha arbitrary", aapopt)
        plt.plot(interp_chip, logfunc(interp_chip, *aapopt), c=fit_col, label="predicted \n arbitrary case", linestyle="--")

    if _best:
        plotscatter(chipsize_col, best_alpha, c=best_col, label='real alpha best')
        abpopt, pcov = curve_fit(logfunc, chipsize_col, best_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(chipsize_col)], [40000, 0.9, max(chipsize_col)]))
        print("alpha best", abpopt)
        plt.plot(interp_chip, logfunc(interp_chip, *abpopt), c=best_fit, label="predicted \n best case", linestyle="--")

    if _mean:
        plt.scatter(chipsize_col, mean_alpha, c=mean_col, label='real alpha mean')
        ampopt, pcov = curve_fit(logfunc, chipsize_col, mean_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(chipsize_col)], [40000, 0.9, max(chipsize_col)]))
        print("alpha mean", ampopt)
        plt.plot(interp_chip, logfunc(interp_chip, *ampopt), c=fit_col, label="predicted \n average case", linestyle='--')

    if _worst:
        plotscatter(chipsize_col, worst_alpha, c=worst_col, label='real alpha worst')
        awpopt, pcov = curve_fit(logfunc, chipsize_col, worst_alpha, p0=(13, 0.05, 10), bounds=([-30000, 0.0001, -max(chipsize_col)], [40000, 0.9, max(chipsize_col)]))
        print("alpha worst", awpopt)
        plt.plot(interp_chip, logfunc(interp_chip, *awpopt), c=fit_col, label="predicted \n worst case", linestyle='--')

    p.set_xlabel("chipsize")
    plt.title("parametrization of alpha for average and best case solvability by "+" ".join(chipsizes.split("_")))
    plt.legend()
    plt.savefig("alpha_param_chipsize_"+chipsizes+".png")
    plt.show()
    plt.clf()

    arbitrary_beta = getdfcol(df,2)
    best_beta = getdfcol(df,6)
    mean_beta = getdfcol(df,4)
    worst_beta = getdfcol(df,8)
    p = plt.subplot(111)
    if _arb:
        plotscatter(chipsize_col, arbitrary_beta, c=arb_col, label='real arbitrary case')
        bapopt, pcov = curve_fit(logfunc, chipsize_col, arbitrary_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -40000], [300, 0.5, 10000]))
        print("beta arbitrary", bapopt)
        plt.plot(interp_chip, logfunc(interp_chip, *bapopt), c=fit_col, label="predicted beta arbitrary", linestyle="--")

    if _best:
        plotscatter(chipsize_col, best_beta, c=best_col, label='real best case')
        bbpopt, pcov = curve_fit(logfunc, chipsize_col, best_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 0.5, 10000]))
        print("beta best", bbpopt)
        plt.plot(interp_chip, logfunc(interp_chip, *bbpopt), c=best_fit,  label="predicted beta best", linestyle="--")

    if _mean:
        plotscatter(chipsize_col, mean_beta, c=mean_col, label='real mean case')
        bapopt, pcov = curve_fit(logfunc, chipsize_col, mean_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("beta arbitrary", bapopt)
        plt.plot(interp_chip, logfunc(interp_chip, *bapopt), c=fit_col, label="predicted beta arbitrary", linestyle="--")

    if _worst:
        plotscatter(chipsize_col, worst_beta, c=worst_col, label='real worst case')
        bbpopt, pcov = curve_fit(logfunc, chipsize_col, worst_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("beta best", bbpopt)
        plt.plot(interp_chip, logfunc(interp_chip, *bbpopt), c=fit_col,  label="predicted beta worst", linestyle='--')

    p.set_xlabel("chipsize")
    plt.title("parametrization of beta for average and best case solvability by " + chipsizes)
    plt.legend()
    plt.savefig("beta_param_chipsize_" + chipsizes + ".png")
    plt.show()


def plotscatter(nl, solvability, c, label=None, alpha=0.4, s=30):
    plt.scatter(nl, solvability, c=c, label=label, alpha=alpha, s=s)

def ABNLfit(nl, solvability, c, fitfunc, label=None, plot=False):
    # popt, pcov = curve_fit(fitfunc, nl, solvability, p0=(40, 0.05), bounds=([-100, -2], [200, 1]))
    popt, pcov = curve_fit(fitfunc, nl, solvability, p0=(0.05, 0.05), bounds=([-1e6, 1e-5], [1e6, 1]))
    return popt, pcov

def ABNL_plot(nl, popts, c, label=None):
    if label:
        plt.plot(nl, regular_logistic(nl, *popt), c=c, linestyle='--', label=label)
    else:
        plt.plot(nl, regular_logistic(nl, *popt), c=c, linestyle='--')



def skiplist(iterable, start, interval=1):
    for i, elem in enumerate(iterable):
        if start <= i:
            if not (i-start) % (interval+1):
                yield elem


def lprint(iterable):
    for elem in iterable:
        print(elem)

def load_ab(_print=False):
    df = pd.read_csv("params.csv", header=1)
    if _print:
        print(df.head())
        lprint(df.columns)
    return df

def dfcprint(df, c):
    print(df.columns[c])
    print(getdfcol(df,c))

@written_math("-const1 * natural log( const2 * (value - const3))")
def logfunc(value, const1, const2, const3):
    return  -const1*np.log(const2* (value-const3))


def getdfcol(df, c):
    """ get c'th column values of dataframe
    """
    return df[df.columns[c]]

def plot_ab(df):
    for col in skiplist(df.columns[1:], 0, interval=1):
        plt.plot(getdfcol(df, 0), eval("df['" + col + "']"), label=col)
    plt.legend()
    plt.show()
    for col in skiplist(df.columns[1:], 1, interval=1):
        plt.plot(getdfcol(df, 0), eval("df['" + col + "']"), label=col)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    fitfunc = expfunc
    save_ab([(i+2)*10 for i in range(9)], fitfunc)
    _arb = False
    _mean = True
    _best = False
    _worst = False
    fitfunc = regular_logistic
    scatter_routability(["mean", "best"], 100, end=True, savename="k.png")
    # plot_fits(["best", "mean"],"" ,fitfunc, scatter=True)
    # plot_fits_dif("best", "mean", fitfunc, "difference best and mean")
    # fit_ab("area", _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    # fit_ab("edge_size", _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    # compare_expected_best(fitfunc)
