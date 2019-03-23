
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp
import numpy as np


def rand_solv_str(chipsize):
    return "solvability by random "+ str(chipsize) +"x" + str(chipsize)


def best_solv_str(iters, chipsize):
    return "solvability best of "+ str(iters) + " " + str(chipsize) +"x" + str(chipsize)


def mean_solv_str(chipsize):
    return "solvability of mean {}x{}".format(chipsize, chipsize)

def expfunc(nl, const1, const2):
    return np.exp(-np.exp(2*(nl+const1)*const2))


def lstr(iterable):
    return [str(elem) for elem in iterable]


def calc_alpha_beta(iters, chipsizes):
    datafile = "compare_solvability_best_of_" + str(iters) + ".csv"

    df = pd.read_csv(datafile)

    param_csv = open("params.csv", "w+")
    param_csv.write(", arbitrary netlists,,mean solvability,,optimized netlist order,\n")
    param_csv.write("chip surface (X by Z), alpha, beta, alpha, beta, alpha" +str(iters)+", beta " +str(iters)+"\n")
    params_r = []
    params_b = []
    params_m = []
    nl = df['netlist length']
    for j, cs in enumerate(chipsizes):
        plt.subplot(3, 3, j+1)
        save_base = os.path.join(os.path.curdir, "fit_plots")
        nl = df['netlist length']
        y_rand = df[rand_solv_str(cs)]
        plt.plot(nl, y_rand, 'b-', label='data')
        popt, pcov = curve_fit(expfunc, nl, y_rand, p0=(-40, 0.05), bounds=([-100, -2], [200, 1]))
        params_r.append(popt)
        plt.plot(nl, expfunc(nl, *popt), 'g--', label='fit rand: alpha=%5.3f, beta=%5.8f' % tuple(popt))

        plt.subplot(3, 3, j+1)
        save_base = os.path.join(os.path.curdir, "fit_plots")
        nl = df['netlist length']
        y_mean = df[mean_solv_str(cs)]
        plt.plot(nl, y_rand, 'b-', label='data')
        poptm, pcov = curve_fit(expfunc, nl, y_mean, p0=(-40, 0.05), bounds=([-100, -2], [200, 1]))
        params_m.append(poptm)
        plt.plot(nl, expfunc(nl, *poptm), 'g--', label='fit rand: alpha=%5.3f, beta=%5.8f' % tuple(popt))


        nl = df['netlist length']
        y_best = df[best_solv_str(iters, cs)]
        plt.plot(nl, y_best, 'b-', label='data')
        poptb, pcov = curve_fit(expfunc, nl, y_best, p0=(-40, 0.05), bounds=([-100, -5], [200, 1]))
        params_b.append(poptb)
        plt.plot(nl, expfunc(nl, *poptb), 'g--', label='fit best: alpha=%5.3f, beta=%5.8f' % tuple(poptb))
        # plt.xlabel('x')
        # plt.ylabel('y')
        param_csv.write(",".join([str(cs)]+lstr(popt)+lstr(poptm)+lstr(poptb))+"\n")
        # savename = os.path.join(save_base, "fitted_" + str(cs) + ".png")
        # if not os.path.exists(savename):
        #     os.makedirs(os.path.pardir(savename))
        # plt.savefig(savename)
        # plt.clf()

    plt.legend()
    plt.show()
    fitted_rs = [expfunc(nl, *ropts) for ropts in params_r]
    fitted_ms = [expfunc(nl, *mopts) for mopts in params_m]
    fitted_bs = [expfunc(nl, *bopts) for bopts in params_b]

    plt.figure()
    plt.subplot(3,1,1)
    for i, fitted_r in enumerate(fitted_rs):
        plt.plot(nl, fitted_r, '--')
    plt.subplot(3,1,3)
    for i, fitted_b in enumerate(fitted_bs):
        plt.plot(nl, fitted_b, '--')
    # plt.subplot(5,1,5)
    # for i, fitted_b in enumerate(fitted_bs):
    #     plt.plot(nl, [fitted_ms[i][j]/(fitted_b[j]+1e-7) for j in range(len(nl))], '--')
    p = plt.subplot(3,1,2)
    for i, fitted_m in enumerate(fitted_ms):
        plt.plot(nl, fitted_m, '--', label=str(chipsizes[i]))
    p.legend(loc=1, bbox_to_anchor=(.6, 1.2, 0.5, 0.5))
    plt.show()
    plt.subplot(1,1,1)
    for i, fitted_m in enumerate(fitted_ms):
        plt.plot(nl, [fitted_bs[i][j]/(fitted_m[j]+1e-3) for j in range(len(nl))], '--', label=str(chipsizes[i]))
    plt.legend()
    plt.show()


def skiplist(iterable, start, interval=1):
    for i, elem in enumerate(iterable):
        if start <= i:
            if not (i-start) % (interval+1):
                yield elem



def load_ab():
    df = pd.read_csv("params.csv", header=1)
    # print(df.head())
    return df

def dfcprint(df, c):
    print(df.columns[c])
    print(getdfcol(df,c))


def logfunc(value, const1, const2, const3):
    return -const1*np.log(const2* (value+const3))


def fit_ab(df):
    chipsize_col = getdfcol(df,0)
    chipsize_col = getdfcol(df,0)**2
    plt.plot(chipsize_col, getdfcol(df,3), 'o-', label='real alpha arbitrary')
    aapopt, pcov = curve_fit(logfunc, chipsize_col, getdfcol(df,3), p0=(13, 0.05, 10), bounds=([-300, 0.00001, -40], [100, 0.5, 30]))
    print("alpha arbitrary", aapopt)
    plt.plot(chipsize_col, logfunc(chipsize_col, *aapopt), 'r--', label="expected alpha arbitrary")

    plt.plot(chipsize_col, getdfcol(df,5), 'o-', label='real alpha best')
    abpopt, pcov = curve_fit(logfunc, chipsize_col, getdfcol(df,5), p0=(13, 0.05, 10), bounds=([-300, 0.00001, -40], [100, 0.5, 30]))
    print("alpha best", abpopt)
    plt.plot(chipsize_col, logfunc(chipsize_col, *abpopt), 'g--', label="expected alpha best")
    plt.title("parametrization of alpha for arbitrary and best")
    plt.legend()
    plt.savefig("alpha_param_chipsize.png")
    plt.show()

    plt.plot(chipsize_col, getdfcol(df,4), 'b-', label='real beta arbitrary')
    bapopt, pcov = curve_fit(logfunc, chipsize_col, getdfcol(df,4), p0=(13, 0.005, 10), bounds=([0, 0.0000001, -40], [300, 0.009, 300]))
    print("beta arbitrary", bapopt)
    plt.plot(chipsize_col, logfunc(chipsize_col, *bapopt), 'r--', label="expected beta arbitrary")

    plt.plot(chipsize_col, getdfcol(df,6), 'o-', label='real beta best')
    bbpopt, pcov = curve_fit(logfunc, chipsize_col, getdfcol(df,6), p0=(13, 0.005, 10), bounds=([0, 0.0000001, -40], [300, 0.009, 300]))
    print("beta best", bbpopt)
    plt.plot(chipsize_col, logfunc(chipsize_col, *bbpopt), 'g--',  label="expected beta best")
    plt.title("parametrization of beta for arbitrary and best")
    plt.legend()
    plt.savefig("beta_param_chipsize.png")


def getdfcol(df, c):
    """ get c'th column values of dataframe
    """
    return df[df.columns[c]]

def plot_ab(df):
    for col in skiplist(df.columns[1:], 0, interval=1):
        plt.plot(df[df.columns[0]], eval("df['" + col + "']"), label=col)
    plt.legend()
    plt.show()
    for col in skiplist(df.columns[1:], 1, interval=1):
        plt.plot(df[df.columns[0]], eval("df['" + col + "']"), label=col)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #calc_alpha_beta(1, [(i+2)*10 for i in range(6)])
    calc_alpha_beta(200, [(i+2)*10 for i in range(9)])
    df = load_ab()
    fit_ab(df)
