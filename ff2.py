
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

        nl = df['netlist length']
        y_best = df[best_solv_str(iters, cs)]
        plt.plot(nl, y_best, 'b-', label='data')
        poptb, pcov = curve_fit(expfunc, nl, y_best, p0=(-40, 0.05), bounds=([-100, -5], [200, 1]))
        params_b.append(poptb)
        plt.plot(nl, expfunc(nl, *poptb), 'g--', label='fit best: alpha=%5.3f, beta=%5.8f' % tuple(poptb))
        # plt.xlabel('x')
        # plt.ylabel('y')
        param_csv.write(",".join([str(cs)]+lstr(popt)+lstr(poptb))+"\n")
        # savename = os.path.join(save_base, "fitted_" + str(cs) + ".png")
        # if not os.path.exists(savename):
        #     os.makedirs(os.path.pardir(savename))
        # plt.savefig(savename)
        # plt.clf()
    plt.legend()
    plt.show()
    plt.subplot(4,1,1)
    fitted_rs = [expfunc(nl, *ropts) for ropts in params_r]
    for i, fitted_r in enumerate(fitted_rs):
        plt.plot(nl, fitted_r, '--', label=str(chipsizes[i]))
    fitted_bs = [expfunc(nl, *bopts) for bopts in params_b]
    plt.subplot(4,1,2)
    for i, fitted_b in enumerate(fitted_bs):
        plt.plot(nl, fitted_b, '--', label=str(chipsizes[i]))
    plt.subplot(4,1,3)
    for i, fitted_r in enumerate(fitted_rs):
        plt.plot(nl, [fitted_bs[i][j]/(fitted_r[j]+1e-7) for j in range(len(nl))], '--', label='ratio b/r' + str(chipsizes[i]))
    plt.subplot(4,1,4)
    for i, fitted_b in enumerate(fitted_bs):
        plt.plot(nl, [fitted_rs[i][j]/(fitted_b[j]+1e-7) for j in range(len(nl))], '--', label='ratio b/r' + str(chipsizes[i]))
    plt.legend()
    plt.show()


def skiplist(iterable, start, interval=1):
    for i, elem in enumerate(iterable):
        if start <= i:
            if not (i-start) % (interval+1):
                yield elem



def load_ab():
    df = pd.read_csv("params.csv", header=1)
    print(df.head())
    for c in df.columns:
        print(c)
    for col in skiplist(df.columns[1:], 0, interval=2):
        plt.plot(df[df.columns[0]], eval("df['" + col + "']"), label=col)

    plt.legend()
    plt.show()
    for col in skiplist(df.columns[1:], 1, interval=2):
        plt.plot(df[df.columns[0]], eval("df['" + col + "']"), label=col)
    plt.legend()
    plt.show()

    for col in skiplist(df.columns[1:], 2, interval=2):
        plt.plot(df[df.columns[0]], eval("df['" + col + "']"), label=col)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    #calc_alpha_beta(1, [(i+2)*10 for i in range(6)])
    calc_alpha_beta(200, [(i+2)*10 for i in range(9)])
    load_ab()
