import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp, floor
import numpy as np

from ff_routability import regular_logistic, \
                            load_ab, \
                            getdfcol, \
                            named_function



# initial_col = '#1b9e77'
fit_col = '#1b9e77'
best_fit = '#d95f02'
best_col = '#d95f02'
mean_col = 'b'
worst_col = 'magenta'
black = 'k'

meta_fit_vars = {}
meta_fit_vars['upper'] = [30000, 20, 30000]
meta_fit_vars['lower'] = [-30000, 0.000000000001, -30000]
meta_fit_vars['upper_s'] = [300, 2, 10000]
meta_fit_vars['lower_s'] = [-40, -0.5, -400]

meta_fit_vars['start'] = (-13, 0.005, 10)
meta_fit_vars['start_s'] = (13, 0.005, 10)
MFV = meta_fit_vars

@named_function('log_of_meshsize')
def log_of_meshsize(value, const1, *args):
    return const1*np.log(value)

@named_function('log_of_meshsize_over_terminals')
def log_of_meshsize_over_terminals(value, const1, *args):
    return const1*np.log(value*(1./100.))

@named_function('log_of_meshsize_over_found_lambda')
def log_of_meshsize_over_found_lambda(value, const1, *args):
    return const1*np.log(value*0.04)

@named_function('log_of_meshsize_over_nice_lambda')
def log_of_meshsize_over_nice_lambda(value, const1, *args):
    return const1*np.log(value*0.02)

@named_function('original_logfunc')
def original_logfunc(value, const1, const2, const3):
    """original was with -const1 """
    return  const1 * np.log( const2 * (value - const3))


def round_after_comma(number, significant):
    s_n = str(number)
    i = s_n.index('.')
    return float(s_n[:i] + '.' + s_n[i+1:i+1+significant])



def log10(num):
    return np.log(num)/np.log(10)

def get_significant(x, n):
   r = round(x, -int(floor(log10(abs(x)))) + (n))
   return r


def format_found_params(const1, const2, const3):
    return "r'" + str(get_significant(const1, 2)) + " $\cdot$ ln( " + str(get_significant(const2, 2)) + " $\cdot$ (x - " + str(int(get_significant(const3, 2))) + "))'"

def params_to_graph_data(function_name, params):
    func2graph_data = {}
    func2graph_data['log_of_meshsize'] = {'title': 'scalar (k) * ln(meshsize)', 'label': 'k = ' + str(params[0])}
    func2graph_data['log_of_meshsize_over_terminals'] = {'title': 'scalar (k) * ln(meshsize/100)', 'label': 'k = ' + str(params[0])}
    func2graph_data['log_of_meshsize_over_found_lambda'] = {'title': 'scalar (k) * ln(meshsize*0.046)', 'label': 'k = ' + str(params[0])}
    return func2graph_data[function_name]['title'], func2graph_data[function_name]['label']

def meta_fit(mesh_metric, param_func, meta_param_func, _arb=False, _mean=False, _best=False, _worst=False):

    print("\n" + mesh_metric)
    meta_param_func_str = meta_param_func
    meta_param_func = eval(meta_param_func)

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(111)

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

    print("shift")
    # for t in types:
    #
    if _arb:
        ax.scatter(mesh_metric_col, initial_shift, c=mean_col, label='initial pathlist')
        aapopt, pcov = curve_fit(meta_param_func, mesh_metric_col, initial_shift, p0=MFV['start'], bounds=(MFV['lower'], MFV['upper']))
        print(*aapopt, interp_mesh)
        print("shift arbitrary", aapopt)
        ax.plot(interp_mesh, meta_param_func(interp_mesh, *aapopt), c=black, linestyle="--")


    if _best:
        ax.scatter(mesh_metric_col, best_shift, c=best_col, label='after permutation')
        abpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, best_shift, p0=MFV['start'], bounds=(MFV['lower'], MFV['upper']))
        print("shift best", abpopt)
        ax.plot(interp_mesh, meta_param_func(interp_mesh, *abpopt), c=black, linestyle="--")

    if _mean:
        ax.scatter(mesh_metric_col, mean_shift, c=mean_col, label='discovered shift mean')
        ampopt, pcov = curve_fit(meta_param_func, mesh_metric_col, mean_shift, p0=MFV['start'], bounds=(MFV['lower'], MFV['upper']))
        print("shift mean", ampopt)
        ax.plot(interp_mesh, meta_param_func(interp_mesh, *ampopt), c=black, linestyle='--')

    if _worst:
        ax.scatter(mesh_metric_col, worst_shift, c=worst_col, label='discovered shift worst')
        awpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, worst_shift, p0=MFV['start'], bounds=(MFV['lower'], MFV['upper']))
        print("shift worst", awpopt)
        ax.plot(interp_mesh, meta_param_func(interp_mesh, *awpopt), c=black, linestyle='--')

    plt.xlabel("mesh area")
    plt.ylabel("routability point")
    # plt.title('routability point ' + meta_param_func.name)
    ax.text(5000, 60, eval(format_found_params(*abpopt)), fontsize=12)
    ax.text(4900, 30, eval(format_found_params(*aapopt)), fontsize=12)

    ax.legend()
    print(param_func + "/shift_param_chipsize_" + mesh_metric + ".png")
    plt.savefig(param_func + "/shift_param_chipsize_" + mesh_metric + ".png")
    plt.show()

    initial_slope = getdfcol(df,2)
    best_slope = getdfcol(df,6)
    mean_slope = getdfcol(df,4)
    worst_slope = getdfcol(df,8)
    plt.plot([],[],c="#000000", linestyle="--", label="generalization")
    print("slope")
    if _arb:
        plt.scatter(mesh_metric_col, initial_slope, c=mean_col, label='initial pathlist')
        bapopt, pcov = curve_fit(meta_param_func, mesh_metric_col, initial_slope, p0=MFV['start_s'], bounds=(MFV['lower_s'], MFV['upper_s']))
        print("slope arbitrary", bapopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bapopt), c=black, linestyle="--")

    if _best:
        plt.scatter(mesh_metric_col, best_slope, c=best_col, label='after permutation')
        bbpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, best_slope, p0=MFV['start_s'], bounds=(MFV['lower_s'], MFV['upper_s']))
        print("slope best", bbpopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bbpopt), c=black, linestyle="--")

    if _mean:
        plt.scatter(mesh_metric_col, mean_slope, c=mean_col, label='discovered mean sequence')
        bapopt, pcov = curve_fit(meta_param_func, mesh_metric_col, mean_slope, p0=MFV['start_s'], bounds=(MFV['lower_s'], MFV['upper_s']))
        print("slope arbitrary", bapopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bapopt), c=black, linestyle="--")

    if _worst:
        plt.scatter(mesh_metric_col, worst_slope, c=worst_col, label='discovered worst sequence')
        bbpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, worst_slope, p0=MFV['start_s'], bounds=(MFV['lower_s'], MFV['upper_s']))
        print("slope best", bbpopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bbpopt), c=black, linestyle='--')

    plt.xlabel("mesh area")
    plt.title('slope ' + meta_param_func.name)

    plt.legend()
    plt.savefig(param_func +"/slope_param_chipsize_" + mesh_metric + ".png")
    plt.show()


if __name__ == '__main__':
    N = 10
    _arb = True
    _mean = False
    _best = True
    _worst = False
    fitfunc_names = []
    chipsizes = [(i+2)*10 for i in range(9)]
    # meta_function = "log_of_meshsize"
    # meta_fit("area", "regular_logistic", meta_function, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    # meta_function = "log_of_meshsize_over_terminals"
    # meta_fit("area", "regular_logistic", meta_function, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    # meta_function = "log_of_meshsize_over_found_lambda"
    # meta_fit("area", "regular_logistic", meta_function, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    # meta_function = "log_of_meshsize_over_nice_lambda"
    # meta_fit("area", "regular_logistic", meta_function, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    meta_function = "original_logfunc"
    meta_fit("area", "regular_logistic", meta_function, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    print(get_significant(12.34567, 2))
    print(get_significant(0.777, 2))
