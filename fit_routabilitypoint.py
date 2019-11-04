import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp
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
meta_fit_vars['upper'] = 100
meta_fit_vars['lower'] = -100
meta_fit_vars['upper_s'] = -0.1
meta_fit_vars['start_routability'] = 1
meta_fit_vars['start_slope'] = -0.2
MFV = meta_fit_vars

@named_function('log_of_meshsize')
def log_of_meshsize(value, const1):
    return const1*np.log(value)

@named_function('log_of_meshsize_over_terminals')
def log_of_meshsize_over_terminals(value, const1):
    return const1*np.log(value*(1./100.))

@named_function('log_of_meshsize_over_found_lambda')
def log_of_meshsize_over_found_lambda(value, const1):
    return const1*np.log(value*0.02)

def params_to_label(function_name, params):
    func2graph_data = {}
    func2graph_data['log_of_meshsize'] = {'title': 'scalar (k) * ln(meshsize)', 'label': 'k = ' + str(params[0])}
    func2graph_data['log_of_meshsize_over_terminals'] = {'title': 'scalar (k) * ln(meshsize/100)', 'label': 'k = ' + str(params[0])}
    func2graph_data['log_of_meshsize_over_found_lambda'] = {'title': 'scalar (k) * ln(meshsize*0.046)', 'label': 'k = ' + str(params[0])}
    return func2graph_data[function_name]['title'], func2graph_data[function_name]['label']

def meta_fit(mesh_metric, param_func, meta_param_func, _arb=False, _mean=False, _best=False, _worst=False):
    print("\n" + mesh_metric)
    meta_param_func_str = meta_param_func
    meta_param_func = eval(meta_param_func)
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

    plt.plot([],[],c="#000000", linestyle='--', label="generalization")
    print("shift")
    # for t in types:
    #
    if _arb:
        plt.scatter(mesh_metric_col, initial_shift, c=mean_col, label='initial')
        aapopt, pcov = curve_fit(meta_param_func, mesh_metric_col, initial_shift, p0=MFV['start_routability'], bounds=([MFV['lower']], [MFV['upper']]))
        print(*aapopt, interp_mesh)
        print("shift arbitrary", aapopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *aapopt), c=black, linestyle="--")


    if _best:
        plt.scatter(mesh_metric_col, best_shift, c=best_col, label='after')
        abpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, best_shift, p0=MFV['start_routability'], bounds=([MFV['lower']], [MFV['upper']]))
        print("shift best", abpopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *abpopt), c=black, linestyle="--")

    if _mean:
        plt.scatter(mesh_metric_col, mean_shift, c=mean_col, label='discovered shift mean')
        ampopt, pcov = curve_fit(meta_param_func, mesh_metric_col, mean_shift, p0=MFV['start_routability'], bounds=([MFV['lower']], [MFV['upper']]))
        print("shift mean", ampopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *ampopt), c=black, linestyle='--')

    if _worst:
        plt.scatter(mesh_metric_col, worst_shift, c=worst_col, label='discovered shift worst')
        awpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, worst_shift, p0=MFV['start_routability'], bounds=([MFV['lower']], [MFV['upper']]))
        print("shift worst", awpopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *awpopt), c=black, linestyle='--')

    plt.xlabel("mesh area")
    plt.ylabel("routability point")
    # plt.title("shift of routability function w.r.t chipsize " + mesh_metric)
    plt.legend()
    print(param_func + "/shift_param_chipsize_" + mesh_metric + ".png")
    plt.savefig(param_func + "/shift_param_chipsize_" + mesh_metric + ".png")
    plt.show()
    plt.clf()

    initial_slope = getdfcol(df,2)
    best_slope = getdfcol(df,6)
    mean_slope = getdfcol(df,4)
    worst_slope = getdfcol(df,8)
    plt.plot([],[],c="#000000", linestyle="--", label="generalization")
    print("slope")
    if _arb:
        plt.scatter(mesh_metric_col, initial_slope, c=mean_col, label='initial')
        bapopt, pcov = curve_fit(meta_param_func, mesh_metric_col, initial_slope, p0=MFV['start_slope'], bounds=([MFV['lower']], [MFV['upper_s']]))
        print("slope arbitrary", bapopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bapopt), c=black, linestyle="--")

    if _best:
        plt.scatter(mesh_metric_col, best_slope, c=best_col, label='after')
        bbpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, best_slope, p0=MFV['start_slope'], bounds=([MFV['lower']], [MFV['upper_s']]))
        print("slope best", bbpopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bbpopt), c=black, linestyle="--")

    if _mean:
        plt.scatter(mesh_metric_col, mean_slope, c=mean_col, label='discovered mean sequence')
        bapopt, pcov = curve_fit(meta_param_func, mesh_metric_col, mean_slope, p0=MFV['start_slope'], bounds=([MFV['lower']], [MFV['upper_s']]))
        print("slope arbitrary", bapopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bapopt), c=black, linestyle="--")

    if _worst:
        plt.scatter(mesh_metric_col, worst_slope, c=worst_col, label='discovered worst sequence')
        bbpopt, pcov = curve_fit(meta_param_func, mesh_metric_col, worst_slope, p0=MFV['start_slope'], bounds=([MFV['lower']], [MFV['upper_s']]))
        print("slope best", bbpopt)
        plt.plot(interp_mesh, meta_param_func(interp_mesh, *bbpopt), c=black, linestyle='--')

    plt.xlabel("mesh area")
    plt.title("slope of routability function w.r.t " + mesh_metric)

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
    meta_function = "log_of_meshsize_over_found_lambda"
    meta_fit("area", "regular_logistic", meta_function, _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
