import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp
import numpy as np

datafile = "compare_solvability_best_of_1.csv"

df = pd.read_csv(datafile)

def rand_solv_str(chipsize):
    return "solvability by random "+ str(chipsize) +"x" + str(chipsize)


def best_solv_str(iters, chipsize):
    return "solvability best of "+ str(iters) + " " + str(chipsize) +"x" + str(chipsize)


print(df[['netlist length', rand_solv_str(20), best_solv_str(1, 20)]])
# print(df.head())


def func(c):
    def expfunc(n, const1, const2):
        return np.exp(-np.exp(2*n*const1)*c*const2)
    return expfunc



for cs in [20,30,40,50,60,70,80,90,100]:
    nl = df['netlist length']
    y_rand = df[rand_solv_str(cs)]
    expfunc = func(cs)
    plt.plot(nl, y_rand, 'b-', label='data')
    popt, pcov = curve_fit(expfunc, nl, y_rand, bounds=(0, [0.08, 0.1]))
    plt.plot(nl, expfunc(nl, *popt), 'g--', label='fit: const1=%5.3f, const2=%5.8f' % tuple(popt))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("fitted_rand_" + str(chipsize) + ".png")

    nl = df['netlist length']
    y_rand = df[best_solv_str(200, cs)]
    plt.plot(nl, y_rand, 'b-', label='data')
    popt, pcov = curve_fit(expfunc, nl, y_rand, bounds=(0, [0.08, 0.1]))
    plt.plot(nl, expfunc(nl, *popt), 'g--', label='fit: const1=%5.3f, const2=%5.8f' % tuple(popt))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig("fitted_rand_" + str(chipsize) + ".png")
