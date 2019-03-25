
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import exp
import numpy as np

arb_col = 'g'
fit = 'g'
best_fit = 'r'
best_col = 'orange'
mean_col = 'b'
worst_col = 'magenta'

def format_chipsize(cs):
    return str(cs)+"x"+str(cs)

def rand_solv_str(chipsize):
    return "solvability by random "+ str(chipsize) +"x" + str(chipsize)

def best_solv_str(iters, chipsize):
    return "solvability best of "+ str(iters) + " " + str(chipsize) +"x" + str(chipsize)

def worst_solv_str(iters, chipsize):
    return "solvability worst of "+ str(iters) + " " + str(chipsize) +"x" + str(chipsize)

def mean_solv_str(chipsize):
    return "solvability of mean {}x{}".format(chipsize, chipsize)

def expfunc(nl, const1, const2):
    return np.exp(-np.exp(2*(nl-const1)*const2))

def lstr(iterable):
    return [str(elem) for elem in iterable]

def determine_9x9_y(elem_n):
    return not elem_n % 3

def determine_9x9_solv(elem_n):
    return elem_n == 3

def determine_9x9_nl(elem_n):
    return elem_n == 7

def determine_9x9_x(elem_n):
    return elem_n // 6



def calc_alpha_beta(iters, chipsizes):
    datafile = "compare_solvability_best_of_" + str(iters) + ".csv"
    df = pd.read_csv(datafile)

    param_csv = open("params.csv", "w+")
    param_csv.write(", arbitrary netlists,,mean solvability,,optimized netlist order,,worst order,\n")
    param_csv.write("chipsize (XxZ), alpha arb, beta arb, alpha mean, beta mean, alpha best, beta best, alpha worst, beta worst\n")
    params_r = []
    params_b = []
    params_m = []
    params_w = []
    nl = [v for v in df['netlist length']]
    _best = True
    _mean = True
    _arb = False
    _worst = False
    fig=plt.figure(figsize=(7,7))
    fig.suptitle('effect of permutation on predicted solavability') # or plt.suptitle('Main title')
    legend_loc = 5
    plotfit = True
    for j, cs in enumerate(chipsizes):
        print((1+j+j//4)//4, (j+j//3)%4)
        ax = plt.subplot2grid((3, 4), ((1+j+j//4)//4, (j+j//3)%4))
        ax.set_title(format_chipsize(cs))
        y_arb = df[rand_solv_str(cs)]
        if not determine_9x9_x(j):
            ax.set_xticks([])
        else:
            ax.set_xticks([10,50,90])
        if determine_9x9_nl(j):
            ax.set_xlabel("netlist length")
        if not determine_9x9_y(j):
            ax.set_yticks([])
        if determine_9x9_solv(j):
            ax.set_ylabel("solvability %")

        if _arb:
            if j==legend_loc:
                plotscatter(nl, y_arb, c=arb_col, s=6, label="arbitrary case")
                popt, pcov = ABNLfit(nl, y_arb, c=fit, plot=plotfit, label="predicted arbitrary \n case")
            else:
                plotscatter(nl, y_arb, c=arb_col, s=6)
                popt, pcov = ABNLfit(nl, y_arb, c=fit, plot=plotfit)
        else:
            popt, pcov = ABNLfit(nl, y_arb, c=fit, plot=False)

        y_mean = df[mean_solv_str(cs)]
        if _mean:
            if j==legend_loc:
                plotscatter(nl, y_mean, c=mean_col, s=6, label="permutated average \n case")
                poptm, pcov = ABNLfit(nl, y_mean, c=fit, plot=plotfit, label="predicted average \n case")
            else:
                plotscatter(nl, y_mean, c=mean_col, s=6)
                poptm, pcov = ABNLfit(nl, y_mean, c=fit, plot=plotfit)
        else:
            poptm, pcov = ABNLfit(nl, y_mean, c=fit, plot=False)

        y_best = df[best_solv_str(iters, cs)]
        if _best:
            if j==legend_loc:
                plotscatter(nl, y_best, c=best_col, s=6, label="permutated best case")
                poptb, pcov = ABNLfit(nl, y_best, c=best_fit, plot=plotfit, label="predicted permutated \n best case")
            else:
                plotscatter(nl, y_best, c=best_col, s=6)
                poptb, pcov = ABNLfit(nl, y_best, c=best_fit, plot=plotfit)
        else:
            poptb, pcov = ABNLfit(nl, y_best, c=fit, plot=False)

        y_worst = df[worst_solv_str(iters, cs)]
        if _worst:
            if j==legend_loc:
                plotscatter(nl, y_worst, c=worst_col, s=6, label="permutated worst case")
                poptw, pcov = ABNLfit(nl, y_worst, c=fit, plot=plotfit, label="predicted permutated \n worst case")
            else:
                plotscatter(nl, y_worst, c=worst_col, s=6)
                poptw, pcov = ABNLfit(nl, y_worst, c=fit, plot=plotfit)
        else:
            poptw, pcov = ABNLfit(nl, y_worst, c=fit, plot=False)

        params_r.append(popt)
        params_m.append(poptm)
        params_b.append(poptb)
        params_w.append(poptw)
        param_csv.write(",".join([str(cs)]+lstr(popt)+lstr(poptm)+lstr(poptb)+lstr(poptw))+"\n")
        if j==legend_loc:
            # Put a legend to the right of the current axis
            lgd = plt.legend(bbox_to_anchor=(1, 1.0))
    plt.suptitle("predicted solvability for different chipsizes")
    plt.savefig("permutation_exp_compare_9x9.png", bbox_extra_artists=(lgd,))
    plt.show()
    param_csv.close()

    fig=plt.figure(figsize=(7,7))
    fig.suptitle('effect of permutation on solavability') # or plt.suptitle('Main title')
    legend_loc = 5
    plotfit=False
    for j, cs in enumerate(chipsizes):
        print((1+j+j//4)//4, (j+j//3)%4)
        ax = plt.subplot2grid((3, 4), ((1+j+j//4)//4, (j+j//3)%4))
        ax.set_title(format_chipsize(cs))
        y_arb = df[rand_solv_str(cs)]
        if not determine_9x9_x(j):
            ax.set_xticks([])
        else:
            ax.set_xticks([10,50,90])
        if determine_9x9_nl(j):
            ax.set_xlabel("netlist length")
        if not determine_9x9_y(j):
            ax.set_yticks([])
        if determine_9x9_solv(j):
            ax.set_ylabel("solvability %")
        if _arb:
            if j==legend_loc:
                plotscatter(nl, y_arb, c=arb_col, s=6, label="arbitrary case")
                popt, pcov = ABNLfit(nl, y_arb, c=fit, plot=plotfit)
            else:
                plotscatter(nl, y_arb, c=arb_col, s=6)
                popt, pcov = ABNLfit(nl, y_arb, c=fit, plot=plotfit)
        else:
            popt, pcov = ABNLfit(nl, y_arb, c=fit, plot=False)

        y_mean = df[mean_solv_str(cs)]
        if _mean:
            if j==legend_loc:
                plotscatter(nl, y_mean, c=mean_col, s=6, label="permutated \n average case")
                poptm, pcov = ABNLfit(nl, y_mean, c=fit, plot=plotfit)
            else:
                plotscatter(nl, y_mean, c=mean_col, s=6)
                poptm, pcov = ABNLfit(nl, y_mean, c=fit, plot=plotfit)
        else:
            poptm, pcov = ABNLfit(nl, y_mean, c=fit, plot=False)

        y_best = df[best_solv_str(iters, cs)]
        if _best:
            if j==legend_loc:
                plotscatter(nl, y_best, c=best_col, s=6, label="permutated \n best case")
                poptb, pcov = ABNLfit(nl, y_best, c=best_fit, plot=plotfit)
            else:
                plotscatter(nl, y_best, c=best_col, s=6)
                poptb, pcov = ABNLfit(nl, y_best, c=best_fit, plot=plotfit)
        else:
            poptb, pcov = ABNLfit(nl, y_best, c=fit, plot=False)

        y_worst = df[worst_solv_str(iters, cs)]
        if _worst:
            if j==legend_loc:
                plotscatter(nl, y_worst, c=worst_col, s=6, label="permutated \n worst case")
                poptw, pcov = ABNLfit(nl, y_worst, c=fit, plot=plotfit)
            else:
                plotscatter(nl, y_worst, c=worst_col, s=6)
                poptw, pcov = ABNLfit(nl, y_worst, c=fit, plot=plotfit)
        else:
            poptw, pcov = ABNLfit(nl, y_worst, c=fit, plot=False)
        if j==legend_loc:
            # Put a legend to the right of the current axis
            lgd = plt.legend(bbox_to_anchor=(1, 1.0))
    plt.savefig("permutation_real_compare_9x9.png", bbox_extra_artists=(lgd,))
    plt.show()

    cs = 60

    fitted_rs = [expfunc(nl, *ropts) for ropts in params_r]
    fitted_ms = [expfunc(nl, *mopts) for mopts in params_m]
    fitted_bs = [expfunc(nl, *bopts) for bopts in params_b]
    fitted_ws = [expfunc(nl, *wopts) for wopts in params_w]

    plt.figure(figsize=(7,7))
    p = plt.subplot(2,1,1)
    plt.setp( p.get_xticklabels(), visible=False)
    for i, fitted_m in enumerate(fitted_ms):
        plt.plot(nl, fitted_m)
    box = p.get_position()
    p.set_position([box.x0, box.y0 + box.height*0.1,
                 box.width * 0.7, box.height])
    p.set_ylabel("solvability %")
    p.set_title("without permutation")

    p = plt.subplot(2,1,2)
    for i, fitted_b in enumerate(fitted_bs):
        plt.plot(nl, fitted_b, label=str(chipsizes[i])+"x"+str(chipsizes[i]))
    box = p.get_position()
    p.set_position([box.x0, box.y0 + box.height*0.1,
                 box.width * 0.7, box.height])
    p.set_title("with permutation")
    plt.suptitle("predicted solvability for differently sized chips\n\n")
    p.set_ylabel("solvability %")
    p.set_xlabel("netlist length")

    p.legend(loc="upper left", bbox_to_anchor=(1.05, 0.8, 1.0, 1.0))
    plt.savefig("predicted_chipsize_compare.png")
    plt.show()


    # for i, fitted_m in enumerate(fitted_ms):
    #     plt.plot(nl, [fitted_bs[i][j]/(fitted_m[j]+1e-3) for j in range(len(nl))], '--', label=str(chipsizes[i]))
    # plt.legend()
    # plt.show()

    plt.figure()
    p = plt.subplot(111)
    y_rand = df[rand_solv_str(cs)]
    plt.scatter(nl, y_rand, c=arb_col, label='arbitrary case', alpha=0.4)
    plt.title("solvability of arbitrary netlists on a "+format_chipsize(cs) + " chip")
    p.set_ylabel("solvability %")
    p.set_xlabel("netlist length")
    plt.legend()
    plt.savefig("arbitrary_case_"+format_chipsize(cs)+".png")
    plt.show()

    p = plt.subplot(2,1,2)
    y_mean = df[mean_solv_str(cs)]
    plotscatter(nl, y_mean, c=mean_col, label='average case', alpha=0.4)
    p.set_ylabel("solvability %")
    p.set_xlabel("netlist length")
    p.legend()

    p = plt.subplot(2,1,1)
    y_rand = df[rand_solv_str(cs)]
    plt.scatter(nl, y_rand, c=arb_col, label='arbitrary case', alpha=0.4)
    p.set_ylabel("solvability %")
    plt.suptitle("comparison arbitrary and average case on a " + format_chipsize(cs) + " chip")
    p.legend()
    plt.savefig("comp_arb_avg_case_"+format_chipsize(cs)+".png")
    plt.show()


    p = plt.subplot(111)
    plotscatter(nl, y_mean, c=mean_col, label='average case')
    ABNLfit(nl, y_mean, c=fit, label="predicted \n average case", plot=True)
    p.set_ylabel("solvability %")
    p.set_xlabel("netlist length")
    p.legend(loc=1)
    plt.title("predicting average case solvability on a " + format_chipsize(cs) + " chip")
    plt.savefig("predicted_average_case_"+format_chipsize(cs)+".png")
    plt.show()

    p = plt.subplot(111)
    y_best = df[best_solv_str(200, cs)]
    plotscatter(nl, y_best, c=best_col, label="best case")

    y_rand = df[rand_solv_str(cs)]
    plotscatter(nl, y_mean, c=mean_col, label="average case")
    p.set_ylabel("solvability %")
    p.set_xlabel("netlist length")
    p.legend(loc=1)
    plt.title("avergae and best case solvability")
    plt.savefig("comp_avg_best_case.png")
    plt.show()

    p = plt.subplot(111)
    ABNLfit(nl, y_mean, c=fit, label='predicted \n average case', plot=True)
    ABNLfit(nl, y_best, c=best_fit, label='predicted \n best case', plot=True)
    p.set_ylabel("solvability %")
    p.set_xlabel("netlist length")
    p.set_title("predicted average and best case solvability")
    p.legend(loc=1)
    plt.savefig("exp_avg_best")
    plt.show()
    fit_ab("area", _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)
    fit_ab("edge size", _arb=_arb, _mean=_mean, _best=_best, _worst=_worst)


def fit_ab(chipsizes, _arb=False, _mean=False, _best=False, _worst=False):
    df = load_ab()
    if chipsizes == "area":
        chipsize_col = getdfcol(df,0)**2
    elif chipsizes == "edge size":
        chipsize_col = getdfcol(df,0)

    arbitrary_alpha = getdfcol(df,1)
    best_alpha = getdfcol(df,5)
    mean_alpha = getdfcol(df,3)
    worst_alpha = getdfcol(df,7)
    p = plt.subplot(111)

    if _arb:
        plotscatter(chipsize_col, arbitrary_alpha, c=arb_col, label='real alpha arbitrary')
        aapopt, pcov = curve_fit(logfunc, chipsize_col, arbitrary_alpha, p0=(13, 0.05, 10), bounds=([-300, 0.0001, -max(chipsize_col)], [400, 0.9, max(chipsize_col)]))
        print("alpha arbitrary", aapopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *aapopt), c=fit, label="predicted \n arbitrary case", linestyle="--")

    if _best:
        plotscatter(chipsize_col, best_alpha, c=best_col, label='real alpha best')
        abpopt, pcov = curve_fit(logfunc, chipsize_col, best_alpha, p0=(13, 0.05, 10), bounds=([-300, 0.0001, -max(chipsize_col)], [400, 0.9, max(chipsize_col)]))
        print("alpha best", abpopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *abpopt), c=best_fit, label="predicted \n best case", linestyle="--")

    if _mean:
        plt.scatter(chipsize_col, mean_alpha, c=mean_col, label='real alpha mean')
        ampopt, pcov = curve_fit(logfunc, chipsize_col, mean_alpha, p0=(13, 0.05, 10), bounds=([-300, 0.0001, -max(chipsize_col)], [40000, 0.9, max(chipsize_col)]))
        print("alpha mean", ampopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *ampopt), c=fit, label="predicted \n average case", linestyle='--')

    if _worst:
        plotscatter(chipsize_col, worst_alpha, c=worst_col, label='real alpha worst')
        awpopt, pcov = curve_fit(logfunc, chipsize_col, worst_alpha, p0=(13, 0.05, 10), bounds=([-300, 0.0001, -max(chipsize_col)], [400, 0.9, max(chipsize_col)]))
        print("alpha worst", awpopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *awpopt), c=fit, label="predicted \n worst case", linestyle='--')

    p.set_xlabel("chipsize")
    plt.title("parametrization of alpha for average and best case solvability by "+chipsizes)
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
        plt.plot(chipsize_col, logfunc(chipsize_col, *bapopt), c=fit, label="predicted beta arbitrary", linestyle="--")

    if _best:
        plotscatter(chipsize_col, best_beta, c=best_col, label='real best case')
        bbpopt, pcov = curve_fit(logfunc, chipsize_col, best_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 0.5, 10000]))
        print("beta best", bbpopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *bbpopt), c=best_fit,  label="predicted beta best", linestyle="--")

    if _mean:
        plotscatter(chipsize_col, mean_beta, c=mean_col, label='real mean case')
        bapopt, pcov = curve_fit(logfunc, chipsize_col, mean_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("beta arbitrary", bapopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *bapopt), c=fit, label="predicted beta arbitrary", linestyle="--")

    if _worst:
        plotscatter(chipsize_col, worst_beta, c=worst_col, label='real worst case')
        bbpopt, pcov = curve_fit(logfunc, chipsize_col, worst_beta, p0=(13, 0.005, 10), bounds=([-40, -0.5, -400], [300, 2, 10000]))
        print("beta best", bbpopt)
        plt.plot(chipsize_col, logfunc(chipsize_col, *bbpopt), c=fit,  label="predicted beta worst", linestyle='--')

    p.set_xlabel("chipsize")
    plt.title("parametrization of beta for average and best case solvability by " + chipsizes)
    plt.legend()
    plt.savefig("beta_param_chipsize_" + chipsizes + ".png")
    plt.show()


def plotscatter(nl, solvability, c, label=None, alpha=0.4, s=30):
    plt.scatter(nl, solvability, c=c, label=label, alpha=alpha, s=s)

def ABNLfit(nl, solvability, c, label=None, plot=False):
    popt, pcov = curve_fit(expfunc, nl, solvability, p0=(40, 0.05), bounds=([-100, -2], [200, 1]))
    if plot:
        if label:
            plt.plot(nl, expfunc(nl, *popt), c=c, linestyle='--', label=label)
        else:
            plt.plot(nl, expfunc(nl, *popt), c=c, linestyle='--')
    return popt, pcov



def skiplist(iterable, start, interval=1):
    for i, elem in enumerate(iterable):
        if start <= i:
            if not (i-start) % (interval+1):
                yield elem


def lprint(iterable):
    for elem in iterable:
        print(elem)

def load_ab():
    df = pd.read_csv("params.csv", header=1)
    print(df.head())
    lprint(df.columns)
    return df

def dfcprint(df, c):
    print(df.columns[c])
    print(getdfcol(df,c))


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
    #calc_alpha_beta(1, [(i+2)*10 for i in range(6)])
    calc_alpha_beta(200, [(i+2)*10 for i in range(9)])
