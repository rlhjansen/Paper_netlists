from random import sample
from math import floor

def swap_two_X_times(net_path, X):
    npath = list(net_path)[:]
    end_index = len(npath) - 1
    swaps = sample([i for i in range(end_index)], X*2)
    for i in range(X):
        npath[swaps[2*i]], npath[swaps[2*i+1]] = npath[swaps[2*i+1]], npath[swaps[2*i]]
    return tuple(npath)


def combine_score(connections, length, scoring, total_nets):
    """ combines cnnections & length components into a single score

    scoring='div' ==> connections, length --> connections + 1/length
    """
    if scoring == "div":
        frac_part = 1.0/float(length)
        return float(connections)+frac_part
    elif scoring == "percdiv":
        frac_part = 1.0/float(length)
        return float(connections)/float(total_nets) + frac_part


def split_score(combination, scoring="div"):
    """ splits score into connections & length component

    scoring='div' ==> connections + 1/length --> connections, length
    """
    if scoring == "div":
        connections = floor(combination)
        length = 1.0/(combination - float(connections))
        return connections, length
