
import numpy as np
from ff2 import load_ab, skiplist, getdfcol

def lprint(itera):
    for e in itera:
        print(e)


def to_reasonable(number):
    if number // 1:
        return str(np.around(number, decimals=1))
    zeros = 0
    nstr = str(number)
    i = 0
    while nstr[i] == "0" or nstr[i] == '.':
        if nstr[i] == "0":
            zeros += 1
        elif nstr[i] == '.':
            pass
        else:
            break
        i += 1

    sigval = np.around(number*10**i, decimals=1)
    return "0." + "0"*(i-1) + "".join([c for c in str(sigval)])



def maketable():
    df = load_ab()
    colinds = [0,3,4,5,6]
    cols = np.array([getdfcol(df, cn) for cn in colinds])
    matrix = np.vectorize(to_reasonable)(cols).T

    linesep = "\\\\\n"
    colnames = [df.columns[i] for i in colinds]
    data = " & ".join(colnames) + linesep + "\\hline\n" + \
        (linesep +"\t").join([" & ".join([c for c in row]) for row in matrix]) + linesep

    start = """
\\begin{center}
\\begin{tabular}{ |""" + " | ".join(["c" for _ in range(cols.shape[0])]) + " |}\n\t" + \
"""\\hline
\\multicolumn{"""+str(cols.shape[0])+"""}{|c|}{parameters for differntly sized chips} \\\\
\\hline
"""


    end = """\\hline
\\end{tabular}
\\end{center}
    """
    return start + data + end

print (maketable())
abtable = open("abtable.txt", "w+")
abtable.write(maketable())
