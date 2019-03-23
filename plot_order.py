

import os
import matplotlib.pyplot as plt
import imageio
import numpy as np

from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization!

from code.classes.grid import file_to_grid
from code.classes.independent_functions import create_fpath, lprint, get_name_netfile, get_name_circuitfile
import code.algorithms.simplyX as simple

GIF_SUBDIR = "Gifs"
c = 100
cX = 0
n = 45
nX = 10
x = 90
y = 90
MESH_HEIGHT = 7
LONG_ORD = "n16,n38,n14,n2,n37,n43,n3,n0,n6,n4,n34,n41,n26,n13,n31,n17,n33,n28,n11,n9,n22,n15,n32,n8,n29,n7,n10,n21,n25,n12,n23,n5,n24,n30,n1,n42,n36,n19,n39,n35,n20,n27,n44,n18,n40".split(",")
BASE_LAYER_TITLE = ""
RESIZE = 1

# Marker properties on plot
shapes_string = "-"
SHAPES = shapes_string.split(' ')
COLOURS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

col_len = len(COLOURS)



###############################################################################
# Modelling
###############################################################################

def get_markers(n):
    return [SHAPES[i % len(SHAPES)] + COLOURS[i % col_len] for i in range(n)]


def paths_to_plotlines(paths):
    """ transforms paths format from the grid to a series of plottable points

    :param paths: list of tuples of tuples
        each outer tuple represents a path for how a certain net is laid
        each inner tuple represents a specific (x,y,z) location on the circuit
    :return xpp, ypp, zpp: multiple list of lists where the inner list is a
        series of points to be plotted for a single netlist (on x,y,z axis
        respectively)
    """
    xpp = []
    ypp = []
    zpp = []

    for path in paths:
        pxs = [spot[0] for spot in path]
        pys = [spot[1] for spot in path]
        pzs = [spot[2] for spot in path]

        xpp.append(pxs)
        ypp.append(pys)
        zpp.append(pzs)
    return xpp, ypp, zpp


def remove_empty_paths(paths, order):
    """ Removes empty paths before plotting.

    :param paths: paths taken from grid.
    :param order: list of accompanying net names in congruent order.
    :return: same paths & order, but with empty paths removed.
    """
    clean_paths = []
    clean_order = []
    print(paths)
    for i, path in enumerate(paths):
        if len(path)-1:
            clean_paths.append(path)
            clean_order.append(order[i])
    return clean_paths, clean_order


def plot_circuit(paths, order, gates, mesh_height, x, y, select, save_name, title=None, resize=1):
    """ Plots the complete circuit in 3D, filled by way of the paths.

    :param paths: List of tuples of tuples
        Each outer tuple represents a path for how a certain net is laid
        Each inner tuple represents a specific (x,y,z) location on the circuit
    :param order: List, instance of an order of netlists
    :param gates: List of tuples, each representing a gate on the circuit
    :param mesh_height: Int height unto which meshes will be drawn (empty layers)
    :param select: Bool, True to manually select direction from which to view
        the plotted circuit, False for autoview-save.
    :param save_name: filename for autoview-save
    :param title: creates title
    :param resize: resizes the plot when saving (make lower when experimenting
        to save time, e.g. 0.3)
    """

    # init figure
    fig = plt.figure()
    fig_size = fig.get_size_inches()
    fig.set_size_inches(fig_size*resize)
    ax = Axes3D(fig)

    #title & nets
    c_paths, c_order = remove_empty_paths(paths, order)
    print(c_paths, c_order)
    xpp, ypp, zpp = paths_to_plotlines(c_paths)
    plotcount = len(xpp)
    ax.set_title(title)

    # nets
    markers = get_markers(len(xpp))
    for i in range(plotcount):
        ax.plot(xpp[i], ypp[i], zpp[i], markers[i], label=c_order[i])

    #gates
    xgs, ygs, zgs = split_gates(gates)
    s = ax.scatter3D(xgs, ygs, zgs, s=resize)

    #s.set_edgecolors = s.set_facecolors = lambda *args: None

    #ticks
    ax.set_zticks(np.arange(0, mesh_height + 1, 1.0))
    ax.set_xticks(np.array([]))
    ax.set_yticks(np.array([]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #layer meshes
    if mesh_height:
        add_mesh(ax, mesh_height, x, y)
    else:
        add_mesh(ax, mesh_height+1, x, y)
        ax.set_zlim(np.array([0, mesh_height+1]))

    #fig.subplots_adjust(left=0, right=1, bottom=0, top=7)
    plt.draw()

    # manual selection?
    if select:
        plt.show()
    else:
        fig.savefig(save_name)
        plt.close()


def add_mesh(ax, h, x, y):
    for height in range(h):
        # for mesh overlay
        xint = np.array([i for i in range(x)])
        yint = np.array([i for i in range(y)])
        zint = np.array([height for _ in range(x)])
        X, Y = np.meshgrid(xint, yint)
        Z, _ = np.meshgrid(zint, yint)
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, linewidth=0.5/(h+1),
                          color=(0, 0, 0))


def create_plot_title(placed):
    return str(placed) + " nets placed"


def split_gates(gates):
    """

    :param gates: list of 3D vertices (tuples)
    :return: lists of their components
    """
    xgs = [gate[0] for gate in gates]
    ygs = [gate[1] for gate in gates]
    zgs = [gate[2] for gate in gates]
    return xgs, ygs, zgs


def paths_to_buildcomp(paths):
    """ Splits tuple of all paths into a list of parts, which consist of all

    Wirepaths up untill i for index i.
    e.g. (with P1 and P2 both having 3 elements):
    (P1, P2,) --> [((P11),) , ((P11, P12),) , ((P11, P12, P13),) ,
        ((P11, P12, P13), (P21),) , ((P11, P12, P13), (P21, P22),) ,
        ((P11, P12, P13), (P21, P22, P23),)]

    :param paths: Tuple of paths like (P1, P2,)
    :return: List of parts like above
    """
    totlen = sum([len(elem) for elem in paths])
    tot_path = []
    for i in range(totlen):
        leftover_i = i
        temp_path = []
        for path in paths:
            if leftover_i > len(path):
                temp_path.append(path)
                leftover_i -= len(path)
            else:
                temp_path.append(path[:leftover_i])
                break
        tot_path.append(temp_path)
    return tot_path


def get_circuit_basics(circuit, order):
    """ Returns necessities for plotting a laid circuit with certain order

    :param circuit: circuit instance
    :param order: netlist order, if None then random.
    :return: gate coordinates, paths, partitioned paths
    """
    print(order)
    if order:
        print("max g", circuit.max_g)
        paths = circuit.solve_order_paths(order)[-1]
        for i, net in enumerate(order):
            gates = circuit.net_gate.get(net)
            start_gate = circuit.gate_coords.get(gates[0])
            end_gate = circuit.gate_coords.get(gates[1])
            paths[i] = (start_gate,) + paths[i][:-1] + (end_gate,)

        build_paths = paths_to_buildcomp(paths)
    else:
        paths = circuit.get_solution_placement(G.get_random_net_order())
        build_paths = paths_to_buildcomp(paths)
    _, g_coords = circuit.get_gate_coords()
    return g_coords, paths, build_paths


###############################################################################
# Model types
###############################################################################
""" Selecting netlist, net order, circuit layout (gates) & other parameters

are all done at the top of the file as of now, functions here only take account of what is needed for the drawing.

I REPEAT: GLOBAL VARIABLES GALORE!
"""


def create_model(titlestring="", height=7, select=False, save_name="net model"):
    s = simple.SIMPLY(c, cX, n, nX, x, y, 'NEIN', iters=1, no_save=True)
    s.circuit.connect()
    circuit = s.circuit

    g_coords, paths, _ = get_circuit_basics(circuit, LONG_ORD)
    c_save_name = save_name + ",".join([str(x),str(n),str(nX)]) + ".png"
    plot_circuit(paths, LONG_ORD, g_coords, height, x, y, select, c_save_name, title=titlestring, resize=RESIZE)


def create_base_model(titlestring, select=False, save_name="bottom layer model.png"):
    s = simple.SIMPLY(c, cX, n, nX, x, y, 'NEIN', iters=1, no_save=True)
    s.circuit.connect()
    circuit = s.circuit

    g_coords, paths, _ = get_circuit_basics(circuit, LONG_ORD)
    plot_circuit([], [], g_coords, False, select, X, Y, save_name, title=titlestring)


def create_model_gif(subdir, select=False, height=7, save_name="gif_model_parts", resize=1, redraw=False, del_after=False):

    g_coords, _, build_paths = get_circuit_basics(GRIDNUM, NL_NUM, X, Y, G, LONG_ORD)

    if redraw:
        for i in range(len(build_paths)):

            c_save_name = save_name + str(i).zfill(5) + ".png"
            c_save_name = create_fpath(subdir, c_save_name)
            plot_circuit(build_paths[i], order, g_coords, height, X, Y, select, c_save_name, title=" ", resize=resize)

    with imageio.get_writer(os.path.join(subdir,'model_gif.gif'), mode='I', subrectangles=True) as writer:
        for filename in os.listdir(subdir):
            print(filename)
            if filename[:9] == "gif_model":
                image = imageio.imread(os.path.join(os.path.join(os.path.curdir, subdir), filename))
                writer.append_data(image)
                if del_after:
                    os.remove(os.path.join(os.path.join(os.path.curdir, subdir),
                                         filename))




create_model("", select=True, save_name="bottom layer model.png")


""" Selecting netlist, net order, circuit layout (gates) & other parameters

are all done at the top of the file as of now, functions here only take account of what is needed for the drawing.

I REPEAT: WHEN CALLING THESE FUNCTIONS, GLOBAL VARIABLES GALORE!
"""


#create_model_gif("Gifs", ordlength=83, redraw=True, height=MESH_HEIGHT, order=LONG_ORD, resize=RESIZE, del_after=True)
