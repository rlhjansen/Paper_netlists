

import os
import matplotlib.pyplot as plt
import imageio
import numpy as np

from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization!

from code.classes.grid import file_to_grid
from code.classes.independent_functions import create_fpath, lprint, get_name_netfile, get_name_circuitfile
import code.algorithms.simplyX as simple


GIF_SUBDIR = "Gifs"
BASE_LAYER_TITLE = ""
RESIZE = 1

# Marker properties on plot




###############################################################################
# Modelling
###############################################################################

def get_markers(n):
    shapes_string = "-"
    SHAPES = shapes_string.split(' ')
    COLOURS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    col_len = len(COLOURS)
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


def plot_circuit(paths, order, gates, x, y, select, save_name, title, resize=1, mesh_height=None):
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
    :param title: creates title, title="placed" makes title into:
        'placement of "+str(len(paths)) + "/" + str(len(order)) + "nets"'
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
    xpp, ypp, zpp = paths_to_plotlines(c_paths)
    plotcount = len(xpp)
    ax.set_title(title)

    # nets
    markers = get_markers(plotcount)
    for i in range(plotcount):
        ax.plot(xpp[i], ypp[i], zpp[i], markers[i], label=c_order[i])

    #gates
    xgs, ygs, zgs = split_gates(gates)
    s = ax.scatter3D(xgs, ygs, zgs, s=resize*9, alpha=1)

    #s.set_edgecolors = s.set_facecolors = lambda *args: None

    #ticks
    ax.set_xticks(np.array([]))
    ax.set_yticks(np.array([]))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zlim((0,7))
    if title == "placed":
        ax.set_title("placement of "+str(len(paths)) + "/" + str(len(order)) + "nets")


    #layer meshes
    if mesh_height=="fitting":
        mesh_fit = get_wire_height(paths)
        ax.set_zticks(np.arange(0, mesh_fit + 1, 1.0))
        add_mesh(ax, mesh_fit+1, x, y)
    elif mesh_height=="full":
        add_mesh(ax, 7, x, y)
        ax.set_zticks(np.arange(0, 8, 1.0))
        ax.set_zlim(np.array([0, mesh_height+1]))
    elif isinstance(mesh_height, int):
        add_mesh(ax, mesh_height, x, y)
        ax.set_zticks(np.arange(0, mesh_height, 1))
    else:
        ax.set_zticks(np.arange(0, 1, 1.0))

    #fig.subplots_adjust(left=0, right=1, bottom=0, top=7)
    plt.draw()

    # manual selection?
    if select:
        plt.show()
    else:
        fig.savefig(save_name)
        plt.close()


def get_wire_height(paths):
    return max([max([node[2] for node in path]) for path in paths])

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
            try:
                paths[i] = (start_gate,) + paths[i][:-1] + (end_gate,)
            except:
                paths[i] = "remove"
        paths =[path for path in paths if path != "remove"]
        build_paths = paths_to_buildcomp(paths)
    else:
        paths = circuit.get_solution_placement(circuit.get_random_net_order())
        build_paths = paths_to_buildcomp(paths)
    _, g_coords = circuit.get_gate_coords()
    return g_coords, paths, build_paths



def create_model(simple_obj, ord, title, select=False, save_name="net model", mesh_height=None):
    simple_obj.circuit.connect()
    circuit = simple_obj.circuit
    if not ord:
        ord = circuit.get_random_net_order()
    g_coords, paths, _ = get_circuit_basics(circuit, ord)
    c_save_name = save_name + ",".join([str(e) for e in circuit.platform_params]) + ".png"
    print(c_save_name)
    plot_circuit(paths, LONG_ORD, g_coords, simple_obj.x, simple_obj.y, select, c_save_name, title, resize=RESIZE, mesh_height=mesh_height)


def create_base_model(simple_obj, title, select=False, save_name="bottom layer model.png", mesh_height=None, resize=1):
    simple_obj.circuit.connect()
    circuit = s.circuit

    g_coords, _, _ = get_circuit_basics(circuit, "")
    paths = []
    plot_circuit([], [], g_coords, simple_obj.x, simple_obj.y, select, save_name, title=title, mesh_height=1, resize=resize)


def create_model_gif(subdir, select=False, height=7, save_name="gif_model_parts", resize=1, redraw=False, del_after=False):

    g_coords, _, build_paths = get_circuit_basics(GRIDNUM, NL_NUM, X, Y, G, LONG_ORD)

    if redraw:
        for i in range(len(build_paths)):

            c_save_name = save_name + str(i).zfill(5) + ".png"
            c_save_name = create_fpath(subdir, c_save_name)
            plot_circuit(build_paths[i], order, g_coords, height, X, Y, select, c_save_name, title="", resize=resize)

    with imageio.get_writer(os.path.join(subdir,'model_gif.gif'), mode='I', subrectangles=True) as writer:
        for filename in os.listdir(subdir):
            print(filename)
            if filename[:9] == "gif_model":
                image = imageio.imread(os.path.join(os.path.join(os.path.curdir, subdir), filename))
                writer.append_data(image)
                if del_after:
                    os.remove(os.path.join(os.path.join(os.path.curdir, subdir),
                                         filename))


# c = 100
# cX = 0
# n = 50
# nX = 15
# x = 60
# y = 60
# LONG_ORD = "n38,n3,n37,n47,n28,n18,n9,n36,n39,n10,n26,n21,n1,n33,n16,n29,n13,n0,n17,n44,n34,n24,n35,n31,n6,n23,n48,n40,n8,n11,n41,n42,n7,n43,n49,n2,n12,n30,n46,n32,n19,n5,n25,n45,n20,n15,n14,n22,n27,n4".split(",")
# s = simple.SIMPLY(c, cX, n, nX, x, y, 'NEIN', iters=1, no_save=True)
# create_model(s, LONG_ORD, "", select=True, save_name="bottom layer model.png")


c = 100
cX = 0
n = 40
nX = 0
x = 30
y = 30
LONG_ORD = "n27,n10,n40,n56,n31,n57,n71,n72,n23,n2,n53,n75,n39,n16,n78,n49,n12,n4,n61,n5,n20,n35,n63,n41,n50,n34,n37,n58,n51,n55,n38,n26,n67,n47,n65,n69,n66,n62,n64,n29,n25,n1,n60,n43,n28,n79,n8,n13,n0,n74,n33,n14,n19,n7,n76,n54,n70,n11,n3,n42,n68,n52,n30,n36,n24,n17,n21,n77,n9,n48,n15,n32,n46,n18,n6,n59,n44,n73,n22,n45".split(",")
s = simple.SIMPLY(c, cX, n, nX, x, y, 'NEIN', iters=1, no_save=True)
# create_model(s, "", "", mesh_height="fitting")


# c = 100
# cX = 0
# n = 40
# nX = 0
# x = 30
# y = 30
# mesh_height = 8
# LONG_ORD = "n4,n14,n17,n10,n9,n5,n13,n15,n16,n1,n2,n11,n12,n19,n6,n8,n7,n18,n0,n3".split(",")
s = simple.SIMPLY(c, cX, n, nX, x, y, 'NEIN', iters=1, no_save=True)
# create_model(s, LONG_ORD, "", mesh_height="fitting")
create_base_model(s, "", save_name="bottom layer model.png", mesh_height="full", resize=5)


""" Selecting netlist, net order, circuit layout (gates) & other parameters

are all done at the top of the file as of now, functions here only take account of what is needed for the drawing.

"""


#create_model_gif("Gifs", ordlength=83, redraw=True, height=MESH_HEIGHT, order=LONG_ORD, resize=RESIZE, del_after=True)
