

import os
import matplotlib.pyplot as plt
import imageio
import numpy as np

from mpl_toolkits.mplot3d import Axes3D #<-- Note the capitalization!

from Gridfile import file_to_grid
from independent_functions import create_fpath, lprint, get_name_netfile, get_name_circuitfile

CIRCUIT_SUBDIR = "circuit_map_100"
GIF_SUBDIR = "Gifs"
GRIDNUM = 0
NL_NUM = 1
X = 30
Y = 30
G = 100
NETL_LEN = 100
MESH_HEIGHT = 5
NET_MAX = 20
LONG_ORD = ['n75', 'n73', 'n53', 'n31', 'n83', 'n41', 'n30', 'n67', 'n60', 'n77', 'n57', 'n70', 'n16', 'n26', 'n76', 'n56', 'n64', 'n63', 'n90', 'n65', 'n45', 'n10', 'n86', 'n87', 'n20', 'n55', 'n69', 'n18', 'n2', 'n71', 'n61', 'n43', 'n47', 'n23', 'n93', 'n68', 'n34', 'n13', 'n52', 'n84', 'n94', 'n8', 'n78', 'n66', 'n79', 'n39', 'n54', 'n9', 'n85', 'n44', 'n4', 'n37', 'n3', 'n92', 'n82', 'n0', 'n59', 'n74', 'n81', 'n72', 'n11', 'n42', 'n80', 'n99', 'n17', 'n21', 'n7', 'n98', 'n1', 'n96', 'n49', 'n95', 'n91', 'n35', 'n62', 'n25', 'n88', 'n14', 'n28', 'n51', 'n24', 'n50', 'n89']
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
    for i, path in enumerate(paths):
        if len(path)-1:
            clean_paths.append(path)
            clean_order.append(order[i])
    return clean_paths, clean_order


def plot_circuit(paths, order, gates, mesh_height, x, y, select, save_name, alt_title=None, resize=1):
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
    :param alt_title: creates title "X nets placed" in cae no alternative is given
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
    if not alt_title:
        ax.set_title(create_plot_title(plotcount))
    else:
        ax.set_title(alt_title)

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


def get_circuit_basics(grid_num, net_num, x, y, g, order, ordlength=-1):
    """ Returns necessities for plotting a laid circuit with certain order

    :param grid_num: grid number
    :param net_num: netlist number
    :param x: x dimension of circuit
    :param y: y dimension of circuit
    :param g: gates of circuit
    :param order: netlist order, if None then random.
    :return: gate coordinates, paths, partitioned paths
    """
    net_file = get_name_netfile(grid_num, net_num)
    grid_file = get_name_circuitfile(grid_num, x, y, g)
    grid_file = create_fpath(CIRCUIT_SUBDIR, grid_file)

    G = file_to_grid(grid_file, None)
    G.read_nets(CIRCUIT_SUBDIR, net_file)
    if order:
        paths = G.get_solution_placement(order[:ordlength])
        build_paths = paths_to_buildcomp(paths)
    else:
        paths = G.get_solution_placement(G.get_random_net_order())
        build_paths = paths_to_buildcomp(paths)
    _, g_coords = G.get_gate_coords()
    return g_coords, paths, build_paths


###############################################################################
# Model types
###############################################################################


def create_model(titlestring="", height=None, order=None, ordlength=None, select=False, save_name="net model"):
    g_coords, paths, _ = get_circuit_basics(GRIDNUM, NL_NUM, X, Y, G, order, ordlength=ordlength)
    c_save_name = save_name + str(ordlength) + ".png"
    plot_circuit(paths, order, g_coords, height, X, Y, select, c_save_name, alt_title=titlestring, resize=RESIZE)


def create_base_model(titlestring, select=False, save_name="bottom layer model.png"):
    g_coords, _, _= get_circuit_basics(GRIDNUM, NL_NUM, X,
                                                           Y, G, [])
    plot_circuit([], [], g_coords, False, select, X, Y, save_name, alt_title=titlestring)


def create_model_gif(subdir, select=False, height=None, order=None, ordlength=None, save_name="gif_model_parts", resize=1, redraw=False, del_after=False):

    g_coords, _, build_paths = get_circuit_basics(GRIDNUM, NL_NUM, X, Y, G, order)

    if redraw:
        for i in range(len(build_paths)):

            c_save_name = save_name + str(i).zfill(5) + ".png"
            c_save_name = create_fpath(subdir, c_save_name)

            plot_circuit(build_paths[i], order, g_coords, height, X, Y, select, c_save_name, alt_title=" ", resize=resize)

    with imageio.get_writer(os.path.join(subdir,'model_gif.gif'), mode='I', subrectangles=True) as writer:
        for filename in os.listdir(subdir):
            print(filename)
            if filename[:9] == "gif_model":
                image = imageio.imread(os.path.join(os.path.join(os.path.curdir, subdir), filename))
                writer.append_data(image)
                if del_after:
                    os.remove(os.path.join(os.path.join(os.path.curdir, subdir),
                                         filename))


create_base_model(" ")
create_model(" ",height=5, order=LONG_ORD,ordlength=30)
#create_model_gif("Gifs", ordlength=83, redraw=True, height=MESH_HEIGHT, order=LONG_ORD, resize=RESIZE, del_after=True)
