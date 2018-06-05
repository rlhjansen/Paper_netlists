

import func_objects as fo
from independent_functions import plot_circuit
from Gridfile import Grid, SRC



SUBDIR = "circuit_map_100"
GRIDNUM = 0
NETLISTNUM = 1
X = 30
Y = 30
G = 100
NETL_LEN = 100
MESH_HEIGHT = 3
NET_MAX = 20
LONG_ORD = ['n75', 'n73', 'n53', 'n31', 'n83', 'n41', 'n30', 'n67', 'n60', 'n77', 'n57', 'n70', 'n16', 'n26', 'n76', 'n56', 'n64', 'n63', 'n90', 'n65', 'n45', 'n10', 'n86', 'n87', 'n20', 'n55', 'n69', 'n18', 'n2', 'n71', 'n61', 'n43', 'n47', 'n23', 'n93', 'n68', 'n34', 'n13', 'n52', 'n84', 'n94', 'n8', 'n78', 'n66', 'n79', 'n39', 'n54', 'n9', 'n85', 'n44', 'n4', 'n37', 'n3', 'n92', 'n82', 'n0', 'n59', 'n74', 'n81', 'n72', 'n11', 'n42', 'n80', 'n99', 'n17', 'n21', 'n7', 'n98', 'n1', 'n96', 'n49', 'n95', 'n91', 'n35', 'n62', 'n25', 'n88', 'n14', 'n28', 'n51', 'n24', 'n50', 'n89']
BASE_LAYER_TITLE = ""

def create_model(titlestring="", height=None, order=None, ordlength=None, select=False, save_name="net model"):
    rc = fo.RC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G,["test_vis"],1,batch_size=1, ask=False)
    if not order:
        order = rc.G.get_random_net_order()
    print(order)
    if ordlength:
        order = order[:ordlength]
    print(len(order), order)
    paths = rc.G.get_solution_placement(order)
    g_names, g_coords = rc.G.get_gate_coords()

    c_save_name = save_name + str(ordlength) + ".png"
    plot_circuit(paths, order, g_coords, g_names, height, select, c_save_name, alt_title=titlestring)


def create_base_model(titlestring, select=False, save_name="bottom layer model.png"):
    rc = fo.RC(SUBDIR, GRIDNUM, NETLISTNUM, X, Y, G, ["test_vis"], 1,
               batch_size=1, ask=False)
    order = rc.G.get_random_net_order()
    order = order[:0]
    paths = rc.G.get_solution_placement(order)
    g_names, g_coords = rc.G.get_gate_coords()

    plot_circuit(paths, order, g_coords, g_names, False, select, save_name, alt_title=titlestring)


create_model(" ", ordlength=20, order=LONG_ORD, select=True)
#create_base_model(" ", select=True)