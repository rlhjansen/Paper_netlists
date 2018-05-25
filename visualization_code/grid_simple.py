from PIL import Image, ImageDraw, ImageFont


from PIL import Image, ImageDraw


def new_base(x,y):
    img = Image.new('RGBA', (x, y))
    return img


def base_grid(image, draw, offset, gline_col=(128, 128, 128, 255)):
    print(image.size)
    n_hlines = int((image.size[1]) / offset)
    start = int(offset/2)
    for i in range(n_hlines):
        draw.line([(0, start + i * offset), (image.size[0], start + i * offset)], fill=gline_col)
    n_vlines = int((image.size[0]) / offset)
    for i in range(n_vlines):
        draw.line([(start + i * offset, 0), (start + i * offset, image.size[0])], fill=gline_col)


def circuit_border(image, draw, width=3, cb_col=(0,0,200,255)):
    border(draw, [0,0,image.size[0], image.size[1]], width=width)


def border(draw, outline, width=2, b_col=(200,0,0,255)):
    draw.line([outline[0]-1, outline[1], outline[0]-1, outline[3]], width=width, fill=b_col)
    draw.line([outline[2], outline[1], outline[2], outline[3]], width=width, fill=b_col)
    draw.line([outline[0], outline[1]-1, outline[2], outline[1]-1], width=width, fill=b_col)
    draw.line([outline[0], outline[3], outline[2], outline[3]], width=width, fill=b_col)


def gate(draw, offset, gatesize, loc, g_num, font, g_col=(255,255,0,255)):
    dif = (offset-gatesize)/2
    gate_outline = [loc[0]*offset + dif,
                    loc[1]*offset + dif,
                    loc[0]*offset + gatesize + dif,
                    loc[1]*offset + gatesize + dif]
    border(draw, gate_outline)
    draw.rectangle(gate_outline, fill=(255,255,255,255))
    draw.text((loc[0]*offset+3/12*offset, loc[1]*offset+1/3*offset), "G"+str(g_num), (0, 0, 0, 255), font=font)


def gates(draw, offset, gatesize, locs, gate_nums, font, g_col=(255,255,0,255)):
    for index, loc in enumerate(locs):
        gate(draw, offset, gatesize, loc, gate_nums[index], font, g_col=g_col)



def net_segment(draw, offset, seg_start, seg_end, width=1, n_col=(50,200,50,255)):
    half_offset = int(offset/2)
    new_start = (seg_start[0]*offset + half_offset, seg_start[1]*offset + half_offset)
    new_end = (seg_end[0]*offset + half_offset, seg_end[1]*offset + half_offset)
    draw.line([new_start, new_end], width=width, fill=n_col)


def net(draw, offset, netlocs, width=1, n_col=(50,200,100,255)):
    for index in range(len(netlocs)-1):
        net_segment(draw, offset, netlocs[index], netlocs[index+1], width=width, n_col=n_col)


def netlist(draw, offset, _netlist, cols, width=1, n_col=(50,200,100,255)):
    for index, _netlocs in enumerate(_netlist):
        net(draw, offset, _netlocs, width=width, n_col=cols[index])


def create_circuit(offset, gatesize, gatelist, gate_nums, gate_font, net_list, net_colors, filename):
    img = new_base(10 * offset, 5 * offset)
    draw = ImageDraw.Draw(img)

    base_grid(img, draw, offset)
    circuit_border(img, draw)
    netlist(draw, offset, net_list, net_colors, width=2)
    gates(draw, offset, gatesize, gatelist, gate_nums, gate_font)

    del draw
    img.save(filename)


OFFSET = 30
GATESIZE = int(2/3*OFFSET)
LETTERSIZE = 2/3*OFFSET

GATE_FONT = ImageFont.truetype("arial.ttf", 10, encoding="unic")



""" Test netlists
GATE_LOCS = [(1,1), (2,2), (4,0), (7,3)]
NET_1 = [(1,1),(2,1),(2,2)]
NET_2 = [(2,2), (3, 2), (4, 2), (4,1), (4,0)]

NETLIST = [NET_1, NET_2]
"""

GATE_FONT = ImageFont.truetype("arial.ttf", 10, encoding="unic")


NET_1 = [(1,1),(2,1),(2,2)]
NET_2 = [(2,2), (3, 2), (4, 2), (4,1), (4,0)]
NETLIST = [NET_1, NET_2]


GATE_LOCS_LONG = [(1,2), (8,2)]
GATE_LOCS_SHORT = [(3,0), (3,4), (6,0), (6,4)]
GATE_LOCS_TOT = GATE_LOCS_LONG + GATE_LOCS_SHORT
GATE_NUMS = [i for i in range(len(GATE_LOCS_TOT))]

NET_COLORS = [(143, 61, 94, 255), (127,133,73,255), (66,79,52,255)]



# Long route
#"""
NETLIST_LONG = [(1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2)]
NETLIST_SHORT_1 = [(3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3), (3,4)]
NETLIST_SHORT_2 = [(6,0), (6,1), (7,1), (8,1), (9,1), (9,2), (9,3), (8,3), (7,3), (6,3), (6,4)]
NETLIST_1 = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

#"""
NETLIST_LONG = [(1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2)]
NETLIST_SHORT_1 = []
NETLIST_SHORT_2 = []
NETLIST_1_1 = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

#"""
NETLIST_LONG = [(1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2)]
NETLIST_SHORT_1 = []
NETLIST_SHORT_2 = [(6,0), (6,1), (7,1), (8,1), (9,1), (9,2), (9,3), (8,3), (7,3), (6,3), (6,4)]
NETLIST_1_1b = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

#"""
NETLIST_LONG = [(1,2), (2,2), (3,2), (4,2), (5,2), (6,2), (7,2), (8,2)]
NETLIST_SHORT_1 = [(3,0), (2,0), (1,0), (0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (3,3), (3,4)]
NETLIST_SHORT_2 = []
NETLIST_1_1a = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""


#"""
NETLIST_LONG = []
NETLIST_SHORT_1 = [(3,0), (3,1), (3,2), (3,4), (3,5)]
NETLIST_SHORT_2 = [(6,0), (6,1), (6,2), (6,3), (6,4)]
NETLIST_2 = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

#"""
NETLIST_LONG = []
NETLIST_SHORT_1 = []
NETLIST_SHORT_2 = [(6,0), (6,1), (6,2), (6,3), (6,4)]
NETLIST_3 = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

#"""
NETLIST_LONG = []
NETLIST_SHORT_1 = [(3,0), (3,1), (3,2), (3,4), (3,5)]
NETLIST_SHORT_2 = []
NETLIST_4 = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

#"""
NETLIST_LONG = []
NETLIST_SHORT_1 = []
NETLIST_SHORT_2 = []
NETLIST_EMPTY = [NETLIST_LONG, NETLIST_SHORT_1, NETLIST_SHORT_2]
#"""

create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_EMPTY, NET_COLORS, 'empty.png')

create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_1, NET_COLORS, 'solution.png')

create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_1_1, NET_COLORS, 'sol_step1.png')
create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_1_1a, NET_COLORS, 'sol_step2a.png')
create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_1_1b, NET_COLORS, 'sol_step2b.png')


create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_2, NET_COLORS, 'false_2.png')
create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_3, NET_COLORS, 'false_1b.png')
create_circuit(OFFSET, GATESIZE, GATE_LOCS_TOT, GATE_NUMS, GATE_FONT, NETLIST_4, NET_COLORS, 'false_1a.png')




