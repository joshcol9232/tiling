import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import os
from multiprocessing import Pipe, Process

DIMS = (1550, 1582)

def stitch_video(fps):
    cmd = "cd frames && ffmpeg -r %d -y -loglevel 24 -s %dx%d -i frame%%d.png -pix_fmt yuv420p output.mp4" % (fps, DIMS[0], DIMS[1])
    print("Stitching video...")
    os.system(cmd)

def render_construction(ax, basis, k_range, x_range):
    cols = ["r", "g", "b", "y", "m", "c", "k"]
    x = np.linspace(-x_range, x_range)

    for i, vec in enumerate(basis.vecs):
        for k in range(1-k_range, k_range):
            y = (basis.offsets[i] + k - (vec[0] * x))/vec[1]

            # Check if line is vertical
            if float("inf") in y or float("-inf") in y:
                ax.axvline(x=basis.offsets[i] + k, color=cols[i%len(cols)])
            else:
                ax.plot(x, y, color=cols[i%len(cols)])



def render_cells_at_intersections(
    cells,
    ax,
    colourmap_str="viridis",
    opacity=1.0,
    edge_thickness=1.0,
    edge_colour="k",
    scale=0.2,
    axis_size=5.0,
):
    def make_polygon(cell_verts, scale, intersection):
         # copy to new array in draw-order
        verts = np.array([cell_verts[0], cell_verts[1], cell_verts[3], cell_verts[2]])
        if scale < 1.0:
            for v in verts:
                v -= (v - intersection) * (1.0 - scale)
        
        return Polygon(verts)

    # Group by smallest internal angle. This will serve as the colour index
    INDEX_DECIMALS = 4  # Significant figures used in grouping cells together
    poly_dict = {} # Dictionary of {size index: [matplotlib polygon]}


    for cell_index, c in enumerate(cells):
        # CENTRE CELLS ON INTERSECTION
        middle = np.mean(c.verts, axis=0)
        diff = c.intersection - middle
        c.verts += diff

        size_ratio = np.around(abs(np.dot(c.verts[0] - c.verts[1], c.verts[0] - c.verts[2])), decimals=4)
        p = make_polygon(c.verts, scale, c.intersection)
        if size_ratio not in poly_dict:
            poly_dict[size_ratio] = [p]
        else:
            poly_dict[size_ratio].append(p)

    # Render
    if colourmap_str == "":
        clrmap = lambda s: "w"
    else:
        clrmap = cm.get_cmap(colourmap_str)

    for size_ratio, polygons in poly_dict.items():
        colour = clrmap(size_ratio)
        shape_coll = PatchCollection(polygons, edgecolor=edge_colour, facecolor=colour, linewidth=edge_thickness, antialiased=True)
        ax.add_collection(shape_coll)


    plt.xlim(-axis_size, axis_size)
    plt.ylim(-axis_size, axis_size)
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio

def run(da, i):
    a = 0.0

    vecs = []
    while a < np.pi * 2.0: # Go around full circle
        vecs.append(np.array([np.cos(a), np.sin(a)]))
        a += da

    vecs = np.array(vecs)
    basis = dg.Basis(vecs, dg.utils.generate_offsets(len(vecs), False, centred=True))

    k_range = 2
    axis_size = 2

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("equal")
    render_construction(ax, basis, k_range, axis_size)

    cells = dg.dualgrid_method(basis, k_range)
    render_cells_at_intersections(cells, ax, scale=0.06, axis_size=axis_size)

    """
    scale = 2.0
    x_origin = -axis_size + scale + 1.0
    y_origin = -axis_size + scale + 1.0
    plt.quiver([x_origin for _i in range(len(vecs))], [y_origin for _i in range(len(vecs))], vecs[:,0] * scale, vecs[:,1] * scale, width=0.003, scale_units="inches")  # Plot unit vectors
    """

    ax.set_axis_off()
    plt.title("Angle between vecs: %.4f. n: %d." % (da, len(vecs)))
    plt.savefig("frames/frame%d.png" % i, dpi=200, bbox_inches='tight', transparent=False, pad_inches=0)

def generate(smallest_division, num, cpu_cores):
    a_list = np.linspace(np.pi + 0.01, np.pi/smallest_division, num=num)
    print("Total angles:\t", len(a_list))


    for i in range(0, len(a_list), cpu_cores):
        processes = []
        for j, da in enumerate(a_list[i:i + cpu_cores]):
            p = Process(target=run, args=(da, i+j))
            p.start()
            processes.append(p)

        for k, p in enumerate(processes):
            p.join()
            print("Frame done:", k + i)

FPS = 30
NUM = FPS * 10
generate(7, NUM, 4)
stitch_video(FPS)
