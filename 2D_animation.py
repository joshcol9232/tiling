import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from multiprocessing import Pipe, Process

DIMS = (1162, 1218)

def stitch_video(fps):
    cmd = "cd frames && ffmpeg -r %d -y -loglevel 24 -s %dx%d -i frame%%d.png -pix_fmt yuv420p output.mp4" % (fps, DIMS[0], DIMS[1])
    print("Stitching video...")
    os.system(cmd)

def run(da, i):
    a = 0.0

    vecs = []
    while a < np.pi * 2.0: # Go around full circle
        vecs.append(np.array([np.cos(a), np.sin(a)]))
        a += da

    vecs = np.array(vecs)
    basis = dg.Basis(vecs, dg.utils.generate_offsets(len(vecs), False, centred=True))

    filt_dist = 11.0
    k_range = max(20//len(vecs), 3)
    print("K RANGE:\t", k_range)

    axis_size = 2 * filt_dist  # Size of matplotlib axis

    cells = dg.dualgrid_method(basis, k_range=k_range)

    G, cells = dg.utils.filtered_graph_from_cells(cells, filter=dg.utils.is_point_within_radius, filter_args=[filt_dist], filter_centre=np.zeros(2))

    fig, ax = plt.subplots(1, figsize=(5, 5))
    ax.axis("equal")

    dg.utils.render_cells_solid(cells, ax, scale=0.85, edge_thickness=0.0, axis_size=axis_size, centre_of_interest=np.zeros(3))

    scale = 2.0
    x_origin = -axis_size + scale + 1.0
    y_origin = -axis_size + scale + 1.0
    plt.quiver([x_origin for _i in range(len(vecs))], [y_origin for _i in range(len(vecs))], vecs[:,0] * scale, vecs[:,1] * scale, width=0.003, scale_units="inches")  # Plot unit vectors

    ax.set_axis_off()
    plt.title("Angle between vecs: %.4f. n: %d." % (da, len(vecs)))
    plt.savefig("frames/frame%d.png" % i, dpi=300, bbox_inches='tight', transparent=False, pad_inches=0)

def generate(smallest_division, num, cpu_cores):
    a_list = np.linspace(np.pi/2.0, np.pi/smallest_division, num=num)
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

generate(15, 512 * 6, 8)
stitch_video(60)
