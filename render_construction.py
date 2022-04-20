import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import sys

R = int(sys.argv[1])
# Make a Basis object. There are some presets available in the `utils`.
basis = dg.utils.surface_with_n_rotsym(R, centred=True)   # 2D structure with 11-fold rotational symmetry
# basis = dg.utils.penrose_basis()          # Penrose tiling.
# basis = dg.utils.icosahedral_basis()      # 3D quasicrystalline structure
# basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure

print("OFFSETS:", basis.offsets)

def render_construction(ax, k_range, x_range):
    x = np.linspace(-x_range, x_range)

    for i, vec in enumerate(basis.vecs):
        for k in range(1-k_range, k_range):
            y = (basis.offsets[i] + k - (vec[0] * x))/vec[1]

            # Check if line is vertical
            if float("inf") in y or float("-inf") in y:
                print("VERTICAL LINE AT:", basis.offsets[i] + k)
                ax.axvline(x=basis.offsets[i] + k)
            else:
                ax.plot(x, y)

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.axis("equal")
k_range = 2
render_construction(ax, k_range, 5)


cells = dg.dualgrid_method(basis, k_range)
print("Cells found.")

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


render_cells_at_intersections(cells, ax, scale=0.1, axis_size=2)

ax.set_axis_off()
# plt.savefig("%d-fold_kmax_%d.pdf" % (R, filt_dist), bbox_inches="tight", transparent=True, pad_inches=0)
plt.show()

# Built-in plotting functions in networkx can be used to view the graph form in 2D.
# See networkx's documentation for more. A simple example is below.

# nx.draw_circular(G)
# plt.show()
