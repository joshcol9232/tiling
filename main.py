import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import sys

R = int(sys.argv[1])
# Make a Basis object. There are some presets available in the `utils`.
basis = dg.utils.surface_with_n_rotsym(R, centred=True, sum_zero=True)   # 2D structure with 11-fold rotational symmetry
# basis = dg.utils.penrose_basis()          # Penrose tiling.
# basis = dg.utils.icosahedral_basis()      # 3D quasicrystalline structure
# basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure

print("OFFSETS:", basis.offsets)

# Set the filtering distance. In this example we will take a sphere out of the centre of the
# generated structure.
filt_dist = int(sys.argv[2])

# Set the k range, i.e the number of construction planes used in generating the vertices.
# In 2D this corresponds to having line sets with lines of index -1, 0, 1 for a k range of 2 for example.
# Higher k_range -> more vertices generated.
k_range = filt_dist+1

# NOTE: It is advised to use smaller numbers here for 3D+ structures as 
# matplotlib starts to struggle with large numbers of shapes. I have
# done an if statement here to change it for 3D+.
if basis.dimensions > 2:
    filt_dist = 2.0
    k_range = 2


# Run the algorithm. k_ranges sets the number of construction planes used in the method.
# The function outputs a list of Cell objects.
cells = dg.dualgrid_method(basis, k_range)
print("Cells found.\nFiltering & generating graph...")
# Filter the output cells by some function. Pre-defined ones are: is_point_within_cube, is_point_within_radius.
# Then outputs a networkx graph with real space positions and indices of each node embedded.

# Filter out a radius
# cells = dg.utils.filter_cells(cells, dg.utils.is_point_within_radius, filter_args=[filt_dist], invert_filter=False, filter_centre=np.zeros(2))

# Get networkx graph of generated structure.
#G = dg.utils.graph_from_cells(cells)

# To filter by highest index allowed (not advisable for 3D):
cells = dg.utils.filter_cells(cells, filter=dg.utils.elements_are_below, filter_args=[filt_dist], filter_indices=True, invert_filter=False)

# Filtering is important so that outliers are not included in the graph.
# e.g tiles that are not connected to the rest of the tiling 
#       - generate a 2D penrose without a filter and zoom out to see for yourself.
# This is one minor caveat of the de Bruijn dualgrid method. Easily remedied by filtering.

# Set up matplotlib axes
if basis.dimensions == 2:
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("equal")
else:
    ax = plt.axes(projection="3d")

# Render the graph using matplotlib. Support for 2D and 3D crystals, 4D and above gets truncated.
# i.e, First 3 elements of vectors are plotted.
if basis.dimensions == 2:   # Fill 2D tiling with colour purely for aesthetics.
    # dg.utils.render_cells_solid(cells, ax, scale=1.0, edge_thickness=1.0, axis_size=10, centre_of_interest=np.zeros(2), colourmap_str="")
    dg.utils.render_cells_solid(cells, ax, scale=0.85, edge_thickness=1.0, axis_size=10, centre_of_interest=np.zeros(2))

    # dg.utils.graph_from_cells(cells) # Uncomment to see graph render.
    # dg.utils.render_graph_wire(G, ax)
else:
    G = dg.utils.graph_from_cells(cells)
    dg.utils.render_graph_wire(G, ax, edge_alpha=1.0)


ax.set_axis_off()
ax.autoscale(enable=True)
plt.savefig("%d-fold_kmax_%d.pdf" % (R, filt_dist), bbox_inches="tight", transparent=True, pad_inches=0)

# Built-in plotting functions in networkx can be used to view the graph form in 2D.
# See networkx's documentation for more. A simple example is below.

# nx.draw_circular(G)
# plt.show()
