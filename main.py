import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Make a Basis object. There are some presets available in the `utils`.
R = 5
basis = dg.utils.surface_with_n_rotsym(R, centred=True)   # 2D structure with 11-fold rotational symmetry
# basis = dg.utils.penrose_basis()          # Penrose tiling.
# basis = dg.utils.icosahedral_basis()      # 3D quasicrystalline structure
# basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure

print("OFFSETS:", basis.offsets)

# Set the filtering distance. In this example we will take a sphere out of the centre of the
# generated structure.
filt_dist = 4

# Set the k range, i.e the number of construction planes used in generating the vertices.
# In 2D this corresponds to having line sets with lines of index -1, 0, 1 for a k range of 2 for example.
# Higher k_range -> more vertices generated.
k_range = 8

axis_size = 20  # Size of matplotlib axis

# NOTE: It is advised to use smaller numbers here for 3D+ structures as 
# matplotlib starts to struggle with large numbers of shapes. I have
# done an if statement here to change it for 3D+.
if basis.dimensions > 2:
    filt_dist = 2.0
    k_range = 2
    axis_size = 3

# Run the algorithm. k_ranges sets the number of construction planes used in the method.
# The function outputs a list of Cell objects.
cells = dg.dualgrid_method(basis, k_range=k_range)
print("Cells found.\nFiltering & generating graph...")
# Filter the output cells by some function. Pre-defined ones are: is_point_within_cube, is_point_within_radius.
# Then outputs a networkx graph with real space positions and indices of each node embedded.
G, cells = dg.utils.filtered_graph_from_cells(cells, filter=dg.utils.contains_value, filter_args=[filt_dist], filter_indices=True, invert_filter=True)
# Filtering is important so that outliers are not included in the graph.
# e.g tiles that are not connected to the rest of the tiling 
#       - generate a 2D penrose without a filter and zoom out to see for yourself.
# This is one minor caveat of the de Bruijn dualgrid method. Easily remedied by filtering.

print("Graph generated..")

# Set up matplotlib axes
if basis.dimensions == 2:
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("equal")
else:
    ax = plt.axes(projection="3d")

# Render the graph using matplotlib. Support for 2D and 3D crystals, 4D and above gets truncated.
# i.e, First 3 elements of vectors are plotted.
if basis.dimensions == 2:   # Fill 2D tiling with colour purely for aesthetics.
    dg.utils.render_cells_solid(cells, ax, scale=0.85, edge_thickness=0.0, axis_size=axis_size)
    # dg.utils.render_graph_wire(G, ax) # Uncomment to see graph render.
else:
    dg.utils.render_cells_solid(cells, ax, axis_size=axis_size)
    # dg.utils.render_graph_wire(G, ax, edge_alpha=1.0)


# plt.title("Change basis in main.py to see other examples.")
# plt.show()

# Built-in plotting functions in networkx can be used to view the graph form in 2D.
# See networkx's documentation for more. A simple example is below.

# nx.draw_circular(G)
# plt.show()

ax.set_axis_off()
plt.title("%d-Fold rotationally symmetrical tiling." % R)
# plt.title("Icosahedral basis centred.")
plt.show()
# plt.savefig("%d-fold_output.png" % R, dpi=400, bbox_inches='tight', transparent=True, pad_inches=0)
# plt.savefig("icosahedral_quasi_output.png", dpi=400, bbox_inches='tight', transparent=True, pad_inches=0)