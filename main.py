import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Make a Basis object. There are some presets available in the `utils`.
# basis = dg.utils.ammann_basis()
# basis = dg.utils.penrose_basis()
# basis = dg.utils.surface_with_n_rotsym(11)   # 2D structure with 7-fold rotational symmetry
basis = dg.utils.icosahedral_basis()
# basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure

# Set the filtering distance. In this example we will take a cube out of the centre of the
# generated structure.
# Note that you may want to use smaller numbers here for 3D+ structures as 
# matplotlib starts to struggle with large numbers of shapes.
filt_dist = 2.0
# Set the k range, i.e the number of construction planes used in generating the vertices.
# In 2D this corresponds to having line sets with lines of index -1, 0, 1 for a k range of 2 for example.
# Higher k_range -> more vertices generated.
k_range = 4

# Run the algorithm. k_ranges sets the number of construction planes used in the method.
# The function outputs a list of Cell objects.
cells = dg.dualgrid_method(basis, k_range=k_range)
print("Cells found.\nFiltering & generating graph...")
# Filter the output cells by some function. Pre-defined ones are: is_point_within_cube, is_point_within_radius.
# Then outputs a networkx graph with real space positions and indices of each node embedded.
G, cells = dg.utils.filtered_graph_from_cells(cells, filter=dg.utils.is_point_within_radius, filter_args=[filt_dist])
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
dg.utils.render_graph_wire(G, ax, vert_size=16.0)
# dg.utils.render_cells_solid(cells, ax, scale=1.0/1.618, edge_thickness=0.0)
plt.show()

# Built-in plotting functions in networkx can be used to view the graph form in 2D.
# See networkx's documentation for more
# nx.draw_circular(G)
# plt.show()
