import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import meshgen

# Make a Basis object. There are some presets available in the `utils`.
# basis = dg.utils.surface_with_n_rotsym(11, centred=True)   # 2D structure with 11-fold rotational symmetry
basis = dg.utils.penrose_basis()          # Section of Penrose tiling.
# basis = dg.utils.icosahedral_basis()      # 3D quasicrystalline structure
# basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure

print("OFFSETS:", basis.offsets)
G = None

# Set the k range, i.e the number of construction planes used in generating the vertices.
# In 2D this corresponds to having line sets with lines of index -1, 0, 1 for a k range of 2 for example.
# Higher k_range -> more vertices generated.
# The results will later be filtered to remove outliers.
k_range = 2

# NOTE: It is advised to use a smaller k_range for 3D+ structures as
# matplotlib starts to struggle with large numbers of shapes. I have
# done an if statement here to change it for 3D+.
if basis.dimensions > 2:
    k_range = 3

# Run the algorithm. k_ranges sets the number of construction planes used in the method.
# The function outputs a list of Cell objects.
cells = dg.dualgrid_method(basis, k_range)
print("Cells found.\nFiltering...")
# Filter the output cells by some function. Pre-defined ones are: is_point_within_cube, is_point_within_radius, elements_are_below, contains_value. Each one can be toggled
# to use the real space positions of vertices, or their indices in grid space.


# To filter by highest index allowed (good for 2D, odd N-fold tilings):
# cells = dg.utils.filter_cells(cells, filter=dg.utils.elements_are_below, filter_args=[max(k_range-1, 0)], filter_indices=True, invert_filter=False)

# To filter out a radius of R:
R = 11
if basis.dimensions != 2:
    R = 2 # Reduce for 3D+ to reduce lag

cells = dg.utils.filter_cells(cells, filter=dg.utils.is_point_within_radius, filter_args=[R])

print("Cells filtered.")

# To get networkx graph of generated structure:
#G = dg.utils.graph_from_cells(cells)

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
    # dg.utils.render_cells_solid(cells, ax, scale=0.85, edge_thickness=0.0)
    # print(cells)

    G = dg.utils.graph_from_cells([cells[0]])
    print("Nodes:", G.nodes[0], G.adj)
    dg.utils.save_graph_to_file(G, "graph_out.stl", 0.1)

    # dg.utils.graph_from_cells(cells) # Uncomment to see graph render.
    dg.utils.render_graph_wire(G, ax)
    ax.autoscale(enable=True)  # Zoom out to fit whole tiling
    ax.set_axis_off()



elif basis.dimensions == 3:
    dg.utils.render_cells_solid(cells, ax)
    ax.autoscale(enable=True)
else:
    print("Generating graph...")
    G = dg.utils.graph_from_cells(cells)
    # We find edges to draw in the process of making G
    dg.utils.render_graph_wire(G, ax, edge_alpha=1.0)

plt.title("Change basis in main.py to see other examples.")
plt.show()

# Built-in plotting functions in networkx can be used to view the graph form in 2D.
# See networkx's documentation for more. A simple example is below.

# if type(G) == type(None):
#     print("Generating graph.")
#     G = dg.utils.graph_from_cells(cells)
# nx.draw_circular(G)
# plt.show()

# Example of generating wireframe:
"""
if type(G) == type(None):
    print("Generating graph.")
    G = dg.utils.graph_from_cells(cells)

# Generate the wireframe:
wireframe = dg.utils.generate_wires(G)
"""
# print(wireframe)

# Example of generating an STL wireframe mesh. NOTE: This can take a long time, depending on node count etc.
# Recommended to use a low k_range value (defined above).
"""
mesh = dg.utils.generate_wire_mesh_stl(G, verbose=True)
mesh.write("G.stl")
"""
