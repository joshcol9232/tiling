import dualgrid as dg
import matplotlib.pyplot as plt


# Make a Basis object. There are some presets available in the `utils`.
# basis = dg.utils.ammann_basis()
basis = dg.utils.surface_with_n_rotsym(7)
# basis = dg.utils.icosahedral_basis()

# Set the filtering distance. In this example we will take a 1x1 cube out of the centre of the
# generated structure.
filt_dist = 2.0

# Run the algorithm. k_ranges sets the number of construction planes used in the method.
# The function outputs a list of Cell objects.
cells = dg.dualgrid_method(basis, k_range=2)
G = dg.utils.graph_from_cells(cells, filter=dg.utils.is_point_within_radius, filter_args=[filt_dist], fast_filter=False)
print("Generated graph.")

dg.utils.render_graph(G)
plt.show()

mesh = dg.utils.generate_wire_mesh(G, verbose=True)
mesh.write("7fold.stl")
