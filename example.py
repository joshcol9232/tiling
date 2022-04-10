import dualgrid as dg
import matplotlib.pyplot as plt

# Make a Basis object. There are some presets available in the `utils`.
basis_obj = dg.utils.icosahedral_basis()
# Set up matplotlib axes.
ax = plt.axes(projection="3d")

# Set the filtering distance. In this example we will take a 1x1 cube out of the centre of the
# generated structure.
filt_dist = 2.0

# Run the algorithm. k_ranges sets the number of construction planes used in the method, but you can also
# put in a 2D array specifying what indices to go through for each plane set if you wish.
# The function outputs:
# `rhombohedra` -> A dictionary of { cell volume: [ generated rhombohedra, ... ], ... }.
# `possible_cells` -> All of the possible cell volumes you can generate with the given basis.
cell_dict, _possible_cells, _offsets = dg.dualgrid_method(basis_obj, k_range=2, old=False)
verts, edges = dg.utils.verts_and_edges_from_cells(cell_dict, filter=dg.utils.is_point_within_cube, filter_args=[filt_dist])

print("Generated rhombohedra.")

# Render the output in matplotlib. `filter` can be any function that takes a point, centre of interest
# and extra parameters (filter_args), and then outputs a boolean value.
# "ocean" here is the name of the matplotlib colourmap we would like to use.

dg.utils.render_cells(ax, cell_dict, "ocean", filter=dg.utils.is_point_within_cube, filter_args=[filt_dist], shape_opacity=0.6)

# dg.utils.render_verts_and_edges(ax, new_verts, new_edges)


# TODO: Make core function output vertices, edges and then edge groups, i.e rhombs in the form of edge indices.

plt.show()
