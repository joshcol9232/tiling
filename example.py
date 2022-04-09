import dualgrid as dg
import matplotlib.pyplot as plt

# Make a Basis object. There are some presets available in the `utils`.
basis_obj = dg.utils.cubic_basis()
# Set up matplotlib axes.
ax = plt.axes(projection="3d")

# Run the algorithm. k_ranges sets the number of construction planes used in the method, but you can also
# put in a 2D array specifying what indices to go through for each plane set if you wish.
# The function outputs:
# `rhombohedra` -> A dictionary of { cell volume: [ generated rhombohedra, ... ], ... }.
# `possible_cells` -> All of the possible cell volumes you can generate with the given basis.
rhombohedra, _possible_cells = dg.dualgrid_method(basis_obj, k_range=2)
print("Generated rhombohedra.")

# Set the filtering distance. In this example we will take a 1x1 cube out of the centre of the
# generated structure.
filt_dist = 2.0

# Render the output in matplotlib. `filter` can be any function that takes a point, centre of interest
# and extra parameters (filter_args), and then outputs a boolean value.
# "ocean" here is the name of the matplotlib colourmap we would like to use.
dg.utils.render_rhombohedra(ax, rhombohedra, "ocean", filter=dg.utils.is_point_within_cube, filter_args=[filt_dist])
plt.show()
