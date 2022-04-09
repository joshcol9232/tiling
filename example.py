import dualgrid as dg
import matplotlib.pyplot as plt

# Make a Basis object. There are some presets available in the `utils`.
basis_obj = dg.utils.hexagonal_basis()
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

old_rhombohedra, _possible_cells, old_intersections, offsets = dg.dualgrid_method(basis_obj, k_range=2, old=True)
old_verts, old_edges = dg.utils.verts_and_edges_from_rhombs(old_rhombohedra, filter=dg.utils.is_point_within_cube, filter_args=[filt_dist])
print("Generated old rhombohedra.")

new_rhombohedra, _possible_cells, new_intersections, _offsets = dg.dualgrid_method(basis_obj, k_range=2, old=False, offsets=offsets)
new_verts, new_edges = dg.utils.verts_and_edges_from_rhombs(new_rhombohedra, filter=dg.utils.is_point_within_cube, filter_args=[filt_dist])

print("Generated rhombohedra.")

print("OLD INTERSECTIONS:", old_intersections)
print("NEW INTERSECTIONS:", new_intersections)

# Render the output in matplotlib. `filter` can be any function that takes a point, centre of interest
# and extra parameters (filter_args), and then outputs a boolean value.
# "ocean" here is the name of the matplotlib colourmap we would like to use.

# dg.utils.render_rhombohedra(ax, old_rhombohedra, "Greys", filter=dg.utils.is_point_within_cube, filter_args=[filt_dist], shape_opacity=0.3)

# dg.utils.render_verts_and_edges(ax, new_verts, new_edges)

ax.plot(old_intersections[:,0], old_intersections[:,1], old_intersections[:,2], "k.", markersize=20.0, label="Old")
ax.plot(new_intersections[:,0], new_intersections[:,1], new_intersections[:,2], "g.", markersize=13.0, label="New")
plt.legend()

plt.show()
