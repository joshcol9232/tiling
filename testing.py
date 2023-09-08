import dualgrid as dg
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import time
import sys

#basis = dg.utils.surface_with_n_rotsym(5, centred=False)   # 2D structure with 7-fold rotational symmetry
basis = dg.utils.penrose_basis()          # Section of Penrose tiling.
# basis = dg.utils.icosahedral_basis()      # 3D quasicrystalline structure
#basis = dg.utils.n_dimensional_cubic_basis(4) # 4D cubic structure

print("OFFSETS:", basis.offsets)

k_range = int(sys.argv[1])

# TIME how long
start_time = time.time()

cells = dg.dualgrid_method(basis, k_range, single_threaded=True)

duration = time.time() - start_time
print("Done! Took: %.8f seconds" % duration)
print("Cells generated per second: %.2f" % (len(cells) / duration))

# print("Cells found.\nFiltering...")
# To filter by highest index allowed (good for 2D, odd N-fold tilings):
# cells = dg.utils.filter_cells(cells, filter=dg.utils.elements_are_below, filter_args=[max(k_range-1, 0)], filter_indices=True, invert_filter=False)

"""
R = 11
if basis.dimensions != 2:
    R = 2 # Reduce for 3D+ to reduce lag
cells = dg.utils.filter_cells(cells, filter=dg.utils.is_point_within_radius, filter_args=[R])
"""

# Set up matplotlib axes
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.axis("equal")

dg.utils.render_cells_solid(cells, ax, scale=0.85, edge_thickness=0.0)
ax.autoscale(enable=True)  # Zoom out to fit whole tiling
ax.set_axis_off()

plt.show()

