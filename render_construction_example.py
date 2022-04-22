# Example of rendering the construction sets in 2D
import dualgrid as dg
import matplotlib.pyplot as plt


basis = dg.utils.surface_with_n_rotsym(7, centred=True) # Choose 7-fold rotational sym

fig, ax = plt.subplots(1, figsize=(10, 10)) # Set up axis
ax.axis("equal")
ax_size = 1
k_range = 3

dg.utils.render_2D_construction(ax, basis, k_range, ax_size)
# OPTIONAL: Cells can be generated and rendered on top of their
# parent intersection, i.e the intersection that formed each rhombus.
# Comment these two lines out if you just want grid lines.
cells = dg.dualgrid_method(basis, k_range)
dg.utils.render_2D_cells_at_intersections(ax, cells, scale=0.05, axis_size=ax_size)

plt.show()
