import dualgrid as dg
import utils
import matplotlib.pyplot as plt


basis_obj = utils.icosahedral_basis()
ax = plt.axes(projection="3d")

rhombohedra, _possible_cells = dg.dualgrid_method(basis_obj, k_ranges=2)
utils.render_rhombohedra(ax, rhombohedra, "ocean", render_distance=2.0)
plt.show()

