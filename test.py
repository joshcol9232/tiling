import dualgrid as dg
import utils
import matplotlib.pyplot as plt


basis_obj = utils.icosahedral_basis()
ax = plt.axes(projection="3d")

rhombohedra, _possible_cells = dg.dualgrid_method(basis_obj)
utils.render_rhombohedra(ax, rhombohedra, "ocean")
plt.show()

