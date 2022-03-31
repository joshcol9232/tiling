import dualgrid as dg
import matplotlib.pyplot as plt

basis_obj = dg.utils.icosahedral_basis()
ax = plt.axes(projection="3d")

"""
k_ranges = [
    [0],
    [],
    [],
    [],
    [0],
    [0]
]
"""

filt_dist = 3.0

rhombohedra, _possible_cells = dg.dualgrid_method(basis_obj, k_ranges=3)
dg.utils.render_rhombohedra(ax, rhombohedra, "ocean", filter_distance=filt_dist, filtering_type="cubic")
plt.show()

print("Generating mesh...")
# mesh = dg.utils.generate_wire_mesh(rhombohedra, verbose=True, mesh_min_length=0.005, mesh_max_length=0.03, filter_distance=filt_dist, filtering_type="cubic")
mesh = dg.utils.generate_wire_mesh(rhombohedra, verbose=True, mesh_min_length=0.005, mesh_max_length=0.05, filter_args=[1.5])
# mesh = dg.utils.generate_solid_mesh(rhombohedra)
print("Done. Saving...")
mesh.write("rhombs.stl")
