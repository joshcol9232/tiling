import dualgrid as dg
import matplotlib.pyplot as plt

basis = dg.utils.icosahedral_basis()

filt_dist = 1.5
cells = dg.dualgrid_method(basis, k_range=3)
# Take a chunk of the crystal out of the middle
cells = dg.utils.filter_cells(cells, filter=dg.utils.is_point_within_cube, filter_args=[filt_dist])
G = dg.utils.graph_from_cells(cells) # Make graph
print("Generated graph.")

# Generate & save a mesh.
print("Generating mesh...")
mesh = dg.utils.generate_wire_mesh(G, verbose=True)
print("Saving mesh...")
mesh.write("G.stl")
