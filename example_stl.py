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
print("Saving mesh to ./graph_out.stl ...")
dg.utils.export_graph_to_stl(G, "graph_out.stl", 0.1)
