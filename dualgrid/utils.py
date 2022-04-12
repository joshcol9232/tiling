import numpy as np
import dualgrid as dg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.pyplot as plt
import pygmsh
import time
import networkx as nx


""" OFFSET generation
"""
def generate_offsets(num, random, below_one=False, sum_zero=False):
    if random:
        rng = np.random.default_rng(int(time.time() * 10))
    else:
        rng = np.random.default_rng(37123912)  # Arbitrary seed

    offsets = rng.random(num)
    if below_one:
        offsets /= num

    if sum_zero:
        offsets[-1] = -np.sum(offsets)

    print("OFFSETS:", offsets)
    return offsets


""" BASES
    Various pre-defined bases to play around with
"""
def icosahedral_basis(random_offsets=True):
    # Generate grid offsets for use in the algorithm.
    offsets = generate_offsets(6, random_offsets)

    # From: https://physics.princeton.edu//~steinh/QuasiPartII.pdf
    sqrt5 = np.sqrt(5)
    icos = [
        np.array([(2.0 / sqrt5) * np.cos(2 * np.pi * n / 5),
                  (2.0 / sqrt5) * np.sin(2 * np.pi * n / 5),
                  1.0 / sqrt5])
        for n in range(5)
    ]
    icos.append(np.array([0.0, 0.0, 1.0]))
    return dg.Basis(np.array(icos), offsets)

def cubic_basis(random_offsets=True):
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]), generate_offsets(3, random_offsets))

def hypercubic_basis(random_offsets=True):
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]), generate_offsets(4, random_offsets))

def cube5D_basis(random_offsets=True):
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
    ]), generate_offsets(5, random_offsets))

def surface_with_n_rotsym(n, sum_to_zero=False, below_one=False, random_offsets=True):
    """
    Basis for generating a 2D structure with `n` rotational symmetry.
    """
    N = n    # Save N for finding the angles
    if n % 2 == 0:
        n //= 2   # To stop identical basis sets being created for even symmetries

    vecs = np.array([[np.cos(j * np.pi * 2.0/N), np.sin(j * np.pi * 2.0/N)] for j in range(n)])
    offsets = generate_offsets(n, random_offsets, below_one=below_one, sum_zero=sum_to_zero)

    return dg.Basis(vecs, offsets)

def penrose_basis(random_offsets=True):
    return surface_with_n_rotsym(5, sum_to_zero=True, below_one=True, random_offsets=random_offsets)

def ammann_basis(random_offsets=True):
    return surface_with_n_rotsym(8)  # TODO: Do you need to sum to 0 for ammann?

def hexagonal_basis(random_offsets=True):
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0/2.0, np.sqrt(3)/2.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]), generate_offsets(3, random_offsets))


""" Filtering functions. Must take form (point, filter_centre, param_1, param_2, ..., param_N)
"""
def is_point_within_radius(r, filter_centre, radius):
    """
    Used to retain cells within a certain radius of the filter_centre.
    """
    return np.linalg.norm(r - filter_centre) < radius

def is_point_within_cube(r, filter_centre, size):
    """
    Used to retain cells within a cube centred at filter_centre.
    N dimensional cube
    """
    diff = r - filter_centre
    sizediv2 = size/2.0

    return np.sum([abs(d) > sizediv2 for d in diff]) == 0

def get_centre_of_interest(cells):
    """ Used to centre the camera/filter on the densest part of the generated crystal.
    """
    all_verts = []
    for c in cells:
        for v in c.verts:
            all_verts.append(v)

    return np.mean(all_verts, axis=0)  # centre of interest is mean of all the vertices

""" Graph
"""
def graph_from_cells(cells, filter=None, filter_whole_cells=True, filter_args=[], filter_centre=None, fast_filter=False):
    """ Returns a list of all vertices and edges with no duplicates, given a
        list of cells.
        Takes a function to filter out points with, along with it's arguments
    """
    unique_indices = []  # Edges will be when distance between indices is 1
    vert_arr_indices = []
    G = nx.Graph()

    curr_node_count = 0

    if not filter_centre:
        # Find centre of interest
        filter_centre = get_centre_of_interest(cells)
        print("COI:", filter_centre)

    for c in cells:
        if filter and filter_whole_cells:
            # Check whole cell is in filter before continuing
            include_cell = c.is_in_filter(filter, filter_centre, filter_args, fast=fast_filter)
            if include_cell:
                for arr_index, i in enumerate(c.indices):
                    i = list(i)
                    if i not in unique_indices:
                        node_ind = curr_node_count + arr_index
                        unique_indices.append(i)
                        vert_arr_indices.append(node_ind)
                        G.add_node(
                            node_ind,
                            position=c.verts[arr_index],
                            indices=i,
                        )
        else:
            for arr_index, i in enumerate(c.indices):
                i = list(i)
                # Check vertex inside filtering distance
                in_range = True
                if filter:  # If a filtering function is given, then filter out the point
                    in_range = filter(c.verts[arr_index], filter_centre, *filter_args)

                if in_range and i not in unique_indices:
                    unique_indices.append(i)
                    node_ind = curr_node_count + arr_index
                    vert_arr_indices.append(node_ind)
                    G.add_node(
                        node_ind,
                        position=c.verts[arr_index],
                        indices=i,
                    )
        
        curr_node_count += len(c.verts)

    # Indices with distance 1 are edges
    for i in range(len(unique_indices)-1):
        for j in range(i+1, len(unique_indices)):
            if np.linalg.norm(np.array(unique_indices[j]) - np.array(unique_indices[i])) == 1:
                # linked
                G.add_edge(vert_arr_indices[i], vert_arr_indices[j])

    return G


""" RENDERING
"""

def vertex_positions_from_graph(G):
    return np.array([v[1]["position"] for v in G.nodes.data()])


def render_graph(G, **kwargs):
    if len(list(G.nodes(data=True))[0][1]["position"]) == 2:
        _render_2D_wire(G, **kwargs)
    else:
        _render_3D_wire(G, **kwargs)

def _render_2D_wire(
    G,
    vert_size=5.0,
    vert_alpha=1.0,
    edge_thickness=2.0,
    edge_alpha=1.0,
    vert_colour="r",
    edge_colour="k",
    axis_size=5.0,
    filter_centre=None
):
    fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
    ax.axis("equal")

    for edge in G.edges:
        vs = np.array([G.nodes[e]["position"] for e in edge])
        ax.plot(vs[:,0], vs[:,1], "%s-" % edge_colour, linewidth=edge_thickness, alpha=edge_alpha)

    verts = vertex_positions_from_graph(G)
    ax.plot(verts[:,0], verts[:,1], "%s." % vert_colour, markersize=vert_size, alpha=vert_alpha)


    if not filter_centre:
        # Find centre of interest
        filter_centre = np.mean(verts, axis=0)

    plt.xlim(filter_centre[0] - axis_size, filter_centre[0] + axis_size)
    plt.ylim(filter_centre[1] - axis_size, filter_centre[1] + axis_size)
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio

def _render_3D_wire(
    G,
    vert_size=15.0,
    vert_alpha=1.0,
    edge_thickness=4.0,
    edge_alpha=0.5,
    vert_colour="r",
    edge_colour="k",
    axis_size=5.0,
    filter_centre=None,
):

    # Set up matplotlib axes.
    ax = plt.axes(projection="3d")
    # Aggregate vertex positions
    verts = np.array([v[1]["position"] for v in G.nodes.data()])

    if not filter_centre:
        # Find centre of interest
        filter_centre = np.mean(verts, axis=0)

    # Plot edges
    for edge in G.edges:
        vs = np.array([G.nodes[e]["position"] for e in edge])
        ax.plot(vs[:,0], vs[:,1], vs[:,2], "%s-" % edge_colour, linewidth=edge_thickness, alpha=edge_alpha)

    # Plot vertices
    ax.plot(verts[:,0], verts[:,1], verts[:,2], "%s." % vert_colour, markersize=vert_size, alpha=vert_alpha)

    # Set axis scaling equal and set size
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))

    axes_bounds = [
        filter_centre[:3] - np.array([axis_size, axis_size, axis_size]),  # Lower
        filter_centre[:3] + np.array([axis_size, axis_size, axis_size])  # Upper
    ]
    ax.set_xlim(axes_bounds[0][0], axes_bounds[1][0])
    ax.set_ylim(axes_bounds[0][1], axes_bounds[1][1])
    ax.set_zlim(axes_bounds[0][2], axes_bounds[1][2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


# def _get_cell_size_ratio(cell, cell_edges):
#     # Well defined for 2D and 3D, truncate to 3D for ND.
#     dims = len(cell.verts[0])
#     if dims == 2:
#         max( np.dot(cell_edges[0]) )
#     else:
#         return _triple_product()

    

# def render_cells(
#         ax,
#         cells,
#         colormap_str,
#         filter=None,
#         filter_centre=None,
#         filter_args=[],
#         fast_filter=False, # False: checks 1 node per rhombohedron, fast checks all 8 are within range
#         shape_opacity=0.6,
#         axis_size=5.0,
# ):
#     """ Renders cells with matplotlib
#     Has to filter whole cells due to the nature of the render.
#     """
#     clrmap = cm.get_cmap(colormap_str)

#     if not filter_centre:
#         # Find centre of interest
#         filter_centre = get_centre_of_interest(cell_dict)

#     for c in cells:
#         clrindex = c.get_size_ratio()
#         color = clrmap(clrindex)

#         in_render = True
#         # apply filter if there is one
#         if filter:
#             in_render = c.is_in_filter(filter, filter_centre, filter_args, fast=fast_filter)

#         if in_render:
#             faces = c.get_faces()
#             shape_col = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors="k", alpha=shape_opacity)
#             ax.add_collection(shape_col)


#     # Set axis scaling equal and display
#     world_limits = ax.get_w_lims()
#     ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))

#     axes_bounds = [
#         filter_centre - np.array([axis_size, axis_size, axis_size]),  # Lower
#         filter_centre + np.array([axis_size, axis_size, axis_size])  # Upper
#     ]
#     ax.set_xlim(axes_bounds[0][0], axes_bounds[1][0])
#     ax.set_ylim(axes_bounds[0][1], axes_bounds[1][1])
#     ax.set_zlim(axes_bounds[0][2], axes_bounds[1][2])

#     ax.set_xlabel("x")
#     ax.set_ylabel("y")
#     ax.set_zlabel("z")

#     return filter_centre


""" STL Output
"""
def generate_wire_mesh(
    G,
    wire_radius=0.1,
    vertex_radius=None,
    mesh_min_length=0.005,   # Arbitrary defaults
    mesh_max_length=0.05,
    **kwargs                # Keyword arguments to the mesh generator
):
    # Get vertex positions
    verts = vertex_positions_from_graph(G)

    # If 2D, add a z coordinate of 0
    if len(verts[0]) == 2:
        new_verts = []
        for v in verts:
            new_verts.append([v[0], v[1], 0.0])
        verts = np.array(new_verts)

    # Make wiremesh and saves to stl file
    if not vertex_radius:
        vertex_radius = wire_radius
    print("VERTEX RAD:", vertex_radius)

    print("CELL COUNT:", len(verts)//8)

    with pygmsh.occ.Geometry() as geom:       # Use CAD-like commands
        geom.characteristic_length_max = mesh_max_length
        geom.characteristic_length_min = mesh_min_length

        for v in verts:
            geom.add_ball(v, vertex_radius)

        for edge in G.edges:
            vs = np.array([G.nodes[e]["position"][:3] for e in edge]) # Truncate to 3D
            # If 2D, add a z coordinate of 0
            if len(vs[0]) == 2:
                new_verts = []
                for v in vs:
                    new_verts.append([v[0], v[1], 0.0])
                vs = np.array(new_verts)

            geom.add_cylinder(vs[0], vs[1] - vs[0], wire_radius)

        mesh = geom.generate_mesh(**kwargs)


    print("CELL COUNT:", len(verts)//8)

    return mesh

"""
def generate_wire_mesh(
        cell_dict,
        wire_radius=0.1,
        vertex_radius=None,
        mesh_min_length=0.005,   # Arbitrary defaults
        mesh_max_length=0.05,
        filter=is_point_within_cube,
        filter_centre=np.zeros(3),
        filter_whole_cells=True,
        filter_args=[],
        fast_filter=False,
        **kwargs                # Keyword arguments to the mesh generator
):
    # Make wiremesh and saves to stl file
    if not vertex_radius:
        vertex_radius = wire_radius

    verts, edges = verts_and_edges_from_cells(cell_dict, filter=filter, filter_centre=filter_centre, filter_args=filter_args, filter_whole_cells=filter_whole_cells, fast_filter=fast_filter)

    print("CELL COUNT:", len(verts)//8)

    with pygmsh.occ.Geometry() as geom:       # Use CAD-like commands
        geom.characteristic_length_max = mesh_max_length
        geom.characteristic_length_min = mesh_min_length

        for v in verts:
            geom.add_ball(v, vertex_radius)

        for e in edges:
            geom.add_cylinder(verts[e[0]], verts[e[1]] - verts[e[0]], wire_radius)

        mesh = geom.generate_mesh(**kwargs)


    print("CELL COUNT:", len(verts)//8)

    return mesh
"""

def generate_solid_mesh(cell_dict, **kwargs):
    with pygmsh.geo.Geometry() as geom:
        for cells in cell_dict.values():
            for c in cells:
                for face in c.get_faces():
                    geom.add_polygon(face, mesh_size=0.1)

        mesh = geom.generate_mesh(**kwargs)

    return mesh
