import numpy as np
import dualgrid as dg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pygmsh
import time
import networkx as nx


""" OFFSET generation
"""
def offsets_fixed_around_centre(n):
    """
    Gives offsets that give a tiling around centre of rotation.
    """
    return np.array([1/n for _i in range(n)])

def generate_offsets(num, random, below_one=False, sum_zero=False, centred=False):
    """
    Generates offsets for use in the dualgrid method.
    num: Number of offsets to generate. Usually the same as the number of basis vectors.
    random: Generate random offsets? Or the same random ones each time (fixed seed).
    below_one: Keep all of the offsets below one - divides by num. Only makes a difference when sum_zero=True
    sum_zero: Make the offsets sum to 0. For example Penrose tiling offsets must sum to 0.
    centred: Give a tiling centred around the centre of rotation. (NOTE: Not sure if this is working in 3D)
    """
    if centred:
        return offsets_fixed_around_centre(num)

    if random:
        rng = np.random.default_rng(int(time.time() * 10))
    else:
        rng = np.random.default_rng(37123912)  # Arbitrary seed

    offsets = rng.random(num)
    if below_one:
        offsets /= num

    if sum_zero:
        offsets[-1] = -np.sum(offsets)

    return offsets


""" BASES
    Various pre-defined bases to play around with
"""
def icosahedral_basis(random_offsets=True, **kwargs):
    offsets = generate_offsets(6, random_offsets, **kwargs)

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

def n_dimensional_cubic_basis(dims, random_offsets=True, **kwargs):
    """
    Returns an N-dimensional cubic basis.
    dims: N
    """
    offsets = generate_offsets(dims, random_offsets, **kwargs)
    basis_vecs = np.asarray(np.eye(dims))
    return dg.Basis(basis_vecs, offsets)

def cubic_basis(random_offsets=True, **kwargs):
    return n_dimensional_cubic_basis(3, random_offsets=random_offsets, **kwargs)


def surface_with_n_rotsym(n, random_offsets=True, **kwargs):
    """
    Basis for generating a 2D structure with `n` rotational symmetry.
    """
    N = n
    if n % 2 == 0:
        n //= 2

    vecs = np.array([[np.cos(j * np.pi * 2.0/N), np.sin(j * np.pi * 2.0/N)] for j in range(n)])

    offsets = generate_offsets(n, random_offsets, **kwargs)

    return dg.Basis(vecs, offsets)

def penrose_basis(random_offsets=True, **kwargs):
    return surface_with_n_rotsym(5, sum_to_zero=True, below_one=True, random_offsets=random_offsets, **kwargs)

def hexagonal_basis(random_offsets=True, **kwargs):
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0/2.0, np.sqrt(3)/2.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]), generate_offsets(3, random_offsets, **kwargs))


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

def contains_value(r, filter_centre, value):
    """
    Checks for index "r" within indices. Not for use with real space.
    """
    return value in list(r)

def elements_are_below(r, filter_centre, value):
    """
    Checks if all elements are below the given value
    """
    for element in r:
        if element > value:
            return False

    return True



def get_centre_of_interest(cells):
    """
    Used to centre the camera/filter on the densest part of the generated crystal.
    """
    all_verts = []
    for c in cells:
        for v in c.verts:
            all_verts.append(v)

    return np.mean(all_verts, axis=0)  # centre of interest is mean of all the vertices

""" Graph & filtering
"""

def filter_cells(cells, filter, filter_args=[], filter_centre=None, fast_filter=False, filter_indices=False, invert_filter=False):
    """
    The position and indices of the vertex is embedded in each node.
    Takes a function to filter out points with, along with it's arguments.
    """

    if type(filter_centre) == type(None):
        print("FINDING COI")
        # Find centre of interest
        filter_centre = get_centre_of_interest(cells)
        print("COI:", filter_centre)

    cells = np.array(cells)
    filter_results = [c.is_in_filter(filter, filter_centre, filter_args, fast=fast_filter, filter_indices=filter_indices, invert_filter=invert_filter) for c in cells]

    return cells[filter_results]

def graph_from_cells(cells):
    """ 
    Returns a networkx graph given a list of cells.
    """
    G = nx.Graph()

    unique_indices = []  # Edges will be when distance between indices is 1
    vert_arr_indices = []
    cell_vertices = [] # Cell vertices (array index form).
                       # [ [cell_0_vertices,], ... , [cell_N_vertices,]  ]

    for cell_index, c in enumerate(cells):
        for arr_index, i in enumerate(c.indices):
            i = list(i)
            if i not in unique_indices:
                node_ind = (cell_index * 8) + arr_index
                unique_indices.append(i)
                vert_arr_indices.append(node_ind)
                G.add_node(
                    node_ind,
                    position=c.verts[arr_index],
                    indices=i,
                )

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
    """
    Extracts vertex positions from the graph data.
    """
    return np.array([v[1]["position"] for v in G.nodes.data()])

def render_graph_wire(G, *args, **kwargs):
    """
    Render the graph as nodes connected by edges.
    """
    if len(list(G.nodes(data=True))[0][1]["position"]) == 2:
        _render_2D_wire(G, *args, **kwargs)
    else:
        _render_3D_wire(G, *args, **kwargs)

def render_2D_construction(ax, basis, k_range, x_range):
    cols = ["r", "g", "b", "y", "m", "c", "k"]
    x = np.linspace(-x_range - 0.5, x_range + 0.5)

    for i, vec in enumerate(basis.vecs):
        for k in range(1-k_range, k_range):
            y = (basis.offsets[i] + k - (vec[0] * x))/vec[1]

            # Check if line is vertical
            if float("inf") in y or float("-inf") in y:
                ax.axvline(x=basis.offsets[i] + k, color=cols[i%len(cols)])
            else:
                ax.plot(x, y, color=cols[i%len(cols)])
    
    plt.xlim(-x_range, x_range)
    plt.ylim(-x_range, x_range)

def render_2D_cells_at_intersections(
    ax,
    cells,
    colourmap_str="viridis",
    opacity=1.0,
    edge_thickness=1.0,
    edge_colour="k",
    scale=0.1,
    axis_size=5.0,
):
    def make_polygon(cell_verts, scale, intersection):
         # copy to new array in draw-order
        verts = np.array([cell_verts[0], cell_verts[1], cell_verts[3], cell_verts[2]])
        if scale < 1.0:
            for v in verts:
                v -= (v - intersection) * (1.0 - scale)
        
        return Polygon(verts)

    # Group by smallest internal angle. This will serve as the colour index
    INDEX_DECIMALS = 4  # Significant figures used in grouping cells together
    poly_dict = {} # Dictionary of {size index: [matplotlib polygon]}


    for cell_index, c in enumerate(cells):
        # CENTRE CELLS ON INTERSECTION
        middle = np.mean(c.verts, axis=0)
        diff = c.intersection - middle
        c.verts += diff

        size_ratio = np.around(abs(np.dot(c.verts[0] - c.verts[1], c.verts[0] - c.verts[2])), decimals=4)
        p = make_polygon(c.verts, scale, c.intersection)
        if size_ratio not in poly_dict:
            poly_dict[size_ratio] = [p]
        else:
            poly_dict[size_ratio].append(p)

    # Render
    if colourmap_str == "":
        clrmap = lambda s: "w"
    else:
        clrmap = cm.get_cmap(colourmap_str)

    for size_ratio, polygons in poly_dict.items():
        colour = clrmap(size_ratio)
        shape_coll = PatchCollection(polygons, edgecolor=edge_colour, facecolor=colour, linewidth=edge_thickness, antialiased=True)
        ax.add_collection(shape_coll)


    plt.xlim(-axis_size, axis_size)
    plt.ylim(-axis_size, axis_size)
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio


def _render_2D_wire(
    G,
    ax,
    vert_size=7.0,
    vert_alpha=1.0,
    edge_thickness=2.0,
    edge_alpha=1.0,
    vert_colour="r",
    edge_colour="k",
    axis_size=5.0,
    filter_centre=None
):
    for edge in G.edges:
        vs = np.array([G.nodes[e]["position"] for e in edge])
        ax.plot(vs[:,0], vs[:,1], "%s-" % edge_colour, linewidth=edge_thickness, alpha=edge_alpha)

    verts = vertex_positions_from_graph(G)
    ax.plot(verts[:,0], verts[:,1], "%s." % vert_colour, markersize=vert_size, alpha=vert_alpha)


    if type(filter_centre) == type(None):
        # Find centre of interest
        filter_centre = np.mean(verts, axis=0)

    plt.xlim(filter_centre[0] - axis_size, filter_centre[0] + axis_size)
    plt.ylim(filter_centre[1] - axis_size, filter_centre[1] + axis_size)
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio

def _render_3D_wire(
    G,
    ax,
    vert_size=15.0,
    vert_alpha=1.0,
    edge_thickness=4.0,
    edge_alpha=0.5,
    vert_colour="r",
    edge_colour="k",
    axis_size=5.0,
    filter_centre=None,
):
    # Aggregate vertex positions
    verts = np.array([v[1]["position"] for v in G.nodes.data()])

    if type(filter_centre) == type(None):
        # Find centre of interest
        filter_centre = np.mean(verts, axis=0)

    # Plot edges
    for edge in G.edges:
        vs = np.array([G.nodes[e]["position"] for e in edge])
        ax.plot(vs[:,0], vs[:,1], vs[:,2], "%s-" % edge_colour, linewidth=edge_thickness, alpha=edge_alpha)

    # Plot vertices
    ax.plot(verts[:,0], verts[:,1], verts[:,2], "%s." % vert_colour, markersize=vert_size, alpha=vert_alpha)

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

    # Set axis scaling equal and set size
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))


def _triple_product(a, b, c):
    return np.dot( a, np.cross(b, c) )

def _get_cell_size_ratio(cell, decimals):
    # Well defined for 2D and 3D, truncate to 3D for ND.
    # cell_edges should be decided beforehand and is
    # the same for every cell. 2D and 3D are known,
    # 4D and above cannot be rendered anyway (currently)
    if len(cell.verts[0]) == 2:
        dot = abs(np.dot(cell.verts[0] - cell.verts[1], cell.verts[0] - cell.verts[2]))
            # np.dot(cell.verts)
        return np.around(dot, decimals=decimals)
    else:
        return np.around(abs(_triple_product(cell.verts[1][:3] - cell.verts[0][:3], cell.verts[2][:3] - cell.verts[0][:3], cell.verts[4][:3] - cell.verts[0][:3])), decimals=decimals)

def _render_cells_solid_2D(
    cells,
    ax,
    colourmap_str="viridis",
    opacity=1.0,
    edge_thickness=1.0,
    edge_colour="k",
    scale=1.0,
    centre_of_interest=None,
    axis_size=5.0,
):
    def make_polygon(cell_verts, scale):
         # copy to new array in draw-order
        verts = np.array([cell_verts[0], cell_verts[1], cell_verts[3], cell_verts[2]])
        if scale < 1.0:
            middle = np.mean(verts, axis=0)
            for v in verts:
                v -= (v - middle) * (1.0 - scale)
        
        return Polygon(verts)

    # Group by smallest internal angle. This will serve as the colour index
    INDEX_DECIMALS = 4  # Significant figures used in grouping cells together
    poly_dict = {} # Dictionary of {size index: [matplotlib polygon]}

    for cell_index, c in enumerate(cells):
        size_ratio = _get_cell_size_ratio(c, INDEX_DECIMALS)
        p = make_polygon(c.verts, scale)
        if size_ratio not in poly_dict:
            poly_dict[size_ratio] = [p]
        else:
            poly_dict[size_ratio].append(p)

    # Render
    clrmap = cm.get_cmap(colourmap_str)
    for size_ratio, polygons in poly_dict.items():
        colour = clrmap(size_ratio)
        shape_coll = PatchCollection(polygons, edgecolor=edge_colour, facecolor=colour, linewidth=edge_thickness, antialiased=True)
        ax.add_collection(shape_coll)

    if type(centre_of_interest) == type(None):
        # Find coi
        centre_of_interest = get_centre_of_interest(cells)

    plt.xlim(centre_of_interest[0] - axis_size, centre_of_interest[0] + axis_size)
    plt.ylim(centre_of_interest[1] - axis_size, centre_of_interest[1] + axis_size)
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio
    


def _render_cells_solid_3D(
    cells,
    ax,
    colourmap_str="viridis",
    shape_opacity=0.6,
    axis_size=5.0,
    edge_thickness=0.5,
    edge_colour="k",
    scale=1.0,
    centre_of_interest=None,
):
    """
    Renders solid 3D cells with matplotlib
    """
    def get_scaled_faces(cell, scale):
        """
        Returns the vertices of each face in draw order (ACW) for the 3D cell
        """
        # Definitions for each rhombohedron. The same for _every_ 3D rhomb
        FACE_INDICES = np.array([  # Faces of every rhombohedron (ACW order).
            [0, 2, 3, 1],
            [0, 1, 5, 4],
            [5, 7, 6, 4],
            [2, 6, 7, 3],
            [0, 4, 6, 2],
            [3, 7, 5, 1]
        ])
        faces = np.zeros((6, 4, 3), dtype=float)
        for i, face in enumerate(FACE_INDICES):
            for j, face_index in enumerate(face):
                faces[i][j] = cell.verts[face_index][:3]

        if scale < 1.0:
            # Find middle of rhomb to scale around
            middle = np.mean(cell.verts, axis=0)[:3]
            for face in faces:
                for v in face:
                    v -= (v - middle) * (1.0 - scale)

        return faces

    clrmap = cm.get_cmap(colourmap_str)

    if type(centre_of_interest) == type(None):
        # Find centre of interest (mean of all vertices)
        centre_of_interest = get_centre_of_interest(cells)[:3]

    INDEX_DECIMALS = 4
    for c in cells:
        clrindex = _get_cell_size_ratio(c, INDEX_DECIMALS)
        color = clrmap(clrindex)

        faces = get_scaled_faces(c, scale)
        shape_col = Poly3DCollection(faces, facecolors=color, linewidths=edge_thickness, edgecolors=edge_colour, alpha=shape_opacity)
        ax.add_collection(shape_col)

    axes_bounds = [
        centre_of_interest - np.array([axis_size, axis_size, axis_size]),  # Lower
        centre_of_interest + np.array([axis_size, axis_size, axis_size])  # Upper
    ]
    ax.set_xlim(axes_bounds[0][0], axes_bounds[1][0])
    ax.set_ylim(axes_bounds[0][1], axes_bounds[1][1])
    ax.set_zlim(axes_bounds[0][2], axes_bounds[1][2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


    # Set axis scaling equal for display
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))



def render_cells_solid(cells, *args, **kwargs):
    if len(cells[0].verts[0]) == 2:
        _render_cells_solid_2D(cells, *args, **kwargs)
    else:
        _render_cells_solid_3D(cells, *args, **kwargs)


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
    if type(vertex_radius) == type(None):
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
