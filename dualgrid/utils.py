import numpy as np
import dualgrid as dg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.pyplot as plt
import pygmsh

""" BASES
    Various pre-defined bases to play around with
"""
def icosahedral_basis():
    # From: https://physics.princeton.edu//~steinh/QuasiPartII.pdf
    sqrt5 = np.sqrt(5)
    icos = [
        np.array([(2.0 / sqrt5) * np.cos(2 * np.pi * n / 5),
                  (2.0 / sqrt5) * np.sin(2 * np.pi * n / 5),
                  1.0 / sqrt5])
        for n in range(5)
    ]
    icos.append(np.array([0.0, 0.0, 1.0]))
    return dg.Basis(np.array(icos), 3)

def cubic_basis():
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ]), 3)

def hypercubic_basis():
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    ]), 4)

def penrose_basis():
    penrose = [np.array([np.cos(j * np.pi * 2.0 / 5.0), np.sin(j * np.pi * 2.0 / 5.0), 0.0]) for j in range(5)]
    penrose.append(np.array([0.0, 0.0, 1.0]))
    return dg.Basis(np.array(penrose), 3, sum_to_zero=True)

def ammann_basis():
    am = [np.array([np.cos(j * np.pi * 2.0 / 8.0), np.sin(j * np.pi * 2.0 / 8.0)]) for j in range(4)]
    return dg.Basis(np.array(am), 2)

def hexagonal_basis():
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0/2.0, np.sqrt(3)/2.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]), 3)


""" Filtering functions. Must take form (point, filter_centre, param_1, param_2, ..., param_N)
"""
def is_point_within_radius(r, filter_centre, radius):
    return np.linalg.norm(r - filter_centre) < radius

def is_point_within_cube(r, filter_centre, size):
    d = r - filter_centre
    sizediv2 = size/2.0
    return abs(d[0]) < sizediv2 and abs(d[1]) < sizediv2 and abs(d[2]) < sizediv2

def get_centre_of_interest(cell_dict):
    """ Used to centre the camera/filter on the densest part of the generated crystal.
    """
    all_verts = []
    for volume, cells in cell_dict.items():
        for c in cells:
            for v in c.verts:
                all_verts.append(v)

    return np.mean(all_verts, axis=0)  # centre of interest is mean of all the vertices

""" Graph
"""

def verts_and_edges_from_cells(cell_dict, filter=None, filter_whole_cells=True, filter_args=[], filter_centre=None, fast_filter=False):
    """ Returns a list of all vertices and edges with no duplicates, given a
        dictionary of cells.
        Takes a function to filter out points with, along with it's arguments
    """
    unique_indices = []  # Edges will be when distance between indices is 1
    verts = []
    edges = []

    if not filter_centre:
        # Find centre of interest
        filter_centre = get_centre_of_interest(cell_dict)

    for _vol, cells in cell_dict.items():
        for c in cells:
            if filter and filter_whole_cells:
                # Check whole cell is in filter before continuing
                include_cell = c.is_in_filter(filter, filter_centre, filter_args, fast=fast_filter)
                if include_cell:
                    for arr_index, i in enumerate(c.indices):
                        i = list(i)
                        if i not in unique_indices:
                            unique_indices.append(i)
                            verts.append(c.verts[arr_index])
            else:
                for arr_index, i in enumerate(c.indices):
                    i = list(i)
                    # Check vertex inside filtering distance
                    in_range = True
                    if filter:  # If a filtering function is given, then filter out the point
                        in_range = filter(c.verts[arr_index], filter_centre, *filter_args)

                    if in_range and i not in unique_indices:
                        unique_indices.append(i)
                        verts.append(c.verts[arr_index])

    # Indices with distance 1 are edges
    for i in range(len(unique_indices)-1):
        for j in range(i+1, len(unique_indices)):
            if np.linalg.norm(np.array(unique_indices[j]) - np.array(unique_indices[i])) == 1:
                # linked
                edges.append([i, j])

    return np.array(verts), np.array(edges)


""" RENDERING
"""
def render_verts_and_edges(
    ax,
    verts,
    edges,
    vert_size=10.0,
    edge_thickness=2.0,
    vert_colour="r",
    edge_colour="k",
    axis_size=5.0,
    filter_centre=None,
):
    if not filter_centre:
        # Find centre of interest
        filter_centre = np.mean(verts, axis=0)

    # Plot edges
    for edge in edges:
        vs = np.array([verts[e] for e in edge])
        ax.plot(vs[:,0], vs[:,1], vs[:,2], "%s-" % edge_colour, linewidth=edge_thickness)

    # Plot vertices
    ax.plot(verts[:,0], verts[:,1], verts[:,2], "%s." % vert_colour, markersize=vert_size)

    # Set axis scaling equal and set size
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))

    axes_bounds = [
        filter_centre - np.array([axis_size, axis_size, axis_size]),  # Lower
        filter_centre + np.array([axis_size, axis_size, axis_size])  # Upper
    ]
    ax.set_xlim(axes_bounds[0][0], axes_bounds[1][0])
    ax.set_ylim(axes_bounds[0][1], axes_bounds[1][1])
    ax.set_zlim(axes_bounds[0][2], axes_bounds[1][2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


def render_cells(
        ax,
        cell_dict,
        colormap_str,
        filter=None,
        filter_centre=None,
        filter_args=[],
        fast_filter=False, # False: checks 1 node per rhombohedron, fast checks all 8 are within range
        shape_opacity=0.6,
        axis_size=5.0,
):
    """ Renders cells with matplotlib
    Has to filter whole cells due to the nature of the render.
    """
    clrmap = cm.get_cmap(colormap_str)

    if not filter_centre:
        # Find centre of interest
        filter_centre = get_centre_of_interest(cell_dict)

    for volume, cells in cell_dict.items():
        color = clrmap(volume)

        for c in cells:
            in_render = True
            # apply filter if there is one
            if filter:
                in_render = c.is_in_filter(filter, filter_centre, filter_args, fast=fast_filter)

            if in_render:
                faces = c.get_faces()
                shape_col = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors="k", alpha=shape_opacity)
                ax.add_collection(shape_col)


    # Set axis scaling equal and display
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))

    axes_bounds = [
        filter_centre - np.array([axis_size, axis_size, axis_size]),  # Lower
        filter_centre + np.array([axis_size, axis_size, axis_size])  # Upper
    ]
    ax.set_xlim(axes_bounds[0][0], axes_bounds[1][0])
    ax.set_ylim(axes_bounds[0][1], axes_bounds[1][1])
    ax.set_zlim(axes_bounds[0][2], axes_bounds[1][2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return filter_centre

""" STL Output
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
        filter_args=[2.0],
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

def generate_solid_mesh(cell_dict, **kwargs):
    with pygmsh.geo.Geometry() as geom:
        for cells in cell_dict.values():
            for c in cells:
                for face in c.get_faces():
                    geom.add_polygon(face, mesh_size=0.1)

        mesh = geom.generate_mesh(**kwargs)

    return mesh
