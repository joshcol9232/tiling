import numpy as np
import dualgrid as dg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
import matplotlib.pyplot as plt


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
    return dg.Basis(np.array(penrose), 2, sum_to_zero=True)

def ammann_basis():
    am = [np.array([np.cos(j * np.pi * 2.0 / 8.0), np.sin(j * np.pi * 2.0 / 8.0), 0.0]) for j in range(4)]
    am.append(np.array([0.0, 0.0, 1.0]))
    return dg.Basis(np.array(am), 2)

def hexagonal_basis():
    return dg.Basis(np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0/2.0, np.sqrt(3)/2.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]), 3)


""" RENDERING
"""
def render_rhombohedra(
        ax,
        rhombohedra,
        colormap_str,
        render_distance=None,
        render_distance_type="spherical",   # Choices are: spherical, cubic
        fast_render_dist_checks=False,
        shape_opacity=0.6,
        coi=None,
):
    """ Renders rhombohedra with matplotlib
    """
    clrmap = cm.get_cmap(colormap_str)

    if type(coi) == type(None):
        # Find centre of interest
        print("Finding COI")
        all_verts = []
        for volume, rhombs in rhombohedra.items():
            for r in rhombs:
                for v in r.verts:
                    all_verts.append(v)
        coi = np.mean(all_verts, axis=0)  # centre of interest is mean of all the vertices

    for volume, rhombs in rhombohedra.items():
        color = clrmap(volume)

        if type(render_distance) == type(None):
            for r in rhombs:
                faces = r.get_faces()
                shape_col = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors="k", alpha=shape_opacity)
                ax.add_collection(shape_col)
            render_distance = 10.0  # Default render distance used for setting axis limits later

        else:
            for r in rhombs:
                inside_render = False
                if render_distance_type == "cubic":
                    inside_render = r.is_inside_box(render_distance, centre=coi, fast=fast_render_dist_checks)
                else:  # Defaults to spherical
                    inside_render = r.is_within_radius(render_distance, centre=coi, fast=fast_render_dist_checks)

                if inside_render:
                    faces = r.get_faces()
                    shape_col = Poly3DCollection(faces, facecolors=color, linewidths=0.2, edgecolors="k", alpha=shape_opacity)
                    ax.add_collection(shape_col)

    # Set axis scaling equal and display
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))

    axes_bounds = [
        coi - np.array([render_distance, render_distance, render_distance]),  # Lower
        coi + np.array([render_distance, render_distance, render_distance])  # Upper
    ]
    ax.set_xlim(axes_bounds[0][0], axes_bounds[1][0])
    ax.set_ylim(axes_bounds[0][1], axes_bounds[1][1])
    ax.set_zlim(axes_bounds[0][2], axes_bounds[1][2])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    return coi