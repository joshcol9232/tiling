import numpy as np
from scipy.spatial.transform import Rotation as Rot

class Shape:  # Group of triangles to be transformed. Fixed
    def __init__(self):
        self.triangles = np.array([])

    def from_triangles(tri):
        s = Shape()
        s.triangles = np.array(tri)
        return s

    def write(self, fo):
        for tri in self.triangles:
            write_triangle(fo, tri)

    def transform(self, matrix):
        for i in range(len(self.triangles)):
            self.triangles[i] = np.dot(matrix, self.triangles[i].T).T

def vec_from_angle_xyplane(a):
    return np.array([ np.cos(a), np.sin(a), 0.0 ])

def circle_eq(v1, v2, centre, radius, a):
    return centre + radius * v1 * np.cos(a) + radius * v2 * np.sin(a)

def write_triangle(fo, verts, normal=np.zeros(3)):
    assert len(verts) == 3

    fo.write("facet normal %.8e %.8e %.8e\n\touter loop\n" % (normal[0], normal[1], normal[2]))
    for v in verts:
        fo.write("\t\tvertex %.8e %.8e %.8e\n" % (v[0], v[1], v[2]))
    fo.write("\tendloop\nendfacet\n")

def make_circle(centre, radius, normal, seg=32): # seg = segments
    # ensure normal is normalized
    normal /= np.linalg.norm(normal)
    triangles = []
    # Find v1 and v2 - forms circle's basis.
    # (some point on the plane - centre) x normal = some vector in the plane.
    # pick arbitrary x and z.
    # then plug in x and z to get y. This is point on the plane.
    # d = ax0 + by0 + cz0 where x0 etc is centre
    point_on_plane = np.array([1.0, 0.1, 2.11]) # TODO: maybe make random in the future
    d = np.dot(centre, normal)

    if normal[1] != 0:
        point_on_plane[1] = (d - point_on_plane[2] * normal[2] - point_on_plane[0] * normal[0]) / normal[1]
    else:
        point_on_plane[2] = (d - point_on_plane[1] * normal[1] - point_on_plane[0] * normal[0]) / normal[2]

    v1 = np.cross(point_on_plane, normal)
    v1 /= np.linalg.norm(v1)  # normalize
    v2 = np.cross(v1, normal) # last orthogonal vec

    # Fan of triangles
    da = 2 * np.pi/trinum
    anext = 0
    prev = circle_eq(v1, v2, centre, radius, anext)
    for t in range(trinum-1):
        anext += da
        prevtmp = circle_eq(v1, v2, centre, radius, anext)
        triangles.append(np.array([centre, prev, prevtmp]))
        prev = prevtmp

    triangles.append(np.array([
        centre,
        circle_eq(v1, v2, centre, radius, anext),
        circle_eq(v1, v2, centre, radius, 0)       # Ensures last triangle connects
    ]))

    return Shape.from_triangles(triangles)


def make_cylinder(start, end, radius, circle_seg=32):
    lengthvec = end - start
    normal = lengthvec / np.linalg.norm(lengthvec) # normal vector from bottom -> top
    triangles = []
    # Find v1 and v2 - forms circle's basis.
    # (some point on the plane - centre) x normal = some vector in the plane.
    # pick arbitrary x and z.
    # then plug in x and z to get y. This is point on the plane.
    # d = ax0 + by0 + cz0 where x0 etc is centre
    point_on_plane = np.array([1.0, 0.1, 2.11]) # TODO: maybe make random in the future
    d = np.dot(centre, normal)

    if normal[1] != 0:
        point_on_plane[1] = (d - point_on_plane[2] * normal[2] - point_on_plane[0] * normal[0]) / normal[1]
    else:
        point_on_plane[2] = (d - point_on_plane[1] * normal[1] - point_on_plane[0] * normal[0]) / normal[2]

    # Vectors in plane of cylinder faces
    v1 = np.cross(point_on_plane, normal)
    v1 /= np.linalg.norm(v1)  # normalize
    v2 = np.cross(v1, normal) # last orthogonal vec

    # Fan of triangles for each end
    da = 2 * np.pi/trinum
    anext = 0
    prevstart = circle_eq(v1, v2, centre, radius, anext)
    prevend = circle_eq(v1, v2, centre, radius, anext)
    for t in range(trinum-1):
        anext += da
        prevtmpstart = circle_eq(v1, v2, start, radius, anext)
        prevtmpend = circle_eq(v1, v2, end, radius, anext)
        triangles.append(np.array([centre, prev, prevtmp]))
        prevstart = prevtmpstart
        prevend = prevtmpend

    triangles.append(np.array([
        centre,
        circle_eq(v1, v2, centre, radius, anext),
        circle_eq(v1, v2, centre, radius, 0)       # Ensures last triangle connects
    ]))

    return Shape.from_triangles(triangles)


def new_stl(filepath):
    fo = open(filepath, "w")
    fo.write("solid \n")
    return fo


# verts = np.array([ [0.5, 0.5, 0], [0, 1.5, 0.1], [1, 1, 0] ])
# verts = np.array([[0.0, 0.0, 0.0], [0, 1, 0], [0.5, 0.5, 0.0]])
# c = Shape.from_triangles([verts])
c = make_circle(np.array([1.0, 0.0, 0.2]), 1.0, np.array([1.0, 0.0, 1.0]), trinum=128)
fo = new_stl("meshgenout.stl")
c.write(fo)
fo.close()
