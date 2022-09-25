import numpy as np

class Shape:  # Group of triangles to be transformed. Fixed
    def __init__(self):
        self.triangles = np.array([])

    def __iadd__(self, rhs):
        assert len(rhs) == 3
        for i in range(len(self.triangles)):
            self.triangles[i] += rhs
        return self

    def __imul__(self, rhs):
        self.transform(rhs)
        return self

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

def add_hemisphere_segment(triangles, v1, v2, centre, dir, r0, a0, a1, longitudes=8): # (without base)
    """
    Edits triangles array by reference
    """
    phi = np.linspace(0, np.pi/2, num=longitudes)[:-1]
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    #print(phi)
    for i in range(len(phi)-1):
        #print("DOING PHI:", phi[i])
        # v1, v2, centre, radius, a
        # anticlockwise from bottom corner 0 -> 3 of square tile.
        r_lower = r0 * cosphi[i]
        r_higher = r0 * cosphi[i+1]
        vs = [
            circle_eq(v1, v2, centre, r_lower, a0) + dir * r0 * sinphi[i],
            circle_eq(v1, v2, centre, r_lower, a1) + dir * r0 * sinphi[i],
            circle_eq(v1, v2, centre, r_higher, a1) + dir * r0 * sinphi[i+1],
            circle_eq(v1, v2, centre, r_higher, a0) + dir * r0 * sinphi[i+1]
        ]

        # Make triangles
        triangles.append([vs[0], vs[1], vs[2]])
        triangles.append([vs[0], vs[2], vs[3]])

    # final cap triangle
    triangles.append([
        circle_eq(v1, v2, centre, r0 * cosphi[-1], a0) + dir * r0 * sinphi[-1],
        circle_eq(v1, v2, centre, r0 * cosphi[-1], a1) + dir * r0 * sinphi[-1],
        centre + dir * r0, # Top of sphere
    ])
    #print("Cap placed:", triangles[-3:])


def make_rounded_cylinder(rodvecs, radius, circle_seg=32, **kwargs):
    start = rodvecs[0]
    end = rodvecs[1]

    lengthvec = end - start
    normal = lengthvec / np.linalg.norm(lengthvec) # normal vector from bottom -> top
    triangles = []
    # Find v1 and v2 - forms circle's basis.
    # (some point on the plane - centre) x normal = some vector in the plane.
    # pick arbitrary x and z.
    # then plug in x and z to get y. This is point on the plane.
    # d = ax0 + by0 + cz0 where x0 etc is centre
    point_on_plane = np.random.rand(3) #np.array([1.0, 0.1, 2.11]) # TODO: maybe make random in the future
    d = np.dot(start, normal)

    if abs(normal[1]) > 0.000001:
        point_on_plane[1] = (d - point_on_plane[2] * normal[2] - point_on_plane[0] * normal[0]) / normal[1]
    elif abs(normal[2]) > 0.000001:
        point_on_plane[2] = (d - point_on_plane[1] * normal[1] - point_on_plane[0] * normal[0]) / normal[2]
    elif abs(normal[0]) > 0.000001:
        point_on_plane[0] = (d - point_on_plane[1] * normal[1] - point_on_plane[2] * normal[2]) / normal[0]
    else:
        raise ValueError("meshgen: Normal vector can't be (0, 0, 0): ", normal)

    # Vectors in plane of cylinder faces
    v1 = np.cross(point_on_plane, normal)
    v1 /= np.linalg.norm(v1)  # normalize
    v2 = np.cross(v1, normal) # last orthogonal vec

    da = 2 * np.pi/circle_seg
    anext = 0
    prev = circle_eq(v1, v2, start, radius, anext)
    for t in range(circle_seg):
        anext += da
        prevtmp = circle_eq(v1, v2, start, radius, anext) # next segment
        # Top rounded end
        add_hemisphere_segment(triangles, v1, v2, start, -normal, radius, anext-da, anext, **kwargs)

        # Bottom rounded end. TODO: top_tris can be copied, flipped and moved to the bottom of the cylinder
        add_hemisphere_segment(triangles, v1, v2, end, normal, radius, anext-da, anext, **kwargs)

        # Side pannels (two to form square)
        triangles.append(np.array([prev, prevtmp, prevtmp + lengthvec])) # top -> bottom
        triangles.append(np.array([prevtmp + lengthvec, prev + lengthvec, prev])) # bottom -> top

        prev = prevtmp

    return Shape.from_triangles(triangles)


def new_stl(filepath):
    fo = open(filepath, "w")
    fo.write("solid \n")
    return fo

if __name__ == "__main__":
    # verts = np.array([ [0.5, 0.5, 0], [0, 1.5, 0.1], [1, 1, 0] ])
    # verts = np.array([[0.0, 0.0, 0.0], [0, 1, 0], [0.5, 0.5, 0.0]])
    # c = Shape.from_triangles([verts])
    # c = make_circle(np.array([1.0, 0.0, 0.2]), 1.0, np.array([1.0, 0.0, 1.0]), trinum=128)
    # c = make_cylinder(np.zeros(3), np.array([0.0, 0.0, 3.0]), 1.0, circle_seg=8)
    c = make_rounded_cylinder(np.zeros(3), np.array([0, 0, 3]), 1.0, circle_seg=32, longitudes=8)
    # c += np.array([2.0, 2.0, 2.0])
    # m = Rot.from_euler("x", np.pi/4.0).as_matrix()
    #c.transform(m)
    # c *= m

    fo = new_stl("meshgenout.stl")
    c.write(fo)
    fo.close()
