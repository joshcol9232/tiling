import numpy as np
import time

DEFAULT_K_RANGE = 3

class PlaneSet:
    def __init__(self, normal, offset, setnum, k_range):
        self.normal = normal
        self.offset = offset
        self.setnum = setnum
        self.k_range = k_range

    def __getitem__(self, k):
        """ Returns point on plane of index k
            PlaneSet[0] -> plane 0 of planeset
        """
        return self.normal + self.normal * (self.offset + k)

    def get_intersections_with(self, other1, other2):
        """
        If this plane intersects with two other planes at a point, this function
        will return the location of this intersection in real space.
        """
        # Checks:
        if np.dot(self.normal, np.cross(other1.normal, other2.normal)) == 0:
            print("WARNING: Sets (%s, %s, %s) may not cross at single points." % (self.setnum, other1.setnum, other2.setnum))
            return []

        coef = np.matrix([
            self.normal,
            other1.normal,
            other2.normal
        ])

        # Check for singular matrix
        if np.linalg.det(coef) == 0:
            print("WARNING: Unit vectors form singular matrices.")
            return []

        # inverse of coefficient matrix
        coef_inv = np.linalg.inv(coef)

        intersections = []

        for k1 in self.k_range:
            for k2 in other1.k_range:
                for k3 in other2.k_range:
                    # remaining part of cartesian form (d)
                    ds = np.matrix([
                        [self.offset + k1],
                        [other1.offset + k2],
                        [other2.offset + k3]
                    ])  # Last row of the matrix -> i.e last element of cartesian form ax + bx + c = d, `d`
                    xyz = np.matmul(coef_inv, ds)
                    intersections.append({   # Append information about the intersection
                        "location": np.array([xyz[0, 0], xyz[1, 0], xyz[2, 0]]),   # column matrix -> vector
                        "ks": [k1, k2, k3],
                        "js": [self.setnum, other1.setnum, other2.setnum],
                    })

        return intersections


def realspace(indices, basis):
    out = np.zeros(3, dtype=float)
    for j, e in enumerate(basis):
        out += e * indices[j]

    return out

def gridspace(r, basis, offsets):
    out = np.zeros(len(basis), dtype=int)

    for j, e in enumerate(basis):
        out[j] = int(np.ceil( np.dot( r, basis[j] ) - offsets[j] ))

    return out


def get_neighbours(intersection, basis, offsets):
    directions = np.array([   # Each possible neighbour of intersection. Derived from eq. 4.5 in de Bruijn paper
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],

        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ])

    indices = gridspace(intersection["location"], basis, offsets)

    # DEBUG print("Root indices before loading:", indices)
    # Load known indices into indices array
    for index, j in enumerate(intersection["js"]):
        indices[j] = intersection["ks"][index]

    # DEBUG print("Getting neighbours for:", intersection)
    # DEBUG print("Root indices:", indices)
    # First off copy the intersection indices 8 times
    neighbours = [ np.array([ v for v in indices ]) for _i in range(8) ]

    # Quick note: Kronecker delta function -> (i == j) = False (0) or True (1) in python. Multiplication of bool is allowed
    delta1 = np.array([ (j == intersection["js"][0]) * 1 for j in range(len(basis)) ])
    delta2 = np.array([ (j == intersection["js"][1]) * 1 for j in range(len(basis)) ])
    delta3 = np.array([ (j == intersection["js"][2]) * 1 for j in range(len(basis)) ])

    # Apply equation 4.5 in de Bruijn's paper 1, expanded for any basis len and extra third dimension
    for i, e in enumerate(directions): # Corresponds to epsilon in paper
        neighbours[i] += e[0] * delta1 + e[1] * delta2 + e[2] * delta3

    return neighbours


def get_largest_node_displacement(basis):
    l = np.zeros(3)
    l_norm = 0.0
    for i in range(len(basis)-1):
        for j in range(i+1, len(basis)):
            l_temp = basis[i] + basis[j]
            l_temp_norm = np.linalg.norm(l_temp)
            if l_temp_norm > l_norm:  #  If bigger
                l_norm = l_temp_norm
                l = l_temp

    return l, l_norm

def triple_product(a, b, c):
    return np.dot( a, np.cross(b, c) )


class Basis:
    def __init__(self, vecs, dimensions, sum_to_zero=False):
        self.vecs = vecs
        self.dimensions = dimensions
        self.is_2d = dimensions % 2 == 0
        self.sum_to_zero = sum_to_zero

    def get_offsets(self, is_random):
        if is_random:
            offsets = []
            N = len(self.vecs) - 1 * self.is_2d

            rng = np.random.default_rng(int(time.time() * 10))
            for i in range(N - 1):
                offsets.append(rng.random())

            if self.sum_to_zero:
                if not self.is_2d:
                    offsets.append(rng.random())
                # Sum of all sigma needs to equal 0
                s = np.sum(np.array(offsets))
                offsets.append(-s)
                if self.is_2d:
                    offsets.append(0.0)
            else:
                for _i in range(2):
                    offsets.append(rng.random())

            return offsets
        else:
            if self.is_2d:
                a = [-1.0 / len(self.vecs) for _i in range(len(self.vecs) - 1)]
                a.append(0.0)
                return a
            else:
                return [-1.0 / len(self.vecs) for _i in range(len(self.vecs))]  # Centre of rotational symmetry in penrose-like case

    def get_possible_cells(self, decimals):
        """ Function that finds all possible cell shapes in the final mesh.
            Number of decimal places required for finite hash keys (floats are hard to == )

            Returns a dictionary of volume : [all possible combinations of basis vector to get that volume]
        """
        shapes = {}  # volume : set indices

        for i in range(len(self.vecs-2)):  # Compare each vector set
            for j in range(i+1, len(self.vecs)-1):
                for k in range(j+1, len(self.vecs)):
                    vol = abs(triple_product(self.vecs[i], self.vecs[j], self.vecs[k]))
                    vol = np.around(vol, decimals=decimals)
                    if vol not in shapes.keys():
                        shapes[vol] = []

                    shapes[vol].append([i, j, k])

        return shapes

""" MAIN ALGORITHM """
# Some definitions for each rhombohedron
FACE_INDICES = [  # Faces of every rhombohedron (ACW order). Worked out on paper
    [0, 2, 3, 1],
    [0, 1, 5, 4],
    [5, 7, 6, 4],
    [2, 6, 7, 3],
    [0, 4, 6, 2],
    [3, 7, 5, 1]
]

EDGES = [  # Connections between neighbours for every cell
    [0, 2],
    [2, 3],
    [3, 1],
    [1, 0],
    [4, 6],
    [6, 7],
    [7, 5],
    [5, 4],
    [0, 4],
    [1, 5],
    [3, 7],
    [2, 6]
]

class Rhombahedron:
    def __init__(self, vertices, indices, parent_sets):
        self.verts = vertices
        self.indices = indices
        self.parent_sets = parent_sets

    def __repr__(self):
        return "Rhombahedron(%s parents %s)" % (self.indices[0], self.parent_sets)

    def get_volume(self): # General for every rhombahedron given that vertices are generated the same way every time (they are due to get_neighbours function)
        return abs(triple_product(self.verts[1] - self.verts[0], self.verts[2] - self.verts[0], self.verts[4] - self.verts[0]))

    def get_faces(self):
        """ Returns the vertices of each face in draw order (ACW) for the rhombahedron
        """
        faces = np.zeros((6, 4, 3), dtype=float)
        for i, face in enumerate(FACE_INDICES):
            for j, face_index in enumerate(face):
                faces[i][j] = self.verts[face_index]

        return faces

    def is_within_radius(self, radius, fast=False, centre=np.zeros(3)):
        """ Utility function for checking whever the rhombohedron is in rendering distance
        `fast` just checks the first vertex
        """
        # If any of the vertices are within range, then say yes
        if fast:
            return np.linalg.norm(self.verts[0] - centre) < radius
        else:
            for v in self.verts:
                if np.linalg.norm(v - centre) < radius:
                    return True

    def is_inside_box(self, size, fast=False, centre=np.zeros(3)):
        """ Checks if any of the vertices are within a box of size given with centre given
        `fast` just checks 1 vertex of the rhombus
        """
        sizediv2 = size / 2
        if fast:
            d = self.verts[0] - centre
            return abs(d[0]) < sizediv2 and abs(d[1]) < sizediv2 and abs(d[2]) < sizediv2
        else:
            for v in self.verts:
                d = v - centre
                if abs(d[0]) < sizediv2 and abs(d[1]) < sizediv2 and abs(d[2]) < sizediv2: # simple axis aligned cube collision
                    return True


def dualgrid_method(basis_obj, k_ranges=None, offsets=None, random=True, shape_accuracy=4):
    """ de Bruijn dual grid method.
    Generates and returns rhombohedra from basis given in the range given.
    Shape accuracy is the number of decimal places used to classify cell shapes

    Returns: rhombohedra, possible cell shapes

    :param basis_obj:
    :param k_ranges:
    :return:
    """
    possible_cells = basis_obj.get_possible_cells(shape_accuracy)
    basis = basis_obj.vecs

    if type(offsets) == type(None):
        offsets = basis_obj.get_offsets(random)

    # Get k range
    if type(k_ranges) == int or type(k_ranges) == type(None):
        k_range = DEFAULT_K_RANGE
        if type(k_ranges) == int:  # Can input 3 for example, and get -3, -2, -1, 0, 1, 2, 3 as a range for each basis vec
            k_range = k_ranges

            k_ranges = [range(1 - k_range, k_range) for _i in range(len(basis))]
            if basis_obj.is_2d:
                k_ranges[-1] = [0]

    # Get each set of parallel planes
    plane_sets = [ PlaneSet(e, offsets[i], i, k_ranges[i]) for (i, e) in enumerate(basis) ]

    rhombohedra = {}
    for possible_volume in possible_cells.keys():
        rhombohedra[possible_volume] = []

    # Find intersections between each of the plane sets
    for p in range(len(basis) - 2):
        for q in range(p+1, len(basis)-1):
            for r in range(q+1, len(basis)):
                intersections = plane_sets[p].get_intersections_with(plane_sets[q], plane_sets[r])
                # DEBUG print("Intersections between plane sets p:%s, q:%s, r:%s : %d" % (p, q, r, len(intersections)))
                for i in intersections:
                    # Calculate neighbours for this intersection
                    indices_set = get_neighbours(i, basis, offsets)
                    vertices_set = []

                    for indices in indices_set:
                        vertex = realspace(indices, basis)
                        # DEBUG print("Vertex output for %s:\t%s" % (indices, vertex))
                        vertices_set.append(vertex)

                    vertices_set = np.array(vertices_set)
                    r = Rhombahedron(vertices_set, indices_set, i["js"])
                    # Get volume and append to appropriate rhombohedra list
                    volume = r.get_volume()
                    volume = np.around(volume, shape_accuracy)
                    rhombohedra[volume].append(r)

    return rhombohedra, possible_cells
