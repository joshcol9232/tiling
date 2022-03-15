import numpy as np
import matplotlib.pyplot as plt
import time

RANDOM = True
OFFSET_SUM_ZERO = True
K_RANGE = 4
RENDER_DISTANCE = 5.0

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


    def get_intersections_with(self, other1, other2, k_range):
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
        # for k1 in [0]:
        #     for k2 in [0]:
        #         for k3 in [0]:
                    # Multiply by remaining part of catesian form (d)
                    ds = np.matrix([
                        [self.offset + k1],
                        [other1.offset + k2],
                        [other2.offset + k3]
                    ])  # Last row of the matrix -> i.e last element of cartesian form ax + bx + c = d, `d`
                    xyz = np.matmul(coef_inv, ds)
                    intersections.append({
                        "location": np.array([xyz[0, 0], xyz[1, 0], xyz[2, 0]]),
                        "ks": [k1, k2, k3],
                        "js": [self.setnum, other1.setnum, other2.setnum],
                    })

        return intersections


def get_offsets(n, is_random, sum_to_zero, is_2d=False):
    if is_random:
        offsets = []

        rng = np.random.default_rng(int(time.time() * 10))
        for i in range(n - 2):
            offsets.append(rng.random())

        if sum_to_zero:
            if not is_2d:
                offsets.append(rng.random())
            # Sum of all sigma needs to equal 0
            s = np.sum(np.array(offsets))
            offsets.append(-s)
            if is_2d:
                offsets.append(0.0)
        else:
            for _i in range(2):
                offsets.append(rng.random())

        return offsets
    else:
        if is_2d:
            a = [-1.0/len(basis) for _i in range(n-1)]
            a.append(0.0)
            return a
        else:
            return [-1.0 / len(basis) for _i in range(n)]  # Centre of rotational symmetry in penrose-like case

def realspace(indices, basis):
    out = np.zeros(3, dtype=float)
    for j, e in enumerate(basis):
        out += e * indices[j]

    # Remove float error
    # for k in range(3):
    #     if abs(out[k]) < 1e-14:
    #         out[k] = 0.0

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

    print("Root indices before loading:", indices)
    # # Load known indices into indices array
    for index, j in enumerate(intersection["js"]):
        indices[j] = intersection["ks"][index]

    print("Getting neighbours for:", intersection)
    print("Root indices:", indices)
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

# def get_neighbours_via_sample(intersection, basis, offsets, m):
#     # Uses multiplier `m`
#     indices = []
#     for i in range(8):



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

"""
BASES
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
    return np.array(icos)

def cubic_basis():
    return np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0])
    ])

def penrose_basis():
    penrose = [np.array([np.cos(j * np.pi * 2.0 / 5.0), np.sin(j * np.pi * 2.0 / 5.0), 0.0]) for j in range(5)]
    penrose.append(np.array([0.0, 0.0, 1.0]))
    return np.array(penrose)

def ammann_basis():
    am = [np.array([np.cos(j * np.pi * 2.0 / 8.0), np.sin(j * np.pi * 2.0 / 8.0), 0.0]) for j in range(4)]
    am.append(np.array([0.0, 0.0, 1.0]))
    return np.array(am)

def hexagonal_basis():
    return np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0/2.0, np.sqrt(3)/2.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])

def test_basis():
    # return np.array([
    #     np.array([1.0, 0.0, 0.0]),
    #     np.array([1.0/2.0, np.sqrt(3)/2.0, 0.0]),
    #     np.array([np.sqrt(3) / 2.0, 1.0 / 2.0, 0.0]),
    #     np.array([0.0, 0.0, 1.0]),
    # ])
    return np.array([
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([np.cos(1.5), np.sin(1.5), 0.0]),
        np.array([np.cos(1.1), np.sin(1.1), 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ])


FACES = [  # Faces of every rhombahedron (ACW order). Worked out on paper
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


if __name__ == "__main__":
    get_basis = penrose_basis
    basis = get_basis()
    _largest_disp, largest_disp_mag = get_largest_node_displacement(basis)

    print("BASIS:", basis)
    offsets = get_offsets(len(basis), RANDOM, OFFSET_SUM_ZERO, is_2d=True)

    """ Penrose test offsets
    offsets = [
        0.321,
        0.56321,
        0.11345,
        0.744,
        -(0.744 + 0.11345 + 0.56321 + 0.321),
        0.0,
    ]
    """
    """ test offsets
    offsets = [
        0.321,
        0.56321,
        -(0.321 + 0.56321),
        0.0,#0.744,
    ]
    # offsets = np.zeros(4)
    """

    # default k ranges
    # k_ranges = [range(1 - K_RANGE, K_RANGE) for _i in range(len(basis))]

    # pre-define k ranges for testing ### DEBUG
    k_ranges = [
        # [-1, 0, 1],
        # [-1, 0, 1],
        # [-1, 0, 1],
        # [-1, 0, 1],
        # [-1, 0, 1],
        # [0],
        # [0],
        # [0],
        # [0],
        range(-5, 5),
        range(-5, 5),
        range(-5, 5),
        range(-5, 5),
        range(-5, 5),
        [0],
    ]

    # Get each set of parallel planes
    planesets = [ PlaneSet(e, offsets[i], i, k_ranges[i]) for (i, e) in enumerate(basis) ]

    all_intersections = []
    all_original_indices = []
    vertices = []
    in_render_dist_range = []

    ax = plt.axes(projection="3d")
    # ax.set_box_aspect(aspect=(1, 1, 1))

    colourmap = plt.get_cmap("viridis")

    # Find intersections between each of the planesets
    for p in range(len(basis) - 2):
        for q in range(p+1, len(basis)-1):
            for r in range(q+1, len(basis)):
                intersections = planesets[p].get_intersections_with(planesets[q], planesets[r], K_RANGE)
                print("Intersections between plane sets p:%s, q:%s, r:%s : %d" % (p, q, r, len(intersections)))
                for i in intersections:
                    all_intersections.append(i)
                    # Calculate neighbours for this intersection
                    indices_set = get_neighbours(i, basis, offsets)
                    vertices_set = []
                    # NOTE: debug
                    # if np.linalg.norm(indices) > 1000:
                    #     print("HUGE INDEX", i)
                    #     print(indices)

                    for indices in indices_set:
                        vertex = realspace(indices, basis)
                        print("Vertex output for %s:\t%s" % (indices, vertex))
                        # print("Indices, Vertex:", indices, vertex)
                        vertices_set.append(vertex)
                        vertices.append(vertex)

                    # Plot connections if in render distance
                    in_range = False
                    for v in vertices_set:
                        if np.linalg.norm(v) < RENDER_DISTANCE:
                            in_range = True
                            break

                    for _i in range(8):
                        in_render_dist_range.append(in_range)

                    if in_range:
                        # Get colour
                        volume = abs(triple_product(vertices_set[1] - vertices_set[0], vertices_set[2] - vertices_set[0], vertices_set[4] - vertices_set[0]))
                        col = colourmap(volume)
                        print("Vol:", volume)
                        for edge in EDGES:
                            v1 = vertices_set[edge[0]]
                            v2 = vertices_set[edge[1]]
                            ax.plot([v2[0], v1[0]], [v2[1], v1[1]], [v2[2], v1[2]], "-", color=col)

                print()

    vertices = np.array(vertices)

    vertices_in_range = vertices[in_render_dist_range]
    ax.plot(vertices_in_range[:,0], vertices_in_range[:,1], vertices_in_range[:,2], "k.", markersize=10)

    # Set axis scaling equal
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1] - world_limits[0], world_limits[3] - world_limits[2], world_limits[5] - world_limits[4]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()