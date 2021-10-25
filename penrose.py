import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time
import sys

SYMMETRY = 5
if len(sys.argv) > 1:
    SYMMETRY = int(sys.argv[1])

ANGLE_OFFSET = 0.04         # Prevents divisions by 0 etcetc. Angle offset is undone at the end
K_RANGE = 10   # In both directions
USE_RANDOM_SIGMA = True
COLOUR = True       # Use colour? Colour is based on the smallest internal angle of the rhombus
PLOT_CONSTRUCTION = True       # Plot construction lines beforehand? (Useful for debugging)

COS_ANGLE_OFF_INV = np.cos(-ANGLE_OFFSET)
SIN_ANGLE_OFF_INV = np.sin(-ANGLE_OFFSET)
ANGLE_OFF_ROT_MAT_INV = np.array([[COS_ANGLE_OFF_INV, -SIN_ANGLE_OFF_INV], [SIN_ANGLE_OFF_INV, COS_ANGLE_OFF_INV]])

def construction_line(x, j, k, sigma, symmetry=SYMMETRY, angle_offset=ANGLE_OFFSET):
    angle = (j * 2.0 * np.pi/symmetry) + angle_offset
    return (k - sigma - x * np.cos(angle)) / np.sin(angle)


def get_indices(r, sigmas, es, symmetry=SYMMETRY, angle_offset=ANGLE_OFFSET):
    """
    Returns the indices for any point on the plane.
    [a j0, b j1, c j2, d j3, e j4] where a,b,c,d,e are integers.
    `es` are the 5 normal vectors in real space that are separated by 2pi/5.
    """
    
    # Dot product with the unit vector will return the index without the original sigma shift, so then add the shift.
    # Note that since k is an integer, the distance between the lines is just 1 so it is already normalised.
    indices = np.zeros(int(symmetry), dtype=int)
    i = 0
    for e in es:
        indices[i] = int(np.ceil(np.dot(r, e) + sigmas[i]))

        i += 1

    return indices



def colour_palette(angle):
    """
    Converts an angle in radians into an RGB value from 0.0 to 1.0. Palette repeats every pi/2 (or 90 degrees).
    It's purpose is for colouring rhombuses based on their smallest internal angle.
    """
    # NOTE: Only need to measure 1 internal angle. If the internal angle is bigger than 90, then you can just take away 90 degrees
    # to get the smallest angle in the rhombus. This is equivalent to doing the modulus of the angle with pi/2
    # Also need to account for the special case of a perfect square
    
    if angle > np.pi/2.0 - 0.001 and angle < np.pi/2.0 + 0.001: # If close to 90 degrees then it is a square
        angle_index = 1.0
    else:
        pio2 = np.pi/2.0
        a = angle % pio2
        angle_index = a/pio2

    return (angle_index, 1.0 - angle_index, 0.0)



class Intersection:
    """
    Takes r vector position, and the j and k values of the lines that intersect each other
    """
    def __init__(self, j1, k1, j2, k2, sigma1, sigma2, symmetry=SYMMETRY):  # Init calculates the intersection from given params
        self.r = self.find_intersection(j1, k1, j2, k2, sigma1, sigma2, symmetry)
        self.j1 = j1
        self.j2 = j2
        self.k1 = k1
        self.k2 = k2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __repr__(self):
        return "%s" % self.r

    def find_intersection(self, j1, k1, j2, k2, sigma1, sigma2, symmetry=SYMMETRY, angle_offset=ANGLE_OFFSET):
        """
        Returns position vector of intersection between line j1,k1 and j2,k2
        """
        a1 = (j1 * 2.0 * np.pi/symmetry) + angle_offset
        a2 = (j2 * 2.0 * np.pi/symmetry) + angle_offset

        x = ( (k1 - sigma1)/np.sin(a1) - (k2 - sigma2)/np.sin(a2) ) / ( (1.0/np.tan(a1)) - (1.0/np.tan(a2)) )

        return np.array([x, construction_line(x, j1, k1, sigma1, symmetry=SYMMETRY)])

    def find_surrounding_indices(self, sigmas, es, symmetry=SYMMETRY, angle_offset=ANGLE_OFFSET):
        """
        Finds the indices for the spaces surrounding this intersection.
        Each intersection will be the corner between 4 spaces. Those 4 spaces will be either side
        of the two lines that intersected.
        """
        # Get indices that the point is located at
        point_indices = get_indices(self.r, sigmas, es, symmetry, angle_offset)
        point_indices[self.j1] = self.k1  # These are already known
        point_indices[self.j2] = self.k2
        # The indices of the spaces surrounding the points are then (for j1 = 0, j2 = 1):
        # [k1 + 1 or 0, k2 + 1 or 0, point_indices...]
        # There should be 4 total
        # had to copy each member of point_indices due to it just putting the array in by reference each time
        surrounding_indices = np.array([np.array([point_indices[j] for j in range(len(point_indices))]) for i in range(4)])
        # Do each permutation
        surrounding_indices[1][self.j1] += 1
        surrounding_indices[2][self.j2] += 1
        surrounding_indices[3][self.j1] += 1
        surrounding_indices[3][self.j2] += 1
        return surrounding_indices


class Rhombus:
    """
    Struct used for drawing a rhombus
    """
    def __init__(self, vertices, colour):
        self.vertices = vertices
        self.colour = colour


def vertex_position_from_pentagrid(indices, es):
    """
    Calculates vertex positions in real space from pentagrid indices.
    """
    vertex = np.zeros(2)    # vector in 2D space
    for i in range(len(indices)):
        vertex += es[i] * indices[i]

    return vertex

def generate_sigma(symmetry=SYMMETRY, random=USE_RANDOM_SIGMA):
    """
    Generates offsets of each set of lines.
    These offsets must sum to 0.
    """
    sigma = []
    rng = np.random.default_rng(32187)
    if random:
        rng = np.random.default_rng( int(time.time() * 10) )

    for i in range(symmetry-1):
        sigma.append(rng.random())
    # Sum of all sigma needs to equal 0
    s = np.sum(np.array(sigma))
    sigma.append(-s)
    return np.array(sigma)

def ccw_sort(v):
    """
    Make sorting function to make sure the polygons draw properly.
    Sorts the points in counter-clockwise order (like a radar)
    """
    mean = np.mean(v, axis=0)
    d = v - mean    # Difference from mean
    s = np.arctan2(d[:,0], d[:,1])
    v_new = []
    for a in np.argsort(s):
        v_new.append(v[a])

    return np.array(v_new)



# Define normal unit vectors for each of the sets. Required for finding indices
es = [np.array([ np.cos( (j * 2 * np.pi/SYMMETRY) + ANGLE_OFFSET ), np.sin( (j * 2 * np.pi/SYMMETRY) + ANGLE_OFFSET ) ]) for j in range(SYMMETRY)]


sigmas = generate_sigma()
# sigmas = np.array([0.2, 0.4, 0.3, -0.8, -0.1])
# sigmas = np.array([0.1, 0.2, 0.3, -0.8, 0.3, -0.1, 0.5, -0.5])
# sigmas = np.zeros(SYMMETRY)
print("Offset sum:", np.sum(sigmas))

# Just find intersections along one line for now
# Let's choose j1 = 1, k1 = 1 and compare that with every other line
# Haven't drawn set 0 properly yet so set 1 is fine
x_intersections = []
y_intersections = []
intersections = []

for j1 in range(SYMMETRY):
    for j2 in range(j1 + 1, SYMMETRY):   # Compares 0 1, 0 2, 0 3, 0 4, 1 2, 1 3, ... 3 4
        for k1 in range(-K_RANGE, K_RANGE):
            for k2 in range(-K_RANGE, K_RANGE):   # Go through each line of the set j2
                intersection = Intersection(j1, k1, j2, k2, sigmas[j1], sigmas[j2])
                intersections.append(intersection)
                x_intersections.append(intersection.r[0])
                y_intersections.append(intersection.r[1])

# ----------------- FOR EXPERIMENTING WITH 1 ONLY -----------
"""
j1 = 0
k1 = 0
for j2 in range(SYMMETRY):
    if j2 != j1:
        for k2 in range(K_RANGE):
            intersection = Intersection(j1, k1, j2, k2, sigmas[j1], sigmas[j2])
            intersections.append(intersection)
            x_intersections.append(intersection.r[0])
            y_intersections.append(intersection.r[1])
"""
# ------------------------------------------------


# Plot construction lines to check beforehand
if PLOT_CONSTRUCTION:
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio
    line_set_colours = ["r", "g", "b", "y", "m", "c", "k"]

    xspace = np.linspace(-20, 20)
    for j in range(SYMMETRY):
        for k in range(-K_RANGE, K_RANGE):
            plt.plot( xspace, construction_line(xspace, j, k, sigmas[j]), color=line_set_colours[j % len(line_set_colours)] )

    plt.plot(x_intersections, y_intersections, "xr")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.show()


indices = [i.find_surrounding_indices(sigmas, es) for i in intersections]


rhombuses = []
colours = {}
i = 0
for indices_set in indices:
    vset = []
    #sums = np.sum(indices_set, axis=1)
    #print(sums)

    for i in indices_set:
        # NOTE: The vertex only exists in the tiling if the sum of the indices is <= SYMMETRY and > 0.
        # http://www.neverendingbooks.org/de-bruijns-pentagrid
        s = np.sum(i)
        if s > 0 and s <= SYMMETRY:
            v = vertex_position_from_pentagrid(i, es)
            vset.append( v )


    # If the vertex set is not empty and has length 4 (all shapes should be 4 sided polygons), append it to the list of vertices
    if len(vset) == 4:
        # Sort the vertices in draw order (anticlockwise).
        vset = ccw_sort(vset)
        # Rotate the vectors to undo the angle offset
        for i in range(4):
            vset[i] = np.dot(ANGLE_OFF_ROT_MAT_INV, vset[i])

        rhombus_colour = None
        if COLOUR:
            # Find an internal angle to find out what rhombus it is. Take 3 points and use dot product.
            # Copy them to new variables. Bear in mind that the sides will be vectors relative
            # to the second point (the middle one, think of a v shape).
            # NOTE: There is most likely a much better way to do this that I haven't thought of enough.
            #       For example, using the fact we know what construction line sets are crossing each other.
            #       For penrose this would be simple (e.g 1, 3 crossing = thick, 1, 2 crossing = thin)
            #       but since this is generalised for different symmetries then it has to work for every case,
            #       which I haven't figured out just yet in terms of indices.
            side1 = vset[0] - vset[1]
            side2 = vset[2] - vset[1]
            sidedot = np.dot(side1, side2)
            angle = np.arccos( np.linalg.norm(sidedot) ) # normalise the vector and then cos^{-1} is the angle.

            rhombus_colour = colour_palette(angle)

        rhombuses.append(Rhombus(vset, rhombus_colour))

    i += 1



fig, ax = plt.subplots(1, figsize=(5, 5), dpi=200)
ax.axis("equal")

""" FOR PLOTTING SHAPES
"""

shapes = {}
for r in rhombuses:
    if r.colour not in shapes.keys():
        shapes[r.colour] = [Polygon(r.vertices)]
    else:
        shapes[r.colour].append(Polygon(r.vertices))

    # p = Polygon(r.vertices, edgecolor="b", facecolor=r.colour, linewidth=2.0, antialiased=False)
    # shapes.append(p)

for colour, shape in shapes.items():
    shape_coll = PatchCollection(shape, edgecolor="k", facecolor=colour, linewidth=0.4, antialiased=True)
    ax.add_collection(shape_coll)


""" FOR DRAWING THE VERTICES ONLY
x = []
y = []
for vertex_set in vertices:
    print(len(vertex_set))
    for v in vertex_set:
        x.append(v[0])
        y.append(v[1])

plt.plot(x, y, ".")
"""


plotrange = 30
plt.xlim(-plotrange, plotrange)
plt.ylim(-plotrange, plotrange)
plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio
plt.show()
