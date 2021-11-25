import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time
import sys
from matplotlib import cm
import threading


SYMMETRY = 5
# Default symmetry can be set above, but can also be passed in as argument to the program
# e.g `python penrose.py 8` -> make Ammann-Beenker tiling (8-fold symmetry)
if len(sys.argv) > 1:
    SYMMETRY = int(sys.argv[1])

ANGLE_OFFSET = 0.05         # Prevents divisions by 0 etcetc. Angle offset is undone at the end
K_RANGE = 20   # Number of lines per construction line set (in both directions)
USE_RANDOM_SIGMA = True
COLOUR = True       # Use colour? Colour is based on the smallest internal angle of the rhombus
PLOT_CONSTRUCTION = False       # Plot construction lines beforehand? (Useful for debugging)

SQUARE_ACCURACY_RANGE = 0.001   # Detection of if the rhombus is actually a square needs some accuracy


def construction_line(x, j, k, sigma, symmetry=SYMMETRY, angle_offset=ANGLE_OFFSET):
    angle = (j * 2.0 * np.pi/symmetry) + angle_offset
    return (k - sigma - x * np.cos(angle)) / np.sin(angle)


def get_indices(r, sigmas, es):
    """
    Returns the indices for any point on the plane.
    [a j0, b j1, c j2, d j3, e j4] where a,b,c,d,e are integers.
    `es` are the 5 normal vectors in real space that are separated by 2pi/5.
    """
    
    # Dot product with the unit vector will return the index without the original sigma shift, so then add the shift.
    # Note that since k is an integer, the distance between the lines is just 1 so it is already normalised.
    return np.array( [int(np.ceil(np.dot( r, es[i] ) + sigmas[i] )) for i in range(len(es))] )



def get_angle_index(angle):
    """
    Converts an angle in radians into a normalised value from 0.0 to 1.0. Palette repeats every pi/2 (or 90 degrees).
    It's purpose is for colouring rhombuses based on their smallest internal angle.
    """
    # NOTE: Only need to measure 1 internal angle. If the internal angle is bigger than 90, then you can just take away 90 degrees
    # to get the smallest angle in the rhombus. This is equivalent to doing the modulus of the angle with pi/2
    # Also need to account for the special case of a perfect square
    
    if angle > np.pi/2.0 - SQUARE_ACCURACY_RANGE and angle < np.pi/2.0 + SQUARE_ACCURACY_RANGE: # If close to 90 degrees then it is a square
        angle_index = 1.0
    else:
        pio2 = np.pi/2.0
        a = angle % pio2
        angle_index = a/pio2

    return np.around(angle_index, 5)   # Round it to a couple of sig figs so that colours for one rhombus is uniform



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

    def find_surrounding_indices(self, sigmas, es):
        """
        Finds the indices for the spaces surrounding this intersection.
        Each intersection will be the corner between 4 spaces. Those 4 spaces will be either side
        of the two lines that intersected.
        """
        # Get indices that the point is located at
        point_indices = get_indices(self.r, sigmas, es)
        point_indices[self.j1] = self.k1  # These are already known
        point_indices[self.j2] = self.k2
        # The indices of the spaces surrounding the points are then (for j1 = 0, j2 = 1):
        # [k1 + 1 or 0, k2 + 1 or 0, point_indices...]
        # There should be 4 total
        # had to copy each member of point_indices due to it just putting the array in by reference each time
        surrounding_indices = np.array([np.array([point_indices[j] for j in range(len(point_indices))]) for i in range(4)])
        # Do each permutation
        surrounding_indices[1][self.j1] += 1
        surrounding_indices[3][self.j2] += 1
        surrounding_indices[2][self.j1] += 1
        surrounding_indices[2][self.j2] += 1
        return surrounding_indices


class Rhombus:
    """
    Struct used for drawing a rhombus. Just holds data for vertices and the colour.
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
        vertex += es[i % len(es)] * indices[i]

    return vertex

def generate_sigma(j_range, random=USE_RANDOM_SIGMA):
    """
    Generates offsets of each set of lines.
    These offsets must sum to 0.
    """
    sigma = []
    rng = np.random.default_rng(32187)
    if random:
        rng = np.random.default_rng( int(time.time() * 10) )

    for i in range(j_range-1):
        sigma.append(rng.random())
    # Sum of all sigma needs to equal 0
    s = np.sum(np.array(sigma))
    sigma.append(-s)
    return np.array(sigma)


def make_rhombus_from_intersection(i, sigmas, es):
    """
    Generates rhombus from intersection "i"
    """
    indices_set = i.find_surrounding_indices(sigmas, es)

    vset = [vertex_position_from_pentagrid(i, es_no_offset) for i in indices_set]

    rhombus_colour = None
    if COLOUR:
        # Find an internal angle to find out what rhombus it is. Take 3 points and use dot product.
        # Copy them to new variables. Bear in mind that the sides will be vectors relative
        # to the second point (the middle one, think of a v shape).
        side1 = vset[0] - vset[1]
        side2 = vset[2] - vset[1]
        sidedot = np.dot(side1, side2)
        angle = np.arccos( np.linalg.norm(sidedot) ) # normalise the vector and then cos^{-1} is the angle.

        angle_index = get_angle_index(angle)    # Number between 0 and 1
        rhombus_colour = colour_palette(angle_index)

    return Rhombus(vset, rhombus_colour)


j_range = SYMMETRY
if SYMMETRY % 2 == 0:
    j_range /= 2
    j_range = int(j_range)

# Define normal unit vectors for each of the sets. Required for finding indices
es_no_offset = [np.array([ np.cos( j * 2 * np.pi/SYMMETRY ), np.sin( j * 2 * np.pi/SYMMETRY ) ]) for j in range(j_range)]
# Define with an offset so that finding intersections does not result in maths errors, as one of the vectors will be
# completely vertical otherwise.
es = [np.array([ np.cos( (j * 2 * np.pi/SYMMETRY) + ANGLE_OFFSET ), np.sin( (j * 2 * np.pi/SYMMETRY) + ANGLE_OFFSET ) ]) for j in range(j_range)]


sigmas = generate_sigma(j_range)     # Generate offsets for each set of lines.
print("Offset sum:", np.sum(sigmas))    # Should sum to 0

x_intersections = []
y_intersections = []
intersections = []

for j1 in range(j_range):
    for j2 in range(j1 + 1, j_range):   # Compares 0 1, 0 2, 0 3, 0 4, 1 2, 1 3, ... 3 4
        for k1 in range(-K_RANGE, K_RANGE):
            for k2 in range(-K_RANGE, K_RANGE):   # Go through each line of the set j2
                intersection = Intersection(j1, k1, j2, k2, sigmas[j1], sigmas[j2])
                intersections.append(intersection)
                x_intersections.append(intersection.r[0])
                y_intersections.append(intersection.r[1])

print("Found %s intersections." % len(intersections))

# Plot construction lines to debug beforehand (optional)
if PLOT_CONSTRUCTION:
    plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio
    line_set_colours = ["r", "g", "b", "y", "m", "c", "k"]

    xspace = np.linspace(-20, 20)
    for j in range(j_range):
        for k in range(-K_RANGE, K_RANGE):
            plt.plot( xspace, construction_line(xspace, j, k, sigmas[j]), color=line_set_colours[j % len(line_set_colours)] )

    plt.plot(x_intersections, y_intersections, ".r", label="Intersections")
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.legend()
    plt.title("de Bruijn Construction Lines")
	
    plt.show()


rhombuses = []      # List of Rhombus objects
colour_palette = cm.get_cmap("viridis", 8)  # Use a numpy colour palette

for i in intersections:
    # Each line intersection corresponds to a single rhombus.
    rhombuses.append( make_rhombus_from_intersection(i, sigmas, es) )

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.axis("equal")

shapes = {}     # Dictionary of colours already found. Used to group rhombuses into PatchCollection objects
for r in rhombuses:
    if r.colour not in shapes.keys():
        shapes[r.colour] = [Polygon(r.vertices)]
    else:
        shapes[r.colour].append(Polygon(r.vertices))

for colour, shape in shapes.items():
    # Add to plot
    shape_coll = PatchCollection(shape, edgecolor="k", facecolor=colour, linewidth=0.4, antialiased=True)
    ax.add_collection(shape_coll)



plotrange = 30
plt.xlim(-plotrange, plotrange)
plt.ylim(-plotrange, plotrange)
plt.gca().set_aspect("equal")   # Make sure plot is in an equal aspect ratio
plt.title("%d-fold symmetry." % SYMMETRY)
plt.show()
