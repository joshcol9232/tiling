import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

ANGLE_OFFSET = np.pi/2.0

class Intersection:
    """
    Takes r vector position, and the j and k values of the lines that intersect each other
    """
    def __init__(self, j1, k1, j2, k2, sigma1, sigma2, symmetry=5):  # Init calculates the intersection from given params
        self.r = self.find_intersection(j1, k1, j2, k2, sigma1, sigma2, symmetry)
        self.j1 = j1
        self.j2 = j2
        self.k1 = k1
        self.k2 = k2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __repr__(self):
        return "%s" % self.r

    def find_intersection(self, j1, k1, j2, k2, sigma1, sigma2, symmetry=5, angle_offset=ANGLE_OFFSET):
        """
        Returns position vector of intersection between line j1,k1 and j2,k2
        """
        a1 = (j1 * 2.0 * np.pi/symmetry) + angle_offset
        a2 = (j2 * 2.0 * np.pi/symmetry) + angle_offset
        print("Angles: ", a1, a2)
        x = ( (k1 - sigma1)/np.sin(a1) - (k2 - sigma2)/np.sin(a2) ) / ( (1.0/np.tan(a1)) - (1.0/np.tan(a2)) )
        return np.array([x, construction_line(x, j1, k1, sigma1, symmetry=5)])


def construction_line(x, j, k, sigma, symmetry=5, angle_offset=ANGLE_OFFSET):
    angle = (j * 2.0 * np.pi/symmetry) + angle_offset
    return (k - sigma - x * np.cos(angle)) / np.sin(angle)


"""
Plan:
1) Find intersections of line with another
2) Calculate what indices this line is by finding the neighbouring lines to that point.
3) Use those indices in de Bruijn's formula to find the vertices in 2D space.
"""


xspace = np.linspace(-5, 5)
rng = np.random.default_rng(32187)

K_RANGE = 3
J_RANGE = 5

sigmas = []


for j in range(J_RANGE):
    offset = 1 + rng.random() * 2
    sigmas.append(offset)
    for k in range(K_RANGE):
        linestyle = "-"
        if j == 1 and k == 1:
            linestyle = "--"

        plt.plot(xspace, construction_line(xspace, j, k, offset), linestyle)

plt.legend()
# Just find intersections along one line for now
# Let's choose j1 = 1, k1 = 1 and compare that with every other line
# Haven't drawn set 0 properly yet so set 1 is fine
x_intersections = []
y_intersections = []

for j2 in range(J_RANGE):   
    if j2 != 1: # every set apart from set j1 = 1
        for k2 in range(K_RANGE):   # Go through each line of the set j2
            intersection = Intersection(1, 1, j2, k2, sigmas[1], sigmas[j2])
            print(intersection)
            x_intersections.append(intersection.r[0])
            y_intersections.append(intersection.r[1])

plt.plot(x_intersections, y_intersections, "xr")
plt.show()