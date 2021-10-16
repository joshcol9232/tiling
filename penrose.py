import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

class Intersection:
    """
    Takes r vector position, and the j and k values of the lines that intersect each other
    """
    def __init__(self, j1, k1, j2, k2, sigma1, sigma2, symmetry=5):  # Init calculates the intersection from given params
        self.r = find_intersection(j1, k1, j2, k2, sigma1, sigma2, symmetry=symmetry)
        self.j1 = j1
        self.j2 = j2
        self.k1 = k1
        self.k2 = k2
        self.sigma1 = sigma1
        self.sigma2 = sigma2

    def __repr__(self):
        return "%s" % self.r

    def find_intersection(j1, k1, j2, k2, sigma1, sigma2, symmetry=5):
        """
        Returns position vector of intersection between line j1,k1 and j2,k2
        """
        a1 = j1 * 2.0 * np.pi/symmetry
        a2 = j2 * 2.0 * np.pi/symmetry
        x = ( (k1 - sigma1)/np.cos(a1) ) - ( (k2 - sigma2)/np.sin(a2) )
        return np.array([x, construction_line(x, j1, k1, sigma1, symmetry=5)])


def construction_line(x, j, k, sigma, symmetry=5):
    angle = j * 2.0 * np.pi/symmetry
    return (k - sigma - x * np.cos(angle)) / np.sin(angle)


"""
Plan:
1) Try to at least generate a conway worm
"""


xspace = np.linspace(-5, 5)
rng = np.random.default_rng(32187)

K_RANGE = 3
J_RANGE = 5

sigmas = []

js = [j+0.25 for j in range(J_RANGE)]

for j in js:
    offset = 1 + rng.random() * 2
    sigmas.append(offset)
    for k in range(K_RANGE):
        plt.plot(xspace, construction_line(xspace, j, k, offset))


# Just find intersections along one line for now
# Let's choose j1 = 0, k1 = 0 and compare that with every other line
for j2 in js[1:]:   # Every set apart from set 0
    for k2 in range(K_RANGE):   # Go through each line of the set j2
        intersection = Intersection()
    

plt.show()