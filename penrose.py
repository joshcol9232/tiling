import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

SIDE_LENGTH = 1
J_RANGE = 5
K_RANGE = 1
SIGMA = 0.5

    

def draw_construction_line(xspace, j, k, sigma):
    """
    Function of the k'th grid line of set j, with normal offset sigma. 
    """
    angle = j * 2 * np.pi/5
    s = np.sin(angle)
    c = np.cos(angle)
    # Special cases. E.g angle = 0 or 90 degrees
    if c == 0:
        print("Vertical")
        plt.axhline(y=( (k - sigma)/s ))
    elif s == 0:
        yval = (k - sigma)/s
        plt.plot(xspace, [yval for i in range(len(xspace))])
    else:
        plt.plot(xspace, (k - sigma - xspace * c)/s)


def find_crossing(j1, j2, k1, k2, sigmaj1, sigmaj2):
    """
    Finds the crossing points between two of the lines given.
    Lines of different sets *will* cross once because they are each infinite lines, and each line
    is at some angle of a circle. No two lines of *different* sets will be parallel.
    """
    a1 = j1 * 2 * np.pi/5
    a2 = j2 * 2 * np.pi/5
    # y = ( np.cos(a2) * (k1 - sigmaj1) - np.cos(a1) * (k2 - sigmaj2) )/np.sin( a2 - a1 )
    # x = (k2 - sigmaj2 - y * np.sin(a2))/np.cos(a2)
    mu1 = k1 - sigmaj1
    mu2 = k2 - sigmaj2
    c1 = np.cos(a1)
    c2 = np.cos(a2)
    s1 = np.sin(a1)
    s2 = np.sin(a2)

    x = mu1/c1 + mu2/c2 - (mu1 * s2)/(s1 * c2) - (mu2 * s1)/(s2 * c1)
    y = mu1 - x * c1/s1
    return x, y

xspace = np.linspace(-11, 11)

for j in range(J_RANGE):
    for k in range(K_RANGE):
        draw_construction_line(xspace, j, k, SIGMA)

# Find where the lines cross. Go through each set and compare against each other line in each other set.
# There is probably a much better way to do this but this will do for now as a concept
x_crossings = []
y_crossings = []

for j1 in range(J_RANGE): # Compare each set with each other set
    for j2 in range(j1+1, J_RANGE):   # If at j1 = 3, we would have already compared 3-0, 3-1, 3-2.
        for k1 in range(K_RANGE):
            for k2 in range(K_RANGE):
                # Compare k1 with k2 of set j1 and j2 respectively. 
                x, y = find_crossing(j1, j2, k1, k2, SIGMA, SIGMA)
                print("Crossing found:", x, y)
                x_crossings.append(x)
                y_crossings.append(y)

print("Number of crossings found:", len(x_crossings))
plt.plot(x_crossings, y_crossings, "x")

plt.show()