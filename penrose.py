import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

SIDE_LENGTH = 1
J_RANGE = 5
K_RANGE = 1
SIGMA = 1.2

# Calculate angle, sin and cos values between each set beforehand
set_angles = [((j/5) * 2 * np.pi, np.sin((j/5) * 2 * np.pi), np.cos((j/5) * 2 * np.pi)) for j in range(5)]
print(set_angles)

def draw_construction_line(xspace, j, k, sigma, set_angles):
    """
    Function of the k'th grid line of set j, with normal offset sigma. 
    """
    # Special cases. E.g angle = 0 or 90 degrees
    if set_angles[j][1] == 0 or set_angles[j][2] == 0:
        plt.axvline(x=( (k - sigma)/set_angles[j][2] ))
    else:
        plt.plot(xspace, (k - sigma - xspace * set_angles[j][2])/set_angles[j][1])

def find_crossing(j1, j2, k1, k2, sigmaj1, sigmaj2):
    y = ( np.cos(j2 * 2 * np.pi/5) * (k1 - sigmaj1) - np.cos(j1 * 2 * np.pi/5) * (k2 - sigmaj2) )/np.sin( (j2 - j1) * 2 * np.pi/5 )
    x = (k1 - sigmaj1 - y * np.sin(j1 * 2 * np.pi/5))/np.cos(j1 * 2 * np.pi/5)
    return x, y

xspace = np.linspace(-5, 5)
crossings = []



for j in range(J_RANGE):
    for k in range(K_RANGE):
        draw_construction_line(xspace, j, k, SIGMA, set_angles)

# Find where the lines cross. Go through each set and compare against each other line in each other set.
# There is probably a much better way to do this but this will do for now as a concept
for j1 in range(J_RANGE): # Compare each set with each other set
    for j2 in range(j1, J_RANGE):   # If at j1 = 3, we would have already compared 3-0, 3-1, 3-2.
        for k1 in range(K_RANGE):
            for k2 in range(K_RANGE):



plt.show()