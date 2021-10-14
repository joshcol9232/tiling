import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

DELTA_SIGMA = 0.1   # Distance between lines of a set
SIDE_LENGTH = 1

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

xspace = np.linspace(-3, 3)

for j in range(5):
    for k in range(3):
        draw_construction_line(xspace, j, k, 1.2, set_angles)



plt.show()