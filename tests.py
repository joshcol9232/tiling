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
    # Special cases. E.g angle = 0
    if set_angles[j][0] == 0:
        plt.axvline(x=( (k - sigma)/set_angles[j][2] ))
    else:
        plt.plot(xspace, (k - sigma - xspace * set_angles[j][2])/set_angles[j][1])

xspace = np.linspace(-7, 7)

for j in range(5):
    # for k in range(10):
    draw_construction_line(xspace, j, 1, 0.1, set_angles)



plt.show()