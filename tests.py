import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

DELTA_SIGMA = 0.1   # Distance between lines of a set

def construction_line(x, j, k, sigma):
    """
    Function of the k'th grid line of set j, with normal offset sigma. 
    """
    s = j * 2 * np.pi/5
    return (k - sigma - x * np.cos(s))/np.sin(s)


xspace = np.linspace(-7, 7)

for j in range(5):
    plt.plot( xspace, construction_line(xspace, j, 1, 0.1) )


plt.show()

# plt.axes()
# line = plt.Line2D((0.5, 10), (1, 20), lw=1.5)
# plt.gca().add_line(line)
# plt.axis("scaled")
# plt.show()
