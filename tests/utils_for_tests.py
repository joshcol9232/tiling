import dualgrid as dg
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_PATH = "tests/testout/"
INPUT_PATH = "tests/testinput/"

def save_test_figure(filename, cells):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.axis("equal")
    dg.utils.render_cells_solid(cells, ax)

    ax.autoscale(enable=True)  # Zoom out to fit whole tiling
    fig.savefig(OUTPUT_PATH+filename)

def save_verts(filename, verts):
    np.save(OUTPUT_PATH+filename, verts)

def load_verts(filename):
    return np.load(INPUT_PATH+filename)
