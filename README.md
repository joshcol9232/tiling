# tiling
Tool for generating arbitrary rhombic tilings in arbitrary numbers of dimensions, based on the de Bruijn grid method.

WORK IN PROGRESS: Breaking changes may be made in the early stages.

- 11-Fold rotationally symmetric tiling.
![11-Fold](11-fold_output.png?raw=true "11-Fold rotationally symmetric tiling.")

- Extracted section of icosahedral quasicrystal viewed down an axis of symmetry.
![3D Icosahedral](icosahedral_quasi_output.png?raw=true "Icosahedral quasicrystal seen down an axis of symmetry.")


## Usage:

The `dualgrid` module contains the method itself, which will return a list of rhombohedra. There are then various
methods in the `utils` module that enable you to render the shapes, save them to an STL file, and choose
pre-defined bases. See `main.py` for some examples.

Ensure that `main.py` (or your Python file) is being executed from within `tiling-main/` (or the folder containing the `dualgrid` folder), as the Python interpreter needs to be able to find the `dualgrid` module.

## Dependencies:

- `numpy`
- `matplotlib`
- `networkx`
- `pygmsh`

## Notes:

- Filtering the generated nodes into a smaller region is important so that outliers are not included in the graph. E.g tiles that are not connected to the rest of the tiling - generate a 2D Penrose tiling without applying a filter and zoom out if you want to see for yourself. This is one minor caveat of the de Bruijn dualgrid method, but it is easily remedied by filtering.
