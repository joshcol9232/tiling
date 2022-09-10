# tiling
Tool for generating arbitrary rhombic tilings in arbitrary numbers of dimensions, based on the de Bruijn grid method.

Please see `final_report.pdf` for a complete overview of the method implemented in the code if you are interested.

If you find any issues/want to request a feature, please don't hesitate to contact me here: joshcol9232@gmail.com.

- 11-Fold rotationally symmetric tiling.
![11-Fold](11-fold_output.png?raw=true "11-Fold rotationally symmetric tiling.")

- Extracted section of icosahedral quasicrystal viewed down an axis of symmetry.
![3D Icosahedral](icosahedral_quasi_output.png?raw=true "Icosahedral quasicrystal seen down an axis of symmetry.")


## Usage:

The `dualgrid` module contains the method itself, which will return a list of rhombohedra. There are then various
methods in the `utils` module that enable you to render the shapes, save them to an STL file, and choose
pre-defined bases. See `main.py` to get started, or the other examples in the folder.

Note: Ensure that `main.py` (or your Python file) is being executed from within `tiling-main/` (or the folder containing the `dualgrid` folder), as the Python interpreter needs to be able to find the `dualgrid` module.

## Dependencies:

- `numpy`
- `matplotlib`
- `networkx`
- `pygmsh`

## Collaboration, making changes:

Please raise an issue if a bug is found, or if a potential new feature should be discussed.
For making changes to the code, please open a separate branch, following the git flow pattern:

i.e Make a new branch for a new feature: `feature/new_feature`, a bug: `bugfix/name_of_bug`, etc.

Then please open a pull request for review.

## Notes:

- Filtering the generated nodes into a smaller region is important so that outliers are not included in the graph. E.g tiles that are not connected to the rest of the tiling - generate a 2D Penrose tiling without applying a filter and zoom out if you want to see for yourself. This is one minor caveat of the de Bruijn dualgrid method, but it is easily remedied by filtering.
