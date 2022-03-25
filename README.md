# tiling
Tool for generating generalised aperiodic tilings in 2D, 3D and 4D.

## Aims:

To provide capability to generate arbitrary tiling data using the de Bruijn grid method in:

- 2D (Done?: Yes)

- 3D (Done?: Yes)

- 4D (Done?: No)

- ND (Done?: No)


## Usage:

The `dualgrid` module contains the method itself, which will return a list of rhombohedra. There are then various
methods in the `utils` module that enable you to render the shapes, save them to an STL file (soon), and choose
pre-defined bases. See `example.py` for and example of rendering an icosahedral quasicrystal.
