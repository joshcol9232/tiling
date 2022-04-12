# tiling
Tool for generating arbitrary rhombic tilings in arbitrary numbers of dimensions, based on the de Bruijn grid method.

## Usage:

The `dualgrid` module contains the method itself, which will return a list of rhombohedra. There are then various
methods in the `utils` module that enable you to render the shapes, save them to an STL file, and choose
pre-defined bases. See `main.py` for some examples.

## Dependencies:

- `numpy`
- `matplotlib`
- `networkx`
- `pygmesh`

## Notes:

- Invalid edges may be generated at the edge of the generated tiling (due to lone tiles on the outside being connected to the rest of the tiling by an edge that may not be valid).
    Therefore ensure you use a large enough k range so that the portion you filter out is not at the edge,
    and hence has valid edges.
