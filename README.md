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

- Filtering the generated nodes into a smaller region is important so that outliers are not included in the graph. E.g tiles that are not connected to the rest of the tiling - generate a 2D Penrose tiling without applying a filter and zoom out if you want to see for yourself. This is one minor caveat of the de Bruijn dualgrid method, but it is easily remedied by filtering.