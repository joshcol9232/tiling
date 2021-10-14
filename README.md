# tiling
Tool for generating various quasi-crystalline patterns in 2D and 3D (perhaps 4D soon).

## Aims:

To provide capability to generate arbitrary tiling data using the de Bruijn grid method in:
    - 2D (Done?: No)
    - 3D (Done?: No)
    - 4D (Done?: No)

## Plan:

Create a python module that generates the required data to form the aperiodic tilings along with their decoration (e.g filled tiles, filled volumes etc). Just a black box that you put a request into, and it gives you graph data out. More powerful/useful as it can just be used as a standalone python module.

Then perhaps have a second utility module that builds on top of the first which has the capability to render the points as images (through matplotlib), or as 3D printer files (something like `stl`, `obj` or `vrml` files). Not sure how 4D objects are usually rendered yet.
