import numpy as np
import itertools
from multiprocessing import Pool
from functools import partial

def _get_k_combos(k_range, dimensions):
    """ 
    Returns all possible comparison between two sets of lines for dimension number "dimensions" and max k_range (index range)
    E.g for 2D with a k range of 1 this is: 

      k_combos = [[-1 -1] [-1  0] [-1  1] [ 0 -1] [ 0  0] [ 0  1] [ 1 -1] [ 1  0] [ 1  1]]
     
    Then, when comparing two 2D construction sets, this compares line (-1) of set 1, with line (-1) of set 2, etc...
    """
    return np.array(list(itertools.product(*[ [k for k in range(1-k_range, k_range)] for _d in range(dimensions) ])))

class ConstructionSet:
    """
    A class to represent a set of parallel lines / planes / n dimensional parallel structure.
    It implements a single method to return all intersections with another ConstructionSet.
    """
    def __init__(self, normal, offset):
        """
        normal: Normal vector to this construction set.
        offset: Offset of these lines from the origin.
        """
        self.normal = normal
        self.offset = offset

    def get_intersections_with(self, k_range, others):
        """
        Calculates all intersections between this set of lines/planes and another.
        """
        dimensions = len(self.normal)
        # Pack Cartesian coefficients into matrix.
        # E.g ax + by + cz = d.     a, b, c for each
        coef_matrix = np.array([self.normal, *[ o.normal for o in others ]])

        # Check for singular matrix
        if np.linalg.det(coef_matrix) == 0:
            print("WARNING: Unit vectors form singular matrices.")
            return [], []

        # get inverse of coefficient matrix
        coef_inv = np.linalg.inv(coef_matrix)

        k_combos = _get_k_combos(k_range, dimensions)

        # last part (d) of Cartiesian form.
        # Pack offsets into N dimensional vector, then + [integers] to get specific planes within set
        base_offsets = np.array([self.offset, *[ o.offset for o in others ]])

        ds = k_combos + base_offsets # remaining part of cartesian form (d)
        intersections = np.asarray( (coef_inv * np.asmatrix(ds).T).T )

        return intersections, k_combos

def _get_neighbours(intersection, js, ks, basis):
    """
    For a given intersection, this function returns the grid-space indices of the spaces surrounding the intersection.
    A "grid-space index" is an N dimensional vector of integer values where N is the number of basis vectors. Each element
    corresponds to an integer multiple of a basis vector, which gives the final location of the tile vertex.

    There will always be a set number of neighbours depending on the number of dimensions. For 2D this is 4 (to form a tile),
    for 3D this is 8 (to form a cube), etc...
    """
    # Each possible neighbour of intersection. See eq. 4.5 in de Bruijn paper
    # For example:
    # [0, 0], [0, 1], [1, 0], [1, 1] for 2D
    directions = np.array(list(itertools.product(*[[0, 1] for _i in range(basis.dimensions)])))

    indices = basis.gridspace(intersection)

    # Load known indices into indices array
    for index, j in enumerate(js):
        indices[j] = ks[index]

    # Copy the intersection indices. This is then incremented for the remaining indices depending on what neighbour it is.
    neighbours = [ np.array([ v for v in indices ]) for _i in range(len(directions)) ]

    # Quick note: Kronecker delta function -> (i == j) = False (0) or True (1) in python. Multiplication of bool is allowed
    # Also from de Bruijn paper 1.
    deltas = [np.array([(j == js[i]) * 1 for j in range(len(basis.vecs))]) for i in range(basis.dimensions)]

    # Apply equation 4.5 in de Bruijn's paper 1, expanded for any basis len and extra third dimension
    for i, e in enumerate(directions): # e Corresponds to epsilon in paper
        neighbours[i] += np.dot(e, deltas)

    return neighbours


class Basis:
    """
    Utility class for defining a set of basis vectors. Has conversion functions between different spaces.
    """
    def __init__(self, vecs, offsets):
        self.vecs = vecs
        self.dimensions = len(self.vecs[0])
        self.offsets = offsets

    def realspace(self, indices):
        """
        Gives position of given indices in real space.
        """
        out = np.zeros(self.dimensions, dtype=float)
        for j, e in enumerate(self.vecs):
            out += e * indices[j]

        return out

    def gridspace(self, r):
        """
        Returns where a "real" point lies in grid space.
        """
        out = np.zeros(len(self.vecs), dtype=int)

        for j, e in enumerate(self.vecs):
            out[j] = int(np.ceil( np.dot( r, self.vecs[j] ) - self.offsets[j] ))

        return out

    def get_possible_cells(self, decimals):
        """
        Function that finds all possible cell shapes in the final mesh.
        Number of decimal places required for finite hash keys (floats are hard to == )
        Returns a dictionary of volume : [all possible combinations of basis vector to get that volume]
        """
        shapes = {}  # volume : set indices

        for inds in itertools.combinations(range(len(self.vecs)), self.dimensions):
            vol = abs(np.linalg.det(np.matrix([self.vecs[j] for j in inds]))) # Determinant ~ volume

            if vol != 0:
                vol = np.around(vol, decimals=decimals)
                if vol not in shapes.keys():
                    shapes[vol] = [inds]
                else:
                    shapes[vol].append(inds)

        return shapes


class Cell:
    """
    Class to hold a set of four vertices, along with additional information
    """
    def __init__(self, vertices, indices, intersection):
        """
        verts: Corner vertices of the real tile/cell.
        indices: The "grid space" indices of each vertex.
        
        """
        self.verts = vertices
        self.indices = indices
        self.intersection = intersection # The intersection which caused this cell's existance. Used for plotting

    def __repr__(self):
        return "Cell(%s)" % (self.indices[0])

    def __eq__(self, other):
        return self.indices == other.indices

    def is_in_filter(self, *args, **kwargs):
        """
        Utility function for checking whever the rhombohedron is in rendering distance
        `fast` just checks the first vertex and exits, otherwise if any of the vertices are inside the filter
        then the whole cell is inside filter
        """
        def run_filter(filter, filter_centre, filter_args=[], filter_indices=False, fast=False, invert_filter=False):
            if fast:
                if filter_indices:
                    return filter(self.indices[0], np.zeros_like(self.indices), *filter_args)
                else:
                    return filter(self.verts[0], filter_centre, *filter_args)
            else:
                if filter_indices:
                    zero_centre = np.zeros_like(self.indices)
                    for i in self.indices:
                        if filter(i, zero_centre, *filter_args):
                            return True
                    return False
                else:
                    for v in self.verts:
                        if filter(v, filter_centre, *filter_args):
                            return True
                    return False
        
        result = run_filter(*args, **kwargs)
        if kwargs["invert_filter"]:
            return not result
        else:
            return result
    
@classmethod
def get_edges_from_indices(indices):
    """
    Gets the edges from vertices given.
    Edges will be found when the indices difference
    has a length of 1. I.e sum(index1 - index2)) = 1
    NOTE: Should be the same for all cells for the particular dimension.
            Therefore it only needs to be run once.
    """
    edges = []
    # Compare every index set with every other index set
    for ind1 in range(len(indices)-1):
        for ind2 in range(ind1+1, len(indices)):
            if abs(np.sum( indices[ind1] - indices[ind2] )) == 1:
                edges.append([ind1, ind2])

    return np.array(edges)

def _get_cells_from_construction_sets(construction_sets, k_range, basis, shape_accuracy, js):
    """
    Retrieves all intersections between the first construction set in the index list, and the rest.
    """
    intersections, k_combos = construction_sets[js[0]].get_intersections_with(k_range, [construction_sets[j] for j in js[1:]])

    cells = []
    for i, intersection in enumerate(intersections):
        # Calculate neighbours for this intersection
        indices_set = _get_neighbours(intersection, js, k_combos[i], basis)
        vertices_set = []

        for indices in indices_set:
            vertex = basis.realspace(indices)
            vertices_set.append(vertex)

        vertices_set = np.array(vertices_set)
        c = Cell(vertices_set, indices_set, intersection)
        cells.append(c)

    return cells

def construction_sets_from_basis(basis):
    return [ ConstructionSet(e, basis.offsets[i]) for (i, e) in enumerate(basis.vecs) ]

def dualgrid_method(basis, k_range, shape_accuracy=4, single_threaded=False):
    """
    de Bruijn dual grid method.
    Generates and returns cells from basis given in the range given.
    Shape accuracy is the number of decimal places used to classify cell shapes
    Returns: cells, possible cell shapes
    """
    # Get each set of parallel planes
    construction_sets = construction_sets_from_basis(basis)

    # `j_combos` corresponds to a list of construction sets to compare. For a 2D basis of length 3, it would be:
    #   (0, 1), (0, 2), (1, 2)
    # Which covers all possible combinations.
    j_combos = itertools.combinations(range(len(construction_sets)), basis.dimensions)

    # Find intersections between each of the plane sets, and retrive cells
    cells = []
    if single_threaded:
        for js in j_combos:
            cells.append(_get_cells_from_construction_sets(construction_sets, k_range, basis, shape_accuracy, js))
    else:
        # Use a `Pool` to distribute work between CPU cores.
        p = Pool()
        work_func = partial(_get_cells_from_construction_sets, construction_sets, k_range, basis, shape_accuracy)
        cells = p.map(work_func, j_combos)
        p.close()

    # Cells is a list of lists -> flatten to a flat 1D list
    return [cell_list for worker_result in cells for cell_list in worker_result]

