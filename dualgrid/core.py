import numpy as np
import itertools

def _get_k_combos(k_range, dimensions):
    return np.array(list(itertools.product(*[ [k for k in range(1-k_range, k_range)] for _d in range(dimensions) ])))

class ConstructionSet:
    def __init__(self, normal, offset, setnum):
        self.normal = normal
        self.offset = offset
        self.setnum = setnum

    def get_intersections_with(self, k_range, others):
        """
        If this plane intersects with two other planes at a point, this function
        will return the location of this intersection in real space.
        """
        dimensions = len(self.normal)
        # Pack Cartesian coefficients into matrix.
        # E.g ax + by + cz = d.     a, b, c for each
        coef = np.matrix([self.normal, *[ o.normal for o in others ]])

        # Check for singular matrix
        if np.linalg.det(coef) == 0:
            print("WARNING: Unit vectors form singular matrices.")
            return [], []

        # get inverse of coefficient matrix
        coef_inv = np.linalg.inv(coef)

        k_combos = _get_k_combos(k_range, dimensions)

        # last part (d) of Cartiesian form.
        # Pack offsets into N dimensional vector, then + [integers] to get specific planes within set
        base_offsets = np.array([self.offset, *[ o.offset for o in others ]])

        ds = k_combos + base_offsets # remaining part of cartesian form (d)
        intersections = np.asarray( (coef_inv * np.asmatrix(ds).T).T )

        return intersections, k_combos

def _get_neighbours(intersection, js, ks, basis):
    # Each possible neighbour of intersection. See eq. 4.5 in de Bruijn paper
    # For example:
    # [0, 0], [0, 1], [1, 0], [1, 1] for 2D
    directions = np.array(list(itertools.product(*[[0, 1] for _i in range(basis.dimensions)])))

    indices = basis.gridspace(intersection)

    # DEBUG print("Root indices before loading:", indices)
    # Load known indices into indices array
    for index, j in enumerate(js):
        indices[j] = ks[index]

    # DEBUG print("Getting neighbours for:", intersection)
    # DEBUG print("Root indices:", indices)
    # First off copy the intersection indices 8 times
    neighbours = [ np.array([ v for v in indices ]) for _i in range(len(directions)) ]

    # Quick note: Kronecker delta function -> (i == j) = False (0) or True (1) in python. Multiplication of bool is allowed
    deltas = [np.array([(j == js[i]) * 1 for j in range(len(basis.vecs))]) for i in range(basis.dimensions)]

    # Apply equation 4.5 in de Bruijn's paper 1, expanded for any basis len and extra third dimension
    for i, e in enumerate(directions): # e Corresponds to epsilon in paper
        neighbours[i] += np.dot(e, deltas)

    return neighbours

class Basis:
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
        Returns where a point lies in grid space.
        """
        out = np.zeros(len(self.vecs), dtype=int)

        for j, e in enumerate(self.vecs):
            out[j] = int(np.ceil( np.dot( r, self.vecs[j] ) - self.offsets[j] ))

        return out

    def get_possible_cells(self, decimals):
        """ Function that finds all possible cell shapes in the final mesh.
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

""" MAIN ALGORITHM """
class Cell:
    def __init__(self, vertices, indices, parent_sets, intersection):
        self.verts = vertices
        self.indices = indices
        self.parent_sets = parent_sets
        self.intersection = intersection # The intersection corresponding to this cell

    def __repr__(self):
        return "Cell(%s parents %s)" % (self.indices[0], self.parent_sets)

    def is_in_filter(self, *args, **kwargs):
        """ Utility function for checking whever the rhombohedron is in rendering distance
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

    print("FOUND EDGES:", edges)
    return np.array(edges)

def _get_cells_from_construction_sets(construction_sets, js, cells, k_range, basis, shape_accuracy):
    intersections, k_combos = construction_sets[js[0]].get_intersections_with(k_range, [construction_sets[j] for j in js[1:]])
    # DEBUG print("Intersections between plane sets %s : %d" % (js, len(intersections)))

    for i, intersection in enumerate(intersections):
        # Calculate neighbours for this intersection
        indices_set = _get_neighbours(intersection, js, k_combos[i], basis)
        vertices_set = []

        for indices in indices_set:
            vertex = basis.realspace(indices)
            # DEBUG print("Vertex output for %s:\t%s" % (indices, vertex))
            vertices_set.append(vertex)

        vertices_set = np.array(vertices_set)
        c = Cell(vertices_set, indices_set, js, intersection)
        cells.append(c)



def dualgrid_method(basis, k_range, shape_accuracy=4):
    """
    de Bruijn dual grid method.
    Generates and returns cells from basis given in the range given.
    Shape accuracy is the number of decimal places used to classify cell shapes
    Returns: cells, possible cell shapes
    """
    # possible_cells = basis.get_possible_cells(shape_accuracy)

    # Get each set of parallel planes
    construction_sets = [ ConstructionSet(e, basis.offsets[i], i) for (i, e) in enumerate(basis.vecs) ]

    cells = []
    # Find intersections between each of the plane sets
    # NOTE: Could very easily be made multithreaded (future task?).
    for js in itertools.combinations(range(len(construction_sets)), basis.dimensions):
        _get_cells_from_construction_sets(construction_sets, js, cells, k_range, basis, shape_accuracy)

    return cells

