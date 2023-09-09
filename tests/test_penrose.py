import unittest
import utils_for_tests as testutil

import dualgrid as dg
import numpy as np

PENROSE_BASIS = dg.utils.penrose_basis(random_offsets=False)

class TestPenrose(unittest.TestCase):
    def test_k_max_1(self):
        ## -- KGO
        KGO_VERTS = [[[ 3.09016994e-01, -9.51056516e-01],
                      [ 6.18033989e-01, -1.11022302e-16],
                      [ 1.30901699e+00, -9.51056516e-01],
                      [ 1.61803399e+00, -1.11022302e-16]],

                     [[ 3.09016994e-01,  9.51056516e-01],
                      [-5.00000000e-01,  1.53884177e+00],
                      [ 1.30901699e+00,  9.51056516e-01],
                      [ 5.00000000e-01,  1.53884177e+00]],

                     [[ 3.09016994e-01, -9.51056516e-01],
                      [-5.00000000e-01, -1.53884177e+00],
                      [ 1.30901699e+00, -9.51056516e-01],
                      [ 5.00000000e-01, -1.53884177e+00]],

                     [[ 3.09016994e-01,  9.51056516e-01],
                      [ 6.18033989e-01, -1.11022302e-16],
                      [ 1.30901699e+00,  9.51056516e-01],
                      [ 1.61803399e+00, -1.11022302e-16]],

                     [[-5.00000000e-01, -1.53884177e+00],
                      [-1.30901699e+00, -9.51056516e-01],
                      [-1.90983006e-01, -5.87785252e-01],
                      [-1.00000000e+00,  1.11022302e-16]],

                     [[ 3.09016994e-01, -9.51056516e-01],
                      [-5.00000000e-01, -1.53884177e+00],
                      [ 6.18033989e-01, -1.11022302e-16],
                      [-1.90983006e-01, -5.87785252e-01]],

                     [[-1.61803399e+00,  2.22044605e-16],
                      [-1.30901699e+00, -9.51056516e-01],
                      [-1.30901699e+00,  9.51056516e-01],
                      [-1.00000000e+00,  1.11022302e-16]],

                     [[ 6.18033989e-01, -1.11022302e-16],
                      [-1.90983006e-01, -5.87785252e-01],
                      [-1.90983006e-01,  5.87785252e-01],
                      [-1.00000000e+00,  1.11022302e-16]],

                     [[ 3.09016994e-01,  9.51056516e-01],
                      [ 6.18033989e-01, -1.11022302e-16],
                      [-5.00000000e-01,  1.53884177e+00],
                      [-1.90983006e-01,  5.87785252e-01]],

                     [[-5.00000000e-01,  1.53884177e+00],
                      [-1.90983006e-01,  5.87785252e-01],
                      [-1.30901699e+00,  9.51056516e-01],
                      [-1.00000000e+00,  1.11022302e-16]]]
        ## ------
        print("--:: TestPenrose::test_k_max_1 starting.")

        k_range = 1
        cells = dg.dualgrid_method(PENROSE_BASIS, k_range)
        verts = np.array([ cell.verts for cell in cells ])

        testutil.save_test_figure("TestPenrose_test_k_max_1.pdf", cells)
        np.testing.assert_allclose(verts, KGO_VERTS, err_msg="Vertices not equal for Penrose Kmax=1 case: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))
        print("--:: TestPenrose::test_k_max_1 finished.")

    def test_k_max_2(self):
        print("--:: TestPenrose::test_k_max_2 starting.")

        k_range = 2
        cells = dg.dualgrid_method(PENROSE_BASIS, k_range)
        verts = np.array([ cell.verts for cell in cells ])
        testutil.save_test_figure("TestPenrose_test_k_max_2.pdf", cells)

        testutil.save_verts("penrose_k2.out.npy", verts)
        KGO_VERTS = testutil.load_verts("penrose_k2.ref.npy")
        np.testing.assert_allclose(verts, KGO_VERTS, err_msg="Vertices not equal for Penrose Kmax=2 case: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))
        print("--:: TestPenrose::test_k_max_2 finished.")

if __name__ == '__main__':
    unittest.main()


