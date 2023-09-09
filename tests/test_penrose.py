import unittest
import utils_for_tests as testutil

import dualgrid as dg
import numpy as np

PENROSE_BASIS = dg.utils.penrose_basis(random_offsets=False)


def add_test_for_k_range(k_range, single_threaded=True):
    print("--:: TestPenrose::test_k_max_%d, single_threaded=%s starting." % (k_range, single_threaded))

    cells = dg.dualgrid_method(PENROSE_BASIS, k_range, single_threaded=single_threaded)
    verts = np.array([ cell.verts for cell in cells ])
    testutil.save_test_figure("TestPenrose_test_k_max_%d_single_thread_%s.pdf" % (k_range, single_threaded), cells)

    testutil.save_verts("penrose_k%d_single_thread_%s.out.npy" % (k_range, single_threaded), verts)

    # NOTE: Always use single-threaded result as the known good output. Should detect differences between single/multithreaded.
    KGO_VERTS = testutil.load_verts("penrose_k%d.ref.npy" % k_range)
    np.testing.assert_allclose(verts, KGO_VERTS, err_msg="Vertices not equal for Penrose Kmax=%d, singlethreaded=%s case: \n%s\n----------------------------\n%s" % (k_range, single_threaded, verts, KGO_VERTS))
    print("--:: TestPenrose::test_k_max_%d, single_threaded=%s finished." % (k_range, single_threaded))


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
        cells = dg.dualgrid_method(PENROSE_BASIS, k_range, single_threaded=True)
        verts = np.array([ cell.verts for cell in cells ])

        testutil.save_test_figure("TestPenrose_test_k_max_1.pdf", cells)
        np.testing.assert_allclose(verts, KGO_VERTS, err_msg="Vertices not equal for Penrose Kmax=1 case: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))
        print("--:: TestPenrose::test_k_max_1 finished.")

    def test_k_max_2_single_thread(self):
        add_test_for_k_range(2, single_threaded=True)

    def test_k_max_3_single_thread(self):
        add_test_for_k_range(3, single_threaded=True)

    def test_k_max_4_single_thread(self):
        add_test_for_k_range(4, single_threaded=True)

    def test_k_max_3_multithreaded(self):
        add_test_for_k_range(3, single_threaded=False)

    def test_k_max_4_multithreaded(self):
        add_test_for_k_range(4, single_threaded=False)

if __name__ == '__main__':
    unittest.main()


