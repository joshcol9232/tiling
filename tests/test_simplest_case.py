import unittest

import dualgrid as dg
import numpy as np

import utils_for_tests as testutil

# Square basis - simplest case
BASIS_VECS = np.array([[0.0, 1.0], [1.0, 0.0]])
BASIS_OFFSETS = np.array([0.0, 0.0])
BASIS = dg.Basis(BASIS_VECS, BASIS_OFFSETS)

class TestSimplestCase(unittest.TestCase):
    def test_square_k1(self):
        KGO_VERTS = np.array([[0., 0.],   # Single square
                              [1., 0.],
                              [0., 1.],
                              [1., 1.]])
        # ---
        print("--:: TestSimplestCase::test_square_k1 starting.")

        cells = dg.dualgrid_method(BASIS, 1, single_threaded=True)  # Should be a single square
        verts = np.array([ cell.verts for cell in cells ])
        print(verts)

        testutil.save_test_figure("TestSimplestCase_test_square_k1.pdf", cells)
        np.testing.assert_allclose(verts[0], KGO_VERTS, err_msg="Vertices not equal for square Kmax=1 case: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))

        print("--:: TestSimplestCase::test_square_k1 finished.")

    def test_square_k2(self):
        print("--:: TestSimplestCase::test_square_k2 starting.")

        cells = dg.dualgrid_method(BASIS, 2, single_threaded=True)
        verts = np.array([ cell.verts for cell in cells ])
        testutil.save_test_figure("TestSimplestCase_test_square_k2.pdf", cells)

        testutil.save_verts("square_k2.out.npy", verts)
        KGO_VERTS = testutil.load_verts("square_k2.ref.npy")
        np.testing.assert_allclose(verts, KGO_VERTS, err_msg="Vertices not equal for square Kmax=2 case: \n%s\n----------------------------\n%s" % (verts, KGO_VERTS))

        print("--:: TestSimplestCase::test_square_k2 finished.")



if __name__ == '__main__':
    unittest.main()




