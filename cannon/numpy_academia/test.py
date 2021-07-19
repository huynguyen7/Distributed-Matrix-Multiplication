#!/Users/huynguyen/miniforge3/envs/math/bin/python3


from cannon import cannon_matrix_mult
import numpy as np
import unittest


class MatMulTest(unittest.TestCase):
    def test_1(self):
        mat1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        mat2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
        truth = mat1@mat2
        rs = cannon_matrix_mult(mat1, mat2)
        np.testing.assert_array_equal(truth, rs)

    def test_2(self):
        mat1 = np.ones(shape=(2,2))
        mat2 = np.ones(shape=(2,2))
        truth = mat1@mat2
        rs = cannon_matrix_mult(mat1, mat2)
        np.testing.assert_array_equal(truth, rs)

    def test_3(self):
        mat1 = np.ones(shape=(1,1))
        mat2 = np.ones(shape=(1,1))
        truth = mat1@mat2
        rs = cannon_matrix_mult(mat1, mat2)
        np.testing.assert_array_equal(truth, rs)

    def test_4(self):
        mat1 = np.ones(shape=(16,16))
        mat2 = np.ones(shape=(16,16))
        truth = mat1@mat2
        rs = cannon_matrix_mult(mat1, mat2)
        np.testing.assert_array_equal(truth, rs)

    def test_5(self):
        mat1 = np.arange(0,25).reshape(5,-1)
        mat2 = np.arange(0,25).reshape(5,-1)
        truth = mat1@mat2
        rs = cannon_matrix_mult(mat1, mat2)
        np.testing.assert_array_equal(truth, rs)

    def test_6(self):
        mat1 = np.zeros(shape=(1,1))
        mat2 = np.zeros(shape=(1,1))
        truth = mat1@mat2
        rs = cannon_matrix_mult(mat1, mat2)
        np.testing.assert_array_equal(truth, rs)


if __name__ == "__main__":
    unittest.main()
