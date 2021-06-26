#!/Users/huynguyen/miniforge3/envs/math/bin/python3


import numpy as np


"""

    @author: Huy Nguyen
    - Cannon's Algorithm for Matrix Multiplication (Non-parallel, non-distributed version).
    - This implementation does not include any part of multiprocessing (just for math intuition only).
    - The main advantage of the algorithm is that its storage requirements remain constant and are independent of the number of processes.
    - ONLY ACCEPT SQUARED MATRICES.
    - Source:
        +https://www3.nd.edu/~zxu2/acms60212-40212/Lec-07-3.pdf
        +https://www.cs.utah.edu/~hari/teaching/paralg/tutorial/05_Cannons.html
        +https://en.wikipedia.org/wiki/Cannon%27s_algorithm
        +https://people.eecs.berkeley.edu/~demmel/cs267/lecture11/lecture11.html#link_5
        +https://iq.opengenus.org/cannon-algorithm-distributed-matrix-multiplication/
        +http://cseweb.ucsd.edu/classes/fa12/cse260-b/Lectures/Lec13.pdf

"""


# Time: O(m*n*k), space: O(m*n)
def naive_matrix_mult(mat1, mat2):
    assert mat1 is not None and mat2 is not None \
        and isinstance(mat1, np.ndarray) and isinstance(mat2, np.ndarray)

    mat3 = np.zeros(shape=(mat1.shape[0],mat2.shape[1]), dtype=np.float64)
    for i in range(mat3.shape[0]):
        for j in range(mat3.shape[1]):
            mat3[i,j] = np.sum(mat1[i,:]*mat2[:,j])
    return mat3


def shift_left(mat, i, amount):  # Shift left `i` row `amount` step.
    tmp = np.zeros(p_sqrt)
    for j in range(p_sqrt):
        tmp[j] = mat[i,(j+amount)%p_sqrt]
    mat[i,:] = tmp


def shift_up(mat, j, amount):  # Shift up `j` row `amount` step.
    tmp = np.zeros(p_sqrt)
    for i in range(p_sqrt):
        tmp[i] = mat[(i+amount)%p_sqrt,j]
    mat[:,j] = tmp


def cannon_matrix_mult(mat1, mat2):
    assert mat1 is not None and mat2 is not None \
            and isinstance(mat1, np.ndarray) and isinstance(mat2, np.ndarray)
    assert mat1.shape[0] == mat1.shape[1] and mat2.shape[0] == mat2.shape[1] \
            and mat1.shape[0] == mat2.shape[0], "NON-SQUARED MATRICES ARE NOT ACCEPTED FOR CANNON'S ALGORITHM"

    mat3 = np.zeros(shape=(mat1.shape[0],mat2.shape[1]), dtype=np.float64)

    global p_sqrt
    p_sqrt = mat1.shape[0]

    for i in range(p_sqrt):
        shift_left(mat1,i,i)

    for j in range(p_sqrt):
        shift_up(mat2,j,j)

    for k in range(p_sqrt):
        for i in range(p_sqrt):
            for j in range(p_sqrt):
                m = (i+j+k) % p_sqrt
                mat3[i,j] += mat1[i,m]*mat2[m,j]

                # Shift left 1 step for mat1.
                shift_left(mat1,i,1)

                # Shift up 1 step for mat2.
                shift_up(mat2,j,1)

    return mat3


if __name__ == "__main__":
    A = np.arange(0,9).reshape(3,-1)
    B = np.arange(0,9).reshape(3,-1)
    C = A@B
    CALC = cannon_matrix_mult(A,B)
    print('TRUTH:\n', C)
    print('CALC:\n', CALC)
