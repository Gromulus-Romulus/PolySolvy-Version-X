"""
Project: PolySolvy Version X
MATH 351 - Numerical Analysis
Under the instruction of Professor Eric Merchant.

Team: Nathan, Jaeger, Ronnie

File Description:

vmond.py programs the routines for solving for bivariate polynomial
coefficients using a Vandermonde matrix. A Gaussian elimination procedure
for solving linear systems (aka Matrix Row Reduction) is also implemented
as an auxiliary function.
"""

import doctest
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#  TODO List:
#
#  + Rigorously test these functions against ALL possible edge cases.
#  + Connect this code to the polynomial interpolation file (yet to be written).
#  + Gauss needs guardian code to stop it from solving inconsistent systems (1 = 0)
#  + Vandermonde and Gauss both need guardian code to ensure the arrays passed are of
#    type np.double exclusively.
#
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class MatrixSolveError(Exception):
    """ Any error intentionally raised by Gauss or Vandermonde. """
    pass

def Gauss(A: np.matrix, b: np.array) -> np.array:
    """
    Gaussian elimination procedure (aka Row Reduction) for
    solving linear systems of the form Ax = b.

    Scaled Partial Pivoting (SPP) is used in order to avoid
    numerical errors associated with the Naive Algorithm
    (read C&K sections 7.1-7.2 for more details).

    Assumes matrix A is square (N x N) and
    solution vector b is N-dimensional.

    Returns values of vector x.

    WARNING: To avoid numerical divide by zero / overflow errors,
    make sure you specify the datatype (dtype) to be np.double!

    References:
        + Phase 1 Row Reduction Alg: C&K 6th ed. p. 267
        + Phase 2 Back Substitution Alg: C&K 6th ed. p. 269

    # - - - - - #
    #  TESTING  #
    # - - - - - #

    Basic Case: The following system can be solved by the naive algorithm (no SPP).

         x1 + 2x2 = 1
        3x1 + 4x2 = 1

         x1 + 2x2 = 1
             -2x2 = -2

        Therefore, x2 = 1, which means x1 = -1

    >>> A = np.matrix([[1, 2], [3, 4]], dtype=np.double)
    >>> b = np.array([1, 1], dtype=np.double)
    >>> Gauss(A, b)
    array([-1,  1])

    Basic Case: Let's get a little more adventurous.

        5x1 + 4x2 -  x3 = 0
             10x2 - 3x3 = 11
                     x3 = 3

        The expected solution: x1 = -1, x2 = 2, x3 = 3.

    >>> A = np.matrix([[5, 4, -1], [0, 10, -3], [0, 0, 1]], dtype=np.double)
    >>> b = np.array([0, 11, 3], dtype=np.double)
    >>> Gauss(A, b)
    array([-1,  2,  3])

    Edge Case: This system cannot be solved by the naive algorithm (needs SSP).

        0x1 + x2 = 1
         x1 + x2 = 2

    >>> A = np.matrix([[0, 1], [1, 1]], dtype=np.double)
    >>> b = np.array([1, 2], dtype=np.double)
    >>> Gauss(A, b)
    array([1, 1])

    """
    N = A.shape[0]

    # Build coefficient vector
    x = np.array([0 for i in range(N)])

    # Index list prioritizes which row
    # should be scaled first.
    #
    # This is necessary because Naive
    # Gaussian elimination is prone to
    # yield errors if the natural order
    # is followed (C&K 6th ed., p. 261-262)
    Index = [i for i in range(N)]

    # Scale list keeps track of what scale
    # factors are needed to normalize the
    # coefficients for row i in matrix A.
    Scale = [0 for i in range(N)]

    # - - - - - - - - - - - - - - - - - - - - - - - #
    #   PHASE 1 - Row Reduction w/ Scaled Pivots    #
    # - - - - - - - - - - - - - - - - - - - - - - - #

    # init step - compute values of scale factors for each row
    for i in range(N):
        SMAX = 0
        for j in range(N):
            SMAX = max(SMAX, A[i, j])
        Scale[i] = SMAX

    # Row Reduction - go through rows, prioritize which one
    # to use as a "pivot", and eliminate entries in corresponding
    # column in all other rows.
    for k in range(N - 1):
        RMAX = 0
        for i in range(k, N):
            l_i = Index[i]
            s_i = Scale[i]
            r = abs(A[l_i, k]/s_i)

            # If scaled ratio is higher than previous encounter,
            # then this row should take higher priority.
            if r > RMAX:
                RMAX = r
                j = i

        # Swap terms to reorganize which rows take higher priority.
        SWP      = Index[j]
        Index[j] = Index[k]
        Index[k] = SWP

        # Eliminate entries in all other rows beneath row k.
        for i in range(k + 1, N):
            l_i = Index[i]
            l_k = Index[k]

            XMULT = A[l_i, k] / A[l_k, k]
            A[l_i, k] = XMULT

            for j in range(k + 1, N):
                A[l_i, j] = A[l_i, j] - XMULT * A[l_k, j]

    # - - - - - - - - - - - - - - - - #
    #   PHASE 2 - Back Substitution   #
    # - - - - - - - - - - - - - - - - #

    # Using the current values of Index and Scale,
    # we will now alter the values of solution vector b
    # to match the manipulations we have already done
    # to the coefficient matrix A.
    for k in range(N - 1):
        l_k = Index[k]
        for i in range(k + 1, N):
            l_i = Index[i]
            b[l_i] = b[l_i] - A[l_i, k] * b[l_k]

    l_n = Index[N - 1]
    x[N - 1] = b[l_n] / A[l_n, N - 1]

    # We are now well equipped to solve for the values
    # of our x-vector. This is the final step of the algorithm.
    for i in range(N-1, -1, -1):
        l_i = Index[i]
        SUM = b[l_i]
        for j in range(i + 1, N):
            SUM = SUM - A[l_i, j] * x[j]

        x[i] = SUM / A[l_i, i]

    return x

def Vandermonde(X: np.array, Y: np.array, Z: np.array) -> np.matrix:
    """
    Given a series of X, Y, and corresponding Z values,
    Vandermonde will return the coefficients associated
    with the interpolating polynomial P(x, y) -> z.

    If X, Y, and Z are unbalanced (not of the same size),
    a RuntimeError will be thrown to the interpreter.

    Calls auxiliary function Gauss to solve for C.

    # - - - - - #
    #  TESTING  #
    # - - - - - #

    Reference page for test cases:
    https://ece.uwaterloo.ca/~dwharder/NumericalAnalysis/05Interpolation/multi/

    Base case 1:

    >>> X = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.double)
    >>> Y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.double)
    >>> Z = np.array([3.2, 4.4, 6.5, 2.5, 4.7, 5.8, 5.1, 3.6, 2.9], dtype=np.double)
    >>> Vandermonde(X, Y, Z)
    array([0.975, -5.275, 5.95, -3.925, 19.825, -21.55, 3.4, -14.7, 18.5])

    """

    # Make sure data value arrays are balanced.
    try:
        assert X.size == Y.size == Z.size
    except AssertionError:
        raise MatrixSolveError("Tried to call Vandermonde with unbalanced X, Y, and Z arrays!")

    # Check for vacuous condition.
    N = X.size
    C = np.array([])
    if N == 0:
        return C

    # NumPy matrices must have memory allocated prior to any computations.
    # Vandermonde Matrix encodes linear system f[X, Y] = Z.
    VanderMat = np.matrix([[0 for i in range(N)] for j in range(N)], dtype=np.double)

    for i in range(N):
        x = X[i]
        for j in range(N):
            y = Y[j]
            VanderMat[i, j] = pow(x, i) * pow(y, j)

    # Extract coefficients from constructed VanderMat.
    # In a sense, we are solving the linear system VanderMat * C = Z.
    C = Gauss(VanderMat, Z)

    return C

if __name__ == '__main__':
    print(doctest.testmod())