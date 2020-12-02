"""
Project: PolySolvy Version X
MATH 351 - Numerical Analysis
Under the instruction of Professor Eric Merchant.

Team: Nathan, Jaeger, Ronnie

File Description:

vmond.py programs the routines for solving for bivariate polynomial
coefficients using a Vandermonde array. A Gaussian elimination procedure
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


def Gauss(A: np.array, b: np.array) -> np.array:
    """
    Gaussian elimination procedure (aka Row Reduction) for
    solving linear systems of the form Ax = b.

    Scaled Partial Pivoting (SPP) is used in order to avoid
    numerical errors associated with the Naive Algorithm
    (read C&K sections 7.1-7.2 for more details).

    Assumes array A is square (N x N) and
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

    Basic Case: Let's get a little more adventurous.

    >> A = np.array([[3, -13, 9, 3], [-6, 4, 1, -18], [6, -2, 2, 4], [12, -8, 6, 10]], dtype=np.double)
    >> b = np.array([-19, -34, 16, 26], dtype=np.double)
    >> Gauss(A, b)
    array([ 3.,  1., -2.,  1.])

    """
    N = A.shape[0]

    # Build coefficient vector
    x = np.array([0 for i in range(N)], dtype=np.double)

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
    # coefficients for row i in array A.
    Scale = [0 for i in range(N)]

    # - - - - - - - - - - - - - - - - - - - - - - - #
    #   PHASE 1 - Row Reduction w/ Scaled Pivots    #
    # - - - - - - - - - - - - - - - - - - - - - - - #

    # init step - compute values of scale factors for each row
    for i in range(N):
        SMAX = 0
        for j in range(N):
            SMAX = max(SMAX, abs(A[i, j]))
        Scale[i] = SMAX

    # Row Reduction - go through rows, prioritize which one
    # to use as a "pivot", and eliminate entries in corresponding
    # column in all other rows.
    for k in range(N - 1):
        RMAX = 0
        for i in range(k, N):
            l_i = Index[i]
            s_i = Scale[l_i]
            r = abs(A[l_i, k] / s_i)

            # If scaled ratio is higher than previous encounter,
            # then this row should take higher priority.
            if r > RMAX:
                RMAX = r
                j = i

        # Swap terms to reorganize which rows take higher priority.
        SWP = Index[j]
        Index[j] = Index[k]
        Index[k] = SWP

        l_k = Index[k]
        # Eliminate entries in all other rows != row k.
        for i in range(k + 1, N):
            l_i = Index[i]

            XMULT = A[l_i, k] / A[l_k, k]
            A[l_i, k] = XMULT

            for j in range(k + 1, N):
                A[l_i, j] = A[l_i, j] - XMULT * A[l_k, j]

    # - - - - - - - - - - - - - - - - #
    #   PHASE 2 - Back Substitution   #
    # - - - - - - - - - - - - - - - - #

    # Using the current values of Index and the stored coefficients in A,
    # we will now alter the values of solution vector b
    # to match the manipulations we have already done
    # to the coefficient array A.
    for k in range(N - 1):
        l_k = Index[k]
        for i in range(k + 1, N):
            l_i = Index[i]
            b[l_i] = b[l_i] - A[l_i, k] * b[l_k]

    l_n = Index[N - 1]
    x[N - 1] = b[l_n] / A[l_n, N - 1]

    # We are now well equipped to solve for the values
    # of our x-vector. This is the final step of the algorithm.
    for i in range(N - 2, -1, -1):
        l_i = Index[i]
        SUM = b[l_i]
        for j in range(i + 1, N):
            SUM = SUM - A[l_i, j] * x[j]

        x[i] = SUM / A[l_i, i]

    return x


def Vandermonde(X: np.array, Y: np.array, Z: np.array) -> np.array:
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

    >>> X = np.array([1, 3, 5, 4], dtype=np.double)
    >>> Y = np.array([2, 5, 4, 1], dtype=np.double)
    >>> Z = np.array([4.9, 2.6, 3.7, 7.8], dtype=np.double)
    >>> Vandermonde(X, Y, Z)
    array([0.31111, -0.47778, 1.48889, 1.07778])

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
    VanderMat = [ [] for i in range(N) ]

    # TODO

    for i in range(N):
        x = X[i]
        y = Y[i]
        for j in range(N):
            VanderMat[i, j] = pow(x, i) * pow(y, j)

    # Extract coefficients from constructed VanderMat.
    # In a sense, we are solving the linear system VanderMat * C = Z.
    C = Gauss(VanderMat, Z)

    return C


if __name__ == '__main__':
    print(doctest.testmod())
