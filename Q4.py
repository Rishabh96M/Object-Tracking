# Q4.py
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Answers for Q4 of first assignment

import numpy as np
import estimations


if __name__ == '__main__':
    x = [5, 150, 150, 5]
    y = [5, 5, 150, 150]
    xp = [100, 200, 220, 100]
    yp = [100, 80, 80, 200]

    A = np.array([[-x[0], -y[0], -1, 0, 0, 0, x[0]*xp[0], y[0]*xp[0], xp[0]],
                  [0, 0, 0, -x[0], -y[0], -1, x[0]*yp[0], y[0]*yp[0], yp[0]],
                  [-x[1], -y[1], -1, 0, 0, 0, x[1]*xp[1], y[1]*xp[1], xp[1]],
                  [0, 0, 0, -x[1], -y[1], -1, x[1]*yp[1], y[1]*yp[1], yp[1]],
                  [-x[2], -y[2], -1, 0, 0, 0, x[2]*xp[2], y[2]*xp[2], xp[2]],
                  [0, 0, 0, -x[2], -y[2], -1, x[2]*yp[2], y[2]*yp[2], yp[2]],
                  [-x[3], -y[3], -1, 0, 0, 0, x[3]*xp[3], y[3]*xp[3], xp[3]],
                  [0, 0, 0, -x[3], -y[3], -1, x[3]*yp[3], y[3]*yp[3], yp[3]]])

    U, S, VT = estimations.svd_decomposition(A)
    print('U Matrix is:')
    print(U)
    print('\nS Matrix is:')
    print(S)
    print('\nVT Matrix is:')
    print(VT)
    print('\nComputed A matrix is:')
    print(np.round(np.matmul(np.matmul(U, S), VT), 4))
    print('\nOriginal matrix:')
    print(A)
    print('\nSolution for Ax=0 is, x:')
    print(VT[-1])
    print('\nHomography matrix H is :')
    print(VT[-1].reshape((3, 3)))
