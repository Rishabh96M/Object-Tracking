# Object-Tracking
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Finding the top and bottom of the red ball in each frame and
#              performing least squares for estimating the path of the ball

import numpy as np


def svd_decomposition(X):
    evals, U = np.linalg.eig(np.matmul(X, np.transpose(X)))
    evals2, V = np.linalg.eig(np.matmul(np.transpose(X), X))

    idx = evals.argsort()[::-1]
    evals = evals[idx]
    U = U[:, idx]
    idx = evals2.argsort()[::-1]
    V = V[:, idx]

    S = np.matmul(np.matmul(U, X), np.transpose(V))

    return np.transpose(U), S, V


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

    U, S, VT = svd_decomposition(A)
    u, s, v = np.linalg.svd(A)
    # print(s)
    # s = np.transpose(np.vstack(((np.transpose(np.diag(s))), np.zeros(8))))
    # print(u)
    # print('\n')
    # print(U)
    # print('\n')
    # print(v)
    # print('\n')
    # print(VT)
    # print('\n')
    # print(s)
    # print('\n')
    # print(S)
    # print('\n')
    print(np.matmul(np.matmul(U, S), VT))
    print('\n')
    print(A)
