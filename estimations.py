# estimations.py>
#
# Estimation Functions
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Library of linear algebra estimation functions used for line and
#              curve fitting.


import numpy as np
import random


def nim_max_normalisation(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))


def ols(x, y, n):
    A = np.ones(len(x))
    for i in range(1, n+1):
        A = np.vstack((np.power(x, i), A))
    A = A.T

    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
    return np.matmul(A, X)


def tls(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    w = np.sum((y - y_mean)**2) - np.sum((x - x_mean)**2)
    r = 2 * (np.sum((x - x_mean).dot(y - y_mean)))

    b = (w + (np.sqrt((w**2) + (r**2)))) / r
    a = y_mean - b * x_mean
    return x, x*b + a


def cov(x, y):
    x_mean, y_mean = x.mean(), y.mean()
    return ((np.sum((x - x_mean) * (y - y_mean))) / (len(x) - 1))


def cov_mat(x):
    return np.array([[cov(x[0], x[0]), cov(x[0], x[1])],
                     [cov(x[1], x[0]), cov(x[1], x[1])]])


def dist(pt, a, b):
    return abs(((b[1]-a[1])*pt[0])-((b[0]-a[0])*pt[1])+(b[0]*a[1])
               - ([b[1]*a[0]]))/(np.sqrt((b[1]-a[1])**2+(b[0]-a[0])**2))


def ransac(x, y):
    n = 1000
    t = 0.08
    sample = 0
    best_line = []
    best_in = []

    while (n > sample):
        rand = random.sample(range(len(x)), 2)
        p1 = (x[rand[0]], y[rand[0]])
        p2 = (x[rand[1]], y[rand[1]])
        inliers = []

        for i in range(len(x)):
            if dist([x[i], y[i]], p1, p2) <= t:
                inliers.append([x[i], y[i]])

        if len(best_in) < len(inliers):
            best_in = inliers
            best_line = [(p1[0], p2[0]), (p1[1], p2[1])]

        e = 1-(len(inliers)/len(x))
        n = (np.log(e))/(np.log(1-(1-e)**2))
        sample += 1

    x_in = []
    y_in = []
    for i in best_in:
        x_in.append(i[0])
        y_in.append(i[1])
    return (best_line[0], best_line[1], x_in, y_in)


def svd_decomposition(x):
    evals, U = np.linalg.eig(np.matmul(x, np.transpose(x)))
    evals2, V = np.linalg.eig(np.matmul(np.transpose(x), x))

    idx = evals.argsort()[::-1]
    evals = evals[idx]
    U = U[:, idx]
    idx = evals2.argsort()[::-1]
    evals2 = evals2[idx]
    V = V[:, idx]

    sig = np.sqrt(evals)
    S = np.zeros(np.shape(x))
    for i in range(len(sig)):
        S[i][i] = sig[i]
        U[:, i] = np.matmul(x, V[:, i]) / sig[i]

    return U, S, np.transpose(V)
