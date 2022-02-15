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
    """
    Definition
    ---
    Method to perform min max normalisation on data

    Parameters
    ---
    x : 1Darray of data

    Returns
    ---
    x_norm : 1Darray of normalised data
    """
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))


def ols(x, y, n):
    """
    Definition
    ---
    Method generate and find original least squares solution for data

    Parameters
    ---
    x, y : Data on which ols is to be performed
    n : Max Degree of solution (1 -> Line, 2 -> 2nd order curve, etc.)

    Returns
    ---
    y_ols : Estimated y points after OLS
    """
    # Generating A matrix based on the maximum degree and x
    A = np.ones(len(x))
    for i in range(1, n+1):
        A = np.vstack((np.power(x, i), A))
    A = A.T

    # dot product of pseudo-inverse of A and y
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
    return np.matmul(A, X)


def tls(x, y):
    """
    Definition
    ---
    Method generate and find total least squares solution for data

    Parameters
    ---
    x, y : Data on which ols is to be performed

    Returns
    ---
    x_tls, y_tls : Estimated x and y points after TLS
    """
    # Mean of data
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    w = np.sum((y - y_mean)**2) - np.sum((x - x_mean)**2)
    r = 2 * (np.sum((x - x_mean).dot(y - y_mean)))

    b = (w + (np.sqrt((w**2) + (r**2)))) / r
    a = y_mean - b * x_mean
    return x, x*b + a


def cov(x, y):
    """
    Definition
    ---
    Method to calculate the covariance between two sets of data

    Parameters
    ---
    x, y : Data

    Returns
    ---
    cov_xy : Covariance between x and y
    """
    x_mean, y_mean = x.mean(), y.mean()
    return ((np.sum((x - x_mean) * (y - y_mean))) / (len(x) - 1))


def cov_mat(x):
    """
    Definition
    ---
    Method to generate covariance matrix

    Parameters
    ---
    x : matrix

    Returns
    ---
    cov_mat : covariance matrix of x
    """
    return np.array([[cov(x[0], x[0]), cov(x[0], x[1])],
                     [cov(x[1], x[0]), cov(x[1], x[1])]])


def dist(pt, a, b):
    """
    Definition
    ---
    Method to calculate distance between a point and line (given by 2 points)

    Parameters
    ---
    pt : point of intererst
    a, b : points on the line

    Returns
    ---
    dist : Distance between point and line
    """
    return abs(((b[1]-a[1])*pt[0])-((b[0]-a[0])*pt[1])+(b[0]*a[1])
               - ([b[1]*a[0]]))/(np.sqrt((b[1]-a[1])**2+(b[0]-a[0])**2))


def ransac(x, y, t):
    """
    Definition
    ---
    Method to generate RANSAC on data

    Parameters
    ---
    x, y : data to estimate RANSAC
    t : Threshold

    Returns
    ---
    x_lins, y_line : X and Y coordinates of points on the best line
    x_in, y_in : X and Y coordinates of all inliers
    """
    n = 1000  # Initital number of iterations
    sample = 0
    best_line = []
    best_in = []
    p = 0.95

    while (n > sample):
        # Extracting 2 random points
        rand = random.sample(range(len(x)), 2)
        p1 = (x[rand[0]], y[rand[0]])
        p2 = (x[rand[1]], y[rand[1]])
        inliers = []

        for i in range(len(x)):
            # Checking for distance from point to random line
            if dist([x[i], y[i]], p1, p2) <= t:
                inliers.append([x[i], y[i]])
        # Updating Best Model
        if len(best_in) < len(inliers):
            best_in = inliers
            best_line = [(p1[0], p2[0]), (p1[1], p2[1])]
        # Updating number of iterations
        e = 1-(len(inliers)/len(x))
        n = (np.log(1-p))/(np.log(1-(1-e)**2))
        sample += 1
    x_in = []
    y_in = []
    for i in best_in:
        x_in.append(i[0])
        y_in.append(i[1])
    return (best_line[0], best_line[1], x_in, y_in)


def svd_decomposition(x):
    """
    Definition
    ---
    Method to calculate SVD of a matrix

    Parameters
    ---
    x : 2DArray of data

    Returns
    ---
    U : Left singular Vectors
    S : Matrix with diagnoal elements as singular eigen values
    VT : Transpose of Right Singular Vectors
    """
    evals, U = np.linalg.eig(np.matmul(x, np.transpose(x)))
    evals2, V = np.linalg.eig(np.matmul(np.transpose(x), x))

    # Sorting Eigen Values and Corresponding Eigen Vectors
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
