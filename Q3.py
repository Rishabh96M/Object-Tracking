# Q3.py
# Copyright (c) 2022 Rishabh Mukund
# MIT License
#
# Description: Answers for Q3 of first assignment

import matplotlib.pyplot as plt
import numpy as np
import estimations


def plot_ev(x_mean, y_mean, mat):
    """
    Definition
    ---
    Method to plot eigen vectors and values of Matrix

    Parameters
    ---
    x_mean : Mean of the all points in X
    y_mean : Mean of the all points in Y
    mat : Matrix to plot eigen vectors and values of
    """
    # Getting Eigen values and Eigen Vectors of matrix
    w, v = np.linalg.eig(mat)

    # Sorting Eigen Values and Corresponding Eigen Vectors
    idx = np.argsort(w)[::-1]
    w = w[idx]
    v = v[:, idx]

    ev1 = v[:, 0]
    ev2 = v[:, -1]
    print("\nLargest and Smallest Eigen Values of Matrix:")
    print(w[0], w[-1])
    print("\nCorrsponding Eigen Vectors of Matrix:")
    print(ev1)
    print(ev2)
    plt.quiver(x_mean, y_mean, ev1[0], ev1[1],
               color=['r'], scale=1/np.sqrt(w[0]))
    plt.quiver(x_mean, y_mean, ev2[0], ev2[1],
               color=['b'], scale=1/np.sqrt(w[-1]))


if __name__ == '__main__':
    # Opening file to extract csv data
    with open('Data/hw1_linear_regression_dataset.csv') as f:
        # Reading data from csv file
        lines = f.readlines()

        ages = []
        charges = []
        for line in lines:
            temp = line.replace("\n", "").split(',')
            ages.append(float(temp[0]))
            charges.append(float(temp[-1]))

    # Normalising the data to convert them into similar scales
    x_norm = np.array(estimations.nim_max_normalisation(ages))
    y_norm = np.array(estimations.nim_max_normalisation(charges))
    cm = estimations.cov_mat(np.vstack((x_norm, y_norm)))
    print("Covariance Matris:")
    print(cm)

    plt.figure()
    plt.title('PCA')
    plt.plot(x_norm, y_norm, 'ro')
    plt.xlabel('ages')
    plt.ylabel('charges')
    plot_ev(np.mean(x_norm), np.mean(y_norm), cm)

    y_ols = estimations.ols(x_norm, y_norm, 1)
    plt.figure()
    plt.title('OLS')
    plt.plot(x_norm, y_norm, 'ro')
    plt.xlabel('ages')
    plt.ylabel('charges')
    plt.plot(x_norm, y_ols, 'b-')

    x_tls, y_tls = estimations.tls(x_norm, y_norm)
    plt.figure()
    plt.title('TLS')
    plt.plot(x_norm, y_norm, 'ro')
    plt.xlabel('ages')
    plt.ylabel('charges')
    plt.plot(x_tls, y_tls, 'b-')

    # threshold for RANSAC
    t = 0.08
    x_line, y_line, x_in, y_in = estimations.ransac(x_norm, y_norm, t, 0.99)
    print("\nThreshold for RANSAC is:")
    print(t)
    print("\nBest Fitting Line for RANSAC is through points:")
    print(x_line[0], y_line[0])
    print(x_line[1], y_line[1])
    plt.figure()
    plt.title('RANSAC')
    plt.plot(x_norm, y_norm, 'ro')
    plt.xlabel('ages')
    plt.ylabel('charges')
    # Plotting In-Liers
    plt.plot(x_in, y_in, 'bo', label='inlires')
    # Plotting Line
    plt.plot(x_line, y_line, 'y-', lw=4, label='RANSAC')
    plt.legend()

    plt.show()
