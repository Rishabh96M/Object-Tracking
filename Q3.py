import matplotlib.pyplot as plt
import numpy as np
import estimations


def plot_ev(x_mean, y_mean, cov_mat):
    w, v = np.linalg.eig(cov_mat)
    ev1 = v[:, 0]
    ev2 = v[:, 1]
    print("\nEigen Values of Covariance Matrix:")
    print(w)
    print("\nEigen Vectors of Covariance Matrix:")
    print(ev1)
    print(ev2)
    plt.quiver(x_mean, y_mean, ev1[0], ev1[1],
               color=['r'], scale=1/np.sqrt(w[0]))
    plt.quiver(x_mean, y_mean, ev2[0], ev2[1],
               color=['b'], scale=1/np.sqrt(w[1]))


if __name__ == '__main__':
    with open('Data/hw1_linear_regression_dataset.csv') as f:
        lines = f.readlines()

        ages = []
        charges = []
        for line in lines:
            temp = line.replace("\n", "").split(',')
            ages.append(float(temp[0]))
            charges.append(float(temp[-1]))

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

    t = 0.08
    x_line, y_line, x_in, y_in = estimations.ransac(x_norm, y_norm, t)
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
    plt.plot(x_in, y_in, 'bo', label='inlires')
    plt.plot(x_line, y_line, 'y-', lw=4, label='RANSAC')

    plt.legend()
    plt.show()
