import matplotlib.pyplot as plt
import numpy as np


def normalization(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))


def lls(x, y):
    A = np.transpose(np.vstack((ages, np.ones(len(ages)))))
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)),
                  np.transpose(A)), charges)

    plt.plot(ages, np.matmul(A, X), 'g-', label='OLS')


def cov(x, y):
    x_mean, y_mean = x.mean(), y.mean()
    return ((np.sum((x - x_mean) * (y - y_mean))) / (len(x) - 1))


def cov_mat(x):
    return np.array([[cov(x[0], x[0]), cov(x[0], x[1])],
                     [cov(x[1], x[0]), cov(x[1], x[1])]])


def plot_ev(x_mean, y_mean, cov_mat):
    w, v = np.linalg.eig(cov_mat)
    print(v)
    ev1 = v[:, 0]
    ev2 = v[:, 1]
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

        plt.figure()
        plt.plot(ages, charges, 'ro')
        plt.xlabel('Ages')
        plt.ylabel('Charges')

        cm = cov_mat(
            np.vstack((normalization(ages), normalization(charges))))
        plot_ev(np.mean(ages), np.mean(charges), cm)
        lls(ages, charges)
        plt.show()
