import matplotlib.pyplot as plt
import numpy as np


def lls(x, y):
    A = np.transpose(np.vstack((ages, np.ones(len(ages)))))
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)),
                  np.transpose(A)), charges)

    plt.plot(ages, np.matmul(A, X), 'g-', label='OLS')


def cov(x):
    s = np.matmul(x, np.transpose(x)) / (np.shape(x)[0] * np.shape(x)[1])
    print(s)


def plot_ev(x_mean, y_mean, cov_mat):
    w, v = np.linalg.eig(cm)
    origin = [x_mean, y_mean]
    ev1 = v[:, 0]
    ev2 = v[:, 1]
    plt.quiver(origin, ev1, color=['r'], scale=w[0])
    plt.quiver(origin, ev2, color=['b'], scale=w[1])


if __name__ == '__main__':
    with open('Data/hw1_linear_regression_dataset.csv') as f:
        lines = f.readlines()

        ages = []
        charges = []
        for line in lines:
            temp = line.replace("\n", "").split(',')
            ages.append(float(temp[0]))
            charges.append(float(temp[-1]))

        plt.plot(ages, charges, 'ro')
        plt.xlabel('Ages')
        plt.ylabel('Charges')

        cm = np.cov(np.vstack((ages, charges)))
        print(cm)
        # plot_ev(np.mean(ages), np.mean(charges), cm)

        cov(np.vstack((ages, charges)))
        lls(ages, charges)
        plt.show()
