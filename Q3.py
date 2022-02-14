import matplotlib.pyplot as plt
import numpy as np
import random


def normalization(x):
    return ((x - np.min(x)) / (np.max(x) - np.min(x)))


def lls(x, y):
    A = np.transpose(np.vstack((ages, np.ones(len(ages)))))
    X = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(A), A)),
                  np.transpose(A)), charges)

    plt.plot(ages, np.matmul(A, X), 'g-', label='OLS')


def tls(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    w = np.sum((y - y_mean)**2) - np.sum((x - x_mean)**2)
    r = 2 * (np.sum((x - x_mean).dot(y - y_mean)))

    b = (w + (np.sqrt((w**2) + (r**2)))) / r
    a = y_mean - b * x_mean
    plt.plot(x, x*b+a, 'b-', label='TLS')


def cov(x, y):
    x_mean, y_mean = x.mean(), y.mean()
    return ((np.sum((x - x_mean) * (y - y_mean))) / (len(x) - 1))


def cov_mat(x):
    return np.array([[cov(x[0], x[0]), cov(x[0], x[1])],
                     [cov(x[1], x[0]), cov(x[1], x[1])]])


def plot_ev(x_mean, y_mean, cov_mat):
    w, v = np.linalg.eig(cov_mat)
    ev1 = v[:, 0]
    ev2 = v[:, 1]
    print(v)
    plt.quiver(x_mean, y_mean, ev1[0], ev1[1],
               color=['r'], scale=1/np.sqrt(w[0]))
    plt.quiver(x_mean, y_mean, ev2[0], ev2[1],
               color=['b'], scale=1/np.sqrt(w[1]))


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

    x1 = []
    y1 = []
    for i in best_in:
        x1.append(i[0])
        y1.append(i[1])
    plt.plot(x1, y1, 'bo')
    plt.plot(best_line[0], best_line[1], 'y-', lw=4, label='RANSAC')


if __name__ == '__main__':
    with open('Data/hw1_linear_regression_dataset.csv') as f:
        lines = f.readlines()

        ages = []
        charges = []
        for line in lines:
            temp = line.replace("\n", "").split(',')
            ages.append(float(temp[0]))
            charges.append(float(temp[-1]))

        ages = np.array(normalization(ages))
        charges = np.array(normalization(charges))
        plt.figure()
        plt.plot(ages, charges, 'ro')
        plt.xlabel('Ages')
        plt.ylabel('Charges')

        cm = cov_mat(np.vstack((ages, charges)))
        plot_ev(np.mean(ages), np.mean(charges), cm)
        lls(ages, charges)
        tls(ages, charges)
        ransac(ages, charges)
        plt.legend()
        plt.show()
