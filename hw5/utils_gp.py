import numpy as np


def load_data(filepath):
    data = np.zeros((34, 2))
    with open(filepath, 'r') as file:
        lines = file.readlines()
        i = 0
        for line in lines:
            x, y = line.split()
            data[i, :] = [float(x), float(y)]
            i += 1
    return data


def rational_quadratic_kernel(x1: np.ndarray, x2: np.ndarray, sigma=1.0, length_scale=1.0, alpha=1.0):
    k = np.zeros((x1.shape[0], x2.shape[0]))
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            square_diff = (x1[i, 0] - x2[j, 0]) ** 2
            k[i, j] = sigma ** 2 * (1 + square_diff / (2 * alpha * length_scale ** 2)) ** (-alpha)
    return k


def prediction_distribution(x_values, data, cov, sigma=1.0, length_scale=1.0, alpha=1.0, beta=5):
    K = rational_quadratic_kernel
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    K_xstar_x = K(x_values, x, sigma, length_scale, alpha)
    invC = np.linalg.inv(cov)
    means = K_xstar_x @ invC @ y
    variances = np.diag(K(x_values, x_values, sigma, length_scale, alpha) + (1/beta) - K_xstar_x @ invC @ K_xstar_x.T)
    return means, variances
