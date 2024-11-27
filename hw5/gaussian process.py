from scipy.optimize import minimize
import matplotlib.pyplot as plt
from utils_gp import *


def main():
    data = load_data(filepath)
    col = data[:, 0].reshape(-1, 1)
    cov = rational_quadratic_kernel(col, col) + 1 / beta * np.identity(col.shape[0])
    x_values = np.linspace(-60, 60, num=1000).reshape(-1, 1)
    means, variances = prediction_distribution(x_values, data, cov, beta=beta)
    std = np.sqrt(variances)

    x_values = x_values.reshape(-1)
    means = means.reshape(-1)
    plt.plot(data[:, 0], data[:, 1], 'bo')
    plt.plot(x_values, means, 'k-')
    plt.fill_between(x_values, means + 2 * std, means - 2 * std, facecolor='r', edgecolor='r')
    plt.xlim(-60, 60)
    plt.show()


def negative_marginal_log_likelihood(params, x, Y, beta):
    sigma, length_scale, alpha = params
    cov = rational_quadratic_kernel(x, x, sigma, length_scale, alpha) + 1 / beta * np.identity(x.shape[0])
    invC = np.linalg.inv(cov)
    n = Y.shape[0]
    quadratic_term = float(1 / 2 * (Y.T @ invC @ Y))
    log_det_term = float(1 / 2 * (np.linalg.slogdet(cov)[1]))
    normalize_term = float(n / 2 * np.log(2 * np.pi))

    return quadratic_term + log_det_term + normalize_term


def main2():
    data = load_data(filepath)
    initial = np.array([1.0, 1.0, 1.0])
    bounds = [(1e-5, None), (1e-5, None), (1e-5, None)]
    x = data[:, 0].reshape(-1, 1)
    Y = data[:, 1].reshape(-1, 1)

    update_parameter = minimize(
        fun=negative_marginal_log_likelihood,
        x0=initial,
        args=(x, Y, beta),
        bounds=bounds
    )
    new_sigma, new_length_scale, new_alpha = update_parameter.x
    print(f'newsigma={new_sigma}')
    print(f'new_length_scale={new_length_scale}')
    print(f'new_alpha={new_alpha}')
    new_cov = rational_quadratic_kernel(x, x, new_sigma, new_length_scale, new_alpha) + 1 / beta * np.identity(
        x.shape[0])
    x_values = np.linspace(-60, 60, num=1000).reshape(-1, 1)
    means, variances = prediction_distribution(x_values, data, new_cov, new_sigma, new_length_scale, new_alpha, beta)
    std = np.sqrt(variances)

    x_values = x_values.reshape(-1)
    means = means.reshape(-1)
    plt.plot(data[:, 0], data[:, 1], 'bo')
    plt.plot(x_values, means, 'k-')
    plt.fill_between(x_values, means + 2 * std, means - 2 * std, facecolor='r', edgecolor='r')
    plt.xlim(-60, 60)
    plt.show()


if __name__ == '__main__':
    filepath = 'input.data'
    beta = 5
    main()
    main2()
