import numpy as np
import random
import math
from scipy.special import erfinv
import matplotlib.pyplot as plt

np.set_printoptions(precision=5, suppress=True)


def probit(num):
    return math.sqrt(2) * erfinv(2 * num - 1)


def randomNumberGeneratorGaussian(mu, var):
    num = np.random.uniform()
    return math.sqrt(var) * probit(num) + mu


def randomNumberGeneratorBasisLinear(n, var, w_list):
    x = random.uniform(-1, 1)
    error = randomNumberGeneratorGaussian(0, var)
    coefficient = np.array(w_list)
    y = 0
    for i in range(n):
        y += coefficient[i] * pow(x, i)
    y += error
    return x, y


def designMatrix(x, n):
    A = np.zeros((1, n))
    for i in range(n):
        A[0, i] = pow(x, i)
    return A


def polynomial_mean_function(mean, x_values):
    # Calculate the polynomial mean: mean[0]*x^0 + mean[1]*x^1 + ... + mean[n]*x^n
    y_pred = np.zeros_like(x_values)
    for i, coeff in enumerate(mean):
        y_pred += coeff * (x_values ** i)
    return y_pred


def plot_results_4_subplots(means, variances, data_points_sets, titles, n):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create 2x2 grid of subplots
    x_values = np.linspace(-2, 2, 100)
    for i, ax in enumerate(axs.flat):
        mean = means[i]
        variance = variances[i]
        data_points = data_points_sets[i]
        title = titles[i]

        mean_polynomial = polynomial_mean_function(mean, x_values)

        if i == 0:
            ax.plot(x_values, mean_polynomial, 'k-', label="Predictive Mean")
            ax.plot(x_values, mean_polynomial + variance, 'r-', label="+1 Std Dev (Variance)")
            ax.plot(x_values, mean_polynomial - variance, 'r-', label="-1 Std Dev (Variance)")
        else:
            ax.plot(x_values, mean_polynomial, 'k-', label="Predictive Mean", zorder=2)
            var = np.zeros(100)
            for i in range(100):
                A = designMatrix(x_values[i], n)
                var[i] = variances[0] + A @ variance @ A.T
            ax.plot(x_values, mean_polynomial + var, 'r-', label="+1 Std Dev (Variance)", zorder=2)
            ax.plot(x_values, mean_polynomial - var, 'r-', label="-1 Std Dev (Variance)", zorder=2)

            data_x = [point[0] for point in data_points]
            data_y = [point[1] for point in data_points]
            ax.scatter(data_x, data_y, c='blue', label="Seen Data Points", zorder=1, alpha=0.4)

        ax.set_title(title)
        ax.grid(True)

    fig.tight_layout()  # Adjust layout for better spacing
    plt.show()


def baysianLinearRegression(prior_pre, n, like_var, w_list):
    epsilon = 1e-5
    samplepoints = []
    count = 0
    like_pre = 1 / like_var
    prior_covariance = np.linalg.inv(prior_pre * np.identity(n))
    prior_mean = np.zeros((1, n))
    while True:
        x, y = randomNumberGeneratorBasisLinear(n, like_var, w_list)
        samplepoints.append([x, y])
        A = designMatrix(x, n)
        if count == 0:
            covariance = np.linalg.inv(like_pre * A.T @ A + np.linalg.inv(prior_covariance))
            mean = covariance * like_pre @ A.T * y

        else:
            covariance = np.linalg.inv(like_pre * A.T @ A + np.linalg.inv(prior_covariance))
            mean = covariance @ (np.linalg.inv(prior_covariance) @ prior_mean + like_pre * A.T * y)
        count += 1

        predict_mean = A @ mean
        predict_variance = A @ covariance @ A.T + like_var   #MSE

        if count == 10:
            mean10 = mean
            covariance10 = covariance
        elif count == 50:
            mean50 = mean
            covariance50 = covariance

        print(f'Add data point ({x}, {y}):')
        print(f'Postirior mean:')
        print(mean)
        print()
        print(f'Posterior variance:')
        print(covariance)
        print()
        print(f'Predictive distribution ~ N({predict_mean[0, 0]}, {predict_variance[0, 0]})')
        print()

        mean_diff = np.linalg.norm(mean - prior_mean)
        var_diff = np.linalg.norm(covariance - prior_covariance)
        if mean_diff < epsilon and count > 50 and var_diff < epsilon:
            break

        prior_mean = mean
        prior_covariance = covariance

    means = [w_list, mean, mean10, mean50]
    variances = [like_var, covariance, covariance10, covariance50]
    datapoints = [None, samplepoints, samplepoints[:10], samplepoints[:50]]
    titles = ['Ground truth', 'Predict result', 'After 10 incomes', 'After 50 incomes']
    plot_results_4_subplots(means, variances, datapoints, titles, n)


def main():
    b = float(input('precision for prior:'))
    n = int(input('basis number:'))
    a = float(input('variance for likelihood:'))
    w = input('ground truth:')
    w_list = [float(x) for x in w.split()]
    estimate = input('Estimation or not (0 or 1):')

    if estimate == '0':
        x, y = randomNumberGeneratorBasisLinear(n, a, w_list)
        print(f'Random number generator outcome: x:{x},y:{y}')
    else:
        baysianLinearRegression(b, n, a, w_list)
    pass


if __name__ == '__main__':
    main()
