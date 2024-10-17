import numpy as np
import math
from scipy.special import erfinv


def probit(num):
    return math.sqrt(2) * erfinv(2 * num - 1)


"""
if U is distributed uniformly on (0,1), then Φ−1(U) will have the standard normal distribution.
1.num from uniform distribution
2.derive probit function
"""


def randomNumberGeneratorGaussian(mu, var):
    num = np.random.uniform()
    return math.sqrt(var) * probit(num) + mu


"""
count += 1
delta = new_value - mean
mean += delta / count
delta2 = new_value - mean
M2 += delta * delta2
"""


def sequentialEstimator(mu, var):
    epsilon = 0.01
    randNum = randomNumberGeneratorGaussian(mu, var)
    mean = randNum
    variance = 0
    n = 1
    M2 = 0
    square = pow(randNum, 2)
    while True:
        print(f'Add data point:{randNum}')
        print(f'Mean = {mean} Variance = {variance}')
        if abs(mean - mu) < epsilon and abs(variance - var) < epsilon:
            break

        randNum = randomNumberGeneratorGaussian(mu, var)

        mean = (n * mean + randNum) / (n + 1)
        square += pow(randNum, 2)
        variance = (square - (n + 1) * pow(mean, 2)) / n
        n += 1

        # n += 1  # Welford's online algorithm
        # delta = randNum - mean
        # mean += delta / n
        # delta2 = randNum - mean
        # M2 += delta * delta2
        # variance = M2 / (n - 1)

    # print(n)


if __name__ == '__main__':
    mu = float(input('Expectation value or mean:'))
    var = float(input('Variance:'))
    estimate = input('Estimation or not (0 or 1):')

    if estimate == '0':
        randomNum = randomNumberGeneratorGaussian(mu, var)
        print(f'Random number generator outcome:{randomNum}')
    else:
        print(f'Data point source function: N({mu}, {var})')
        sequentialEstimator(mu, var)
