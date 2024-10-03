import numpy as np
import math


def combination(n, m):
    return math.factorial(n) / (math.factorial(n - m) * math.factorial(m))


if __name__ == '__main__':
    filename = 'testfile.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    a = int(input('a:'))
    b = int(input('b:'))
    i = 1

    for line in lines:
        heads = 0
        tails = 0
        for char in line:
            if char == '0':
                tails += 1
            elif char == '1':
                heads += 1
        p = heads / (heads + tails)
        likelihood = combination(heads + tails, heads) * math.pow(p, heads) * math.pow(1 - p, tails)
        print(f'Case {i}: {line[:len(line) - 1]}')
        print(f'Likelihood: {likelihood}')
        print(f'Beta prior: a={a},b={b}')
        a = heads + a
        b = tails + b
        print(f'Beta posterior: a={a},b={b}')
        i += 1
        print('')
