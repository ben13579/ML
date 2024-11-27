from libsvm.svmutil import *
from scipy.spatial.distance import cdist
import prettytable as pt
import numpy as np


def load_mnist(x_train_path, y_train_path, x_test_path, y_test_path):
    x_train = np.genfromtxt(x_train_path, delimiter=',')
    y_train = np.genfromtxt(y_train_path, delimiter=',')
    x_test = np.genfromtxt(x_test_path, delimiter=',')
    y_test = np.genfromtxt(y_test_path, delimiter=',')

    return x_train, y_train, x_test, y_test


def grid_search(x_train, y_train, kernel_type, param_grid):
    best_score = -np.inf
    best_params = None
    results = []
    for C in param_grid['C']:
        for gamma in param_grid.get('gamma', [None]):  # gamma only for RBF and poly
            for degree in param_grid.get('degree', [None]):  # degree only for poly
                params = f"-q -s 0 -t {kernel_type} -c {C}"
                if gamma is not None:
                    params += f" -g {gamma}"
                if degree is not None:
                    params += f" -d {degree}"
                score = svm_train(y_train, x_train, params + " -v 5")  # Cross-validation
                if score > best_score:
                    best_score = score
                    best_params = params
                if kernel_type == 0:
                    results.append([C, score])
                elif kernel_type == 1:
                    results.append([C, gamma, degree, score])
                elif kernel_type == 2:
                    results.append([C, gamma, score])
    if kernel_type == 0:
        print('Linear kernel')
        tb = pt.PrettyTable()
        tb.field_names = ["C", "score"]
        for l in results:
            tb.add_row(l)
        print(tb)
    elif kernel_type == 1:
        print('Polynomial kernel')
        tb = pt.PrettyTable()
        tb.field_names = ["C", "gamma", "degree", "score"]
        for l in results:
            tb.add_row(l)
        print(tb)
    elif kernel_type == 2:
        print('RBF kernel')
        tb = pt.PrettyTable()
        tb.field_names = ["C", "gamma", "score"]
        for l in results:
            tb.add_row(l)
        print(tb)
    return best_params, best_score


def train_and_evaluate(x_train, y_train, x_test, y_test, params):
    model = svm_train(y_train, x_train, params)
    predictions, accuracy, _ = svm_predict(y_test, x_test, model)
    return predictions, accuracy


def task1(x_train, y_train, x_test, y_test):
    params_linear = '-q -t 0'
    params_poly = '-q -t 1'
    params_rbf = '-q -t 2'

    preds_linear, acc_linear = train_and_evaluate(x_train, y_train, x_test, y_test, params_linear)
    preds_poly, acc_poly = train_and_evaluate(x_train, y_train, x_test, y_test, params_poly)
    preds_rbf, acc_rbf = train_and_evaluate(x_train, y_train, x_test, y_test, params_rbf)

    print("Linear Kernel Accuracy:", acc_linear)
    print("Polynomial Kernel Accuracy:", acc_poly)
    print("RBF Kernel Accuracy:", acc_rbf)
    print()


def task2(x_train, y_train, x_test, y_test):
    # Define parameter grids
    param_grid_linear = {'C': [0.1, 1, 10]}
    param_grid_poly = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1], 'degree': [2, 3, 4]}
    param_grid_rbf = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}

    best_params_linear, best_score_linear = grid_search(x_train, y_train, kernel_type=0, param_grid=param_grid_linear)
    best_params_poly, best_score_poly = grid_search(x_train, y_train, kernel_type=1, param_grid=param_grid_poly)
    best_params_rbf, best_score_rbf = grid_search(x_train, y_train, kernel_type=2, param_grid=param_grid_rbf)

    preds_linear, acc_linear = train_and_evaluate(x_train, y_train, x_test, y_test, best_params_linear)
    preds_poly, acc_poly = train_and_evaluate(x_train, y_train, x_test, y_test, best_params_poly)
    preds_rbf, acc_rbf = train_and_evaluate(x_train, y_train, x_test, y_test, best_params_rbf)

    print("Linear Kernel Best Parameters:", best_params_linear[13:], "Accuracy:", acc_linear)
    print("Polynomial Kernel Best Parameters:", best_params_poly[13:], "Accuracy:", acc_poly)
    print("RBF Kernel Best Parameters:", best_params_rbf[13:], "Accuracy:", acc_rbf)


def get_kernel(x):
    num_samples = x.shape[0]
    kernel = np.zeros((num_samples, num_samples + 1))
    kernel[:, 0] = np.arange(1, num_samples + 1)  # First column: row indices
    linear = x @ x.T
    rbf = np.exp(-1 / 784 * cdist(x, x, 'sqeuclidean'))
    kernel[:, 1:] = linear + rbf
    return kernel


def task3(x_train, y_train, x_test, y_test):
    params_linear_rbf = '-q -t 4'
    kernel_train = get_kernel(x_train)
    kernel_test = get_kernel(x_test)
    preds_linear_rbf, acc_linear_kbf = train_and_evaluate(kernel_train, y_train, kernel_test, y_test, params_linear_rbf)
    print(f'Linear Kernel + RBF Kernel Accuracy:{acc_linear_kbf}')


def main(x_train, y_train, x_test, y_test):
    # Normalize the data
    std_train = np.std(x_train, axis=0)
    std_train[std_train == 0] = 1
    std_test = np.std(x_test, axis=0)
    std_test[std_test == 0] = 1
    x_train = (x_train - np.mean(x_train, axis=0)) / std_train
    x_test = (x_test - np.mean(x_test, axis=0)) / std_test

    task1(x_train, y_train, x_test, y_test)
    task2(x_train, y_train, x_test, y_test)
    task3(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    x_train_path = 'X_train.csv'
    y_train_path = 'Y_train.csv'
    x_test_path = 'X_test.csv'
    y_test_path = 'Y_test.csv'
    x_train, y_train, x_test, y_test = load_mnist(x_train_path, y_train_path, x_test_path, y_test_path)
    main(x_train, y_train, x_test, y_test)
