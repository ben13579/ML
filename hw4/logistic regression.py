import numpy as np
import math
from scipy.special import erfinv
import matplotlib.pyplot as plt


def probit(num):
    return math.sqrt(2) * erfinv(2 * num - 1)


def RNGgaussian(mu, var):
    num = np.random.uniform()
    return math.sqrt(var) * probit(num) + mu


def generateData(mx, my, vx, vy, n):
    return np.array([[RNGgaussian(mx, vx), RNGgaussian(my, vy)] for i in range(n)])


def designMatrix(data, n):  # 1,x,y
    A = np.ones((2 * n, 3))
    A[:, 1:] = data
    # print(A)
    return A


def target(n):  # 0,1
    b = np.zeros((2 * n, 1))
    b[n:, :] = 1
    # print(b)
    return b


def gradient_method(A, b):
    L = 0.001
    w = np.zeros((3, 1))
    while True:
        grad = A.T @ (1 / (1 + np.exp(-A @ w)) - b)
        w = w - grad * L
        if np.sqrt(np.sum(grad ** 2)) < 1e-3:
            break

    print('Gradient decent:')
    print('w:')
    print(w)
    return w


def newton_method(A, b, n):
    L = 0.001  # for gradient decent
    w = np.zeros((3, 1))
    while True:
        D = np.zeros((2 * n, 2 * n))
        for i in range(2 * n):
            Aw_scalar = (A[i] @ w).item()
            D[i, i] = np.exp(-Aw_scalar) / (1 + np.exp(-Aw_scalar)) ** 2
        H = A.T @ D @ A
        grad = A.T @ (1 / (1 + np.exp(-A @ w)) - b)
        try:
            invH = np.linalg.inv(H)
        except np.linalg.LinAlgError as error:
            print(str(error) + ',use gradient decent')
            w = w - grad * L
        else:
            w = w - invH @ grad
        if np.sqrt(np.sum(grad ** 2)) < 1e-3:
            break
    print('Newton\'s method:')
    print('w:')
    print(w)
    return w


def output(A, w, n):
    print('Confusion Matrix:')
    TP, FP, TN, FN = 0, 0, 0, 0
    threshold = 0.5
    pred1 = []
    pred2 = []
    for i in range(n):  # ground truth=0
        Aw_scalar = (A[i] @ w).item()
        predict = 1 / (1 + np.exp(-Aw_scalar))  # activation function
        # print(predict)
        if predict >= threshold:
            FP += 1
            pred1.append(1)
        else:
            TN += 1
            pred1.append(0)

    for i in range(n):  # ground truth=1
        Aw_scalar = (A[i + n] @ w).item()
        predict = 1 / (1 + np.exp(-Aw_scalar))
        if predict >= threshold:
            TP += 1
            pred2.append(1)
        else:
            FN += 1
            pred2.append(0)

    print("\nConfusion Matrix:")
    print('              Predict cluster 1 Predict cluster 2')
    print(f'Is cluster 1        {TN}             {FP}')
    print(f'Is cluster 2        {FN}             {TP}')
    print(f'Sensitivity (Successfully predict cluster 1):{TN / (TN + FP)}')
    print(f'Specificity (Successfully predict cluster 2):{TP / (FN + TP)}')
    return pred1, pred2


def visualize(dataset1, dataset2, grad_pred1, grad_pred2, newton_pred1, newton_pred2, n):
    # 設置畫布和子圖
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Visualization of Ground Truth, Gradient Descent, and Newton's Method")

    # Ground Truth Plot
    axes[0].scatter(dataset1[:, 0], dataset1[:, 1], color='blue', label='Class 0', alpha=0.6)
    axes[0].scatter(dataset2[:, 0], dataset2[:, 1], color='red', label='Class 1', alpha=0.6)
    axes[0].set_title('Ground Truth')
    axes[0].legend()
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Gradient Descent Result Plot
    for i in range(n):
        color = 'blue' if grad_pred1[i] == 0 else 'red'
        label = 'Class 0' if grad_pred1[i] == 0 and i == 0 else None  # 為Class 0設置單次label
        axes[1].scatter(dataset1[i, 0], dataset1[i, 1], color=color, label=label, alpha=0.6)
    for i in range(n):
        color = 'blue' if grad_pred2[i] == 0 else 'red'
        label = 'Class 1' if grad_pred2[i] == 1 and i == 0 else None  # 為Class 1設置單次label
        axes[1].scatter(dataset2[i, 0], dataset2[i, 1], color=color, label=label, alpha=0.6)
    axes[1].set_title('Gradient Descent Result')
    axes[1].legend()
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')

    # Newton's Method Result Plot
    for i in range(n):
        color = 'blue' if newton_pred1[i] == 0 else 'red'
        label = 'Class 0' if newton_pred1[i] == 0 and i == 0 else None
        axes[2].scatter(dataset1[i, 0], dataset1[i, 1], color=color, label=label, alpha=0.6)
    for i in range(n):
        color = 'blue' if newton_pred2[i] == 0 else 'red'
        label = 'Class 1' if newton_pred2[i] == 1 and i == 0 else None
        axes[2].scatter(dataset2[i, 0], dataset2[i, 1], color=color, label=label, alpha=0.6)
    axes[2].set_title("Newton's Method Result")
    axes[2].legend()
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')

    # 調整佈局，防止標題與子圖重疊
    plt.subplots_adjust(top=0.85, bottom=0.1)
    plt.tight_layout()
    plt.show()


def main():
    n = int(input('number of data points:'))
    mx1, my1 = [int(x) for x in input('mx1 and my1: ').split()]
    mx2, my2 = [int(x) for x in input('mx2 and my2: ').split()]
    vx1, vy1 = [int(x) for x in input('vx1 and vy1: ').split()]
    vx2, vy2 = [int(x) for x in input('vx2 and vy2: ').split()]
    dataset1 = generateData(mx1, my1, vx1, vy1, n)
    dataset2 = generateData(mx2, my2, vx2, vy2, n)
    data = np.concatenate((dataset1, dataset2))
    A = designMatrix(data, n)
    b = target(n)
    grad_w = gradient_method(A, b)
    grad_pred1, grad_pred2 = output(A, grad_w, n)
    newton_w = newton_method(A, b, n)
    newton_pred1, newton_pred2 = output(A, newton_w, n)
    visualize(dataset1, dataset2, grad_pred1, grad_pred2, newton_pred1, newton_pred2, n)


if __name__ == '__main__':
    main()
