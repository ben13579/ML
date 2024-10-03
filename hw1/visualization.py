import numpy as np
import matplotlib.pyplot as plt


# 讀取 (x, y) 點的資料
def read_xy_data(filename):
    x = []
    y = []
    with open(filename, 'r') as file:
        for line in file:
            xy = line.strip().split(',')  # 用逗號分隔 x 和 y
            x.append(float(xy[0]))
            y.append(float(xy[1]))
    return np.array(x), np.array(y)


# 讀取回歸方法產生的多項式係數
def read_regression_coefficients(filename):
    coeffs = []
    with open(filename, 'r') as file:
        for line in file:
            coeffs.append(float(line.strip()))  # 每行一個係數
    return np.array(coeffs)


# 計算回歸函數的值
def regression_function(coeffs, x):
    y_pred = np.zeros_like(x)
    for i, coeff in enumerate(coeffs):
        y_pred += coeff * (x ** i)  # 依照升冪次數加總
    return y_pred


# 繪製點與回歸曲線
def plot_regression(x, y, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', label='Data points')  # 原始點
    plt.plot(x, y_pred, color='red', label='Regression function')  # 回歸函數
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


# 計算總誤差 (L2 norm)
def calculate_total_error(y, y_pred):
    return np.sum((y - y_pred) ** 2)


def print_regression_equation(coeffs):
    terms = []
    for i, coeff in enumerate(coeffs):
        if i == 0:
            terms.append(f"{coeff:.4f}")  # 常數項
        else:
            terms.append(f"{coeff:.4f}x^{i}")

    equation = " + ".join(terms)
    print(f"Regression function: f(x) = {equation}")


# 主程式
def main():
    # 讀取 (x, y) 資料
    x, y = read_xy_data('testfile.txt')  # 更新檔案為 .txt 格式

    # 回歸方法的係數檔案
    regression_files = ['LSE', 'steepest', 'Newton']  # 例子中的 txt 檔案名稱

    # 繪製每個回歸函數及計算總誤差
    for i, reg_file in enumerate(regression_files):
        coeffs = read_regression_coefficients(reg_file + '.txt')  # 讀取多項式係數
        y_pred = regression_function(coeffs, x)  # 計算回歸函數的預測值
        total_error = calculate_total_error(y, y_pred)  # 計算總誤差

        print(f"{reg_file}:")
        print_regression_equation(coeffs)
        print(f"Total Error (L2 norm): {total_error:.4f}\n")

        # 繪圖
        title = f'{reg_file}, Total Error: {total_error:.4f}'
        plot_regression(x, y, y_pred, title)


# 執行程式
if __name__ == '__main__':
    main()
