import numpy as np
import math
from scipy import stats
from scipy.special import logsumexp


def training_discrete(train_images, train_labels):
    num_images, rows, cols = train_images.shape
    likelihood = np.zeros((10, rows * cols, 32), dtype=float)
    label_sum = np.zeros(10)
    train_images = train_images // 8
    for i in range(num_images):
        label = train_labels[i]
        label_sum[label] += 1
        for j in range(rows):
            for k in range(cols):
                likelihood[label, j * cols + k, train_images[i, j, k]] += 1
    for j in range(10):
        likelihood[j, :, :] /= label_sum[j]

    prior = label_sum / num_images
    return likelihood, prior


# def speed_training_discrete(train_images, train_labels):
#     num_images, rows, cols = train_images.shape
#     likelihood = np.zeros((10, rows * cols, 32), dtype=float)
#     label_sum = np.zeros(10)
#
#     # 將圖像轉換為一維並計算每個像素的 bin 值
#     flattened_images = train_images.reshape(num_images, rows * cols)
#     bins = flattened_images // 8  # 每個像素值映射到 32 個 bin（0-31）
#
#     # 逐類別累計每個 bin 的頻率
#     for label in range(10):
#         label_mask = train_labels == label  # 找到屬於該類別的圖像
#         label_sum[label] = label_mask.sum()  # 計算該類別的圖像數
#         # 將該類別的圖像按 bin 統計次數
#         np.add.at(likelihood[label],
#                   (np.arange(rows * cols)[:, np.newaxis].repeat(label_sum[label], 1), bins[label_mask].T), 1)
#
#     # 對每個類別的 bin 計算條件機率
#     likelihood /= label_sum[:, None, None]
#
#     # 計算每個類別的先驗機率
#     prior = label_sum / num_images
#
#     return likelihood, prior


def testing_discrete(test_images, test_labels, likelihood, prior):
    num_images, rows, cols = test_images.shape
    error = 0
    test_images = test_images // 8
    for i in range(num_images):
        label = test_labels[i]
        post = np.zeros(10)
        for j in range(10):
            for k in range(rows):
                for l in range(cols):
                    post[j] += math.log(max(1e-6, likelihood[j, k * cols + l, test_images[i, k, l]]))
            post[j] += math.log(prior[j])
        post /= sum(post)
        print('Posterior (in log scale):')
        for j in range(10):
            print(f'{j}: {post[j]}')
        print(f'Prediction:{np.argmin(post)}, Ans:{label}')
        if np.argmin(post) != label:
            error += 1
    print(f'error rate:{error / num_images}')

    print('Imagination of numbers in Bayesian classifier:')
    for i in range(10):
        print(f'{i}:')
        for j in range(rows):
            _str = ''
            for k in range(cols):
                intensity = np.argmax(likelihood[i, j * cols + k])
                if intensity < 16:
                    _str += '0 '
                else:
                    _str += '1 '
            print(_str)


# def speed_testing_discrete(test_images, test_labels, likelihood, prior):
#     num_images, rows, cols = test_images.shape
#     test_images = test_images // 8
#
#     print(np.arange(rows * cols)[:, None].shape)
#     print(test_images.reshape(num_images, -1).T.shape)
#     # Compute log-likelihood for all images at once
#     log_likelihood = np.log(
#         np.maximum(1e-6, likelihood[:, np.arange(rows * cols)[:, None], test_images.reshape(num_images, -1).T]))
#     log_likelihood = log_likelihood.reshape(10, rows, cols, num_images).sum(axis=(1, 2))  # [m, r, c, num] --> [m, num]
#
#     # Add log-prior
#     log_posterior = log_likelihood + np.log(prior)[:, None]  # [m, num] + [m, ]
#
#     log_posterior = np.swapaxes(log_posterior, 0, 1)
#
#     # print(log_posterior.shape, np.sum(log_posterior, axis=1).shape)
#     log_posterior /= np.sum(log_posterior, axis=1)[:, np.newaxis]
#     # # Normalize log-posterior
#     # log_posterior -= logsumexp(log_posterior, axis=0)
#
#     # Make predictions
#     predictions = np.argmin(log_posterior, axis=1)
#
#     # Calculate error rate
#     error_rate = np.mean(predictions != test_labels)
#
#     print(f'error rate: {error_rate}')
#
#     return error_rate


def training_continuous(train_images, train_labels):
    num_images, rows, cols = train_images.shape
    label_sum = np.zeros(10)
    means = np.zeros((10, rows * cols), dtype=float)
    variances = np.zeros((10, rows * cols), dtype=float)

    for i in range(10):
        LabelImage = train_images[train_labels == i]
        means[i, :] = np.mean(LabelImage.reshape(-1, rows * cols), axis=0)
        variances[i, :] = np.var(LabelImage.reshape(-1, rows * cols), axis=0) * (rows * cols) / (rows * cols - 1)
        variances[i, :] = np.where(variances[i, :] == 0, 10, variances[i, :])
        # var=0會導致之後除以var會有問題而且var太小testing時的忍受誤差的範圍也會太小

    for i in range(num_images):
        label = train_labels[i]
        label_sum[label] += 1

    prior = label_sum / num_images
    return means, variances, prior


def testing_continuous(test_images, test_labels, means, variances, prior):
    num_images, rows, cols = test_images.shape
    error = 0
    for i in range(num_images):
        label = test_labels[i]
        post = np.zeros(10)
        for j in range(10):
            mu = means[j]
            var = variances[j]
            post[j] = np.sum(stats.norm.logpdf(test_images[i].reshape(-1), mu, np.sqrt(var)))
            post[j] += math.log(prior[j])
        post /= sum(post)
        print('Posterior (in log scale):')
        for j in range(10):
            print(f'{j}: {post[j]}')
        print(f'Prediction:{np.argmin(post)}, Ans:{label}')
        if np.argmin(post) != label:
            error += 1
    print(f'error rate:{error / num_images}')

    print('Imagination of numbers in Bayesian classifier:')
    for i in range(10):
        print(f'{i}:')
        for j in range(rows):
            _str = ''
            for k in range(cols):
                intensity = int(means[i, j * cols + k] + 0.5)
                if intensity < 128:
                    _str += '0 '
                else:
                    _str += '1 '
            print(_str)


"""
prior = sum label(i)/total images
theta:0~9
observe:pixels a1=i1,a2=i2,...
likelihood:p(a1=i1|theta)*p(a2=i2|theta)*...*p(theta)
marginal

"""
