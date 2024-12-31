import random
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial import distance


def transfrom_to_faces(vectors):
    eigenfaces = np.copy(vectors.T)
    fig, axes = plt.subplots(5, 5, figsize=(15, 5))
    axes = axes.flat
    for i, ax in enumerate(axes):
        eigenface = eigenfaces[i].reshape((image_height, image_width))
        eigenface = (eigenface - eigenface.min()) / (eigenface.max() - eigenface.min())
        ax.imshow(eigenface, cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_reconstruction(images, normalized_eigenvectors):
    pick = set()
    while len(pick) < 10:
        pick.add(random.randint(0, images.shape[0] - 1))

    fig, axes = plt.subplots(10, 2, figsize=(5, 15))
    for i, idx in enumerate(pick):
        # reconstruct the image by transfer to low dimension first and then transfer back into high dimension
        reconstruct_image = images[idx] @ normalized_eigenvectors @ normalized_eigenvectors.T
        axes[i, 0].imshow(images[idx].reshape(image_height, image_width), cmap='gray')
        axes[i, 0].axis("off")
        axes[i, 1].imshow(reconstruct_image.reshape(image_height, image_width), cmap='gray')
        axes[i, 1].axis('off')
    plt.tight_layout()
    plt.show()


def face_recognition(z_train, train_labels, z_test, test_labels, k=3):
    correct = 0
    for i in range(z_test.shape[0]):
        distances = np.linalg.norm(z_test[i] - z_train, axis=1)
        dis = np.argsort(distances)
        unique, count = np.unique(train_labels[dis[:k]], return_counts=True)
        idx = np.argmax(count)
        if test_labels[i] == unique[idx]:
            correct += 1
    print(f'accuracy:{correct / z_test.shape[0]}')


def get_covariance(data):
    num_datapoints = data.shape[0]
    mean = np.mean(data, axis=0)
    diff = data - mean
    # Perform PCA with memory-efficient covariance matrix computation. origin: diff_train.T @ diff_train
    small_cov = diff @ diff.T / num_datapoints
    return small_cov


def get_eigenvectors(data, small_cov):
    eigenvalues, eigenvectors = np.linalg.eig(small_cov)

    # turn the eigenvectors of the small matrix back to origin feature space's eigenvectors
    normalized_eigenvectors = data.T @ eigenvectors / np.linalg.norm(data.T @ eigenvectors, axis=0)

    # sort eigenvalue from large to small
    indices = np.argsort(eigenvalues)[::-1]
    normalized_eigenvectors = normalized_eigenvectors[:, indices]
    return normalized_eigenvectors


def get_kernel_function(X, Y, kernel_type):
    # num_points = images.shape[0]
    # construct kernel matrix
    K = np.zeros((X.shape[0], Y.shape[0]))
    if kernel_type == 'rbf':
        gamma = 1.0 / X.shape[1]
        # gamma = 0.1
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                diff = X[i] - Y[j]
                K[i, j] = np.exp(-gamma * diff.dot(diff))
    else:
        degree = 3
        for i in range(X.shape[0]):
            for j in range(Y.shape[0]):
                K[i, j] = np.power(X[i].dot(Y[j]), degree)

    # center kernel
    n_samples_X = X.shape[0]
    n_samples_Y = Y.shape[0]

    one_n_X = np.ones((n_samples_X, n_samples_X)) / n_samples_X
    one_n_Y = np.ones((n_samples_Y, n_samples_Y)) / n_samples_Y

    K_centered = K - one_n_X @ K - K @ one_n_Y + one_n_X @ K @ one_n_Y
    return K_centered


def read_data(folder_path):
    num_images = len(os.listdir(folder_path))
    images = np.zeros((num_images, image_height * image_width), dtype=np.uint8)
    labels = np.zeros(num_images)
    i = 0
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # print(file_path)
        labels[i] = int(file_name.split('.')[0][7:9]) - 1
        images[i, :] = np.asarray(Image.open(file_path).resize((image_width, image_height))).flatten()
        i += 1
    return images, labels


image_height = 195
image_width = 231
