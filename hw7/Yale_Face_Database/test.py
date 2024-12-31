import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import read_data


def kernelPCA_and_kernelLDA(train_images, train_labels, test_images, test_labels, kernel_type='rbf',
                            n_components_pca=30, n_components_lda=None):
    """
    Perform Kernel PCA and Kernel LDA for face recognition.

    Parameters:
        train_images (ndarray): Training images of shape (n_train, n_features).
        train_labels (ndarray): Training labels of shape (n_train,).
        test_images (ndarray): Testing images of shape (n_test, n_features).
        test_labels (ndarray): Testing labels of shape (n_test,).
        kernel_type (str): Kernel type ('rbf' or 'poly').
        n_components_pca (int): Number of components for Kernel PCA.
        n_components_lda (int): Number of components for LDA (default: None, will auto-set to n_classes - 1).
    """
    # Step 1: Kernel PCA
    print("Performing Kernel PCA...")
    kpca = KernelPCA(n_components=n_components_pca, kernel=kernel_type, fit_inverse_transform=False, gamma=0.0001,
                     degree=3)
    z_train_pca = kpca.fit_transform(train_images)
    z_test_pca = kpca.transform(test_images)
    print(f"Kernel PCA output shapes - Train: {z_train_pca.shape}, Test: {z_test_pca.shape}")

    print("Performing face recognition...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(z_train_pca, train_labels)
    predictions = knn.predict(z_test_pca)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Face Recognition Accuracy: {accuracy:.4f}")

    # Step 2: Kernel LDA
    print("Performing LDA...")
    if n_components_lda is None:
        n_components_lda = len(np.unique(train_labels)) - 1

    lda = LDA(n_components=n_components_lda)
    z_train_lda = lda.fit_transform(z_train_pca, train_labels)
    z_test_lda = lda.transform(z_test_pca)
    print(f"LDA output shapes - Train: {z_train_lda.shape}, Test: {z_test_lda.shape}")

    # Step 3: Face recognition using KNN
    print("Performing face recognition...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(z_train_lda, train_labels)
    predictions = knn.predict(z_test_lda)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Face Recognition Accuracy: {accuracy:.4f}")


def main():
    train_images, train_labels = read_data(training_path)
    test_images, test_labels = read_data(testing_path)
    kernelPCA_and_kernelLDA(train_images, train_labels, test_images, test_labels, kernel_type='rbf',
                            n_components_pca=50)
    # PCA(train_images, train_labels, test_images, test_labels)
    # kernelPCA(train_images, train_labels, test_images, test_labels, kernel_type)


if __name__ == '__main__':
    training_path = 'Yale_Face_Database/Training'
    testing_path = 'Yale_Face_Database/Testing'
    image_height = 195
    image_width = 231
    k = 3
    kernel_type = 'rbf'
    # kernel_type = 'polynomial'
    main()
