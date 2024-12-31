from utils import *


def get_LDA_eigenvectors(matrix, labels):
    num_group = 15
    num_features = matrix.shape[1]
    S_W = np.zeros((num_features, num_features))
    S_W += 1e-6 * np.eye(S_W.shape[0])
    S_B = np.zeros((num_features, num_features))
    overall_mean = np.mean(matrix, axis=0)
    for i in range(num_group):
        groupI = matrix[labels == i]
        mean = np.mean(groupI, axis=0)
        S_W += (groupI - mean).T @ (groupI - mean)
        num_groupI = groupI.shape[0]
        mean_diff = (mean - overall_mean).reshape(1, -1)
        S_B += num_groupI * mean_diff.T @ mean_diff

    W = np.linalg.inv(S_W) @ S_B
    eigenvalues, eigenvectors = np.linalg.eig(W)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real.astype('float')
    # print(eigenvectors.dtype)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, indices]
    return eigenvectors


def LDA(train_images, train_labels, test_images, test_labels):
    """
    train_images : (# of images, features)
    """
    # do PCA first to reduce the dimension since the number of features are too large
    small_cov = get_covariance(train_images)
    pca_eigenvectors = get_eigenvectors(train_images, small_cov)
    pca_matrix = train_images @ pca_eigenvectors

    # get LDA eigenvectors
    num_group = 15
    eigenvectors = get_LDA_eigenvectors(pca_matrix, train_labels)
    eigenvectors = pca_eigenvectors @ eigenvectors

    # show the first 25 eigenfaces and use the eigenvectors to reconstruct the images
    transfrom_to_faces(eigenvectors[:, :25])
    show_reconstruction(train_images, eigenvectors[:, :num_group - 1])

    # use the eigenvectors to transfer to low dimension and then do the recognition
    z_train = train_images @ eigenvectors[:, :num_group - 1]
    z_test = test_images @ eigenvectors[:, :num_group - 1]
    face_recognition(z_train, train_labels, z_test, test_labels, k)


def kernelLDA(train_images, train_labels, test_images, test_labels, kernel_type):
    # get the kernel matrix of training dataset
    train_kernel = get_kernel_function(train_images, train_images, kernel_type)
    num_group = 15
    # get LDA eigenvectors
    eigenvectors = get_LDA_eigenvectors(train_kernel, train_labels)

    # use the eigenvectors to transfer to low dimension and then do the recognition
    z_train = train_kernel @ eigenvectors[:, :num_group - 1]
    test_kernel = get_kernel_function(test_images, train_images, kernel_type)
    z_test = test_kernel @ eigenvectors[:, :num_group - 1]
    face_recognition(z_train, train_labels, z_test, test_labels, k)


def main():
    train_images, train_labels = read_data(training_path)
    test_images, test_labels = read_data(testing_path)
    LDA(train_images, train_labels, test_images, test_labels)
    kernelLDA(train_images, train_labels, test_images, test_labels, kernel_type)


if __name__ == '__main__':
    training_path = 'Yale_Face_Database/Training'
    testing_path = 'Yale_Face_Database/Testing'
    image_height = 195
    image_width = 231
    k = 3
    # kernel_type = 'rbf'
    kernel_type = 'polynomial'
    main()
