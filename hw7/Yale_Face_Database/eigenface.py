from utils import *


def PCA(train_images, train_labels, test_images, test_labels):
    """
    train_images : (# of images, features)
    """
    # get the covariance matrix of the training dataset(high-dim)
    small_cov = get_covariance(train_images)

    # get the eigenvectors of the covariance matrix
    normalized_eigenvectors = get_eigenvectors(train_images, small_cov)

    # show the first 25 eigenfaces and use the eigenvectors to reconstruct the images
    transfrom_to_faces(normalized_eigenvectors[:, :25])
    show_reconstruction(train_images, normalized_eigenvectors)

    # transfer the dataset into low-dimensional space and do the face recognition
    z_train = train_images @ normalized_eigenvectors[:, :14]
    z_test = test_images @ normalized_eigenvectors[:, :14]
    face_recognition(z_train, train_labels, z_test, test_labels, k)


def kernelPCA(train_images, train_labels, test_images, test_labels, kernel_type):
    # get the kernel matrix of training dataset
    train_kernel = get_kernel_function(train_images, train_images, kernel_type)

    # calculate the eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(train_kernel)
    indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, indices]

    # use the eigenvectors to transfer to low dimension and then do the recognition
    test_kernel = get_kernel_function(test_images, train_images, kernel_type)
    z_train = train_kernel @ eigenvectors[:, :100]
    z_test = test_kernel @ eigenvectors[:, :100]
    face_recognition(z_train, train_labels, z_test, test_labels, k)


def main():
    train_images, train_labels = read_data(training_path)
    test_images, test_labels = read_data(testing_path)
    PCA(train_images, train_labels, test_images, test_labels)
    kernelPCA(train_images, train_labels, test_images, test_labels, kernel_type)


if __name__ == '__main__':
    training_path = 'Yale_Face_Database/Training'
    testing_path = 'Yale_Face_Database/Testing'
    image_height = 195
    image_width = 231
    k = 3
    # kernel_type = 'rbf'
    kernel_type = 'polynomial'
    main()
