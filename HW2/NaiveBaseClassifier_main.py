import numpy as np
from parse_data import load_mnist
from NaiveBaseClassifier import *
import cv2

train_images_path = 'train-images.idx3-ubyte'
train_labels_path = 'train-labels.idx1-ubyte'
test_images_path = 't10k-images.idx3-ubyte'
test_labels_path = 't10k-labels.idx1-ubyte'

(train_images, train_labels), (test_images, test_labels) = load_mnist(
    train_images_path, train_labels_path, test_images_path, test_labels_path
)

# print(f"訓練圖片形狀: {test_images.shape}")
# print(f"訓練標籤形狀: {test_labels.shape}")
if __name__ == '__main__':
    toggle_option = input('toggle_option(0:discrete,1:continuous): ')
    # print(train_labels[0])
    # cv2.imshow('',train_images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    if toggle_option == '0':
        print('training...')
        likelihood, prior = training_discrete(train_images, train_labels)
        print('testing...')
        testing_discrete(test_images, test_labels, likelihood, prior)
    else:
        print('training...')
        means, variances, prior = training_continuous(train_images, train_labels)
        print('testing...')
        testing_continuous(test_images, test_labels, means, variances, prior)

"""
prior = sum label(i)/total images
theta:0~9
observe:pixels a1=i1,a2=i2,...
likelihood:p(a1=i1|theta)*p(a2=i2|theta)*...*p(theta)
marginal

"""
