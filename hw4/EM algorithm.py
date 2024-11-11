from parse_data import load_mnist
import numpy as np


def Estep(images, lam, bernoulli):  # image:x,bernoulli:theta;class:z
    num_images = images.shape[0]
    height, width = images[0].shape
    w = np.zeros((num_images, 10))
    for i in range(num_images):
        for j in range(10):
            w[i, j] = np.prod(bernoulli[j] * images[i].flatten() + (1 - bernoulli[j]) * (1 - images[i].flatten()))
            w[i, j] *= lam[j, 0]
        # w[i] /= np.sum(w[i])
    sums = np.sum(w, axis=1).reshape(-1, 1)  # normalize (/sum{z})
    sums[sums == 0] = 1
    w = w / sums
    return w


def Mstep(images, w):  # weighted log likelihood
    num_images = images.shape[0]
    height, width = images[0].shape
    lam = np.sum(w, axis=0) / num_images
    sums = np.sum(w, axis=0)
    sums[sums == 0] = 1
    # w_normalized = w / sums
    weighted_images = w.T @ images.reshape(num_images, height * width)  # (10, H*W)
    bernoulli = weighted_images / sums[:, np.newaxis]  # (10, H*W)
    return lam[:, np.newaxis], bernoulli


def assign_labels(w, labels):
    num_images = w.shape[0]
    cluster_labels = np.zeros(10, dtype=int) - 1
    assigned_labels = set()
    all_labels = set(range(10))
    empty_clusters = []

    for i in range(10):  # 找target label
        cluster_indices = np.argmax(w, axis=1) == i  # 針對每張照片照max w
        if np.sum(cluster_indices) > 0:
            # print(i)
            target_label = np.bincount(labels[cluster_indices]).argmax()

            # 如果 target_label 已被使用，從未分配標籤中挑選一個
            if target_label in assigned_labels:
                remaining_labels = all_labels - assigned_labels
                if remaining_labels:
                    target_label = max(remaining_labels, key=lambda x: np.bincount(labels[cluster_indices] == x).sum())

            cluster_labels[i] = target_label
            assigned_labels.add(target_label)
        else:
            empty_clusters.append(i)

    remaining_labels = list(all_labels - assigned_labels)
    for idx, cluster in enumerate(empty_clusters):
        cluster_labels[cluster] = remaining_labels[idx]

    return cluster_labels


def imagination(bernoulli, height, width, labels):
    for i in range(10):
        j = np.where(labels == i)[0]
        # print(j)
        digit_pattern = (bernoulli[j] > 0.5).reshape(height, width).astype(int)
        print(f"\nlabeled class {i}:")
        for row in digit_pattern:
            print("".join(map(str, row)))
        print()


def calculate_metrics(true_labels, predicted_cluster, digit):
    true_positive = np.sum((true_labels == digit) & (predicted_cluster == digit))
    true_negative = np.sum((true_labels != digit) & (predicted_cluster != digit))
    false_positive = np.sum((true_labels != digit) & (predicted_cluster == digit))
    false_negative = np.sum((true_labels == digit) & (predicted_cluster != digit))

    sensitivity = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) > 0 else 0

    confusion_matrix = np.array([
        [true_positive, false_positive],
        [false_negative, true_negative]
    ])
    return confusion_matrix, sensitivity, specificity


def main(train_images, train_labels):
    eps = 0.25
    binning_image = (train_images > 127).astype(int)
    # print(binning_image[0])
    height, width = train_images[0].shape
    # initial lambda and bernoulli
    lam = np.random.rand(10, 1)
    lam /= np.sum(lam)
    bernoulli = np.random.rand(10, height * width)
    count = 0
    while True:
        # print(bernoulli.shape)
        responsibility = Estep(binning_image, lam, bernoulli)  # (num_images, 10)
        newlam, newbernoulli = Mstep(binning_image, responsibility)
        # print(newlam)
        print(f'diff: {np.linalg.norm(newbernoulli - bernoulli) + np.linalg.norm(newlam - lam)}')
        if np.linalg.norm(newbernoulli - bernoulli) + np.linalg.norm(newlam - lam) < eps or count > 20:
            break
        lam = newlam
        bernoulli = newbernoulli
        count += 1
    labels = assign_labels(responsibility, train_labels)
    # print(labels)
    predicted = labels[np.argmax(responsibility, axis=1)]
    imagination(bernoulli, height, width, labels)
    print("\nMetrics for each digit:")
    for digit in range(10):
        confusion_matrix, sensitivity, specificity = calculate_metrics(train_labels, predicted, digit)
        print(f"\nDigit {digit}:")
        print("Confusion Matrix:")
        print("            Predicted")
        print("            Pos  Neg")
        print(f"Actual Pos  {confusion_matrix[0, 0]:4d}  {confusion_matrix[0, 1]:4d}")
        print(f"      Neg  {confusion_matrix[1, 0]:4d}  {confusion_matrix[1, 1]:4d}")
        print(f"Sensitivity: {sensitivity:.3f}")
        print(f"Specificity: {specificity:.3f}")
    print("\nOverall accuracy:", np.mean(predicted == train_labels))


if __name__ == '__main__':
    train_images_path = 'train-images.idx3-ubyte'
    train_labels_path = 'train-labels.idx1-ubyte'
    (train_images, train_labels) = load_mnist(
        train_images_path, train_labels_path)
    main(train_images, train_labels)
