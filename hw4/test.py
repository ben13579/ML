from parse_data import load_mnist
import numpy as np


def Estep(images, lam, bernoulli):
    num_images = images.shape[0]
    w = np.zeros((num_images, 10))

    images_flat = images.reshape(num_images, -1)

    for j in range(10):
        log_probs = (np.log(bernoulli[j] + 1e-10) * images_flat +
                     np.log(1 - bernoulli[j] + 1e-10) * (1 - images_flat))
        w[:, j] = np.exp(np.sum(log_probs, axis=1)) * lam[j, 0]

    sums = np.sum(w, axis=1)
    mask = sums > 0
    w[mask] = w[mask] / sums[mask, np.newaxis]
    w[~mask] = 1.0 / 10

    return w


def Mstep(images, w):
    num_images = images.shape[0]

    lam = np.sum(w, axis=0) / num_images
    sums = np.sum(w, axis=0) + 1e-10
    w_normalized = w / sums[np.newaxis, :]

    images_flat = images.reshape(num_images, -1)
    weighted_images = (w_normalized.T @ images_flat)
    bernoulli = np.clip(weighted_images / sums[:, np.newaxis], 1e-10, 1 - 1e-10)

    return lam[:, np.newaxis], bernoulli


def assign_labels(w, labels):
    num_clusters = 10
    cluster_labels = np.full(num_clusters, -1)
    assigned_labels = set()

    cluster_assignments = np.argmax(w, axis=1)
    cluster_sizes = [(i, np.sum(cluster_assignments == i)) for i in range(num_clusters)]
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    for cluster_idx, size in cluster_sizes:
        if size == 0:
            continue

        cluster_mask = cluster_assignments == cluster_idx
        cluster_true_labels = labels[cluster_mask]
        label_counts = np.bincount(cluster_true_labels)

        for label in np.argsort(label_counts)[::-1]:
            if label not in assigned_labels:
                cluster_labels[cluster_idx] = label
                assigned_labels.add(label)
                break

    remaining_labels = set(range(10)) - assigned_labels
    unassigned_clusters = np.where(cluster_labels == -1)[0]
    for cluster_idx, label in zip(unassigned_clusters, remaining_labels):
        cluster_labels[cluster_idx] = label

    return cluster_labels


def calculate_metrics(true_labels, predicted_labels, digit):
    """Calculate confusion matrix, sensitivity, and specificity for each digit"""
    true_pos = np.sum((true_labels == digit) & (predicted_labels == digit))
    true_neg = np.sum((true_labels != digit) & (predicted_labels != digit))
    false_pos = np.sum((true_labels != digit) & (predicted_labels == digit))
    false_neg = np.sum((true_labels == digit) & (predicted_labels != digit))

    sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

    confusion_matrix = np.array([
        [true_pos, false_pos],
        [false_neg, true_neg]
    ])

    return confusion_matrix, sensitivity, specificity


def imagination(bernoulli, height=28, width=28):
    """Visualize the learned digit patterns"""
    print("\nImagined Patterns for Each Digit Class:")
    print("-" * 40)

    for i in range(10):
        # Convert bernoulli parameters to binary pattern
        digit_pattern = (bernoulli[i] > 0.5).reshape(height, width)
        print(f"\nLabeled Class {i}:")
        # Print pattern using blocks for better visibility
        for row in digit_pattern:
            print(''.join(['1' if pixel else '0' for pixel in row]))
    print("-" * 40)


def main(train_images, train_labels, max_iterations=10, convergence_threshold=0.0025):
    binning_image = (train_images > 127).astype(int)
    height, width = train_images[0].shape

    # Initialize parameters
    np.random.seed(42)
    lam = np.random.rand(10, 1)
    lam /= np.sum(lam)
    bernoulli = np.random.rand(10, 784)

    # EM iterations
    responsibility = None
    for iteration in range(max_iterations):
        try:
            responsibility = Estep(binning_image, lam, bernoulli)
            new_lam, new_bernoulli = Mstep(binning_image, responsibility)

            diff = np.linalg.norm(new_bernoulli - bernoulli)
            print(f"Iteration {iteration + 1}, Parameter difference: {diff:.6f}")

            if diff < convergence_threshold:
                print("Converged!")
                break

            bernoulli = new_bernoulli
            lam = new_lam

        except Exception as e:
            print(f"Error in iteration {iteration}: {str(e)}")
            raise

    if responsibility is None:
        raise RuntimeError("EM algorithm failed to complete any iterations")

    # Assign labels and evaluate
    cluster_labels = assign_labels(responsibility, train_labels)
    predicted = cluster_labels[np.argmax(responsibility, axis=1)]

    # Print imagination patterns
    imagination(bernoulli, height=28, width=28)

    # Calculate and print metrics for each digit
    print("\nMetrics for Each Digit:")
    print("=" * 50)

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

    return bernoulli, predicted


if __name__ == '__main__':
    train_images_path = 'train-images.idx3-ubyte'
    train_labels_path = 'train-labels.idx1-ubyte'

    try:
        train_images, train_labels = load_mnist(train_images_path, train_labels_path)
        bernoulli, predicted = main(train_images, train_labels)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise