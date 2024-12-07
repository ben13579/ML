import numpy as np
import matplotlib.pyplot as plt


def ration_cut(W, k=2, init_type='random'):
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    eigenvalues, eigenvectors = np.linalg.eig(L)
    np.save('eigenvalue_ratiocut', eigenvalues)
    np.save('eigenvector_ratiocut', eigenvectors)
    eigenvalues = np.load("eigenvalue_ratiocut.npy")
    eigenvectors = np.load("eigenvector_ratiocut.npy")
    sort_eigval = np.argsort(eigenvalues)
    H = eigenvectors[:, sort_eigval[1:k + 1]]
    return kmeans(H, k=k, init_type=init_type)


def normalized_cut(W, k=2, init_type='random'):
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    sqrtD = np.sqrt(D)
    norL = sqrtD @ L @ sqrtD
    eigenvalues, eigenvectors = np.linalg.eig(norL)
    np.save('eigenvalue_normalizedcut', eigenvalues)
    np.save('eigenvector_normalizedcut', eigenvectors)
    eigenvalues = np.load("eigenvalue_normalizedcut.npy")
    eigenvectors = np.load("eigenvector_normalizedcut.npy")
    sort_eigval = np.argsort(eigenvalues)
    U = eigenvectors[:, sort_eigval[1:k + 1]]
    T = np.array([U[i] / np.sqrt(np.sum(U ** 2, axis=1))[i] for i in range(U.shape[0])])
    return kmeans(T, k=k, init_type=init_type)


def kmeans(G, k=2, init_type='random', epsilon=1e-3, max_iter=100):
    """
    K-Means implementation with support for Kernel K-Means.
    Args:
        G: numpy.ndarray
            Kernel (Gram) matrix or data points (n_samples, n_features).
        k: int
            Number of clusters.
        init_type: str
            Initialization type: "random" or "kmeans++".
    Returns:
        allLabels: list of numpy.ndarray
            Cluster assignments for each iteration.
    """
    points = G.shape[0]

    # Initialization
    if init_type == "kmeans++":
        centers = kmeans_plusplus(G, k)  # Implement or import kmeans_plusplus
        labels = np.zeros(points, dtype=np.int32)
        for i in range(points):
            distances = [np.linalg.norm(G[i] - G[c]) for c in centers]
            labels[i] = np.argmin(distances)
    else:  # Random initialization
        labels = np.random.randint(0, k, size=points)

    allLabels = [labels.copy()]
    count = 0
    while count < max_iter:
        # Compute cluster means
        clusters_mean = []
        for i in range(k):
            cluster_points = np.where(labels == i)[0]
            if len(cluster_points) == 0:  # Handle empty clusters
                clusters_mean.append(np.zeros_like(G[0]))
            else:
                clusters_mean.append(np.mean(G[cluster_points], axis=0))
        clusters_mean = np.array(clusters_mean)
        # Update labels by minimizing distances
        new_labels = np.zeros(points, dtype=np.int32)
        for j in range(points):
            distances = [np.linalg.norm(G[j] - clusters_mean[i]) for i in range(k)]
            new_labels[j] = np.argmin(distances)
        # Append labels to track progress
        allLabels.append(new_labels.copy())
        # Check for convergence
        if np.sum(labels != new_labels) < epsilon * points:
            break
        labels = new_labels.copy()
        count += 1

    return allLabels, G


def kmeans_plusplus(G, k):
    points = G.shape[0]
    # Step 1: Randomly select the first cluster center
    centers = [np.random.choice(points)]

    # Step 2: Iteratively select the next cluster centers
    for _ in range(1, k):
        # Compute squared distances to the closest center
        distances = np.zeros((points, len(centers)))
        for i, center in enumerate(centers):
            for j in range(points):
                distances[j, i] = np.linalg.norm(G[j] - G[center])
        distances = np.min(distances, axis=1)

        # Choose the next center with a probability proportional to distance^2
        probabilities = distances / distances.sum()
        next_center = np.random.choice(points, p=probabilities)
        centers.append(next_center)

    return centers


def plot_eigenspace(H, labels, save_path, img_num):
    """
    Visualize the eigenspace and cluster assignments.

    Args:
        H: numpy.ndarray
            Coordinates in eigenspace (n_samples, k).
        labels: numpy.ndarray
            Cluster assignments.
    """
    if H.shape[1] > 2:
        # If eigenspace is more than 2D, use only the first two dimensions for plotting
        H = H[:, :2]

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(H[:, 0], H[:, 1], c=labels, cmap="tab10", s=50, alpha=0.8)
    # plt.colorbar(scatter, label="Cluster ID")
    plt.xlabel("Eigenvector 1")
    plt.ylabel("Eigenvector 2")
    plt.title("Data Points in the Eigendecomposition Space")
    plt.grid()
    plt.savefig(f"image{img_num}/{save_path}.png")
    plt.show()
