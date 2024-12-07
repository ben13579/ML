import numpy as np


def kernel_kmeans(G, k=2, init_type='random'):
    points = G.shape[0]
    if init_type == "kmeans++":
        centers = kernel_kmeans_plusplus(G, k)
        values = np.zeros((k, points))
        for i in range(k):
            values[i] = np.diagonal(G) - 2 * G[centers[i]] + G[centers[i]] * G[centers[i]]
        labels = np.argmin(values, axis=0)
    else:
        labels = np.random.randint(0, k, size=points)

    epsilon = 0.001
    max_ite = 100
    count = 0
    allLabels = []
    allLabels.append(labels)
    while True:
        clusters_mean = []
        for i in range(k):
            C_j = np.where(labels == i)[0]
            if (len(C_j) == 0):
                clusters_mean.append(np.zeros(points))
            else:
                mean = np.mean(G[C_j, :], axis=0)
                clusters_mean.append(mean)
        clusters_mean = np.array(clusters_mean)
        values = np.zeros_like(clusters_mean)
        for i in range(k):
            values[i] = np.diagonal(G) - 2 * clusters_mean[i] + clusters_mean[i] * clusters_mean[i]
        newlabels = np.argmin(values, axis=0)
        allLabels.append(newlabels)
        count += 1
        if np.sum(labels != newlabels) < epsilon * points or count > max_ite:
            break
        labels = newlabels.copy()
    print(len(allLabels))
    return allLabels


def kernel_kmeans_plusplus(G, k):
    points = G.shape[0]

    # Step 1: Randomly select the first cluster center
    centers = [np.random.choice(points)]

    # Step 2: Iteratively select the next cluster centers
    for _ in range(1, k):
        # Compute squared distances to the closest center
        distances = np.full(points, np.inf)
        for i, center in enumerate(centers):
            current_distances = (
                    G[np.arange(points), np.arange(points)]  # G(i, i)
                    - 2 * G[:, center]  # - 2G(i, center)
                    + G[center, center]  # + G(center, center)
            )
            distances = np.minimum(distances, current_distances)

        # Choose the next center with a probability proportional to distance^2
        probabilities = distances / distances.sum()
        next_center = np.random.choice(points, p=probabilities)
        centers.append(next_center)

    return centers
