import numpy as np
from scipy.spatial.distance import cdist
from matplotlib import colormaps
import imageio.v2 as imageio
import os
import cv2


def computeGramMatrix(img, gammaS, gammaC):
    # Compute spatial coordinates for each pixel
    x_coords, y_coords = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")

    spatial_coords = np.stack([x_coords.ravel(), y_coords.ravel()], axis=-1)

    # Compute pairwise squared distances
    spatial_dist_sq = cdist(spatial_coords, spatial_coords, metric='sqeuclidean')
    color_dist_sq = cdist(img, img, metric='sqeuclidean')

    # Apply kernels
    G = np.exp(-gammaS * spatial_dist_sq) * np.exp(-gammaC * color_dist_sq)
    return G


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def visualization(labels, k, save_path="frames"):
    clear_folder(save_path)
    os.makedirs(save_path, exist_ok=True)
    labels = np.array(labels)
    ite, points = labels.shape
    img_size = int(np.sqrt(points))
    colors = colormaps["Set1"].resampled(k)
    for i in range(ite):
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)

        # Assign colors based on the cluster label
        for j in range(points):
            cluster = labels[i, j]
            color = (np.array(colors(cluster))[:3] * 255).astype(np.uint8)  # Convert to RGB
            x, y = divmod(j, img_size)
            img[x, y] = color

        # Save the frame
        frame_path = f"{save_path}/frame_{i:02d}.png"
        cv2.imwrite(frame_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def create_gif_from_frames(save_path="frames", gif_name="clustering.gif", duration=0.5, img_num=1):
    """
    Create a GIF from saved frames.
    :param save_path: Directory where frames are saved.
    :param gif_name: Name of the output GIF.
    :param duration: Duration of each frame in seconds.
    """
    import os
    images = []
    frame_files = sorted([f for f in os.listdir(save_path) if f.endswith(".png")])
    for frame_file in frame_files:
        frame_path = os.path.join(save_path, frame_file)
        images.append(imageio.imread(frame_path))
    imageio.mimsave(f"image{img_num}/" + gif_name, images, duration=duration)
    print(f"GIF saved as {gif_name}")
