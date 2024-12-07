from kernel_kmeans import kernel_kmeans
from spectral_cluster import ration_cut, plot_eigenspace,normalized_cut
from utils import *


def main():
    img1 = cv2.imread("image1.png")
    img2 = cv2.imread("image2.png")
    img1 = img1.reshape((10000, -1))
    img2 = img2.reshape((10000, -1))
    clusters = 2
    init_type = "random"
    init_type = "kmeans++"
    gammaS = 0.001
    gammaC = 0.001
    G1 = computeGramMatrix(img1, gammaS, gammaC)
    G2 = computeGramMatrix(img2, gammaS, gammaC)
    img_num = 1
    G = G1 if img_num == 1 else G2
    training_mode = "kernel"
    if training_mode == "kernel":
        labels = kernel_kmeans(G, k=clusters, init_type=init_type)
        visualization(labels, clusters)
        create_gif_from_frames(gif_name=f"newkernel kmeans-{init_type}-k={clusters}.gif", img_num=img_num)
    elif training_mode == "ratio":
        labels, H = ration_cut(G, k=clusters, init_type=init_type)
        visualization(labels, clusters)
        plot_eigenspace(H, labels[len(labels) - 1], save_path=f"ratio cut-{init_type}-k={clusters}", img_num=img_num)
        create_gif_from_frames(gif_name=f"ratio cut-{init_type}-k={clusters}.gif", img_num=img_num)
    elif training_mode == "normalized":
        labels, H = normalized_cut(G, k=clusters, init_type=init_type)
        visualization(labels, clusters)
        plot_eigenspace(H, labels[len(labels) - 1], save_path=f"normalized cut-{init_type}-k={clusters}",
                        img_num=img_num)
        create_gif_from_frames(gif_name=f"normalized cut-{init_type}-k={clusters}.gif", img_num=img_num)



if __name__ == "__main__":
    main()
