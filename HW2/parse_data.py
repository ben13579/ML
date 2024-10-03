import numpy as np
import struct

def parse_idx(filename):
    """解析 IDX 檔案格式。"""
    with open(filename, 'rb') as f:
        # 讀取 Magic Number 和數據維度
        magic, num_items = struct.unpack(">II", f.read(8))  # ">II" big endian
        if magic == 2049:  # 標籤檔 (IDX1 格式)
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
        elif magic == 2051:  # 圖片檔 (IDX3 格式)
            num_rows, num_cols = struct.unpack(">II", f.read(8))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, num_rows, num_cols)
            return images
        else:
            raise ValueError("未知的 IDX 檔案格式")

def load_mnist(train_images_path, train_labels_path, test_images_path, test_labels_path):
    """讀取 MNIST 資料集。"""
    train_images = parse_idx(train_images_path)
    train_labels = parse_idx(train_labels_path)
    test_images = parse_idx(test_images_path)
    test_labels = parse_idx(test_labels_path)
    return (train_images, train_labels), (test_images, test_labels)

# 使用範例：

