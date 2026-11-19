import cv2 as cv
import torch
import numpy as np
from os import listdir, path
from torch.utils.data import Dataset


class getImageLabel(Dataset):
    def __init__(self, folder='./Dataset'):
        self.dataset = []
        # Mengambil daftar folder kelas spesies ikan secara terurut[cite: 9]
        self.classes = sorted([d for d in listdir(folder) if path.isdir(path.join(folder, d))])
        num_classes = len(self.classes)
        
        # Inisiasi one-hot encoding sesuai jumlah kelas[cite: 9]
        to_one_hot = np.eye(num_classes)
        
        for i, fish_class in enumerate(self.classes):
            class_folder = path.join(folder, fish_class)
            for image_name in listdir(class_folder):
                img_path = path.join(class_folder, image_name)
                img = cv.imread(img_path)
                if img is not None:
                    # Read image, resize ke (100, 100), dan normalisasi ke rentang 0-1[cite: 9]
                    image = cv.resize(img, dsize=(100, 100)) / 255.0
                    # Append fitur gambar dan label one-hot ke self.dataset[cite: 9]
                    self.dataset.append([image, to_one_hot[i]])

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.dataset)