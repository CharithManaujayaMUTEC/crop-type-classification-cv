import os
import cv2
import numpy as np

IMG_SIZE = 64

def load_dataset(path):
    X, y = [], []
    labels = os.listdir(path)

    for label_idx, label in enumerate(labels):
        folder = os.path.join(path, label)

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0

            X.append(img)
            y.append(label_idx)

    return np.array(X), np.array(y), labels