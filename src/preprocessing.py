import os
import cv2
import numpy as np

# Image size expected by the CNN model
IMG_SIZE = 64


# Function to load all images from the dataset
def load_dataset(path):
    # Lists to store images and their corresponding labels
    X, y = [], []

    # Get all class folder names (e.g., Forest, River, Residential, etc.)
    labels = os.listdir(path)

    # Loop through each class folder
    for label_idx, label in enumerate(labels):
        # Create the full path to the current class folder
        folder = os.path.join(path, label)

        # Loop through every image inside the current folder
        for img_name in os.listdir(folder):
            # Create the full image path
            img_path = os.path.join(folder, img_name)

            # Read the image using OpenCV
            img = cv2.imread(img_path)

            # Skip the file if the image cannot be loaded
            if img is None:
                continue

            # Resize the image to 64 × 64 pixels
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalize pixel values from [0,255] to [0,1]
            img = img / 255.0

            # Store the processed image
            X.append(img)

            # Store the numerical label corresponding to the image class
            y.append(label_idx)

    # Convert lists into NumPy arrays and return them
    return np.array(X), np.array(y), labels