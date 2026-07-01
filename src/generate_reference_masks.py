import os
import cv2
import rasterio
import numpy as np

# -----------------------------
# Folder Paths
# -----------------------------

IMAGE_DIR = "evaluation/images"
OUTPUT_DIR = "evaluation/masks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Baseline Segmentation
# -----------------------------

def generate_reference_mask(image_path):

    with rasterio.open(image_path) as src:
        data = src.read()

    # Sentinel-2
    red = data[3].astype(np.float32)
    nir = data[7].astype(np.float32)

    # NDVI
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Normalize
    ndvi = cv2.normalize(
        ndvi,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # -------- Baseline Pipeline --------

    # Median filter instead of Gaussian
    median = cv2.medianBlur(ndvi, 5)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(median)

    # Adaptive Threshold
    adaptive = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        25,
        3
    )

    kernel = np.ones((3, 3), np.uint8)

    reference = cv2.morphologyEx(
        adaptive,
        cv2.MORPH_OPEN,
        kernel
    )

    reference = cv2.morphologyEx(
        reference,
        cv2.MORPH_CLOSE,
        kernel
    )

    return reference


# -----------------------------
# Generate masks
# -----------------------------

for filename in os.listdir(IMAGE_DIR):

    if not filename.endswith(".tif"):
        continue

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    mask = generate_reference_mask(image_path)

    save_name = filename.replace(".tif", ".png")

    save_path = os.path.join(
        OUTPUT_DIR,
        save_name
    )

    cv2.imwrite(save_path, mask)

    print(f"Saved: {save_name}")

print("\nReference masks generated successfully.")