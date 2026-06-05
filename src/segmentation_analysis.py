import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

file_path = "/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"

# --------------------------------------------------
# Load Sentinel-2 image
# --------------------------------------------------

with rasterio.open(file_path) as src:
    data = src.read()

red = data[3].astype(float)
nir = data[7].astype(float)

# --------------------------------------------------
# NDVI
# --------------------------------------------------

ndvi = (nir - red) / (nir + red + 1e-8)

plt.imshow(ndvi, cmap="RdYlGn")
plt.axis("off")
plt.title("NDVI")
plt.savefig("results/ndvi_map.png")
plt.close()

# --------------------------------------------------
# Gaussian Smoothing
# --------------------------------------------------

ndvi_uint8 = ((ndvi + 1) * 127.5).astype(np.uint8)

blurred = cv2.GaussianBlur(
    ndvi_uint8,
    (5, 5),
    0
)

# --------------------------------------------------
# Threshold Segmentation
# --------------------------------------------------

_, vegetation_mask = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

plt.imshow(vegetation_mask, cmap="gray")
plt.axis("off")
plt.title("Vegetation Mask")
plt.savefig("results/vegetation_mask.png")
plt.close()

# --------------------------------------------------
# Morphological Opening
# --------------------------------------------------

kernel = np.ones((3, 3), np.uint8)

opened = cv2.morphologyEx(
    vegetation_mask,
    cv2.MORPH_OPEN,
    kernel
)

plt.imshow(opened, cmap="gray")
plt.axis("off")
plt.title("Opening Result")
plt.savefig("results/opening_result.png")
plt.close()

# --------------------------------------------------
# Morphological Closing
# --------------------------------------------------

closed = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel
)

plt.imshow(closed, cmap="gray")
plt.axis("off")
plt.title("Closing Result")
plt.savefig("results/closing_result.png")
plt.close()

# --------------------------------------------------
# Contour Detection
# --------------------------------------------------

contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contour_img = cv2.cvtColor(
    closed,
    cv2.COLOR_GRAY2BGR
)

cv2.drawContours(
    contour_img,
    contours,
    -1,
    (0, 255, 0),
    2
)

plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Vegetation Regions")
plt.savefig("results/contours.png")
plt.close()

# --------------------------------------------------
# Region Analysis
# --------------------------------------------------

areas = [cv2.contourArea(c) for c in contours]

print("Number of Vegetation Regions:", len(contours))

if len(areas) > 0:
    print("Largest Region Area:", max(areas))
    print("Average Region Area:", np.mean(areas))

with open("results/vegetation_regions.txt", "w") as f:
    f.write(f"Number of Regions: {len(contours)}\n")

    if len(areas) > 0:
        f.write(f"Largest Region Area: {max(areas)}\n")
        f.write(f"Average Region Area: {np.mean(areas)}\n")

print("Analysis Complete")