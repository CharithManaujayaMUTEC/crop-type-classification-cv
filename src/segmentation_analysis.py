import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

# Create the results folder if it does not already exist
os.makedirs("results", exist_ok=True)

# Path to the multispectral Sentinel-2 image
file_path = "/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"


# --------------------------------------------------
# Load Sentinel-2 Image
# --------------------------------------------------

# Open the multispectral image and read all bands
with rasterio.open(file_path) as src:
    data = src.read()

# Extract the Red and Near Infrared (NIR) bands
red = data[3].astype(float)
nir = data[7].astype(float)


# --------------------------------------------------
# NDVI Calculation
# --------------------------------------------------

# Compute the NDVI using the standard formula
ndvi = (nir - red) / (nir + red + 1e-8)

# Display and save the NDVI map
plt.imshow(ndvi, cmap="RdYlGn")
plt.axis("off")
plt.title("NDVI")
plt.savefig("results/ndvi_map.png")
plt.close()


# --------------------------------------------------
# Gaussian Smoothing
# --------------------------------------------------

# Convert NDVI values from [-1,1] to [0,255] for OpenCV processing
ndvi_uint8 = ((ndvi + 1) * 127.5).astype(np.uint8)

# Apply Gaussian Blur to reduce image noise
blurred = cv2.GaussianBlur(
    ndvi_uint8,
    (5, 5),
    0
)


# --------------------------------------------------
# Threshold Segmentation
# --------------------------------------------------

# Separate vegetation from the background using binary thresholding
_, vegetation_mask = cv2.threshold(
    blurred,
    127,
    255,
    cv2.THRESH_BINARY
)

# Display and save the vegetation mask
plt.imshow(vegetation_mask, cmap="gray")
plt.axis("off")
plt.title("Vegetation Mask")
plt.savefig("results/vegetation_mask.png")
plt.close()


# --------------------------------------------------
# Morphological Opening
# --------------------------------------------------

# Create a 3×3 kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

# Remove small noise using the opening operation
opened = cv2.morphologyEx(
    vegetation_mask,
    cv2.MORPH_OPEN,
    kernel
)

# Display and save the opening result
plt.imshow(opened, cmap="gray")
plt.axis("off")
plt.title("Opening Result")
plt.savefig("results/opening_result.png")
plt.close()


# --------------------------------------------------
# Morphological Closing
# --------------------------------------------------

# Fill small holes inside vegetation regions
closed = cv2.morphologyEx(
    opened,
    cv2.MORPH_CLOSE,
    kernel
)

# Display and save the closing result
plt.imshow(closed, cmap="gray")
plt.axis("off")
plt.title("Closing Result")
plt.savefig("results/closing_result.png")
plt.close()


# --------------------------------------------------
# Contour Detection
# --------------------------------------------------

# Detect the boundaries of vegetation regions
contours, _ = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Convert the binary image into a color image for visualization
contour_img = cv2.cvtColor(
    closed,
    cv2.COLOR_GRAY2BGR
)

# Draw the detected contours in green
cv2.drawContours(
    contour_img,
    contours,
    -1,
    (0, 255, 0),
    2
)

# Display and save the contour image
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Vegetation Regions")
plt.savefig("results/contours.png")
plt.close()


# --------------------------------------------------
# Region Analysis
# --------------------------------------------------

# Calculate the area of each detected vegetation region
areas = [cv2.contourArea(c) for c in contours]

# Print the number of detected vegetation regions
print("Number of Vegetation Regions:", len(contours))

# Print the largest and average region area if any regions exist
if len(areas) > 0:
    print("Largest Region Area:", max(areas))
    print("Average Region Area:", np.mean(areas))

# Save the region statistics to a text file
with open("results/vegetation_regions.txt", "w") as f:
    f.write(f"Number of Regions: {len(contours)}\n")

    if len(areas) > 0:
        f.write(f"Largest Region Area: {max(areas)}\n")
        f.write(f"Average Region Area: {np.mean(areas)}\n")

# Indicate that the analysis has finished
print("Analysis Complete")