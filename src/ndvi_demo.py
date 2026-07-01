import rasterio
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Input File Path
# =============================================================================

# Path to the Sentinel-2 multispectral image
file_path = "/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"


# =============================================================================
# Read the Multispectral Image
# =============================================================================

# Open the .tif file and read all spectral bands
with rasterio.open(file_path) as src:
    data = src.read()


# =============================================================================
# Select Required Bands for NDVI
# =============================================================================

# Sentinel-2 band indices in EuroSAT
# B04 (Red) = band 4 -> index 3
# B08 (Near Infrared - NIR) = band 8 -> index 7

# Extract the Red band
red = data[3].astype(float)

# Extract the NIR band
nir = data[7].astype(float)


# =============================================================================
# Compute NDVI
# =============================================================================

# Calculate NDVI using the standard formula:
# NDVI = (NIR - Red) / (NIR + Red)
# A small value (1e-8) is added to avoid division by zero.
ndvi = (nir - red) / (nir + red + 1e-8)


# =============================================================================
# Display the NDVI Map
# =============================================================================

# Create a figure for visualization
plt.figure(figsize=(6,6))

# Display the NDVI map using the Red-Yellow-Green color map
plt.imshow(ndvi, cmap="RdYlGn")

# Add a color bar to indicate NDVI values
plt.colorbar(label="NDVI")

# Add a title to the figure
plt.title("NDVI Map")

# Hide the axis for a cleaner visualization
plt.axis("off")


# =============================================================================
# Save and Display the Result
# =============================================================================

# Save the NDVI map as an image
plt.savefig("ndvi_map.png")

# Display the plot
plt.show()