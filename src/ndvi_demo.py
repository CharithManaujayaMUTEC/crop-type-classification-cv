import rasterio
import numpy as np
import matplotlib.pyplot as plt

file_path = "/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif"

with rasterio.open(file_path) as src:
    data = src.read()

# Sentinel-2 band indices in EuroSAT
# B04 (Red) = band 4 -> index 3
# B08 (NIR) = band 8 -> index 7

red = data[3].astype(float)
nir = data[7].astype(float)

ndvi = (nir - red) / (nir + red + 1e-8)

plt.figure(figsize=(6,6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI")
plt.title("NDVI Map")
plt.axis("off")

plt.savefig("ndvi_map.png")
plt.show()