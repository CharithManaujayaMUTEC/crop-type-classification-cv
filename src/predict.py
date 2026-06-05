# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ndvi import analyze_vegetation

# ==========================================
# Load Model and Labels
# ==========================================

MODEL_PATH = "models/eurosat_model.keras"
LABELS_PATH = "models/eurosat_labels.npy"

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

# ==========================================
# RGB Image Classification
# ==========================================

IMAGE_PATH = "/content/drive/MyDrive/dataset/EuroSAT_RGB/EuroSAT_RGB/Forest/Forest_1.jpg"

img = cv2.imread(IMAGE_PATH)

if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img = cv2.resize(img, (64, 64))
img = img.astype("float32") / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img, verbose=0)[0]

class_idx = np.argmax(pred)
predicted_class = labels[class_idx]
confidence = pred[class_idx] * 100

# ==========================================
# Vegetation Analysis
# ==========================================

MS_IMAGE_PATH = "/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/Forest/Forest_1.tif"

veg = analyze_vegetation(MS_IMAGE_PATH)

# ==========================================
# Final Report
# ==========================================

print("\n" + "=" * 60)
print("SATELLITE IMAGE ANALYSIS REPORT")
print("=" * 60)

print("\nCLASSIFICATION RESULTS")
print("-" * 40)
print(f"Predicted Class      : {predicted_class}")
print(f"Confidence           : {confidence:.2f}%")

print("\nNDVI STATISTICS")
print("-" * 40)
print(f"Average NDVI         : {veg['avg_ndvi']}")
print(f"NDVI Std Dev         : {veg['std_ndvi']}")
print(f"Maximum NDVI         : {veg['max_ndvi']}")
print(f"Minimum NDVI         : {veg['min_ndvi']}")

print("\nVEGETATION HEALTH")
print("-" * 40)
print(f"Health Status        : {veg['health']}")

print("\nSEGMENTATION RESULTS")
print("-" * 40)
print(f"Otsu Threshold       : {veg['otsu_threshold']}")
print(f"Vegetation Regions   : {veg['regions']}")

print("\nREGION ANALYSIS")
print("-" * 40)
print(f"Largest Region Area  : {veg['largest_area']}")
print(f"Average Region Area  : {veg['average_area']}")
print(f"Total Region Area    : {veg['total_region_area']}")
print(f"Average Perimeter    : {veg['average_perimeter']}")

print("\nCOVERAGE ANALYSIS")
print("-" * 40)
print(f"Vegetation Coverage  : {veg['vegetation_coverage']}%")
print(f"Vegetation Density   : {veg['vegetation_density']}%")
print(f"Stress Percentage    : {veg['stress_percentage']}%")

print("\nEDGE ANALYSIS")
print("-" * 40)
print(f"Edge Pixels          : {veg['edge_pixels']}")

print("\nGENERATED PIPELINE IMAGES")
print("-" * 40)

pipeline_images = [
    "01_ndvi.png",
    "02_normalized_ndvi.png",
    "03_histogram_equalized.png",
    "04_gaussian_blur.png",
    "05_otsu_segmentation.png",
    "06_adaptive_segmentation.png",
    "07_combined_mask.png",
    "08_opening.png",
    "09_closing.png",
    "10_edges.png",
    "11_contours.png",
    "12_health_map.png"
]

for image in pipeline_images:
    print(f"✓ {image}")

print("\nAnalysis Complete.")
print("=" * 60)