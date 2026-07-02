# predict.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ndvi import analyze_vegetation


# =============================================================================
# Load the Trained CNN Model and Class Labels
# =============================================================================

# Path to the trained CNN model
MODEL_PATH = "models/eurosat_model.keras"

# Path to the saved class labels
LABELS_PATH = "models/eurosat_labels.npy"

# Load the trained model
model = load_model(MODEL_PATH)

# Load the class names
labels = np.load(LABELS_PATH, allow_pickle=True)


# =============================================================================
# RGB Image Classification
# =============================================================================

# Path to the RGB satellite image
IMAGE_PATH = "/content/drive/MyDrive/dataset/EuroSAT_RGB/EuroSAT_RGB/Forest/Forest_1.jpg"

# Read the RGB image
img = cv2.imread(IMAGE_PATH)

# Check whether the image was loaded successfully
if img is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Resize the image to the model's required input size
img = cv2.resize(img, (64, 64))

# Normalize pixel values to the range [0,1]
img = img.astype("float32") / 255.0

# Add an extra dimension to create a batch of one image
img = np.expand_dims(img, axis=0)

# Predict the land-cover class
pred = model.predict(img, verbose=0)[0]

# Find the class with the highest probability
class_idx = np.argmax(pred)

# Get the predicted class name
predicted_class = labels[class_idx]

# Convert prediction confidence into percentage
confidence = pred[class_idx] * 100


# =============================================================================
# Vegetation Analysis
# =============================================================================

# Path to the corresponding multispectral Sentinel-2 image
MS_IMAGE_PATH = "/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/Forest/Forest_1.tif"

# Perform NDVI and vegetation health analysis
veg = analyze_vegetation(MS_IMAGE_PATH)


# =============================================================================
# Display Final Analysis Report
# =============================================================================

print("\n" + "=" * 60)
print("SATELLITE IMAGE ANALYSIS REPORT")
print("=" * 60)

# -------------------------------------------------------------------------
# Image Classification Results
# -------------------------------------------------------------------------

print("\nCLASSIFICATION RESULTS")
print("-" * 40)
print(f"Predicted Class      : {predicted_class}")
print(f"Confidence           : {confidence:.2f}%")

# -------------------------------------------------------------------------
# NDVI Statistics
# -------------------------------------------------------------------------

print("\nNDVI STATISTICS")
print("-" * 40)
print(f"Average NDVI         : {veg['avg_ndvi']}")
print(f"NDVI Std Dev         : {veg['std_ndvi']}")
print(f"Maximum NDVI         : {veg['max_ndvi']}")
print(f"Minimum NDVI         : {veg['min_ndvi']}")

# -------------------------------------------------------------------------
# Vegetation Health Status
# -------------------------------------------------------------------------

print("\nVEGETATION HEALTH")
print("-" * 40)
print(f"Health Status        : {veg['health']}")

# -------------------------------------------------------------------------
# Segmentation Results
# -------------------------------------------------------------------------

print("\nSEGMENTATION RESULTS")
print("-" * 40)
print(f"Otsu Threshold       : {veg['otsu_threshold']}")
print(f"Vegetation Regions   : {veg['regions']}")

# -------------------------------------------------------------------------
# Region Analysis
# -------------------------------------------------------------------------

print("\nREGION ANALYSIS")
print("-" * 40)
print(f"Largest Region Area  : {veg['largest_area']}")
print(f"Average Region Area  : {veg['average_area']}")
print(f"Total Region Area    : {veg['total_region_area']}")
print(f"Average Perimeter    : {veg['average_perimeter']}")

# -------------------------------------------------------------------------
# Vegetation Coverage Statistics
# -------------------------------------------------------------------------

print("\nCOVERAGE ANALYSIS")
print("-" * 40)
print(f"Vegetation Coverage  : {veg['vegetation_coverage']}%")
print(f"Vegetation Density   : {veg['vegetation_density']}%")
print(f"Stress Percentage    : {veg['stress_percentage']}%")

# -------------------------------------------------------------------------
# Edge Detection Statistics
# -------------------------------------------------------------------------

print("\nEDGE ANALYSIS")
print("-" * 40)
print(f"Edge Pixels          : {veg['edge_pixels']}")

# -------------------------------------------------------------------------
# Generated Processing Images
# -------------------------------------------------------------------------

print("\nGENERATED PIPELINE IMAGES")
print("-" * 40)

# List of all generated processing results
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

# Display the generated image filenames
for image in pipeline_images:
    print(f"✓ {image}")

# -------------------------------------------------------------------------
# End of Analysis
# -------------------------------------------------------------------------

print("\nAnalysis Complete.")
print("=" * 60)