import cv2
import rasterio
import numpy as np


def analyze_vegetation(ms_image_path):

    # ---------------------------
    # Load Sentinel-2 image
    # ---------------------------

    with rasterio.open(ms_image_path) as src:
        data = src.read()

    red = data[3].astype(np.float32)
    nir = data[7].astype(np.float32)

    # ---------------------------
    # NDVI Computation
    # ---------------------------

    ndvi = (nir - red) / (nir + red + 1e-8)

    avg_ndvi = float(np.mean(ndvi))
    std_ndvi = float(np.std(ndvi))
    max_ndvi = float(np.max(ndvi))
    min_ndvi = float(np.min(ndvi))

    # ---------------------------
    # NDVI Normalization
    # ---------------------------

    ndvi_uint8 = cv2.normalize(
        ndvi,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # ---------------------------
    # Histogram Equalization
    # ---------------------------

    equalized = cv2.equalizeHist(ndvi_uint8)

    # ---------------------------
    # Gaussian Filtering
    # ---------------------------

    blurred = cv2.GaussianBlur(
        equalized,
        (5, 5),
        0
    )

    # ---------------------------
    # Otsu Segmentation
    # ---------------------------

    _, vegetation_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---------------------------
    # Morphological Operations
    # ---------------------------

    kernel = np.ones((3, 3), np.uint8)

    opened = cv2.morphologyEx(
        vegetation_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ---------------------------
    # Edge Detection
    # ---------------------------

    edges = cv2.Canny(
        closed,
        100,
        200
    )

    # ---------------------------
    # Contour Detection
    # ---------------------------

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    areas = [cv2.contourArea(c) for c in contours]

    # ---------------------------
    # Vegetation Coverage
    # ---------------------------

    vegetation_pixels = np.sum(closed > 0)
    total_pixels = closed.size

    vegetation_coverage = (
        vegetation_pixels / total_pixels
    ) * 100

    # ---------------------------
    # Health Assessment
    # ---------------------------

    if avg_ndvi > 0.6:
        health = "Excellent"

    elif avg_ndvi > 0.4:
        health = "Healthy"

    elif avg_ndvi > 0.2:
        health = "Moderate"

    else:
        health = "Poor"

    # ---------------------------
    # Region Statistics
    # ---------------------------

    largest_area = max(areas) if areas else 0

    avg_area = (
        np.mean(areas)
        if len(areas) > 0
        else 0
    )

    return {

        "avg_ndvi": round(avg_ndvi, 4),
        "std_ndvi": round(std_ndvi, 4),
        "max_ndvi": round(max_ndvi, 4),
        "min_ndvi": round(min_ndvi, 4),

        "health": health,

        "regions": len(contours),

        "largest_area": round(float(largest_area), 2),

        "average_area": round(float(avg_area), 2),

        "vegetation_coverage": round(
            float(vegetation_coverage),
            2
        ),

        "edge_pixels": int(
            np.sum(edges > 0)
        )
    }