# src/ndvi.py

import os
import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Save Image Utility Function
# =============================================================================

def save_image(img, title, filename, cmap=None):
    """
    Display and save an image with a title.

    Parameters:
        img      : Image to save.
        title    : Title displayed on the image.
        filename : Output file path.
        cmap     : Color map (optional).
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


# =============================================================================
# Vegetation Analysis Pipeline
# =============================================================================

def analyze_vegetation(ms_image_path):

    # Create the results folder if it does not already exist
    os.makedirs("results", exist_ok=True)

    # ==========================================
    # Load Sentinel-2 Image
    # ==========================================

    # Read all spectral bands from the multispectral image
    with rasterio.open(ms_image_path) as src:
        data = src.read()

    # Extract the Red (B04) and Near Infrared (B08) bands
    red = data[3].astype(np.float32)   # B04
    nir = data[7].astype(np.float32)   # B08

    # ==========================================
    # NDVI Calculation
    # ==========================================

    # Compute NDVI using the standard formula
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Save the NDVI image
    save_image(
        ndvi,
        "NDVI",
        "results/01_ndvi.png",
        cmap="RdYlGn"
    )

    # Calculate basic NDVI statistics
    avg_ndvi = float(np.mean(ndvi))
    std_ndvi = float(np.std(ndvi))
    max_ndvi = float(np.max(ndvi))
    min_ndvi = float(np.min(ndvi))

    # ==========================================
    # Normalize NDVI
    # ==========================================

    # Normalize NDVI values to the range 0-255
    ndvi_uint8 = cv2.normalize(
        ndvi,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype(np.uint8)

    # Save normalized image
    save_image(
        ndvi_uint8,
        "Normalized NDVI",
        "results/02_normalized_ndvi.png",
        cmap="gray"
    )

    # ==========================================
    # Histogram Equalization
    # ==========================================

    # Improve image contrast
    equalized = cv2.equalizeHist(ndvi_uint8)

    save_image(
        equalized,
        "Histogram Equalization",
        "results/03_histogram_equalized.png",
        cmap="gray"
    )

    # ==========================================
    # Gaussian Blur
    # ==========================================

    # Reduce image noise before segmentation
    blurred = cv2.GaussianBlur(
        equalized,
        (3, 3),
        0
    )

    save_image(
        blurred,
        "Gaussian Blur",
        "results/04_gaussian_blur.png",
        cmap="gray"
    )

    # ==========================================
    # Otsu Thresholding
    # ==========================================

    # Automatically determine an optimal threshold value
    otsu_threshold, otsu_mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    save_image(
        otsu_mask,
        "Otsu Segmentation",
        "results/05_otsu_segmentation.png",
        cmap="gray"
    )

    # ==========================================
    # Adaptive Thresholding
    # ==========================================

    # Perform adaptive image segmentation
    adaptive_mask = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    save_image(
        adaptive_mask,
        "Adaptive Thresholding",
        "results/06_adaptive_segmentation.png",
        cmap="gray"
    )

    # ==========================================
    # Combine Segmentation Masks
    # ==========================================

    # Merge both thresholding results
    combined_mask = cv2.bitwise_or(
        otsu_mask,
        adaptive_mask
    )

    save_image(
        combined_mask,
        "Combined Mask",
        "results/07_combined_mask.png",
        cmap="gray"
    )

    # ==========================================
    # Morphological Opening
    # ==========================================

    # Create a small elliptical kernel
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2, 2)
    )

    # Remove small isolated noise
    opened = cv2.morphologyEx(
        combined_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    save_image(
        opened,
        "Morphological Opening",
        "results/08_opening.png",
        cmap="gray"
    )

    # ==========================================
    # Morphological Closing
    # ==========================================

    # Fill small gaps inside vegetation regions
    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    save_image(
        closed,
        "Morphological Closing",
        "results/09_closing.png",
        cmap="gray"
    )

    # ==========================================
    # Edge Detection
    # ==========================================

    # Detect edges using the Canny algorithm
    edges = cv2.Canny(
        closed,
        100,
        200
    )

    save_image(
        edges,
        "Canny Edges",
        "results/10_edges.png",
        cmap="gray"
    )

    # Count the number of detected edge pixels
    edge_pixels = int(np.sum(edges > 0))

    # ==========================================
    # Contour Detection
    # ==========================================

    # Detect vegetation contours
    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Remove very small contours that are likely noise
    filtered_contours = []

    for c in contours:
        if cv2.contourArea(c) > 10:
            filtered_contours.append(c)

    contours = filtered_contours

    # Convert grayscale image to color for contour visualization
    contour_img = cv2.cvtColor(
        closed,
        cv2.COLOR_GRAY2BGR
    )

    # Draw contours in green
    cv2.drawContours(
        contour_img,
        contours,
        -1,
        (0, 255, 0),
        2
    )

    # Save contour visualization
    save_image(
        cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
        "Contours",
        "results/11_contours.png"
    )

    # Calculate the area of each contour
    contour_areas = [
        cv2.contourArea(c)
        for c in contours
    ]

    # ==========================================
    # Vegetation Health Map
    # ==========================================

    # Create an empty health map
    health_map = np.zeros_like(ndvi)

    # Assign vegetation health levels based on NDVI values
    health_map[ndvi < 0.3] = 1
    health_map[(ndvi >= 0.3) & (ndvi < 0.6)] = 2
    health_map[ndvi >= 0.6] = 3

    # Save the vegetation health map
    save_image(
        health_map,
        "Vegetation Health Map",
        "results/12_health_map.png",
        cmap="RdYlGn"
    )

    # ==========================================
    # Vegetation Statistics
    # ==========================================

    # Count vegetation pixels
    vegetation_pixels = np.sum(closed > 0)

    # Total number of image pixels
    total_pixels = closed.size

    # Calculate vegetation coverage percentage
    vegetation_coverage = (
        vegetation_pixels / total_pixels
    ) * 100

    # Count dense vegetation pixels
    dense_pixels = np.sum(ndvi > 0.6)

    # Calculate vegetation density percentage
    vegetation_density = (
        dense_pixels / total_pixels
    ) * 100

    # Count stressed vegetation pixels
    stress_pixels = np.sum(ndvi < 0.3)

    # Calculate stress percentage
    stress_percentage = (
        stress_pixels / total_pixels
    ) * 100

    # Largest detected vegetation region
    largest_area = (
        max(contour_areas)
        if contour_areas else 0
    )

    # Average region area
    average_area = (
        np.mean(contour_areas)
        if contour_areas else 0
    )

    # Total area of all vegetation regions
    total_region_area = (
        np.sum(contour_areas)
        if contour_areas else 0
    )

    # Calculate perimeter of each contour
    perimeters = [
        cv2.arcLength(c, True)
        for c in contours
    ]

    # Average contour perimeter
    average_perimeter = (
        np.mean(perimeters)
        if perimeters else 0
    )

    # ==========================================
    # Overall Vegetation Health Classification
    # ==========================================

    if avg_ndvi >= 0.7:
        health = "Dense Healthy Vegetation"
    elif avg_ndvi >= 0.5:
        health = "Healthy Vegetation"
    elif avg_ndvi >= 0.3:
        health = "Moderately Stressed Vegetation"
    else:
        health = "Highly Stressed Vegetation"

    # ==========================================
    # Return Analysis Results
    # ==========================================

    return {

        # NDVI statistics
        "avg_ndvi": round(avg_ndvi, 4),
        "std_ndvi": round(std_ndvi, 4),
        "max_ndvi": round(max_ndvi, 4),
        "min_ndvi": round(min_ndvi, 4),

        # Overall vegetation health
        "health": health,

        # Otsu threshold value
        "otsu_threshold": round(
            float(otsu_threshold),
            2
        ),

        # Number of vegetation regions
        "regions": len(contours),

        # Region measurements
        "largest_area": round(
            float(largest_area),
            2
        ),

        "average_area": round(
            float(average_area),
            2
        ),

        "total_region_area": round(
            float(total_region_area),
            2
        ),

        "average_perimeter": round(
            float(average_perimeter),
            2
        ),

        # Vegetation coverage statistics
        "vegetation_coverage": round(
            float(vegetation_coverage),
            2
        ),

        "vegetation_density": round(
            float(vegetation_density),
            2
        ),

        "stress_percentage": round(
            float(stress_percentage),
            2
        ),

        # Total detected edge pixels
        "edge_pixels": edge_pixels
    }