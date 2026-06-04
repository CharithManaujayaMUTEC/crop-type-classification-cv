import cv2
import rasterio
import numpy as np

def analyze_vegetation(ms_image_path):

    with rasterio.open(ms_image_path) as src:
        data = src.read()

    red = data[3].astype(float)
    nir = data[7].astype(float)

    ndvi = (nir - red) / (nir + red + 1e-8)

    ndvi_uint8 = ((ndvi + 1) * 127.5).astype(np.uint8)

    blurred = cv2.GaussianBlur(ndvi_uint8, (5,5), 0)

    _, mask = cv2.threshold(
        blurred,
        150,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones((3,3), np.uint8)

    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    closed = cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    areas = [cv2.contourArea(c) for c in contours]

    avg_ndvi = float(np.mean(ndvi))

    if avg_ndvi > 0.5:
        health = "Healthy"
    elif avg_ndvi > 0.2:
        health = "Moderate"
    else:
        health = "Poor"

    return {
        "avg_ndvi": avg_ndvi,
        "health": health,
        "regions": len(contours),
        "largest_area": max(areas) if areas else 0
    }