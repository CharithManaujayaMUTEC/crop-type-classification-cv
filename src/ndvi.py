import numpy as np

def compute_ndvi(img):
    red = img[:, :, 2]
    green = img[:, :, 1]
    return (green - red) / (green + red + 1e-5)