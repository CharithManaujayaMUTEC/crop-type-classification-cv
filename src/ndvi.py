import numpy as np

def compute_ndvi(nir, red):
    return (nir - red) / (nir + red + 1e-5)