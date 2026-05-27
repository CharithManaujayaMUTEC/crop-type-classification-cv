import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("test.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Fake NDVI (since no NIR band)
red = img[:,:,0].astype(float)
nir = img[:,:,1].astype(float)

ndvi = (nir - red) / (nir + red + 1e-5)

plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar()
plt.title("NDVI Map (Simulated)")
plt.show()