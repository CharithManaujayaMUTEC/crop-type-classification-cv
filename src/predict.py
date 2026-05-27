import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("models/crop_model.keras")

labels = [
    "Pepper_healthy", "Potato_healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Tomato_Early_blight"
]

img = cv2.imread("test.jpg")
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
class_id = np.argmax(pred)

print("Predicted class:", labels[class_id])