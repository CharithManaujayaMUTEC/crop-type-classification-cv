import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "models/eurosat_model.keras"
LABELS_PATH = "models/eurosat_labels.npy"

model = load_model(MODEL_PATH)
labels = np.load(LABELS_PATH, allow_pickle=True)

IMAGE_PATH = "/content/drive/MyDrive/dataset/EuroSAT_RGB/EuroSAT_RGB/Forest/Forest_1.jpg"

img = cv2.imread(IMAGE_PATH)
img = cv2.resize(img, (64, 64))
img = img.astype("float32") / 255.0

img = np.expand_dims(img, axis=0)

pred = model.predict(img, verbose=0)[0]

class_idx = np.argmax(pred)

print("Prediction:", labels[class_idx])
print("Confidence:", round(pred[class_idx] * 100, 2), "%")