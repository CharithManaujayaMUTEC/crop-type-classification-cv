import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from preprocessing import load_dataset

DATASET_PATH = "/content/drive/MyDrive/dataset/PlantVillage"

print("Loading dataset...")
X, y, labels = load_dataset(DATASET_PATH)

y_true = np.array(y)

print("Loading model...")
model = load_model("models/crop_model.keras")

print("Predicting...")
y_pred = model.predict(X)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(xticks_rotation=45)

plt.title("Confusion Matrix")
plt.show()