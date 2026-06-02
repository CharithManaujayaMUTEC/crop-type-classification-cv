import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

from preprocessing import load_dataset

DATASET_PATH = "/content/drive/MyDrive/dataset/EuroSAT_RGB/EuroSAT_RGB"

print("Loading dataset...")
X, y, labels = load_dataset(DATASET_PATH)

y_cat = to_categorical(y, num_classes=len(labels))

print("Loading model...")
model = load_model("models/eurosat_model.keras")

print("Predicting...")
y_pred = model.predict(X, verbose=1)

y_true = np.argmax(y_cat, axis=1)
y_pred = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("EuroSAT Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.show()

print("Saved: results/confusion_matrix.png")