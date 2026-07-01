import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

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

print("Generating classification report...")

report = classification_report(
    y_true,
    y_pred,
    target_names=labels
)

print(report)

with open("results/classification_report.txt", "w") as f:
    f.write("EuroSAT Classification Report\n")
    f.write("=" * 40 + "\n\n")
    f.write(report)

print("Saved: results/classification_report.txt")

plt.figure(figsize=(14, 12))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    linewidths=0.5,
    linecolor="gray",
    cbar=True,
    square=True,
    annot_kws={"size": 10}
)

plt.title("EuroSAT Confusion Matrix", fontsize=16, pad=20)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=300)
plt.show()

print("Saved: results/confusion_matrix.png")