import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Import TensorFlow model loader
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Import dataset loading function
from preprocessing import load_dataset


# =============================================================================
# Dataset Path
# =============================================================================

# Path to the EuroSAT RGB dataset
DATASET_PATH = "D:\\7th semester\\Computer Vision\\project\\data_set\\EuroSAT_RGB\\EuroSAT_RGB"


# =============================================================================
# Load Dataset
# =============================================================================

print("Loading dataset...")

# Load images, labels, and class names
X, y, labels = load_dataset(DATASET_PATH)

# Convert integer labels into one-hot encoded labels
y_cat = to_categorical(y, num_classes=len(labels))


# =============================================================================
# Load Trained CNN Model
# =============================================================================

print("Loading model...")

# Load the trained EuroSAT classification model
model = load_model("models/eurosat_model.keras")


# =============================================================================
# Predict Image Classes
# =============================================================================

print("Predicting...")

# Predict class probabilities for all images
y_pred = model.predict(X, verbose=1)

# Convert one-hot encoded labels back to class indices
y_true = np.argmax(y_cat, axis=1)

# Select the class with the highest prediction probability
y_pred = np.argmax(y_pred, axis=1)


# =============================================================================
# Generate Confusion Matrix
# =============================================================================

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)


# =============================================================================
# Generate Classification Report
# =============================================================================

print("Generating classification report...")

# Calculate precision, recall, F1-score, and accuracy
report = classification_report(
    y_true,
    y_pred,
    target_names=labels
)

# Display report in the console
print(report)


# =============================================================================
# Save Classification Report
# =============================================================================

# Save the report as a text file
with open("results/classification_report.txt", "w") as f:
    f.write("EuroSAT Classification Report\n")
    f.write("=" * 40 + "\n\n")
    f.write(report)

print("Saved: results/classification_report.txt")


# =============================================================================
# Plot Confusion Matrix
# =============================================================================

# Create a larger figure for better visibility
plt.figure(figsize=(14, 12))

# Draw the confusion matrix as a heatmap
sns.heatmap(
    cm,
    annot=True,              # Display values inside cells
    fmt="d",                 # Display integers
    cmap="Blues",            # Blue color map
    xticklabels=labels,      # Predicted class names
    yticklabels=labels,      # Actual class names
    linewidths=0.5,
    linecolor="gray",
    cbar=True,               # Display color bar
    square=True,
    annot_kws={"size": 10}
)

# Add title and axis labels
plt.title("EuroSAT Confusion Matrix", fontsize=16, pad=20)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)

# Rotate x-axis labels for readability
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Adjust layout to avoid overlapping labels
plt.tight_layout()

# Save confusion matrix image
plt.savefig("results/confusion_matrix.png", dpi=300)

# Display the confusion matrix
plt.show()

print("Saved: results/confusion_matrix.png")