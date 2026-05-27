import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_dataset
from model import build_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# ---------------- CONFIG ----------------
DATASET_PATH = "/content/drive/MyDrive/dataset/PlantVillage"
MODEL_SAVE_PATH = "models/crop_model.keras"

# ---------------- TRAIN FUNCTION ----------------
def train():
    print("Loading dataset...")
    X, y, labels = load_dataset(DATASET_PATH)

    # Safety check (prevents empty dataset crash)
    if len(X) == 0:
        raise ValueError("Dataset is empty. Check DATASET_PATH.")

    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {len(labels)}")

    # ---------------- PREPROCESS ----------------
    print("Preprocessing labels...")
    y = to_categorical(y)

    # ---------------- SPLIT ----------------
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- BUILD MODEL ----------------
    print("Building model...")
    model = build_model(len(labels))

    model.summary()

    # ---------------- TRAIN ----------------
    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,  # avoids memory issues
        validation_data=(X_test, y_test)
    )

    # ---------------- EVALUATE ----------------
    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Final Accuracy: {acc:.4f}")

    # ---------------- SAVE MODEL ----------------
    print("Saving model...")

    os.makedirs("models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)

    print(f"Model saved at: {MODEL_SAVE_PATH}")

    # ---------------- SAVE LABELS ----------------
    print("Saving class labels...")
    np.save("models/labels.npy", labels)

    # ---------------- PLOTS (FOR REPORT) ----------------
    print("Generating training plots...")

    # Accuracy plot
    plt.figure()
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("models/accuracy_plot.png")
    plt.show()

    # Loss plot
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("models/loss_plot.png")
    plt.show()

    print("Training complete!")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    train()