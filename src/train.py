import os
import numpy as np
import matplotlib.pyplot as plt

from preprocessing import load_dataset
from model import build_model

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


# ---------------- CONFIG ----------------

# Path to the EuroSAT RGB dataset
DATASET_PATH = "/content/drive/MyDrive/dataset/EuroSAT_RGB/EuroSAT_RGB"

# Location where the trained model will be saved
MODEL_SAVE_PATH = "models/eurosat_model.keras"


# ---------------- TRAIN FUNCTION ----------------

# Function to train the CNN model
def train():

    # Load all images and labels from the dataset
    print("Loading dataset...")
    X, y, labels = load_dataset(DATASET_PATH)

    # Check whether the dataset contains images
    if len(X) == 0:
        raise ValueError("Dataset is empty. Check DATASET_PATH.")

    # Display dataset information
    print(f"Total samples: {len(X)}")
    print(f"Number of classes: {len(labels)}")
    print("Classes found:")
    print(labels)

    # ---------------- PREPROCESS ----------------

    # Convert class labels into one-hot encoded vectors
    print("Preprocessing labels...")
    y = to_categorical(y, num_classes=len(labels))

    # ---------------- SPLIT ----------------

    # Split the dataset into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=np.argmax(y, axis=1)
    )

    # ---------------- BUILD MODEL ----------------

    # Create the CNN model
    print("Building model...")
    model = build_model(len(labels))

    # Display the model architecture
    model.summary()

    # ---------------- TRAIN ----------------

    # Train the CNN model using the training dataset
    print("Training...")
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test)
    )

    # ---------------- EVALUATE ----------------

    # Evaluate the trained model on the testing dataset
    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)

    # Display the final accuracy
    print(f"Final Accuracy: {acc:.4f}")

    # ---------------- SAVE MODEL ----------------

    # Create the models folder if it does not exist
    print("Saving model...")
    os.makedirs("models", exist_ok=True)

    # Save the trained CNN model
    model.save(MODEL_SAVE_PATH)

    print(f"Model saved at: {MODEL_SAVE_PATH}")

    # ---------------- SAVE LABELS ----------------

    # Save the class labels for future predictions
    print("Saving class labels...")
    np.save("models/eurosat_labels.npy", labels)

    # ---------------- SAVE ACCURACY PLOT ----------------

    # Plot training and validation accuracy
    print("Generating training plots...")

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("EuroSAT Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Save the accuracy graph
    plt.savefig("models/eurosat_accuracy_plot.png")
    plt.show()

    # ---------------- SAVE LOSS PLOT ----------------

    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("EuroSAT Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Save the loss graph
    plt.savefig("models/eurosat_loss_plot.png")
    plt.show()

    # Display completion message
    print("Training complete!")


# ---------------- MAIN ----------------

# Execute the training function when this file is run directly
if __name__ == "__main__":
    train()