import os
from preprocessing import load_dataset
from model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Dataset path (Colab Drive still fine for data)
DATASET_PATH = "/content/drive/MyDrive/dataset/PlantVillage"

# Save inside repo (IMPORTANT CHANGE)
MODEL_SAVE_PATH = "models/crop_model.keras"

def train():
    print("Loading dataset...")
    X, y, labels = load_dataset(DATASET_PATH)

    #Safety check (prevents your previous crash)
    if len(X) == 0:
        raise ValueError("Dataset is empty. Check DATASET_PATH.")

    print("Splitting data...")
    y = to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Building model...")
    model = build_model(len(labels))

    print("Training...")
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,  #prevents memory warning
        validation_data=(X_test, y_test)
    )

    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print("Accuracy:", acc)

    print("Saving model...")

    # ensure models folder exists
    os.makedirs("models", exist_ok=True)

    # Save in repo (NOT Drive)
    model.save(MODEL_SAVE_PATH)

    print(f"Model saved at {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()