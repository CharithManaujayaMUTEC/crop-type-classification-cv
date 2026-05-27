import os
from preprocessing import load_dataset
from model import build_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

DATASET_PATH = "/content/drive/MyDrive/dataset"

def train():
    print("Loading dataset...")
    X, y, labels = load_dataset(DATASET_PATH)

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
        validation_data=(X_test, y_test)
    )

    print("Evaluating...")
    loss, acc = model.evaluate(X_test, y_test)
    print("Accuracy:", acc)

    print("Saving model...")
    model.save("/content/drive/MyDrive/crop_model.h5")

if __name__ == "__main__":
    train()