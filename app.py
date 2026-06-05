"""
EuroSAT Crop Type Classification — Streamlit Dashboard
Author : Charith Manujaya
Run    : streamlit run app.py
"""

import os
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers, models

st.set_page_config(
    page_title="EuroSAT · Crop Classification",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [class*="css"], .stButton button, .stTextInput input,
.stNumberInput input, .stDataFrame, code, pre {
    font-family: 'Courier New', Courier, monospace !important;
}
header[data-testid="stHeader"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding-top: 16px !important; }

.topbar {
    background: #e8f0fe;
    border-bottom: 2px solid #4a7fcb;
    padding: 10px 20px;
    margin-bottom: 20px;
    font-family: 'Courier New', monospace;
    font-size: 13px;
    color: #1a3a6b;
}
.topbar strong { font-size: 15px; }

div[data-testid="metric-container"] {
    border: 1px solid #b0c8f0;
    border-radius: 4px;
    padding: 10px 14px;
    background: #f4f8ff;
}
div[data-testid="metric-container"] label {
    font-family: 'Courier New', monospace !important;
    font-size: 11px !important;
    color: #4a7fcb !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Courier New', monospace !important;
    color: #1a3a6b !important;
}

.stButton button {
    background: #4a7fcb !important;
    color: white !important;
    border: none !important;
    border-radius: 3px !important;
    font-size: 13px !important;
}
.stButton button:hover { background: #3366b3 !important; }

.result-box {
    background: #f4f8ff;
    border: 1px solid #4a7fcb;
    padding: 16px 20px;
    margin-bottom: 14px;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake"
]
IMG_SIZE = 64
MODEL_PATH = "models/eurosat_model.keras"
LABELS_PATH = "models/eurosat_labels.npy"
RESULTS_DIR = "results"

NAV_ITEMS = [
    "Project Overview",
    "Dataset & Configuration",
    "Train Model",
    "Training Results",
    "Predict Image",
]

if "page" not in st.session_state:
    st.session_state["page"] = "Project Overview"

st.markdown(
    '<div class="topbar"><strong>EuroSAT · Crop Type Classification</strong>'
    ' &nbsp;|&nbsp; Image Processing & Computer Vision Project</div>',
    unsafe_allow_html=True
)

cols = st.columns(len(NAV_ITEMS))
for i, (col, item) in enumerate(zip(cols, NAV_ITEMS)):
    if col.button(f"{i+1}. {item}", key=f"nav_{i}", use_container_width=True):
        st.session_state["page"] = item

page = st.session_state["page"]
st.caption(f"Current page: {page}")
st.markdown("---")

def load_dataset(path):
    X, y = [], []
    labels = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    for idx, label in enumerate(labels):
        folder = os.path.join(path, label)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255.0
            X.append(img)
            y.append(idx)
    return np.array(X), np.array(y), labels

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3,3), activation="relu"),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

if page == "Project Overview":
    st.subheader("Project Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", "EuroSAT RGB")
    c2.metric("Total Images", "27,000")
    c3.metric("Classes", "10")
    c4.metric("Image Size", "64 x 64 px")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Objectives**")
        st.markdown("""
- Classify satellite imagery into land-use and vegetation categories
- Train a CNN on EuroSAT RGB images (10 classes)
- Evaluate with accuracy/loss curves and confusion matrix
- Demonstrate Computer Vision in remote sensing
        """)
        st.markdown("**CNN Architecture**")
        st.code("""
Input (64x64x3)
  Conv2D  32 filters, 3x3, ReLU
  MaxPooling2D 2x2
  Conv2D  64 filters, 3x3, ReLU
  MaxPooling2D 2x2
  Flatten
  Dense   128, ReLU
  Dense   10,  Softmax
        """, language="text")

    with col_b:
        st.markdown("**Land-Cover Classes**")
        for cls in CLASSES:
            st.markdown(f"- {cls}")
        st.markdown("**Technologies**")
        st.markdown("Python, TensorFlow/Keras, OpenCV, NumPy, Matplotlib, Scikit-learn, Seaborn, Streamlit")

    st.info("Use the navigation buttons above to move through each step.")



if page == "Project Overview":
    # ... (previous Project Overview code)
    pass

elif page == "Dataset & Configuration":
    st.subheader("Dataset & Configuration")

    default_path = st.session_state.get("dataset_path", "../DATA_SET/EuroSAT_RGB/EuroSAT_RGB")
    dataset_path = st.text_input(
        "Path to EuroSAT_RGB folder (one sub-folder per class)",
        value=default_path,
    )

    if st.button("Verify Path"):
        if os.path.isdir(dataset_path):
            found = sorted([d for d in os.listdir(dataset_path)
                            if os.path.isdir(os.path.join(dataset_path, d))])
            if not found:
                st.error("No class sub-folders found.")
            else:
                st.success(f"Found {len(found)} class folders.")
                counts = {}
                for cls in found:
                    counts[cls] = len([f for f in os.listdir(os.path.join(dataset_path, cls))
                                       if f.lower().endswith((".jpg", ".png", ".tif"))])
                st.write(f"Total images: {sum(counts.values())}")
                c1, c2 = st.columns(2)
                for i, (cls, cnt) in enumerate(counts.items()):
                    (c1 if i % 2 == 0 else c2).write(f"{cls}: {cnt}")
                st.session_state["dataset_path"] = dataset_path
                st.session_state["dataset_verified"] = True
        else:
            st.error(f"Path not found: {dataset_path}")
            st.session_state["dataset_verified"] = False

    st.markdown("---")
    st.markdown("**Training Hyperparameters**")
    col1, col2, col3 = st.columns(3)
    epochs     = col1.number_input("Epochs",           min_value=1,  max_value=50,  value=10)
    batch_size = col2.number_input("Batch Size",        min_value=8,  max_value=128, value=32, step=8)
    test_split = col3.slider(      "Validation Split",  0.1, 0.4, 0.2, 0.05)

    st.session_state["train_cfg"] = {
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "test_split": float(test_split),
    }

    st.markdown("---")
    st.write("Model save path: `models/eurosat_model.keras`")
    st.write("Results path:    `results/`")

    if st.session_state.get("dataset_verified"):
        st.success("Configuration ready. Proceed to Step 3: Train Model.")
    else:
        st.warning("Verify the dataset path before training.")

elif page == "Train Model":
    st.subheader("Train Model")

    cfg   = st.session_state.get("train_cfg", {"epochs": 10, "batch_size": 32, "test_split": 0.2})
    dpath = st.session_state.get("dataset_path", "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Epochs",           cfg["epochs"])
    col2.metric("Batch Size",       cfg["batch_size"])
    col3.metric("Validation Split", f"{int(cfg['test_split']*100)}%")

    if not dpath or not os.path.isdir(dpath):
        st.warning("Dataset path not set. Go back to Step 2 first.")
        st.stop()

    st.markdown("### 🧠 Computer Vision Training Overview")

    st.code("""
1. Load satellite images from dataset folders
2. Resize images to fixed size (64×64)
3. Normalize pixel values (0–255 → 0–1)
4. Convert labels into one-hot vectors
5. Feed images into CNN model
6. CNN learns spatial features (edges → textures → land patterns)
7. Output class probabilities using Softmax
""", language="text")

    st.write("""
This training process transforms raw satellite images into structured numerical data
so that a neural network can learn visual patterns for land classification.
""")

    if st.button("Start Training", type="primary", use_container_width=True):
        st.session_state["training_done"] = False

        st.markdown("### 🧪 Dataset Loading & Preprocessing")

        with st.spinner("Loading dataset..."):
            X, y, labels = load_dataset(dpath)

        st.success(f"Loaded {len(X)} images, {len(labels)} classes.")

        st.write("""
Each image is:
- Resized to 64×64 pixels
- Normalized to [0,1]
- Stored as NumPy arrays for CNN input
""")

        y_cat = to_categorical(y, num_classes=len(labels))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_cat, test_size=cfg["test_split"],
            random_state=42, stratify=np.argmax(y_cat, axis=1)
        )

        st.write(f"Train: {len(X_train)}  |  Validation: {len(X_test)}")

        model = build_model(len(labels))

        st.markdown("### 🔥 CNN Feature Learning Explanation")

        st.write("""
CNN learns hierarchical features:

- Layer 1 → edges (simple patterns)
- Layer 2 → textures (vegetation, roads)
- Layer 3 → complex structures (urban, forest, water bodies)

Pooling layers reduce image size while preserving important features.
""")


        st.markdown("---")
        st.markdown("**Training Log (Learning Progress)**")

        log_area = st.empty()
        prog_bar = st.progress(0)
        status_tx = st.empty()

        train_acc_list, val_acc_list = [], []
        train_loss_list, val_loss_list = [], []
        log_lines = []

        col_ca, col_cl = st.columns(2)
        chart_acc = col_ca.empty()
        chart_loss = col_cl.empty()

        class StreamlitCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}

                ta = logs.get("accuracy", 0)
                va = logs.get("val_accuracy", 0)
                tl = logs.get("loss", 0)
                vl = logs.get("val_loss", 0)

                ep = epoch + 1
                total_ep = cfg["epochs"]

                train_acc_list.append(ta)
                val_acc_list.append(va)
                train_loss_list.append(tl)
                val_loss_list.append(vl)

                log_lines.append(
                    f"Ep {ep:>2}/{total_ep} | "
                    f"acc={ta:.4f} | val_acc={va:.4f} | "
                    f"loss={tl:.4f} | val_loss={vl:.4f}"
                )

                log_area.code("\n".join(log_lines), language="text")

                prog_bar.progress(ep / total_ep)
                status_tx.write(f"Epoch {ep}/{total_ep} | Validation Accuracy: {va:.4f}")

                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_acc_list, label="Train")
                ax.plot(val_acc_list, label="Validation", linestyle="--")
                ax.set_title("Accuracy Learning Curve")
                ax.legend()
                ax.grid(True, alpha=0.3)
                chart_acc.pyplot(fig)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_loss_list, label="Train")
                ax.plot(val_loss_list, label="Validation", linestyle="--")
                ax.set_title("Loss Learning Curve")
                ax.legend()
                ax.grid(True, alpha=0.3)
                chart_loss.pyplot(fig)
                plt.close(fig)

        history = model.fit(
            X_train, y_train,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            validation_data=(X_test, y_test),
            callbacks=[StreamlitCallback()],
            verbose=0,
        )