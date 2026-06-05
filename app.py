"""
EuroSAT Crop Type Classification — Streamlit Dashboard
Author : Charith Manujaya
Run    : streamlit run app.py
"""

import os
import io
import sys
import tempfile
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

try:
    import rasterio                          # noqa: F401 (used inside analyze_vegetation)
    from ndvi import analyze_vegetation
    NDVI_AVAILABLE = True
except ImportError:
    NDVI_AVAILABLE = False
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
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
IMG_SIZE    = 64
MODEL_PATH  = "models/eurosat_model.keras"
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

# Top bar
st.markdown(
    '<div class="topbar"><strong>EuroSAT · Crop Type Classification</strong>'
    ' &nbsp;|&nbsp; Image Processing & Computer Vision Project</div>',
    unsafe_allow_html=True
)

# Navigation
cols = st.columns(len(NAV_ITEMS))
for i, (col, item) in enumerate(zip(cols, NAV_ITEMS)):
    if col.button(f"{i+1}. {item}", key=f"nav_{i}", use_container_width=True):
        st.session_state["page"] = item

page = st.session_state["page"]
st.caption(f"Current page: {page}")
st.markdown("---")


# ── helpers ──────────────────────────────────
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


# ══════════════════════════════════════════════
#  PAGE 1
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
#  PAGE 2
# ══════════════════════════════════════════════
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


# ══════════════════════════════════════════════
#  PAGE 3
# ══════════════════════════════════════════════
# ══════════════════════════════════════════════
#  PAGE 3
# ══════════════════════════════════════════════
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

    # =========================
    # 🧠 CV TRAINING EXPLANATION (ADDED)
    # =========================
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

        # =========================
        # DATA LOADING
        # =========================
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

        # =========================
        # MODEL BUILDING
        # =========================
        model = build_model(len(labels))

        st.markdown("### 🔥 CNN Feature Learning Explanation")

        st.write("""
CNN learns hierarchical features:

- Layer 1 → edges (simple patterns)
- Layer 2 → textures (vegetation, roads)
- Layer 3 → complex structures (urban, forest, water bodies)

Pooling layers reduce image size while preserving important features.
""")

        # =========================
        # TRAINING VISUALIZATION
        # =========================
        st.markdown("---")
        st.markdown("**Training Log (Learning Progress)**")

        log_area  = st.empty()
        prog_bar  = st.progress(0)
        status_tx = st.empty()

        train_acc_list, val_acc_list   = [], []
        train_loss_list, val_loss_list = [], []
        log_lines = []

        col_ca, col_cl = st.columns(2)
        chart_acc  = col_ca.empty()
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

                # Accuracy plot
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(train_acc_list, label="Train")
                ax.plot(val_acc_list, label="Validation", linestyle="--")
                ax.set_title("Accuracy Learning Curve")
                ax.legend()
                ax.grid(True, alpha=0.3)
                chart_acc.pyplot(fig)
                plt.close(fig)

                # Loss plot
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

        # =========================
        # SAVE MODEL
        # =========================
        os.makedirs("models", exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        model.save(MODEL_PATH)
        np.save(LABELS_PATH, np.array(labels))

        # =========================
        # FINAL PLOTS
        # =========================
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["accuracy"], label="Train")
        ax.plot(history.history["val_accuracy"], label="Validation", linestyle="--")
        ax.set_title("Model Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f"{RESULTS_DIR}/accuracy_plot.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(history.history["loss"], label="Train")
        ax.plot(history.history["val_loss"], label="Validation", linestyle="--")
        ax.set_title("Model Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(f"{RESULTS_DIR}/loss_plot.png", dpi=120)
        plt.close(fig)

        # =========================
        # CONFUSION MATRIX
        # =========================
        y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
        y_true = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title("Confusion Matrix (Class Confusion Analysis)")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Actual Class")
        fig.tight_layout()
        fig.savefig(f"{RESULTS_DIR}/confusion_matrix.png", dpi=120)
        plt.close(fig)

        # =========================
        # STORE RESULTS
        # =========================
        final_val_acc = history.history["val_accuracy"][-1]

        st.session_state.update({
            "training_done": True,
            "train_history": history.history,
            "final_val_acc": final_val_acc,
            "trained_labels": labels,
        })

        prog_bar.progress(1.0)

        st.success(f"Training complete. Final validation accuracy: {final_val_acc:.4f}")

        # =========================
        # FINAL CV INTERPRETATION (ADDED)
        # =========================
        st.markdown("### 📊 Training Interpretation (Computer Vision View)")

        st.write("""
- High accuracy → model learned strong visual patterns
- Loss reduction → better feature representation
- Validation accuracy → ability to generalize to unseen images
- Confusion matrix → which land types look visually similar
""")

        st.info("Proceed to Step 4: Training Results.")

# ══════════════════════════════════════════════
#  PAGE 4
# ══════════════════════════════════════════════
elif page == "Training Results":
    st.subheader("Training Results")

    if not st.session_state.get("training_done"):
        if not os.path.exists(f"{RESULTS_DIR}/accuracy_plot.png"):
            st.warning("No training run found. Complete Step 3 first.")
            st.stop()
        st.info("Showing previously saved figures.")

    history = st.session_state.get("train_history")

    if history:
        final_acc  = history["val_accuracy"][-1]
        best_acc   = max(history["val_accuracy"])
        final_loss = history["val_loss"][-1]
        epochs_run = len(history["accuracy"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Val Accuracy", f"{final_acc*100:.2f}%")
        c2.metric("Best Val Accuracy",  f"{best_acc*100:.2f}%")
        c3.metric("Final Val Loss",     f"{final_loss:.4f}")
        c4.metric("Epochs",             epochs_run)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Accuracy**")
        acc_path = f"{RESULTS_DIR}/accuracy_plot.png"
        if os.path.exists(acc_path):
            st.image(acc_path, use_container_width=True)
        elif history:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history["accuracy"],     label="Train",      color="#4a7fcb", linewidth=2)
            ax.plot(history["val_accuracy"], label="Validation", color="#4a7fcb", linewidth=2, linestyle="--")
            ax.set_title("Accuracy"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown("**Loss**")
        loss_path = f"{RESULTS_DIR}/loss_plot.png"
        if os.path.exists(loss_path):
            st.image(loss_path, use_container_width=True)
        elif history:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(history["loss"],     label="Train",      color="#4a7fcb", linewidth=2)
            ax.plot(history["val_loss"], label="Validation", color="#4a7fcb", linewidth=2, linestyle="--")
            ax.set_title("Loss"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.markdown("**Confusion Matrix**")
    cm_path = f"{RESULTS_DIR}/confusion_matrix.png"
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
    else:
        st.info("Confusion matrix not found. Re-run training.")

    if history:
        st.markdown("---")
        st.markdown("**Epoch Log**")
        import pandas as pd
        n = len(history["accuracy"])
        df = pd.DataFrame({
            "Epoch":      range(1, n+1),
            "Train Acc":  [f"{v:.4f}" for v in history["accuracy"]],
            "Val Acc":    [f"{v:.4f}" for v in history["val_accuracy"]],
            "Train Loss": [f"{v:.4f}" for v in history["loss"]],
            "Val Loss":   [f"{v:.4f}" for v in history["val_loss"]],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.info("Proceed to Step 5: Predict Image.")


# ══════════════════════════════════════════════
#  PAGE 5
# ══════════════════════════════════════════════
# =========================
# PAGE 5 - PREDICT IMAGE
# =========================
# ══════════════════════════════════════════════
#  PAGE 5 - Predict Image (ENHANCED CV VERSION)
# ══════════════════════════════════════════════
elif page == "Predict Image":
    st.subheader("Predict Image")

    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        st.warning("Trained model not found. Complete Step 3 first.")
        st.stop()

    @st.cache_resource
    def load_trained_model():
        m   = tf.keras.models.load_model(MODEL_PATH)
        lbs = list(np.load(LABELS_PATH, allow_pickle=True))
        return m, lbs

    model, labels = load_trained_model()

    # =========================
    # STEP A - INPUT IMAGE
    # =========================
    st.markdown("**Step A — Upload Image**")

    uploaded = st.file_uploader(
        "Select a satellite image (JPG or PNG). Will be resized to 64x64.",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.info("Upload an image to begin.")
        st.stop()

    raw_img = Image.open(uploaded).convert("RGB")

    col_img, col_info = st.columns([1, 2])
    col_img.image(raw_img, caption="Uploaded image", use_container_width=True)
    col_info.write(f"Size: {raw_img.size[0]} x {raw_img.size[1]} px")
    col_info.write(f"Mode: {raw_img.mode}")

    # =========================
    # STEP B - IMAGE PROCESSING
    # =========================
    st.markdown("---")
    st.markdown("**Step B — Image Processing (Computer Vision Pipeline)**")

    img_array   = np.array(raw_img)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_norm    = img_resized.astype("float32") / 255.0
    img_input   = np.expand_dims(img_norm, axis=0)

    # ---- PIPELINE EXPLANATION ----
    st.markdown("**Image Transformation Pipeline**")
    st.code("""
RAW IMAGE
   ↓
RESIZE (64×64)
   ↓
FILTERING (Blur + Edge Detection)
   ↓
NORMALIZATION (0–255 → 0–1)
   ↓
TENSOR SHAPE (1, 64, 64, 3)
   ↓
CNN CLASSIFICATION
""", language="text")

    # =========================
    # ORIGINAL + RESIZE
    # =========================
    st.markdown("**Original vs Resized Image**")
    col1, col2 = st.columns(2)
    col1.image(raw_img, caption="Original Image", use_container_width=True)
    col2.image(img_resized, caption="Resized (64×64)", use_container_width=True)

    # =========================
    # FILTERING (CV FEATURE)
    # =========================
    st.markdown("**Image Filtering (Computer Vision Enhancement)**")

    blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)
    gray    = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    edges   = cv2.Canny(gray, 50, 120)

    col1, col2, col3 = st.columns(3)

    col1.image(img_resized, caption="Original", use_container_width=True)
    col2.image(blurred, caption="Gaussian Blur", use_container_width=True)
    col3.image(edges, caption="Edge Detection", use_container_width=True)

    st.write("""
Gaussian Blur removes noise and smooths image.
Edge Detection highlights boundaries (roads, crops, buildings).
""")

    # =========================
    # RGB CHANNELS
    # =========================
    st.markdown("**RGB Channel Analysis**")

    fig, axes = plt.subplots(1, 3, figsize=(6, 2.5))
    for ci, (name, cmap) in enumerate(zip(["Red","Green","Blue"], ["Reds","Greens","Blues"])):
        axes[ci].imshow(img_resized[:, :, ci], cmap=cmap)
        axes[ci].set_title(name)
        axes[ci].axis("off")

    st.pyplot(fig)
    plt.close(fig)

    st.write("""
Red → soil / roads  
Green → vegetation  
Blue → water / shadow information  
""")

    # =========================
    # HISTOGRAM (IMPORTANT CV PART)
    # =========================
    st.markdown("**Histogram Analysis (Before vs After Normalization)**")

    col1, col2 = st.columns(2)

    fig, ax = plt.subplots()
    ax.hist(img_resized.flatten(), bins=50)
    ax.set_title("Before Normalization (0–255)")
    ax.grid(True)
    col1.pyplot(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.hist(img_norm.flatten(), bins=50)
    ax.set_title("After Normalization (0–1)")
    ax.grid(True)
    col2.pyplot(fig)
    plt.close(fig)

    # =========================
    # TENSOR INFO
    # =========================
    st.markdown("**Tensor Preparation**")

    st.code(f"""
Original Shape : {img_array.shape}
Resized Shape  : {img_resized.shape}
Model Input    : {img_input.shape}

(1 = batch size for CNN)
""", language="text")

    # =========================
    # STEP C - INFERENCE
    # =========================
    st.markdown("---")
    st.markdown("**Step C — Model Inference**")

    if st.button("Run Inference", type="primary", use_container_width=True):
        with st.spinner("Running inference..."):
            preds = model.predict(img_input, verbose=0)[0]
        st.session_state["preds"] = preds
        st.session_state["pred_labels"] = labels

    if "preds" not in st.session_state:
        st.info("Click Run Inference to get results.")
        st.stop()

    # =========================
    # STEP D - RESULTS
    # =========================
    st.markdown("---")
    st.markdown("**Step D — Results**")

    preds      = st.session_state["preds"]
    labels     = st.session_state["pred_labels"]

    top_idx    = int(np.argmax(preds))
    top_class  = labels[top_idx]
    confidence = float(preds[top_idx]) * 100

    st.markdown(
        f'<div class="result-box">'
        f'Predicted class: <strong>{top_class}</strong><br>'
        f'Confidence: <strong>{confidence:.2f}%</strong>'
        f'</div>',
        unsafe_allow_html=True)

    st.progress(int(confidence))

    # =========================
    # TOP-5 RESULTS
    # =========================
    st.markdown("**Top-5 Predictions**")

    sorted_idx = np.argsort(preds)[::-1][:5]

    top5_labels = [labels[i] for i in sorted_idx]
    top5_probs  = [preds[i] * 100 for i in sorted_idx]

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots()
        ax.barh(top5_labels[::-1], top5_probs[::-1], color="#4a7fcb")
        ax.set_xlabel("Confidence (%)")
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        import pandas as pd
        st.dataframe(pd.DataFrame({
            "Class": top5_labels,
            "Confidence": [f"{p:.2f}%" for p in top5_probs]
        }))

    # =========================
    # ALL CLASS DISTRIBUTION
    # =========================
    st.markdown("**All-Class Probability Distribution**")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(labels, preds * 100)
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylabel("Confidence (%)")
    ax.grid(True, axis="y")

    st.pyplot(fig)
    plt.close(fig)

    st.success("Inference complete.")

    # =========================================================
    # STEP E — MULTISPECTRAL IMAGE UPLOAD (NDVI ANALYSIS)
    # =========================================================
    st.markdown("---")
    st.markdown("**Step E — Multispectral Image Upload (NDVI & Vegetation Analysis)**")

    if not NDVI_AVAILABLE:
        st.warning(
            "**rasterio** is not installed. Run `pip install rasterio` then restart "
            "the app to enable NDVI analysis."
        )
    else:
        st.info(
            "Optional: upload a Sentinel-2 multispectral `.tif` image "
            "(same scene as the RGB image above) to run the full NDVI pipeline."
        )

        tif_uploaded = st.file_uploader(
            "Select a Sentinel-2 multispectral image (.tif / .tiff)",
            type=["tif", "tiff"],
            key="tif_uploader",
        )

        if tif_uploaded is not None:

            # ---- Pipeline explanation ----
            st.markdown("**NDVI Processing Pipeline**")
            st.code("""\
MULTISPECTRAL IMAGE (.tif)
   ↓
EXTRACT RED (B04) + NIR (B08) BANDS
   ↓
COMPUTE NDVI  =  (NIR - RED) / (NIR + RED)
   ↓
NORMALIZE  →  HISTOGRAM EQUALIZATION  →  GAUSSIAN BLUR
   ↓
OTSU THRESHOLDING  +  ADAPTIVE THRESHOLDING
   ↓
MORPHOLOGICAL OPENING / CLOSING
   ↓
CANNY EDGE DETECTION  +  CONTOUR ANALYSIS
   ↓
VEGETATION HEALTH MAP (Stressed / Moderate / Healthy / Dense)
""", language="text")

            # Save uploaded bytes to a temp file so rasterio can open it
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(tif_uploaded.read())
                tmp_path = tmp.name

            with st.spinner("Running NDVI & vegetation analysis …"):
                try:
                    veg = analyze_vegetation(tmp_path)
                    ndvi_ok = True
                except Exception as exc:
                    st.error(f"Vegetation analysis failed: {exc}")
                    ndvi_ok = False
            try:
                os.remove(tmp_path)
            except OSError:
                pass

            if ndvi_ok:

                # =====================================================
                # STEP F — NDVI RESULTS
                # =====================================================
                st.markdown("---")
                st.markdown("**Step F — NDVI & Vegetation Analysis Results**")

                # NDVI Statistics
                st.markdown("**NDVI Statistics**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Avg NDVI", veg["avg_ndvi"])
                c2.metric("Std Dev",  veg["std_ndvi"])
                c3.metric("Max NDVI", veg["max_ndvi"])
                c4.metric("Min NDVI", veg["min_ndvi"])

                # Vegetation Health
                st.markdown("**Vegetation Health**")
                health = veg["health"]
                if "Dense" in health or ("Healthy" in health and "Stressed" not in health):
                    st.success(f"Health Status : **{health}**")
                elif "Moderately" in health:
                    st.warning(f"Health Status : **{health}**")
                else:
                    st.error(f"Health Status : **{health}**")

                # Segmentation Results
                st.markdown("**Segmentation Results**")
                c1, c2 = st.columns(2)
                c1.metric("Otsu Threshold",     veg["otsu_threshold"])
                c2.metric("Vegetation Regions", veg["regions"])

                # Region Analysis
                st.markdown("**Region Analysis**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Largest Region Area",  veg["largest_area"])
                c2.metric("Average Region Area",  veg["average_area"])
                c3.metric("Total Region Area",    veg["total_region_area"])
                c4.metric("Average Perimeter",    veg["average_perimeter"])

                # Coverage Analysis
                st.markdown("**Coverage Analysis**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Vegetation Coverage", f"{veg['vegetation_coverage']}%")
                c2.metric("Vegetation Density",  f"{veg['vegetation_density']}%")
                c3.metric("Stress Percentage",   f"{veg['stress_percentage']}%")

                # Edge Analysis
                st.markdown("**Edge Analysis**")
                st.metric("Edge Pixels", veg["edge_pixels"])

                # =====================================================
                # STEP G — PIPELINE IMAGES
                # =====================================================
                st.markdown("---")
                st.markdown("**Step G — Generated Pipeline Images**")

                pipeline_images = [
                    ("01_ndvi.png",                 "01 · NDVI"),
                    ("02_normalized_ndvi.png",       "02 · Normalized NDVI"),
                    ("03_histogram_equalized.png",   "03 · Histogram Equalized"),
                    ("04_gaussian_blur.png",         "04 · Gaussian Blur"),
                    ("05_otsu_segmentation.png",     "05 · Otsu Segmentation"),
                    ("06_adaptive_segmentation.png", "06 · Adaptive Segmentation"),
                    ("07_combined_mask.png",         "07 · Combined Mask"),
                    ("08_opening.png",               "08 · Morphological Opening"),
                    ("09_closing.png",               "09 · Morphological Closing"),
                    ("10_edges.png",                 "10 · Canny Edges"),
                    ("11_contours.png",              "11 · Contours"),
                    ("12_health_map.png",            "12 · Vegetation Health Map"),
                ]

                for row_start in range(0, len(pipeline_images), 3):
                    row = pipeline_images[row_start : row_start + 3]
                    cols = st.columns(3)
                    for col, (fname, title) in zip(cols, row):
                        fpath = os.path.join(RESULTS_DIR, fname)
                        if os.path.exists(fpath):
                            col.image(fpath, caption=f"✓ {title}", use_container_width=True)
                        else:
                            col.info(f"⏳ {fname}")

                # =====================================================
                # FINAL SATELLITE IMAGE ANALYSIS REPORT
                # =====================================================
                st.markdown("---")
                st.markdown("**Final Analysis Report**")

                pipe_list = "".join(
                    f"  ✓ {fname}\n" for fname, _ in pipeline_images
                )
                report = (
                    f"{'=' * 60}\n"
                    f"SATELLITE IMAGE ANALYSIS REPORT\n"
                    f"{'=' * 60}\n\n"
                    f"CLASSIFICATION RESULTS\n"
                    f"{'-' * 40}\n"
                    f"Predicted Class      : {top_class}\n"
                    f"Confidence           : {confidence:.2f}%\n\n"
                    f"NDVI STATISTICS\n"
                    f"{'-' * 40}\n"
                    f"Average NDVI         : {veg['avg_ndvi']}\n"
                    f"NDVI Std Dev         : {veg['std_ndvi']}\n"
                    f"Maximum NDVI         : {veg['max_ndvi']}\n"
                    f"Minimum NDVI         : {veg['min_ndvi']}\n\n"
                    f"VEGETATION HEALTH\n"
                    f"{'-' * 40}\n"
                    f"Health Status        : {veg['health']}\n\n"
                    f"SEGMENTATION RESULTS\n"
                    f"{'-' * 40}\n"
                    f"Otsu Threshold       : {veg['otsu_threshold']}\n"
                    f"Vegetation Regions   : {veg['regions']}\n\n"
                    f"REGION ANALYSIS\n"
                    f"{'-' * 40}\n"
                    f"Largest Region Area  : {veg['largest_area']}\n"
                    f"Average Region Area  : {veg['average_area']}\n"
                    f"Total Region Area    : {veg['total_region_area']}\n"
                    f"Average Perimeter    : {veg['average_perimeter']}\n\n"
                    f"COVERAGE ANALYSIS\n"
                    f"{'-' * 40}\n"
                    f"Vegetation Coverage  : {veg['vegetation_coverage']}%\n"
                    f"Vegetation Density   : {veg['vegetation_density']}%\n"
                    f"Stress Percentage    : {veg['stress_percentage']}%\n\n"
                    f"EDGE ANALYSIS\n"
                    f"{'-' * 40}\n"
                    f"Edge Pixels          : {veg['edge_pixels']}\n\n"
                    f"GENERATED PIPELINE IMAGES\n"
                    f"{'-' * 40}\n"
                    f"{pipe_list}\n"
                    f"Analysis Complete.\n"
                    f"{'=' * 60}"
                )
                st.code(report, language="text")

                st.success("Full satellite image analysis complete.")