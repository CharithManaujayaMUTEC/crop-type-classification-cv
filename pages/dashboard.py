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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

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

# Handle query parameters for page navigation
query_params = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
if "tab" in query_params:
    requested_tab = query_params.get("tab")
    if isinstance(requested_tab, list):
        requested_tab = requested_tab[0]
    if requested_tab in NAV_ITEMS:
        st.session_state["page"] = requested_tab

# Top bar
st.markdown(
    '<div class="topbar"><strong>EuroSAT · Crop Type Classification</strong>'
    ' &nbsp;|&nbsp; Image Processing & Computer Vision Project</div>',
    unsafe_allow_html=True
)

# Navigation and Home link
st.page_link("app.py", label="← Back to Home Page", icon="🏠")

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
#  PAGE 5  —  Predict Image
# ══════════════════════════════════════════════
elif page == "Predict Image":
    import pandas as pd

    # ── lookup tables ────────────────────────────────────────────
    CLASS_INFO = {
        "AnnualCrop":           ("🌾", "Annual Crop",           "Farmland planted with crops harvested within one year (wheat, corn, sunflowers, etc.)"),
        "Forest":               ("🌲", "Forest",                "Dense woodland — natural or managed — with a continuous tree canopy."),
        "HerbaceousVegetation": ("🌿", "Herbaceous Vegetation", "Low-lying grassland, shrubs, or meadow with no significant tree cover."),
        "Highway":              ("🛣️",  "Highway",              "Major roads or motorways, often with clear lane markings visible from above."),
        "Industrial":           ("🏭", "Industrial Area",       "Factories, warehouses, or large facilities with hard surfaces and structures."),
        "Pasture":              ("🐄", "Pasture",               "Open grassy land used for livestock grazing — uniform, managed grass."),
        "PermanentCrop":        ("🍇", "Permanent Crop",        "Long-term crops like vineyards, orchards, or olive groves."),
        "Residential":          ("🏘️",  "Residential Area",     "Housing estates or urban neighbourhoods with buildings, roads, and gardens."),
        "River":                ("🏞️",  "River",                "A flowing body of water — river, canal, or stream."),
        "SeaLake":              ("🌊", "Sea / Lake",            "Large open or still water — ocean, sea, or inland lake."),
    }

    HEALTH_ICON  = {"Dense Healthy Vegetation": "🟢", "Healthy Vegetation": "🟢",
                    "Moderately Stressed Vegetation": "🟡", "Highly Stressed Vegetation": "🔴"}
    HEALTH_GUIDE = {
        "Dense Healthy Vegetation":      ("🌿 Dense & Healthy",   "#1b5e20", "#e8f5e9"),
        "Healthy Vegetation":            ("🌿 Healthy",           "#2e7d32", "#f1f8e9"),
        "Moderately Stressed Vegetation":("⚠️ Moderate Stress",  "#e65100", "#fff8e1"),
        "Highly Stressed Vegetation":    ("🔴 High Stress",       "#b71c1c", "#ffebee"),
    }

    def conf_color(c):
        if c >= 80: return "#2e7d32"
        if c >= 55: return "#f57c00"
        return "#c62828"

    def conf_label(c):
        if c >= 80: return "✅  High confidence"
        if c >= 55: return "⚠️  Moderate confidence"
        return "❓  Low confidence — result may be uncertain"

    def ndvi_bar_html(value, low=0.0, high=1.0, color="#4a7fcb"):
        pct = max(0, min(100, (value - low) / (high - low) * 100))
        return (
            f'<div style="background:#e0e0e0;border-radius:4px;height:8px;width:100%;margin-top:4px">'
            f'<div style="background:{color};border-radius:4px;height:8px;width:{pct:.0f}%"></div></div>'
        )

    # ── model guard ───────────────────────────────────────────────
    if not (os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH)):
        st.warning("Trained model not found. Complete Step 3 first.")
        st.stop()

    @st.cache_resource
    def load_trained_model():
        m   = tf.keras.models.load_model(MODEL_PATH)
        lbs = list(np.load(LABELS_PATH, allow_pickle=True))
        return m, lbs

    model, labels = load_trained_model()

    # ══════════════════════════════════════════
    #  UPLOAD — both files side by side
    # ══════════════════════════════════════════
    st.subheader("Predict Image")

    up_col, tif_col = st.columns(2)

    with up_col:
        st.markdown("**1 · Satellite Image** (required)")
        uploaded = st.file_uploader(
            "JPG or PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    with tif_col:
        st.markdown("**2 · Multispectral .tif** *(optional — for vegetation health)*")
        if NDVI_AVAILABLE:
            tif_uploaded = st.file_uploader(
                ".tif / .tiff",
                type=["tif", "tiff"],
                key="tif_uploader",
                label_visibility="collapsed",
            )
        else:
            tif_uploaded = None
            st.caption("Install `rasterio` to enable vegetation analysis.")

    if uploaded is None:
        st.info("⬆️  Upload a satellite image above to begin.")
        st.stop()

    raw_img     = Image.open(uploaded).convert("RGB")
    img_array   = np.array(raw_img)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_norm    = img_resized.astype("float32") / 255.0
    img_input   = np.expand_dims(img_norm, axis=0)

    st.markdown("---")
    img_col, btn_col = st.columns([1, 2])
    img_col.image(raw_img, caption="Uploaded image", use_container_width=True)
    with btn_col:
        st.write("")
        run = st.button("🔍  Analyse Image", type="primary", use_container_width=True)
        if tif_uploaded:
            st.caption("✅  Multispectral file ready — vegetation analysis will also run.")
        else:
            st.caption("ℹ️  No .tif uploaded — only land classification will run.")

    # ── run inference ─────────────────────────────────────────────
    if run:
        with st.spinner("Classifying …"):
            preds = model.predict(img_input, verbose=0)[0]
        st.session_state["preds"]       = preds
        st.session_state["pred_labels"] = labels

        # run vegetation analysis if TIF was provided
        if tif_uploaded and NDVI_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
                tmp.write(tif_uploaded.read())
                tmp_path = tmp.name
            with st.spinner("Analysing vegetation …"):
                try:
                    st.session_state["veg"]    = analyze_vegetation(tmp_path)
                    st.session_state["veg_ok"] = True
                except Exception as exc:
                    st.session_state["veg_ok"] = False
                    st.error(f"Vegetation analysis failed: {exc}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        else:
            st.session_state.pop("veg", None)
            st.session_state["veg_ok"] = False

    if "preds" not in st.session_state:
        st.stop()

    preds      = st.session_state["preds"]
    labels     = st.session_state["pred_labels"]
    top_idx    = int(np.argmax(preds))
    top_class  = labels[top_idx]
    confidence = float(preds[top_idx]) * 100
    emoji, friendly_name, description = CLASS_INFO.get(top_class, ("📍", top_class, ""))
    color = conf_color(confidence)

    # ══════════════════════════════════════════
    #  SECTION A — LAND TYPE RESULT
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🗺️ Land Type Classification")

    # ── primary result card ───────────────────
    st.markdown(
        f"""
        <div style="background:#f4f8ff;border-left:6px solid {color};
                    border-radius:8px;padding:20px 24px;margin-bottom:12px;
                    font-family:'Courier New',monospace;">
          <div style="font-size:40px;line-height:1.1">{emoji}</div>
          <div style="font-size:28px;font-weight:700;color:#1a3a6b;margin:4px 0 2px">{friendly_name}</div>
          <div style="font-size:13px;color:#555;margin-bottom:14px">{description}</div>
          <div style="font-size:20px;font-weight:600;color:{color}">{confidence:.1f}% confident</div>
          <div style="font-size:12px;color:{color};margin-bottom:8px">{conf_label(confidence)}</div>
          <div style="background:#ddd;border-radius:4px;height:10px;width:100%">
            <div style="background:{color};border-radius:4px;height:10px;
                        width:{min(confidence,100):.0f}%"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── next best guesses ─────────────────────
    st.markdown("**Other possibilities the model considered**")
    sorted_all  = np.argsort(preds)[::-1]          # all 10, best first
    alt_idx     = [i for i in sorted_all if i != top_idx][:3]
    c1, c2, c3  = st.columns(3)
    for col, idx in zip([c1, c2, c3], alt_idx):
        lbl   = labels[idx]
        prob  = float(preds[idx]) * 100
        em, nm, _ = CLASS_INFO.get(lbl, ("", lbl, ""))
        col.metric(f"{em}  {nm}", f"{prob:.1f}%")

    # ── full confidence breakdown (expander) ──
    with st.expander("📊  View confidence breakdown for all 10 land types"):
        chart_labels = [f"{CLASS_INFO.get(l,('','',l,''))[0]} {CLASS_INFO.get(l,('',l,''))[1]}"
                        for l in labels]
        bar_colors   = [color if i == top_idx else "#4a7fcb" for i in range(len(labels))]
        fig, ax = plt.subplots(figsize=(10, 3))
        bars = ax.bar(chart_labels, preds * 100, color=bar_colors)
        ax.set_ylabel("Confidence (%)")
        ax.set_ylim(0, 110)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        ax.bar_label(bars,
                     labels=[f"{v*100:.0f}%" if v*100 > 2 else "" for v in preds],
                     fontsize=8, padding=2)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.dataframe(
            pd.DataFrame({
                "Land Type":   chart_labels,
                "Confidence":  [f"{float(preds[i])*100:.2f}%" for i in range(len(labels))],
            }).sort_values("Confidence", ascending=False).reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    # ══════════════════════════════════════════
    #  SECTION B — VEGETATION HEALTH ANALYSIS
    # ══════════════════════════════════════════
    st.markdown("---")
    st.markdown("### 🌿 Vegetation Health Analysis")

    veg_ok = st.session_state.get("veg_ok", False)

    if not veg_ok:
        st.info(
            "No vegetation data yet.  \n"
            "Upload a matching **Sentinel-2 .tif** file above and click **Analyse Image** again."
        )
    else:
        veg      = st.session_state["veg"]
        health   = veg["health"]
        ndvi_val = veg["avg_ndvi"]
        cov      = veg["vegetation_coverage"]
        dense    = veg["vegetation_density"]
        stress   = veg["stress_percentage"]

        hlabel, hcolor, hbg = HEALTH_GUIDE.get(
            health, ("Unknown", "#555", "#fafafa")
        )

        # ── health status banner ──────────────
        st.markdown(
            f"""
            <div style="background:{hbg};border-left:6px solid {hcolor};
                        border-radius:8px;padding:16px 20px;margin-bottom:16px;
                        font-family:'Courier New',monospace">
              <div style="font-size:22px;font-weight:700;color:{hcolor}">{hlabel}</div>
              <div style="font-size:13px;color:#555;margin-top:4px">
                NDVI Score: <strong>{ndvi_val:.2f}</strong> &nbsp;|&nbsp;
                Scale: −1 (no vegetation) → 0 (bare soil) → +1 (dense forest)
              </div>
              {ndvi_bar_html(ndvi_val, -1.0, 1.0, hcolor)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── plain-English interpretation ──────
        if ndvi_val >= 0.6:
            summary = "The vegetation in this area is **lush and thriving** with strong photosynthetic activity."
        elif ndvi_val >= 0.4:
            summary = "The vegetation is **moderately healthy** — growing well but worth monitoring."
        elif ndvi_val >= 0.2:
            summary = "The vegetation is **under stress** — possible drought, disease, or thinning cover."
        else:
            summary = "**Little or no healthy vegetation** detected — mainly bare soil, built surfaces, or water."

        st.markdown(
            f"{summary}  \n"
            f"Roughly **{cov:.0f}%** of this scene is vegetated, "
            f"**{dense:.0f}%** has dense cover, and **{stress:.0f}%** shows stress signals."
        )

        # ── key metric cards ──────────────────
        st.markdown("#### At a glance")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("NDVI Score",          f"{ndvi_val:.2f}",
                  help="−1 to +1. Above 0.5 = healthy. Below 0.2 = stressed or no vegetation.")
        m2.metric("Vegetation Coverage", f"{cov:.1f}%",
                  help="Percentage of the image area covered by any vegetation.")
        m3.metric("Dense Cover",         f"{dense:.1f}%",
                  help="Percentage with very high vegetation density (NDVI > 0.6).")
        m4.metric("Stress Level",        f"{stress:.1f}%",
                  help="Percentage showing stress signals or bare soil (NDVI < 0.3).")

        # ── visual maps ──────────────────────
        ndvi_path   = os.path.join(RESULTS_DIR, "01_ndvi.png")
        health_path = os.path.join(RESULTS_DIR, "12_health_map.png")

        if os.path.exists(ndvi_path) or os.path.exists(health_path):
            st.markdown("#### Vegetation Maps")
            mc1, mc2 = st.columns(2)
            if os.path.exists(ndvi_path):
                mc1.image(ndvi_path,
                          caption="NDVI Map — green = healthy, red = stressed / bare",
                          use_container_width=True)
            if os.path.exists(health_path):
                mc2.image(health_path,
                          caption="Vegetation Health Map — 3 zones: stressed / moderate / dense",
                          use_container_width=True)

        # ── detailed stats (expander) ─────────
        with st.expander("🔬  Detailed vegetation statistics"):

            st.markdown("**Segmentation**")
            s1, s2 = st.columns(2)
            s1.metric("Otsu Threshold",
                      veg["otsu_threshold"],
                      help="Auto-computed pixel threshold used to separate vegetation from background.")
            s2.metric("Vegetation Regions",
                      veg["regions"],
                      help="Number of distinct vegetation patches detected in the image.")

            st.markdown("**Region Analysis**")
            r1, r2, r3, r4 = st.columns(4)
            r1.metric("Largest Region",  f"{veg['largest_area']:.0f} px²",
                      help="Area of the single largest vegetation patch.")
            r2.metric("Average Region",  f"{veg['average_area']:.0f} px²",
                      help="Mean area across all detected vegetation regions.")
            r3.metric("Total Area",      f"{veg['total_region_area']:.0f} px²",
                      help="Sum of all vegetation region areas.")
            r4.metric("Avg Perimeter",   f"{veg['average_perimeter']:.0f} px",
                      help="Mean perimeter (boundary length) of vegetation regions.")

            st.markdown("**Edge Analysis**")
            st.metric("Edge Pixels", veg["edge_pixels"],
                      help="Number of pixels lying on a vegetation boundary — higher = more fragmented cover.")

            st.markdown("**NDVI Distribution**")
            nd1, nd2, nd3, nd4 = st.columns(4)
            nd1.metric("Mean NDVI", veg["avg_ndvi"])
            nd2.metric("Std Dev",   veg["std_ndvi"])
            nd3.metric("Max NDVI",  veg["max_ndvi"])
            nd4.metric("Min NDVI",  veg["min_ndvi"])

        # ── processing pipeline images (expander) ──
        pipeline_images = [
            ("01_ndvi.png",                 "NDVI"),
            ("02_normalized_ndvi.png",       "Normalized NDVI"),
            ("03_histogram_equalized.png",   "Histogram Equalized"),
            ("04_gaussian_blur.png",         "Gaussian Blur"),
            ("05_otsu_segmentation.png",     "Otsu Segmentation"),
            ("06_adaptive_segmentation.png", "Adaptive Segmentation"),
            ("07_combined_mask.png",         "Combined Mask"),
            ("08_opening.png",               "Morphological Opening"),
            ("09_closing.png",               "Morphological Closing"),
            ("10_edges.png",                 "Canny Edges"),
            ("11_contours.png",              "Contours"),
            ("12_health_map.png",            "Vegetation Health Map"),
        ]

        with st.expander("🖼️  View all 12 processing pipeline images"):
            for row_start in range(0, len(pipeline_images), 3):
                row  = pipeline_images[row_start : row_start + 3]
                cols = st.columns(3)
                for col, (fname, title) in zip(cols, row):
                    fpath = os.path.join(RESULTS_DIR, fname)
                    if os.path.exists(fpath):
                        col.image(fpath, caption=title, use_container_width=True)
                    else:
                        col.caption(f"⏳ {fname} not yet generated")

        # ── full text report (expander) ────────
        with st.expander("📋  Full analysis report"):
            pipe_list = "".join(f"  ✓ {f}\n" for f, _ in pipeline_images)
            report = (
                f"{'='*60}\n"
                f"SATELLITE IMAGE ANALYSIS REPORT\n"
                f"{'='*60}\n\n"
                f"CLASSIFICATION RESULTS\n{'-'*40}\n"
                f"Predicted Class      : {friendly_name}\n"
                f"Confidence           : {confidence:.2f}%\n\n"
                f"NDVI STATISTICS\n{'-'*40}\n"
                f"Average NDVI         : {veg['avg_ndvi']}\n"
                f"NDVI Std Dev         : {veg['std_ndvi']}\n"
                f"Maximum NDVI         : {veg['max_ndvi']}\n"
                f"Minimum NDVI         : {veg['min_ndvi']}\n\n"
                f"VEGETATION HEALTH\n{'-'*40}\n"
                f"Health Status        : {health}\n\n"
                f"SEGMENTATION RESULTS\n{'-'*40}\n"
                f"Otsu Threshold       : {veg['otsu_threshold']}\n"
                f"Vegetation Regions   : {veg['regions']}\n\n"
                f"REGION ANALYSIS\n{'-'*40}\n"
                f"Largest Region Area  : {veg['largest_area']}\n"
                f"Average Region Area  : {veg['average_area']}\n"
                f"Total Region Area    : {veg['total_region_area']}\n"
                f"Average Perimeter    : {veg['average_perimeter']}\n\n"
                f"COVERAGE ANALYSIS\n{'-'*40}\n"
                f"Vegetation Coverage  : {cov}%\n"
                f"Vegetation Density   : {dense}%\n"
                f"Stress Percentage    : {stress}%\n\n"
                f"EDGE ANALYSIS\n{'-'*40}\n"
                f"Edge Pixels          : {veg['edge_pixels']}\n\n"
                f"GENERATED PIPELINE IMAGES\n{'-'*40}\n"
                f"{pipe_list}\n"
                f"Analysis Complete.\n"
                f"{'='*60}"
            )
            st.code(report, language="text")