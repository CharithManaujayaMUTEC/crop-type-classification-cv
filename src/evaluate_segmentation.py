import os
import cv2
import numpy as np

from ndvi import analyze_vegetation

# ==========================================
# CONFIGURATION
# ==========================================

IMAGE_DIR = "evaluation/images"
MASK_DIR = "evaluation/masks"

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "segmentation_evaluation.txt"
)

# ==========================================
# Metric Functions
# ==========================================

def pixel_accuracy(pred, gt):

    return np.mean(pred == gt)


def iou(pred, gt):

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    return intersection / (union + 1e-8)


def dice(pred, gt):

    intersection = np.logical_and(pred, gt).sum()

    return (2 * intersection) / (
        pred.sum() + gt.sum() + 1e-8
    )


def precision(pred, gt):

    tp = np.logical_and(pred, gt).sum()

    fp = np.logical_and(pred, np.logical_not(gt)).sum()

    return tp / (tp + fp + 1e-8)


def recall(pred, gt):

    tp = np.logical_and(pred, gt).sum()

    fn = np.logical_and(
        np.logical_not(pred),
        gt
    ).sum()

    return tp / (tp + fn + 1e-8)


def f1_score(p, r):

    return (2 * p * r) / (p + r + 1e-8)


# ==========================================
# Evaluation
# ==========================================

pixel_scores = []
iou_scores = []
dice_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

report = []

print("=" * 70)
print("SEGMENTATION EVALUATION")
print("=" * 70)

for image_name in sorted(os.listdir(IMAGE_DIR)):

    if not image_name.lower().endswith(".tif"):
        continue

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    mask_name = os.path.splitext(image_name)[0] + ".png"

    mask_path = os.path.join(
        MASK_DIR,
        mask_name
    )

    if not os.path.exists(mask_path):

        print(f"Missing mask : {mask_name}")
        continue

    # --------------------------------------
    # Run NDVI pipeline
    # --------------------------------------

    result = analyze_vegetation(image_path)

    predicted = result["mask"]

    predicted = (predicted > 127).astype(np.uint8)

    # --------------------------------------
    # Load Ground Truth
    # --------------------------------------

    gt = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    gt = cv2.resize(
        gt,
        (
            predicted.shape[1],
            predicted.shape[0]
        )
    )

    gt = (gt > 127).astype(np.uint8)

    # --------------------------------------
    # Metrics
    # --------------------------------------

    pa = pixel_accuracy(predicted, gt)

    iu = iou(predicted, gt)

    dc = dice(predicted, gt)

    pr = precision(predicted, gt)

    rc = recall(predicted, gt)

    f1 = f1_score(pr, rc)

    pixel_scores.append(pa)
    iou_scores.append(iu)
    dice_scores.append(dc)
    precision_scores.append(pr)
    recall_scores.append(rc)
    f1_scores.append(f1)

    print(f"\n{image_name}")

    print(f"Pixel Accuracy : {pa:.4f}")
    print(f"IoU            : {iu:.4f}")
    print(f"Dice           : {dc:.4f}")
    print(f"Precision      : {pr:.4f}")
    print(f"Recall         : {rc:.4f}")
    print(f"F1 Score       : {f1:.4f}")

    report.append(
        f"""
Image : {image_name}

Pixel Accuracy : {pa:.4f}
IoU            : {iu:.4f}
Dice           : {dc:.4f}
Precision      : {pr:.4f}
Recall         : {rc:.4f}
F1 Score       : {f1:.4f}
"""
    )

# ==========================================
# Final Results
# ==========================================

if len(pixel_scores) == 0:

    print("No images evaluated.")
    exit()

avg_pa = np.mean(pixel_scores)
avg_iou = np.mean(iou_scores)
avg_dice = np.mean(dice_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

summary = f"""

======================================================
FINAL SEGMENTATION RESULTS
======================================================

Images Evaluated : {len(pixel_scores)}

Pixel Accuracy : {avg_pa:.4f}
IoU            : {avg_iou:.4f}
Dice           : {avg_dice:.4f}
Precision      : {avg_precision:.4f}
Recall         : {avg_recall:.4f}
F1 Score       : {avg_f1:.4f}

======================================================
"""

print(summary)

with open(REPORT_PATH, "w") as f:

    for r in report:
        f.write(r)

    f.write(summary)

print("Report saved to:")
print(REPORT_PATH)