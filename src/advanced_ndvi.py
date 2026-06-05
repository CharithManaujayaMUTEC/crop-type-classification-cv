import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio

# Load image and compute NDVI

def load_and_compute_ndvi(image_path: str) -> np.ndarray:
    """
    Open a multispectral .tif file and compute the NDVI map.

    Sentinel-2 band layout used by EuroSAT_MS:
        Index 0 → B02 Blue
        Index 1 → B03 Green
        Index 2 → B04 Red       ← used for NDVI
        Index 3 → B05 Red-Edge
        Index 4 → B06 Red-Edge
        Index 5 → B07 Red-Edge
        Index 6 → B08 NIR       ← used for NDVI
        Index 7 → B08A NIR
        Index 8 → B11 SWIR
        Index 9 → B12 SWIR

    NDVI = (NIR - Red) / (NIR + Red)
    Range: -1.0 (water/no vegetation) → +1.0 (dense healthy vegetation)
    """
    with rasterio.open(image_path) as src:
        data = src.read()          # shape: (bands, height, width)

    # Band indices (0-based) for Sentinel-2 in EuroSAT_MS
    red = data[2].astype(np.float32)   # B04
    nir = data[6].astype(np.float32)   # B08

    # Avoid division by zero with a tiny epsilon
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Clip to valid NDVI range [-1, 1]
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi

# Save NDVI color map

def save_ndvi_map(ndvi: np.ndarray, output_path: str) -> None:
    """
    Save a color-coded NDVI map image.
    Color scheme RdYlGn:
        Red   → low/negative NDVI (water, bare soil, urban)
        Yellow → moderate NDVI (sparse/stressed vegetation)
        Green → high NDVI (healthy dense vegetation)
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    plt.colorbar(im, ax=ax, label="NDVI Value", shrink=0.85)
    ax.set_title("NDVI Map", fontsize=14, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] NDVI map saved → {output_path}")


# Generate NDVI histogram

def save_ndvi_histogram(ndvi: np.ndarray, output_path: str) -> None:
    """
    Save a histogram showing the distribution of NDVI values across the image.

    The histogram shows how many pixels fall in each NDVI range.
    Three vertical lines mark the thresholds used for health classification.
    """
    flat_ndvi = ndvi.flatten()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram with 60 bins
    n, bins, patches = ax.hist(
        flat_ndvi,
        bins=60,
        range=(-1.0, 1.0),
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    # Color the bars by NDVI category
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < 0.0:
            patch.set_facecolor("#d73027")    # Poor / non-vegetation (red)
        elif left_edge < 0.3:
            patch.set_facecolor("#fee08b")    # Moderate vegetation (yellow)
        else:
            patch.set_facecolor("#1a9850")    # Healthy vegetation (green)

    # Threshold marker lines
    ax.axvline(x=0.0,  color="#d73027", linestyle="--", linewidth=1.5, label="Poor threshold (0.0)")
    ax.axvline(x=0.3,  color="#1a9850", linestyle="--", linewidth=1.5, label="Healthy threshold (0.3)")

    ax.set_xlabel("NDVI Value", fontsize=12)
    ax.set_ylabel("Number of Pixels", fontsize=12)
    ax.set_title("NDVI Value Distribution Histogram", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-1.0, 1.0)

    # Add category labels below the x-axis
    for x_pos, label, color in [
        (-0.5,  "Non-vegetation", "#d73027"),
        (0.15,  "Moderate",       "#fee08b"),
        (0.65,  "Healthy",        "#1a9850"),
    ]:
        ax.text(
            x_pos, -0.06 * n.max(), label,
            ha="center", fontsize=9, color=color, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [✓] NDVI histogram saved → {output_path}")

# Calculate vegetation health statistics

def calculate_vegetation_statistics(ndvi: np.ndarray) -> dict:
    """
    Classify every pixel into one of three health categories and return stats.

    Classification thresholds:
        NDVI < 0.0          → Poor Vegetation (bare soil, water, urban)
        0.0 ≤ NDVI < 0.3    → Moderate Vegetation (sparse / stressed crops)
        NDVI ≥ 0.3          → Healthy Vegetation (dense, well-nourished crops)

    Returns a dict with counts, percentages, and descriptive statistics.
    """
    total_pixels = ndvi.size

    poor_mask     = ndvi < 0.0
    moderate_mask = (ndvi >= 0.0) & (ndvi < 0.3)
    healthy_mask  = ndvi >= 0.3

    poor_pct     = 100.0 * poor_mask.sum()     / total_pixels
    moderate_pct = 100.0 * moderate_mask.sum() / total_pixels
    healthy_pct  = 100.0 * healthy_mask.sum()  / total_pixels

    stats = {
        "total_pixels":   total_pixels,
        # Counts
        "poor_count":     int(poor_mask.sum()),
        "moderate_count": int(moderate_mask.sum()),
        "healthy_count":  int(healthy_mask.sum()),
        # Percentages (rounded to 2 decimal places)
        "poor_pct":       round(poor_pct,     2),
        "moderate_pct":   round(moderate_pct, 2),
        "healthy_pct":    round(healthy_pct,  2),
        # Descriptive statistics
        "ndvi_mean":      round(float(np.mean(ndvi)),   4),
        "ndvi_std":       round(float(np.std(ndvi)),    4),
        "ndvi_min":       round(float(np.min(ndvi)),    4),
        "ndvi_max":       round(float(np.max(ndvi)),    4),
        "ndvi_median":    round(float(np.median(ndvi)), 4),
    }
    return stats

# Save statistics report to text file

def save_vegetation_statistics(stats: dict, image_path: str, output_path: str) -> None:
    """
    Write the vegetation health statistics to a human-readable text file.
    """
    lines = [
        "=" * 55,
        "  NDVI Vegetation Health Analysis Report",
        "=" * 55,
        f"  Image analysed : {os.path.basename(image_path)}",
        f"  Total pixels   : {stats['total_pixels']:,}",
        "",
        "  -- Vegetation Health Classification --------------",
        f"  Healthy Vegetation  (NDVI >= 0.3) : {stats['healthy_pct']:6.2f}%  ({stats['healthy_count']:,} px)",
        f"  Moderate Vegetation (0 <= NDVI < 0.3): {stats['moderate_pct']:6.2f}%  ({stats['moderate_count']:,} px)",
        f"  Poor / Non-Vegetation (NDVI < 0)   : {stats['poor_pct']:6.2f}%  ({stats['poor_count']:,} px)",
        "",
        "  -- NDVI Descriptive Statistics ------------------",
        f"  Mean   : {stats['ndvi_mean']}",
        f"  Median : {stats['ndvi_median']}",
        f"  Std    : {stats['ndvi_std']}",
        f"  Min    : {stats['ndvi_min']}",
        f"  Max    : {stats['ndvi_max']}",
        "=" * 55,
    ]

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    # Print to console
    print()
    print(report)
    print(f"\n  [✓] Statistics saved → {output_path}")

# Main pipeline

def run_advanced_ndvi_analysis(image_path: str) -> None:
    """
    Full pipeline:
        1. Load image and compute NDVI
        2. Save color-coded NDVI map
        3. Save NDVI histogram
        4. Calculate & save vegetation statistics
    """
    # Ensure output directory exists
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 55)
    print("  Advanced NDVI Analysis  —  Member 3")
    print("=" * 55)
    print(f"  Input image : {image_path}\n")

    # Load and compute NDVI 
    print("  [1/4] Loading image and computing NDVI...")
    ndvi = load_and_compute_ndvi(image_path)
    print(f"        Image shape : {ndvi.shape}  (height × width)")

    # Save NDVI color map 
    print("  [2/4] Saving NDVI map...")
    save_ndvi_map(ndvi, "results/ndvi_map.png")

    # Save NDVI histogram 
    print("  [3/4] Generating NDVI histogram...")
    save_ndvi_histogram(ndvi, "results/ndvi_histogram.png")

    # Calculate and save statistics 
    print("  [4/4] Calculating vegetation health statistics...")
    stats = calculate_vegetation_statistics(ndvi)
    save_vegetation_statistics(stats, image_path, "results/vegetation_statistics.txt")

    print("\n  ✅  Analysis complete. All outputs saved in results/\n")

# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced NDVI Analysis — Satellite Analysis Enhancements (Member 3)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif",
        help="Path to the multispectral .tif satellite image (EuroSAT format)",
    )
    args = parser.parse_args()
    run_advanced_ndvi_analysis(args.image)