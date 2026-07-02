import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import rasterio


# =============================================================================
# Load image and compute NDVI
# =============================================================================

def load_and_compute_ndvi(image_path: str) -> np.ndarray:
    """
    Open a multispectral Sentinel-2 (.tif) image and compute the
    Normalized Difference Vegetation Index (NDVI).

    Sentinel-2 band layout used in the EuroSAT_MS dataset:

        Index 0  -> B02 (Blue)
        Index 1  -> B03 (Green)
        Index 2  -> B04 (Red)        <-- Used for NDVI
        Index 3  -> B05 (Red Edge)
        Index 4  -> B06 (Red Edge)
        Index 5  -> B07 (Red Edge)
        Index 6  -> B08 (Near Infrared - NIR) <-- Used for NDVI
        Index 7  -> B08A (NIR)
        Index 8  -> B11 (SWIR)
        Index 9  -> B12 (SWIR)

    NDVI Formula:
        NDVI = (NIR - Red) / (NIR + Red)

    NDVI Range:
        -1.0 -> Water / Non-vegetation
         0.0 -> Bare soil
         1.0 -> Dense healthy vegetation

    Parameters
    ----------
    image_path : str
        Path to the Sentinel-2 multispectral image.

    Returns
    -------
    np.ndarray
        Computed NDVI image.
    """

    # Open the multispectral image
    with rasterio.open(image_path) as src:
        data = src.read()  # Shape: (bands, height, width)

    # Extract Red and NIR bands
    red = data[2].astype(np.float32)   # Band B04
    nir = data[6].astype(np.float32)   # Band B08

    # Compute NDVI (small epsilon avoids division by zero)
    ndvi = (nir - red) / (nir + red + 1e-8)

    # Restrict values to the valid NDVI range
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi


# =============================================================================
# Save NDVI Color Map
# =============================================================================

def save_ndvi_map(ndvi: np.ndarray, output_path: str) -> None:
    """
    Save the computed NDVI as a color-coded image.

    Colour scheme (RdYlGn):

        Red     -> Low / Negative NDVI
        Yellow  -> Moderate Vegetation
        Green   -> Healthy Vegetation

    Parameters
    ----------
    ndvi : np.ndarray
        Computed NDVI image.

    output_path : str
        Destination path for saving the image.
    """

    fig, ax = plt.subplots(figsize=(6, 6))

    # Display NDVI using a vegetation colour map
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)

    # Add colour scale
    plt.colorbar(im, ax=ax, label="NDVI Value", shrink=0.85)

    ax.set_title("NDVI Map", fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [✓] NDVI map saved → {output_path}")


# =============================================================================
# Save NDVI Histogram
# =============================================================================

def save_ndvi_histogram(ndvi: np.ndarray, output_path: str) -> None:
    """
    Generate a histogram showing the distribution of NDVI values.

    The histogram is colour coded according to vegetation health
    categories and includes threshold markers.

    Parameters
    ----------
    ndvi : np.ndarray
        Computed NDVI image.

    output_path : str
        Destination path for saving the histogram.
    """

    # Convert image into a 1D array
    flat_ndvi = ndvi.flatten()

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot histogram
    n, bins, patches = ax.hist(
        flat_ndvi,
        bins=60,
        range=(-1.0, 1.0),
        color="steelblue",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )

    # Colour histogram bars based on NDVI category
    for patch, left_edge in zip(patches, bins[:-1]):

        if left_edge < 0.0:
            patch.set_facecolor("#d73027")     # Poor vegetation

        elif left_edge < 0.3:
            patch.set_facecolor("#fee08b")     # Moderate vegetation

        else:
            patch.set_facecolor("#1a9850")     # Healthy vegetation

    # Threshold indicators
    ax.axvline(
        x=0.0,
        color="#d73027",
        linestyle="--",
        linewidth=1.5,
        label="Poor threshold (0.0)",
    )

    ax.axvline(
        x=0.3,
        color="#1a9850",
        linestyle="--",
        linewidth=1.5,
        label="Healthy threshold (0.3)",
    )

    ax.set_xlabel("NDVI Value", fontsize=12)
    ax.set_ylabel("Number of Pixels", fontsize=12)
    ax.set_title("NDVI Value Distribution Histogram",
                 fontsize=14,
                 fontweight="bold")

    ax.legend(fontsize=10)
    ax.set_xlim(-1.0, 1.0)

    # Add vegetation category labels
    for x_pos, label, color in [
        (-0.5, "Non-vegetation", "#d73027"),
        (0.15, "Moderate", "#fee08b"),
        (0.65, "Healthy", "#1a9850"),
    ]:
        ax.text(
            x_pos,
            -0.06 * n.max(),
            label,
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  [✓] NDVI histogram saved → {output_path}")


# =============================================================================
# Calculate Vegetation Statistics
# =============================================================================

def calculate_vegetation_statistics(ndvi: np.ndarray) -> dict:
    """
    Compute vegetation health statistics from the NDVI image.

    Classification Rules

        NDVI < 0.0
            Poor / Non-vegetation

        0.0 <= NDVI < 0.3
            Moderate vegetation

        NDVI >= 0.3
            Healthy vegetation

    Returns
    -------
    dict
        Dictionary containing vegetation counts,
        percentages and descriptive statistics.
    """

    total_pixels = ndvi.size

    # Create vegetation masks
    poor_mask = ndvi < 0.0
    moderate_mask = (ndvi >= 0.0) & (ndvi < 0.3)
    healthy_mask = ndvi >= 0.3

    # Calculate percentages
    poor_pct = 100.0 * poor_mask.sum() / total_pixels
    moderate_pct = 100.0 * moderate_mask.sum() / total_pixels
    healthy_pct = 100.0 * healthy_mask.sum() / total_pixels

    # Store all statistics
    stats = {

        "total_pixels": total_pixels,

        # Pixel counts
        "poor_count": int(poor_mask.sum()),
        "moderate_count": int(moderate_mask.sum()),
        "healthy_count": int(healthy_mask.sum()),

        # Percentages
        "poor_pct": round(poor_pct, 2),
        "moderate_pct": round(moderate_pct, 2),
        "healthy_pct": round(healthy_pct, 2),

        # NDVI descriptive statistics
        "ndvi_mean": round(float(np.mean(ndvi)), 4),
        "ndvi_std": round(float(np.std(ndvi)), 4),
        "ndvi_min": round(float(np.min(ndvi)), 4),
        "ndvi_max": round(float(np.max(ndvi)), 4),
        "ndvi_median": round(float(np.median(ndvi)), 4),
    }

    return stats


# =============================================================================
# Save Vegetation Statistics Report
# =============================================================================

def save_vegetation_statistics(stats: dict, image_path: str, output_path: str) -> None:
    """
    Save vegetation statistics into a formatted text report.

    Parameters
    ----------
    stats : dict
        Statistics dictionary.

    image_path : str
        Original image path.

    output_path : str
        Destination report file.
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

    # Display report
    print()
    print(report)
    print(f"\n  [✓] Statistics saved → {output_path}")


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_advanced_ndvi_analysis(image_path: str) -> None:
    """
    Execute the complete NDVI analysis workflow.

    Pipeline

        1. Load multispectral image
        2. Compute NDVI
        3. Save NDVI colour map
        4. Generate NDVI histogram
        5. Calculate vegetation statistics
        6. Save analysis report
    """

    # Create output folder if it does not exist
    os.makedirs("results", exist_ok=True)

    print("\n" + "=" * 55)
    print("  Advanced NDVI Analysis  —  Member 3")
    print("=" * 55)
    print(f"  Input image : {image_path}\n")

    # Step 1
    print("  [1/4] Loading image and computing NDVI...")
    ndvi = load_and_compute_ndvi(image_path)
    print(f"        Image shape : {ndvi.shape}  (height × width)")

    # Step 2
    print("  [2/4] Saving NDVI map...")
    save_ndvi_map(ndvi, "results/ndvi_map.png")

    # Step 3
    print("  [3/4] Generating NDVI histogram...")
    save_ndvi_histogram(ndvi, "results/ndvi_histogram.png")

    # Step 4
    print("  [4/4] Calculating vegetation health statistics...")
    stats = calculate_vegetation_statistics(ndvi)
    save_vegetation_statistics(
        stats,
        image_path,
        "results/vegetation_statistics.txt",
    )

    print("\n  ✅ Analysis complete. All outputs saved in results/\n")


# =============================================================================
# Program Entry Point
# =============================================================================

if __name__ == "__main__":

    # Configure command-line argument parser
    parser = argparse.ArgumentParser(
        description="Advanced NDVI Analysis — Satellite Analysis Enhancements (Member 3)"
    )

    # Input image argument
    parser.add_argument(
        "--image",
        type=str,
        default="/content/drive/MyDrive/dataset/EuroSAT_MS/EuroSAT_MS/AnnualCrop/AnnualCrop_1.tif",
        help="Path to the multispectral .tif satellite image (EuroSAT format)",
    )

    args = parser.parse_args()

    # Execute complete analysis pipeline
    run_advanced_ndvi_analysis(args.image)