# Crop Type Classification and Vegetation Health Assessment Using Satellite Images

## Project Overview

This project combines **Computer Vision**, **Image Processing**, **Deep Learning**, and **Remote Sensing** techniques to classify satellite imagery and assess vegetation health using Sentinel-2 multispectral data.

The system performs:

- Land-cover classification using a Convolutional Neural Network (CNN)
- NDVI (Normalized Difference Vegetation Index) computation
- Vegetation health assessment
- Image enhancement and preprocessing
- Image segmentation
- Morphological processing
- Edge detection and contour analysis
- Region-based vegetation analysis
- Interactive visualization through a Streamlit dashboard

---

## Features

### RGB Satellite Image Analysis

- Satellite image upload
- Image resizing
- Gaussian filtering
- Edge detection
- Histogram analysis
- CNN-based classification
- Top-5 prediction analysis
- Probability distribution visualization

### Multispectral Vegetation Analysis

- Sentinel-2 band extraction
- NDVI computation
- Histogram equalization
- Gaussian smoothing
- Otsu thresholding
- Adaptive thresholding
- Morphological opening
- Morphological closing
- Canny edge detection
- Contour detection
- Region analysis
- Vegetation health assessment

### Interactive Dashboard

- Dataset exploration
- CNN training interface
- Training visualization
- Prediction interface
- NDVI analysis interface
- Automated report generation

---

# Dataset

## EuroSAT RGB Dataset

The RGB version of the EuroSAT dataset is used for CNN classification.

### Dataset Information

| Property   | Value   |
| ---------- | ------- |
| Images     | 27,000  |
| Classes    | 10      |
| Resolution | 64 × 64 |
| Format     | RGB     |

### Classes

- AnnualCrop
- Forest
- HerbaceousVegetation
- Highway
- Industrial
- Pasture
- PermanentCrop
- Residential
- River
- SeaLake

---

## EuroSAT Multispectral Dataset

Used for NDVI and vegetation health analysis.

### Characteristics

- 13 Sentinel-2 spectral bands
- TIFF format
- 64 × 64 pixels

Used for:

- NDVI generation
- Vegetation analysis
- Segmentation
- Morphological processing
- Region extraction

---

# Technologies Used

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras

## Computer Vision

- OpenCV

## Scientific Computing

- NumPy
- Rasterio
- Scikit-Learn

## Visualization

- Matplotlib
- Seaborn

## Dashboard

- Streamlit

---

# Project Structure

```text
crop-type-classification-cv/
│
├── app.py
│
├── models/
│   ├── eurosat_model.keras
│   ├── eurosat_labels.npy
│   ├── crop_model.keras
│   ├── labels.npy
│   ├── eurosat_accuracy_plot.png
│   └── eurosat_loss_plot.png
│
├── results/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── ndvi_map.png
│   ├── false_color.png
│   ├── vegetation_health_map.png
│   ├── ndvi_histogram.png
│   ├── vegetation_statistics.txt
│   ├── 01_ndvi.png
│   ├── 02_normalized_ndvi.png
│   ├── 03_histogram_equalized.png
│   ├── 04_gaussian_blur.png
│   ├── 05_otsu_segmentation.png
│   ├── 06_adaptive_segmentation.png
│   ├── 07_combined_mask.png
│   ├── 08_opening.png
│   ├── 09_closing.png
│   ├── 10_edges.png
│   ├── 11_contours.png
│   └── 12_health_map.png
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── ndvi.py
│   ├── advanced_ndvi.py
│   ├── ndvi_demo.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── satellite_preprocessing.py
│   └── satellite_train.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Computer Vision Pipeline

## RGB Classification Pipeline

```text
RAW RGB SATELLITE IMAGE
            ↓
IMAGE RESIZING (64×64)
            ↓
GAUSSIAN FILTERING
            ↓
EDGE DETECTION
            ↓
NORMALIZATION
            ↓
TENSOR CONVERSION
            ↓
CNN CLASSIFICATION
            ↓
LAND COVER PREDICTION
```

---

## Multispectral NDVI Pipeline

```text
MULTISPECTRAL IMAGE (.tif)
            ↓
RED (B04) + NIR (B08)
            ↓
NDVI COMPUTATION
            ↓
NORMALIZATION
            ↓
HISTOGRAM EQUALIZATION
            ↓
GAUSSIAN FILTERING
            ↓
OTSU THRESHOLDING
            ↓
ADAPTIVE THRESHOLDING
            ↓
MASK COMBINATION
            ↓
MORPHOLOGICAL OPENING
            ↓
MORPHOLOGICAL CLOSING
            ↓
CANNY EDGE DETECTION
            ↓
CONTOUR DETECTION
            ↓
REGION ANALYSIS
            ↓
VEGETATION HEALTH ASSESSMENT
```

---

# Classical Image Processing Techniques

## Image Enhancement

- Image Resizing
- Image Normalization
- Histogram Equalization
- Gaussian Filtering

## Image Segmentation

- Otsu Thresholding
- Adaptive Thresholding

## Morphological Processing

- Morphological Opening
- Morphological Closing

## Feature Extraction

- Canny Edge Detection
- Contour Detection

## Region Analysis

- Region Area Calculation
- Region Perimeter Analysis
- Vegetation Coverage Analysis
- Vegetation Density Analysis

---

# Computer Vision Techniques

## Deep Learning

- Convolutional Neural Networks (CNN)
- Feature Learning
- Pattern Recognition
- Multi-Class Classification

## Classical Computer Vision

- Edge Detection
- Contour Extraction
- Region-Based Analysis
- Shape Analysis

---

# NDVI Computation

NDVI is calculated using Sentinel-2 spectral bands:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

Where:

```text
NIR = Band 8
Red = Band 4
```

### Interpretation

| NDVI Range | Interpretation               |
| ---------- | ---------------------------- |
| < 0.0      | Water / Non-Vegetation       |
| 0.0 - 0.3  | Sparse / Stressed Vegetation |
| 0.3 - 0.5  | Moderate Vegetation          |
| > 0.5      | Healthy Vegetation           |

---

# Vegetation Metrics Generated

The system automatically computes:

- Average NDVI
- NDVI Standard Deviation
- Maximum NDVI
- Minimum NDVI
- Vegetation Coverage (%)
- Vegetation Density (%)
- Stress Percentage (%)
- Number of Vegetation Regions
- Largest Region Area
- Average Region Area
- Total Region Area
- Average Region Perimeter
- Edge Pixel Count

---

# CNN Architecture

```text
Input Image (64×64×3)
        ↓
Conv2D (32 Filters)
        ↓
MaxPooling2D
        ↓
Conv2D (64 Filters)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128)
        ↓
Dense (10 Classes)
        ↓
Softmax Output
```

---

# Training Results

| Metric              | Value       |
| ------------------- | ----------- |
| Dataset             | EuroSAT RGB |
| Classes             | 10          |
| Images              | 27,000      |
| Epochs              | 10          |
| Validation Accuracy | 82.19%      |

---

# Generated Pipeline Images

The system automatically saves all processing stages:

| Stage                  | Output                       |
| ---------------------- | ---------------------------- |
| NDVI                   | 01_ndvi.png                  |
| Normalized NDVI        | 02_normalized_ndvi.png       |
| Histogram Equalization | 03_histogram_equalized.png   |
| Gaussian Blur          | 04_gaussian_blur.png         |
| Otsu Segmentation      | 05_otsu_segmentation.png     |
| Adaptive Segmentation  | 06_adaptive_segmentation.png |
| Combined Mask          | 07_combined_mask.png         |
| Morphological Opening  | 08_opening.png               |
| Morphological Closing  | 09_closing.png               |
| Edge Detection         | 10_edges.png                 |
| Contour Detection      | 11_contours.png              |
| Health Map             | 12_health_map.png            |

---

# Example Output

```text
SATELLITE IMAGE ANALYSIS REPORT

Predicted Class      : Residential
Confidence           : 99.93%

Average NDVI         : 0.6638
NDVI Std Dev         : 0.2776
Maximum NDVI         : 0.8759
Minimum NDVI         : -0.0616

Health Status        : Healthy Vegetation

Vegetation Regions   : 3

Largest Region Area  : 1840.0
Average Region Area  : 842.67

Vegetation Coverage  : 64.23%
Vegetation Density   : 75.32%
Stress Percentage    : 20.61%

Edge Pixels          : 479
```

---

# Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Train Model

```bash
python src/train.py
```

---

## Evaluate Model

```bash
python src/evaluate.py
```

---

## Run Prediction

```bash
python src/predict.py
```

---

## Run NDVI Analysis

```bash
python src/advanced_ndvi.py
```

---

## Launch Dashboard

```bash
streamlit run app.py
```

---

# Applications

- Precision Agriculture
- Crop Monitoring
- Vegetation Health Assessment
- Land Cover Classification
- Environmental Monitoring
- Remote Sensing Analysis
- Smart Farming Systems
- Satellite-Based Decision Support

---

# Future Improvements

- ResNet / EfficientNet architectures
- Real-time Sentinel-2 integration
- GIS integration
- Time-series vegetation monitoring
- Crop disease detection
- Geospatial visualization dashboard
- Object detection for agricultural fields

---

# Conclusion

This project demonstrates the integration of:

- Deep Learning Classification
- Remote Sensing Analysis
- NDVI-Based Vegetation Assessment
- Image Enhancement
- Image Segmentation
- Morphological Processing
- Edge Detection
- Contour Analysis
- Region-Based Computer Vision

The system successfully combines classical Image Processing techniques with modern Deep Learning approaches to perform satellite image classification and vegetation health assessment.

---

# Authors

- Charith Manujaya
- Tharika Akurana
- Thurunu Pabasara
- Team Members

---

# License

This project is developed for academic and educational purposes.
