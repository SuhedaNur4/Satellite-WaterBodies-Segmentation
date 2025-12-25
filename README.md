# Satellite Water Bodies Segmentation  
**Baseline : Classical Image Processing Approach**

This project focuses on **water body segmentation from satellite images** using **classical image processing techniques**, without employing deep learning models.  
The main objective of this baseline is to establish a **reference solution** that can later be improved and quantitatively compared.

---

## Dataset

The dataset used in this project is **Satellite Images of Water Bodies**, obtained from Kaggle:

🔗 https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies

- RGB satellite images with corresponding binary ground truth masks  
- Image–mask pairs share the same filename  
- The dataset is **not included** in this repository due to size limitations  

Expected directory structure after downloading the dataset:
Dataset/
├── Images/
└── Masks/

---

## Baseline Methodology

The baseline segmentation pipeline consists of the following stages:

---

### 1. Image Acquisition (Data Reading)

- Satellite images are read using OpenCV
- Images are converted from **BGR to RGB** format
- Ground truth masks are loaded in grayscale format

---

### 2. Pre-processing: Noise Reduction

- A **Median Filter** is applied to the RGB image
- Purpose:
  - Reduce impulse noise
  - Preserve object boundaries more effectively than linear filters

---

### 3. Image Enhancement: Contrast Improvement

- The RGB image is converted to the **HSV color space**
- The **Saturation (S) channel** is extracted
- **Histogram Equalization** is applied to the S channel to:
  - Improve contrast
  - Enhance separation between water and non-water regions

---

### 4. Color Space Transformation

- RGB → HSV conversion is used to separate:
  - Hue (color information)
  - Saturation (color intensity)
  - Value (brightness)

This transformation provides better robustness against illumination variations compared to raw RGB values.

---

### 5. Segmentation: Thresholding

- **Otsu’s global thresholding** method is applied to the enhanced S channel
- This step automatically determines an optimal threshold
- A **binary segmentation mask** is generated:
  - White pixels represent water regions
  - Black pixels represent non-water regions

---

### 6. Morphological Post-processing

To refine the binary mask, morphological operations are applied:

- **Opening**: removes small isolated noise
- **Closing**: fills small holes within detected water regions

These steps improve the spatial consistency of the segmentation output.

---

### 7. Evaluation Metrics

The segmentation results are evaluated against ground truth masks using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  
- Intersection over Union (IoU)  

Average performance values are computed over the dataset.

---

### 8. Visualization and Results

For qualitative and quantitative analysis, the following outputs are generated:

- Original image  
- Ground truth mask  
- Predicted segmentation mask  
- Error map  
- Histogram with Otsu threshold  
- Overall performance bar chart  

All baseline outputs are saved under:
results/baseline/

---

## Purpose of the Baseline

This baseline implementation serves as:

- A reference point for future improvements  
- A demonstration of classical image processing capabilities  
- A foundation for enhanced segmentation strategies  

---

## Future Work

An improved version of this pipeline will be developed by enhancing feature extraction and segmentation strategies, while keeping the dataset and evaluation protocol unchanged.


