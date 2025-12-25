import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# 1. PATH SETTINGS (RELATIVE PATHS)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "Dataset", "Images")
MASK_DIR  = os.path.join(BASE_DIR, "Dataset", "Masks")
SAVE_DIR  = os.path.join(BASE_DIR, "results", "baseline")

os.makedirs(SAVE_DIR, exist_ok=True)



# 2. READ IMAGES / MASKS

def read_rgb(path: str, verbose: bool = True):
    #Read an image as RGB. Return None if it cannot be read.
    img = cv2.imread(path)
    if img is None:
        if verbose:
            print(f"[WARNING] Could not read image: {path}")
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_gray(path: str, verbose: bool = True):
    #Read a mask as grayscale. Return None if it cannot be read.
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        if verbose:
            print(f"[WARNING] Could not read mask: {path}")
        return None
    return m



# 3. SEGMENTATION PIPELINE

def segmentation_pipeline(img: np.ndarray):
    """
    Median Blur -> HSV -> S channel equalization -> Otsu -> optional inversion -> open/close
    Returns: (pred_mask, s_eq, otsu_threshold)
    """
    # 1) Median filter (for impulse noise; applied to the IMAGE, not the GT mask)
    img_med = cv2.medianBlur(img, 5)

    # 2) Convert to HSV
    hsv = cv2.cvtColor(img_med, cv2.COLOR_RGB2HSV)

    # 3) Equalize S channel
    s_channel = hsv[:, :, 1]
    s_eq = cv2.equalizeHist(s_channel)

    # 4) Otsu threshold
    otsu_val, mask = cv2.threshold(
        s_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 5) Invert if the white area is too large
    white_ratio = np.sum(mask == 255) / mask.size
    if white_ratio > 0.5:
        mask = cv2.bitwise_not(mask)

    # 6) Morphological post-processing
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask, s_eq, otsu_val



# 4. METRICS

def compute_metrics(gt: np.ndarray, pred: np.ndarray):
    #Compute Accuracy, Precision, Recall, F1, IoU for binary masks.
    y_true = gt.flatten() > 127
    y_pred = pred.flatten() > 127

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    iou = intersection / union if union != 0 else 0.0

    return acc, prec, rec, f1, iou



# 5. VISUALIZATION / SAVING

def save_detailed_visuals(img, s_eq, gt, pred, otsu_val, name: str):
    """
    Save:
      - 4-panel figure (original, GT, prediction, error map)
      - histogram of equalized S channel with Otsu threshold
    Always saves as .png
    """
    base = os.path.splitext(name)[0]

    error_map = np.logical_xor(gt > 127, pred > 127).astype(np.uint8) * 255

    # 4-panel overview
    plt.figure(figsize=(16, 4))
    titles = ["Original", "Ground Truth", "Otsu Prediction", "Error Map"]
    images = [img, gt, pred, error_map]
    cmaps  = [None, "gray", "gray", "hot"]

    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(images[i], cmap=cmaps[i])
        plt.title(titles[i])
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"segmentation_{base}.png"), dpi=150)
    plt.close()

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(s_eq.ravel(), 256, [0, 256], alpha=0.7, color="gray")
    plt.axvline(x=otsu_val, color="red", linestyle="--", label=f"Otsu = {otsu_val:.0f}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"histogram_{base}.png"), dpi=150)
    plt.close()

def save_metrics_barchart(avg_metrics: dict):
    """Save a bar chart of average metrics to results/overall_performance.png."""
    labels = list(avg_metrics.keys())
    values = [avg_metrics[k] * 100 for k in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Performance (%)")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "overall_performance.png"), dpi=150)
    plt.close()



# 6. DATASET PATHS (SIMPLE, ASSUMES MATCHING ORDER/NAMES)

def get_dataset_paths(verbose: bool = True):
    """
    Returns sorted image & mask paths.
    Assumes filenames correspond in the same sorted order.
    """
    images = sorted(glob(os.path.join(IMAGE_DIR, "*")))
    masks  = sorted(glob(os.path.join(MASK_DIR, "*")))

    if verbose:
        print(f"[INFO] Found {len(images)} images and {len(masks)} masks.")
        if len(images) == 0 or len(masks) == 0:
            print("[WARNING] Dataset folders look empty or paths are incorrect.")
        elif len(images) != len(masks):
            print("[WARNING] Image/mask counts differ. Check the dataset folders.")

    return images, masks

def ensure_mask_size_like_image(gt: np.ndarray, img: np.ndarray):

    if gt is None or img is None:
        return gt

    if img.shape[:2] != gt.shape:
        gt = cv2.resize(
            gt,
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
    return gt
