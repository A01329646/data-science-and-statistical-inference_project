# PCA Dataset Analysis - Detailed Documentation

## Overview
This script performs **Principal Component Analysis (PCA)** on facial image data to identify the main patterns of variation in an unsupervised manner. PCA is a dimensionality reduction technique that transforms high-dimensional image data into a lower-dimensional representation while preserving maximum variance.

---

## Statistical Theory: Principal Component Analysis

### What is PCA?

**Principal Component Analysis** is an unsupervised statistical method that:
1. Identifies directions (principal components) of maximum variance in data
2. Projects data onto these orthogonal directions
3. Reduces dimensionality while retaining most information

### Mathematical Foundation

**Goal:** Find orthogonal directions that maximize variance

Given standardized data matrix **X** (n × p):
- n = number of samples (images)
- p = number of features (pixels)

**Steps:**
1. **Compute covariance matrix:** Σ = (1/n) X^T X
2. **Eigendecomposition:** Σ = V Λ V^T
   - V = eigenvectors (principal components)
   - Λ = eigenvalues (variance explained)
3. **Project data:** Z = X V_k (keep k components)

**Key Properties:**
- PCs are orthogonal (uncorrelated)
- PCs are ordered by variance explained
- Total variance = Σ(eigenvalues)

### Why Use PCA for Face Images?

1. **Dimensionality reduction:** 178×218 image = 38,804 features → 50 components
2. **Noise reduction:** Lower components often capture noise
3. **Visualization:** Can plot first 2-3 components
4. **Computational efficiency:** Smaller feature space for downstream analysis

---

## Line-by-Line Code Explanation

### Header and Imports (Lines 1-11)

```python
# ==============================
# PCA Transfer Between Datasets (with Gender Labels)
# ==============================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
```

**Imports explained:**
- **os:** File system operations
- **numpy:** Numerical computing (arrays, linear algebra)
- **matplotlib.pyplot:** Plotting and visualization
- **PCA:** Implements PCA algorithm from scikit-learn
- **StandardScaler:** Standardizes features (mean=0, std=1)
- **joblib:** Efficient serialization for saving models
- **PIL.Image:** Image loading and processing

**Theory:** Scikit-learn's PCA uses efficient SVD (Singular Value Decomposition) for eigendecomposition.

---

### Parameters (Lines 13-18)

```python
TRAIN_DIR = "dataset/train"      # Folder containing training images
VAL_DIR   = "dataset/validation" # Folder containing validation images
N_COMPONENTS = 50                # Number of PCA components to keep
```

**Line 15:** `TRAIN_DIR` - Training data location (fit PCA on this)
**Line 16:** `VAL_DIR` - Validation data location (transform using fitted PCA)
**Line 17:** `N_COMPONENTS = 50` - Keep first 50 principal components

**Why 50 components?**
- Balances information retention with dimensionality reduction
- Typically captures 80-95% of total variance
- Reduces 38,804 features to 50 (99.87% reduction)

**Theory:** Number of components is a hyperparameter. Too few loses information, too many keeps noise.

---

### Function: load_images_and_labels (Lines 20-44)

```python
def load_images_and_labels(folder, max_images=None):
    data, labels = [], []
    files = sorted(os.listdir(folder))
    if max_images:
        files = files[:max_images]
    for filename in files:
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path)
            arr = np.array(img, dtype=np.float32) / 255.0  # normalize to [0,1]
            data.append(arr.flatten())

            # Extract gender from filename
            if filename.lower().startswith("male"):
                labels.append("male")
            elif filename.lower().startswith("female"):
                labels.append("female")
            else:
                labels.append("unknown")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return np.array(data), np.array(labels)
```

**Line 23:** `sorted(os.listdir(folder))` - Lists files in alphabetical order (ensures reproducibility)

**Line 30:** `arr = np.array(img, dtype=np.float32) / 255.0`
- Converts PIL image to numpy array
- `dtype=np.float32` - Uses 32-bit floats (memory efficient, sufficient precision)
- `/ 255.0` - **Normalizes pixel values from [0, 255] to [0, 1]**

**Why normalize?**
- Prevents numerical instability in algorithms
- Puts all features on similar scale
- Standard practice in image processing

**Line 31:** `arr.flatten()` - Converts 2D image (178×218) to 1D vector (38,804)

**Theory:** 
- Flattening treats each pixel as an independent feature
- Position information is preserved in the ordering
- Alternative: Could use 2D convolutions, but PCA works on vectors

**Lines 34-39:** Extract gender labels from filenames
- Male images start with "male"
- Female images start with "female"
- Others marked as "unknown"

**Note:** Labels aren't used by PCA (unsupervised), but stored for visualization

---

### Load Datasets (Lines 46-54)

```python
print("Loading images...")
X_train, y_train = load_images_and_labels(TRAIN_DIR, max_images=1000)
X_val,   y_val   = load_images_and_labels(VAL_DIR,   max_images=500)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
```

**Line 49:** `max_images=1000` - Loads up to 1000 training images
**Line 50:** `max_images=500` - Loads up to 500 validation images

**Why limit images?**
- Computational efficiency (PCA on large datasets is expensive)
- Sufficient for statistical analysis
- Faster prototyping

**Expected shapes:**
- `X_train`: (1000, 38804) - 1000 samples, 38,804 features
- `X_val`: (500, 38804) - 500 samples, 38,804 features

---

### Standardization (Lines 56-61)

```python
scaler = StandardScaler(with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)  # use same mean/std
```

**Line 59:** `StandardScaler(with_std=True)` - Creates standardizer
- Subtracts mean (centering)
- Divides by standard deviation (scaling)

**Line 60:** `fit_transform()` - Fits on training data AND transforms it
- Computes mean and std from training data
- Applies transformation: z = (x - μ) / σ

**Line 61:** `transform()` - Transforms validation using TRAINING statistics
- **Critical:** Uses training mean/std, not validation mean/std
- Prevents data leakage
- Ensures consistent preprocessing

**Mathematical formula:**
```
x_scaled = (x - μ_train) / σ_train
```

**Why standardize before PCA?**
1. **PCA is scale-sensitive:** Features with larger variance dominate
2. **All pixels on same scale:** Each pixel contributes equally
3. **Numerical stability:** Prevents overflow/underflow
4. **Improves convergence:** Faster eigendecomposition

**Theory:** Standardization makes covariance matrix = correlation matrix

---

### Fit PCA (Lines 63-70)

```python
pca = PCA(n_components=N_COMPONENTS, random_state=42)
pca.fit(X_train_scaled)

print(f"Explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
print(f"Total variance explained by {N_COMPONENTS} components: {np.sum(pca.explained_variance_ratio_):.2f}")
```

**Line 66:** `PCA(n_components=50, random_state=42)`
- `n_components=50` - Keep 50 principal components
- `random_state=42` - Seed for reproducibility (in randomized solvers)

**Line 67:** `pca.fit(X_train_scaled)` - Fits PCA model
- Computes covariance matrix: Σ = X^T X / n
- Performs eigendecomposition (via SVD)
- Sorts eigenvectors by eigenvalues (descending)
- Stores top 50 components

**What PCA computes:**
- `pca.components_` - Principal component vectors (50 × 38,804)
- `pca.explained_variance_` - Variance of each PC (50 values)
- `pca.explained_variance_ratio_` - Proportion of variance (sums to ≤ 1)
- `pca.mean_` - Feature means (for centering)

**Line 69:** Shows variance explained by first 10 PCs

**Theory:**
- First PC captures most variance
- Subsequent PCs capture decreasing variance
- Variance ratio: λ_i / Σ(λ_j)

**Typical output:**
- PC1 might explain 15-20% of variance
- First 50 PCs might explain 85-95% of total variance

---

### Transform Data (Lines 72-75)

```python
X_train_pca = pca.transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)
```

**What happens:**
- Projects data onto principal components
- Transforms from 38,804D to 50D

**Mathematical operation:**
```
X_pca = X_scaled @ V_k
```
Where:
- X_scaled: (n × 38,804) standardized data
- V_k: (38,804 × 50) first 50 eigenvectors
- X_pca: (n × 50) transformed data

**Result:**
- `X_train_pca`: (1000, 50)
- `X_val_pca`: (500, 50)

**Important:** Same transformation applied to both train and validation using training PCA

---

### Visualization Function (Lines 77-90)

```python
def plot_pca_projection(X_pca, y_labels, title):
    plt.figure(figsize=(8,6))
    for gender, color in zip(["male", "female"], ["blue", "red"]):
        mask = (y_labels == gender)
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                    alpha=0.5, s=10, label=gender.capitalize(), color=color)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
```

**Purpose:** Visualizes first 2 principal components colored by gender

**Line 83:** `mask = (y_labels == gender)` - Boolean mask for filtering

**Line 84-85:** Scatter plot of PC1 vs PC2
- `X_pca[mask, 0]` - First component (PC1) for this gender
- `X_pca[mask, 1]` - Second component (PC2) for this gender
- `alpha=0.5` - 50% transparency (shows overlap)
- `s=10` - Small point size

**What to look for in plot:**
- **Separation:** Do male/female clusters separate?
- **Overlap:** How much do clusters overlap?
- **Outliers:** Are there unusual points?

**Theory:** 
- If PCA separates genders well → gender info in top 2 PCs
- If not → gender info in other PCs or not linearly separable
- PCA is unsupervised, so separation not guaranteed

---

### Plot PCA Projections (Lines 92-96)

```python
print("Plotting PCA projection for training set...")
plot_pca_projection(X_train_pca, y_train, "Training Set: PCA Projection by Gender")

print("Plotting PCA projection for validation set...")
plot_pca_projection(X_val_pca, y_val, "Validation Set: PCA Projection by Gender")
```

**What it does:** Creates 2D scatter plots for both datasets

**Expected behavior:**
- Training and validation plots should look similar
- If very different → potential issues (overfitting, data shift)

**Interpretation:**
- **Good separation:** Gender is major source of variance
- **Poor separation:** Gender variation is subtle, higher PCs needed
- **For face images:** Usually moderate separation, overlap common

---

### Save Models (Lines 98-106)

```python
os.makedirs("models", exist_ok=True)
joblib.dump({"scaler": scaler, "pca": pca}, "models/pca_model.pkl")
print("✅ PCA model and scaler saved to models/pca_model.pkl")
```

**Line 101:** `os.makedirs("models", exist_ok=True)` 
- Creates 'models' directory if it doesn't exist
- `exist_ok=True` - No error if directory already exists

**Line 102:** `joblib.dump({...}, "models/pca_model.pkl")`
- Saves both scaler and PCA model as dictionary
- `joblib` is efficient for numpy arrays (better than pickle)
- `.pkl` extension indicates pickle format

**Why save both scaler and PCA?**
- **Scaler:** Needed to preprocess new data consistently
- **PCA:** Needed to transform new data to PC space
- **Together:** Complete preprocessing pipeline

**Usage later:**
```python
loaded = joblib.load("models/pca_model.pkl")
scaler = loaded["scaler"]
pca = loaded["pca"]
```

---

## Statistical Interpretation

### Variance Explained

**What it means:**
- High variance directions = major patterns in data
- PC1: Might capture overall brightness/contrast
- PC2-3: Might capture facial shape variations
- Later PCs: Finer details, potentially noise

**Cumulative variance:**
If first 10 PCs explain 60% of variance:
- 60% of variability captured in 10 numbers (vs 38,804)
- Massive compression with moderate information loss

### PCA Limitations for Classification

**PCA is unsupervised:**
- Doesn't use gender labels
- Maximizes variance, not class separation
- Might capture variations irrelevant to gender

**Why use PCA then?**
1. Reduces dimensionality (computational efficiency)
2. Removes noise (regularization effect)
3. Decorrelates features (independence assumption)
4. Provides interpretable components

**Alternative:** LDA (Linear Discriminant Analysis) - supervised, maximizes class separation

---

## Practical Considerations

### Memory Requirements

**Training data:** 1000 × 38,804 × 4 bytes = ~155 MB
**PCA components:** 50 × 38,804 × 8 bytes = ~15 MB
**Transformed data:** 1000 × 50 × 8 bytes = ~0.4 MB

**Speedup:** Working with 50 features vs 38,804 is ~776× faster

### Choosing Number of Components

**Rule of thumb:**
- Keep components explaining 95% of variance
- Or use elbow method (plot cumulative variance)
- Or cross-validation for downstream task

**For this project:** 50 components chosen empirically

### Reproducibility

**Sources of randomness:**
- `random_state=42` in PCA (for randomized SVD solver)
- `sorted()` for file loading
- Fixing these ensures reproducibility

---

## Integration with Project Pipeline

**Position in pipeline:**
```
image_converter.py → PCA_dataset.py ← You are here → run_regression_analysis.py
                            ↓
                     lda_dataset.py
```

**PCA provides:**
- Dimensionality reduction for LDA
- Feature decorrelation
- Noise reduction
- PC scores as predictors in regression

**Output used by:**
- `run_regression_analysis.py` - PC scores as independent variables (X)
- Visualization and interpretation

---

## Common Questions

**Q: Why 50 components specifically?**
A: Empirical choice balancing dimensionality reduction and information retention. Typically explains 85-95% variance.

**Q: Why standardize if images already normalized?**
A: Normalization scales [0,255]→[0,1]. Standardization makes mean=0, std=1 across images. Different purposes.

**Q: Can PCA classify gender?**
A: No, PCA is unsupervised. But PC scores can be inputs to classifiers (like LDA or logistic regression).

**Q: What do principal components represent?**
A: Linear combinations of original pixels. PC1 might be "average face", PC2 might be "face shape", etc.

**Q: Why not use all components?**
A: Computational cost, overfitting risk, noise in later components.

---

## Expected Output

**Console output:**
```
Loading images...
Training set shape: (1000, 38804)
Validation set shape: (500, 38804)
Explained variance ratio (first 10): [0.15 0.08 0.05 0.04 0.03 ...]
Total variance explained by 50 components: 0.89
Plotting PCA projection for training set...
Plotting PCA projection for validation set...
✅ PCA model and scaler saved to models/pca_model.pkl
```

**Generated files:**
- `models/pca_model.pkl` - Saved PCA model and scaler

**Plots:**
- 2D scatter plots showing PC1 vs PC2 colored by gender
