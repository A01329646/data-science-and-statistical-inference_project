# LDA Gender Classification - Detailed Documentation

## Overview
This script implements **Linear Discriminant Analysis (LDA)** for binary gender classification using facial images. Unlike PCA (unsupervised), LDA is a **supervised** method that uses class labels to find the linear combination of features that maximizes the separation between classes.

---

## Statistical Theory: Linear Discriminant Analysis

### What is LDA?

**Linear Discriminant Analysis** (Fisher's Linear Discriminant) is a supervised dimensionality reduction and classification method that:
1. Maximizes between-class variance
2. Minimizes within-class variance
3. Finds optimal linear decision boundary

### Mathematical Foundation

**Goal:** Find direction **w** that maximizes class separation

Given two classes (Male=0, Female=1):
- **μ₁**, **μ₂** = class means
- **Σ₁**, **Σ₂** = class covariances
- **Σ_w** = within-class scatter = Σ₁ + Σ₂
- **Σ_b** = between-class scatter = (μ₁ - μ₂)(μ₁ - μ₂)ᵀ

**Fisher's criterion:**
```
w = argmax J(w) = (w^T Σ_b w) / (w^T Σ_w w)
```

**Solution (for 2 classes):**
```
w = Σ_w^(-1) (μ₁ - μ₂)
```

**Projection (LD1 score):**
```
y = w^T x
```

### LDA vs PCA

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Type** | Unsupervised | Supervised |
| **Goal** | Maximize variance | Maximize class separation |
| **Uses labels?** | No | Yes |
| **Max components** | min(n, p) | C - 1 (C = classes) |
| **For 2 classes** | Many components | 1 component |

**For binary classification:** LDA produces exactly **1 discriminant** (LD1)

---

## Line-by-Line Code Explanation

### Header and Imports (Lines 1-15)

```python
# ==============================
# LDA for Gender Classification (Statistical Approach)
# ==============================

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
from PIL import Image
```

**Key imports:**
- **LinearDiscriminantAnalysis:** sklearn's LDA implementation
- **StandardScaler:** Data standardization (critical for LDA)
- **scipy.stats:** Statistical tests (t-test, Shapiro-Wilk)
- **joblib:** Model persistence
- **matplotlib:** Visualization

**Theory:** Scikit-learn's LDA uses efficient matrix decomposition (SVD-based)

---

### Parameters (Lines 17-22)

```python
TRAIN_DIR = "dataset/train"      # Folder containing training images
VAL_DIR   = "dataset/validation" # Folder containing validation images
N_COMPONENTS = 1                 # LDA with 2 classes can have max 1 component
```

**Line 22:** `N_COMPONENTS = 1` 
- **Fundamental constraint:** For C classes, max components = C - 1
- **Binary classification:** 2 classes → max 1 component
- This single component (LD1) is the optimal separating direction

**Mathematical reason:**
- Between-class scatter matrix rank = C - 1
- Cannot extract more discriminants than rank allows

---

### Function: load_images_and_labels (Lines 24-54)

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
                labels.append(0)  # Male = 0
            elif filename.lower().startswith("female"):
                labels.append(1)  # Female = 1
            else:
                labels.append(-1)  # Unknown
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    
    # Filter out unknown labels
    valid_mask = labels != -1
    return data[valid_mask], labels[valid_mask]
```

**Key differences from PCA version:**
- **Line 40:** Labels are numeric (0, 1) not strings ("male", "female")
- **Line 42:** Labels = -1 for unknown gender
- **Lines 50-52:** Filters out unknown labels (LDA requires valid labels)

**Why numeric labels?**
- LDA algorithms expect numeric class identifiers
- Simplifies mathematical operations
- Standard convention: 0 = negative class, 1 = positive class

---

### Load Datasets (Lines 56-68)

```python
print("=" * 60)
print("LOADING DATA")
print("=" * 60)
X_train, y_train = load_images_and_labels(TRAIN_DIR, max_images=1000)
X_val,   y_val   = load_images_and_labels(VAL_DIR,   max_images=500)

print(f"Training set shape: {X_train.shape}")
print(f"  - Male samples: {np.sum(y_train == 0)}")
print(f"  - Female samples: {np.sum(y_train == 1)}")
print(f"Validation set shape: {X_val.shape}")
print(f"  - Male samples: {np.sum(y_val == 0)}")
print(f"  - Female samples: {np.sum(y_val == 1)}")
```

**Why count samples per class?**
- **Class imbalance check:** LDA assumes balanced classes
- **Statistical power:** Need sufficient samples per class
- **Validation:** Ensures proper label extraction

**Ideal:** Roughly equal samples per class (balanced dataset)

---

### Standardization (Lines 70-78)

```python
print("\n" + "=" * 60)
print("STANDARDIZING DATA")
print("=" * 60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
print("✓ Data standardized using training set statistics")
```

**Why standardization is CRITICAL for LDA:**
1. **Scale sensitivity:** LDA uses covariance matrices
2. **Feature equality:** Each pixel should contribute equally
3. **Numerical stability:** Prevents singular covariance matrices
4. **Assumption:** LDA assumes features on comparable scales

**Mathematical impact:**
- Without standardization: High-variance features dominate
- With standardization: All features have variance = 1

**Formula:** x_scaled = (x - μ) / σ

---

### Fit LDA Model (Lines 80-93)

```python
print("\n" + "=" * 60)
print("FITTING LDA MODEL")
print("=" * 60)
lda = LinearDiscriminantAnalysis(n_components=N_COMPONENTS)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)

print(f"✓ LDA fitted with {N_COMPONENTS} component(s)")
print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
print(f"LDA means for each class:")
print(f"  - Male (class 0): {lda.means_[0][:5]}... (first 5 features)")
print(f"  - Female (class 1): {lda.means_[1][:5]}... (first 5 features)")
```

**Line 85:** `lda.fit_transform(X_train_scaled, y_train)`
- **fit:** Computes LDA parameters from training data
- **transform:** Projects training data onto LD1

**What LDA computes:**

1. **Class means** (`lda.means_`):
   - μ_male: Mean feature vector for males (38,804 dimensions)
   - μ_female: Mean feature vector for females (38,804 dimensions)

2. **Covariance matrices:**
   - Within-class scatter: Σ_w = Σ_male + Σ_female
   - Between-class scatter: Σ_b = (μ₁ - μ₂)(μ₁ - μ₂)ᵀ

3. **Discriminant vector** (`lda.scalings_`):
   - w = Σ_w^(-1) (μ_male - μ_female)
   - This is the "optimal" direction for separation

4. **Explained variance ratio:**
   - How much of discriminative information is captured
   - For 1 component: typically 1.0 (100%) for binary case

**Output:**
- `X_train_lda`: (1000, 1) - LD1 scores for training samples
- Each score represents position along discriminant direction

---

### Transform Validation Set (Lines 95-97)

```python
X_val_lda = lda.transform(X_val_scaled)
print(f"✓ Validation set transformed to LDA space")
```

**Critical:** Uses the SAME LDA fitted on training data
- Same discriminant direction
- Same projection
- No refitting on validation data (prevents data leakage)

**Result:** `X_val_lda` (500, 1) - LD1 scores for validation

---

### Statistical Analysis (Lines 99-131)

```python
print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS (SECTION 5)")
print("=" * 60)

# Training set statistics
train_male_scores = X_train_lda[y_train == 0, 0]
train_female_scores = X_train_lda[y_train == 1, 0]

print(f"\nTraining Set - LD1 Score Statistics:")
print(f"  Male:   mean={train_male_scores.mean():.4f}, std={train_male_scores.std():.4f}")
print(f"  Female: mean={train_female_scores.mean():.4f}, std={train_female_scores.std():.4f}")
print(f"  Separation: {abs(train_male_scores.mean() - train_female_scores.mean()):.4f} units")

t_stat, p_value = stats.ttest_ind(train_male_scores, train_female_scores)
print(f"\n  Two-sample t-test:")
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value: {p_value:.6f}")
if p_value < 0.05:
    print(f"    ✓ Classes are significantly different (p < 0.05)")
else:
    print(f"    ✗ Classes are NOT significantly different (p >= 0.05)")
```

#### Two-Sample t-test Theory

**Null hypothesis (H₀):** μ_male = μ_female (no difference)
**Alternative (H₁):** μ_male ≠ μ_female (difference exists)

**Test statistic:**
```
t = (μ₁ - μ₂) / SE
SE = sqrt(s₁²/n₁ + s₂²/n₂)
```

**Interpretation:**
- **Large |t|, small p:** Strong evidence of difference
- **p < 0.05:** Reject H₀, classes significantly different
- **p ≥ 0.05:** Fail to reject H₀, no significant difference

**Why this matters:**
- Validates LDA found meaningful separation
- Quantifies statistical significance
- Supports classification validity

---

### Goodness-of-Fit Test (Lines 138-152)

```python
print("\n" + "=" * 60)
print("SECTION 4.2: GOODNESS-OF-FIT TEST")
print("=" * 60)
print("Testing if train_male_scores are normally distributed...")

shapiro_stat, shapiro_p = stats.shapiro(train_male_scores)

print(f"  Shapiro-Wilk Statistic: {shapiro_stat:.4f}")
print(f"  p-value: {shapiro_p:.6f}")

alpha = 0.05
if shapiro_p > alpha:
    print(f"  Conclusion (p > {alpha}): Fail to reject H0. Data appears to be normally distributed.")
else:
    print(f"  Conclusion (p <= {alpha}): Reject H0. Data does not appear to be normally distributed.")
```

#### Shapiro-Wilk Test Theory

**Null hypothesis (H₀):** Data comes from a normal distribution
**Alternative (H₁):** Data does not come from a normal distribution

**Test statistic:**
```
W = (Σ a_i x_(i))² / Σ (x_i - x̄)²
```
Where x_(i) are ordered sample values and a_i are tabulated constants.

**Interpretation:**
- **W close to 1:** Suggests normality
- **p > 0.05:** Data consistent with normal distribution
- **p ≤ 0.05:** Data significantly different from normal

**Why test normality?**
- **LDA assumption:** Assumes normal class-conditional distributions
- **t-test validity:** t-test robust but prefers normality
- **Diagnostic:** Identifies distribution violations

**Note:** Many real datasets violate normality, but LDA often works well anyway (robust)

---

### Classification with Fisher's Threshold (Lines 155-183)

```python
print("\n" + "=" * 60)
print("CLASSIFICATION USING FISHER'S LINEAR DISCRIMINANT")
print("=" * 60)

# Calculate optimal threshold (midpoint between means)
threshold = (train_male_scores.mean() + train_female_scores.mean()) / 2
print(f"Decision threshold: {threshold:.4f}")
print(f"  Rule: If LD1 score < {threshold:.4f} → Male (0), else → Female (1)")

# Classify training set
train_predictions = (X_train_lda[:, 0] >= threshold).astype(int)
train_accuracy = np.mean(train_predictions == y_train)
print(f"\nTraining Set Accuracy: {train_accuracy:.2%}")

# Classify validation set
val_predictions = (X_val_lda[:, 0] >= threshold).astype(int)
val_accuracy = np.mean(val_predictions == y_val)
print(f"Validation Set Accuracy: {val_accuracy:.2%}")
```

#### Fisher's Linear Discriminant Threshold

**Optimal threshold (assuming equal priors and covariances):**
```
threshold = (μ_male + μ_female) / 2
```

**Classification rule:**
```
if LD1_score < threshold:
    predict Male (class 0)
else:
    predict Female (class 1)
```

**Why midpoint?**
- Minimizes total classification error (under assumptions)
- Equidistant from both class means
- Bayes-optimal for equal priors and equal covariances

**Alternative thresholds:**
- **Unequal priors:** Shift threshold toward rarer class
- **Unequal costs:** Weight errors differently
- **ROC optimization:** Maximize sensitivity/specificity tradeoff

**Accuracy calculation:**
```
accuracy = (# correct predictions) / (# total samples)
```

**Training vs Validation Accuracy:**
- **Training:** How well model fits training data
- **Validation:** How well model generalizes to new data
- **Overfitting indicator:** Large gap between them

---

### Confusion Matrix Details (Lines 172-183)

```python
# Confusion matrix for validation set
val_male_correct = np.sum((val_predictions == 0) & (y_val == 0))
val_male_total = np.sum(y_val == 0)
val_female_correct = np.sum((val_predictions == 1) & (y_val == 1))
val_female_total = np.sum(y_val == 1)

print(f"\nValidation Set Breakdown:")
print(f"  Male:   {val_male_correct}/{val_male_total} correct ({val_male_correct/val_male_total:.2%})")
print(f"  Female: {val_female_correct}/{val_female_total} correct ({val_female_correct/val_female_total:.2%})")
```

**Confusion Matrix Components:**

|                    | **Predicted Male** | **Predicted Female** |
|--------------------|-------------------|---------------------|
| **Actual Male**    | True Negative (TN) | False Positive (FP) |
| **Actual Female**  | False Negative (FN) | True Positive (TP) |

**Metrics computed:**
- **Male accuracy (Specificity):** TN / (TN + FP)
- **Female accuracy (Sensitivity):** TP / (TP + FN)

**Why per-class accuracy matters:**
- **Balanced performance:** Ensures model works well for both classes
- **Bias detection:** Model might favor one class
- **Diagnostic:** Identifies which class is harder to classify

---

### Visualizations (Lines 185-285)

#### Plot 1: Distribution Histograms (Lines 195-220)

```python
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(train_male_scores, bins=30, alpha=0.6, label='Male', color='blue', density=True)
plt.hist(train_female_scores, bins=30, alpha=0.6, label='Female', color='red', density=True)
plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
plt.xlabel('LD1 Score')
plt.ylabel('Density')
plt.title('Training Set: LDA Projection by Gender')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Similar for validation set...
```

**Purpose:** Shows distribution of LD1 scores for each gender

**What to look for:**
- **Separation:** Do distributions overlap?
- **Shape:** Are they approximately normal?
- **Threshold position:** Does it divide classes well?
- **Consistency:** Training vs validation similarity

**Interpretation:**
- **Good separation:** Minimal overlap, clear threshold
- **Poor separation:** Large overlap, many misclassifications
- **Typical:** Moderate overlap, 70-90% accuracy

---

#### Plot 2: Box Plots (Lines 222-241)

```python
plt.figure(figsize=(10, 6))
data_to_plot = [train_male_scores, train_female_scores, val_male_scores, val_female_scores]
labels = ['Train\nMale', 'Train\nFemale', 'Val\nMale', 'Val\nFemale']
colors = ['lightblue', 'lightcoral', 'blue', 'red']

box_parts = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True)
```

**Purpose:** Compares distributions using summary statistics

**Box plot components:**
- **Box:** Interquartile range (IQR, 25th-75th percentile)
- **Line in box:** Median (50th percentile)
- **Whiskers:** 1.5 × IQR or min/max
- **Points:** Outliers
- **Notch:** 95% confidence interval for median

**What to look for:**
- **Median separation:** Do medians differ significantly?
- **Overlap:** Do boxes overlap?
- **Symmetry:** Are distributions skewed?
- **Outliers:** Are there unusual samples?

---

#### Plot 3: Scatter with Misclassifications (Lines 243-285)

```python
plt.subplot(1, 2, 1)
# Correct predictions in normal colors
train_correct_mask = train_predictions == y_train
plt.scatter(np.arange(len(X_train_lda))[train_correct_mask & (y_train == 0)], 
           X_train_lda[train_correct_mask & (y_train == 0), 0], 
           c='blue', alpha=0.3, s=10, label='Male (correct)')
# ...
# Misclassifications in black
plt.scatter(np.arange(len(X_train_lda))[~train_correct_mask], 
           X_train_lda[~train_correct_mask, 0], 
           c='black', marker='x', s=50, label='Misclassified')
```

**Purpose:** Identifies which samples are misclassified

**Color coding:**
- **Blue points:** Correctly classified males
- **Red points:** Correctly classified females
- **Black X's:** Misclassified samples

**What to look for:**
- **Misclassification location:** Are errors near threshold?
- **Patterns:** Do misclassifications cluster?
- **Outliers:** Are misclassifications extreme values?

**Insight:** Helps identify problematic samples or threshold issues

---

### Save Model and Results (Lines 287-338)

```python
print("\n" + "=" * 60)
print("SAVING MODEL AND RESULTS")
print("=" * 60)
os.makedirs("models", exist_ok=True)

model_data = {
    "scaler": scaler,
    "lda": lda,
    "threshold": threshold,
    "train_accuracy": train_accuracy,
    "val_accuracy": val_accuracy,
    "train_male_mean": train_male_scores.mean(),
    "train_female_mean": train_female_scores.mean(),
    "p_value": p_value
}
joblib.dump(model_data, "models/lda_model.pkl")
print("✓ LDA model and statistics saved to models/lda_model.pkl")
```

**What's saved:**
1. **scaler:** For preprocessing new data
2. **lda:** Trained LDA model
3. **threshold:** Decision boundary
4. **Accuracies:** Performance metrics
5. **Class means:** Statistical summaries
6. **p_value:** Statistical significance

**Why save statistics?**
- Reproducibility
- Downstream analyses (regression script uses these)
- Documentation
- Model evaluation

---

#### Text Report Generation (Lines 313-338)

```python
with open("outputs/lda/lda_analysis_report.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("LDA GENDER CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training Set Size: {len(X_train)} samples\n")
    # ... writes detailed statistics
```

**Purpose:** Creates human-readable summary of results

**Contents:**
- Dataset sizes
- Class distributions
- LDA statistics
- Statistical test results
- Classification performance

**Use cases:**
- Quick reference
- Documentation
- Sharing results
- Report writing

---

## Statistical Interpretation

### What LD1 Scores Mean

**LD1 score** = projection onto discriminant direction
- **Negative values:** More male-like
- **Positive values:** More female-like
- **Near zero:** Ambiguous

**Scale:**
- Arbitrary units (depends on standardization)
- Magnitude indicates distance from decision boundary
- Larger |score| = more confident classification

### Model Assumptions

**LDA assumes:**
1. **Multivariate normality:** Each class is normally distributed
2. **Equal covariances:** Σ_male ≈ Σ_female
3. **Linear separability:** Classes separable by hyperplane

**Violations:**
- **Normality:** LDA often robust to violations (Shapiro-Wilk test checks this)
- **Equal covariances:** Quadratic DA (QDA) relaxes this
- **Nonlinearity:** Kernel methods or neural networks needed

### Performance Interpretation

**Typical accuracy ranges:**
- **60-70%:** Weak signal, difficult task
- **70-85%:** Moderate signal, typical for face gender classification
- **85-95%:** Strong signal, good separation
- **95-100%:** Excellent separation (or overfitting)

**For gender classification:**
- Human performance: ~95%+ accuracy
- Automated (LDA): 70-85% typical (depends on image quality, preprocessing)

---

## Integration with Project Pipeline

**Position in pipeline:**
```
image_converter.py → pca_dataset.py → lda_dataset.py ← You are here
                                            ↓
                                 regression_analysis.py
```

**LDA provides:**
- **LD1 scores:** Used as dependent variable (Y) in regression
- **Class separation:** Validates gender information in faces
- **Statistical tests:** Quantifies separation significance
- **Baseline classifier:** Performance benchmark

**Output used by:**
- `regression_analysis.py` - LD1 as response variable
- Model comparison with PCA

---

## Expected Output

**Console output:**
```
============================================================
LOADING DATA
============================================================
Training set shape: (1000, 38804)
  - Male samples: 500
  - Female samples: 500
Validation set shape: (500, 38804)
  - Male samples: 250
  - Female samples: 250

============================================================
FITTING LDA MODEL
============================================================
✓ LDA fitted with 1 component(s)
Explained variance ratio: [1.0]

Training Set - LD1 Score Statistics:
  Male:   mean=-2.0542, std=0.9147
  Female: mean=2.0542, std=1.0767
  Separation: 4.1084 units

  Two-sample t-test:
    t-statistic: -40.2345
    p-value: 0.000000
    ✓ Classes are significantly different (p < 0.05)

============================================================
CLASSIFICATION USING FISHER'S LINEAR DISCRIMINANT
============================================================
Decision threshold: 0.0000
Training Set Accuracy: 82.40%
Validation Set Accuracy: 79.60%
```

**Generated files:**
1. `outputs/lda/lda_gender_distribution.png` - Distribution histograms
2. `outputs/lda/lda_boxplot_comparison.png` - Box plot comparison
3. `outputs/lda/lda_classification_results.png` - Scatter with misclassifications
4. `outputs/lda/lda_analysis_report.txt` - Text summary
5. `models/lda_model.pkl` - Saved model