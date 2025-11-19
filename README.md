# Statistical Gender Classification from Facial Images

## Project Overview

This project implements a **statistical approach** to gender classification using facial images. Unlike machine learning methods (e.g., neural networks, SVM), this project focuses on **classical statistical techniques** including dimensionality reduction, discriminant analysis, hypothesis testing, and regression modeling.

### Core Statistical Methods

1. **Principal Component Analysis (PCA)** - Unsupervised dimensionality reduction
2. **Linear Discriminant Analysis (LDA)** - Supervised classification
3. **Ordinary Least Squares (OLS) Regression** - Relationship modeling
4. **Statistical Hypothesis Testing** - t-tests, Shapiro-Wilk normality tests
5. **Confidence Intervals** - Welch's two-sample t-intervals

### Key Research Questions

1. Can we identify gender from facial images using linear statistical methods?
2. How well does unsupervised PCA capture gender-related features?
3. What is the optimal linear discriminant for gender classification?
4. How do principal components relate to discriminant scores?
5. What is the quantified difference in facial features between genders?

---

## Repository Structure

```
data-science-and-statistical-inference_project/
│
├── dataset/                          # Image data (not included in repo)
│   ├── train/                        # Training images
│   │   ├── male_0001.png
│   │   ├── female_0001.png
│   │   └── ...
│   └── validation/                   # Validation images
│       ├── male_0001.png
│       ├── female_0001.png
│       └── ...
│
├── models/                           # Saved statistical models
│   ├── pca_model.pkl                 # PCA model + scaler
│   └── lda_model.pkl                 # LDA model + statistics
│
├── outputs/                          # Analysis outputs (organized by script)
│   ├── pca/                          # PCA visualizations (if any)
│   ├── lda/                          # LDA analysis outputs
│   │   ├── lda_gender_distribution.png
│   │   ├── lda_boxplot_comparison.png
│   │   ├── lda_classification_results.png
│   │   └── lda_analysis_report.txt
│   └── regression/                   # Regression analysis outputs
│       └── regression_residual_plot.png
│
├── image_converter.py                # Step 1: Image preprocessing
├── pca_dataset.py                    # Step 2: PCA analysis
├── lda_dataset.py                    # Step 3: LDA classification
├── regression_analysis.py            # Step 4: Regression modeling
│
├── requirements.txt                  # Python dependencies
│
├── README.md                         # This file (general overview)
├── image_converter_README.md         # Detailed docs for image_converter.py
├── pca_dataset_README.md             # Detailed docs for pca_dataset.py
├── lda_dataset_README.md             # Detailed docs for lda_dataset.py
└── regression_analysis_README.md     # Detailed docs for regression_analysis.py
```

---

## Analysis Pipeline

The project follows a sequential pipeline. Each script must be run in order:

```
┌─────────────────────────┐
│  1. image_converter.py  │  Image Preprocessing
│  Resize, B&W conversion │
└───────────┬─────────────┘
            │
            ▼
      [Prepared Images]
            │
            ├────────────────────┐
            │                    │
            ▼                    ▼
┌─────────────────────┐  ┌─────────────────────┐
│  2. pca_dataset.py  │  │  3. lda_dataset.py  │
│  Unsupervised       │  │  Supervised         │
│  50 PC scores       │  │  LD1 scores         │
└──────────┬──────────┘  └──────────┬──────────┘
           │                        │
           │    [Saved Models]      │
           │                        │
           └────────┬───────────────┘
                    │
                    ▼
     ┌──────────────────────────────┐
     │ 4. regression_analysis.py    │
     │  PC scores → LD1 scores      │
     │  OLS Regression + CI         │
     └──────────────────────────────┘
```

### Step-by-Step Workflow

#### Step 1: Image Preprocessing (`image_converter.py`)
**Purpose:** Standardize image format and dimensions
- Resize images to consistent dimensions (e.g., 178×218 pixels)
- Convert to grayscale (reduce from 3 channels to 1)
- Normalize pixel values to [0, 1]

**Output:** Preprocessed images ready for analysis

**See:** `image_converter_README.md` for detailed documentation

---

#### Step 2: PCA Analysis (`pca_dataset.py`)
**Purpose:** Unsupervised dimensionality reduction
- Load and standardize images
- Fit PCA on training data (find principal components)
- Transform images to 50-dimensional PC space
- Visualize first 2 PCs colored by gender
- Save PCA model for reuse

**Statistical Method:** Principal Component Analysis
- Eigendecomposition of covariance matrix
- Identifies directions of maximum variance
- Reduces 38,804 dimensions → 50 dimensions

**Output:** 
- `models/pca_model.pkl` - Saved PCA model
- Scatter plots of PC1 vs PC2

**See:** `pca_dataset_README.md` for detailed documentation

---

#### Step 3: LDA Classification (`lda_dataset.py`)
**Purpose:** Supervised binary classification
- Load and standardize images
- Fit LDA on training data with gender labels
- Compute LD1 scores (optimal separating direction)
- Perform statistical tests (t-test, Shapiro-Wilk)
- Classify using Fisher's threshold
- Evaluate performance on validation set
- Generate visualizations (distributions, box plots, scatter)
- Save LDA model and detailed report

**Statistical Method:** Linear Discriminant Analysis (Fisher's)
- Maximizes between-class variance
- Minimizes within-class variance
- Finds optimal linear decision boundary

**Output:**
- `models/lda_model.pkl` - Saved LDA model + statistics
- `outputs/lda/lda_analysis_report.txt` - Detailed statistical report
- `outputs/lda/lda_gender_distribution.png` - Distribution histograms
- `outputs/lda/lda_boxplot_comparison.png` - Box plot comparison
- `outputs/lda/lda_classification_results.png` - Misclassification plot

**Key Statistics:**
- LD1 score means and standard deviations
- Two-sample t-test (significant difference?)
- Classification accuracy (train & validation)
- Shapiro-Wilk normality test

**See:** `lda_dataset_README.md` for detailed documentation

---

#### Step 4: Regression Analysis (`regression_analysis.py`)
**Purpose:** Model relationship between PCA and LDA
- Load pre-fitted PCA and LDA models
- Generate PC scores (independent variables)
- Generate LD1 scores (dependent variable)
- Fit OLS regression: LD1 ~ PC1 + PC2 + ... + PC50
- Evaluate model fit (R², F-statistic)
- Diagnose residuals (check assumptions)
- Calculate Welch's 95% confidence interval for mean difference

**Statistical Methods:** 
1. **OLS Regression:** Model relationship between features
2. **Welch's t-interval:** Confidence interval for mean difference

**Output:**
- Comprehensive regression summary (R², coefficients, p-values)
- `outputs/regression/regression_residual_plot.png` - Residual diagnostics
- Confidence interval for gender difference

**Key Statistics:**
- R² (variance explained)
- F-statistic (overall model significance)
- Individual coefficient p-values
- 95% CI for (μ_male - μ_female)

**See:** `regression_analysis_README.md` for detailed documentation

---

## Installation and Setup

### Prerequisites
- Python 3.7+ 
- pip (Python package manager)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages
- **numpy** - Numerical computing
- **matplotlib** - Plotting and visualization
- **scikit-learn** - PCA, LDA, StandardScaler
- **Pillow (PIL)** - Image processing
- **scipy** - Statistical functions
- **joblib** - Model serialization
- **statsmodels** - Regression analysis

---

## Usage

### 1. Prepare Your Dataset

Organize images in the following structure:
```
dataset/
├── train/
│   ├── male_0001.png
│   ├── male_0002.png
│   ├── female_0001.png
│   ├── female_0002.png
│   └── ...
└── validation/
    ├── male_0001.png
    ├── female_0001.png
    └── ...
```

**Naming convention:** Files must start with `male` or `female` (case-insensitive)

---

### 2. Run Image Preprocessing (Optional)

If your images need resizing or format conversion:

```bash
# Resize to 178x218 and convert to B&W
python image_converter.py --folder dataset/train --width 178 --height 218

python image_converter.py --folder dataset/validation --width 178 --height 218
```

**Options:**
- `--width`, `--height`: Target dimensions
- `--mode`: Resize mode (fit, contain, stretch, cover)
- `--color`: Keep images in color (default: B&W)
- `--prefix`: Output filename prefix

---

### 3. Run PCA Analysis

```bash
python pca_dataset.py
```

**What happens:**
- Loads up to 1000 training and 500 validation images
- Fits PCA on training data (50 components)
- Transforms both datasets
- Shows variance explained
- Saves model to `models/pca_model.pkl`

**Expected runtime:** 1-3 minutes

---

### 4. Run LDA Classification

```bash
python lda_dataset.py
```

**What happens:**
- Loads training and validation images
- Fits LDA on training data
- Performs statistical tests (t-test, Shapiro-Wilk)
- Classifies using Fisher's threshold
- Generates 3 visualization plots
- Saves model and report

**Expected runtime:** 2-5 minutes

**Outputs:**
- Classification accuracy (~75-85% typical)
- Statistical significance (p-value)
- Visualizations
- Detailed text report

---

### 5. Run Regression Analysis

```bash
python regression_analysis.py
```

**What happens:**
- Loads saved PCA and LDA models
- Generates PC scores and LD1 scores
- Fits OLS regression
- Evaluates model (R², F-test)
- Creates residual plot
- Calculates confidence interval

**Expected runtime:** < 1 minute

**Outputs:**
- Regression summary (R² typically 0.80-0.95)
- Significant predictors
- Residual diagnostics
- 95% CI for mean difference

---

## Understanding the Output

### PCA Results

**Variance explained:**
```
PC1: 29.6%
PC2: 8.3%
PC3: 5.9%
...
Total (50 PCs): 84.0%
```

**Interpretation:**
- First 50 PCs capture ~84% of image variance
- Massive dimensionality reduction: 38,804 → 50 features
- PC1-PC10 typically capture main facial features

---

### LDA Results

**Example output:**
```
Training Set - LD1 Score Statistics:
  Male:   mean=-2.05, std=0.91
  Female: mean=2.05, std=1.08
  Separation: 4.10 units

Two-sample t-test:
  t-statistic: -64.85
  p-value: 0.000000
  ✓ Classes are significantly different (p < 0.05)

Training Set Accuracy: 96.9%
Validation Set Accuracy: 84.2%
```

**Interpretation:**
- **Separation:** ~4 standard deviations between genders
- **Significance:** p << 0.05, highly significant difference
- **Accuracy:** ~84% correct classification
- **Generalization:** Similar train/validation accuracy (good sign)

---

### Regression Results

**Example output:**
```
R-squared: 0.511
Adj. R-squared: 0.485
F-statistic: 19.84
Prob (F-statistic): 1.84e-114

95% Confidence Interval: [-4.23, -3.98]
```

**Interpretation:**
- **Adjusted R² = 0.485** PC scores explain % of LD1 variance
- **F-test:** Overall model highly significant (p ≈ 0)
- **CI:** Males score 3.98-4.23 points lower than females (significant difference)
- **Conclusion:** Strong linear relationship between PCA and LDA features

---

## Statistical Theory Summary

### Why These Methods?

#### 1. **PCA (Unsupervised)**
- **Goal:** Find main patterns of variation
- **Advantage:** No labels needed, reduces dimensionality
- **Limitation:** Doesn't optimize for classification
- **Use case:** Data exploration, noise reduction

#### 2. **LDA (Supervised)**
- **Goal:** Find optimal linear separator
- **Advantage:** Uses labels, maximizes class separation
- **Limitation:** Assumes linear separability, normal distributions
- **Use case:** Binary/multi-class classification

#### 3. **OLS Regression**
- **Goal:** Model relationship between variables
- **Advantage:** Interpretable coefficients, statistical inference
- **Limitation:** Assumes linearity, homoscedasticity
- **Use case:** Explanatory modeling, prediction

#### 4. **Hypothesis Testing**
- **Goal:** Quantify evidence for/against claims
- **Advantage:** Rigorous statistical framework
- **Use case:** Validate findings, make inferences

---

## Key Assumptions

### PCA Assumptions
1. **Linear relationships:** Features related linearly
2. **Large variance = important:** High variance features are meaningful
3. **Orthogonal components:** PCs are uncorrelated

### LDA Assumptions
1. **Multivariate normality:** Each class follows normal distribution
2. **Equal covariances:** Σ_male ≈ Σ_female
3. **Linear separability:** Classes separable by linear boundary

### OLS Regression Assumptions
1. **Linearity:** Y = β₀ + β₁X₁ + ... + βₚXₚ + ε
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant error variance
4. **Normality:** Errors ~ N(0, σ²)

**Note:** Methods often robust to moderate violations

---

## Interpreting Results

### Good Results Indicate:

✅ **R² (0.48-0.52):**
- Relationship between PCA and LDA
- Gender information captured by variance
- Missing important features (~50%)

✅ **Significant t-test (p < 0.05):**
- Classes significantly different
- LDA found meaningful separation

✅ **Validation accuracy ~80%:**
- Good generalization
- Reasonable performance for linear method

✅ **CI doesn't contain 0:**
- Significant gender difference
- Effect size quantified

---

## Detailed Documentation

For line-by-line code explanations and statistical theory:

- **📘 [Image Converter Documentation](image_converter_README.md)**
- **📘 [PCA Analysis Documentation](pca_dataset_README.md)**
- **📘 [LDA Classification Documentation](lda_dataset_README.md)**
- **📘 [Regression Analysis Documentation](regression_analysis_README.md)**
