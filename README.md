# Gender Classification Using Linear Discriminant Analysis

A comprehensive statistical analysis project for gender classification from face images using Linear Discriminant Analysis (LDA). This project demonstrates the application of probability and statistics theory for supervised classification, comparing unsupervised (PCA) vs supervised (LDA) approaches.

## 📊 Project Overview

This repository contains a complete pipeline for gender classification using classical statistical methods:

1. **Dataset Preparation**: Face image dataset with train/validation splits
2. **Image Preprocessing**: Batch image conversion utility for standardization
3. **PCA Analysis**: Initial unsupervised dimensionality reduction attempt
4. **LDA Analysis**: Fisher's Linear Discriminant for supervised classification
5. **Statistical Testing**: Hypothesis testing, t-tests, confidence intervals
6. **Comprehensive Documentation**: Theory, formulas, and visual guides

### Key Distinction: Statistical Approach, Not Machine Learning

This project emphasizes **classical statistical inference** rather than modern machine learning:
- Uses Fisher's Linear Discriminant (1936)
- Focuses on hypothesis testing and p-values
- Provides interpretable results with statistical significance
- Perfect for probability and statistics coursework

---

## 🎯 Main Features

### Statistical Classification Pipeline
- **Fisher's Linear Discriminant Analysis** for supervised classification
- **Statistical significance testing** (two-sample t-test, p-values)
- **Bayes' optimal decision rule** for threshold calculation
- **Confusion matrix analysis** with sensitivity/specificity metrics
- **Effect size calculation** (Cohen's d)
- **Confidence intervals** for accuracy estimates

### Image Processing Utilities
- Batch image converter with multiple resize modes
- Grayscale or color output options
- Customizable output file naming
- Support for multiple image formats (JPG, PNG, BMP, GIF, TIFF, WEBP)

### Visualization & Reporting
- Distribution comparison plots (train vs validation)
- Boxplot analysis for class separation
- Misclassification scatter plots
- Automated text report generation
- Model persistence for reproducibility

---

## 📁 Repository Structure

```
├── dataset/
│   ├── train/          # Training images (labeled by filename)
│   └── validation/     # Validation images (labeled by filename)
├── models/             # Saved LDA models and scalers
├── lda_gender_classification.py    # Main LDA analysis script
├── compare_pca_lda.py              # PCA vs LDA comparison
├── pca_dataset.py                  # Original PCA implementation
├── image_converter.py              # Image preprocessing utility
├── requirements.txt                # Python dependencies
├── 00_DOCUMENTATION_INDEX.md       # Master documentation guide
├── LDA_THEORY_EXPLAINED.md         # Complete statistical theory
├── LDA_QUICK_REFERENCE.md          # Formula quick reference
├── LDA_VISUAL_GUIDE.md             # Visual intuition guide
└── PCA_vs_LDA_Guide.md             # Comparison and recommendations
```

---

## 🚀 Quick Start

### 1. Installation

```powershell
# Clone the repository
git clone <repository-url>
cd data-science-and-statistical-inference_project

# Create and activate virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your face images in the following structure:
```
dataset/
├── train/
│   ├── male_001.jpg
│   ├── male_002.jpg
│   ├── female_001.jpg
│   └── female_002.jpg
└── validation/
    ├── male_101.jpg
    └── female_101.jpg
```

**Important**: Filenames must start with "male" or "female" for automatic labeling.

### 3. Run LDA Analysis

```powershell
# Option 1: Using activated venv
python lda_gender_classification.py

# Option 2: Direct venv execution
.\venv\Scripts\python.exe lda_gender_classification.py
```

### 4. Compare PCA vs LDA (Optional)

```powershell
python compare_pca_lda.py
```

---

## 📊 Main Scripts

### `lda_gender_classification.py` - Complete LDA Analysis

**What it does:**
- Loads training and validation datasets
- Standardizes features using training set statistics
- Fits Fisher's Linear Discriminant on training data
- Performs statistical significance testing (t-test)
- Calculates optimal classification threshold
- Generates confusion matrix and performance metrics
- Creates 3 visualization plots
- Saves detailed text report and model

**Outputs:**
- `lda_gender_distribution.png` - Distribution comparison
- `lda_boxplot_comparison.png` - Boxplot analysis
- `lda_classification_results.png` - Misclassification visualization
- `lda_analysis_report.txt` - Detailed statistical report
- `models/lda_model.pkl` - Saved LDA model

**Usage:**
```powershell
python lda_gender_classification.py
```

---

### `compare_pca_lda.py` - Visual Comparison

**What it does:**
- Runs both PCA and LDA on the same dataset
- Compares separation quality using t-tests
- Creates side-by-side visualization
- Shows why LDA works better for classification

**Outputs:**
- `pca_vs_lda_comparison.png` - 2×2 comparison plot
- Terminal summary of both methods

**Usage:**
```powershell
python compare_pca_lda.py
```

---

### `image_converter.py` - Image Preprocessing Utility

**What it does:**
- Batch converts images to standard format
- Resizes to custom dimensions with aspect ratio handling
- Converts to grayscale or keeps in color
- Useful for preparing datasets

**Features:**
- Multiple resize modes: fit, contain, stretch, cover
- Optional grayscale conversion (default) or color preservation
- Customizable output filename prefix
- Automatic skip of already processed files

**Usage:**

Basic conversion to grayscale with resizing:
```powershell
python image_converter.py --folder images --width 800 --height 600
```

Keep color and original resolution:
```powershell
python image_converter.py --folder images --color
```

Custom prefix with specific mode:
```powershell
python image_converter.py --folder <folder_name> --color
```

### Custom Prefix for Output Files
```bash
python image_converter.py --folder <folder_name> --width 800 --height 600 --prefix thumbnail
```

## Examples

### Example 1: Resize to 800x600 with fit mode (default, black & white)
```bash
python image_converter.py --folder images --width 800 --height 600
```

### Example 2: Resize to 1920x1080 with contain mode (letterbox/pillarbox)
```bash
python image_converter.py --folder photos --width 1920 --height 1080 --mode contain
```

### Example 3: Resize to 512x512 with cover mode (crop to fill)
```bash
python image_converter.py --folder pics --width 512 --height 512 --mode cover
```

### Example 4: Resize to exact dimensions with stretch mode
```bash
python image_converter.py --folder wallpapers --width 1024 --height 768 --mode stretch
```

### Example 5: Keep images in color (no black & white conversion)
```bash
python image_converter.py --folder portraits --width 800 --height 800 --mode cover --prefix thumbnail
```

See full documentation at the end of this README for all image converter options.

---

## 📚 Documentation

This project includes extensive documentation for understanding the theory and implementation:

### **[00_DOCUMENTATION_INDEX.md](00_DOCUMENTATION_INDEX.md)** - Start Here!
Master guide with navigation to all documentation, reading order recommendations, and project overview.

### **[LDA_THEORY_EXPLAINED.md](LDA_THEORY_EXPLAINED.md)** - Complete Theory
In-depth explanation of:
- Fisher's Linear Discriminant mathematical derivation
- Probability and statistics foundations
- Statistical tests (t-test, confidence intervals, hypothesis testing)
- Multivariate normal distributions
- Bayes' decision theory
- Assumptions and when they're violated

### **[LDA_QUICK_REFERENCE.md](LDA_QUICK_REFERENCE.md)** - Quick Lookup
- All important formulas on one page
- p-value and accuracy interpretation tables
- Code-to-theory mapping
- Example calculations

### **[LDA_VISUAL_GUIDE.md](LDA_VISUAL_GUIDE.md)** - Visual Intuition
- ASCII art diagrams showing how LDA works
- Step-by-step visual explanations
- Before/after comparisons
- Distribution plot interpretations
- Common error patterns

### **[PCA_vs_LDA_Guide.md](PCA_vs_LDA_Guide.md)** - Comparison
- Why PCA failed for classification
- Why LDA is better for this project
- Supervised vs unsupervised comparison
- Project recommendations

---

## 🎓 Key Concepts & Theory

### Fisher's Linear Discriminant

LDA finds a projection vector **w** that maximizes the ratio:

```
J(w) = (Between-Class Variance) / (Within-Class Variance)
     = (w^T S_B w) / (w^T S_W w)
```

where:
- **S_B** = Between-class scatter matrix (separation of means)
- **S_W** = Within-class scatter matrix (spread within classes)

**Optimal solution:**
```
w = S_W^(-1) (μ_male - μ_female)
```

### Statistical Significance Testing

**Two-sample t-test** to verify classes are different:

```
H₀: μ_male = μ_female (null hypothesis)
H₁: μ_male ≠ μ_female (alternative hypothesis)
```

**Interpretation:**
- p < 0.001: ★★★ Extremely significant separation
- p < 0.01: ★★☆ Very significant
- p < 0.05: ★☆☆ Significant (conventional threshold)
- p ≥ 0.05: ☆☆☆ Not statistically significant

### Decision Rule

**Optimal threshold** (assuming equal priors):
```
threshold = (μ_male + μ_female) / 2
```

**Classification:**
```
If LD1_score < threshold → Male
If LD1_score ≥ threshold → Female
```

---

## 📈 Expected Results

### Good Separation Scenario
- **p-value**: < 0.05 (statistically significant)
- **Accuracy**: > 70%
- **Visualization**: Clear separation in distribution plots
- **Conclusion**: Gender can be classified from face images

### Poor Separation Scenario
- **p-value**: > 0.05 (not significant)
- **Accuracy**: ≈ 50% (random guessing)
- **Visualization**: Large overlap in distributions
- **Conclusion**: Gender cannot be reliably classified from raw pixels


---

## 🔬 PCA vs LDA Comparison

| Aspect | PCA | LDA |
|--------|-----|-----|
| **Type** | Unsupervised | Supervised |
| **Uses labels?** | ❌ No | ✅ Yes |
| **Goal** | Maximize variance | Maximize class separation |
| **Best for** | Visualization, noise reduction | Classification |
| **# Components** | Up to n features | Up to k-1 classes |
| **For 2 classes** | Many components | Exactly 1 component |

**Why LDA for this project:**
- Uses gender labels to find discriminative direction
- Maximizes separation between males and females
- Statistical foundation (Fisher's discriminant)
- Perfect for probability & statistics course

---

## 📊 Performance Metrics

The analysis reports multiple metrics:

- **Accuracy**: Overall correct classification rate
- **Sensitivity**: True positive rate (male classification rate)
- **Specificity**: True negative rate (female classification rate)
- **Confusion Matrix**: Breakdown of correct/incorrect predictions
- **Confidence Intervals**: Statistical uncertainty estimates
- **Effect Size (Cohen's d)**: Magnitude of separation

---

## 🛠️ Requirements

```
Pillow>=10.0.0           # Image processing
scikit-learn>=1.3.0      # PCA and LDA
numpy>=1.24.0            # Numerical operations
matplotlib>=3.7.0        # Plotting
joblib>=1.3.0            # Model persistence
scipy>=1.11.0            # Statistical tests
```

# Appendix: Image Converter Detailed Usage

## Image Converter Features

- Batch converts images to standard format
- Multiple resize modes (fit, contain, stretch, cover)
- Optional grayscale conversion or color preservation
- Customizable output filename prefix
- Supports JPG, PNG, BMP, GIF, TIFF, WEBP

## Usage Examples

### Basic Conversion
```powershell
python image_converter.py --folder images --width 800 --height 600
