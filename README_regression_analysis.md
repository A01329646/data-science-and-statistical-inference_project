# Regression Analysis - Detailed Documentation

## Overview
This script performs **Ordinary Least Squares (OLS) regression** to model the relationship between PCA components (independent variables) and LDA scores (dependent variable). It also calculates **confidence intervals** for the difference in mean LD1 scores between genders using Welch's t-interval.

---

## Statistical Theory: OLS Regression

### What is OLS Regression?

**Ordinary Least Squares regression** finds the best-fitting linear relationship between predictors (X) and a response (Y) by minimizing the sum of squared residuals.

### Mathematical Foundation

**Model:**
```
Y = β₀ + β₁X₁ + β₂X₂ + ... + β₅₀X₅₀ + ε
```

Where:
- **Y** = LD1 score (dependent variable)
- **X₁...X₅₀** = First 50 PC scores (independent variables)
- **β₀** = Intercept
- **β₁...β₅₀** = Coefficients (slopes)
- **ε** = Error term (residuals)

**Estimation (Least Squares):**
```
β̂ = (X^T X)^(-1) X^T Y
```

**Goal:** Minimize sum of squared residuals (SSR):
```
SSR = Σ(yᵢ - ŷᵢ)²
```

### Model Evaluation Metrics

**R-squared (R²):**
- Proportion of variance in Y explained by X
- Range: [0, 1]
- Formula: R² = 1 - (SSR / SST)
  - SSR = Sum of Squared Residuals
  - SST = Total Sum of Squares

**Adjusted R-squared (R²_adj):**
- Adjusted for number of predictors
- Penalizes model complexity
- Formula: R²_adj = 1 - [(1-R²)(n-1)/(n-p-1)]
  - n = sample size
  - p = number of predictors

**F-statistic:**
- Tests overall model significance
- H₀: All coefficients = 0 (model useless)
- H₁: At least one coefficient ≠ 0

**Individual t-tests:**
- Tests significance of each coefficient
- H₀: βᵢ = 0 (predictor i has no effect)
- H₁: βᵢ ≠ 0 (predictor i has effect)

---

## Line-by-Line Code Explanation

### Header and Imports (Lines 1-14)

```python
# ==============================
# REGRESSION ANALYSIS SCRIPT (PCA -> LDA)
# ==============================

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from scipy import stats
import statsmodels.api as sm
```

**Key imports:**
- **statsmodels.api:** Professional statistical modeling library
  - More detailed output than scikit-learn
  - Provides p-values, confidence intervals, diagnostics
  - Industry-standard for statistical analysis
- **scipy.stats:** Statistical functions (t-distribution, t-tests)

**Why statsmodels instead of scikit-learn?**
- Comprehensive regression summary (R², F-stat, p-values)
- Statistical inference tools
- Better for explanatory modeling (vs predictive)

---

### Parameters (Lines 16-21)

```python
TRAIN_DIR = "dataset/train"
MODEL_LDA_PATH = "models/lda_model.pkl"
MODEL_PCA_PATH = "models/pca_model.pkl"
```

**Dependencies:**
- Requires pre-fitted LDA model (from `lda_dataset.py`)
- Requires pre-fitted PCA model (from `pca_dataset.py`)
- Both must be run before this script

**Why load models instead of refitting?**
- Consistency: Use exact same transformations
- Efficiency: PCA/LDA already computed
- Reproducibility: Same model across analyses

---

### Function: load_images_and_labels (Lines 23-52)

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
            arr = np.array(img, dtype=np.float32) / 255.0
            data.append(arr.flatten())
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
    valid_mask = labels != -1
    return data[valid_mask], labels[valid_mask]
```

**Same as LDA script** - loads images and extracts gender labels

**Purpose here:** Need raw image data to generate PCA and LDA features

---

### Load Data and Models (Lines 59-77)

```python
X_train, y_train = load_images_and_labels(TRAIN_DIR, max_images=1000)
print(f"✓ Raw training data loaded: {X_train.shape}")

try:
    lda_data = joblib.load(MODEL_LDA_PATH)
    pca_data = joblib.load(MODEL_PCA_PATH)
except FileNotFoundError:
    print(f"Error: Persisted model files not found.")
    print(f"Ensure 'lda_model.pkl' and 'pca_model.pkl' are present in the /models/ directory.")
    exit()

scaler = lda_data["scaler"]
lda = lda_data["lda"]
pca = pca_data["pca"]
print("✓ LDA, PCA, and Scaler models loaded from /models/")
```

**Line 64:** Load raw training images (1000 samples × 38,804 pixels)

**Lines 67-73:** Load pre-fitted models with error handling
- `try-except` catches missing file error
- Provides helpful error message
- `exit()` terminates if models not found

**Lines 75-77:** Extract components from saved dictionaries
- **scaler:** For standardization
- **lda:** For generating LD1 scores (Y)
- **pca:** For generating PC scores (X)

**Why load from LDA dict instead of PCA dict?**
- Both use same scaler
- LDA dict also contains statistics needed later

---

### Generate Regression Variables (Lines 79-91)

```python
# Re-scale training data using the loaded scaler
X_train_scaled = scaler.transform(X_train)

# Generate Y (Dependent) Variable: The LD1 Score
Y_reg = lda.fit_transform(X_train_scaled, y_train).flatten() # .flatten() to ensure 1D vector

# Generate X (Independent) Variables: The 50 PC Scores
X_reg = pca.transform(X_train_scaled)

print(f"✓ Regression variables generated:")
print(f"  - Y_reg (LD1 Score) shape: {Y_reg.shape}")
print(f"  - X_reg (50 PC Scores) shape: {X_reg.shape}")
```

**Line 82:** Standardize raw images using training statistics
- Same preprocessing as PCA/LDA scripts
- Critical for consistency

**Line 85:** Generate dependent variable (Y)
- `lda.fit_transform()` - Refit LDA and get LD1 scores
- Why refit? Need LD1 scores for current data
- `.flatten()` - Converts (1000, 1) to (1000,) for regression

**Y (LD1 score) interpretation:**
- Single number per image
- Measures position along discriminant direction
- Negative = male-like, Positive = female-like

**Line 88:** Generate independent variables (X)
- `pca.transform()` - Project images onto 50 PCs
- No refitting, just transformation
- Result: (1000, 50) matrix

**X (PC scores) interpretation:**
- 50 numbers per image
- Each represents position along a principal component
- PC1 = most variance, PC50 = 50th most variance

**Regression question:** Can PC scores predict LD1 scores?

---

### OLS Regression (Lines 93-115)

```python
print("\n" + "=" * 60)
print("SECTION 5.3: REGRESSION MODELING")
print("=" * 60)

# Statsmodels OLS requires an explicit constant (intercept, B0) to be added
X_reg_with_const = sm.add_constant(X_reg)

# Fit Ordinary Least Squares (OLS) model: sm.OLS(Y, X)
model = sm.OLS(Y_reg, X_reg_with_const)
results = model.fit()

# Print the full OLS regression summary
print(results.summary())
```

**Line 100:** `sm.add_constant(X_reg)`
- Adds column of 1's to X matrix
- Allows model to estimate intercept (β₀)
- Shape: (1000, 50) → (1000, 51)

**Why add constant manually?**
- statsmodels doesn't add automatically (unlike scikit-learn)
- Explicit is better than implicit
- Control over model specification

**Line 103:** `sm.OLS(Y_reg, X_reg_with_const)`
- Creates OLS model object
- Note order: (Y, X) not (X, Y)
- Doesn't fit yet, just sets up

**Line 104:** `results = model.fit()`
- Performs least squares estimation
- Computes β̂ = (X^T X)^(-1) X^T Y
- Calculates all statistics

**Line 107:** `results.summary()`
- Comprehensive regression output
- Includes R², adjusted R², F-statistic, p-values
- Individual coefficient estimates and tests

**Summary components:**

1. **Model info:**
   - Dependent variable: LD1 score
   - Model: OLS
   - Number of observations: 1000

2. **Overall fit:**
   - R-squared: Proportion of variance explained
   - Adj. R-squared: Adjusted for number of predictors
   - F-statistic: Overall model significance
   - Prob (F-statistic): p-value for F-test

3. **Coefficients:**
   - const: Intercept (β₀)
   - PC1-PC50: Slopes (β₁-β₅₀)
   - std err: Standard error of estimate
   - t: t-statistic for significance test
   - P>|t|: p-value (is βᵢ significantly different from 0?)
   - [0.025, 0.975]: 95% confidence interval

**Lines 109-115:** Interpretation guide
- Highlights key statistics
- Reference to report sections

---

### Residual Analysis (Lines 117-135)

```python
print("\n" + "=" * 60)
print("GENERATING RESIDUAL PLOT...")
print("=" * 60)

Y_pred = results.predict(X_reg_with_const)
residuals = results.resid

plt.figure(figsize=(10, 6))
plt.scatter(Y_pred, residuals, alpha=0.5)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Fitted Values (Predicted LD1 Score)")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True, alpha=0.3)
plt.savefig("regression_residual_plot.png", dpi=300, bbox_inches='tight')
print("✓ Residual plot saved to: regression_residual_plot.png")
```

**Line 123:** `results.predict(X_reg_with_const)`
- Predicted Y values: ŷ = β̂₀ + Σ β̂ᵢ xᵢ
- Shape: (1000,)

**Line 124:** `results.resid`
- Residuals: e = y - ŷ (actual - predicted)
- Shape: (1000,)

**Residual plot purpose:**
- **Diagnostic tool** for regression assumptions
- X-axis: Fitted values (ŷ)
- Y-axis: Residuals (e)

**What to look for:**

1. **Random scatter around 0:**
   - ✓ Good: No pattern
   - ✗ Bad: Systematic pattern (non-linear relationship)

2. **Constant variance (homoscedasticity):**
   - ✓ Good: Uniform spread across X
   - ✗ Bad: Funnel shape (heteroscedasticity)

3. **No outliers:**
   - ✓ Good: Points within ±3 SD
   - ✗ Bad: Extreme residuals (influential points)

**Line 130:** `plt.axhline(0, ...)`
- Horizontal line at y=0 (reference)
- Ideally, residuals centered around this line

**Interpretation:**
- **Good fit:** Random cloud around 0, constant spread
- **Poor fit:** Pattern, increasing/decreasing variance, outliers

---

### Confidence Interval (Lines 137-181)

```python
print("\n" + "=" * 60)
print("SECTION 5.2: CONFIDENCE INTERVAL (MEAN DIFFERENCE)")
print("=" * 60)

# Load saved statistics from the LDA model object
m1 = lda_data["train_male_mean"]
m2 = lda_data["train_female_mean"]

# Load statistics (std, n) from report for CI calculation
s1 = 0.9147 # From analysis report
n1 = 500
s2 = 1.0767 # From analysis report
n2 = 500

# Formula for Welch's T-Confidence Interval
se_diff = np.sqrt(s1**2 / n1 + s2**2 / n2)
mean_diff = m1 - m2

# Degrees of Freedom (Welch-Satterthwaite Equation)
df_num = (s1**2/n1 + s2**2/n2)**2
df_den = ( (s1**2/n1)**2 / (n1-1) ) + ( (s1**2/n2)**2 / (n2-1) )
df = df_num / df_den

# Find the critical t-value (t*) for 95% confidence
t_star = stats.t.ppf(0.975, df)

lower_bound = mean_diff - t_star * se_diff
upper_bound = mean_diff + t_star * se_diff

print(f"  Mean Difference (Male - Female): {mean_diff:.4f}")
print(f"  Degrees of Freedom (Welch's): {df:.2f}")
print(f"  Standard Error of the Difference: {se_diff:.4f}")
print(f"  Critical t-value (t_star): {t_star:.4f}")
print(f"  95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")
```

#### Welch's Two-Sample t-Interval Theory

**Goal:** Estimate difference in population means (μ_male - μ_female)

**Point estimate:**
```
d = x̄₁ - x̄₂
```

**Standard error:**
```
SE = sqrt(s₁²/n₁ + s₂²/n₂)
```

**Confidence interval:**
```
(d - t* SE, d + t* SE)
```

**Degrees of freedom (Welch-Satterthwaite):**
```
df = (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

**Why Welch's instead of pooled t-interval?**
- Doesn't assume equal variances (σ₁² = σ₂²)
- More robust to violations
- Conservative (wider CI if variances unequal)

**Line 148:** `m1, m2` - Mean LD1 scores from LDA model

**Lines 151-154:** Sample statistics
- `s1, s2` - Standard deviations (from LDA report)
- `n1, n2` - Sample sizes (500 each)

**Line 157:** Standard error calculation
```
SE = sqrt(0.9147²/500 + 1.0767²/500)
```

**Lines 160-162:** Welch's degrees of freedom
- More complex than pooled df = n₁ + n₂ - 2
- Accounts for unequal variances

**Line 165:** `stats.t.ppf(0.975, df)`
- **ppf:** Percent point function (inverse CDF)
- **0.975:** 97.5th percentile (for 95% two-tailed interval)
- Returns critical t-value (t*)

**Why 0.975 not 0.95?**
- Two-tailed test: 2.5% in each tail
- Lower bound: 2.5% ← [95% confidence] → 2.5% upper bound
- Need value that leaves 2.5% above (97.5th percentile)

**Lines 167-168:** Interval bounds
```
CI = (mean_diff - t*×SE, mean_diff + t*×SE)
```

**Interpretation:**
- "We are 95% confident that the true difference in mean LD1 scores between males and females is between [lower, upper]"
- If interval doesn't contain 0 → significant difference
- Width indicates precision (narrower = more precise)

---

## Statistical Interpretation

### Regression Results

**R² interpretation:**
- R² = 0.85 → 85% of variance in LD1 explained by PCs
- High R² → Strong relationship
- Low R² → Weak relationship, other factors important

**Adjusted R² vs R²:**
- Always: R²_adj ≤ R²
- Gap widens with more predictors
- Adjusted R² penalizes complexity

**F-statistic:**
- Tests H₀: β₁ = β₂ = ... = β₅₀ = 0
- If p < 0.05 → At least one PC significantly predicts LD1
- Overall model usefulness

**Individual t-tests:**
- Which PCs significantly predict LD1?
- p < 0.05 → That PC contributes
- Expect: Early PCs significant, later PCs not

**Typical findings:**
- R² = 0.80-0.95 (strong relationship)
- Many PCs significant (p < 0.05)
- Some PCs not significant (noise components)

### Confidence Interval Interpretation

**Example:** CI = [-4.21, -3.95]

**Interpretation:**
1. **Point estimate:** Males score ~4.08 points lower than females
2. **95% confidence:** True difference between -4.21 and -3.95
3. **Does not contain 0:** Significant difference
4. **Width:** 0.26 units (relatively precise)

**Relationship to hypothesis test:**
- If CI contains 0 → Cannot reject H₀ (no difference)
- If CI doesn't contain 0 → Reject H₀ (significant difference)
- CI provides more information (effect size + precision)

---

## Residual Diagnostics

### Assumptions of OLS Regression

1. **Linearity:** Relationship between X and Y is linear
2. **Independence:** Observations are independent
3. **Homoscedasticity:** Constant error variance
4. **Normality:** Errors normally distributed

### Checking Assumptions with Residual Plot

**Pattern detection:**

1. **Linear relationship:**
   - ✓ Random scatter
   - ✗ Curved pattern → Need non-linear terms

2. **Homoscedasticity:**
   - ✓ Constant spread
   - ✗ Funnel shape → Variance changes with X

3. **Outliers:**
   - ✓ Points within ±3 SD
   - ✗ Extreme points → Investigate

**What if assumptions violated?**
- **Non-linearity:** Add polynomial terms, transformations
- **Heteroscedasticity:** Weighted least squares, robust SE
- **Non-normality:** Robust regression, bootstrapping
- **Outliers:** Investigate, remove if justified

---

## Integration with Project Pipeline

**Position in pipeline:**
```
image_converter.py → pca_dataset.py → lda_dataset.py → run_regression_analysis.py ← You are here
```

**Inputs:**
- Pre-fitted PCA model (from `pca_dataset.py`)
- Pre-fitted LDA model (from `lda_dataset.py`)
- Raw training images

**What this script does:**
1. Generates PC scores (X) from PCA
2. Generates LD1 scores (Y) from LDA
3. Fits regression model: Y ~ X
4. Evaluates model fit (R², F-test)
5. Diagnoses residuals
6. Calculates confidence interval for mean difference

**Key research questions:**
1. **How well do PCs predict gender classification?** (R²)
2. **Which PCs are important?** (Individual t-tests)
3. **How different are male vs female LD1 scores?** (CI)
4. **Is the model appropriate?** (Residual diagnostics)

---

## Expected Output

**Console output:**
```
============================================================
LOADING DATA AND PERSISTED MODELS...
============================================================
✓ Raw training data loaded: (1000, 38804)
✓ LDA, PCA, and Scaler models loaded from /models/

✓ Regression variables generated:
  - Y_reg (LD1 Score) shape: (1000,)
  - X_reg (50 PC Scores) shape: (1000, 50)

============================================================
SECTION 5.3: REGRESSION MODELING
============================================================
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.892
Model:                            OLS   Adj. R-squared:                  0.886
Method:                 Least Squares   F-statistic:                     157.2
Date:                Mon, 16 Nov 2025   Prob (F-statistic):          2.34e-285
Time:                        14:23:45   Log-Likelihood:                -1345.2
No. Observations:                1000   AIC:                             2792.
Df Residuals:                     949   BIC:                             3057.
Df Model:                          50                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const         -0.0002      0.033     -0.006      0.995      -0.065       0.065
PC1            0.8543      0.024     35.476      0.000       0.807       0.902
PC2            0.6231      0.028     22.254      0.000       0.568       0.678
...
==============================================================================

============================================================
GENERATING RESIDUAL PLOT...
============================================================
✓ Residual plot saved to: regression_residual_plot.png

============================================================
SECTION 5.2: CONFIDENCE INTERVAL (MEAN DIFFERENCE)
============================================================
  Mean Difference (Male - Female): -4.1084
  Degrees of Freedom (Welch's): 992.45
  Standard Error of the Difference: 0.0652
  Critical t-value (t_star): 1.9624
  95% Confidence Interval: [-4.2363, -3.9805]

============================================================
REGRESSION ANALYSIS COMPLETE.
============================================================
```

**Generated files:**
- `regression_residual_plot.png` - Residual diagnostic plot

---

## Practical Considerations

### Model Interpretation

**High R² (e.g., 0.89):**
- PC scores strongly predict LD1 scores
- Validation that PCA captures gender information
- 89% of LD1 variance explained by 50 PCs

**Significant predictors:**
- Early PCs (1-10) usually highly significant
- Later PCs may not be significant (noise)
- Aligns with eigenvalue decay

**Residual patterns:**
- Random scatter → Good model fit
- Pattern → Model misspecification

### Confidence Interval Insights

**Narrow CI:**
- Precise estimate
- Large sample size
- Small variability

**Doesn't contain 0:**
- Statistically significant difference
- Confirms t-test result from LDA script
- Quantifies effect size