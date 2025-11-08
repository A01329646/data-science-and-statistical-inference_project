# Linear Discriminant Analysis (LDA) - Complete Theory Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Step-by-Step Theory Explanation](#step-by-step-theory-explanation)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Statistical Tests Used](#statistical-tests-used)
5. [Interpretation Guide](#interpretation-guide)

---

## Introduction

This document explains the **probability and statistics theory** behind each step of the LDA gender classification code. Linear Discriminant Analysis is a **supervised classification method** rooted in classical statistics.

### Key Difference: Supervised vs Unsupervised

- **PCA (Principal Component Analysis)**: Unsupervised - finds directions of maximum variance without using labels
- **LDA (Linear Discriminant Analysis)**: Supervised - finds directions that maximize class separation using labels

---

## Step-by-Step Theory Explanation

### Step 1: Data Loading and Preparation

```python
X_train, y_train = load_images_and_labels(TRAIN_DIR, max_images=1000)
X_val, y_val = load_images_and_labels(VAL_DIR, max_images=500)
```

#### Theory: Random Sampling and Population vs Sample

**Probability Concept**: Your dataset is a **sample** from a larger **population** of all possible face images.

- **Population (Ω)**: All possible face images of males and females
- **Sample (S)**: The subset we actually have (training + validation)
- **Random Variable**: Each pixel value is a random variable X₁, X₂, ..., Xₙ

**Why This Matters**: 
- We assume our sample is **representative** of the population
- Statistical inference allows us to make conclusions about the population from the sample
- The validation set tests if our findings **generalize** to unseen data

**Statistical Assumptions**:
1. **Independence**: Each image is independently sampled
2. **Identically Distributed**: All images come from the same distribution
3. **Representative**: Sample captures the true population characteristics

---

### Step 2: Standardization (Z-score Normalization)

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
```

#### Theory: Standardization and Z-scores

**Formula**: For each feature (pixel) xᵢ:
```
z = (x - μ) / σ
```
where:
- μ = mean of feature
- σ = standard deviation of feature

**Why Standardize?**

1. **Different Scales**: Pixel values might have different ranges
2. **Equal Importance**: Prevents features with larger scales from dominating
3. **Numerical Stability**: Improves mathematical computations

**Probability Distribution**:
- Original data: X ~ (μ, σ²)
- Standardized data: Z ~ (0, 1)
- This is a **location-scale transformation**

**Key Point**: We use training set statistics (μ, σ) for both train and validation to avoid **data leakage**.

**Statistical Property**:
```
E[Z] = 0  (expected value is zero)
Var[Z] = 1  (variance is one)
```

---

### Step 3: Linear Discriminant Analysis Fitting

```python
lda = LinearDiscriminantAnalysis(n_components=1)
X_train_lda = lda.fit_transform(X_train_scaled, y_train)
```

#### Theory: Fisher's Linear Discriminant

**Goal**: Find a linear projection that maximizes the separation between classes.

**Mathematical Objective**:

LDA seeks a vector **w** that maximizes the **Fisher criterion**:

```
J(w) = (w^T S_B w) / (w^T S_W w)
```

where:
- **S_B**: Between-class scatter matrix (variance between class means)
- **S_W**: Within-class scatter matrix (variance within each class)

**Intuition**: 
- Numerator: How far apart are the class means when projected onto w?
- Denominator: How much do points within each class spread when projected?
- We want: Large separation between classes, small spread within classes

#### Between-Class Scatter Matrix (S_B)

```
S_B = (μ₁ - μ₂)(μ₁ - μ₂)^T
```

where:
- μ₁ = mean vector of class 1 (males)
- μ₂ = mean vector of class 2 (females)

**Interpretation**: Measures how different the class centers are.

#### Within-Class Scatter Matrix (S_W)

```
S_W = Σ₁ + Σ₂
```

where:
- Σ₁ = covariance matrix of class 1
- Σ₂ = covariance matrix of class 2

**Interpretation**: Measures the variability within each class.

#### Optimal Solution

The optimal projection vector **w** is:

```
w = S_W^(-1) (μ₁ - μ₂)
```

**This is the direction that maximally separates the two classes!**

#### Number of Discriminants

**Important Statistical Property**:
- With K classes, LDA can find at most K-1 discriminant functions
- With 2 classes (male/female), we get exactly **1 discriminant (LD1)**
- This is why n_components=1 in our code

**Explained Variance Ratio**:
- Shows how much of the between-class variance is captured
- With 2 classes and 1 component, this is 100% (since we can't have more discriminants)

---

### Step 4: Projection onto Discriminant Axis

```python
X_val_lda = lda.transform(X_val_scaled)
```

#### Theory: Linear Transformation

Each data point **x** (a high-dimensional vector) is projected onto the discriminant axis:

```
y = w^T x
```

where:
- x ∈ ℝⁿ (original n-dimensional space)
- y ∈ ℝ (1-dimensional discriminant score)
- w = discriminant vector

**Probability Distribution**:

If the original data follows a multivariate normal distribution:
```
X ~ N(μ, Σ)
```

Then the projection follows a univariate normal:
```
Y = w^T X ~ N(w^T μ, w^T Σ w)
```

**This is crucial**: The discriminant scores should be approximately normally distributed within each class.

---

### Step 5: Statistical Significance Testing

```python
t_stat, p_value = stats.ttest_ind(train_male_scores, train_female_scores)
```

#### Theory: Two-Sample t-Test

**Null Hypothesis (H₀)**: μ_male = μ_female (no difference between genders)

**Alternative Hypothesis (H₁)**: μ_male ≠ μ_female (genders are different)

**Test Statistic**:
```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

where:
- x̄₁, x̄₂ = sample means
- s₁², s₂² = sample variances
- n₁, n₂ = sample sizes

**Degrees of Freedom** (Welch's t-test):
```
df ≈ (s₁²/n₁ + s₂²/n₂)² / [(s₁²/n₁)²/(n₁-1) + (s₂²/n₂)²/(n₂-1)]
```

**p-value Interpretation**:
- p < 0.001: Extremely strong evidence against H₀ (classes very different)
- p < 0.01: Strong evidence against H₀
- p < 0.05: Significant evidence against H₀ (conventional threshold)
- p ≥ 0.05: Insufficient evidence to reject H₀ (classes not distinguishable)

**Type I and Type II Errors**:
- **Type I Error (α)**: False positive - concluding classes differ when they don't
- **Type II Error (β)**: False negative - failing to detect real difference
- **Significance level α = 0.05** means we accept 5% chance of Type I error

---

### Step 6: Decision Boundary (Classification Threshold)

```python
threshold = (train_male_scores.mean() + train_female_scores.mean()) / 2
```

#### Theory: Bayes' Decision Rule

**Optimal Decision Rule** (assuming equal priors and equal covariance):

Classify as male if:
```
P(Male | x) > P(Female | x)
```

Using Bayes' theorem:
```
P(Male | x) = P(x | Male) × P(Male) / P(x)
```

**Assumptions**:
1. **Equal Priors**: P(Male) = P(Female) = 0.5
2. **Normal Distributions**: Each class follows a Gaussian distribution
3. **Equal Variances**: σ²_male ≈ σ²_female

**Under these assumptions**, the optimal threshold is:

```
threshold = (μ_male + μ_female) / 2
```

This is the **midpoint between class means**.

**Decision Rule**:
```
If y < threshold → Classify as Male
If y ≥ threshold → Classify as Female
```

(or vice versa depending on which class has higher mean)

#### Posterior Probability

The probability that a sample belongs to a class given its discriminant score:

```
P(Class = c | y) = exp(-(y - μ_c)² / 2σ_c²) / Σ_k exp(-(y - μ_k)² / 2σ_k²)
```

This is a **softmax function** for Gaussian distributions.

---

### Step 7: Classification Accuracy

```python
train_accuracy = np.mean(train_predictions == y_train)
val_accuracy = np.mean(val_predictions == y_val)
```

#### Theory: Estimation of Error Rate

**True Error Rate** (θ):
```
θ = P(Prediction ≠ True Class)
```

**Sample Accuracy** (θ̂):
```
θ̂ = (Number of Correct Predictions) / (Total Predictions)
```

**Statistical Properties**:

1. **θ̂ is an unbiased estimator of θ**:
   ```
   E[θ̂] = θ
   ```

2. **Standard Error**:
   ```
   SE(θ̂) = √[θ(1-θ) / n]
   ```

3. **95% Confidence Interval**:
   ```
   θ̂ ± 1.96 × SE(θ̂)
   ```

**Example**: If validation accuracy = 75% with n=500:
```
SE = √[0.75 × 0.25 / 500] = 0.0194
95% CI = [0.75 - 1.96×0.0194, 0.75 + 1.96×0.0194]
       = [0.712, 0.788]
```

**Interpretation**: We're 95% confident the true accuracy is between 71.2% and 78.8%.

---

### Step 8: Confusion Matrix Analysis

```python
val_male_correct = np.sum((val_predictions == 0) & (y_val == 0))
val_female_correct = np.sum((val_predictions == 1) & (y_val == 1))
```

#### Theory: Confusion Matrix and Performance Metrics

**Confusion Matrix**:
```
                Predicted
              Male   Female
Actual  Male   TP      FN
       Female  FP      TN
```

where:
- **TP** (True Positive): Correctly classified males
- **TN** (True Negative): Correctly classified females
- **FP** (False Positive): Females classified as males
- **FN** (False Negative): Males classified as females

**Key Metrics**:

1. **Sensitivity (True Positive Rate)**:
   ```
   Sensitivity = TP / (TP + FN)
   ```
   Probability of correctly identifying a male.

2. **Specificity (True Negative Rate)**:
   ```
   Specificity = TN / (TN + FP)
   ```
   Probability of correctly identifying a female.

3. **Precision (Positive Predictive Value)**:
   ```
   Precision = TP / (TP + FP)
   ```
   When we predict male, how often are we correct?

4. **F1 Score** (Harmonic mean of precision and recall):
   ```
   F1 = 2 × (Precision × Recall) / (Precision + Recall)
   ```

**Balanced Accuracy**:
```
Balanced Accuracy = (Sensitivity + Specificity) / 2
```

This is important when classes are imbalanced!

---

## Mathematical Foundations

### 1. Multivariate Normal Distribution

**Assumption**: Data from each class follows a multivariate normal distribution:

```
X ~ N(μ, Σ)
```

**Probability Density Function**:
```
f(x) = (1 / √((2π)^n |Σ|)) × exp(-½(x-μ)^T Σ^(-1) (x-μ))
```

where:
- n = dimensionality
- |Σ| = determinant of covariance matrix
- Σ^(-1) = inverse of covariance matrix

**Mahalanobis Distance**:
```
D²(x, μ) = (x-μ)^T Σ^(-1) (x-μ)
```

This measures distance accounting for correlations between features.

---

### 2. Maximum Likelihood Estimation (MLE)

**For each class**, we estimate parameters:

**Mean Estimator**:
```
μ̂ = (1/n) Σ xᵢ
```

**Covariance Estimator**:
```
Σ̂ = (1/n) Σ (xᵢ - μ̂)(xᵢ - μ̂)^T
```

**Properties**:
- These are **unbiased estimators**
- They are **consistent**: converge to true values as n → ∞
- They are **efficient**: have minimum variance among unbiased estimators

---

### 3. Fisher's Discriminant Ratio

**Objective Function**:
```
J(w) = (μ₁ᵖʳᵒʲ - μ₂ᵖʳᵒʲ)² / (s₁²ᵖʳᵒʲ + s₂²ᵖʳᵒʲ)
```

where:
- μᵢᵖʳᵒʲ = mean of class i after projection
- sᵢ²ᵖʳᵒʲ = variance of class i after projection

**In matrix form**:
```
J(w) = (w^T S_B w) / (w^T S_W w)
```

**Solution** (generalized eigenvalue problem):
```
S_B w = λ S_W w
```

The eigenvector with the largest eigenvalue λ is our discriminant!

---

### 4. Assumptions of LDA

LDA makes the following **statistical assumptions**:

1. **Linearity**: Decision boundary is linear
2. **Normality**: Each class follows a multivariate normal distribution
3. **Homoscedasticity**: Classes have equal covariance matrices (Σ₁ = Σ₂)
4. **Independence**: Samples are independently drawn

**When assumptions are violated**:
- Normality violation: LDA still often works (robust)
- Equal covariance violation: Consider Quadratic Discriminant Analysis (QDA)
- Non-linear boundary: Consider kernel methods or neural networks

---

## Statistical Tests Used

### 1. Two-Sample t-Test (Welch's t-test)

**Purpose**: Test if two population means are different.

**Hypotheses**:
- H₀: μ₁ = μ₂
- H₁: μ₁ ≠ μ₂

**Assumptions**:
- Samples are independent
- Data approximately normally distributed (robust to mild violations)
- Does NOT assume equal variances (Welch's version)

**When to use**: Compare male vs female discriminant scores

---

### 2. Effect Size (Cohen's d)

**Formula**:
```
d = (μ₁ - μ₂) / √[(s₁² + s₂²) / 2]
```

**Interpretation**:
- |d| < 0.2: Small effect
- |d| ≈ 0.5: Medium effect
- |d| > 0.8: Large effect

**Why it matters**: p-value tells if difference exists, effect size tells if it's meaningful.

---

### 3. Confidence Intervals

**For a mean**:
```
CI = x̄ ± t_(α/2, df) × (s / √n)
```

**For accuracy**:
```
CI = p̂ ± z_(α/2) × √[p̂(1-p̂) / n]
```

**Interpretation**: 95% CI means if we repeated the experiment many times, 95% of intervals would contain the true parameter.

---

## Interpretation Guide

### How to Read Results

#### 1. p-value < 0.001
✅ **Excellent separation**
- Classes are extremely different
- High confidence in classification ability
- Strong statistical evidence

#### 2. p-value between 0.001 and 0.05
✅ **Good separation**
- Classes are statistically different
- Moderate confidence in classification
- Acceptable for practical use

#### 3. p-value between 0.05 and 0.10
⚠️ **Marginal separation**
- Weak evidence of difference
- Classification may be unreliable
- Consider more data or better features

#### 4. p-value > 0.10
❌ **No separation**
- No statistical evidence of difference
- Classification not better than random
- Need fundamentally different approach

---

### Accuracy Interpretation

#### Random Baseline
With 2 balanced classes, random guessing gives 50% accuracy.

#### Meaningful Thresholds
- **50-60%**: Barely better than random (not useful)
- **60-70%**: Weak classification (needs improvement)
- **70-80%**: Moderate classification (acceptable for some applications)
- **80-90%**: Good classification (reliable)
- **90-100%**: Excellent classification (very reliable)

#### Statistical Significance of Accuracy

**Test if accuracy is better than chance** (50%):

**Binomial Test**:
```
z = (p̂ - 0.5) / √[0.5 × 0.5 / n]
```

If z > 1.96, accuracy is significantly better than random (p < 0.05).

---

### Distribution Overlap

**Visual Inspection**:
- **No overlap**: Perfect separation (accuracy → 100%)
- **Small overlap**: Good separation (accuracy 80-95%)
- **Moderate overlap**: Fair separation (accuracy 65-80%)
- **Large overlap**: Poor separation (accuracy 50-65%)
- **Complete overlap**: No separation (accuracy ≈ 50%)

**Quantitative Measure** (Overlap Coefficient):
```
OVL = ∫ min(f₁(x), f₂(x)) dx
```

where f₁, f₂ are the probability density functions.

---

## References

- Fisher, R. A. (1936). "The Use of Multiple Measurements in Taxonomic Problems"
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). "The Elements of Statistical Learning"
- Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"
- Duda, R. O., Hart, P. E., & Stork, D. G. (2001). "Pattern Classification"
