# LDA Quick Reference - Key Formulas & Concepts

## 🎯 Core Objective

**Find vector w that maximizes**:
```
J(w) = Between-Class Variance / Within-Class Variance
     = (w^T S_B w) / (w^T S_W w)
```

---

## 📐 Key Matrices

### Between-Class Scatter (S_B)
```
S_B = (μ₁ - μ₂)(μ₁ - μ₂)^T
```
Measures: How far apart are the class means?

### Within-Class Scatter (S_W)
```
S_W = Σ₁ + Σ₂
```
Measures: How much variance within each class?

### Optimal Direction
```
w = S_W^(-1) (μ₁ - μ₂)
```

---

## 📊 Statistical Tests

### Two-Sample t-Test
```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
```

**Interpretation**:
- p < 0.001: ★★★ Extremely significant
- p < 0.01:  ★★☆ Very significant  
- p < 0.05:  ★☆☆ Significant
- p ≥ 0.05:  ☆☆☆ Not significant

---

## 🎲 Decision Rule (Bayes)

**Optimal Threshold** (equal priors & variances):
```
threshold = (μ_male + μ_female) / 2
```

**Classification**:
```
If LD1_score < threshold → Class 1
If LD1_score ≥ threshold → Class 2
```

---

## 📈 Performance Metrics

### Accuracy
```
Accuracy = Correct Predictions / Total Predictions
```

### Confidence Interval (95%)
```
CI = Accuracy ± 1.96 × √[Accuracy × (1-Accuracy) / n]
```

### Effect Size (Cohen's d)
```
d = (μ₁ - μ₂) / √[(s₁² + s₂²) / 2]
```
- Small: |d| < 0.2
- Medium: |d| ≈ 0.5
- Large: |d| > 0.8

---

## 🔢 Standardization (Z-score)

```
z = (x - μ) / σ
```

**Result**: Mean = 0, Std = 1

---

## 🎓 Key Assumptions

1. ✓ **Multivariate Normality**: Each class ~ N(μ, Σ)
2. ✓ **Equal Covariance**: Σ₁ ≈ Σ₂
3. ✓ **Independence**: Samples drawn independently
4. ✓ **Linearity**: Decision boundary is linear

---

## 💡 Quick Interpretation Guide

| p-value | Separation | Action |
|---------|------------|--------|
| < 0.001 | Excellent | ✅ Trust results |
| 0.001-0.05 | Good | ✅ Use with confidence |
| 0.05-0.10 | Marginal | ⚠️ Be cautious |
| > 0.10 | None | ❌ Rethink approach |

| Accuracy | Quality | Interpretation |
|----------|---------|----------------|
| 90-100% | Excellent | Highly reliable |
| 80-90% | Good | Reliable |
| 70-80% | Fair | Acceptable |
| 60-70% | Weak | Barely useful |
| 50-60% | Poor | Not better than random |

---

## 🔍 Confusion Matrix

```
                Predicted
              Male   Female
Actual  Male   TP      FN
       Female  FP      TN
```

### Derived Metrics
```
Sensitivity = TP / (TP + FN)    [Recall for males]
Specificity = TN / (TN + FP)    [Recall for females]
Precision   = TP / (TP + FP)    [Male prediction accuracy]
F1-Score    = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 🎯 Fisher's Discriminant (Intuition)

**Goal**: Project high-dimensional data to 1D line such that:
- ✅ Class means are far apart (maximize)
- ✅ Within-class spread is small (minimize)

**Analogy**: Finding the best viewing angle to see two clusters as separate.

---

## 📚 LDA vs PCA

| Aspect | PCA | LDA |
|--------|-----|-----|
| Type | Unsupervised | Supervised |
| Uses labels? | ❌ No | ✅ Yes |
| Maximizes | Total variance | Class separation |
| # Components | Up to n features | Up to k-1 classes |
| Best for | Visualization | Classification |

---

## ⚡ One-Line Summary

**LDA finds the direction that maximally separates classes by maximizing the ratio of between-class to within-class variance.**

---

## 🧮 Example Calculation

Given two classes with:
- Male: μ₁ = -2.5, σ₁ = 1.2, n₁ = 100
- Female: μ₂ = 1.8, σ₂ = 1.1, n₂ = 120

**Threshold**:
```
threshold = (-2.5 + 1.8) / 2 = -0.35
```

**t-statistic**:
```
t = (-2.5 - 1.8) / √(1.2²/100 + 1.1²/120)
  = -4.3 / 0.149
  = -28.86
```

**Conclusion**: p << 0.001, extremely significant separation!

**Classification Rule**:
```
If LD1 < -0.35 → Male
If LD1 ≥ -0.35 → Female
```

---

## 💻 Code-to-Theory Mapping

| Code | Theory |
|------|--------|
| `StandardScaler()` | z = (x - μ) / σ |
| `LinearDiscriminantAnalysis()` | w = S_W^(-1)(μ₁ - μ₂) |
| `lda.fit_transform()` | y = w^T x |
| `stats.ttest_ind()` | t = (x̄₁ - x̄₂) / SE |
| `threshold = (m1 + m2)/2` | Bayes optimal decision |
| `accuracy` | θ̂ = correct / total |