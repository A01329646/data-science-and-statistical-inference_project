# LDA Gender Classification - Documentation Index

Welcome to the comprehensive documentation for the Linear Discriminant Analysis (LDA) gender classification project!

## 📚 Documentation Files

### 1. **LDA_THEORY_EXPLAINED.md** 📖
**Complete theoretical foundation with probability & statistics**

**Contents:**
- Step-by-step theory for each code section
- Mathematical formulas and derivations
- Statistical tests explained (t-test, confidence intervals)
- Probability distributions and assumptions
- Detailed interpretation guidelines

---

### 2. **LDA_QUICK_REFERENCE.md** ⚡
**Quick formulas and key concepts at a glance**

**Contents:**
- All important formulas on one page
- Quick interpretation tables (p-values, accuracy)
- Code-to-theory mapping
- Example calculations

---

### 3. **LDA_VISUAL_GUIDE.md** 🎨
**Visual intuition with ASCII art and examples**

**Contents:**
- Visual representations of how LDA works
- Before/after diagrams
- Distribution plots explained
- Real example walkthrough
- Common error patterns
- Decision boundary visualization

---

## 📂 Project Files Overview

### Main Analysis Scripts
- **`lda_gender_classification.py`** - Complete LDA analysis with statistical tests
- **`compare_pca_lda.py`** - Side-by-side comparison of PCA vs LDA
- **`pca_dataset.py`** - Original PCA implementation (for reference)

### Documentation
- **`LDA_THEORY_EXPLAINED.md`** - Complete theoretical explanation
- **`LDA_QUICK_REFERENCE.md`** - Quick reference card
- **`LDA_VISUAL_GUIDE.md`** - Visual intuition guide
- **`PCA_vs_LDA_Guide.md`** - Comparison and project recommendations
- **`README.md`** - Image converter documentation

### Support Files
- **`requirements.txt`** - Python dependencies
- **`image_converter.py`** - Utility for image preprocessing

---

## 🎯 Key Concepts Covered

### Probability & Statistics Theory
✅ Multivariate normal distributions  
✅ Maximum likelihood estimation  
✅ Hypothesis testing (t-tests)  
✅ p-values and statistical significance  
✅ Confidence intervals  
✅ Effect size (Cohen's d)  
✅ Type I and Type II errors  
✅ Bayes' decision theory  
✅ Fisher's discriminant criterion  

### Linear Algebra
✅ Matrix operations  
✅ Eigenvalues and eigenvectors  
✅ Covariance matrices  
✅ Linear transformations  
✅ Scatter matrices  

### Machine Learning Concepts
✅ Supervised vs unsupervised learning  
✅ Train/validation split  
✅ Generalization  
✅ Overfitting  
✅ Cross-validation  
✅ Confusion matrix  
✅ Performance metrics  

---

## 📊 What Each File Generates

### lda_gender_classification.py
**Generates:**
1. `lda_gender_distribution.png` - Distribution comparison
2. `lda_boxplot_comparison.png` - Train vs validation boxplots
3. `lda_classification_results.png` - Misclassification visualization
4. `lda_analysis_report.txt` - Detailed text report
5. `models/lda_model.pkl` - Saved model

**Terminal Output:**
- Loading statistics
- LDA fitting information
- Statistical test results (t-test, p-value)
- Classification accuracy
- Summary of findings

---

### compare_pca_lda.py
**Generates:**
1. `pca_vs_lda_comparison.png` - 2×2 comparison plot

**Terminal Output:**
- PCA separation metrics
- LDA separation metrics
- Recommendation on which to use