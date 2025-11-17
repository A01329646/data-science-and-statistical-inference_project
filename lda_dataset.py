# ==============================
# LDA for Gender Classification (Statistical Approach)
# ==============================
# Linear Discriminant Analysis is a statistical method that finds
# the linear combination of features that best separates two or more classes.
# Unlike PCA (unsupervised), LDA uses class labels to maximize separation.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
from PIL import Image

# ------------------------------
# 1. Parameters
# ------------------------------
TRAIN_DIR = "dataset/train"      # Folder containing training images
VAL_DIR   = "dataset/validation" # Folder containing validation images
N_COMPONENTS = 1                 # LDA with 2 classes can have max 1 component

# ------------------------------
# 2. Helper: Load images and labels
# ------------------------------
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

# ------------------------------
# 3. Load datasets
# ------------------------------
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

# ------------------------------
# 4. Standardize the data (important for LDA)
# ------------------------------
print("\n" + "=" * 60)
print("STANDARDIZING DATA")
print("=" * 60)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
print("✓ Data standardized using training set statistics")

# ------------------------------
# 5. Fit LDA on training data
# ------------------------------
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

# ------------------------------
# 6. Transform validation set using the SAME LDA
# ------------------------------
X_val_lda = lda.transform(X_val_scaled)
print(f"✓ Validation set transformed to LDA space")

# ------------------------------
# 7. Statistical Analysis - Distribution Comparison
# ------------------------------
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

# Validation set statistics
val_male_scores = X_val_lda[y_val == 0, 0]
val_female_scores = X_val_lda[y_val == 1, 0]

print(f"\nValidation Set - LD1 Score Statistics:")
print(f"  Male:   mean={val_male_scores.mean():.4f}, std={val_male_scores.std():.4f}")
print(f"  Female: mean={val_female_scores.mean():.4f}, std={val_female_scores.std():.4f}")

# ------------------------------
# Goodness-of-fit test (Shapiro-Wilk)
# ------------------------------
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


# ------------------------------
# 8. Simple Classification using Fisher's threshold
# ------------------------------
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

# Confusion matrix for validation set
val_male_correct = np.sum((val_predictions == 0) & (y_val == 0))
val_male_total = np.sum(y_val == 0)
val_female_correct = np.sum((val_predictions == 1) & (y_val == 1))
val_female_total = np.sum(y_val == 1)

print(f"\nValidation Set Breakdown:")
print(f"  Male:   {val_male_correct}/{val_male_total} correct ({val_male_correct/val_male_total:.2%})")
print(f"  Female: {val_female_correct}/{val_female_total} correct ({val_female_correct/val_female_total:.2%})")

# ------------------------------
# 9. Visualization - Distribution Comparison
# ------------------------------
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# Plot 1: Training Set Distribution
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

# Plot 2: Validation Set Distribution
plt.subplot(1, 2, 2)
plt.hist(val_male_scores, bins=30, alpha=0.6, label='Male', color='blue', density=True)
plt.hist(val_female_scores, bins=30, alpha=0.6, label='Female', color='red', density=True)
plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
plt.xlabel('LD1 Score')
plt.ylabel('Density')
plt.title('Validation Set: LDA Projection by Gender')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lda_gender_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved distribution plot: lda_gender_distribution.png")

# Plot 3: Box plot comparison
plt.figure(figsize=(10, 6))
data_to_plot = [train_male_scores, train_female_scores, val_male_scores, val_female_scores]
labels = ['Train\nMale', 'Train\nFemale', 'Val\nMale', 'Val\nFemale']
colors = ['lightblue', 'lightcoral', 'blue', 'red']

box_parts = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True)
for patch, color in zip(box_parts['boxes'], colors):
    patch.set_facecolor(color)

plt.ylabel('LD1 Score')
plt.title('LDA Score Distribution by Gender and Dataset')
plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('lda_boxplot_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved boxplot: lda_boxplot_comparison.png")

# Plot 4: Scatter plot with misclassifications highlighted
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
# Correct predictions in normal colors
train_correct_mask = train_predictions == y_train
plt.scatter(np.arange(len(X_train_lda))[train_correct_mask & (y_train == 0)], 
           X_train_lda[train_correct_mask & (y_train == 0), 0], 
           c='blue', alpha=0.3, s=10, label='Male (correct)')
plt.scatter(np.arange(len(X_train_lda))[train_correct_mask & (y_train == 1)], 
           X_train_lda[train_correct_mask & (y_train == 1), 0], 
           c='red', alpha=0.3, s=10, label='Female (correct)')
# Misclassifications in black
plt.scatter(np.arange(len(X_train_lda))[~train_correct_mask], 
           X_train_lda[~train_correct_mask, 0], 
           c='black', marker='x', s=50, label='Misclassified')
plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('LD1 Score')
plt.title(f'Training Set (Accuracy: {train_accuracy:.2%})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Correct predictions in normal colors
val_correct_mask = val_predictions == y_val
plt.scatter(np.arange(len(X_val_lda))[val_correct_mask & (y_val == 0)], 
           X_val_lda[val_correct_mask & (y_val == 0), 0], 
           c='blue', alpha=0.3, s=10, label='Male (correct)')
plt.scatter(np.arange(len(X_val_lda))[val_correct_mask & (y_val == 1)], 
           X_val_lda[val_correct_mask & (y_val == 1), 0], 
           c='red', alpha=0.3, s=10, label='Female (correct)')
# Misclassifications in black
plt.scatter(np.arange(len(X_val_lda))[~val_correct_mask], 
           X_val_lda[~val_correct_mask, 0], 
           c='black', marker='x', s=50, label='Misclassified')
plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label='Threshold')
plt.xlabel('Sample Index')
plt.ylabel('LD1 Score')
plt.title(f'Validation Set (Accuracy: {val_accuracy:.2%})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lda_classification_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved classification results: lda_classification_results.png")

# ------------------------------
# 10. Save LDA model and statistics
# ------------------------------
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

# Save detailed report
with open("lda_analysis_report.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("LDA GENDER CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Training Set Size: {len(X_train)} samples\n")
    f.write(f"  - Male: {np.sum(y_train == 0)}\n")
    f.write(f"  - Female: {np.sum(y_train == 1)}\n\n")
    f.write(f"Validation Set Size: {len(X_val)} samples\n")
    f.write(f"  - Male: {np.sum(y_val == 0)}\n")
    f.write(f"  - Female: {np.sum(y_val == 1)}\n\n")
    f.write(f"LDA Statistics:\n")
    f.write(f"  - Components: {N_COMPONENTS}\n")
    f.write(f"  - Explained variance: {lda.explained_variance_ratio_[0]:.4f}\n")
    f.write(f"  - Decision threshold: {threshold:.4f}\n\n")
    f.write(f"Training Set LD1 Scores:\n")
    f.write(f"  - Male mean: {train_male_scores.mean():.4f} (std: {train_male_scores.std():.4f})\n")
    f.write(f"  - Female mean: {train_female_scores.mean():.4f} (std: {train_female_scores.std():.4f})\n")
    f.write(f"  - t-statistic: {t_stat:.4f}\n")
    f.write(f"  - p-value: {p_value:.6f}\n\n")
    f.write(f"Classification Results:\n")
    f.write(f"  - Training accuracy: {train_accuracy:.2%}\n")
    f.write(f"  - Validation accuracy: {val_accuracy:.2%}\n")
    f.write(f"  - Male classification rate (val): {val_male_correct/val_male_total:.2%}\n")
    f.write(f"  - Female classification rate (val): {val_female_correct/val_female_total:.2%}\n")

print("✓ Detailed report saved to lda_analysis_report.txt")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print("\nKey Findings:")
print(f"  • LDA found {'significant' if p_value < 0.05 else 'NO significant'} separation between genders")
print(f"  • Training accuracy: {train_accuracy:.2%}")
print(f"  • Validation accuracy: {val_accuracy:.2%}")
print(f"  • This is a statistical approach using Fisher's Linear Discriminant")
print("\nFiles generated:")
print("  1. lda_gender_distribution.png")
print("  2. lda_boxplot_comparison.png")
print("  3. lda_classification_results.png")
print("  4. lda_analysis_report.txt")
print("  5. models/lda_model.pkl")