# ==============================
# Quick Comparison: PCA vs LDA
# ==============================
# This script runs both PCA and LDA side-by-side to compare results

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
from PIL import Image

# ------------------------------
# Load Data
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
            arr = np.array(img, dtype=np.float32) / 255.0
            data.append(arr.flatten())
            if filename.lower().startswith("male"):
                labels.append(0)
            elif filename.lower().startswith("female"):
                labels.append(1)
            else:
                labels.append(-1)
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    
    data = np.array(data)
    labels = np.array(labels)
    valid_mask = labels != -1
    return data[valid_mask], labels[valid_mask]

print("Loading data...")
X_train, y_train = load_images_and_labels("dataset/train", max_images=1000)
print(f"Loaded {len(X_train)} training samples")

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ------------------------------
# PCA (Unsupervised)
# ------------------------------
print("\n--- PCA Analysis ---")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

# Calculate separation
male_pca = X_pca[y_train == 0]
female_pca = X_pca[y_train == 1]
pca_t_stat, pca_p_value = stats.ttest_ind(male_pca[:, 0], female_pca[:, 0])
print(f"PC1 t-test p-value: {pca_p_value:.6f}")

# ------------------------------
# LDA (Supervised)
# ------------------------------
print("\n--- LDA Analysis ---")
lda = LinearDiscriminantAnalysis(n_components=1)
X_lda = lda.fit_transform(X_train_scaled, y_train)

# Calculate separation
male_lda = X_lda[y_train == 0]
female_lda = X_lda[y_train == 1]
lda_t_stat, lda_p_value = stats.ttest_ind(male_lda, female_lda)
print(f"LD1 t-test p-value: {lda_p_value:.6f}")

# ------------------------------
# Visual Comparison
# ------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# PCA - Scatter
axes[0, 0].scatter(male_pca[:, 0], male_pca[:, 1], alpha=0.3, s=10, c='blue', label='Male')
axes[0, 0].scatter(female_pca[:, 0], female_pca[:, 1], alpha=0.3, s=10, c='red', label='Female')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
axes[0, 0].set_title(f'PCA: 2D Projection\n(Unsupervised - No Label Info Used)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# PCA - Distribution
axes[0, 1].hist(male_pca[:, 0], bins=30, alpha=0.6, label='Male', color='blue', density=True)
axes[0, 1].hist(female_pca[:, 0], bins=30, alpha=0.6, label='Female', color='red', density=True)
axes[0, 1].set_xlabel('PC1 Score')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title(f'PCA: PC1 Distribution\np-value: {pca_p_value:.6f}')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# LDA - Scatter (1D, so plot against index)
male_indices = np.where(y_train == 0)[0]
female_indices = np.where(y_train == 1)[0]
axes[1, 0].scatter(male_indices, male_lda, alpha=0.3, s=10, c='blue', label='Male')
axes[1, 0].scatter(female_indices, female_lda, alpha=0.3, s=10, c='red', label='Female')
threshold = (male_lda.mean() + female_lda.mean()) / 2
axes[1, 0].axhline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('LD1 Score')
axes[1, 0].set_title(f'LDA: Discriminant Scores\n(Supervised - Uses Label Info)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# LDA - Distribution
axes[1, 1].hist(male_lda, bins=30, alpha=0.6, label='Male', color='blue', density=True)
axes[1, 1].hist(female_lda, bins=30, alpha=0.6, label='Female', color='red', density=True)
axes[1, 1].axvline(threshold, color='green', linestyle='--', linewidth=2, label=f'Threshold={threshold:.2f}')
axes[1, 1].set_xlabel('LD1 Score')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title(f'LDA: LD1 Distribution\np-value: {lda_p_value:.6f}')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_vs_lda_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Comparison saved: pca_vs_lda_comparison.png")
plt.show()

# ------------------------------
# Summary
# ------------------------------
print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print(f"\nPCA (Unsupervised):")
print(f"  - Uses: Maximum variance direction")
print(f"  - PC1 separates genders: {'YES' if pca_p_value < 0.05 else 'NO'} (p={pca_p_value:.6f})")
print(f"  - Male PC1 mean: {male_pca[:, 0].mean():.4f}")
print(f"  - Female PC1 mean: {female_pca[:, 0].mean():.4f}")
print(f"  - Separation: {abs(male_pca[:, 0].mean() - female_pca[:, 0].mean()):.4f}")

print(f"\nLDA (Supervised):")
print(f"  - Uses: Maximum class separation")
print(f"  - LD1 separates genders: {'YES' if lda_p_value < 0.05 else 'NO'} (p={lda_p_value:.6f})")
print(f"  - Male LD1 mean: {male_lda.mean():.4f}")
print(f"  - Female LD1 mean: {female_lda.mean():.4f}")
print(f"  - Separation: {abs(male_lda.mean() - female_lda.mean()):.4f}")

print(f"\n{'='*60}")
print(f"RECOMMENDATION: Use {'LDA' if lda_p_value < pca_p_value else 'PCA'}")
print(f"  LDA is designed for classification and should perform better")
print(f"  when the goal is to separate known classes.")
print(f"{'='*60}")
