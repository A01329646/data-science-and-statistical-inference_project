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

# ------------------------------
# 1. Parameters
# ------------------------------
TRAIN_DIR = "dataset/train"      # Folder containing training images
VAL_DIR   = "dataset/validation" # Folder containing validation images
N_COMPONENTS = 50                # Number of PCA components to keep

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
                labels.append("male")
            elif filename.lower().startswith("female"):
                labels.append("female")
            else:
                labels.append("unknown")
        except Exception as e:
            print(f"Skipping {filename}: {e}")
    return np.array(data), np.array(labels)

# ------------------------------
# 3. Load datasets
# ------------------------------
print("Loading images...")
X_train, y_train = load_images_and_labels(TRAIN_DIR, max_images=1000)
X_val,   y_val   = load_images_and_labels(VAL_DIR,   max_images=500)

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# ------------------------------
# 4. Standardize the data (important before PCA)
# ------------------------------
scaler = StandardScaler(with_std=True)
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)  # use same mean/std

# ------------------------------
# 5. Fit PCA on training data
# ------------------------------
pca = PCA(n_components=N_COMPONENTS, random_state=42)
pca.fit(X_train_scaled)

print(f"Explained variance ratio (first 10): {pca.explained_variance_ratio_[:10]}")
print(f"Total variance explained by {N_COMPONENTS} components: {np.sum(pca.explained_variance_ratio_):.2f}")

# ------------------------------
# 6. Transform both sets using the SAME PCA
# ------------------------------
X_train_pca = pca.transform(X_train_scaled)
X_val_pca   = pca.transform(X_val_scaled)

# ------------------------------
# 7. Visualization (PCA colored by gender)
# ------------------------------
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

print("Plotting PCA projection for training set...")
plot_pca_projection(X_train_pca, y_train, "Training Set: PCA Projection by Gender")

print("Plotting PCA projection for validation set...")
plot_pca_projection(X_val_pca, y_val, "Validation Set: PCA Projection by Gender")

# ------------------------------
# 8. Save PCA model and scaler
# ------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump({"scaler": scaler, "pca": pca}, "models/pca_model.pkl")
print("✅ PCA model and scaler saved to models/pca_model.pkl")
