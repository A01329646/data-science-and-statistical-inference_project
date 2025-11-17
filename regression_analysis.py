# ==============================
# REGRESSION ANALYSIS SCRIPT (PCA -> LDA)
# ==============================
# This script performs the core statistical modeling for the project.
# It loads pre-fitted PCA and LDA models to generate the predictor (X)
# and target (Y) variables, then fits an OLS regression model.

import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from PIL import Image
from scipy import stats
import statsmodels.api as sm

# ------------------------------
# 1. Parameters
# ------------------------------
TRAIN_DIR = "dataset/train"
MODEL_LDA_PATH = "models/lda_model.pkl"
MODEL_PCA_PATH = "models/pca_model.pkl"

# ------------------------------
# 2. Helper Function: Image Data Loading
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
            # Assumes images are pre-processed to 178x218 as per report
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

print("=" * 60)
print("LOADING DATA AND PERSISTED MODELS...")
print("=" * 60)

# ------------------------------
# 3. Load Raw Data and Fitted Models
# ------------------------------
# Load raw training data
X_train, y_train = load_images_and_labels(TRAIN_DIR, max_images=1000)
print(f"✓ Raw training data loaded: {X_train.shape}")

# Load persisted models
try:
    lda_data = joblib.load(MODEL_LDA_PATH)
    pca_data = joblib.load(MODEL_PCA_PATH)
except FileNotFoundError:
    print(f"Error: Persisted model files not found.")
    print(f"Ensure 'lda_model.pkl' and 'pca_model.pkl' are present in the /models/ directory.")
    exit()

# Extract model objects
scaler = lda_data["scaler"]
lda = lda_data["lda"]
pca = pca_data["pca"]
print("✓ LDA, PCA, and Scaler models loaded from /models/")

# ------------------------------
# 4. Generate Regression Variables (X_reg, Y_reg)
# ------------------------------
# Re-scale training data using the loaded scaler
X_train_scaled = scaler.transform(X_train)

# Generate Y (Dependent) Variable: The LD1 Score
Y_reg = lda.fit_transform(X_train_scaled, y_train).flatten() # .flatten() to ensure 1D vector

# Generate X (Independent) Variables: The 50 PC Scores
X_reg = pca.transform(X_train_scaled)

print(f"✓ Regression variables generated:")
print(f"  - Y_reg (LD1 Score) shape: {Y_reg.shape}")
print(f"  - X_reg (50 PC Scores) shape: {X_reg.shape}")

# =======================================================
# 5. REGRESSION ANALYSIS 
# =======================================================
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

# ----- Summary Interpretation Guide -----
# R-squared / Adj. R-squared: This is the primary finding (Section 5.3)
# Prob (F-statistic): The overall model p-value (Section 5.3)
# coef: The coefficient values (B0, B1, ... B50)
# P>|t|: The p-values for each coefficient (Section 5.3)
# ----------------------------------------

# =======================================================
# 6. RESIDUAL ANALYSIS 
# =======================================================
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
os.makedirs("outputs/regression", exist_ok=True)
plt.savefig("outputs/regression/regression_residual_plot.png", dpi=300, bbox_inches='tight')
print("✓ Residual plot saved to: outputs/regression/regression_residual_plot.png")


# =======================================================
# 7. CONFIDENCE INTERVAL 
# =======================================================
print("\n" + "=" * 60)
print("SECTION 5.2: CONFIDENCE INTERVAL (MEAN DIFFERENCE)")
print("=" * 60)

# Load saved statistics from the LDA model object
# (Values correspond to lda_analysis_report.txt)
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
df_den = ( (s1**2/n1)**2 / (n1-1) ) + ( (s2**2/n2)**2 / (n2-1) )
df = df_num / df_den

# Find the critical t-value (t*) for 95% confidence
# stats.t.ppf(0.975) gives the t-value for the 97.5th percentile
t_star = stats.t.ppf(0.975, df)

lower_bound = mean_diff - t_star * se_diff
upper_bound = mean_diff + t_star * se_diff

print(f"  Mean Difference (Male - Female): {mean_diff:.4f}")
print(f"  Degrees of Freedom (Welch's): {df:.2f}")
print(f"  Standard Error of the Difference: {se_diff:.4f}")
print(f"  Critical t-value (t_star): {t_star:.4f}")
print(f"  95% Confidence Interval: [{lower_bound:.4f}, {upper_bound:.4f}]")

print("\n" + "=" * 60)
print("REGRESSION ANALYSIS COMPLETE.")
print("=" * 60)