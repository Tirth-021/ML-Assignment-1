import numpy as np
import matplotlib.pyplot as plt
import tsfel
import pandas as pd

X = np.load("training_data.npy")  # shape (N, 500, 3)

acc_magnitude = np.sqrt(np.sum(X**2, axis=2))  # shape (N, 500)

# TSFEL Features
cfg = tsfel.get_features_by_domain("statistical")  # choose statistical features
tsfel_list = []
for i in range(acc_magnitude.shape[0]):
    df_feat = tsfel.time_series_features_extractor(cfg,
                                                   acc_magnitude[i].reshape(-1, 1),
                                                   fs=50,
                                                   verbose=0)
    tsfel_list.append(df_feat.values[0])
tsfel_features = np.array(tsfel_list)

# Save TSFEL feature names
tsfel_feature_names = list(df_feat.columns)

# Correlation matrix for TSFEL
tsfel_corr = np.corrcoef(tsfel_features, rowvar=False)

plt.figure(figsize=(8, 6))
plt.matshow(tsfel_corr, cmap="bwr", fignum=1, vmin=-1, vmax=1)
plt.colorbar(label="Correlation strength")
plt.title("Correlation map - TSFEL features", pad=20)
plt.show()

# Identify highly correlated TSFEL pairs
cutoff_val = 0.85
tsfel_idx = np.argwhere(np.abs(tsfel_corr) > cutoff_val)
tsfel_pairs = [(tsfel_feature_names[i], tsfel_feature_names[j], tsfel_corr[i, j]) 
               for i, j in tsfel_idx if i < j]

print("TSFEL features with correlation above 0.85:")
print(tsfel_pairs if tsfel_pairs else "None found")

# Dataset Features 
X_train = np.loadtxt("UCI HAR Dataset/train/X_train.txt")
X_test  = np.loadtxt("UCI HAR Dataset/test/X_test.txt")
X_full  = np.vstack((X_train, X_test))

# Load dataset feature names
with open("UCI HAR Dataset/features.txt") as f:
    dataset_feature_names = [line.strip().split()[1] for line in f]

# Correlation matrix for dataset features
dataset_corr = np.corrcoef(X_full, rowvar=False)

plt.matshow(dataset_corr, cmap="bwr", fignum=2, vmin=-1, vmax=1)
plt.colorbar(label="Correlation strength")
plt.title("Correlation map - Dataset features", pad=20)
plt.show()

# Identify highly correlated dataset pairs
dataset_idx = np.argwhere(np.abs(dataset_corr) > cutoff_val)
dataset_pairs = [(dataset_feature_names[i], dataset_feature_names[j], dataset_corr[i, j]) 
                 for i, j in dataset_idx if i < j]

print("Dataset features with correlation above 0.85:")
print(dataset_pairs if dataset_pairs else "None found")

# Observation 
if tsfel_pairs or dataset_pairs:
    print("\nObservation: Strongly correlated features found — possible redundancy.")
else:
    print("\nObservation: No redundant features found (no strong correlations).")
