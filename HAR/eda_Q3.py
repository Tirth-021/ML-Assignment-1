import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tsfel   

# Activity labels
ACTIVITIES = {
    1: "Walking",
    2: "Walking Upstairs",
    3: "Walking Downstairs",
    4: "Sitting",
    5: "Standing",
    6: "Laying"
}

# colors
colors = {
    1: 'red', 
    2: 'blue', 
    3: 'purple', 
    4: 'orange', 
    5: 'green', 
    6: 'brown'
}


X = np.load("training_data.npy")   # shape (N, 500, 3)
y = np.load("training_labels.npy") # shape (N,)

# PCA Total Acceleration
acc_magnitude = np.sqrt(np.sum(X**2, axis=2))   # (N, 500)

acc_pca = PCA(n_components=2)
X_acc_pca = acc_pca.fit_transform(acc_magnitude)

# PCA TSFEL Features 
cfg = tsfel.get_features_by_domain("statistical")

feature_list = []
for i in range(X.shape[0]):
    df_feat = tsfel.time_series_features_extractor(cfg, 
                                                   acc_magnitude[i].reshape(-1,1),
                                                   fs=50, 
                                                   verbose=0)
    feature_list.append(df_feat.values[0])
tsfel_features = np.array(feature_list)

tsfel_pca = PCA(n_components=2)
X_tsfel_pca = tsfel_pca.fit_transform(tsfel_features)

# PCA Dataset Features
features_train = np.loadtxt("UCI HAR Dataset/train/X_train.txt")
labels_train   = np.loadtxt("UCI HAR Dataset/train/y_train.txt").astype(int)

features_test  = np.loadtxt("UCI HAR Dataset/test/X_test.txt")
labels_test    = np.loadtxt("UCI HAR Dataset/test/y_test.txt").astype(int)

X_full = np.vstack((features_train, features_test))
y_full = np.concatenate((labels_train, labels_test))

dataset_pca = PCA(n_components=2)
X_dataset_pca = dataset_pca.fit_transform(X_full)

# Matplotlib Plotting 
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for label in np.unique(y):
    axes[0].scatter(X_acc_pca[y==label, 0], X_acc_pca[y==label, 1],
                    c=colors[label], s=10, label=ACTIVITIES[label], alpha=0.7, marker="x")
axes[0].set_title("PCA: Acceleration Magnitude")

for label in np.unique(y):
    axes[1].scatter(X_tsfel_pca[y==label, 0], X_tsfel_pca[y==label, 1],
                    c=colors[label], s=10, label=ACTIVITIES[label], alpha=0.7, marker="x")
axes[1].set_title("PCA: TSFEL Features")

for label in np.unique(y_full):
    axes[2].scatter(X_dataset_pca[y_full==label, 0], X_dataset_pca[y_full==label, 1],
                    c=colors[label], s=10, label=ACTIVITIES[label], alpha=0.7, marker="x")
axes[2].set_title("PCA: Dataset Features")
axes[2].legend(loc="best", fontsize=8)

plt.tight_layout()
plt.show()
