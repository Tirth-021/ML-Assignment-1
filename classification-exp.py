import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

np.random.seed(42)

# Code given in the question
X_np, y_np = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X_np[:, 0], X_np[:, 1], c=y_np)

# Write the code for Q2 a) and b) below. Show your results.
X = pd.DataFrame(X_np, columns=["feature1", "feature2"])
y = pd.Series(y_np, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
) #stratify=y

tree = DecisionTree(criterion="information_gain")
tree.fit(X_train, y_train)

y_pred_raw = tree.predict(X_test)
y_pred = (y_pred_raw >= 0.5).astype(int) 

print("=== Part (a) Results ===")
print("Accuracy: ", accuracy(y_pred, y_test))

for cls in np.unique(y):
    prec = precision(y_pred,y_test,cls)
    rec = recall(y_pred,y_test,cls)
    print(f"Class {cls} -> Precision: {prec:.4f}, Recall: {rec:.4f}")


################ Part (b): #####################

outer_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #tried with Kfold and SKfold as well
depth_range = range(1, 11)  # depths to search over
outer_scores = []
best_depths = []

for train_idx, test_idx in outer_kf.split(X,y):
    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
    
    inner_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    depth_scores = []

    for d in depth_range:
        inner_fold_scores = []
        
        for inner_train_idx, inner_val_idx in inner_kf.split(X_train_outer,y_train_outer):
            X_train_inner, X_val_inner = X_train_outer.iloc[inner_train_idx], X_train_outer.iloc[inner_val_idx]
            y_train_inner, y_val_inner = y_train_outer.iloc[inner_train_idx], y_train_outer.iloc[inner_val_idx]
            
            tree = DecisionTree(criterion='information_gain', max_depth=d)
            tree.fit(X_train_inner, y_train_inner)
            
            y_pred_inner = (tree.predict(X_val_inner) >= 0.5).astype(int)
            inner_fold_scores.append(accuracy(y_pred_inner,y_val_inner))
        
        depth_scores.append(np.mean(inner_fold_scores))
    
    # Best depth
    best_d = depth_range[int(np.argmax(depth_scores))]
    best_depths.append(best_d)
    
    # Train with best depth and evaluate on outer fold
    final_tree = DecisionTree(criterion="information_gain")
    final_tree.fit(X_train, y_train)
    y_pred = final_tree.predict(X_test)
    outer_scores.append(accuracy(y_pred, y_test))

print("\n=== Part (b) Nested CV Results ===")
for i, d in enumerate(best_depths, 1):
    print(f"Fold {i}: Best Depth = {d}, Accuracy = {outer_scores[i-1]:.4f}")

print(f"Overall Mean Accuracy (Nested CV): {np.mean(outer_scores):.4f}")

