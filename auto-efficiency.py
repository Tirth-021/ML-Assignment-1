import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,na_values='?',
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# Compare the performance of your model with the decision tree module from scikit learn

#Data cleaning

df =data.dropna() #removing rows with missing values
df = df.drop('car name',axis=1) #car name is not relevant 
X = df.drop('mpg',axis=1) 
y= df['mpg']

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size=0.2, random_state=42)

sk_tree = DecisionTreeRegressor(max_depth=6,random_state=42) #max depth same as our custom decision tree
sk_tree.fit(X_train,y_train)

y_pred= sk_tree.predict(X_test)

sl_mae = mae(y_pred,y_test)
print(f"MAE for sklearn model is {sl_mae}")


custom_tree = DecisionTree(criterion="information_gain")
custom_tree.fit(X_train,y_train)

y_pred= custom_tree.predict(X_test)

custom_mae = mae(y_pred,y_test)
print(f"MAE for custom tree is {custom_mae}")