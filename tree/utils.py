"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    encoded_dataframe =pd.DataFrame(index=X.index)

    for column in X.columns:
        if (X[column].dtype == 'object') or (str(X[column].dtype == 'category')):
            unique_values = X[column].unique()
            for value in unique_values:
                encoded_dataframe[f"{column}_{value}"] = (X[column]==value).astype(int) #creating one hot encoding for each unique categorical value, for ex. outlook_sunny
        else:
            encoded_dataframe[column] = X[column]
    
    return encoded_dataframe #returning this new encoded dataframe to work on rather than original df

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    if((y.dtype == 'object') or (str(y.dtype) == 'category')):
        return False
    
    return True


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    _ ,counts = np.unique(Y, return_counts=True)
    total = len(Y)
    entropy = 0.0
    for count in counts:
        p =count/total
        entropy -=p*np.log2(p) # can add a little term 1e-12 so if class does not exists, it doesn't give -inf
    return entropy


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    _ ,counts = np.unique(Y, return_counts=True)

    total = len(Y)
    gini = 1.0
    for count in counts:
        p = count / total
        gini -= p ** 2
    return gini


def mse(y: pd.Series) -> float:
    mean = y.mean()
    error = (y-mean)
    square_error = error**2
    return square_error.mean()

CRITERION_FUNC = {
    "information_gain": entropy,  
    "gini_index": gini_index,     
    "mse": mse
}

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    criterion_func = CRITERION_FUNC[criterion]

    impurity_before_split = criterion_func(Y)
    total = len(Y)

    impurity_after_split = 0.0
    
    for value in attr.unique():
        sub_Y = Y[attr==value]
        weight = len(sub_Y) / total #the proportion of data that  falls in this subset
        impurity_after_split +=weight*criterion_func(sub_Y) #refer page 24 of PDF for this calculation

    return impurity_before_split - impurity_after_split



def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """
    best_feature =None
    best_threshold =None
    best_gain =-float('inf')

    for feature in features:
        col =X[feature]

        if((col.dtype == 'object') or (str(col.dtype) == 'category')):
            # If discrete feature-use info gain directly
            gain = information_gain(y, col, criterion)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = None
        else:
            # If real feature-try splits at midpoints between sorted unique values
            sorted_vals = np.unique(col)
            for i in range(len(sorted_vals) - 1):
                thresh = (sorted_vals[i] + sorted_vals[i+1]) / 2.0
                left_mask =(col <= thresh) 
                right_mask =(col > thresh)
                if((left_mask.sum() == 0) or (right_mask.sum() == 0)):
                    continue #checking if all samples are on one side then invalid split, either right or left, don't do anything just look for next split

                if criterion == 'mse':
                    impurity_before = mse(y)
                    impurity_after = (
                        (left_mask.sum()/len(y))*mse(y[left_mask]) +
                        (right_mask.sum()/len(y))*mse(y[right_mask])
                    )
                    gain = impurity_before - impurity_after
                else:
                    # Classification
                    if criterion=='entropy':
                        impurity_before =entropy(y)
                        impurity_left = entropy(y[left_mask])
                        impurity_right =entropy(y[right_mask])
                        impurity_after = ((left_mask.sum()/len(y))*impurity_left)+((right_mask.sum()/len(y))*impurity_right)
                    else:
                        impurity_before =gini_index(y)
                        impurity_left = gini_index(y[left_mask])
                        impurity_right =gini_index(y[right_mask])
                        impurity_after = ((left_mask.sum()/len(y))*impurity_left)+((right_mask.sum()/len(y))*impurity_right)

                    gain = impurity_before - impurity_after

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = thresh

    return best_feature, best_threshold


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    column = X[attribute]

    if ((column.dtype == 'object') or (str(column.dtype) == 'category')):
        # Discrete feature-left all matching values and right different ones
        mask_left = (column == value)
        mask_right = (column != value)
    else:
        # Real feature- left = rows <= threshold
        mask_left = (column <= value)
        mask_right = (column > value)

    X_left =X[mask_left].reset_index(drop=True)
    y_left =y[mask_left].reset_index(drop=True)
    X_right =X[mask_right].reset_index(drop=True)
    y_right =y[mask_right].reset_index(drop=True)

    return X_left,y_left,X_right,y_right
