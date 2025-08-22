from typing import Union
import pandas as pd
import numpy as np


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size, "Predictions and labels must be of same size"
    assert y_hat.size > 0, "Input series must not be empty"
    return (y_hat == y).sum() / y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Predictions and labels must be of same size"
    assert y_hat.size > 0, "Input series must not be empty"

    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    denom = tp + fp
    return tp / denom if denom != 0 else 0.0    


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Predictions and labels must be of same size"
    assert y_hat.size > 0, "Input series must not be empty"

    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    denom = tp + fn
    return tp / denom if denom != 0 else 0.0


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    assert y_hat.size == y.size, "Predictions and labels must be of same size"
    assert y_hat.size > 0, "Input series must not be empty"
    errors = (y_hat - y).astype(float)
    return np.sqrt(np.mean(errors ** 2))


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "Predictions and labels must be of same size"
    assert y_hat.size > 0, "Input series must not be empty"
    errors = (y_hat - y).astype(float)
    return np.mean(np.abs(errors))
