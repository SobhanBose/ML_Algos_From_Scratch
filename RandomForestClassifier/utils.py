import random
import numpy as np
import pandas as pd

def train_valid_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int=10) -> tuple:
    """
    Performs train-validation split

    Parameters
    ----------
    X: pd.DataFrame
        Input Features
    y: pd.DataFrame
        Target Label
    test_size: float, optional
        Size of validation set, by default 0.2
    random_state: int
        Random State

    Returns
    -------
    train_data, valid_data
    """
    
    random.seed(random_state)

    test_size = round(test_size * len(df))
    indices = df.index.tolist()

    test_index = random.sample(indices, k=test_size)

    train_data = df.drop(test_index)
    valid_data = df.iloc[test_index]

    return train_data, valid_data


def get_accuracy_score(y_true: pd.DataFrame, y_pred: np.ndarray) -> float:
    """
    Computes the accuracy of a classification model

    Parameters
    ----------
    y_true : pd.DataFrame
        True Labels
    y_pred : np.ndarray
        Predicted Labels

    Returns
    -------
    accuracy_score: float
        The accuracy of the model
    """

    y_true = y_true.to_numpy().flatten()
    correct_predictions = np.sum(y_true == y_pred)
    return correct_predictions/len(y_true)