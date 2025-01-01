"""
utils.py

This file contains helper functions for data loading, preprocessing,
and any miscellaneous utilities used by your quantum-classical models.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_iris_data(test_size=0.2, random_state=42):
    """
    Loads the Iris dataset, splits into train/test, and returns normalized arrays.
    
    Returns:
        x_train (ndarray): Training features
        x_test  (ndarray): Testing features
        y_train (ndarray): Training labels
        y_test  (ndarray): Testing labels
    """
    data = load_iris()
    X = data['data']    # shape (150, 4)
    y = data['target']  # shape (150,)

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return x_train, x_test, y_train, y_test


def one_hot_encode(labels, num_classes=3):
    """
    Convert integer labels into one-hot vectors.

    Args:
        labels (ndarray): 1D array of integer labels.
        num_classes (int): Number of classes.

    Returns:
        one_hot (ndarray): 2D array of shape (len(labels), num_classes)
    """
    one_hot = np.zeros((len(labels), num_classes))
    for idx, val in enumerate(labels):
        one_hot[idx, val] = 1.0
    return one_hot
