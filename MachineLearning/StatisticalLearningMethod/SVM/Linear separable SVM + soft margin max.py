import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import numpy as np


def get_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test