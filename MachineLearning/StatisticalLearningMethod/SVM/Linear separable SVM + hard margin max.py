import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import numpy as np


# def get_data():
#     X, y = datasets.load_breast_cancer(return_X_y=True)
#     X_train, X_test, y_train, y_test =\
#         train_test_split(X, y, test_size=0.25, random_state=33)
#     return X_train, X_test, y_train, y_test


train_set1 = [[1, 1, 3], [1, 2, 2], [1, 3, 8], [1, 2, 6]]  # Positives
train_set2 = [[-1, 2, 1], [-1, 4, 1], [-1, 6, 2], [-1, 7, 3]]  # Negatives
train_set = train_set1 + train_set2  # concatenate

class LSSVM(object):
    def __init__(self):
        self.X = None
        self.y = None



