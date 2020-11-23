import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor as DTR
import numpy as np


def get_data():
    X, y = datasets.fetch_california_housing(return_X_y=True)
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test


def train(x, y, number_clf):
    GBDT = {}
    for i in range(number_clf):
        weak_clf = DTR(max_depth=6)
        weak_clf.fit(x, y)
        pred = weak_clf.predict(x)
        residual = y - pred
        GBDT[i] = weak_clf
        y = residual



X_train, X_test, y_train, y_test = get_data()

# tree = DTR()
# tree.fit(X_train, y_train)
# predict = tree.score(X_test, y_test)
print(1)





#
# ones_matrix = np.ones((len(X_train), len(X_train)))
# utm = np.triu(ones_matrix)              # Upper triangular matrix
# ltm = ones_matrix - utm                 # Lower triangular matrix
# c1 = np.dot(y_train, utm) / [i for i in range(len(X_train))]
# c2 = np.dot(y_train, ltm) / list.reverse([i for i in range(len(X_train))])
# error = sum((y_train-c1)**2 + (y_train-c2)**2)


