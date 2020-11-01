import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split

import numpy as np
import time

'''
The core of the boost method is that the judge of many experts better than only one for a complicated task.

The main questions of boost method need to be answered  are followed:
    1. How to change the weight of training sample or probability distribution in each iteration.
        ---> Increase the weight of misclassified points in the previous round of weak classifiers. 
    2. How to combine the weakly classifiers into a strongly classifier.
        ---> Using a weighted majority voting method.
'''

def get_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.25, random_state=33)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    return X_train, X_test, y_train, y_test

class AdaBoost(object):
    def init_args(self,X, Y):
        self.X = X
        self.Y = Y
        self.D = np.ones(len(self.Y)) / len(self.Y)
        self.classifier_set = {}
        self.m = 100        # the number of weakly classifier
        self.error = 0.0    # the Classification error rate

    def add(self, index, classifier):
        self.classifier_set[index] = classifier

    def train(self, features, labels):
        self.init_args(features, labels)




if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
