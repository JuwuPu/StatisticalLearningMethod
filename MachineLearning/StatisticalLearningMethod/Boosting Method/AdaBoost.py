import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

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
    def __init__(self):
        self.clf_set = {}

    def add(self, index, clf, alpha):
        self.clf_set[index] = (clf, alpha)

    def train(self, X_train, y_train, M=20):
        w = np.ones(len(X_train)) / len(X_train)
        # weak_clf = DecisionTreeClassifier(max_depth=1)
        for i in range(M):
            weak_clf = DecisionTreeClassifier(max_depth=1)
            clf_i = weak_clf.fit(X_train, y_train, sample_weight=w)
            pred_train_i = clf_i.predict(X_train)
            error_sample = [int(x) for x in (pred_train_i != y_train)]
            if sum(error_sample) / len(X_train) < 0.01:
                break
            error_i = sum(error_sample*w)
            alpha_i = 1/2*np.log(1/error_i - 1)
            self.add(i, clf_i, alpha_i)
            w = np.multiply(w, np.exp(-1 * alpha_i * y_train * pred_train_i))
            w = w / sum(w)


    def predict(self, X_test):
        temp_ = np.zeros(len(X_test))
        for i in range(len(self.clf_set)):
            temp_ += self.clf_set[i][0].predict(X_test) * self.clf_set[i][1]
        return np.sign(temp_)

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_data()
    AdaBoost = AdaBoost()
    AdaBoost.train(X_train,y_train)
    pred = AdaBoost.predict(X_test)
    score = sum(pred == y_test) / len(y_test)
    print("the AdaBoost's score is %.4f" % score)