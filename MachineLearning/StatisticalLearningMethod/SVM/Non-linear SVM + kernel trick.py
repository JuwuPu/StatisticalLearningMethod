import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
import random
import numpy as np
import time

class SVM(object):
    def __init__(self, features=None, labels=None, alpha=None, b=None):
        self.features = features
        self.labels = labels
        self.alpha =alpha
        self.b = b

    def predict(self, test, test_labels):
        test_labels[test_labels == 0] = -1
        pred = []
        a = []
        for item in test:
            temp = sum(kernel(item, self.features) * self.labels * self.alpha) + self.b
            a.append(temp)
            pred.append(np.sign(temp))
        return sum(pred == test_labels) / len(test_labels)


def get_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.25, random_state=33)
    return X_train, X_test, y_train, y_test

def kernel(x,z,sigma=1,p=2):
    """
    Gaussian kernel function
    """
    if len(z.shape) == 2:
        return np.exp(-(np.sum((z-x)**2, axis=1)**0.5)) / (2 * sigma**2)
    else:
        return np.exp(-(np.sum((z-x)**2)**0.5)) / (2 * sigma**2)

    """
    Polynomial kernel function
    """
    # return (np.inner(x, z)+1)**p

def SMO_train(features, labels, max_passes=100, C=5, tol=0.01):
    """
    max_passes      # max # of times to iterate over alpha's without changing
    C               # regularization parameter
    tol             # numerical tolerance
    """
    passes = 0
    alpha = np.zeros(len(labels))
    b = 0.0
    labels[labels == 0] = -1
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(len(labels)):
            error_i = sum(kernel(features[i], features) * alpha * labels) + b - labels[i]
            if not (labels[i] * error_i < -tol and alpha[i] < C)\
                    or (labels[i] * error_i > tol and alpha[i] > 0):
                temp_list = [i for i in range(len(labels))]
                temp_list.pop(i)
                j = random.choice(temp_list)
                error_j = sum(kernel(features[j], features) * alpha * labels) + b - labels[j]
                temp_alpha_i_old = alpha[i]
                temp_alpha_j_old = alpha[j]
                if labels[i] != labels[j]:
                    L = max(0, alpha[j]-alpha[i])
                    H = min(C, alpha[j]-alpha[i]+C)
                else:
                    L = max(0, alpha[j]+alpha[i]-C)
                    H = min(C, alpha[i]+alpha[j])
                if L == H:
                    continue
                eta = 2 * kernel(features[i], features[j]) - \
                    kernel(features[i], features[i]) - kernel(features[j], features[j])
                if eta >= 0:
                    continue
                alpha[j] = alpha[j] - labels[j] * (error_i - error_j) / eta
                if alpha[j] > H:
                    alpha[j] = H
                elif alpha[j] < L:
                    alpha[j] = L
                else:
                    pass
                if abs(alpha[j] - temp_alpha_j_old) < 10**(-5):
                    continue
                alpha[i] = alpha[i] + labels[i] * labels[j] * (temp_alpha_j_old - alpha[j])

                b1 = b - error_i - labels[i]*(alpha[i]-temp_alpha_i_old)* \
                     kernel(features[i],features[i]) - labels[j]*(alpha[j]-temp_alpha_j_old) * \
                     kernel(features[i],features[j])
                b2 = b - error_j - labels[i]*(alpha[i]-temp_alpha_i_old) * \
                     kernel(features[i],features[j]) - labels[j]*(alpha[j]-temp_alpha_j_old) * \
                     kernel(features[j],features[j])
                if 0 < alpha[i] < C:
                    b = b1
                elif 0 < alpha[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
            # End if
        # End for
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    svm = SVM()
    svm.labels = labels
    svm.alpha = alpha
    svm.b = b
    svm.features = features
    return svm

time1 = time.perf_counter()
X_train, X_test, y_train, y_test = get_data()

time2 = time.perf_counter()
print('loading data cost %s s' % (time2 - time1))
svm = SMO_train(X_train, y_train)
time3 = time.perf_counter()
print('training cost %s s' % (time3-time2))

score = svm.predict(X_test, y_test)
time4 = time.perf_counter()
print('predicting cost %s s' % (time4-time3))
print('the score is %f' %(score))












