import pandas as pd
import numpy as np
import time
from collections import Counter

def load_data(trainfile, testfile):
    train = pd.read_csv(trainfile, header=None)
    test = pd.read_csv(testfile, header=None)
    train_label = train.iloc[:, 0]
    test_label = test.iloc[:, 0]
    train = train.drop(labels=train.columns[0], axis=1)
    test = test.drop(labels=test.columns[0], axis=1)
    return np.array(train), np.array(train_label), np.array(test), np.array(test_label)

def bayesianModel(train, train_label):
    '''
    This section contain two parts
    Part 1 : prior probability
    Part 2 : posterior probability
    '''
    # Part 1
    totalNum = train.shape[0]
    classNum = Counter(train_label)
    prioriP = np.array([classNum[i]/totalNum for i in range(10)])

    # Part 2
    posteriorNum = np.zeros((10, train.shape[1]))
    posteriorP = np.zeros((10, train.shape[1]))
    for j in range(10):
        posteriorNum[j] = train[np.where(train_label == j)].sum(axis=0)
        # laplace smoothing
        posteriorP[j] = (posteriorNum[j] + 1) / (classNum[j] + 2)
    return prioriP, posteriorP

def main():
    train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time cost {0:.2f}s'.format(end - start))