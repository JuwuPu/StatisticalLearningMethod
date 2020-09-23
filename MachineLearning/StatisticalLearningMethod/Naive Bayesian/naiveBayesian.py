import pandas as pd
import numpy as np


def load_data(trainfile, testfile):
    train = pd.read_csv(trainfile, header=None)
    test = pd.read_csv(testfile, header=None)
    train_label = train.iloc[:, 0]
    test_label = test.iloc[:, 0]
    train = train.drop(labels=train.columns[0], axis=1)
    test = test.drop(labels=test.columns[0], axis=1)
    return np.array(train), np.array(train_label), np.array(test), np.array(test_label)

train, train_label, test, test_label = load_data(
    'D:\PythonDocument'
    '\MachineLearning'
    '\StatisticalLearningMethod'
    '\DataSources'
    '\MNIST'
    '\mnist_train.csv',
    'D:\PythonDocument'
    '\MachineLearning'
    '\StatisticalLearningMethod'
    '\DataSources'
    '\MNIST'
    '\mnist_test.csv'
)