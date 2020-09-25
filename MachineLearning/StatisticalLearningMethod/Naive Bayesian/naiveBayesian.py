import pandas as pd
import numpy as np
import time
from collections import Counter
import progressbar as pgb

def load_data(trainfile, testfile):
    train = pd.read_csv(trainfile, header=None)
    test = pd.read_csv(testfile, header=None)
    train_label = train.iloc[:, 0]
    test_label = test.iloc[:, 0]
    train = train.drop(labels=train.columns[0], axis=1)
    test = test.drop(labels=test.columns[0], axis=1)
    print('data is loaded')
    return np.array(train), np.array(train_label), np.array(test), np.array(test_label)

def normalize(data):
    '''
    Binarized data, let data becoming 0-1 distribution.
    '''
    m = data.shape[0]
    n = np.array(data).shape[1]
    progress = pgb.ProgressBar()
    for i in progress(range(m)):
        for j in range(n):
            if data[i, j] != 0:
                data[i, j] = 1
            else:
                data[i, j] = 0
    print('Data is binarization')
    return data

def bayesianModel(train, train_label):
    '''
    This section contain two parts
    Part 1 : prior probability
    Part 2 : posterior probability (Bernoulli distribution)
    '''
    # Part 1
    totalNum = train.shape[0]
    classNum = Counter(train_label)
    # laplace smoothing
    prioriP = np.array([(classNum[i] + 1) / (totalNum + 10) for i in range(10)])

    # Part 2
    posteriorNum = np.zeros((10, train.shape[1]))
    posteriorP = np.zeros((10, train.shape[1]))
    for j in range(10):
        posteriorNum[j] = train[np.where(train_label == j)].sum(axis=0)
        # laplace smoothing
        posteriorP[j] = (posteriorNum[j] + 1) / (classNum[j] + 2)
    print('Probability calculated')
    return prioriP, posteriorP

def predict(test, test_label, prioriP, posteriorP):
    print('Predict starting')
    predict_test = np.zeros(test.shape[0])
    progress = pgb.ProgressBar()
    for i in progress(range(test.shape[0])):
        probability = np.zeros(10)
        for j in range(10):
            temp = sum([np.log(1 - posteriorP[j][m])
                        if test[i][m] == 0 else np.log(posteriorP[j][m])
                        for m in range(test.shape[1])])
            probability[j] = np.array(np.log(prioriP[j]) + temp)
        predict_test[i] = np.argmax(probability)
    accuracy = (predict_test == test_label).sum() / test.shape[0]
    print('Prediction completed')
    return accuracy

def main():
    train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')
    train = normalize(train)
    test = normalize(test)
    prioriP, posteriorP = bayesianModel(train, train_label)
    accuracy = predict(test, test_label, prioriP, posteriorP)
    print("The accuracy of MNIST for the Naive Bayes Classifier is {0:.2f}%".format(accuracy*100))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time cost {0}min{1:.2f}s'.format(int((end - start)//60), (end - start)%60))