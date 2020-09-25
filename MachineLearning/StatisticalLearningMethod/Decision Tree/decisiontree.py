import pandas as pd
import numpy as np
import time
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


def main():
    train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time cost {0} min {1:.2f}s'.format(int((end - start)//60), (end - start) % 60))