import pandas as pd
import numpy as np
import time
import progressbar as pgb
from collections import Counter
'''
This file will use the ID3 algorithms to predict the MNIST dataset.

    It will be divided into the following 5 sections:
    
    Section[1] - Preprocess Data
            Load data
            Image Binarization
            Split data into train set and cross-validation set
                 
    Section[2] - Feature Selection
            calculate the shanno entropy
            calculate the conditional  entropy
            calculate the Information Gain
    
    Section[3] - Create Decision Tree
    
    Section[4] - Pruning
    
    Section[5] - Predict
'''

# Section[1]
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


# Section[2]
'''
The information gain g(D, A) of feature A to training set D is defined as
the difference between the empirical entropy H(D) of set D and the empirical
conditional entropy H(D|A) of D under the given condition of feature A,
namely
                    g(D,A) = H(D) - H(D|A) [mutual information].

'''
def _cal_entropy(x):
    '''
        calculate Shanno entropy of x
        H(X) = -Σ(pi * logpi )
    '''
    classNum = len(x)   # total number of sample
    cnt_x = Counter(x)  # calculate the number of each category
    entropy = float(0)
    for i in cnt_x:
        p = cnt_x[i] / classNum    # probability of each category
        logp = np.log2(p)
        entropy += -p * logp
    return entropy

def _cal_conditionalEnt(x, y):
    # x means dataset category
    # y means
    '''
    H(Y|X) = Σ( pi * H(Y|X = xi) )
    '''
    classNum = len(x)
    cnt_x = Counter(x)
    entropy = float(0)
    for i in cnt_x:
        yNum = y[np.where(x == i)].sum()
    return entropy

def _cal_infoGain( x, y):
    '''
    g(D,A) = H(D) - H(D|A)
    '''
    Hd = _cal_entropy(y)
    condEnt = _cal_conditionalEnt(x, y)
    infoGain = Hd - condEnt
    return infoGain












def main():
    train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')
    train = normalize(train)
    test = normalize(test)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time cost {0} min {1:.2f}s'.format(int((end - start)//60), (end - start) % 60))