import time
import numpy as np
import pandas as pd
import progressbar as pgb
from collections import Counter
from sklearn.model_selection import train_test_split
'''
This file will use the ID3 algorithms to predict the MNIST dataset.

    It will be divided into the following 5 sections:
    
    Section[1] - Preprocess Data
            Load data and split train into train set and cross-validation set
            Image Binarization
               
    Section[2] - Feature Selection
            calculate the shanno entropy
            calculate the conditional  entropy
            calculate the Information Gain
    
    Section[3] - Create Decision Tree
            ID3 algorithm
    
    Section[4] - Pruning
    
    Section[5] - Predict
'''

# Section[1]
def load_data(trainfile, testfile):
    '''
    load data
    Divide the train set into train set and cross-validation set  in ratio.
    '''
    train = pd.read_csv(trainfile, header=None)
    test = pd.read_csv(testfile, header=None)
    np.array(train)
    train, cv = train_test_split(train)
    m, n = train.shape
    p, q = cv.shape
    train_label = train[:, 0]
    train = train[:, 1:].reshape(m, n-1)
    cv_label = cv[:, 0]
    cv = cv[:, 1:].reshape(p, q-1)
    test_label = test.iloc[:, 0]
    test = test.drop(labels=test.columns[0], axis=1)
    print('data is loaded')
    return train, train_label, \
           cv, cv_label, \
           np.array(test), np.array(test_label)

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
    # x means condition -> feature vector
    # y means dataset category(such as MNIST classNum = 10) -> label
    '''
    H(Y|X) = Σ( pi * H(Y|X = xi) )
    '''
    classNum = len(x)
    cnt_x = Counter(x)  # due to image binarization, there are only two category(0-1)
    entropy = float(0)
    for i in cnt_x:
        temp_y = y[np.where(x == i)]
        cond_ent = _cal_entropy(temp_y)
        entropy += (cnt_x[i] / classNum) * cond_ent
    return entropy

def _cal_infoGain(x, y):
    '''
    g(D,A) = H(D) - H(D|A)
    '''
    Hd = _cal_entropy(y)
    condEnt = _cal_conditionalEnt(x, y)
    infoGain = Hd - condEnt
    return infoGain


# Section[3]
'''
ID3 algorithm: The core of ID3 is to apply information gain criteria to 
               select features on each node of the decision tree, 
               and build the decision tree recursively. 
               
    INPUT: train set D, feature set A， Threshold ε
    OUTPUT： decision tree T
    
    (1) if all instance in D belong to the same category Ck, then T is Single node tree,
        and use class Ck as the class tag of the node's category, return T.
    
    (2) if A = Ø, then T is Single node tree. And take the class Ck with the largest 
        number of instances in D as the class mark of the node, return T.
    
    (3) otherwise, calculate the information gain of each feature(A) to D,
        select the feature with maximum information gain(Ag).
    
    (4) if the info gain of Ag less than threshold(ε), then let the T as single node tree,
        And take the class Ck with the largest number of instances in D as the class mark of the node, return T.
        
    (5) otherwise, for each possible value ai of Ag, divide D into several non-empty 
        subsets Di according to Ag = ai, and use the largest class in Di as a mark to 
        construct sub-nodes, which are composed of nodes and their sub-nodes Tree T, 
        return T.
        
    (6) for the i-th sub-node, use Di as the training set and A-{Ag} as the feature set, 
        recursively call step (1) ~ step (5) to get the subtree Ti, and return Ti.
        
'''
class Tree():
    def __init__(self, node_type, Class = None, feature = None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add(self, value, tree):
        self.dict[value] = tree

    def predict(self, features):
        if self.node_type == 'leaf':
            return self.Class
        tree = self.dict[features[self.feature]]
        return tree.predict(features)

def CreateTree(train, train_label, features, epsilon):
    classNum = len(Counter(train_label))
    leaf = 'leaf'
    internal = 'internal'

    # Step 1












def main():
    train, train_label, cv, cv_label, test, test_label\
        = load_data('mnist_train.csv', 'mnist_test.csv')
    train = normalize(train)
    cv = normalize(cv)
    test = normalize(test)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time cost {0} min {1:.2f}s'.format(int((end - start)//60), (end - start) % 60))