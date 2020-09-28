import time
import numpy as np
import pandas as pd
import progressbar as pgb
from collections import Counter

'''
This file will use the C4.5 algorithms to predict the MNIST dataset.

    It will be divided into the following 4 sections:

    Section[1] - Preprocess Data
            Load data and split train into train set and cross-validation set
            Image Binarization

    Section[2] - Feature Selection
            calculate the shanno entropy
            calculate the conditional  entropy
            calculate the information gain ratio

    Section[3] - Create Decision Tree
            C4.5 algorithm
            
    Section[4] - Pruning
            calculate the empirical entropy in each node
            pruning the leaf node

    Section[4] - Predict
'''


# Section[1]
def load_data(trainfile, testfile):
    """
        load data
        Divide the train set into train set and cross-validation set  in ratio.
    """
    train = pd.read_csv(trainfile, header=None)
    test = pd.read_csv(testfile, header=None)
    train_label = train.iloc[:, 0]
    test_label = test.iloc[:, 0]
    train = train.drop(labels=train.columns[0], axis=1)
    test = test.drop(labels=test.columns[0], axis=1)
    print('data is loaded')
    return np.array(train), np.array(train_label), np.array(test), np.array(test_label)


def normalize(data, threshold):
    """
        Binarized data, let data becoming 0-1 distribution.
    """
    m = data.shape[0]
    n = np.array(data).shape[1]
    progress = pgb.ProgressBar()
    for i in progress(range(m)):
        for j in range(n):
            if data[i, j] > int(threshold):
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
    """
        calculate Shanno entropy of x
        H(X) = -Σ(pi * logpi )
    """
    classNum = len(x)  # total number of sample
    cnt_x = Counter(x)  # calculate the number of each category
    entropy = float(0)
    for i in cnt_x:
        p = cnt_x[i] / classNum  # probability of each category
        logp = np.log2(p)
        entropy -= p * logp
    return entropy


def _cal_conditionalEnt(x, y):
    # x means condition -> feature vector
    # y means dataset category(such as MNIST classNum = 10) -> label
    """
    H(Y|X) = Σ( pi * H(Y|X = xi) )
    """
    classNum = len(x)
    cnt_x = Counter(x)  # due to image binarization, there are only two category(0-1)
    entropy = float(0)
    for i in cnt_x:
        temp_y = y[np.where(x == i)]
        cond_ent = _cal_entropy(temp_y)
        entropy += (cnt_x[i] / classNum) * cond_ent
    return entropy


def _cal_igr(x, y):
    """
    gR(D,A) = (H(D) - H(D|A)) / H(D|A)
    """
    return (_cal_entropy(y) - _cal_conditionalEnt(x, y)) / _cal_conditionalEnt(x, y)


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

    (3) otherwise, calculate the information gain ratio of each feature(A) to D,
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


class Tree(object):
    def __init__(self, node_type, Class=None, feature=None):
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
    leaf = 'leaf'
    internal = 'internal'

    # Step 1
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(leaf, Class=label_set.pop())

    # Step 2
    # (max_class, max_len) = \
    #     max([(i, len(list(filter(lambda x: x == i, train_label))))
    #          for i in range(10)], key=lambda x: x[1])
    max_class = Counter(train_label).most_common(1)[0][0]
    if len(features) == 0:
        return Tree(leaf, Class=max_class)

    # Step 3
    _feature = 0
    _igr = 0
    for item in features:
        A = np.array(train[:, item])  # feature vector
        igr = _cal_igr(A, train_label)
        if igr > _igr:
            _igr, _feature = igr, item

    # Step 4
    if _igr < epsilon:
        return Tree(leaf, Class=max_class)

    # Step 5
    sub_features = list(filter(lambda x: x != _feature, features))
    tree = Tree(internal, feature=_feature)

    # Step 6
    feature_vec = np.array(train[:, _feature])
    feature_val = set([feature_vec[i] for i in range(feature_vec.shape[0])])
    for val in feature_val:
        sample_index = []
        for i in range(len(train_label)):
            if train[i][_feature] == val:
                sample_index.append(i)
        sub_train = train[sample_index]
        sub_train_label = train_label[sample_index]
        sub_tree = CreateTree(sub_train, sub_train_label, sub_features, epsilon)
        tree.add(val, sub_tree)

    return tree

# Section[4]
'''

'''


# Section[5]
def predict(test, tree):
    result = []
    for features in test:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    print('Prediction complete')
    return np.array(result)


def main():
    train, train_label, test, test_label \
        = load_data('mnist_train.csv', 'mnist_test.csv')
    train = normalize(train, threshold=50)
    test = normalize(test, threshold=50)
    tree = CreateTree(train, train_label, features=[i for i in range(train.shape[1])], epsilon=0.1)
    print('Decision Tree Created')
    test_pred = predict(test, tree)
    score = (test_pred == test_label).sum() / len(test_label)
    print("the accuracy score is {:.4f}".format(score))


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Time cost {0} min {1:.2f}s'.format(int((end - start) // 60), (end - start) % 60))