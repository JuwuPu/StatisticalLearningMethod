import time
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import cv2
import sys
import copy
sys.setrecursionlimit(1000000)

# Section[1]
def load_data(trainfile, testfile):
    """
        load data
        Divide the train set into train set and cross-validation set  in ratio.
    """
    raw_train = pd.read_csv(trainfile, header=None)
    raw_test = pd.read_csv(testfile, header=None)
    train = raw_train.values
    test = raw_test.values
    train_features = train[0::, 1::]
    train_label = train[::, 0]
    test_features = test[0::, 1::]
    test_label = test[::, 0]
    train, cv , train_label, cv_label = train_test_split(train_features,train_label, test_size=0.33, random_state=42)
    return train, train_label, \
           cv, cv_label, \
           test_features, test_label


def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img,50,1,cv2.THRESH_BINARY,cv_img)
    return cv_img


def binaryzation_features(trainset):
    features = []
    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        img_b = binaryzation(cv_img)
        features.append(img_b)

    features = np.array(features)
    features = np.reshape(features,(-1,784))
    return features

# Section[2]
def Gini(D):
    sampleNum = len(D)
    cnt_D = Counter(D)
    p2 = 0
    for i in cnt_D:
        p = cnt_D[i] / sampleNum
        p2 += p**2
    return 1 - p2

def joint_Gini(D, A):
    sampleNum = len(A)
    cnt_A = Counter(A)
    _gini = 0
    for i in cnt_A:
        temp_D = D[np.where(A == i)]
        cond_p = Gini(temp_D)
        _gini += cond_p * cnt_A[i] / sampleNum
    return _gini

# Section[3]

class Tree(object):
    def __init__(self, node_type, Class=None, feature_index=None, max_Class=None):
        self.node_type = node_type
        self.dict = {}
        self.Class = Class
        self.max_Class = max_Class
        self.feature_index = feature_index

    def add(self, val, tree):
        self.dict[val] = tree

    # def predict(self, features):
    #     if self.node_type == 'leaf':
    #         return self.max_Class
    #     tree = self.dict[features[self.feature_index]]
    #     return tree.perdict(features)

    # def pre_oder(self, tree):         # preorder traverse
    #     if tree == None:
    #         return []
    #     stack = []
    #     class_set = []
    #     nodetype_set = []
    #     while tree or stack:
    #         while tree:
    #             class_set.append(tree.Class)
    #             nodetype_set.append(tree.node_type)
    #             if len(tree.dict) == 0:
    #                 break
    #             stack.append(tree)
    #             tree = tree.dict[0]
    #         if len(stack) == 0:
    #             break
    #         tree = stack.pop()
    #         if len(tree.dict) == 0:
    #             continue
    #         tree = tree.dict[1]
    #     return class_set, nodetype_set

    @classmethod
    def post_order(self, tree):
        stack = [tree]
        stack2 = []
        while len(stack) > 0:
            tree = stack.pop()
            stack2.append(tree)
            if len(tree.dict) != 0:
                stack.append(tree.dict[0])
                stack.append(tree.dict[1])
        sub_tree = []
        class_set = []
        nodetype_set = []
        while len(stack2) > 0:
            tmp_subtree = stack2.pop()
            class_set.append(tmp_subtree.Class)
            nodetype_set.append(tmp_subtree.node_type)
            sub_tree.append(tmp_subtree)
        return sub_tree, class_set, nodetype_set

    @classmethod
    def delete(self, tree, key):
        if tree == key:
            tree.dict = {}
            tree.max_Class = max(tree.Class)
            tree.node_type = 'leaf'
            return tree
        if len(tree.dict) > 0:
            self.delete(tree.dict[0], key)
            self.delete(tree.dict[1], key)

def CART(train, train_label, features_index, epsilon):
    leaf = 'leaf'
    internal = 'internal'

    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(leaf, Class=train_label, max_Class=label_set.pop())

    max_class = Counter(train_label).most_common(1)[0][0]
    if len(features_index) == 0:
        return Tree(leaf, Class=train_label, max_Class=max_class)

    select_feature = 0
    tmp_gini = 1    # gini less than 1
    for item in features_index:
        A = np.array(train[:, item])
        gini = joint_Gini(train_label, A)
        if tmp_gini > gini:
            tmp_gini, select_feature = gini, item

    if tmp_gini < epsilon:
        return Tree(leaf, Class=train_label, max_Class=max_class)

    sub_features = list(filter(lambda x: x != select_feature, features_index))
    tree = Tree(internal, feature_index=select_feature, Class=train_label)

    for val in (0, 1):
        sample_index = []
        for i in range(len(train_label)):
            if train[i][select_feature] == val:
                sample_index.append(i)
        sub_train = train[sample_index]
        sub_train_label = train_label[sample_index]
        sub_tree = CART(sub_train, sub_train_label, sub_features, epsilon)
        tree.add(val, sub_tree)

    return tree

# Section[4]
def calc_ent(x):
    classNum = len(x)  # total number of sample
    cnt_x = Counter(x)  # calculate the number of each category
    entropy = float(0)
    for i in cnt_x:
        p = cnt_x[i] / classNum  # probability of each category
        logp = np.log2(p)
        entropy -= p * logp
    return entropy

def costF(class_set):
    return len(class_set) * calc_ent(class_set)

def pruning(tree, cv, cv_label):
    # k, T = 0, tree        # (1)

    alpha = float('inf')  # (2)
    # subtree_array = [tree]
    T = copy.deepcopy(tree)


    optimal_subtree = tree
    cv_pred = predict(T, cv)
    score = (cv_pred == cv_label).sum() / len(cv_label)
    best_score = score

    while 1:
        if len(T.dict) == 0:
            break
        sub_tree, class_set, nodetype_set = Tree.post_order(T)   # (3)
        tmp_alpha = alpha
        location_subtree = 0
        for i in range(len(nodetype_set)):
            if nodetype_set[i] == 'internal':
                Ct = costF(class_set[i])        # the error of subtree, if subtree is a single tree
                tmp_tree, tmp_class_set, tmp_nodetype_set = Tree.post_order(sub_tree[i])
                Tt = Counter(tmp_nodetype_set)['leaf']      # the number of leaf in subtree
                CTt = float(0)                  # the error of subtree
                for j in range(len(tmp_nodetype_set)):
                    if tmp_nodetype_set[j] == 'leaf':
                        CTt += costF(tmp_class_set[j])
                gt = (Ct - CTt) / (Tt - 1)
                if tmp_alpha > gt:
                    tmp_alpha = gt
                    location_subtree = i
        pruned_subtree = sub_tree[location_subtree]
        Tree.delete(T, pruned_subtree)
        # tmp_T = copy.deepcopy(T)
        # subtree_array.append(tmp_T)

        cv_pred = predict(T, cv)
        score = (cv_pred == cv_label).sum() / len(cv_label)
        if best_score < score:
            best_score = score
            optimal_subtree = copy.deepcopy(T)

    return optimal_subtree

# Section[5]
def node_pred(tree, features):
    if tree.node_type == 'leaf':
        return tree.max_Class
    tree = tree.dict[features[tree.feature_index]]
    return node_pred(tree, features)

def predict(tree, test):
    result = []
    for features in test:
        tmp_predict = node_pred(tree, features)
        result.append(tmp_predict)
    return np.array(result)

def main():
    start = time.perf_counter()
    train, train_label, cv, cv_label, test, test_label \
        = load_data('mnist_train.csv', 'mnist_test.csv')
    time1 = time.perf_counter()
    print('data is loaded, cost {}s'.format(time1 - start))

    train = binaryzation_features(train)
    cv = binaryzation_features(cv)
    test = binaryzation_features(test)
    time2 = time.perf_counter()
    print('binaryzation features, cost {}s'.format(time2 - time1))

    tree = CART(train, train_label, features_index=[i for i in range(train.shape[1])], epsilon=0.1)
    time3 = time.perf_counter()
    print('Created tree, cost {}s'.format(time3 - time2))

    optimal_tree = pruning(tree, cv, cv_label)
    time4 = time.perf_counter()
    print('Pruned, cost {}s'.format(time4 - time3))

    test_pred = predict(optimal_tree, test)
    score = (test_pred == test_label).sum() / len(test_label)
    print("the accuracy score is {0:.4f}. Total time cost {1}".format(score, time.perf_counter()-start))


if __name__ == '__main__':
    main()
