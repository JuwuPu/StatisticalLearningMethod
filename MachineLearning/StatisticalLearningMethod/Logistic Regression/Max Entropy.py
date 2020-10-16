import cv2
import numpy as np
import pandas as pd
import time
from collections import defaultdict


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

def rebuilt_features(features):
    new_features = []
    for feature in features:
        new_feature = []
        for i, f in enumerate(feature):
            new_feature.append((str(i) + '_' + str(f)))
        new_features.append(new_feature)
    return new_features

def IIS(train, train_label, iter=1000):
    N = len(train)      # number of train sample

    label = set(train_label)
    Vxy = defaultdict(int)
    for i in range(N):
        x_, y = train[i], train_label[i]
        for x in x_:
            Vxy[(x, y)] += 1

    id2xy = {}
    xy2id = {}
    for i, (x, y) in enumerate(Vxy):
        id2xy[i] = (x, y)
        xy2id[(x, y)] = i

    Pxy = defaultdict(float)
    for id in range(len(Vxy)):
        (x, y) = id2xy[id]
        Pxy[id] = float(Vxy[(x, y)]) / float(N)














train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')
train = binaryzation_features(train)
test = binaryzation_features(test)

train = rebuilt_features(train)