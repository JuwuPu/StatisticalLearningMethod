import cv2
import numpy as np
import pandas as pd
import time
from collections import defaultdict
import progressbar as pgb


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

class maxEntropy(object):
    def init__(self, X, Y):
        self.X_ = X
        self.Y_ = set()
        self.calc_Vxy(X, Y)
        self.N = len(X)             # sample size
        self.n = len(self.Vxy)
        self.M = 10000.0            # learning rate

    def calc_Vxy(self, X, Y):
        """
        calculate v(X=x, Y=y)
        """
        self.Vxy = defaultdict(int)
        for i in range(len(X)):
            x_, y = X[i], Y[i]
            self.Y_.add(y)
            for x in x_:
                self.Vxy[(x, y)] += 1

    def build_dict(self):
        self.id2xy = {}
        self.xy2id = {}
        for i, (x, y) in enumerate(self.Vxy):
            self.id2xy[i] = (x, y)
            self.xy2id[(x, y)] = i

    def calc_Pxy(self):
        """
        calculate the P(X=x, Y=y)
        """
        self.Pxy = defaultdict(float)
        for id in range(self.n):
            (x, y) = self.id2xy[id]
            self.Pxy[id] = float(self.Vxy[(x, y)]) / float(self.N)

    def calc_Zx(self, X, y):
        """
        calculate Zw(x/yi)
        """
        tmp_ = float(0)
        for x in X:
            if (x, y) in self.xy2id:
                id = self.xy2id[(x, y)]
                tmp_ += self.w[id]
        return (np.exp(tmp_), y)

    def calc_Pyx(self, X):
        """
        calculate P(y|x)
        """
        Pyxs = [(self.calc_Zx(X, y)) for y in self.Y_]
        Zwx = sum([prob for prob, y in Pyxs])
        return [(prob / Zwx, y) for prob, y in Pyxs]

    def calc_Epfi(self):
        """
        calculate Ep(fi)
        """
        self.Epfi = [0.0 for i in range(self.n)]

        for i, X in enumerate(self.X_):
            Pyxs = self.calc_Pyx(X)
            for x in X:
                for Pyx, y in Pyxs:
                    if (x, y) in self.xy2id:
                        id = self.xy2id[(x, y)]
                        self.Epfi[id] += Pyx * (1.0 / self.N)


    def IIS(self, X, Y):
        self.init__(X, Y)
        self.build_dict()
        self.w = [0.0 for i in range(self.n)]
        max_iteration = 500

        for times in range(max_iteration):
            detas = []
            self.calc_Epfi()
            for i in range(self.n):
                deta = 1 / self.M * np.log(self.Pxy[i] / self.Epfi[i])
                detas.append(deta)

            self.w = [self.w[i] + detas[i] for i in range(self.n)]

    def predict(self, test):
        temp = []
        for item in test:
            temp_ = self.calc_Pyx(item)
            temp.append(max(temp_, key=lambda x: x[0])[1])
        return temp



time_1 = time.time()
train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')
train = binaryzation_features(train)
test = binaryzation_features(test)
train = rebuilt_features(train)
test = rebuilt_features(test)
time_2 = time.time()
print('read data cost %f seconds' % (time_2 - time_1))

print('Start training...')
model = maxEntropy()
model.IIS(train, train_label)
time_3 = time.time()
print('training cost %f seconds' % (time_3 - time_2))

print('Start predicting...')
test_predict = model.predict(test)
time_4 = time.time()
print('predicting cost %f seconds' % (time_4 - time_3))

score = accuracy = (test_predict == test_label).sum() / len(test_label)
print("The accruacy score is %f" % score)