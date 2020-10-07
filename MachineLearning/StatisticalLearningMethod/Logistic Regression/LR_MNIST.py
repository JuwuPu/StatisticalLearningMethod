import numpy as np
import pandas as pd
import time


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

def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+np.exp(-inx))
    else:
        return np.exp(inx)/(1+np.exp(inx))

def lr_train(x, y, label_change,
             learning_step=0.00001,
             max_iteration=5000,
             lambda_=0.06):
    w = np.zeros((1 + x.shape[1], 1))
    iteration = 0
    _x = np.column_stack((np.ones((x.shape[0], 1)), x))
    _y = np.array(list(map(lambda q: q == label_change, y))).astype(int)
    m, n = _x.shape
    index_set = np.random.permutation(m)
    while iteration < max_iteration:
        # stochastic gradient descent
        index = index_set[iteration]
        tmp_x, tmp_y = _x[index].reshape(1, n), _y[index].reshape(1, 1)

        # w[0] -= learning_step / m * ((1 + np.exp(np.dot(tmp_x, w))) - tmp_y))
        # w -= learning_step * (
        #     ((1 + np.exp(-np.dot(tmp_x, w))) - tmp_y)*tmp_x.T + lambda_ * w
        # )
        w -= learning_step * (
                (sigmoid(np.dot(tmp_x, w)) - tmp_y) * tmp_x.T + lambda_ * w
        )

        w.reshape((n, 1))
        iteration += 1
    return w

def coef_matrix(train, train_label):
    w_matrix = np.zeros((train.shape[1]+1, 10))
    for i in [i for i in range(10)]:
        w = lr_train(train, train_label, i)
        w_matrix[:, [i]] = w
    return w_matrix

def predict(features, w):
    _features = np.column_stack((np.ones((features.shape[0], 1)), features))
    wx = np.dot(_features, w)
    probability = np.exp(wx) / (1 + np.exp(wx))
    result = probability.argmax(axis=1)
    return result


start = time.perf_counter()
train, train_label, test, test_label = load_data('mnist_train.csv', 'mnist_test.csv')
time1 = time.perf_counter()
print('data is loaded, cost {}s'.format(time1 - start))

w_matrix = coef_matrix(train, train_label)
time2 = time.perf_counter()
print('Training completed, cost {}s'.format(time2 - time1))

test_pred = predict(test, w_matrix)
score = (test_pred == test_label).sum() / len(test_label)
print("the accuracy score is {0:.4f}. Total time cost {1}".format(score, time.perf_counter() - start))







