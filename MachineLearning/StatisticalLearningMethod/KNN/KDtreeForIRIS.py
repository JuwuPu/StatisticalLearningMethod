import kdtree
import pandas as pd
import numpy as np

# step1: reading data
def load_data(train_file, test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    y = train.iloc[:, -1]
    train = train.drop(labels=train.columns[-1], axis=1)
    return np.array(train), np.array(y), np.array(test)


train, y, test = load_data('train.csv', 'test.csv')
train = train[:, 1:]  # removing id
test_id = test[:, 0]
test = test[:, 1:]  # removing id

# step2: constructig kdtree for training data
tree = kdtree.KdTree(n_dim=train.shape[1])
tree.createTree(np.c_[train, y])

# vote for prediction
y_test = []  # storing the y of testing data
for i in range(test.shape[0]):
    classCounter = {}  # vote
    dis, k_nearest = tree.k_nearest_neighbor(k, test[i], tree.root, kdtree.LargeHeap())
    for pos in k_nearest:
        classCounter[pos[-1]] = classCounter.get(pos[-1], 0) + 1
    y_test.append(sorted(classCounter)[0])

