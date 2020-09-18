import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def df2ndarray(data):
    rows, columns = data.shape
    data_matrix = np.zeros((rows-1, columns))
    data = data.values
    for i in range(rows-1):
        for j in range(columns):
            if j != 4:
                data_matrix[i][j] = float(data[i+1][j])
            elif data[i+1][j] == 'Iris-setosa':
                data_matrix[i][j] = 0
            elif data[i+1][j] == 'Iris-versicolor':
                data_matrix[i][j] = 1
            else:
                data_matrix[i][j] = 2
    return data_matrix

def SplitData(data,ratio):
    np.random.shuffle(data)
    m, n = data.shape
    train_set = data[0:int(m*ratio[0]), :]
    cv_set = data[int(m*ratio[0]):int(m*(ratio[0]+ratio[1])), :]
    test_set = data[int(m*(ratio[0]+ratio[1])):, :]
    return train_set, cv_set, test_set

def k_value(train, cv, Max_k):
    m, n = train.shape      # the rows of train --> m, the column of train -->n
    p, q = cv.shape         # the rows of train --> p, the column of train --> q
    dist_matrix = np.zeros((p, m))
    cnt1, cnt2 = 0, 0
    Accuracylist = []
    for i in cv:
        cnt2 = 0
        for j in train:
            # each row --> The distance from one point in CV to each training set point.
            dist_matrix[cnt1][cnt2] = np.linalg.norm(i[0:3] - j[0:3])
            cnt2 += 1
        cnt1 += 1
    for r in range(1, Max_k+1):
        cv_Predict = np.zeros((p, 1))
        cnt3 = 0
        for item in dist_matrix:
            # Combine the dist(m x 1) and the label of training sample(m x 1).
            dist = np.hstack((item.reshape((m, 1)), train[:, 4].reshape((m, 1))))
            # Sort the combined matrix by the distance of the first column
            sort_dist = dist[dist[:, 0].argsort()]
            # Select the label with the most frequent label occurrences among the first k distances.
            cv_Predict[cnt3] = stats.mode(sort_dist[0:r, 1])[0][0]
            cnt3 += 1
        bool_vector = cv_Predict == cv[:, -1].reshape((p, 1))
        Accuracylist.append(stats.mode(bool_vector)[1][0]/p)
    plt.figure()
    k = list(range(1, Max_k+1))
    plt.plot(k, Accuracylist, color='r', label='k value for Classification error')
    plt.xlabel("k value")
    plt.ylabel("Cross-Validation Accuracy")
    plt.title(" Value of different k")
    plt.show()
    accuracy_matrix = np.hstack((
        np.reshape(k, (Max_k, 1),), np.reshape(Accuracylist, (Max_k, 1))))
    accuracy_matrix = accuracy_matrix[
        accuracy_matrix[:, 1].argsort()
    ]
    return int(accuracy_matrix[-1][0])

def KNN(train, test, k_value):
    m, n = train.shape
    p, q = test.shape
    dist_matrix = np.zeros((p, m))
    cnt1, cnt2 = 0, 0
    test_Predict = []
    for i in test:
        cnt2 = 0
        for j in train:
            dist_matrix[cnt1][cnt2] = np.linalg.norm(i[0:3] - j[0:3])
            cnt2 += 1
        cnt1 += 1
    for item in dist_matrix:
        dist = np.hstack((item.reshape((m, 1)), train[:, 4].reshape((m, 1))))
        sort_dist = dist[dist[:, 0].argsort()]
        test_Predict.append(stats.mode(sort_dist[0:k_value, 1])[0][0])
    test_Predict = np.reshape(test_Predict, (p, 1))
    bool_vector = test_Predict == test[:, -1].reshape((p, 1))
    Accuracy = (stats.mode(bool_vector)[1][0])/p
    return Accuracy, test_Predict

if __name__ == '__main__':
    data = pd.read_csv('IRIS.csv', header=None)
    data_matrix = df2ndarray(data)
    train, cv, test = SplitData(data_matrix, (0.5, 0.25, 0.25))
    k = k_value(train, cv, 30)
    print('k value is {0}'.format(k))
    Accuracy, predict_set = KNN(train, test, k)
    print("The classifier's accuracy is {0:.2f}%.\n".format(float(Accuracy) * 100))
