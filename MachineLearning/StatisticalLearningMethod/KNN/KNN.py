import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
    data = pd.read_csv('IRIS.csv', header=None)
    data_matrix = df2ndarray(data)