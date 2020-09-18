import numpy as np
import pandas as pd

class LoadData:
    '''
    Data pre-processing.
    '''

    def __init__(self):

        '''
        Load data class to input original data.
        '''

        self.load_data = None
        self.df2ndarray = None
        self.SplitData = None

    def load_data(loaction):
        data = pd.read_csv(loaction, header=None)
        return data

    def df2ndarray(data):

        '''
        This function only for IRIS data precess.
        '''

        rows, columns = data.shape
        data_matrix = np.zeros((rows - 1, columns))
        data = data.values
        for i in range(rows - 1):
            for j in range(columns):
                if j != 4:
                    data_matrix[i][j] = float(data[i + 1][j])
                elif data[i + 1][j] == 'Iris-setosa':
                    data_matrix[i][j] = 0
                elif data[i + 1][j] == 'Iris-versicolor':
                    data_matrix[i][j] = 1
                else:
                    data_matrix[i][j] = 2
        return data_matrix

    def SplitData(data, ratio):
        np.random.shuffle(data)
        m, n = data.shape
        train_set = data[0:int(m * ratio[0]), :]
        cv_set = data[int(m * ratio[0]):int(m * (ratio[0] + ratio[1])), :]
        test_set = data[int(m * (ratio[0] + ratio[1])):, :]
        return train_set, cv_set, test_set
