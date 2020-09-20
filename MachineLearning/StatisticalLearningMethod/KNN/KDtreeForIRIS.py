from LoadData import LoadData as LD
import numpy as np
from scipy.spatial import KDTree





def main():
    data = LD.load_data('IRIS.CSV')
    dataMatrix = LD.df2ndarray(data)
    train, CV, test = LD.SplitData(dataMatrix, (0.6, 0, 0.4))  # CrossValidation is empty.
    KDTree


if __name__ == '__main__':
    main()

