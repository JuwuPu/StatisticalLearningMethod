import numpy as np

x = np.array([i for i in range(1, 11)])
y = np.array([4.5, 4.75, 4.91, 5.34, 5.80, 7.05, 7.90, 8.23, 8.70, 9.00])

'''
least squares regression tree

INPUT: train set D
OUTPUT: regression tree T

'''

def SE(x, y, index):
    c1 = y[[j <= x[index] for j in x]]
    c2 = y[[j > x[index] for j in x]]
    if len(c2) == 0:
        c2 = [0]
    error1 = np.sum((c1-np.mean(c1))**2)
    error2 = np.sum((c2-np.mean(c2))**2)
    return error1 + error2

def SplitPoint(x,y):
    error_list = []
    for i in range(len(x)):
        error_list.append(SE(x,y,i))
    error_list = np.array(error_list)
    index = np.argmin(error_list)
    min_error = np.min(error_list)
    return x[index], index, min_error

class RT(object): # create the node of tree
    def __init__(self, val=None, features=None):
        self.feature = features
        self.val = val
        self.left = None
        self.right = None

def createTR(feature, val):
    if len(val) >= 1:
        _feature, _split, _val = SplitPoint(feature, val)

        root = RT(val=_val, features=_feature)

        f_left = feature[:_split]
        v_left = val[:_split]
        f_right = feature[(_split+1):]
        v_right = val[(_split+1):]

        root.left = createTR(f_left, v_left)
        root.right = createTR(f_right, v_right)
    else:
        root = None
    return root

if __name__ == '__main__':
    tree = createTR(x,y)








