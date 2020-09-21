import numpy as np
import itertools
import pandas as pd
import math
"""
    kd-tree for quick nearest-neighbor lookup

    Data Structure
        Property name       |     Type      |       Description
        Position            |    double     |   节点对应的空间坐标（描述样本点的特征数组）
        Axis                |     int       |   坐标轴（描述分裂维度的索引）
        Value               |    double     |   分割超平面在分裂维度上的取值（本属性可用Position[Axis])
        OriginalIndex       |     int       |   本节点对应样本在原训练集中的位置索引（import）
        Left                |  KDTreeNode   |   左子节点
        Right               |  KDTreeNode   |   右子节点

    """
class Node:
    def __init__(self, point=None, axis=None):
        """
        :param point: type list, indicates a sample, not contain id&label
        :param axis: type int, the splitting axis on this splitting
        """
        # self.parent = parent
        self.left = None
        self.right = None
        self.point = point  # indicates the point sample for this node
        self.axis = axis   # indicates the splitting axis for this node
        # self.flag = 0   # flag used in traverse, to indicate whether visited, 0 means not visited


class LargeHeap:
    def __init__(self):
        self.len = 0
        self.heaplist = []   # here we use a list to store heap

    def adjust(self):
        # adjust to a large heap, assuming that only the last element in heaplist is not legal
        i = self.len - 1
        while i > 0:
            if self.heaplist[i][1] > self.heaplist[(i - 1) / 2][1]:
                self.heaplist[i], self.heaplist[(i - 1) / 2] = self.heaplist[(i - 1) / 2], self.heaplist[i]
                i = (i - 1) / 2
            else:
                break

    def add(self, x, distance):
        """
        :param x: type Node, indicates a sample
        :param distance: type double, use to indicate "large"
        :return: None
        """
        # add a point and adjust it to a large heap
        self.len += 1
        self.heaplist.append([x, distance])   # append it to the end, and use adjust()
        self.adjust()

    def adjust2(self):
        # adjust to a large heap, assuming that only the first element(top) in heaplist is not legal
        i = 0
        # attention to exchange with the large one of the children
        while (2*i + 1) < self.len:
            tt = self.heaplist[(2*i + 1)][1]
            if 2*i+2 < self.len:
                tt = self.heaplist[(2*i + 2)][1]
            max_ind = (2*i + 1) if (self.heaplist[(2*i + 1)][1] > self.heaplist[(2*i + 2)][1] or \
                                                     (2*i+2 >= self.len)) else (2*i + 2)
            if self.heaplist[i][1] < self.heaplist[max_ind][1]:
                self.heaplist[i], self.heaplist[max_ind] = self.heaplist[max_ind], self.heaplist[i]
                i = max_ind
            else:
                break

    def pop(self):
        # pop the top of the heap
        if self.len == 1:
            self.heaplist = []
            self.len = 0
            return
        # exchange for the last ele, and use adjust2()
        self.heaplist[0] = self.heaplist[-1]
        self.len -= 1
        self.heaplist = self.heaplist[:self.len]
        self.adjust2()

class KdTree:
    def __init__(self, n_dim):
        """
        :param n_dim:  the dimension of this kd_tree
        """
        self.n_dim = n_dim   # the dimension of this tree
        self.root = None
        self.size = 0

    def distance(self, a, b):
        """
        :param a: type iterable
        :param b: type iterable
        :return: type double, return the Euclidean distance between a and b
        """
        s = 0.0
        for x, y in itertools.izip(a, b):
            d = x - y
            s += d * d
        return math.sqrt(s)

    def createTree(self, data, axis=0, current_node=Node()):
        """
        function to create a kd-tree, recursion
        :type data: array-like, samples used to construct a kd-tree or sub_kd-tree, the last column is label
        :type axis: int, between 0 and n_dim, dimension used to split data
        :type current_node: Node, the current 'root' node
        :return: None
        """
        if self.size == 0:
            self.root = current_node
        self.size += 1
        if data.shape[0] == 1:   # if no more than one sample, then stop iterating
            current_node.point = data[0, :]
            current_node.axis = axis
            return
        """
        step1: split the points with the median on this axis
        To find the median on target axis, we've got two ways:
        A. simply sort on target axis each time
        B. presort on each axis? but I haven't solved this at present
        """
        temp = data[data[:, axis].argsort()]
        med = len(temp)/2 if len(temp) % 2 else len(temp)/2 - 1  # get the median of this axis
        # find the 'first' med, this means that "<" goto left child, ">" goto right child
        while med > 0 and temp[med,axis] == temp[med-1, axis]:
            med -= 1
        current_node.axis = axis
        current_node.point = temp[med]
        tt = temp[med]
        axis = (axis + 1) % self.n_dim
        if temp[:med, :].shape[0] >= 1:
            tt = temp[:med, :]
            current_node.left = Node()
            self.createTree(temp[:med, :], axis, current_node.left)
        if temp[(med+1):, :].shape[0] >= 1:
            tt = temp[(med+1):, :]
            current_node.right = Node()
            self.createTree(temp[(med+1):, :], axis, current_node.right)

    def k_nearest_neighbor(self, k, target, current_root, k_nearest_heap = LargeHeap()):
        """
        function used to find the k nearest neighbor of a given target
        :param k: type int, indicates how many nearest neighbors to find
        :param target: type list, the target point
        :return: k_nearest_heap, type list
        """
        iter_list = []  # a stack to store iteration path
        # step1: find the 'nearest' leaf
        nearest_leaf = current_root
        while nearest_leaf is not None:
            iter_list.append(nearest_leaf)  # store the path
            tt = nearest_leaf.point
            tt1 = nearest_leaf.axis
            if target[nearest_leaf.axis] < nearest_leaf.point[nearest_leaf.axis]:
                if nearest_leaf.left is not None:  # then go to the left child
                    nearest_leaf = nearest_leaf.left
                else:
                    break
            else:
                if nearest_leaf.right is not None:   # else, go to the right child
                    nearest_leaf = nearest_leaf.right
                else:
                    break
        while nearest_leaf.left is not None or nearest_leaf.right is not None:
            if nearest_leaf.left is not None:
                nearest_leaf = nearest_leaf.left
                iter_list.append(nearest_leaf)
            if nearest_leaf.right is not None:
                nearest_leaf = nearest_leaf.right
                iter_list.append(nearest_leaf)
        tt = nearest_leaf.point
        """
        step2: find the k nearest by backtracking upside
        Two situations to add the point into the heap k_nearest_heap
        A. when len(k_nearest_heap) < k
        B. when dis(point, target) < current_max_dis
        """
        # k_nearest_heap = LargeHeap()  # the large heap to store the current 'nearest' neighbors
        # the max distance is actually the distance between target and the top of the heap
        '''
        current_max_dis = self.distance(target, nearest_leaf.point[:self.n_dim])
        k_nearest_heap.add(nearest_leaf, current_max_dis)
        tmp = iter_list.pop()
        '''
        former_node = nearest_leaf  # the former 'current_node', to indicate whether go through this child
        while iter_list != []:
            if k_nearest_heap.len > 0:
                current_max_dis = k_nearest_heap.heaplist[0][1]
            else:
                current_max_dis = -1
            current_pointer = iter_list.pop()
            tt = current_pointer.point
            dis = self.distance(current_pointer.point[:self.n_dim], target)
            if k_nearest_heap.len < k:
                k_nearest_heap.add(current_pointer, dis)
            elif dis < current_max_dis:
                k_nearest_heap.pop()
                k_nearest_heap.add(current_pointer, dis)
            # current_max_dis = self.distance(k_nearest_heap.heaplist[0][0].point[:self.n_dim], target)
            current_max_dis = k_nearest_heap.heaplist[0][1]
            axis = current_pointer.axis
            if abs(target[axis] - current_pointer.point[axis]) >= current_max_dis:
                former_node = current_pointer
                # if not intersect with
                continue
            if current_pointer.left is not None and current_pointer.left != former_node:
                tt = current_pointer.left
                # iter_list.append(current_pointer.left)
                self.k_nearest_neighbor(k, target, current_pointer.left, k_nearest_heap)
            if current_pointer.right is not None and current_pointer.right != former_node:
                tt = current_pointer.right
                # iter_list.append(current_pointer.righat)
                self.k_nearest_neighbor(k, target, current_pointer.right, k_nearest_heap)
            former_node = current_pointer
        rlist = []
        rdis = []
        for ele in k_nearest_heap.heaplist:
            rlist.append(ele[0].point)
            rdis.append(ele[1])
        return rdis, rlist





