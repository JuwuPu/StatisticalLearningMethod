class KDTree(object):
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
    class Node(object):
        def __init__(self):
            self.parent = None
            self.left = None
            self.right = None
            self.split = None





