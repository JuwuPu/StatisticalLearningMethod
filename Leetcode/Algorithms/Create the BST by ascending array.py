class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def sorted_array_to_bst(nums):
    if not nums:
        return None
    mid_val = len(nums) // 2
    node = TreeNode(nums[mid_val])
    node.left = sorted_array_to_bst(nums[:mid_val])
    node.right = sorted_array_to_bst(nums[mid_val + 1:])
    return node

nums = [1,2,3,4,5]

bst = sorted_array_to_bst(nums)