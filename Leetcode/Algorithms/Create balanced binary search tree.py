class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

nums = [1,2,3,4,5]


# Write a Python program to create a Balanced Binary Search Tree (BST)
#   using an array (given) elements where array elements are sorted in ascending order.
def sorted_array_to_bst(nums):
    if not nums:
        return None

    mid_val = len(nums)//2
    node = TreeNode(nums[mid_val])
    node.left = sorted_array_to_bst(nums[:mid_val])
    node.right = sorted_array_to_bst(nums[mid_val + 1:])
    return node

bst = sorted_array_to_bst(nums)
