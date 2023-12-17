import numpy as np

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
    def check(self, value):
        if value == self.value:
            return 1
        else:
            return 0
    def add_children(self, value):
        if self.is_child(value):
            if self.value == value:
                return 1
            else:
                for child in self.children:
                    if child.add_children(value):
                        return 1
                self.children.append(Node(value))
            
        else:
            return 0

    def is_child(self, value):
        #  0011010
        # 0001010 other
        # 0001010 and
        # 0000000 xor

        # 0011010
        # 0101010 other
        # 0001010 and
        # 0100000 xor
        if np.sum(np.logical_xor(np.logical_and(self.value, value), value)) > 0:
            return 0
        else:
            return 1
