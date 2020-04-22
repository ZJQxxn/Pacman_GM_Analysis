'''
Description:
    Construct a utility tree along the estimated path.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 21 2020
'''

import pandas as pd
import numpy as np
import anytree
import sys
from collections import deque

sys.path.append('./')
from TreeAnalysisUtils import adjacent_data


class PathTree:

    def __init__(self, root, depth = 10):
        # The root node contains only the pathstarting point.
        # Other tree nodes should contain:
        #   (1) location
        #   (2) the direction from its to parent to itself
        #   (3) utility of this node (reward and risk are separated)
        #   (4) the cumulative utility so far (reward and risk are separated)
        self.root = anytree.Node(root)
        # A list of moving directions of the path with the highest utility
        self.best_path = []
        # A queue used for append nodes on the tree
        self.node_queue = deque()
        # The maximize depth (i.e., the path length)
        self.depth = depth

    def construct(self):
        #TODO: when walking on the path, do not go back
        pass

    def _computeUtility(self):
        # TODO: compute utility for a certain node
        pass