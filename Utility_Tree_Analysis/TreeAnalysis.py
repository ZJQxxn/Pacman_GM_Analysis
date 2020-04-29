'''
Description:
    Path tree analysis

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 21 2020
'''

import pandas as pd
import numpy as np
import anytree
import sys
import pickle
import pprint

sys.path.append('./')
from PathTreeConstructor import PathTree

class Analyzer:

    def __init__(self, filename):
        # TODO: how to split different path
        # TODO: 现在只有一盘游戏的数据
        with open(filename, 'rb') as file:
            self.data = pickle.load(file)
        print()


    def analysis(self):
        # TODO: 拿出每一条整个 global graze 的路径来进行分析； fruit， bean按照这个path上的情况来进行计算；鬼的距离怎么确定？
        tree = PathTree(self.data.pacmanPos.values[0], self.data, depth =15)
        tree.construct()
        # print(anytree.RenderTree(tree.root))
        pprint.pprint([(each.name, each.cumulative_utility) for each in tree.root.leaves])
        best_leaf = tree.root
        for leaf in tree.root.leaves:
            if leaf.cumulative_utility > best_leaf.cumulative_utility:
                best_leaf = leaf
        best_path = best_leaf.ancestors
        print("\n Path with the highest cumulative utility {} is: ".format(best_leaf.cumulative_utility))
        for each in [each.name for each in best_path]:
            print(each)
        print(best_leaf.name)


if __name__ == '__main__':
    a = Analyzer("extracted_data/test_data.pkl")
    a.analysis()