'''
Description:
    Main function for the utility tree analysis.

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
        energizer_data = self.data.energizers.values[0]
        bean_data = self.data.beans.values[0]
        ghost_data = np.array([self.data.distance1.values[0], self.data.distance2.values[0]])
        ghost_status = self.data[["ifscared1", "ifscared2"]].values
        reward_type = self.data.Reward.values
        fruit_pos = self.data.fruitPos.values
        tree = PathTree(
            self.data.pacmanPos.values[0],
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth = 15,
            ghost_attractive_thr = 34,
            ghost_repulsive_thr = 10,
            fruit_attractive_thr = 10
        )
        root, highest_utility, best_path = tree.construct()
        print(highest_utility)
        print(best_path)


if __name__ == '__main__':
    a = Analyzer("extracted_data/test_data.pkl")
    a.analysis()