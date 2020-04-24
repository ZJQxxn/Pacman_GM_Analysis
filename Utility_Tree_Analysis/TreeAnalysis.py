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

sys.path.append('./')
from PathTreeConstructor import PathTree

class Analyzer:

    def __init__(self, filename):
        # TODO: how to split different path
        self.data = pd.read_csv(filename)
        for each in ["pacmanPos", "ghost1Pos", "ghost2Pos", "energizers", "beans", "next_eat_rwd"]:
            self.data[each] = self.data[each].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
        self.data.ifscared1 = self.data.ifscared1.apply(lambda x: int(eval(x)) if not isinstance(x, float) else np.nan)
        self.data.ifscared2 = self.data.ifscared2.apply(lambda x: int(eval(x)) if not isinstance(x, float) else np.nan)


    def analysis(self):
        # TODO: 拿出每一条整个 global graze 的路径来进行分析； fruit， bean按照这个path上的情况来进行计算；鬼的距离怎么确定？
        tree = PathTree(self.data.pacmanPos.values[0], self.data, depth =10)
        tree.construct()
        # print(anytree.RenderTree(tree.root))
        print("Path with the highest cumulative utility is: ")
        for each in  [each.name for each in tree.best_path[-1].path]:
            print(each)


if __name__ == '__main__':
    a = Analyzer("extracted_data/test_data.csv")
    a.analysis()