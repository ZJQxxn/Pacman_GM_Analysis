'''
Description:
    The simple global agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Dec. 31 2020
'''

import pandas as pd
import numpy as np
import anytree
from anytree.exporter import DotExporter
from collections import deque
import sys
import time
import copy

sys.path.append('./')
from TreeAnalysisUtils import unitStepFunc, scaleOfNumber, makeChoice


class SimpleEnergizer:

    def __init__(self, adjacent_data, adjacent_path, locs_df, reward_amount, cur_pos, energizer_data, ghost_data, ghost_status, beans, last_dir,
                 ghost_attractive_thr = 15, energizer_attractive_thr = 15, beans_attractive_thr =15, randomness_coeff = 1.0, laziness_coeff = 1.0):
        # Game status (energizer)
        self.cur_pos = cur_pos
        self.energizer_data = energizer_data
        self.ghost_data = ghost_data
        self.ghost_status = np.array(ghost_status)
        self.beans = beans
        self.last_dir = last_dir
        # Auxiliary data
        self.adjacent_data=  adjacent_data
        self.adjacent_path = adjacent_path
        self.locs_df = locs_df
        self.reward_amount = reward_amount
        # Adjacent position and available direction
        self.adjacent_pos = adjacent_data[self.cur_pos]
        self.available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.adjacent_pos = [self.adjacent_pos[each] for each in self.available_dir]
        # Utility (Q-value) for every direction
        self.Q_value = [0, 0, 0, 0]
        # Direction list
        self.dir_list = ['left', 'right', 'up', 'down']
        # For randomness and laziness
        self.randomness_coeff = randomness_coeff
        self.laziness_coeff = laziness_coeff
        self.ghost_attractive_thr = ghost_attractive_thr
        self.energizer_attractive_thr = energizer_attractive_thr
        self.beans_attractive_thr = beans_attractive_thr


    def _dirArea(self,dir):
        # x: 1~28 | y: 1~33
        left_bound = 1
        right_bound = 28
        upper_bound = 1
        lower_bound = 33
        # Area corresponding to the direction
        if dir == "left":
            area = [
                (left_bound, upper_bound),
                (max(1, self.cur_pos[0]-1), lower_bound)
            ]
        elif dir == "right":
            area = [
                (min(right_bound, self.cur_pos[0]+1), upper_bound),
                (right_bound, lower_bound)
            ]
        elif dir == "up":
            area = [
                (left_bound, upper_bound),
                (right_bound, min(lower_bound, self.cur_pos[1]+1))
            ]
        elif dir == "down":
            area = [
                (left_bound, min(lower_bound, self.cur_pos[1]+1)),
                (right_bound, lower_bound)
            ]
        else:
            raise ValueError("Undefined direction {}!".format(dir))
        return area


    def _countEnergizers(self, upper_left, lower_right):
        area_loc = []
        # Construct a grid area
        for i in range(upper_left[0], lower_right[0]+1):
            for j in range(upper_left[1], lower_right[1]+1):
                area_loc.append((i,j))
        if isinstance(self.energizer_data, float) or self.energizer_data is None:
            return 0
        else:
            energizer_num = 0
            for each in self.energizer_data:
                if each in area_loc:
                    energizer_num += 1
            return energizer_num


    def nextDir(self, return_Q=False):
        available_directions_index = [self.dir_list.index(each) for each in self.available_dir]
        self.Q_value = [0.0, 0.0, 0.0, 0.0]
        for dir in self.available_dir:
            area = self._dirArea(dir)
            energizer_num = self._countEnergizers(area[0], area[1])
            self.Q_value[self.dir_list.index(dir)] = energizer_num
        self.Q_value = np.array(self.Q_value, dtype=np.float)
        # self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(self.Q_value)))
        if len(available_directions_index) > 0:
            randomness = np.random.normal(loc=0, scale=0.1, size=len(available_directions_index)) * Q_scale
            self.Q_value[available_directions_index] += (self.randomness_coeff * randomness)
        if self.last_dir is not None and self.dir_list.index(self.last_dir) in available_directions_index:
            self.Q_value[self.dir_list.index(self.last_dir)] += (self.laziness_coeff * Q_scale)
        if return_Q:
            return makeChoice(self.Q_value), self.Q_value
        else:
            return makeChoice(self.Q_value)


if __name__ == '__main__':
    import sys

    sys.path.append('./')
    from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, makeChoice

    # Read data
    locs_df = readLocDistance("./extracted_data/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    adjacent_path = readAdjacentPath("./extracted_data/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    print("Finished reading auxiliary data!")

    # cur_pos = (27, 33) # 24
    # ghost_data = [(14, 33), (17, 27)]
    # ghost_status = [1, 1]
    # energizer_data = [(13, 9), (22, 26)]
    # bean_data = [(7, 5), (11, 5), (17, 5), (23, 5), (2, 8), (18, 9), (27, 11), (13, 12), (24, 12), (7, 14),
    #              (7, 15), (10, 15), (13, 15), (16, 15), (19, 15), (22, 15), (7, 16), (2, 18), (25, 18), (22, 19),
    #              (7, 21), (17, 21), (27, 24), (15, 27), (13, 31)]
    # reward_type = 6
    # fruit_pos = (2, 7)
    # last_dir = "right"

    cur_pos = (14, 27)
    ghost_data = [(13, 27), (15, 27)]
    ghost_status = [1, 1]
    energizer_data = [(7, 5), (17, 5), (7, 26), (24, 30)]
    bean_data = [(2, 5), (4, 5), (5, 5), (16, 5), (18, 5), (20, 5), (24, 5), (25, 5), (16, 6), (7, 7), (13, 7), (27, 7),
                 (2, 8), (16, 8), (22, 8), (2, 9), (6, 9), (8, 9), (9, 9), (13, 9), (14, 9), (16, 9), (17, 9), (19, 9),
                 (24, 9), (26, 9), (27, 9), (10, 10), (22, 10), (2, 11), (10, 11), (22, 11), (5, 12), (7, 12), (22, 12),
                 (22, 13), (7, 14), (13, 14), (16, 14), (7, 15), (7, 17), (22, 17), (7, 19), (10, 23), (2, 24), (3, 24),
                 (6, 24), (8, 24), (9, 24), (13, 24), (17, 24), (20, 24), (22, 24), (25, 24), (7, 25), (22, 25),
                 (2, 26),
                 (27, 27), (2, 28), (27, 28), (10, 29), (22, 29), (2, 30), (7, 30), (11, 30), (16, 30), (18, 30),
                 (19, 30),
                 (22, 30), (27, 30), (2, 32), (5, 33), (9, 33), (12, 33), (14, 33), (15, 33), (16, 33), (17, 33),
                 (18, 33),
                 (20, 33), (24, 33), (25, 33), (26, 33)]
    reward_type = 5
    fruit_pos = (3, 30)
    last_dir = "left"

    # Global agent
    agent = SimpleEnergizer(
        adjacent_data,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status,
        last_dir,
        15,
        5,
        34,
        34,
        34,
        reward_coeff=1.0, risk_coeff=0.0,
        randomness_coeff=1.0, laziness_coeff=1.0
    )
    _, Q = agent.nextDir(return_Q=True)
    choice = agent.dir_list[makeChoice(Q)]
    print("Energizer Choice : ", choice, Q)

