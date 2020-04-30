'''
Description:
    The random agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 29 2020
'''

import numpy as np
import sys
sys.path.append('./')
from TreeAnalysisUtils import adjacent_data


class LazyAgent:

    def __init__(self, cur_pos, last_dir):
        self.cur_pos = cur_pos
        self.adjacent_pos = adjacent_data[adjacent_data.pos == self.cur_pos]
        self.available_dir = []
        for dir in self.adjacent_pos.columns.values:
            if None != self.adjacent_pos[dir]:
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.last_dir = last_dir  # moving direction for the last time step
        self.dir_list = ['left', 'right', 'up', 'down']

    def nextDir(self):
        # If this is the starting of the game, randomly choose a direction among available choices
        if None == self.last_dir:
            choice = np.random.choice(range(len(self.available_dir)), 1)
            choice = self.available_dir[choice]
        # else, stay moving to the same direction until the number of passed crossroads surpasses a threshold (default = 5)
        else:
            self.available_dir.remove(self.last_dir)
            choice = np.random.choice(range(len(self.available_dir)), 1)
            choice = self.available_dir[choice]
        return choice