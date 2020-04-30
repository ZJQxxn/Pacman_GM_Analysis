'''
Description:
    The lazy agent.

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

    def __init__(self, cur_pos, last_dir, loop_count, max_loop = 5):
        self.cur_pos = cur_pos
        self.adjacent_pos = adjacent_data[adjacent_data.pos == self.cur_pos]
        self.available_dir = []
        for dir in self.adjacent_pos.columns.values:
            if None != self.adjacent_pos[dir]:
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.last_dir = last_dir  # moving direction for the last time step
        self.not_turn = False # True only if the Pacman passes a crossroad without turning
        self.loop_count = loop_count  # the number of crossroads passed for now
        self.max_loop = max_loop  # the maximum number of the crossroads Pacman can directly pass
        self.dir_list = ['left', 'right', 'up', 'down']


    def _atTunnel(self):
        if 2 == len(self.available_dir):
            if "left" in self.available_dir and "right" in self.available_dir:
                return True
            elif "up" in self.available_dir and "down" in self.available_dir:
                return True
            else:
                return False
        else:
            return False


    def _atCorner(self):
        if 2 == len(self.available_dir):
           if not self._atTunnel():
               return True
           else:
               return False
        else:
            return False


    def _atTJunction(self):
        if 3 == len(self.available_dir):
            if not self.last_dir in self.available_dir:
                return True
            else:return False
        else:
            return False


    def nextDir(self):
        # If this is the starting of the game, randomly choose a direction among available choices
        if None == self.last_dir:
            choice = np.random.choice(range(len(self.available_dir)), 1)
            choice = self.available_dir[choice]
        # else, stay moving to the same direction until the number of passed crossroads surpasses a threshold (default = 5)
        else:
            # If at the corner or in the tunnel (i.e., only one direction can be chosen except the last moving direction)
            if self._atTunnel() or self._atCorner():
                self.available_dir.remove(self.last_dir)
                choice = self.available_dir[0]
            # else if at the T-junction and the Pacman must turn (i.e., two direction can be chosen except the last moving direction)
            elif self._atTJunction():
                self.available_dir.remove(self.last_dir)
                choice = np.random.choice(range(len(self.available_dir)), 1)
                choice = self.available_dir[choice]
            # else if at the crossroads or T-junctions (without mandatory turning)
            else:
                # don't have to turn
                if self.loop_count < self.max_loop:
                    choice = self.last_dir
                    self.not_turn = True
                # turn for breaking the loop
                else:
                    self.available_dir.remove(self.last_dir)
                    choice = np.random.choice(range(len(self.available_dir)), 1)
                    choice = self.available_dir[choice]
        return (choice, self.not_turn)