'''
Description:
    The random agent. An agent randomly chooses a direction at crossroads.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 29 2020
'''

import numpy as np


class RandomAgent:

    def __init__(self, adjacent_data, cur_pos, last_dir, random_seed = None):
        '''
        Initialization of the random agent.
        :param adjacent_data: The adjacent tiles of each tile in the map. Should be a data with the type pf pandas.DataFrame.
        :param cur_pos: The current position of Pacman, should be a 2-tuple.
        :param last_dir: The moving direction of the last time step, should be a string from {``left'', ``right'', ``up'',
                                  ``down''}.
        :param random_seed: The random seed.
        '''
        np.random.seed(random_seed)
        self.cur_pos = cur_pos
        self.adjacent_pos = adjacent_data[self.cur_pos]
        self.available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.last_dir = last_dir  # moving direction for the last time step
        # opposite direction; to avoid turn back
        self.opposite_dir = {"left": "right", "right": "left", "up": "down", "down": "up"}
        # Utility (Q-value) for every direction
        self.Q_value = [0, 0, 0, 0]
        # Direction list
        self.dir_list = ['left', 'right', 'up', 'down']


    def nextDir(self, return_Q = False):
        '''
        Estimate the moving direction. 
        :return: The moving direction {`left'', ``right'', ``up'', ``down''}.
        '''
        choice = np.random.choice(range(len(self.available_dir)), 1).item()
        choice = self.available_dir[choice]
        random_Q_value = np.tile(1 / len(self.available_dir), len(self.available_dir))
        for index, each in enumerate(self.available_dir):
            self.Q_value[self.dir_list.index(each)] = random_Q_value[index]
        self.Q_value = np.array(self.Q_value)
        self.Q_value = self.Q_value / np.sum(self.Q_value)
        if return_Q:
            return choice, self.Q_value
        else:
            return choice

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from TreeAnalysisUtils import readAdjacentMap
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    cur_pos = (22, 24)
    last_dir = "right"
    agent = RandomAgent(adjacent_data, cur_pos, last_dir)
    choice = agent.nextDir(return_Q = True)
    print("Choice : ", choice)
