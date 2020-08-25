'''
Description:
    The lazy agent. An agent chooses to stay the moving direction at crossroads for saving energy. Specifically, in 
    order to avoid hovering in a sub-area, the agent randomly chooses a direction at the crossroads when the number of 
    passed crossroads surpasses a threshold. 
    
    (Updated 12 Aug. 2020) The lazy agent chooses to repeat the last direction. If the last direction is not available, 
    randomly select a direction.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 29 2020
'''

import numpy as np


class LazyAgent:

    def __init__(self, adjacent_data, cur_pos, last_dir):
        '''
        Initialization of the lazy agent.
        :param adjacent_data: The adjacent tiles of each tile in the map. Should be a data with the type pf pandas.DataFrame.
        :param cur_pos: The current position of Pacman, should be a 2-tuple.
        :param last_dir: The moving direction of the last time step, should be a string from {``left'', ``right'', ``up'',
                          ``down''}.
        :param loop_count: The number of the passed crossroads, should be a non-negative integer.
        :param max_loop: The maximal number of the passed crossroads. The agent should randomly choose a direction at 
                          crossroads when loop_count > max_loop.
        '''
        self.cur_pos = cur_pos
        self.adjacent_pos = adjacent_data[self.cur_pos] # all the adjacent positions of the current position
        self.available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.last_dir = last_dir  # moving direction for the last time step
        # Utility (Q-value) for every direction
        self.Q_value = [0, 0, 0, 0]
        # Direction list
        self.dir_list = ['left', 'right', 'up', 'down']
        # self.not_turn = False # True only if the Pacman passes a crossroad without turning
        # self.loop_count = loop_count  # the number of crossroads passed for now
        # self.max_loop = max_loop  # the maximum number of the crossroads Pacman can directly pass
        # # opposite direction; to avoid turn back
        # self.opposite_dir = {"left":"right", "right":"left", "up":"down", "down":"up"}

    def nextDir(self, return_Q=False):
        if self.last_dir is not None and self.last_dir in self.available_dir:
            choice = self.last_dir
            self.Q_value[self.dir_list.index(choice)] = 1
            for each in self.available_dir:
                if each != self.last_dir:
                    # set to 1 because all the 0 will be considered as unavailable directions
                    self.Q_value[self.dir_list.index(each)] = -1
        else:
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
            random_Q_value = np.tile(1, len(self.available_dir))
            for index, each in enumerate(self.available_dir):
                self.Q_value[self.dir_list.index(each)] = random_Q_value[index]
        self.Q_value = np.array(self.Q_value)
        # self.Q_value = self.Q_value / np.sum(self.Q_value)
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
    last_dir = "up"
    agent = LazyAgent(adjacent_data, cur_pos, last_dir)
    print(agent.available_dir)
    choice = agent.nextDir(return_Q = True)
    print("Choice : ", choice)
