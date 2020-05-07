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

    def __init__(self, adjacent_data, cur_pos, last_dir, random_seed):
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
            if None != self.adjacent_pos[dir]:
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.last_dir = last_dir  # moving direction for the last time step
        self.dir_list = ['left', 'right', 'up', 'down']


    def nextDir(self):
        '''
        Estimate the moving direction. 
        :return: The moving direction {`left'', ``right'', ``up'', ``down''}.
        '''
        # If this is the starting of the game, randomly choose a direction among available choices
        if None == self.last_dir:
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
        # else, stay moving to the same direction until the number of passed crossroads surpasses a threshold (default = 5)
        else:
            self.available_dir.remove(self.last_dir)
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
        return choice

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from TreeAnalysisUtils import adjacent_data
    cur_pos = (22, 24)
    last_dir = "right"
    loop_count = 1
    agent = RandomAgent(adjacent_data, cur_pos, last_dir)
    choice = agent.nextDir()
    print(choice)
