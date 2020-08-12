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
        # self.not_turn = False # True only if the Pacman passes a crossroad without turning
        # self.loop_count = loop_count  # the number of crossroads passed for now
        # self.max_loop = max_loop  # the maximum number of the crossroads Pacman can directly pass
        self.dir_list = ['left', 'right', 'up', 'down']
        # # opposite direction; to avoid turn back
        # self.opposite_dir = {"left":"right", "right":"left", "up":"down", "down":"up"}


    # def _atTunnel(self):
    #     '''
    #     Determine whether the Pacman is in the tunnel (i.e., no possibility to turn its direction unless going back).
    #     :return: Return true if in the tunnel; otherwise return False.
    #     '''
    #     if 2 == len(self.available_dir):
    #         if "left" in self.available_dir and "right" in self.available_dir:
    #             return True
    #         elif "up" in self.available_dir and "down" in self.available_dir:
    #             return True
    #         else:
    #             return False
    #     else:
    #         return False
    #
    #
    # def _atCorner(self):
    #     '''
    #     Determine whether the Pacman is in the corner.
    #     :return: Return true if in the corner; otherwise return False.
    #     '''
    #     if 2 == len(self.available_dir):
    #        if not self._atTunnel():
    #            return True
    #        else:
    #            return False
    #     else:
    #         return False
    #
    #
    # def _atTJunction(self):
    #     '''
    #     Determine whether the Pacman is in the T-juntion.
    #     :return: Return true if in the T-junction; otherwise return False.
    #     '''
    #     if 3 == len(self.available_dir):
    #         if not self.last_dir in self.available_dir:
    #             return True
    #         else:return False
    #     else:
    #         return False


    # def nextDir(self):
    #     '''
    #     Estimate the moving direction.
    #     :return: A 2-tuple. The first element is the moving direction {`left'', ``right'', ``up'', ``down''}.
    #     '''
    #     # If this is the starting of the game, randomly choose a direction among available choices
    #     if None == self.last_dir:
    #         choice = np.random.choice(range(len(self.available_dir)), 1).item()
    #         choice = self.available_dir[choice]
    #     # else, stay moving to the same direction until the number of passed crossroads surpasses a threshold (default = 5)
    #     else:
    #         # If at the corner or in the tunnel (i.e., only one direction can be chosen except the last moving direction)
    #         if self._atTunnel() or self._atCorner():
    #             if self.opposite_dir[self.last_dir] in self.available_dir:
    #                 self.available_dir.remove(self.opposite_dir[self.last_dir])
    #             choice = self.available_dir[0]
    #         # else if at the T-junction and the Pacman must turn (i.e., two direction can be chosen except the last moving direction)
    #         elif self._atTJunction():
    #             if self.opposite_dir[self.last_dir] in self.available_dir:
    #                 self.available_dir.remove(self.opposite_dir[self.last_dir])
    #             choice = np.random.choice(range(len(self.available_dir)), 1).item()
    #             choice = self.available_dir[choice]
    #         # else if at the crossroads or T-junctions (without mandatory turning)
    #         else:
    #             # don't have to turn
    #             if self.loop_count < self.max_loop:
    #                 choice = self.last_dir
    #                 self.not_turn = True
    #             # turn for breaking the loop
    #             else:
    #                 if self.opposite_dir[self.last_dir] in self.available_dir:
    #                     self.available_dir.remove(self.opposite_dir[self.last_dir])
    #                 choice = np.random.choice(range(len(self.available_dir)), 1).item()
    #                 choice = self.available_dir[choice]
    #     return choice

    def nextDir(self):
        if self.last_dir is not None and self.last_dir in self.available_dir:
            return self.last_dir
        else:
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
            return choice




if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from TreeAnalysisUtils import readAdjacentMap
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    cur_pos = (22, 24)
    last_dir = "up"
    loop_count = 1
    agent = LazyAgent(adjacent_data, cur_pos, last_dir)
    print(agent.available_dir)
    choice = agent.nextDir()
    print(choice)
