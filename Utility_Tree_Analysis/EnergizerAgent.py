'''
Description:
    Planned hunting agent.If the ghosts are scared or no energizer exists, the planned hunting agent degenerates to 
    random agent. Else, when ghosts are normal, the Pacman has chances to plan hunting by reaching out to the energizer 
    intentionally.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    16 Dec. 2020
'''

import numpy as np
import sys
sys.path.append("./")
from TreeAnalysisUtils import scaleOfNumber
from PathTreeAgent import PathTree


class EnergizerAgent:

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

    def _descendantUtility(self, node):
        leaves_utility = []
        for each in node.leaves:
            leaves_utility.append(each.path_utility)
        return sum(leaves_utility) / len(leaves_utility)


    def nextDir(self, return_Q = False):
        # If ghosts are scared or no energizer exists, degenerate to random agent
        if np.all(self.ghost_status >= 3) or np.any(self.ghost_status > 3) or isinstance(self.energizer_data, float) or self.energizer_data == []:
            if np.any(self.ghost_status > 3):
                self.Q_value = np.array([0.0, 0.0, 0.0, 0.0])
        # Else, has chance to plan hunting
        else:
            # Compute the distance between adjacent positions of Pacman and energizers
            P_E_distance = [] # (# of adjacent positions, # of energizers)
            for each_adjacent_pos in self.adjacent_pos:
                temp_P_E_distance = []
                # energizer distance
                for each_energizer_pos in self.energizer_data:
                    if each_energizer_pos in self.locs_df[each_adjacent_pos]:
                        temp_P_E_distance.append(self.locs_df[each_adjacent_pos][each_energizer_pos])
                    elif each_energizer_pos == each_adjacent_pos:
                        temp_P_E_distance.append(0.0)
                    else:
                        print("Lost path : {} to {}".format(each_adjacent_pos, each_energizer_pos))
                P_E_distance.append(temp_P_E_distance)
            P_E_distance = np.array(P_E_distance)
            # distance for closest energizer and closest ghost
            closest_energizer_index = np.argmin(P_E_distance, axis = 1) # closest energizer index for every adjacent position
            closest_P_E_distance = []
            for index, each in enumerate(closest_energizer_index):
                closest_P_E_distance.append(P_E_distance[index][each])
            closest_E_G_distance = []
            # Compute utility of each adjacent positions (i.e., each moving direction)
            available_dir_utility = []
            for adjacent_index in range(len(self.available_dir)):
                P_E = closest_P_E_distance[adjacent_index]
                temp_utility = 0.0
                # Energizer reward
                energizer_attractive_thr = self.energizer_attractive_thr
                if P_E < energizer_attractive_thr:
                    R = self.reward_amount[2]
                    temp_utility += R
                available_dir_utility.append(temp_utility)
            available_dir_utility = np.array(available_dir_utility)
            for index, each in enumerate(self.available_dir):
                self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]
            self.Q_value = np.array(self.Q_value)
        self.Q_value = np.array(self.Q_value, dtype = np.float32)
        available_directions_index = [self.dir_list.index(each) for each in self.available_dir]
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(self.Q_value)))
        randomness = np.random.normal(loc=0, scale=0.1, size=len(available_directions_index)) * Q_scale
        self.Q_value[available_directions_index] += (self.randomness_coeff * randomness)
        if self.last_dir is not None and self.dir_list.index(self.last_dir) in available_directions_index:
            self.Q_value[self.dir_list.index(self.last_dir)] += (self.laziness_coeff * Q_scale)
        choice = np.argmax(self.Q_value[available_directions_index])
        choice = self.available_dir[choice]
        if return_Q:
            return choice, self.Q_value
        else:
            return choice


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
    # Planned hunting agent
    cur_pos = (23, 33) # 1457
    ghost_data = [(4, 33), (2, 32)]
    ghost_status = [1, 1]
    energizer_data = [(12, 9), (27, 10), (25, 24), (6, 30)]
    bean_data = [(16, 9), (7, 12), (25, 12), (21, 9), (22, 6), (7, 11), (14, 9), (2, 5), (16, 14), (12, 9), (22, 20), (13, 12), (8, 24), (22, 27), (7, 6), (22, 14), (7, 19), (27, 10), (7, 28), (22, 28), (27, 28), (26, 12), (27, 32), (7, 14), (4, 5), (22, 17), (24, 24), (22, 11), (7, 22), (3, 12), (20, 24), (22, 12), (25, 24), (13, 5), (23, 9), (7, 30), (25, 33), (21, 24), (7, 21), (27, 6), (2, 12), (13, 8), (23, 30), (10, 30), (27, 33), (2, 9), (5, 12), (15, 9), (10, 12), (17, 24), (22, 7), (7, 10), (27, 5), (10, 22), (7, 23), (10, 9), (18, 24), (22, 15), (2, 10), (6, 30), (26, 30), (20, 5), (27, 31), (6, 12), (7, 9), (23, 5), (13, 14), (27, 7), (26, 5), (24, 12), (22, 30)]
    reward_type = None
    fruit_pos = None
    last_dir = "right"

    agent = EnergizerAgent(
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        ghost_data,
        ghost_status,
        bean_data,
        last_dir,
        ghost_attractive_thr=15,
        energizer_attractive_thr=15,
        beans_attractive_thr=5,
        randomness_coeff = 0.0,
        laziness_coeff = 0.0
    )
    _, Q = agent.nextDir(return_Q=True)
    choice = agent.dir_list[makeChoice(Q)]
    print("Choice : ", choice, Q)
    print("Position : ", agent.cur_pos)
    print("Available directions : ", agent.available_dir)