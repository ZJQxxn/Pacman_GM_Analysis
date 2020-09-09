'''
Description:
    Planned hunting agent.If the ghosts are scared or no energizer exists, the planned hunting agent degenerates to 
    random agent. Else, when ghosts are normal, the Pacman has chances to plan hunting by reaching out to the energizer 
    intentionally.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    25 Aug. 2020
'''

import numpy as np
import sys
sys.path.append("./")
from TreeAnalysisUtils import scaleOfNumber


class PlannedHuntingAgent:

    def __init__(self, adjacent_data, adjacent_path, locs_df, reward_amount, cur_pos, energizer_data, ghost_data, ghost_status, last_dir,
                 randomness_coeff = 1.0, laziness_coeff = 1.0):
        # Game status (energizer)
        self.cur_pos = cur_pos
        self.energizer_data = energizer_data
        self.ghost_data = ghost_data
        self.ghost_status = np.array(ghost_status)
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


    def nextDir(self, return_Q = False):
        # If ghosts are scared or no energizer exists, degenerate to random agent
        if np.all(self.ghost_status >= 3) or isinstance(self.energizer_data, float) or self.energizer_data == []:
            self.Q_value = np.array([0.0, 0.0, 0.0, 0.0])
        # Else, has chance to plan hunting
        else:
            # Compute the distance between energizers and ghosts
            E_G_distance = [] # (# of energizers, # of ghosts)
            for each_energizer in self.energizer_data:
                temp_E_G_distance = []
                for each_ghost in self.ghost_data:
                    if tuple(each_ghost) != each_energizer:
                        temp_E_G_distance.append(self.locs_df[each_energizer][tuple(each_ghost)])
                    else:
                        temp_E_G_distance.append(0.0)
                E_G_distance.append(temp_E_G_distance)
            E_G_distance = np.array(E_G_distance)
            # Compute the distance between adjacent positions of Pacman and energizers
            P_E_distance = [] # (# of adjacent positions, # of energizers)
            for each_adjacent_pos in self.adjacent_pos:
                temp_P_E_distance = []
                # energizer distance
                for each_energizer_pos in self.energizer_data:
                    if each_energizer_pos in self.locs_df[each_adjacent_pos]:
                        temp_P_E_distance.append(self.locs_df[each_adjacent_pos][each_energizer_pos])
                    elif each_energizer_pos == each_adjacent_pos:
                        temp_P_E_distance.append(1.0)
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
            for each in closest_energizer_index:
                closest_E_G_distance.append(np.min(E_G_distance[each]))
            # Compute utility of each adjacent positions (i.e., each moving direction)
            available_dir_utility = []
            for adjacent_index in range(len(self.available_dir)):
                P_E = closest_P_E_distance[adjacent_index]
                P_E = 1 if 0 == P_E else P_E
                E_G = closest_E_G_distance[adjacent_index]
                E_G = 1 if 0 == E_G else E_G
                # temp_utility = (E_G - P_E) # pacman is closer to energizer compared with ghost

                temp_utility = (
                    self.reward_amount[2] / P_E # reward for energizer
                    + self.reward_amount[8] / E_G # reward for ghost
                )
                if -5 < (E_G - P_E) < 0: # only compute risk when ghost is more closer to energizer than the Pacman
                    temp_utility = temp_utility + self.reward_amount[9] / (E_G - P_E) # risk for being eaten by ghost
                available_dir_utility.append(temp_utility)
            available_dir_utility = np.array(available_dir_utility)
            for index, each in enumerate(self.available_dir):
                self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]
            self.Q_value = np.array(self.Q_value)
        self.Q_value = np.array(self.Q_value, dtype = np.float32)
        available_directions_index = [self.dir_list.index(each) for each in self.available_dir]
        self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
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
    from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath

    # Read data
    locs_df = readLocDistance("./extracted_data/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    adjacent_path = readAdjacentPath("./extracted_data/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    print("Finished reading auxiliary data!")
    # Planned hunting agent
    cur_pos = (7, 16)
    ghost_data = [(21, 5), (22, 5)]
    ghost_status = [1, 1]
    energizer_data = [(19, 27), (19, 28)]
    last_dir = "up"
    agent = PlannedHuntingAgent(
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        ghost_data,
        ghost_status,
        last_dir,
        randomness_coeff = 0.0,
        laziness_coeff = 0.0
    )
    choice = agent.nextDir(return_Q = True)
    print("Position : ", agent.cur_pos)
    print("Available directions : ", agent.available_dir)
    print("Choice : ", choice)

