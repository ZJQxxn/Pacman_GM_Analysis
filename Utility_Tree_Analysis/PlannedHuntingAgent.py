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
from PathTreeAgent import PathTree


class PlannedHuntingAgent:

    def __init__(self, adjacent_data, adjacent_path, locs_df, reward_amount, cur_pos, energizer_data, ghost_data, ghost_status, last_dir,
                 ghost_attractive_thr = 15, energizer_attractive_thr = 15, randomness_coeff = 1.0, laziness_coeff = 1.0):
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
        self.ghost_attractive_thr = ghost_attractive_thr
        self.energizer_attractive_thr = energizer_attractive_thr

    def _descendantUtility(self, node):
        leaves_utility = []
        for each in node.leaves:
            leaves_utility.append(each.path_utility)
        return sum(leaves_utility) / len(leaves_utility)


    def nextDir(self, return_Q = False):
        # If ghosts are scared or no energizer exists, degenerate to random agent
        if np.all(self.ghost_status >= 3) or isinstance(self.energizer_data, float) or self.energizer_data == []:
            if np.any(self.ghost_status > 3):
                # # Build a local path tree based on the current position
                # cur_pos_tree, _, _ = PathTree(
                #     self.adjacent_data,
                #     self.locs_df,
                #     self.reward_amount,
                #     self.cur_pos,
                #     np.nan, # ignore energizers
                #     np.nan, # ignore beans
                #     self.ghost_data,
                #     np.nan, # ignore fruits
                #     np.nan, # inore fruits
                #     self.ghost_status,
                #     self.last_dir,
                #     depth = 15,
                #     ignore_depth = 0,
                #     ghost_attractive_thr=15,
                #     ghost_repulsive_thr=15,
                #     fruit_attractive_thr=15,
                #     reward_coeff = 1.0,
                #     risk_coeff = 0.0,
                #     randomness_coeff=0.0,
                #     laziness_coeff=0.0
                # )._construct()
                # available_directions = [each.dir_from_parent for each in cur_pos_tree.children]
                # available_dir_utility = np.array([self._descendantUtility(each) for each in cur_pos_tree.children])
                # for index, each in enumerate(available_directions):
                #     self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]

                # Compute the distance between adjacent positions of Pacman and ghosts
                P_G_distance = []  # (# of adjacent positions, # of ghosts)
                for each_adjacent_pos in self.adjacent_pos:
                    temp_P_G_distance = []
                    for index, each_ghost in enumerate(self.ghost_data):
                        if self.ghost_status[index] == 3:
                            continue
                        if tuple(each_ghost) != each_adjacent_pos:
                            temp_P_G_distance.append(self.locs_df[each_adjacent_pos][tuple(each_ghost)])
                        else:
                            temp_P_G_distance.append(0.0)
                    P_G_distance.append(temp_P_G_distance)
                P_G_distance = np.array(P_G_distance)
                closest_P_G_distance = []
                for each in range(len(self.adjacent_pos)):
                    closest_P_G_distance.append(np.min(P_G_distance[each]))
                # Compute utility of each adjacent positions (i.e., each moving direction)
                available_dir_utility = []
                for adjacent_index in range(len(self.available_dir)):
                    P_G = closest_P_G_distance[adjacent_index]
                    temp_utility = 0.0
                    # Ghost reward
                    ghost_attractive_thr = 15
                    if P_G < ghost_attractive_thr:
                        R = self.reward_amount[8]
                        T = ghost_attractive_thr
                        if P_G <= ghost_attractive_thr:
                            temp_utility += (-R / T) * P_G + R
                    available_dir_utility.append(temp_utility)
                available_dir_utility = np.array(available_dir_utility)
                for index, each in enumerate(self.available_dir):
                    self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]
                self.Q_value = np.array(self.Q_value)
            else:
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
                        temp_P_E_distance.append(0.0)
                    else:
                        print("Lost path : {} to {}".format(each_adjacent_pos, each_energizer_pos))
                P_E_distance.append(temp_P_E_distance)
            P_E_distance = np.array(P_E_distance)
            # Compute the distance between adjacent positions of Pacman and ghosts
            P_G_distance = []  # (# of adjacent positions, # of ghosts)
            for each_adjacent_pos in self.adjacent_pos:
                temp_P_G_distance = []
                for each_ghost in self.ghost_data:
                    if tuple(each_ghost) != each_adjacent_pos:
                        temp_P_G_distance.append(self.locs_df[each_adjacent_pos][tuple(each_ghost)])
                    else:
                        temp_P_G_distance.append(0.0)
                P_G_distance.append(temp_P_G_distance)
            P_G_distance = np.array(P_G_distance)
            # distance for closest energizer and closest ghost
            closest_energizer_index = np.argmin(P_E_distance, axis = 1) # closest energizer index for every adjacent position
            closest_P_E_distance = []
            for index, each in enumerate(closest_energizer_index):
                closest_P_E_distance.append(P_E_distance[index][each])
            closest_E_G_distance = []
            for each in closest_energizer_index:
                closest_E_G_distance.append(np.min(E_G_distance[each]))
            closest_P_G_distance = []
            for each in range(len(self.adjacent_pos)):
                closest_P_G_distance.append(np.min(P_G_distance[each]))
            # Compute utility of each adjacent positions (i.e., each moving direction)
            available_dir_utility = []
            for adjacent_index in range(len(self.available_dir)):
                P_E = closest_P_E_distance[adjacent_index]
                E_G = closest_E_G_distance[adjacent_index]
                P_G = closest_P_G_distance[adjacent_index]

                temp_utility = 0.0
                # Energizer reward
                energizer_attractive_thr = self.energizer_attractive_thr
                if P_E < energizer_attractive_thr:
                    R = self.reward_amount[2] + self.reward_amount[8]
                    T = energizer_attractive_thr
                    if P_E <= energizer_attractive_thr:
                        temp_utility += (-R / T) * P_E + R
                # temp_utility *= 2
                # temp_utility = (
                #     self.reward_amount[2] / P_E # reward for energizer
                #     # + self.reward_amount[8] / E_G # reward for ghost
                # )

                # Ghost risk
                ghost_repulsive_thr = self.ghost_attractive_thr
                if P_G < ghost_repulsive_thr:
                    R = self.reward_amount[8]
                    T = ghost_repulsive_thr
                    if P_G <= ghost_repulsive_thr:
                        temp_utility -= (-R / T) * P_G + R
                # if -5 < (E_G - P_E) < 0: # only compute risk when ghost is more closer to energizer than the Pacman
                #     temp_utility = temp_utility + self.reward_amount[9] / (E_G - P_E) # risk for being eaten by ghost
                available_dir_utility.append(temp_utility)
            available_dir_utility = np.array(available_dir_utility)
            for index, each in enumerate(self.available_dir):
                self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]
            self.Q_value = np.array(self.Q_value)
        self.Q_value = np.array(self.Q_value, dtype = np.float32)
        available_directions_index = [self.dir_list.index(each) for each in self.available_dir]
        # self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
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
    cur_pos = (22, 27) # 1457
    ghost_data = [(16, 26), (26, 24)]
    ghost_status = [3, 5]
    energizer_data = [(18, 5)]
    bean_data = [(13, 8), (3, 9), (4, 9), (13, 9), (10, 22), (26, 24), (27, 24), (27, 25), (27, 28), (10, 30), (13, 30), (19, 30), (2, 33), (9, 33)]
    reward_type = None
    fruit_pos = None
    last_dir = "right"

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
        ghost_attractive_thr=12,
        energizer_attractive_thr=12,
        randomness_coeff = 0.0,
        laziness_coeff = 0.0
    )
    _, Q = agent.nextDir(return_Q=True)
    choice = agent.dir_list[makeChoice(Q)]
    print("Choice : ", choice, Q)
    print("Position : ", agent.cur_pos)
    print("Available directions : ", agent.available_dir)