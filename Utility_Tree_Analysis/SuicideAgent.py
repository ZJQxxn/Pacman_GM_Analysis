'''
Description:
    Suicide agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    21 Aug. 2020
'''

import numpy as np


import sys
sys.path.append("./")
from PathTreeConstructor import PathTree

class SuicideAgent:

    def __init__(self, adjacent_data, adjacent_path, locs_df, reward_amount, cur_pos, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status, last_dir,
                 depth = 10, ghost_attractive_thr = 34,ghost_repulsive_thr = 10,  fruit_attractive_thr = 10):
        '''
        Initialization.
        :param adjacent_data: Map adjacent data (dict).
        :param locs_df: Locations distance (dict).
        :param reward_amount: Reward amount (dict).
        :param root: Pacman position of 2-tuple.
        :param energizer_data: A list of positions of energizers. Each position should be a 2-tuple.
        :param bean_data: A list of positions of bens. Each position should be a 2-tuple.
        :param ghost_data: A list of positions of ghosts. Each position should be a 2-tuple. If no ghost exists, pass np.nan.
        :param reward_type: The type pf reward (int).
        :param fruit_pos: The position of fruit. Should be a 2-tuple.
        :param ghost_status: A list of ghost status. Each status should be either 1(normal) or 4 (scared). If no ghost exists, pass np.nan.
        :param depth: The maximum depth of tree. 
        :param ghost_attractive_thr: Ghost attractive threshold.
        :param ghost_repulsive_thr: Ghost repulsive threshold.
        :param fruit_attractive_thr: Fruit attractive threshold.
        '''
        # Parameter type check
        if not isinstance(cur_pos, tuple):
            raise TypeError("The root should be a 2-tuple, but got a {}.".format(type(cur_pos)))
        if not isinstance(depth, int):
            raise TypeError("The depth should be a integer, but got a {}.".format(type(depth)))
        if depth <= 0:
            raise ValueError("The depth should be a positive integer.")
        # Game status
        self.reborn_pos = (14, 27)
        self.cur_pos = cur_pos
        self.cur_pos = cur_pos
        self.ghost_pos = ghost_data
        self.ghost_status = ghost_status
        self.reward_pos = fruit_pos
        self.last_dir = last_dir
        self.is_suicide = False
        # Construct two utility tree with current position and reborn position as the tree root
        self.cur_pos_tree = PathTree(
            adjacent_data,
            locs_df,
            reward_amount,
            self.cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth = depth,
            ghost_attractive_thr = ghost_attractive_thr,
            ghost_repulsive_thr = ghost_repulsive_thr,
            fruit_attractive_thr = fruit_attractive_thr
        )
        self.reborn_pos_tree = PathTree(
            adjacent_data,
            locs_df,
            reward_amount,
            self.reborn_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth=depth,
            ghost_attractive_thr=ghost_attractive_thr,
            ghost_repulsive_thr=ghost_repulsive_thr,
            fruit_attractive_thr = fruit_attractive_thr
        )
        # Obtain available directions from the current location
        self.adjacent_pos = adjacent_data[self.cur_pos]
        self.available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        # Directions
        self.dir_list = ['left', 'right', 'up', 'down']
        self.opposite_dir = {"left": "right", "right": "left", "up": "down", "down": "up"}
        # Other pre-computed data
        self.adjacent_data = adjacent_data
        self.adjacent_path = adjacent_path
        self.locs_df = locs_df
        self.reward_amount = reward_amount
        self.locs_df = locs_df  # distance between locations
        # Utility (Q-value) for every direction
        self.Q_value = [0, 0, 0, 0]
        # Direction list
        self.dir_list = ['left', 'right', 'up', 'down']


    def _relativeDir(self, cur_pos, destination):
        '''
        Determine the relative direction of the adjacent destination given the current location of Pacman.
        :param cur_pos: Current position of Pacman.
        :param destination: Location of destination.
        :return: Relative direction. "left"/"right"/"up"/"down"/None. Heere, None denoting that two positions are the same.
        '''
        if cur_pos[0] < destination[0]:
            return "right"
        elif cur_pos[0] > destination[0]:
            return "left"
        elif cur_pos[1] > destination[1]:
            return "up"
        elif cur_pos[1] < destination[1]:
            return "down"
        else:
            return None


    def nextDir(self, return_Q = False):
        # TODO: need reconstruction; this function is a little complicated; simplify
        # Construct paths and compute global utilities
        self.cur_pos_tree, _, _ = self.cur_pos_tree._construct()
        self.reborn_pos_tree, _, _ = self.reborn_pos_tree._construct()
        cur_pos_utility = sum([each.cumulative_utility for each in self.cur_pos_tree.leaves])
        reborn_pos_utility = sum([each.cumulative_utility for each in self.reborn_pos_tree.leaves])
        is_suicide_better = (reborn_pos_utility > cur_pos_utility)
        # Compute istance between Pacman and ghosts
        P_G_distance = []  # d
        for each in self.ghost_pos:
            each = tuple(each)
            if each in self.locs_df[self.cur_pos]:
                P_G_distance.append(self.locs_df[self.cur_pos][each])
            else:
                print("Lost path : {} to {}".format(self.cur_pos, each))
        P_G_distance = np.array(P_G_distance)
        closest_ghost_index = np.argmin(P_G_distance)
        # Suicide. Run to ghosts.
        if is_suicide_better:
            self.is_suicide = True
            if self.cur_pos != tuple(self.ghost_pos[closest_ghost_index]):
                choice = self._relativeDir(
                    self.cur_pos,
                    self.adjacent_path[
                        (self.adjacent_path.pos1 == self.cur_pos) &
                        (self.adjacent_path.pos2 == tuple(self.ghost_pos[closest_ghost_index]))
                    ].path.values[0][0][1]
                )
                # TODO: the escape direction and suicide direction
                # The escape direction (cur_opposite_dir) should be not the same as the suicide direction (choice)
                available_wo_suicide_dir = self.available_dir.copy()
                available_wo_suicide_dir.remove(choice)
                if self.last_dir is not None and self.opposite_dir[self.last_dir] in available_wo_suicide_dir:
                    cur_opposite_dir = self.opposite_dir[self.last_dir]
                else:
                    cur_opposite_dir = np.random.choice(available_wo_suicide_dir, 1).item()
                self.Q_value[self.dir_list.index(cur_opposite_dir)] = cur_pos_utility
                self.Q_value[self.dir_list.index(choice)] = reborn_pos_utility
            else:  # Pacman meets the ghost
                choice = self.last_dir
                self.Q_value[self.dir_list.index(self.last_dir)] = 1
            if choice is None:
                choice = np.random.choice(range(len(self.available_dir)), 1).item()
                choice = self.available_dir[choice]
                random_Q_value = np.tile(1 / len(self.available_dir), len(self.available_dir))
                for index, each in enumerate(self.available_dir):
                    self.Q_value[self.dir_list.index(each)] = random_Q_value[index]
        # Evade. Run away to the opposite direction
        elif self.last_dir is not None:
            cur_opposite_dir = self.opposite_dir[self.last_dir]
            if cur_opposite_dir in self.available_dir:
                choice = cur_opposite_dir
                if self.cur_pos != tuple(self.ghost_pos[closest_ghost_index]):
                    suicide_direction = self._relativeDir(
                        self.cur_pos,
                        self.adjacent_path[
                            (self.adjacent_path.pos1 == self.cur_pos) &
                            (self.adjacent_path.pos2 == tuple(self.ghost_pos[closest_ghost_index]))
                            ].path.values[0][0][1]
                    )
                    self.Q_value[self.dir_list.index(suicide_direction)] = reborn_pos_utility
                self.Q_value[self.dir_list.index(cur_opposite_dir)] = cur_pos_utility

            else:
                choice = np.random.choice(range(len(self.available_dir)), 1).item()
                choice = self.available_dir[choice]
        # Else, random choice
        else:
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
            random_Q_value = np.tile(1 / len(self.available_dir), len(self.available_dir))
            for index, each in enumerate(self.available_dir):
                self.Q_value[self.dir_list.index(each)] = random_Q_value[index]
        # Normalization
        self.Q_value = np.array(self.Q_value)
        self.Q_value = self.Q_value / np.sum(self.Q_value)
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
    # Suicide agent
    cur_pos = (7, 16)
    ghost_data = [(21, 5), (22, 5)]
    ghost_status = [4, 4]
    reward_pos = [(13, 9)]
    energizer_data = [(19, 27)]
    bean_data = [(20, 27)]
    reward_type = 3
    fruit_pos = (22, 27)
    last_dir = "down"
    agent = SuicideAgent(
        adjacent_data, adjacent_path, locs_df, reward_amount,
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status,
        last_dir,
        depth = 10, ghost_attractive_thr = 34, ghost_repulsive_thr = 34, fruit_attractive_thr = 34)
    choice= agent.nextDir(return_Q = True)
    print("Choice : ", choice)
    print("Is suicide : ", agent.is_suicide)
