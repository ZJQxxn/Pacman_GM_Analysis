'''
Description:
    Suicide agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    21 Aug. 2020
'''

import numpy as np
import copy

import sys
sys.path.append("./")
from PathTreeAgent import PathTree
from TreeAnalysisUtils import scaleOfNumber


class SuicideAgent:

    def __init__(self, adjacent_data, adjacent_path, locs_df, reward_amount, cur_pos, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status, last_dir,
                 depth = 10, ghost_attractive_thr = 34,ghost_repulsive_thr = 10,  fruit_attractive_thr = 10, randomness_coeff = 1.0, laziness_coeff = 1.0):
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
            last_dir,
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
            last_dir,
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
        # For randomness
        self.randomness_coeff = randomness_coeff
        self.is_random = False
        # For laziness
        self.laziness_coeff = laziness_coeff


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


    def _descendantUtility(self, node):
        utility = 0.0
        for each in node.leaves:
            utility += each.cumulative_utility
        return utility / len(node.leaves)


    def nextDir(self, return_Q = False):
        # Construct paths and compute global utilities
        self.cur_pos_tree, _, _ = self.cur_pos_tree._construct()
        self.reborn_pos_tree, _, _ = self.reborn_pos_tree._construct()
        cur_pos_utility = {each.dir_from_parent : self._descendantUtility(each) for each in self.cur_pos_tree.children}
        reborn_pos_utility = np.max(
            [each.cumulative_utility for each in self.reborn_pos_tree.leaves]
        )
        self.is_suicide_better = np.all(reborn_pos_utility > np.array(list(cur_pos_utility.values())))
        # Compute distance between Pacman and ghosts for normalizing
        P_G_distance = []
        for each in self.ghost_pos:
            each = tuple(each)
            if each in self.locs_df[self.cur_pos]:
                P_G_distance.append(self.locs_df[self.cur_pos][each])
            else:
                P_G_distance.append(0.0)
                print("Lost path : {} to {}".format(self.cur_pos, each))
        P_G_distance = np.array(P_G_distance)
        # Determine the suicide direction and escape direction
        if self.cur_pos != tuple(self.ghost_pos[0]) and self.cur_pos != tuple(self.ghost_pos[1]):
            # The suicide direction: relative direction to all the ghosts
            suicide_direction = [
                self._relativeDir(
                    self.cur_pos,
                    self.adjacent_path[
                        (self.adjacent_path.pos1 == self.cur_pos) &
                        (self.adjacent_path.pos2 == tuple(self.ghost_pos[index]))
                        ].path.values[0][0][1]
                )
                for index in range(2)
            ]
        # Pacman meets ghosts: can not be caught at the beginning, so doesn't have to consider last_dir == None
        else:
            suicide_direction = []
            for index in range(2):
                if self.cur_pos == tuple(self.ghost_pos[index]) and self.last_dir in self.available_dir:
                    suicide_direction.append(self.last_dir)
                elif self.cur_pos == tuple(self.ghost_pos[index]) and self.last_dir not in self.available_dir:
                    suicide_direction.append(np.random.choice(self.available_dir, 1).item())
                else:
                    suicide_direction.append(self._relativeDir(
                                        self.cur_pos,
                                        self.adjacent_path[
                                            (self.adjacent_path.pos1 == self.cur_pos) &
                                            (self.adjacent_path.pos2 == tuple(self.ghost_pos[index]))
                                        ].path.values[0][0][1])
                    )
        # The escape direction: directions other than the suicide direction
        # escape_dir = self.available_dir.copy()
        escape_dir = copy.deepcopy(self.available_dir)
        for each in suicide_direction:
            if each in escape_dir:
                escape_dir.remove(each)
        # Assign utilities for escape directions
        for each in escape_dir:
            self.Q_value[self.dir_list.index(each)] = cur_pos_utility[each]
        # Assign utilities for suicide directions:
        # If relative directions w.r.t. two ghosts are the same, no need to normalize the utility.
        if suicide_direction[0] == suicide_direction[1]:
            self.Q_value[self.dir_list.index(suicide_direction[0])] = reborn_pos_utility
        # Else, normalize the utility based on their distance, a closer ghost might have a larger utility
        else:
            PG_normalizing_factor = P_G_distance / np.sum(P_G_distance) if np.sum(P_G_distance) != 0 else np.array([1.0, 1.0])
            PG_normalizing_factor = PG_normalizing_factor[::-1]
            for index, each in enumerate(suicide_direction):
                self.Q_value[self.dir_list.index(each)] = reborn_pos_utility * PG_normalizing_factor[index]
        self.Q_value = np.array(self.Q_value)
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
    # Suicide agent
    cur_pos = (7, 18)
    ghost_data = [(10, 15), (5, 18)]
    ghost_status = [1, 2]
    reward_pos = [(13, 9)]
    energizer_data = [(2, 5), (16, 5), (6, 33)]
    bean_data = [(3, 5), (11, 5), (18, 5), (20, 5), (21, 5), (22, 5), (27, 5), (27, 6), (22, 7), (27, 7), (13, 8),
                 (14, 9), (16, 9), (18, 9), (27, 9), (27, 10), (19, 11), (16, 12), (19, 12), (22, 12), (24, 12),
                 (16, 13), (13, 14), (16, 14), (2, 29), (2, 31), (7, 33), (8, 33), (2, 5), (16, 5), (6, 33)]
    reward_type = 3
    fruit_pos = (22, 13)
    last_dir = "up"
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
        depth = 5, ghost_attractive_thr = 10, ghost_repulsive_thr = 10, fruit_attractive_thr = 10,
        randomness_coeff = 0.0, laziness_coeff = 0.0
    )
    choice= agent.nextDir(return_Q = True)
    print("Choice : ", choice)
    print("Is suicide : ", agent.is_suicide)
