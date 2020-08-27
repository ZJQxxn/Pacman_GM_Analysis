'''
Description:
    Construct a utility tree along the estimated path. Integrate global, local, optimistic, and pessimistic agents. Add 
    randomness and laziness.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Aug. 25 2020
    
Update History:
    25 Aug. 2020: Set (14, 16) and (15, 16) to be a wall in common_data/map_info_brian.csv; reconstruct map adjacent_map.csv
'''


import pandas as pd
import numpy as np
import anytree
from anytree.exporter import DotExporter
from collections import deque
import sys
import time

sys.path.append('./')
from TreeAnalysisUtils import unitStepFunc, scaleOfNumber


class PathTree:

    def __init__(self, adjacent_data, locs_df, reward_amount, root, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status, last_dir,
                 depth = 10, ignore_depth = 0, ghost_attractive_thr = 34, ghost_repulsive_thr = 10, fruit_attractive_thr = 10,
                 randomness_coeff = 1.0, laziness_coeff = 1.0, reward_coeff = 1.0, risk_coeff = 1.0):
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
        :param ignore_depth: Ignore this depth of nodes. 
        :param ghost_attractive_thr: Ghost attractive threshold.
        :param ghost_repulsive_thr: Ghost repulsive threshold.
        :param fruit_attractive_thr: Fruit attractive threshold.
        :param reward_coeff: Coefficient for the reward.
        :param risk_coeff: Coefficient for the risk.
        '''
        # Parameter type check
        if not isinstance(root, tuple):
            raise TypeError("The root should be a 2-tuple, but got a {}.".format(type(root)))
        if not isinstance(depth, int):
            raise TypeError("The depth should be a integer, but got a {}.".format(type(depth)))
        if depth <= 0:
            raise ValueError("The depth should be a positive integer.")
        # Other initialization
        # The root node is the path starting point.
        # Other tree nodes should contain:
        #   (1) location ("name")
        #   (2) parent location ("parent")
        #   (3) the direction from its to parent to itself ("dir_from_parent")
        #   (4) utility of this node, reward and  risk are separated ("cur_reward", "cur_risk", "cur_utility")
        #   (5) the cumulative utility so far, reward and risk are separated ("cumulative_reward", "cumulative_risk", "cumulative_utility")
        self.root = anytree.Node(root,
                                 cur_utility = 0,
                                 cumulative_utility = 0,
                                 cur_reward = 0,
                                 cumulative_reward = 0,
                                 cur_risk = 0,
                                 cumulative_risk = 0
                                 )
        # The current node
        self.current_node = self.root
        # A queue used for append nodes on the tree
        self.node_queue = deque()
        self.node_queue.append(self.root)
        # The maximize depth (i.e., the path length)
        self.depth = depth
        # The ignore depth (i.e., exclude this depth of nodes)
        self.ignore_depth = ignore_depth
        # Game status
        self.energizer_data = energizer_data
        self.bean_data = bean_data
        self.ghost_data = [tuple(each) for each in ghost_data]
        self.ghost_status = ghost_status
        # Fruit data
        self.reward_type = reward_type
        self.fruit_pos = fruit_pos
        # Pre-defined thresholds for utility computation
        self.ghost_attractive_thr = ghost_attractive_thr
        self.fruit_attractive_thr = fruit_attractive_thr
        self.ghost_repulsive_thr = ghost_repulsive_thr
        # Other pre-computed data
        self.adjacent_data = adjacent_data
        self.locs_df = locs_df
        self.reward_amount = reward_amount
        self.existing_bean = bean_data
        self.existing_energizer = energizer_data
        self.existing_fruit = fruit_pos
        # Utility (Q-value) for every direction
        self.Q_value = [0, 0, 0, 0]
        # Direction list
        self.dir_list = ['left', 'right', 'up', 'down']
        # Last direction
        self.last_dir = last_dir
        # Trade-off between risk and reward
        self.reward_coeff = reward_coeff
        self.risk_coeff = risk_coeff
        # For randomness and laziness
        self.randomness_coeff = randomness_coeff
        self.laziness_coeff = laziness_coeff
        # Pacman is eaten? If so, the path will be ended
        self.is_eaten = False


    def _construct(self):
        '''
        Construct the utility tree.
        :return: The tree root node (anytree.Node).
        '''
        # construct the first layer firstly (depth = 1)
        self._attachNode(ignore = True if self.ignore_depth > 0 else False) # attach all children of the root (depth = 1)
        self.node_queue.append(None) # the end of layer with depth = 1
        self.node_queue.popleft()
        self.current_node = self.node_queue.popleft()
        cur_depth = 2
        # construct the other parts
        while cur_depth <= self.depth:
            if cur_depth <= self.ignore_depth:
                ignore = True
            else:
                ignore = False
            while None != self.current_node :
                self._attachNode(ignore = ignore)
                self.current_node = self.node_queue.popleft()
            self.node_queue.append(None)
            if 0 == len(self.node_queue):
                break
            self.current_node = self.node_queue.popleft()
            cur_depth += 1
        # Find the best path with the highest utility
        best_leaf = self.root.leaves[0]
        for leaf in self.root.leaves:
            if leaf.cumulative_utility > best_leaf.cumulative_utility:
                best_leaf = leaf
        highest_utility = best_leaf.cumulative_utility
        best_path = best_leaf.ancestors
        best_path = [(each.name, each.dir_from_parent) for each in best_path[1:]]
        if best_path == []: # only one step is taken
            best_path = [(best_leaf.name, best_leaf.dir_from_parent)]
        return self.root, highest_utility, best_path


    def _attachNode(self, ignore = False):
        # Find adjacent positions and the corresponding moving directions for the current node
        tmp_data = self.adjacent_data[self.current_node.name]
        for each in ["left", "right", "up", "down"]:
            # do not walk on the wall or walk out of boundary
            # do not turn back # TODO: turn back?
            if None == self.current_node.parent and isinstance(tmp_data[each], float):
                continue
            elif None != self.current_node.parent and \
                    (isinstance(tmp_data[each], float)
                     or tmp_data[each] == self.current_node.parent.name):
                continue
            else:
                # Compute utility
                cur_pos = tmp_data[each]
                if ignore:
                    cur_reward = 0.0
                    cur_risk = 0.0
                else:
                    cur_reward = self._computeReward(cur_pos)
                    # if the position is visited before, do not add up the risk to cumulative
                    if cur_pos in [each.name for each in self.current_node.path]:
                        cur_risk = 0.0
                    else:
                        cur_risk = self._computeRisk(cur_pos)
                # Construct the new node
                new_node = anytree.Node(
                        cur_pos,
                        parent = self.current_node,
                        dir_from_parent = each,
                        cur_utility = self.reward_coeff * cur_reward + self.risk_coeff * cur_risk,
                        cumulative_utility = self.current_node.cumulative_utility + self.reward_coeff * cur_reward + self.risk_coeff * cur_risk,
                        cur_reward = cur_reward,
                        cumulative_reward = self.current_node.cumulative_reward + cur_reward,
                        cur_risk = cur_risk,
                        cumulative_risk = self.current_node.cumulative_risk + cur_risk
                        )
                # If the Pacman is eaten, end this path
                if self.is_eaten:
                    self.is_eaten = False
                else:
                    self.node_queue.append(new_node)


    def _computeReward(self, cur_position):
        reward = 0
        # Bean reward
        if isinstance(self.existing_bean, float):
            reward += 0
        elif cur_position in self.existing_bean:
            reward += self.reward_amount[1]
            self.existing_bean.remove(cur_position)
        else:
            reward += 0
        # Energizer reward
        if isinstance(self.energizer_data, float) or cur_position not in self.energizer_data:
            reward += 0
        else:
            # Reward for eating the energizer
            if cur_position in self.existing_energizer:
                reward += self.reward_amount[2]
                self.existing_energizer.remove(cur_position)
                self.ghost_status = [4 if each != 3 else 3 for each in self.ghost_status]  # change ghost status
                # Potential reward for ghosts
                ifscared1 = self.ghost_status[0] if not isinstance(self.ghost_status[0], float) else 0
                ifscared2 = self.ghost_status[1] if not isinstance(self.ghost_status[1], float) else 0
                if 4 == ifscared1 or 4 == ifscared2:  # ghosts are scared
                    if 3 == ifscared1:
                        ghost_dist = self.locs_df[cur_position][self.ghost_data[1]]
                    elif 3 == ifscared2:
                        ghost_dist = self.locs_df[cur_position][self.ghost_data[0]]
                    else:
                        if cur_position != self.ghost_data[0] and cur_position != self.ghost_data[1]:
                            ghost_dist = min(
                                self.locs_df[cur_position][self.ghost_data[0]], self.locs_df[cur_position][self.ghost_data[1]]
                            )
                        else:
                            ghost_dist = 1 #TODO: change to 0 and revise the division
                    if ghost_dist < self.ghost_attractive_thr:
                        reward += self.reward_amount[8] * (1 / ghost_dist)
            else:
                reward += 0

        # Ghost reward (check whether ghosts are scared)
        ifscared1 = self.ghost_status[0] if not isinstance(self.ghost_status[0], float) else 0
        ifscared2 = self.ghost_status[1] if not isinstance(self.ghost_status[1], float) else 0
        if 4 == ifscared1 or 4 == ifscared2:  # ghosts are scared
            if cur_position not in self.ghost_data:
                # compute ghost dist
                if 3 == ifscared1:
                    ghost_dist = self.locs_df[cur_position][self.ghost_data[1]]
                elif 3 == ifscared2:
                    ghost_dist = self.locs_df[cur_position][self.ghost_data[0]]
                else:
                    ghost_dist = min(
                        self.locs_df[cur_position][self.ghost_data[0]], self.locs_df[cur_position][self.ghost_data[1]]
                    )
                if ghost_dist < self.ghost_attractive_thr:
                    reward += self.reward_amount[8] * (1 / ghost_dist)
            elif cur_position in self.ghost_data:
                reward += self.reward_amount[8]
                if cur_position == self.ghost_data[0]:
                    self.ghost_status[0] = 3
                else:
                    self.ghost_status[1] = 3
            else:
                reward += 0
        # Fruit reward 
        if not isinstance(self.existing_fruit, float):
            if cur_position == self.fruit_pos:
                reward += self.reward_amount[int(self.reward_type)]
                self.existing_fruit = np.nan
            else:
                fruit_dist = self.locs_df[cur_position][self.fruit_pos]
                if fruit_dist < self.fruit_attractive_thr:
                    reward += self.reward_amount[int(self.reward_type)] * (1 / fruit_dist)
        return reward


    def _computeRisk(self, cur_position):
        # Compute ghost risk when ghosts are normal
        ifscared1 = self.ghost_status[0] if not isinstance(self.ghost_status[0], float) else 0
        ifscared2 = self.ghost_status[1] if not isinstance(self.ghost_status[1], float) else 0
        if ifscared1 <= 2 or ifscared2 <= 2: # ghosts are normal; use "or" for dealing with dead ghosts
            if 3 == ifscared1:
                # Pacman is eaten
                if cur_position == self.ghost_data[1]:
                    risk = -self.reward_amount[9]
                    self.is_eaten = True
                    return risk
                ghost_dist = self.locs_df[cur_position][self.ghost_data[1]]
            elif 3 == ifscared2:
                # Pacman is eaten
                if cur_position == self.ghost_data[0]:
                    risk = -self.reward_amount[9]
                    self.is_eaten = True
                    return risk
                ghost_dist = self.locs_df[cur_position][self.ghost_data[0]]
            else:
                # Pacman is eaten
                if cur_position == self.ghost_data[0] or cur_position == self.ghost_data[1]:
                    risk = -self.reward_amount[9]
                    self.is_eaten = True
                    return risk
                # Potential risk
                else:
                    ghost_dist = min(
                        self.locs_df[cur_position][self.ghost_data[0]],
                        self.locs_df[cur_position][self.ghost_data[1]]
                    )
            if ghost_dist < self.ghost_repulsive_thr:
                risk = -self.reward_amount[9] * 1 / ghost_dist #TODO: [8] or [9] ?
            else:
                risk = 0
        # Ghosts are not scared
        else:
            risk = 0
        return risk


    def _descendantUtility(self, node):
        utility = 0.0
        for each in node.leaves:
            utility += each.cumulative_utility
        return utility


    def nextDir(self, return_Q = False):
        _, highest_utility, best_path = self._construct()
        available_directions = [each.dir_from_parent for each in self.root.children]
        available_dir_utility = np.array([self._descendantUtility(each) for each in self.root.children])
        for index, each in enumerate(available_directions):
            self.Q_value[self.dir_list.index(each)] = available_dir_utility[index]
        self.Q_value = np.array(self.Q_value)
        available_directions_index = [self.dir_list.index(each) for each in available_directions]
        self.Q_value[available_directions_index] += 1.0 # avoid 0 utility
        # Add randomness and laziness
        self.Q_value[available_directions_index] += (self.randomness_coeff * np.random.normal(size=len(available_directions_index)))
        if self.last_dir is not None and self.dir_list.index(self.last_dir) in available_directions_index:
            self.Q_value[self.dir_list.index(self.last_dir)] += (self.laziness_coeff * scaleOfNumber(np.max(np.abs(self.Q_value))))
        if return_Q:
            return best_path[0][1], self.Q_value
        else:
            return best_path[0][1]



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

    cur_pos = (13, 27)
    ghost_data = [(14, 17), (14, 18)]
    ghost_status = [1, 1]
    energizer_data = [(22,6)]
    bean_data = [(22, 5), (23, 5), (24, 5), (26, 5), (27, 6), (16, 7), (22, 8), (17, 9), (19, 9), (22, 9), (23, 9),
                 (24, 9), (27, 9), (19, 10), (22, 10), (19, 11), (19, 12), (22, 12), (23, 12), (25, 12), (22, 15),
                 (1, 18), (5, 18), (6, 18), (22, 18), (23, 18), (26, 18), (27, 18), (22, 19), (27, 25), (2, 26), (27, 26),
                 (2, 27), (27, 30), (27, 31), (3, 33), (8, 33), (19, 33), (20, 33), (22, 33), (23, 33), (27, 33), (22, 6),
                 (27, 32)]
    reward_type = 6
    fruit_pos = (19, 23)
    last_dir = "left"

    # Global agent
    agent = PathTree(
        adjacent_data,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status,
        last_dir,
        15,
        5,
        34,
        34,
        34
    )
    choice = agent.nextDir(return_Q = True)
    print("Global Agent Q : ", choice)

    # Local agent
    agent = PathTree(
        adjacent_data,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status,
        last_dir,
        5,
        0,
        5,
        5,
        5
    )
    choice = agent.nextDir(return_Q=True)
    print("Local Agent Q : ", choice)

    # Optimistice
    agent = PathTree(
        adjacent_data,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status,
        last_dir,
        10,
        0,
        34,
        34,
        12,
        reward_coeff = 1.0,
        risk_coeff = 0.0
    )
    choice = agent.nextDir(return_Q=True)
    print("Optimistic Q : ", choice)

    # Pessimistic
    agent = PathTree(
        adjacent_data,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status,
        last_dir,
        10,
        0,
        34,
        34,
        12,
        reward_coeff = 0.0,
        risk_coeff = 1.0
    )
    choice = agent.nextDir(return_Q=True)
    print("Pessimistic Q : ", choice)