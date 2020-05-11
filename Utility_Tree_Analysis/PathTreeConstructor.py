'''
Description:
    Construct a utility tree along the estimated path.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 30 2020
'''


import pandas as pd
import numpy as np
import anytree
from anytree.exporter import DotExporter
from collections import deque
import sys
import time

sys.path.append('./')
from TreeAnalysisUtils import unitStepFunc


class PathTree:

    def __init__(self, adjacent_data, locs_df, reward_amount, root, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status,
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


    def construct(self):
        '''
        Construct the utility tree.
        :return: The tree root node (anytree.Node).
        '''
        # construct the first layer firstly (depth = 1)
        self._attachNode() # attach all children of the root (depth = 1)
        self.node_queue.append(None) # the end of layer with depth = 1
        self.node_queue.popleft()
        self.current_node = self.node_queue.popleft()
        cur_depth = 2
        # construct the other parts
        while cur_depth <= self.depth:
            while None != self.current_node :
                self._attachNode()
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
        return self.root, highest_utility, best_path


    def _attachNode(self):
        # Find adjacent positions and the corresponding moving directions for the current node
        tmp_data = self.adjacent_data[self.current_node.name]
        for each in ["left", "right", "up", "down"]:
            # do not walk on the wall or walk out of boundary
            # do not turn back
            if None == self.current_node.parent and isinstance(tmp_data[each], float):
                continue
            elif None != self.current_node.parent and \
                    (isinstance(tmp_data[each], float)
                     or tmp_data[each] == self.current_node.parent.name):
                continue
            else:
                # Compute utility
                cur_pos = tmp_data[each]
                cur_reward = self._computeReward(cur_pos)
                cur_risk = self._computeRisk(cur_pos)
                # Construct the new node
                new_node = anytree.Node(
                        cur_pos,
                        parent=self.current_node,
                        dir_from_parent = each,
                        cur_utility = cur_reward + cur_risk,
                        cumulative_utility = self.current_node.cumulative_utility + cur_reward + cur_risk,
                        cur_reward = cur_reward,
                        cumulative_reward = self.current_node.cumulative_reward + cur_reward,
                        cur_risk = cur_risk,
                        cumulative_risk = self.current_node.cumulative_risk + cur_risk
                        )
                self.node_queue.append(new_node)


    def _computeReward(self, cur_position):
        reward = 0
        # Bean reward
        if cur_position in self.existing_bean:
            reward += self.reward_amount[1]
            self.existing_bean.remove(cur_position)
        else:
            reward += 0
        # Energizer reward #TODO: revise this; split this into energizer reward and ghost reward
        if isinstance(self.energizer_data, float) or cur_position not in self.energizer_data:
            reward += 0
        else:
            # Potential reward for ghosts
            ifscared1 = self.ghost_status[0] if not isinstance(self.ghost_status[0], float) else 0
            ifscared2 = self.ghost_status[1] if not isinstance(self.ghost_status[1], float) else 0
            if 4 == ifscared1 or 4 == ifscared2:  # ghosts are scared
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
            # Reward for eating the energizer
            if cur_position in self.existing_energizer:
                reward += self.reward_amount[2]
                self.existing_energizer.remove(cur_position)
                self.ghost_status = [4 if each != 3 else 3 for each in self.ghost_status] # change ghost status
        # Ghost reward
        if cur_position in self.ghost_data: #TODO: ghost status
                reward += self.reward_amount[8]
        # Fruit reward 
        if not isinstance(self.fruit_pos, float):
            if np.all(np.array(cur_position) == np.array(self.fruit_pos)):
                reward += self.reward_amount[int(self.reward_type)]
            else:
                # fruit_dist = self.locs_df[(self.locs_df.pos1 == cur_position) & (self.locs_df.pos2 == self.fruit_pos)].dis.values.item()
                fruit_dist = self.locs_df[cur_position][self.fruit_pos]
                if fruit_dist < self.fruit_attractive_thr:
                    reward += self.reward_amount[int(self.reward_type)] * (1 / fruit_dist)
        return reward


    def _computeRisk(self, cur_position):
        # Compute ghost risk when ghosts are normal
        ifscared1 = self.ghost_status[0] if not isinstance(self.ghost_status[0], float) else 0
        ifscared2 = self.ghost_status[1] if not isinstance(self.ghost_status[1], float) else 0
        if 1 == ifscared1 or 1 == ifscared2: # ghosts are normal; use "or" for dealing with dead ghosts
            if 3 == ifscared1:
                # Pacman is eaten
                if cur_position == self.ghost_data[1]:
                    risk = -self.reward_amount[8]
                    self.ghost_data[1] = ()
                    self.ghost_status[1] = 3
                    return risk
                ghost_dist = self.locs_df[cur_position][self.ghost_data[1]]
            elif 3 == ifscared2:
                # Pacman is eaten
                if cur_position == self.ghost_data[0]:
                    risk = -self.reward_amount[8]
                    self.ghost_data[0] = ()
                    self.ghost_status[0] = 3
                    return risk
                ghost_dist = self.locs_df[cur_position][self.ghost_data[0]]
            else:
                # Pacman is eaten
                if cur_position == self.ghost_data[0] or cur_position == self.ghost_data[1]:
                    if cur_position == self.ghost_data[0]:
                        self.ghost_data[0] = ()
                        self.ghost_status[0] = 3
                    else:
                        self.ghost_data[1] = ()
                        self.ghost_status[1] = 3
                    risk = -self.reward_amount[8]
                    return risk
                # Potential risk
                else:
                    ghost_dist = min(
                        self.locs_df[cur_position][self.ghost_data[0]],
                        self.locs_df[cur_position][self.ghost_data[1]]
                    )
            if ghost_dist < self.ghost_repulsive_thr:
                risk = -self.reward_amount[8] * 1 / ghost_dist
            else:
                risk = 0
        # Ghosts are not scared
        else:
            risk = 0
        return risk