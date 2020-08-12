'''
Description:
    Construct a utility tree along the estimated path.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 21 2020
'''

#TODO: deprecated for multi-agents; only used for analyzing current data


import pandas as pd
import numpy as np
import anytree
from anytree.exporter import DotExporter
from collections import deque
import sys

sys.path.append('./')
from TreeAnalysisUtils import adjacent_data, locs_df, reward_amount
from TreeAnalysisUtils import unitStepFunc



class PathTree:

    def __init__(self, root, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status, depth = 10,
                 ghost_attractive_thr = 34,ghost_repulsive_thr = 10,  fruit_attractive_thr = 10):
        '''
        Initialization.
        :param root: The tree root (i.e., the current position with the shape of 2-tuple).
        :param path_data: 
        :param depth: The deepest 
        :param threshold: 
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
        # All the true data for a certain path
        self.energizer_data = energizer_data
        self.bean_data = bean_data
        self.ghost_data = ghost_data
        self.ghost_status = ghost_status
        # Fruit data
        self.reward_type = reward_type
        self.fruit_pos = fruit_pos
        # Pre-defined thresholds for utility computation
        self.ghost_attractive_thr = ghost_attractive_thr
        self.fruit_attractive_thr = fruit_attractive_thr
        self.ghost_repulsive_thr = ghost_repulsive_thr


    def construct(self):
        '''
        Construct the utility tree.
        :return: The tree root node (anytree.Node).
        '''
        # construct the first layer firstly (depth = 1)
        self._attachNode(cur_depth = 1) # attach all children of the root (depth = 1)
        self.node_queue.append(None) # the end of layer with depth = 1
        self.node_queue.popleft()
        self.current_node = self.node_queue.popleft()
        cur_depth = 2
        # construct the other parts
        while cur_depth <= self.depth:
            while None != self.current_node :
                self._attachNode(cur_depth = cur_depth)
                self.current_node = self.node_queue.popleft()
            self.node_queue.append(None)
            if 0 == len(self.node_queue):
                break
            self.current_node = self.node_queue.popleft()
            cur_depth += 1
        # Find the best path with the highest utility
        best_leaf = self.root
        for leaf in self.root.leaves:
            if leaf.cumulative_utility > best_leaf.cumulative_utility:
                best_leaf = leaf
        highest_utility = best_leaf.cumulative_utility
        best_path = best_leaf.ancestors
        best_path = [(each.name, each.dir_from_parent) for each in best_path[1:]]
        return self.root, highest_utility, best_path


    def _attachNode(self, cur_depth = 0):
        # Find adjacent positions and the corresponding moving directions for the current node
        tmp_data = adjacent_data[adjacent_data.pos == self.current_node.name]
        for each in tmp_data.columns.values[-4:]:
            # do not walk on the wall or wolk out of boundary
            # do not turn back
            if None == self.current_node.parent and isinstance(tmp_data[each].values.item(), float):
                continue
            elif None != self.current_node.parent and \
                    (isinstance(tmp_data[each].values.item(), float)
                     or tmp_data[each].values.item() == self.current_node.parent.name):
                continue
            else:
                # Compute utility
                cur_pos = tmp_data[each].values.item()
                cur_reward = self._computeReward(cur_pos, cur_depth)
                cur_risk = self._computeRisk(cur_depth)
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
                # # Find the path with the biggest utility value
                # if not self.best_flag:
                #     self.best_path.append(new_node)
                #     self.best_flag = True
                # else:
                #     if new_node.cumulative_utility > self.best_path[-1].cumulative_utility:
                #         self.best_path[-1] = new_node


    def _computeReward(self, cur_position, cur_depth):
        reward = 0
        # Bean reward
        if cur_position in self.bean_data:
            reward += reward_amount[1]
            self.bean_data.remove(cur_position)
        else:
            reward += 0
        # Energizer reward
        if isinstance(self.energizer_data, float) or cur_position not in self.energizer_data:
            reward += 0
        else:
            reward += reward_amount[2] * unitStepFunc(self.ghost_attractive_thr - min(self.ghost_data))
        # Ghost reward
        ifscared1 = self.ghost_status[cur_depth, 0] if not isinstance(self.ghost_status[cur_depth, 0], float) else 0
        ifscared2 = self.ghost_status[cur_depth, 1] if not isinstance(self.ghost_status[cur_depth, 1], float) else 0
        if 4 == ifscared1 or 4 == ifscared2: # ghosts are scared
            ghost_dist = min(self.ghost_data)
            if ghost_dist < self.ghost_attractive_thr:
                reward += 8 * (1 / ghost_dist)
        # Fruit reward
        if not isinstance(self.fruit_pos[cur_depth], float):
            if np.all(np.array(cur_position) == np.array(self.fruit_pos[cur_depth])):
                reward += reward_amount[int(self.reward_type[cur_depth])]
            else:
                fruit_dist = locs_df[(locs_df.pos1 == cur_position) & (locs_df.pos2 == self.fruit_pos[cur_depth])].dis.values.item()
                if fruit_dist < self.fruit_attractive_thr:
                    reward += reward_amount[int(self.reward_type[cur_depth])] * (1 / fruit_dist)
        return reward


    def _computeRisk(self, cur_depth):
        # Compute ghost risk when ghosts are normal
        ifscared1 = self.ghost_status[cur_depth, 0] if not isinstance(self.ghost_status[cur_depth, 0], float) else 0
        ifscared2 = self.ghost_status[cur_depth, 1] if not isinstance(self.ghost_status[cur_depth, 1], float) else 0
        if 1 == ifscared1 and 1 == ifscared2: # ghosts are normal
            ghost_dist = min(self.ghost_data)
            # TODO: difficult to directly use repulsive
            repulsive = - (1 / ghost_dist ** 2) * \
                            (1 / self.ghost_repulsive_thr - 1 / min(self.ghost_data))
            if ghost_dist < self.ghost_repulsive_thr:
                risk = -8 * 1 / ghost_dist
            else:
                risk = 0
        else:
            risk = 0
        return risk



if __name__ == '__main__':
    tree = PathTree(root = (10, 24), path_data = [], depth = 10)
    tree.construct()
    print(anytree.RenderTree(tree.root))
    print(tree.best_path[-1])