'''
Description:
    Construct a utility tree along the estimated path.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 21 2020
'''

import pandas as pd
import numpy as np
import anytree
from anytree.exporter import DotExporter
from collections import deque
import sys

sys.path.append('./')
from TreeAnalysisUtils import adjacent_data, all_data, locs_df
from TreeAnalysisUtils import unitStepFunc



class PathTree:

    def __init__(self, root, path_data, depth = 10, energizer_tradeoff = 0.5,
                 weight = {"bean": 1},
                 threshold = {"ghost_dist_thr": 34, "fruit_attractive_thr": 10, "ghost_repulsive_thr": 10}):
        # TODO: path_data? not all the data is required by the tree
        # The root node is the path starting point.
        # Other tree nodes should contain:
        #   (1) location
        #   (2) the direction from its to parent to itself
        #   (3) utility of this node (reward and risk are separated)
        #   (4) the cumulative utility so far (reward and risk are separated)
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
        self.path_data = path_data
        self.energizer_data = path_data.energizers.values[0]
        self.bean_data = path_data.beans.values[0]
        self.ghost_data = np.array([path_data.distance1.values[0], path_data.distance2.values[0]])
        # TODO: For now, take the true starting distance as the ghost distance
        self.fruit_data = list(set(path_data["next_eat_rwd"].dropna().values)) # TODO: no fruit position now
        # A list of moving directions of the path with the highest utility
        self.best_path = [self.root]
        self.best_flag = False
        # Pre-defined weights and thresholds for utility computation
        self.weight = weight
        self.energizer_tradeoff = energizer_tradeoff
        self.threshold = threshold


    def construct(self):
        # Construct a tree
        # construct the layer 1 first (depth = 1)
        self._attachNode(depth = 1) # children of the root (depth = 1)
        self.best_flag = False
        self.node_queue.append(None) # the end of layer with depth = 1
        self.node_queue.popleft()
        self.current_node = self.node_queue.popleft()
        cur_depth = 2
        # construct the other parts
        while cur_depth <= self.depth:
            while None != self.current_node :
                self._attachNode(depth = cur_depth)
                self.current_node = self.node_queue.popleft()
            self.node_queue.append(None)
            self.best_flag = False
            #
            if 0 == len(self.node_queue):
                break
            self.current_node = self.node_queue.popleft()
            cur_depth += 1


    def _attachNode(self, depth = 0):
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
                cur_reward = self._computeReward(cur_pos)
                cur_risk = self._computeRisk(depth)
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
                # Find the path with the biggest utility value
                if not self.best_flag:
                    self.best_path.append(new_node)
                    self.best_flag = True
                else:
                    if new_node.cumulative_utility > self.best_path[-1].cumulative_utility:
                        self.best_path[-1] = new_node



    def _computeReward(self, cur_position):
        reward = 0
        # Bean reward
        if cur_position in self.bean_data:
            reward += 1
            self.bean_data.remove(cur_position)
        else:
            reward += 0
        # Energizer reward
        if isinstance(self.energizer_data, float) or cur_position not in self.energizer_data:
            reward += 0
        else:
            reward += (self.energizer_tradeoff * unitStepFunc(min(self.ghost_data) - self.threshold['ghost_dist_thr'])
                       + (1 - self.energizer_tradeoff) * unitStepFunc(self.threshold['ghost_dist_thr'] - min(self.ghost_data)))
        # Ghost reward
        #TODO: if ghosts are scared, get 5 score if capturing a ghost
        # Fruit reward
        # fruit_dist = [
        #     locs_df[(locs_df.pos1 == cur_position) & (locs_df.pos2 == each)].dis.values.item()
        #     for each in self.fruit_data
        # ]
        # min_fruit_dist = min(fruit_dist)
        # # TODO: can get the position of different type of fruit?
        # attractive_force = 1 if min_fruit_dist <= self.threshold['fruit_attractive_thr'] else 1
        # reward += attractive_force
        return reward


    def _computeRisk(self, depth):
        # Ghost risk when ghosts are scared
        # TODO: how to get the current ghost position and ghost status?
        ifscared1 = self.path_data.iloc[depth].ifscared1.values.item() if not isinstance(self.path_data.iloc[depth].ifscared1, float) else 0
        ifscared2 = self.path_data.iloc[depth].ifscared2.values.item() if not isinstance(self.path_data.iloc[depth].ifscared2, float) else 0
        if 1 == ifscared1 and 1 == ifscared2:
            ghost_dist = min(self.ghost_data )
        elif 1 == ifscared1:
            ghost_dist = self.ghost_data[0]
        else:
            ghost_dist = self.ghost_data[1]
        repulsive_force = - (1 / ghost_dist ** 2) * \
                        (1 / self.threshold['ghost_repulsive_thr'] - 1 / min(self.ghost_data))
        return repulsive_force



if __name__ == '__main__':
    tree = PathTree(root = (10, 24), path_data = [], depth = 5)
    tree.construct()
    print(anytree.RenderTree(tree.root))
    print(tree.best_path[-1])