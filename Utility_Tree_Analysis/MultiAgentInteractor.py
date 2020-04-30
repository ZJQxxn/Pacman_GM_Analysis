'''
Description:
    Multi-agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 29 2020
'''

import pandas as pd
import numpy as np
from scipy.special import softmax
import anytree
import sys
import warnings

sys.path.append('./')
from PathTreeConstructor import PathTree
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent


class MultiAgentInteractor:

    def __init__(self, agent_weight = [1, 0, 0, 0],
                 global_depth = 15, local_depth = 5,
                 global_ghost_attractive_thr = 34, local_ghost_attractive_thr = 5,
                 global_ghost_repulsive_thr=34, local_ghost_repulsive_thr = 5,
                 global_fruit_attractive_thr = 15, local_fruit_attractive_thr = 5):
        '''
        Initialization for the multi-agent interactor.
        :param global_depth: 
        :param local_depth: 
        :param global_ghost_attractive_thr: 
        :param local_ghost_attractive_thr: 
        :param global_ghost_repulsive_thr: 
        :param local_ghost_repulsive_thr: 
        :param global_fruit_attractive_thr: 
        :param local_fruit_attractive_thr: 
        '''
        # Initialization for game status
        self.cur_pos = None
        self.energizer_data = None
        self.bean_data = None
        self.ghost_data = None
        self.ghost_status = None
        self.reward_type = None
        self.fruit_pos = None
        # Initialization for the weight of each agent
        self.agent_weight = agent_weight
        # Initialization for pre-defined parameters of global agents and local agents
        self.global_depth = global_depth
        self.global_ghost_attractive_thr = global_ghost_attractive_thr
        self.global_ghost_repulsive_thr = global_ghost_repulsive_thr
        self.global_fruit_attractive_thr = global_fruit_attractive_thr
        self.local_depth = local_depth
        self.local_ghost_attractive_thr = local_ghost_attractive_thr
        self.local_ghost_repulsive_thr = local_ghost_repulsive_thr
        self.local_fruit_attractive_thr = local_fruit_attractive_thr
        # Other initializations
        self.dir_list = ['left', 'right', 'up', 'down']
        self.reset_flag = False
        self.last_dir = None  # moving direction for the last time step
        self.loop_count = 0  # the number of crossroads passed for now



    def _oneHot(self, val):
        # Type check
        if val not in self.dir_list:
            raise ValueError("Undefined direction {}!".format(val))
        if not isinstance(val, str):
            raise TypeError("Undefined direction type {}!".format(type(val)))
        # One hot
        onehot_vec = [0, 0, 0, 0]
        onehot_vec[self.dir_list.index(val)] = 1
        return onehot_vec



    def _globalAgent(self):
        # Construct the decision tree for the global agent
        self.global_agent = PathTree(
            self.cur_pos,
            self.energizer_data,
            self.bean_data,
            self.ghost_data,
            self.reward_type,
            self.fruit_pos,
            self.ghost_status,
            depth = self.global_depth,
            ghost_dist_thr = self.global_ghost_attractive_thr,
            fruit_attractive_thr = self.global_fruit_attractive_thr,
            ghost_repulsive_thr = self.global_ghost_repulsive_thr
        )
        # Estimate the moving direction
        root, highest_utility, best_path = self.global_agent.construct()
        return self._oneHot(best_path[0][1])



    def _localAgent(self):
        # Construct the decision tree for the local agent
        self.local_agent = PathTree(
            self.cur_pos,
            self.energizer_data,
            self.bean_data,
            self.ghost_data,
            self.reward_type,
            self.fruit_pos,
            self.ghost_status,
            depth=self.local_depth,
            ghost_dist_thr=self.local_ghost_attractive_thr,
            fruit_attractive_thr=self.local_fruit_attractive_thr,
            ghost_repulsive_thr=self.local_ghost_repulsive_thr
        )
        # Estimate the moving direction
        root, highest_utility, best_path = self.global_agent.construct()
        return self._oneHot(best_path[0][1])



    def _lazyAgent(self):
        self.lazy_agent = LazyAgent(self.cur_pos, self.last_dir, self.loop_count, max_loop = 5) #TODO: set max_loop
        next_dir, not_turn = self.lazy_agent.nextDir()
        if not_turn:
            self.loop_count += 1
        return self._oneHot(next_dir)



    def _randomAgent(self):
        self.random_agent = RandomAgent(self.cur_pos, self.last_dir)
        next_dir = self.random_agent.nextDir()
        return self._oneHot(next_dir)



    def estimateDir(self):
        if not self.reset_flag:
            warnings.warn("The game status is not reset since the last direction estimation! "
                          "Are you sure you want to predict the moving direction in this case?")
        integrate_estimation = np.zeros((4, 4))
        integrate_estimation[:, 0] = self._globalAgent()
        integrate_estimation[:, 1] = self._localAgent()
        integrate_estimation[:, 2] = self._lazyAgent()
        integrate_estimation[:, 3] = self._randomAgent()
        integrate_estimation = integrate_estimation @ self.agent_weight
        integrate_estimation = softmax(integrate_estimation)
        self.reset_flag = True
        # If multiple directions have the highest score, choose the first occurance
        self.last_dir = self.dir_list[np.argmax(integrate_estimation)]
        return integrate_estimation



    def resetStatus(self, cur_pos, energizer_data, bean_data, ghost_data,reward_type,fruit_pos,ghost_status):
        self.cur_pos = cur_pos
        self.energizer_data = energizer_data
        self.bean_data = bean_data
        self.ghost_data = ghost_data
        self.ghost_status = ghost_status
        self.reward_type = reward_type
        self.fruit_pos = fruit_pos
        self.reset_flag = True




if __name__ == '__main__':
    import pickle
    with open("extracted_data/test_data.pkl", 'rb') as file:
        all_data = pickle.load(file)
    for each in all_data.iloc:
        energizer_data = each.energizers.values[0]
        bean_data = each.beans.values[0]
        ghost_data = np.array([each.distance1.values[0], each.distance2.values[0]])
        ghost_status = each[["ifscared1", "ifscared2"]].values
        reward_type = each.Reward.values
        fruit_pos = each.fruitPos.values
