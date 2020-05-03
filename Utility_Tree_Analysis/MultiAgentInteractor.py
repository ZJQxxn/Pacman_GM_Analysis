'''
Description:
    Multi-agent interactor. Specifically, we integrate four types of agents:
    (1) Global agent: an agent uses long-distance features to construct a utility-based decision tree and make predictions. 
    (2) Local agent: an agent uses short-distance features to construct a utility-based decision tree and make predictions.
    (3) Lazy agent: an agent chooses to stay the moving direction at crossroads for saving energy. Specifically, in 
                    order to avoid hovering in a sub-area, the agent randomly chooses a direction at the crossroads when
                    the number of passed crossroads surpasses a threshold. 
    (4) Random agent: an agent randomly chooses a direction at crossroads.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 29 2020
'''

import numpy as np
import sys
import warnings
import json

sys.path.append('./')
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount
from PathTreeConstructor import PathTree
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent


class MultiAgentInteractor:

    def __init__(self, config_file):
        '''
        Initialization for the multi-agent interactor.The config_file is a json  file containing all the configurations.
        :param agent_weight: A vector denoting the weight of every agent.
        :param global_depth: The tree depth of global agent. The default is 15.
        :param local_depth: The tree depth of local agent. The default is 5.
        :param global_ghost_attractive_thr: The ghost attractive threshold of global agent. The default is 34.
        :param local_ghost_attractive_thr: The ghost attractive threshold of local agent. The default is 5.
        :param global_ghost_repulsive_thr: The ghost repulsive threshold of global agent. The default is 34.
        :param local_ghost_repulsive_thr: The ghost attractive threshold of local agent. The default is 5.
        :param global_fruit_attractive_thr: The fruit attractive threshold of global agent. The default is 15.
        :param local_fruit_attractive_thr: The fruit attractive threshold of local agent. The default is 5.
        '''
        # Read config files
        with open(config_file) as file:
            config = json.load(file)
        # Initialization for game status
        self.cur_pos = None
        self.energizer_data = None
        self.bean_data = None
        self.ghost_data = None
        self.ghost_status = None
        self.reward_type = None
        self.fruit_pos = None
        # Initialization for the weight of each agent
        self.agent_weight = config['agent_weight']
        # Pre-computed data
        self.adjacent_data = readAdjacentMap(config['map_file'])
        self.locs_df = readLocDistance(config['loc_distance_file'])
        self.reward_amount = readRewardAmount()
        # Initialization for pre-defined parameters of global agents and local agents
        self.global_depth = config['global_depth'] \
            if 'global_depth' in config else 15
        self.global_ghost_attractive_thr = config['global_ghost_attractive_thr']  \
            if 'global_ghost_attractive_thr' in config else 34
        self.global_ghost_repulsive_thr = config['global_ghost_repulsive_thr']  \
            if 'global_ghost_repulsive_thr' in config else 34
        self.global_fruit_attractive_thr = config['global_fruit_attractive_thr']  \
            if 'global_fruit_attractive_thr' in config else 15
        self.local_depth = config['local_depth']  \
            if 'local_depth' in config else 5
        self.local_ghost_attractive_thr = config['local_ghost_attractive_thr']  \
            if 'local_ghost_attractive_thr' in config else 5
        self.local_ghost_repulsive_thr = config['local_ghost_repulsive_thr']  \
            if 'local_ghost_repulsive_thr' in config else 5
        self.local_fruit_attractive_thr = config['local_fruit_attractive_thr']  \
            if 'local_fruit_attractive_thr' in config else 5
        # Set of available directions
        self.dir_list = ['left', 'right', 'up', 'down']
        # Determine whether the game status is reseted after the last estimation. Notice that the game status should be
        # reseted after every estimation though the function ``resetStatus(...)''
        self.reset_flag = False
        # Moving direction of the last time step
        self.last_dir = None
        # The number of crossroads passed wihtout turning for now
        self.loop_count = 0
        print("Finished all the pre-computation and initializations.")


    def _oneHot(self, val):
        '''
        Convert the direction into a one-hot vector.
        :param val: The direction. should be the type ``str''.
        :return: 
        '''
        # Type check
        if val not in self.dir_list:
            raise ValueError("Undefined direction {}!".format(val))
        if not isinstance(val, str):
            raise TypeError("Undefined direction type {}!".format(type(val)))
        # One-hot
        onehot_vec = [0, 0, 0, 0]
        onehot_vec[self.dir_list.index(val)] = 1
        return onehot_vec


    def _globalAgent(self):
        '''
        Use the global agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of global agent.
        '''
        # Construct the decision tree for the global agent
        self.global_agent = PathTree(
            self.adjacent_data,
            self.locs_df,
            self.reward_amount,
            self.cur_pos,
            self.energizer_data,
            self.bean_data,
            self.ghost_data,
            self.reward_type,
            self.fruit_pos,
            self.ghost_status,
            depth = self.global_depth,
            ghost_attractive_thr = self.global_ghost_attractive_thr,
            fruit_attractive_thr = self.global_fruit_attractive_thr,
            ghost_repulsive_thr = self.global_ghost_repulsive_thr
        )
        # Estimate the moving direction
        root, highest_utility, best_path = self.global_agent.construct()
        return self._oneHot(best_path[0][1])


    def _localAgent(self):
        '''
        Use the local agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of local agent.
        '''
        # Construct the decision tree for the local agent
        self.local_agent = PathTree(
            self.adjacent_data,
            self.locs_df,
            self.reward_amount,
            self.cur_pos,
            self.energizer_data,
            self.bean_data,
            self.ghost_data,
            self.reward_type,
            self.fruit_pos,
            self.ghost_status,
            depth=self.local_depth,
            ghost_attractive_thr=self.local_ghost_attractive_thr,
            fruit_attractive_thr=self.local_fruit_attractive_thr,
            ghost_repulsive_thr=self.local_ghost_repulsive_thr
        )
        # Estimate the moving direction
        root, highest_utility, best_path = self.global_agent.construct()
        return self._oneHot(best_path[0][1])


    def _lazyAgent(self):
        '''
        Use the lazy agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of lazy agent.
        '''
        self.lazy_agent = LazyAgent(self.adjacent_data, self.cur_pos, self.last_dir, self.loop_count, max_loop = 5)
        next_dir, not_turn = self.lazy_agent.nextDir()
        if not_turn:
            self.loop_count += 1
        return self._oneHot(next_dir)


    def _randomAgent(self):
        '''
        Use the random agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of random agent.
        '''
        self.random_agent = RandomAgent(self.adjacent_data, self.cur_pos, self.last_dir)
        next_dir = self.random_agent.nextDir()
        return self._oneHot(next_dir)


    def estimateDir(self):
        '''
        Integrate the estimation of all the agents and obtain the final direction estimation.
        :return: A vector with a length of 4 denoting the selecting probability of all the four directions correspondingly. 
                ([left, right, up, down])
        '''
        # If the game status has not been reset since the last estimation, warn the user
        if not self.reset_flag:
            warnings.warn("The game status is not reset since the last direction estimation! "
                          "Are you sure you want to predict the moving direction in this case?")
        # Obtain estimations of all the agents
        integrate_estimation = np.zeros((4, 4))
        integrate_estimation[:, 0] = self._globalAgent()
        integrate_estimation[:, 1] = self._localAgent()
        integrate_estimation[:, 2] = self._lazyAgent()
        integrate_estimation[:, 3] = self._randomAgent()
        integrate_estimation = integrate_estimation @ self.agent_weight
        self.reset_flag = False
        # If multiple directions have the highest score, choose the first occurrence
        self.last_dir = self.dir_list[np.argmax(integrate_estimation)]
        return integrate_estimation


    def resetStatus(self, cur_pos, energizer_data, bean_data, ghost_data,reward_type,fruit_pos,ghost_status):
        '''
        Reset the game stauts and Pacman status.
        :param cur_pos: The current position of Pacman with the shape of a 2-tuple.
        :param energizer_data: A list contains positions of all the energizers. Each list element is a 2-tuple. 
                               If no energizer exists, just pass np.nan to the function.  
        :param bean_data: A list contains positions of all the beans. Each list element is a 2-tuple. 
        :param ghost_data: A list with a length of 2 denoting the distance between Pacman and two ghosts. 
                           [ghost 1 distance, ghost 2 distance]
        :param reward_type: The integer expression of the type of fruit existing in the game. If no fruit exists, just pass 
                             np.nan to the function. The fruit type and its corresponding integer expression is:
                                3 -- cherry,
                                4 -- strawberry,
                                5 -- orange,
                                6 -- apple,
                                7 -- melon.
        :param fruit_pos: A 2-tuple denoting the position of the fruit. 
        :param ghost_status: A list with a length of 2 denoting the status of two ghosts. 
                             [ghost 1 status, ghost 2 status]. The ghost status and its corresponding meaning is:
                                1 -- chasing Pacman, 
                                2 -- going corner, 
                                3 -- dead ghosts (include ghosts are being eaten), 
                                4 -- scared ghosts, 
                                5 -- flash scared ghosts.
                             Currently, only 1 (normal ghosts) and 4 (scared ghosts) are useful. 
        :return: VOID
        '''
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

    multiagent = MultiAgentInteractor("config.json")
    for index in range(15):
        each = all_data.iloc[index]
        cur_pos = each.pacmanPos
        energizer_data = each.energizers
        bean_data = each.beans
        ghost_data = np.array([each.distance1, each.distance2])
        ghost_status = each[["ifscared1", "ifscared2"]].values
        reward_type = int(each.Reward)
        fruit_pos = each.fruitPos
        multiagent.resetStatus(cur_pos, energizer_data, bean_data, ghost_data,reward_type,fruit_pos,ghost_status)
        dir_prob = multiagent.estimateDir()
        cur_dir = multiagent.dir_list[np.argmax(dir_prob)]
        if "left" == cur_dir:
            next_pos = [cur_pos[0] - 1, cur_pos[1]]
        elif "right" == cur_dir:
                next_pos = [cur_pos[0] + 1, cur_pos[1]]
        elif "up" == cur_dir:
            next_pos = [cur_pos[0], cur_pos[1] - 1]
        else:
            next_pos = [cur_pos[0], cur_pos[1] + 1]
        print(cur_dir, next_pos, dir_prob)

