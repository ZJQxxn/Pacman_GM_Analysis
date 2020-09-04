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
from MultiAgent_Analysis.TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, makeChoice
from MultiAgent_Analysis.PlannedHuntingAgent import PlannedHuntingAgent
from MultiAgent_Analysis.PathTreeAgent import PathTree
from MultiAgent_Analysis.SuicideAgent import SuicideAgent

import time

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
        self.adjacent_path = readAdjacentPath(config['loc_distance_file'])
        self.reward_amount = readRewardAmount()
        # Randomness and laziness
        self.randomness_coeff = 1.0
        self.laziness_coeff = 1.0
        # Configuration (for global agent)
        self.global_depth = 15
        self.ignore_depth = 5
        self.global_ghost_attractive_thr = 34
        self.global_fruit_attractive_thr = 34
        self.global_ghost_repulsive_thr = 34
        # Configuration (for local agent)
        self.local_depth = 5
        self.local_ghost_attractive_thr = 5
        self.local_fruit_attractive_thr = 5
        self.local_ghost_repulsive_thr = 5
        # Configuration (for optimistic agent)
        self.optimistic_depth = 5
        self.optimistic_ghost_attractive_thr = 5
        self.optimistic_fruit_attractive_thr = 5
        self.optimistic_ghost_repulsive_thr = 5
        # Configuration (for pessimistic agent)
        self.pessimistic_depth = 5
        self.pessimistic_ghost_attractive_thr = 5
        self.pessimistic_fruit_attractive_thr = 5
        self.pessimistic_ghost_repulsive_thr = 5
        # Configuration (for suicide agent)
        self.suicide_depth = 5
        self.suicide_ghost_attractive_thr = 5
        self.suicide_fruit_attractive_thr = 5
        self.suicide_ghost_repulsive_thr = 5
        # Set of available directions
        self.dir_list = ['left', 'right', 'up', 'down']
        # Determine whether the game status is reseted after the last estimation. Notice that the game status should be
        # reseted after every estimation though the function ``resetStatus(...)''
        self.reset_flag = False
        # Moving direction of the last time step
        self.last_dir = None
        # random seed for the random agent
        self.random_seed = config["random_seed"]
        # All the agents
        self.global_agent = None
        self.local_agent = None
        self.optimistic_agent = None
        self.pessimistic_agent = None
        self.suicide_agent = None
        self.planned_hunting_agent = None
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
            self.last_dir,
            depth = self.global_depth,
            ignore_depth = self.ignore_depth,
            ghost_attractive_thr = self.global_ghost_attractive_thr,
            fruit_attractive_thr = self.global_fruit_attractive_thr,
            ghost_repulsive_thr = self.global_ghost_repulsive_thr,
            randomness_coeff = self.randomness_coeff,
            laziness_coeff = self.laziness_coeff
        )
        # Estimate the moving direction
        global_result = self.global_agent.nextDir(return_Q = True)
        global_Q =global_result[1]
        return global_Q


    def _localAgent(self):
        '''
        Use the global agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of global agent.
        '''
        # Construct the decision tree for the global agent
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
            self.last_dir,
            depth = self.local_depth,
            ghost_attractive_thr = self.local_ghost_attractive_thr,
            fruit_attractive_thr = self.local_fruit_attractive_thr,
            ghost_repulsive_thr = self.local_ghost_repulsive_thr,
            randomness_coeff = self.randomness_coeff,
            laziness_coeff = self.laziness_coeff
        )
        # Estimate the moving direction
        local_result = self.local_agent.nextDir(return_Q = True)
        local_Q =local_result[1]
        return local_Q


    def _optimisticAgent(self):
        '''
        Use the global agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of global agent.
        '''
        # Construct the decision tree for the global agent
        self.optimistic_agent = PathTree(
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
            self.last_dir,
            depth = self.optimistic_depth,
            ghost_attractive_thr = self.optimistic_ghost_attractive_thr,
            fruit_attractive_thr = self.optimistic_fruit_attractive_thr,
            ghost_repulsive_thr = self.optimistic_ghost_repulsive_thr,
            reward_coeff = 1.0,
            risk_coeff = 0.0,
            randomness_coeff = self.randomness_coeff,
            laziness_coeff = self.laziness_coeff
        )
        # Estimate the moving direction
        optimistic_result = self.optimistic_agent.nextDir(return_Q = True)
        optimistic_Q = optimistic_result[1]
        return optimistic_Q


    def _pessimisticAgent(self):
        '''
        Use the global agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of global agent.
        '''
        # Construct the decision tree for the global agent
        self.pessimistic_agent = PathTree(
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
            self.last_dir,
            depth = self.pessimistic_depth,
            ghost_attractive_thr = self.pessimistic_ghost_attractive_thr,
            fruit_attractive_thr = self.pessimistic_fruit_attractive_thr,
            ghost_repulsive_thr = self.pessimistic_ghost_repulsive_thr,
            reward_coeff = 0.0,
            risk_coeff = 1.0,
            randomness_coeff = self.randomness_coeff,
            laziness_coeff = self.laziness_coeff
        )
        # Estimate the moving direction
        pessimistic_result = self.pessimistic_agent.nextDir(return_Q = True)
        pessimistic_Q = pessimistic_result[1]
        return pessimistic_Q


    def _suicideAgent(self):
        '''
        Use the global agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of global agent.
        '''
        # Construct the decision tree for the global agent
        self.suicide_agent = SuicideAgent(
            self.adjacent_data,
            self.adjacent_path,
            self.locs_df,
            self.reward_amount,
            self.cur_pos,
            self.energizer_data,
            self.bean_data,
            self.ghost_data,
            self.reward_type,
            self.fruit_pos,
            self.ghost_status,
            self.last_dir,
            depth=self.suicide_depth,
            ghost_attractive_thr = self.suicide_ghost_attractive_thr,
            ghost_repulsive_thr =self.suicide_fruit_attractive_thr,
            fruit_attractive_thr = self.suicide_ghost_repulsive_thr,
            randomness_coeff = self.randomness_coeff,
            laziness_coeff = self.laziness_coeff
        )
        # Estimate the moving direction
        suicide_result = self.suicide_agent.nextDir(return_Q = True)
        suicide_Q = suicide_result[1]
        return suicide_Q


    def _plannedHuntingAgent(self):
        '''
        Use the global agent to predict the moving direction given game status of the current time step. 
        :return: The one-hot vector denoting the direction estimation of global agent.
        '''
        # Construct the decision tree for the global agent
        self.planned_hunting_agent = PlannedHuntingAgent(
            self.adjacent_data,
            self.adjacent_path,
            self.locs_df,
            self.reward_amount,
            self.cur_pos,
            self.energizer_data,
            self.ghost_data,
            self.ghost_status,
            self.last_dir,
            randomness_coeff = self.randomness_coeff,
            laziness_coeff = self.laziness_coeff
        )
        # Estimate the moving direction
        planned_hunting_result = self.planned_hunting_agent.nextDir(return_Q = True)
        planned_hunting_Q = planned_hunting_result[1]
        return planned_hunting_Q


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
        agent_estimation = np.zeros((4, 6))
        agent_estimation[:, 0] = self._globalAgent()
        agent_estimation[:, 1] = self._localAgent()
        agent_estimation[:, 2] = self._optimisticAgent()
        agent_estimation[:, 3] = self._pessimisticAgent()
        agent_estimation[:, 4] = self._suicideAgent()
        agent_estimation[:, 5] = self._plannedHuntingAgent()
        integrate_estimation = agent_estimation @ self.agent_weight
        self.reset_flag = False
        # If multiple directions have the highest score, choose the first occurrence
        self.last_dir = self.dir_list[makeChoice(integrate_estimation)]
        return integrate_estimation, agent_estimation


    def resetStatus(self, cur_pos, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status):
        '''
        Reset the game stauts and Pacman status.
        :param cur_pos: The current position of Pacman with the shape of a 2-tuple.
        :param energizer_data: A list contains positions of all the energizers. Each list element is a 2-tuple. 
                               If no energizer exists, just pass np.nan to the function.  
        :param bean_data: A list contains positions of all the beans. Each list element is a 2-tuple. 
        :param ghost_data: A list with a length of 2 denoting the positions of two ghosts. Each position should be a 2-tuple.
                           [ghost 1 position, ghost 2 position]
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
        # Reset all the agents
        self.global_agent = None
        self.local_agent = None
        self.optimistic_agent = None
        self.pessimistic_agent = None
        self.suicide_agent = None
        self.planned_hunting_agent = None



    def resetLastDir(self):
        '''
        Reset the last direction to None.
        :return: VOID
        '''
        self.last_dir = None



if __name__ == '__main__':
    import pickle
    # with open("extracted_data/test_data.pkl", 'rb') as file:
    #     all_data = pickle.load(file)

    multiagent = MultiAgentInteractor("config.json")

    cur_pos = (22, 27)
    ghost_data = [(18, 27), (10, 21)]
    ghost_status = np.array([1, 1])
    energizer_data = [(13, 9), (9, 24), (22, 26)]
    bean_data = [(5, 5), (9, 5), (16, 5), (21, 5), (26, 5), (2, 6), (27, 6), (7, 7), (2, 8), (11, 9), (22, 10),
                 (4, 12), (22, 12), (16, 13), (15, 15), (18, 15), (22, 15), (15, 16), (7, 19), (22, 20), (22, 24),
                 (25, 24), (2, 29), (22, 29), (17, 30), (23, 30), (25, 30), (5, 33), (11, 33)]
    reward_type = 4
    fruit_pos = (2, 7)
    last_dir = "right"

    multiagent.resetStatus(cur_pos, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status)
    multiagent.last_dir = last_dir
    dir_prob, agent_Q = multiagent.estimateDir()
    print("Q : ", agent_Q)
    # for index in range(15):
    #     each = all_data.iloc[index]
    #     cur_pos = each.pacmanPos
    #     energizer_data = each.energizers
    #     bean_data = each.beans
    #     ghost_data = np.array([each.ghost1Pos, each.ghost2Pos]) #TODO: revise to ghost position
    #     ghost_status = each[["ifscared1", "ifscared2"]].values
    #     reward_type = int(each.Reward)
    #     fruit_pos = each.fruitPos
    #     multiagent.resetStatus(cur_pos, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status)
    #     dir_prob, _ = multiagent.estimateDir()
    #     cur_dir = multiagent.dir_list[makeChoice(dir_prob)]
    #     if "left" == cur_dir:
    #         next_pos = [cur_pos[0] - 1, cur_pos[1]]
    #     elif "right" == cur_dir:
    #             next_pos = [cur_pos[0] + 1, cur_pos[1]]
    #     elif "up" == cur_dir:
    #         next_pos = [cur_pos[0], cur_pos[1] - 1]
    #     else:
    #         next_pos = [cur_pos[0], cur_pos[1] + 1]
    #     print(cur_dir, next_pos, dir_prob)

