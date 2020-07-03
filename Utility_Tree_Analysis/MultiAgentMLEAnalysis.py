'''
Description:
    MLE parameter estimation for multi-agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    July 1 2020
'''

import pickle
import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
import scipy.optimize

import sys
# from MultiAgentInteractor import MultiAgentInteractor
sys.path.append('./')
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount
from PathTreeConstructor import PathTree
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent


# Global variables
dir_list = ['left', 'right', 'up', 'down']


def oneHot(val):
    '''
    Convert the direction into a one-hot vector.
    :param val: The direction. should be the type ``str''.
    :return: 
    '''
    # Type check
    if val not in dir_list:
        raise ValueError("Undefined direction {}!".format(val))
    if not isinstance(val, str):
        raise TypeError("Undefined direction type {}!".format(type(val)))
    # One-hot
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[dir_list.index(val)] = 1
    return onehot_vec


def negativeLogLikelihood(param, all_data, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    # TODO: check correctness
    agent_weight = [0.7, 0.0, 0.2, 0.1] #TODO: later, also estimate this weight vector
    # Parameters
    #TODO: fix depth for now
    global_depth = 15
    global_ghost_attractive_thr = int(param[0])
    global_fruit_attractive_thr = int(param[1])
    global_ghost_repulsive_thr = int(param[2])
    local_depth = 5
    local_ghost_attractive_thr =int(param[3])
    local_fruit_attractive_thr = int(param[4])
    local_ghost_repulsive_thr = int(param[5])
    # Compute log likelihood
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    # for index in range(num_samples):
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):  # TODO: use only a part of samples for efficiency for now
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos)
        energizer_data = eval(each.energizers)
        bean_data = eval(each.beans)
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values # TODO: check whether same as ``ifscared''
        reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        # Construct agents
        global_agent = PathTree(
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
            depth=global_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr
        )
        local_agent = PathTree(
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
            depth=local_depth,
            ghost_attractive_thr=local_ghost_attractive_thr,
            fruit_attractive_thr=local_fruit_attractive_thr,
            ghost_repulsive_thr=local_ghost_repulsive_thr
        )
        lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir, loop_count, max_loop=5)
        random_agent = RandomAgent(adjacent_data, cur_pos, last_dir, None)
        # Estimation
        agent_estimation = np.zeros((4, 4))
        _, _, global_best_path = global_agent.construct()
        _, _,local_best_path = local_agent.construct()
        lazy_next_dir, not_turn = lazy_agent.nextDir()
        if not_turn:
            loop_count += 1
        random_next_dir = random_agent.nextDir()
        agent_estimation[:, 0] = oneHot(global_best_path[0][1])
        agent_estimation[:, 1] = oneHot(local_best_path[0][1])
        agent_estimation[:, 2] = oneHot(lazy_next_dir)
        agent_estimation[:, 3] = oneHot(random_next_dir)
        dir_prob = agent_estimation @ agent_weight
        best_dir_index = np.argmax(dir_prob)
        last_dir = dir_list[best_dir_index]
        exp_prob = np.exp(dir_prob)
        log_likelihood = dir_prob[best_dir_index] - np.log(np.sum(exp_prob))
        nll += (-log_likelihood)
        estimation_prob_trajectory.append(exp_prob / np.sum(exp_prob))
    print('Finished')
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def MLE(data_filename, map_filename, loc_distance_filename):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    # Load experiment data
    # with open(data_filename, 'rb') as file:
    #     all_data = pickle.load(file)
    with open(data_filename, 'r') as file:
        all_data = pd.read_csv(file)
    print("Number of sanmples : ", all_data.shape[0])
    # Optimization
    useful_num_samples = 1000 # use a part of samples for efficiency
    print("Number of used samples : ", useful_num_samples)
    bounds = [[1, 40], [1, 40], [1, 40], [1, 40], [1, 40], [1, 40]]
    params = np.array([34, 34, 12, 5, 5, 5]) # TODO: same as the initial guess
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    func = lambda parameter: negativeLogLikelihood(parameter, all_data, adjacent_data, locs_df, reward_amount,
                                                   useful_num_samples = useful_num_samples)
    res = scipy.optimize.minimize(
        func,
        x0 = params,
        method = "SLSQP",
        bounds = bounds,
        tol = 1e-6,
        constraints = cons
    )
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = negativeLogLikelihood(res.x, all_data, adjacent_data, locs_df, reward_amount,
                                              useful_num_samples = useful_num_samples, return_trajectory = True)
    true_dir = all_data.pacman_dir.apply(
            lambda x: np.argmax([float(each) for each in x.strip('[]').split(' ')]) if not isinstance(x, float) else -1
        ).values[:useful_num_samples]
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))




if __name__ == '__main__':
    # TODO: data filename; a function for extracting data from different type of file
    # data_filename = "extracted_data/test_data.pkl"
    data_filename = "stimulus_data/global-graze/diary.csv"
    map_filename = "extracted_data/adjacent_map.csv"
    loc_distance_filename = "extracted_data/dij_distance_map.csv"
    MLE(data_filename, map_filename, loc_distance_filename)