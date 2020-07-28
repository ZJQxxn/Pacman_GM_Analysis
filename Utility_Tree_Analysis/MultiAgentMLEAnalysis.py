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
from sklearn.model_selection import train_test_split
import itertools

import sys
# from MultiAgentInteractor import MultiAgentInteractor
sys.path.append('./')
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from PathTreeConstructor import PathTree, OptimisticAgent, PessimisticAgent
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent
from SuicideAgent import SuicideAgent


# ===========================================================
#               UTILITY FUNCTIONS
# ===========================================================

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


# ===========================================================
#             MAXIMUM LIKELIHOOD ESTIMATION
# ===========================================================
def negativeLogLikelihood(param, all_data, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    '''
    Compute the negative log likelihood. 
    :param param: Parameters.
    :param all_data: All the experimental data (pd.DataFrame).
    :param adjacent_data: Adjacent data (pd.DataFrame).
    :param locs_df: Dij distance map data.
    :param reward_amount: Reward value (dict). 
    :param useful_num_samples: Number of samples used in the computation.
    :param return_trajectory: Whether return the estimated probability for each sample.
    :return: 
    '''
    # Parameters
    global_depth = 5
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    local_depth = 15
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    agent_weight = [param[0], param[1], param[2], param[3]]
    # Compute log likelihood
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):
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
    # print('Finished')
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def MLE(data_filename, map_filename, loc_distance_filename, useful_num_samples = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    # Load experiment data
    with open(data_filename, 'r') as file:
        all_data = pd.read_csv(file)
    true_prob = all_data.pacman_dir
    if "[" in true_prob.values[0]:  # If pacman_dir is a vector
        overall_dir = []
        for index in range(all_data.pacman_dir.values.shape[0]):
            each = all_data.pacman_dir.values[index]
            each = each.strip('[]').split(' ')
            while '' in each:  # For the weird case that '' might exist in the split list
                each.remove('')
            overall_dir.append([float(e) for e in each])
        overall_dir = np.array(overall_dir)
        true_prob = overall_dir
    else:  # If pacman_dir is the name of directions
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    print("Number of samples : ", all_data.shape[0])
    # Optimization
    if useful_num_samples is None:
        useful_num_samples = all_data.shape[0]
    print("Number of used samples : ", useful_num_samples)
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1})
    func = lambda parameter: negativeLogLikelihood(parameter, all_data, adjacent_data, locs_df, reward_amount,
                                                   useful_num_samples=useful_num_samples)
    is_success = False
    retry_num = 0
    while not is_success and retry_num < 10:
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,
            tol=1e-8,
            constraints=cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = negativeLogLikelihood(res.x, all_data, adjacent_data, locs_df, reward_amount,
                                              useful_num_samples=useful_num_samples, return_trajectory=True)
    true_dir = []
    for index in range(all_data.pacman_dir.values.shape[0]):
        each = all_data.pacman_dir.values[index]
        each = each.strip('[]').split(' ')
        while '' in each:  # For the weird case that '' might exist in the split list
            each.remove('')
        true_dir.append(np.argmax([float(e) for e in each]))
    true_dir = np.array(true_dir)
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))


# ===========================================================
#               MINIMUM ERROR ESTIMATION
# ===========================================================
def estimationError(param, all_data, true_prob, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    '''
    Compute the estimation error with global/local/lazy/random agents.
    :param param: Parameters.
    :param all_data: All the experimental data (pd.DataFrame).
    :param: true_prob: True probability of directions.
    :param adjacent_data: Adjacent data (pd.DataFrame).
    :param locs_df: Dij distance map data.
    :param reward_amount: Reward value (dict). 
    :param useful_num_samples: Number of samples used in the computation.
    :param return_trajectory: Whether return the estimated probability for each sample.
    :return: 
    '''
    # Parameters
    global_depth = 5
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    local_depth = 15
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    agent_weight = [param[0], param[1], param[2], param[3]]
    # Compute estimation error
    nll = 0  # estimation error
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)]) \
            if "ghost1_pos" in all_data.columns.values or "ghost2_pos" in all_data.columns.values \
            else np.array([each.ghost1Pos, each.ghost2Pos])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared1])
        if "fruit_type" in all_data.columns.values:
            reward_type = int(each.fruit_type)  if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
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
        error = np.linalg.norm(dir_prob - true_prob.values[index]
                               if isinstance(true_prob, pd.DataFrame) or isinstance(true_prob, pd.Series)
                               else dir_prob - true_prob[index])
        nll += error
        estimation_prob_trajectory.append(dir_prob)
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def estimationErrorOptimism(param, all_data, true_prob, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    '''
    Compute the estimation error with optimistic/pessimistic/lazy/random agents.
    :param param: Parameters.
    :param all_data: All the experimental data (pd.DataFrame).
    :param: true_prob: True probability of directions.
    :param adjacent_data: Adjacent data (pd.DataFrame).
    :param locs_df: Dij distance map data.
    :param reward_amount: Reward value (dict). 
    :param useful_num_samples: Number of samples used in the computation.
    :param return_trajectory: Whether return the estimated probability for each sample.
    :return: 
    '''
    # Parameters
    depth = 10 #TODO:
    ghost_attractive_thr = 34
    fruit_attractive_thr = 34
    ghost_repulsive_thr = 12
    agent_weight = [param[0], param[1], param[2], param[3]]
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    # for index in range(num_samples):
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        # TODO: rename the columns first before this function
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)]) \
            if "ghost1_pos" in all_data.columns.values or "ghost2_pos" in all_data.columns.values \
            else np.array([each.ghost1Pos, each.ghost2Pos])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared1])
        if "fruit_type" in all_data.columns.values:
            reward_type = int(each.fruit_type)  if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
        # Construct agents
        optimistic_agent = OptimisticAgent(
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
            depth=depth,
            ghost_attractive_thr=ghost_attractive_thr,
            fruit_attractive_thr=fruit_attractive_thr,
            ghost_repulsive_thr=ghost_repulsive_thr
        )
        pessimistic_agent = PessimisticAgent(
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
            depth=depth,
            ghost_attractive_thr=ghost_attractive_thr,
            fruit_attractive_thr=fruit_attractive_thr,
            ghost_repulsive_thr=ghost_repulsive_thr
        )
        lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir, loop_count, max_loop=5)
        random_agent = RandomAgent(adjacent_data, cur_pos, last_dir, None)
        # Estimation
        agent_estimation = np.zeros((4, 4))
        _, _, optimistic_best_path = optimistic_agent.construct()
        _, _,pessimistic_best_path = pessimistic_agent.construct()
        lazy_next_dir, not_turn = lazy_agent.nextDir()
        if not_turn:
            loop_count += 1
        random_next_dir = random_agent.nextDir()
        agent_estimation[:, 0] = oneHot(optimistic_best_path[0][1])
        agent_estimation[:, 1] = oneHot(pessimistic_best_path[0][1])
        agent_estimation[:, 2] = oneHot(lazy_next_dir)
        agent_estimation[:, 3] = oneHot(random_next_dir)
        dir_prob = agent_estimation @ agent_weight
        error = np.linalg.norm(dir_prob - true_prob.values[index]
                               if isinstance(true_prob, pd.DataFrame) or isinstance(true_prob, pd.Series)
                               else dir_prob - true_prob[index])
        nll += error
        estimation_prob_trajectory.append(dir_prob)
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def estimationErrorAll(param, all_data, true_prob, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    '''
    Compute the estimation error with optimistic/pessimistic/global/lazy agents.
    :param param: Parameters.
    :param all_data: All the experimental data (pd.DataFrame).
    :param: true_prob: True probability of directions.
    :param adjacent_data: Adjacent data (pd.DataFrame).
    :param locs_df: Dij distance map data.
    :param reward_amount: Reward value (dict). 
    :param useful_num_samples: Number of samples used in the computation.
    :param return_trajectory: Whether return the estimated probability for each sample.
    :return: 
    '''
    # Parameters
    optimisim_depth = 10
    optimisim_ghost_attractive_thr = 34
    optimisim_fruit_attractive_thr = 34
    optimisim_ghost_repulsive_thr = 12
    global_depth = 5
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    local_depth = 15
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    agent_weight = [param[0], param[1], param[2], param[3]]
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    # for index in range(num_samples):
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        # TODO: rename the columns first before this function
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)]) \
            if "ghost1_pos" in all_data.columns.values or "ghost2_pos" in all_data.columns.values \
            else np.array([each.ghost1Pos, each.ghost2Pos])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared1])
        if "fruit_type" in all_data.columns.values:
            reward_type = int(each.fruit_type)  if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
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
        optimistic_agent = OptimisticAgent(
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
            depth=optimisim_depth,
            ghost_attractive_thr=optimisim_ghost_attractive_thr,
            fruit_attractive_thr=optimisim_fruit_attractive_thr,
            ghost_repulsive_thr=optimisim_ghost_repulsive_thr
        )
        pessimistic_agent = PessimisticAgent(
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
            depth=optimisim_depth,
            ghost_attractive_thr=optimisim_ghost_attractive_thr,
            fruit_attractive_thr=optimisim_fruit_attractive_thr,
            ghost_repulsive_thr=optimisim_ghost_repulsive_thr
        )
        # Estimation
        agent_estimation = np.zeros((4, 4))
        _, _, global_best_path = global_agent.construct()
        _, _, local_best_path = local_agent.construct()
        _, _, optimistic_best_path = optimistic_agent.construct()
        _, _,pessimistic_best_path = pessimistic_agent.construct()
        agent_estimation[:, 0] = oneHot(global_best_path[0][1])
        agent_estimation[:, 1] = oneHot(local_best_path[0][1])
        agent_estimation[:, 2] = oneHot(optimistic_best_path[0][1])
        agent_estimation[:, 3] = oneHot(pessimistic_best_path[0][1])
        dir_prob = agent_estimation @ agent_weight
        error = np.linalg.norm(dir_prob - true_prob.values[index]
                               if isinstance(true_prob, pd.DataFrame) or isinstance(true_prob, pd.Series)
                               else dir_prob - true_prob[index])
        nll += error
        estimation_prob_trajectory.append(dir_prob)
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def estimationErrorAllWithSuicide(param, all_data, true_prob, adjacent_data, adjacent_path, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    '''
    Compute the estimation error with optimistic/pessimistic/global/lazy agents.
    :param param: Parameters.
    :param all_data: All the experimental data (pd.DataFrame).
    :param: true_prob: True probability of directions.
    :param adjacent_data: Adjacent data (pd.DataFrame).
    :param locs_df: Dij distance map data.
    :param reward_amount: Reward value (dict). 
    :param useful_num_samples: Number of samples used in the computation.
    :param return_trajectory: Whether return the estimated probability for each sample.
    :return: 
    '''
    # Parameters
    optimisim_depth = 10
    optimisim_ghost_attractive_thr = 34
    optimisim_fruit_attractive_thr = 34
    optimisim_ghost_repulsive_thr = 12
    global_depth = 5
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    local_depth = 15
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    agent_weight = [param[0], param[1], param[2], param[3], param[4]]
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    is_suicide_trajectory = [] # remains the same during the optimization
    is_scared_trajectory = [] # remains the same during the optimization
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    # for index in range(num_samples):
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        # TODO: rename the columns first before this function
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)]) \
            if "ghost1_pos" in all_data.columns.values or "ghost2_pos" in all_data.columns.values \
            else np.array([each.ghost1Pos, each.ghost2Pos])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared1])
        if "fruit_type" in all_data.columns.values:
            reward_type = int(each.fruit_type)  if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
        #
        # bean_data = bean_data if not isinstance(bean_data, float) else None
        # energizer_data = energizer_data if not isinstance(energizer_data, float) else None
        # fruit_pos = fruit_pos if not isinstance(fruit_pos, float) else None
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
        optimistic_agent = OptimisticAgent(
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
            depth=optimisim_depth,
            ghost_attractive_thr=optimisim_ghost_attractive_thr,
            fruit_attractive_thr=optimisim_fruit_attractive_thr,
            ghost_repulsive_thr=optimisim_ghost_repulsive_thr
        )
        pessimistic_agent = PessimisticAgent(
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
            depth=optimisim_depth,
            ghost_attractive_thr=optimisim_ghost_attractive_thr,
            fruit_attractive_thr=optimisim_fruit_attractive_thr,
            ghost_repulsive_thr=optimisim_ghost_repulsive_thr
        )
        reward_data = bean_data if bean_data is not None else []
        if not isinstance(energizer_data, float) and energizer_data is not None:
            reward_data.extend(energizer_data)
        if not isinstance(fruit_pos, float) and fruit_pos is not None:
            reward_data.append(fruit_pos)
        suicide_agent = SuicideAgent(
            adjacent_data,
            adjacent_path,
            locs_df,
            cur_pos,
            [tuple(each) for each in ghost_data],
            [int(each) for each in ghost_status],
            reward_data,
            last_dir
        )
        # Estimation
        agent_estimation = np.zeros((4, 5))
        _, _, global_best_path = global_agent.construct()
        _, _, local_best_path = local_agent.construct()
        _, _, optimistic_best_path = optimistic_agent.construct()
        _, _,pessimistic_best_path = pessimistic_agent.construct()
        try:
            suicide_choice, is_scared, is_suicide = suicide_agent.nextDir()
        except:
            suicide_choice = np.random.choice(range(len(suicide_agent.available_dir)), 1).item()
            suicide_choice = suicide_agent.available_dir[suicide_choice]
            is_scared = False
            is_suicide = False
        agent_estimation[:, 0] = oneHot(global_best_path[0][1])
        agent_estimation[:, 1] = oneHot(local_best_path[0][1])
        agent_estimation[:, 2] = oneHot(optimistic_best_path[0][1])
        agent_estimation[:, 3] = oneHot(pessimistic_best_path[0][1])
        agent_estimation[:, 4] = oneHot(suicide_choice)
        dir_prob = agent_estimation @ agent_weight
        error = np.linalg.norm(dir_prob - true_prob.values[index]
                               if isinstance(true_prob, pd.DataFrame) or isinstance(true_prob, pd.Series)
                               else dir_prob - true_prob[index])
        nll += error
        estimation_prob_trajectory.append(dir_prob)
        is_suicide_trajectory.append(is_suicide)
        is_scared_trajectory.append(is_scared)
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory, is_suicide_trajectory, is_scared_trajectory)


def MEE(data_filename, map_filename, loc_distance_filename, useful_num_samples = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    # Load experiment data
    with open(data_filename, 'r') as file:
        all_data = pd.read_csv(file)
    true_prob = all_data.pacman_dir
    if "[" in true_prob.values[0]: # If pacman_dir is a vector
        overall_dir = []
        for index in range(all_data.pacman_dir.values.shape[0]):
            each = all_data.pacman_dir.values[index]
            each = each.strip('[]').split(' ')
            while '' in each:  # For the weird case that '' might exist in the split list
                each.remove('')
            overall_dir.append([float(e) for e in each])
        overall_dir = np.array(overall_dir)
        true_prob = overall_dir
    else: # If pacman_dir is the name of directions
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    print("Number of samples : ", all_data.shape[0])
    # Optimization
    if useful_num_samples is None:
        useful_num_samples = all_data.shape[0]
    print("Number of used samples : ", useful_num_samples)
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1})
    func = lambda parameter: estimationError(parameter, all_data, true_prob, adjacent_data, locs_df, reward_amount,
                                                   useful_num_samples = useful_num_samples)
    is_success = False
    retry_num = 0
    while not is_success and retry_num < 10:
        res = scipy.optimize.minimize(
            func,
            x0 = params,
            method = "SLSQP",
            bounds = bounds,
            tol = 1e-8,
            constraints = cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = estimationError(res.x, all_data, true_prob, adjacent_data, locs_df, reward_amount,
                                              useful_num_samples = useful_num_samples, return_trajectory = True)
    true_dir = []
    for index in range(all_data.pacman_dir.values.shape[0]):
        each = all_data.pacman_dir.values[index]
        each = each.strip('[]').split(' ')
        while '' in each:  # For the weird case that '' might exist in the split list
            each.remove('')
        true_dir.append(np.argmax([float(e) for e in each]))
    true_dir = np.array(true_dir)
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))



def MEEWithData(X, Y, map_filename, loc_distance_filename, useful_num_samples = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    adjacent_path = readAdjacentPath(loc_distance_filename)
    reward_amount = readRewardAmount()
    print("Number of samples : ", X.shape[0])
    # Optimization
    if useful_num_samples is None:
        useful_num_samples = X.shape[0]
    print("Number of used samples : ", useful_num_samples)
    # X = X.iloc[:useful_num_samples]
    # Y = Y.iloc[:useful_num_samples]
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] - 1})
    func = lambda parameter: estimationErrorAllWithSuicide(parameter, X, Y, adjacent_data, adjacent_path, locs_df, reward_amount,
                                             useful_num_samples=useful_num_samples)
    is_success = False
    retry_num = 0
    while not is_success and retry_num < 5:
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,
            tol=1e-8,
            constraints=cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = estimationError(res.x, X, Y, adjacent_data, locs_df, reward_amount,
                                        useful_num_samples=useful_num_samples, return_trajectory=True)
    true_dir = np.array([np.argmax(each) for each in Y.iloc[:useful_num_samples]])
    # true_dir = [dir_list.index(each) if isinstance(each, str) else -1 for each in X.pacman_dir.values]
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))


# ===========================================================
#               AGENT WEIGHT ANALYSIS
# ===========================================================
def constructDatasetFromCSV(filename, clip = None):
    # Read data and pre-processing
    agent_dir = pd.read_csv(filename)
    overall_dir = []
    for index in range(agent_dir.pacman_dir.values.shape[0]):
        each = agent_dir.pacman_dir.values[index]
        each = each.strip('[]').split(' ')
        while '' in each: # For the weird case that '' might exist in the split list
            each.remove('')
        overall_dir.append(np.argmax([float(e) for e in each]))
    overall_dir = np.array(overall_dir)
    # Construct the dataset
    if clip is not None and clip > agent_dir.shape[0]:
        print("Warning: requir more data than you have. Use the entire dataset by default.")
        clip = None
    X = agent_dir if clip is None else agent_dir[:clip]
    Y = overall_dir if clip is None else overall_dir[:clip]
    return X, Y


def constructDatasetFromOriginalLog(filename, clip = None, trial_name = None):
    '''
    Construct dataset from a .pkl file.
    :param filename: Filename.
    :param clip: Number of samples used for computation.
    :param trial_name: Trial name.
    :return: 
    '''
    # Read data and pre-processing
    with open(filename, "rb") as file:
        # file.seek(0) # deal with the error that "could not find MARK"
        all_data = pickle.load(file)
    if trial_name is not None: # explicitly indicate the trial
        clip = None
        all_data = all_data[all_data.file == trial_name]
    all_data = all_data.reset_index()
    true_prob = all_data.pacman_dir
    start_index = 0
    while pd.isna(true_prob[start_index]):
        start_index += 1
    true_prob = true_prob[start_index:].reset_index(drop = True)
    all_data = all_data[start_index:].reset_index(drop = True)
    for index in range(1, true_prob.shape[0]):
        if pd.isna(true_prob[index]):
            true_prob[index] = true_prob[index - 1]
    true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    # Construct the dataset
    if clip is not None and clip > all_data.shape[0]:
        print("Warning: requir more data than you have. Use the entire dataset by default.")
        clip = None
    X = all_data if clip is None else all_data[:clip]
    Y = true_prob if clip is None else true_prob[:clip]
    return X, Y


def movingWindowAnalysis(X, Y, map_filename, loc_distance_filename, window = 100,
                         trial_name = None, optimism_agent = False, need_random_lazy = True):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    print("Finished pre-processing!")
    print("Start optimizing...")
    print("="*15)
    # Construct constraints for the optimizer
    if need_random_lazy:
        bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
        params = np.array([0.0, 0.0, 0.0, 0.0])
        cons = []  # construct the bounds in the form of constraints
        for par in range(len(bounds)):
            l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
            u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)
        cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3]  - 1})
    else:
        bounds = [[0, 1], [0, 1]]
        params = np.array([0.0, 0.0])
        cons = []  # construct the bounds in the form of constraints
        for par in range(len(bounds)):
            l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
            u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)
        cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})
    # The indices
    subset_index = np.arange(window, len(Y) - window)
    all_coeff = []
    all_correct_rate = []
    all_success = []
    # Moving the window
    for index in subset_index:
        # if index % 20 == 0:
        print("Window at {}...".format(index))
        sub_X = X[index - window:index + window]
        sub_Y = Y[index - window:index + window]
        # X_train, X_test, Y_train, Y_test = train_test_split(sub_X, sub_Y, test_size=0.2)
        # Optimize with minimum error estimation (MEE)
        if optimism_agent:
            func = lambda parameter: estimationErrorOptimism(parameter, sub_X, sub_Y, adjacent_data, locs_df, reward_amount)
        else:
            func = lambda parameter: estimationError(parameter, sub_X, sub_Y, adjacent_data, locs_df, reward_amount)
        is_success = False
        retry_num = 0
        while not is_success and retry_num < 5:
            res = scipy.optimize.minimize(
                func,
                x0 = params,
                method = "SLSQP",
                bounds = bounds,
                tol = 1e-5,
                constraints = cons
            )
            is_success = res.success
            if not is_success:
                print("Fail, retrying...")
                retry_num += 1
        all_success.append(is_success)
        # Make estimations on the testing dataset
        _, estimated_prob = estimationError(res.x, sub_X, sub_Y, adjacent_data, locs_df, reward_amount, return_trajectory=True)
        estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # Save estimated agent weights
    type = "{}_random_lazy-{}".format("with" if need_random_lazy else "without", "optimism" if optimism_agent else "area")
    np.save("MEE-agent-weight-real_data-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-agent-weight-real_data-window{}-{}-{}.npy".format(window, trial_name, type), all_coeff)
    np.save("MEE-is_success-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-is_success-weight-window{}-{}-{}.npy".format(window, trial_name, type), all_success)



def movingWindowAnalysisAll(X, Y, map_filename, loc_distance_filename, window = 100,
                         trial_name = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    print("Finished pre-processing!")
    print("Start optimizing...")
    print("="*15)
    # Construct constraints for the optimizer
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3]  - 1})
    # The indices
    subset_index = np.arange(window, len(Y) - window)
    all_coeff = []
    all_correct_rate = []
    all_success = []
    # Moving the window
    for index in subset_index:
        # if index % 20 == 0:
        print("Window at {}...".format(index))
        sub_X = X[index - window:index + window]
        sub_Y = Y[index - window:index + window]
        # X_train, X_test, Y_train, Y_test = train_test_split(sub_X, sub_Y, test_size=0.2)
        # Optimize with minimum error estimation (MEE)
        func = lambda parameter: estimationErrorAll(parameter, sub_X, sub_Y, adjacent_data, locs_df, reward_amount)
        is_success = False
        retry_num = 0
        while not is_success and retry_num < 5:
            res = scipy.optimize.minimize(
                func,
                x0 = params,
                method = "SLSQP",
                bounds = bounds,
                tol = 1e-5,
                constraints = cons
            )
            is_success = res.success
            if not is_success:
                print("Fail, retrying...")
                retry_num += 1
        all_success.append(is_success)
        # Make estimations on the testing dataset
        _, estimated_prob = estimationError(res.x, sub_X, sub_Y, adjacent_data, locs_df, reward_amount, return_trajectory=True)
        estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # Save estimated agent weights
    type = "area_and_optimisim"
    np.save("MEE-agent-weight-real_data-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-agent-weight-real_data-window{}-{}-{}.npy".format(window, trial_name, type), all_coeff)
    np.save("MEE-is_success-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-is_success-weight-window{}-{}-{}.npy".format(window, trial_name, type), all_success)


def movingWindowAnalysisAllWithSuicide(X, Y, map_filename, loc_distance_filename, window = 100,
                         trial_name = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    adjacent_path = readAdjacentPath(loc_distance_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    print("Finished pre-processing!")
    print("Start optimizing...")
    print("="*15)
    # Construct constraints for the optimizer
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] + x[4] - 1})
    # The indices
    subset_index = np.arange(window, len(Y) - window)
    all_coeff = []
    all_correct_rate = []
    all_success = []
    # Moving the window
    for index in subset_index:
        # if index % 20 == 0:
        print("Window at {}...".format(index))
        sub_X = X[index - window:index + window]
        sub_Y = Y[index - window:index + window]
        # X_train, X_test, Y_train, Y_test = train_test_split(sub_X, sub_Y, test_size=0.2)
        # Optimize with minimum error estimation (MEE)
        func = lambda parameter: estimationErrorAllWithSuicide(parameter, sub_X, sub_Y, adjacent_data, adjacent_path, locs_df, reward_amount)
        is_success = False
        retry_num = 0
        while not is_success and retry_num < 5:
            res = scipy.optimize.minimize(
                func,
                x0 = params,
                method = "SLSQP",
                bounds = bounds,
                tol = 1e-5,
                constraints = cons
            )
            is_success = res.success
            if not is_success:
                print("Fail, retrying...")
                retry_num += 1
        all_success.append(is_success)
        # Make estimations on the testing dataset
        _, estimated_prob, all_is_suicede, all_is_scared = estimationErrorAllWithSuicide(res.x, sub_X, sub_Y, adjacent_data, adjacent_path, locs_df, reward_amount, return_trajectory=True)
        estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # Save estimated agent weights
    type = "area_and_optimisim"
    np.save("MEE-with_suicide-agent-weight-real_data-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-with_suicide-agent-weight-real_data-window{}-{}-{}.npy".format(window, trial_name, type), all_coeff)
    np.save("MEE-with_suicide-is_success-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-with_suicide-is_success-weight-window{}-{}-{}.npy".format(window, trial_name, type), all_success)
    np.save("MEE-with_suicide-is_suicide-real_data-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-with_suicide-is_suicide-real_data-window{}-{}-{}.npy".format(window, trial_name, type),
            np.array(all_is_suicede))
    np.save("MEE-with_suicide-is_scared-real_data-window{}-{}.npy".format(window, type) if trial_name is None
            else "MEE-with_suicide-is_scared-real_data-window{}-{}-{}.npy".format(window, trial_name, type),
            np.array(all_is_scared))


def plotWeightVariation(all_agent_weight, window, is_success = None, reverse_point = None,
                        with_random_lazy = True, optimism_agent = False):
    # Determine agent names
    if optimism_agent:
        agent_name = ["Random Agent", "Lazy Agent", "Pessimistic Agent", "Optimistic Agent"]
    else:
        agent_name = ["Random Agent", "Lazy Agent", "Local Agent", "Global Agent"]
    if not with_random_lazy:
        agent_name = agent_name[2:]
    # Plot weight variation
    all_coeff = np.array(all_agent_weight)
    if is_success is not None: # TODO: deal with fail optimization
        for index in range(1, is_success.shape[0]):
            if not is_success[index]:
                all_agent_weight[index] = all_agent_weight[index - 1]
    # plt.style.use("seaborn")
    plt.subplot(2, 1, 1)
    plt.title("Agent Weights Variation", fontsize = 30)
    if with_random_lazy:
        plt.stackplot(np.arange(all_coeff.shape[0]),
                      all_coeff[:, 3],  # random agent
                      all_coeff[:, 2],  # lazy agent
                      all_coeff[:, 1],  # local agent
                      all_coeff[:, 0],  # global agent
                      labels = agent_name
                      # labels=["Local Agent", "Global Agent"]
                      )
    else:
        plt.stackplot(np.arange(all_coeff.shape[0]),
                      all_coeff[:, 1],  # local agent
                      all_coeff[:, 0],  # global agent
                      labels = agent_name
                      # labels=["Local Agent", "Global Agent"]
                      )
    plt.ylim(0, 1.0)
    plt.ylabel("Agent Percentage (%)", fontsize=20)
    plt.yticks(
        np.arange(0.1, 1.1, 0.1),
        ["0.{}".format(each)  if each < 10 else "1.0" for each in np.arange(1, 11, 1)],
        fontsize=20)
    plt.xlim(0, all_coeff.shape[0]-1)
    # plt.xlabel("Time Step", fontsize=20)
    x_ticks = list(range(0, all_coeff.shape[0], 10))
    if (all_coeff.shape[0]-1) not in x_ticks:
        x_ticks.append(all_coeff.shape[0]-1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=20, ncol = 4)

    plt.subplot(2, 1, 2)
    if with_random_lazy:
        plt.plot(all_coeff[:, 3], "o-", label=agent_name[0], ms=3, lw=5)
        plt.plot(all_coeff[:, 2], "o-", label=agent_name[1], ms=3, lw=5)
    plt.plot(all_coeff[:, 1], "o-", label=agent_name[2], ms=3, lw=5)
    plt.plot(all_coeff[:, 0], "o-", label=agent_name[3], ms=3, lw=5)
    if reverse_point is not None:
        plt.plot([reverse_point-window, reverse_point-window], [0.0, np.max(all_coeff)+0.1], "k--", lw = 3, alpha = 0.5)
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.05, np.max(all_coeff) + 0.3)
    plt.yticks(
        np.arange(0, 1.3, 0.2),
        ["0.{}".format(each) if each < 10 else "1.0" for each in np.arange(0, 12, 2)],
        fontsize = 20)
    plt.xlim(0, all_coeff.shape[0]-1)
    plt.xlabel("Time Step", fontsize=20)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=20, ncol = 4)
    plt.show()


def plotWeightVariationAllAgent(all_agent_weight, window, is_success = None, reverse_point = None):
    # Determine agent names
    agent_name = ["Pessimistic Agent", "Optimistic Agent", "Local Agent", "Global Agent"]
    # Plot weight variation
    all_coeff = np.array(all_agent_weight)
    if is_success is not None: # TODO: deal with fail optimization
        for index in range(1, is_success.shape[0]):
            if not is_success[index]:
                all_agent_weight[index] = all_agent_weight[index - 1]
    # plt.style.use("seaborn")
    plt.subplot(2, 1, 1)
    plt.title("Agent Weights Variation", fontsize = 30)
    plt.stackplot(np.arange(all_coeff.shape[0]),
                  all_coeff[:, 3],  # pessimistic agent
                  all_coeff[:, 2],  # optimistic agent
                  all_coeff[:, 1],  # local agent
                  all_coeff[:, 0],  # global agent
                  labels=agent_name
                  )
    plt.ylim(0, 1.0)
    plt.ylabel("Agent Percentage (%)", fontsize=20)
    plt.yticks(
        np.arange(0.1, 1.1, 0.1),
        ["0.{}".format(each)  if each < 10 else "1.0" for each in np.arange(1, 11, 1)],
        fontsize=20)
    plt.xlim(0, all_coeff.shape[0]-1)
    # plt.xlabel("Time Step", fontsize=20)
    x_ticks = list(range(0, all_coeff.shape[0], 10))
    if (all_coeff.shape[0]-1) not in x_ticks:
        x_ticks.append(all_coeff.shape[0]-1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=20, ncol = 4)

    plt.subplot(2, 1, 2)
    plt.plot(all_coeff[:, 3], "o-", label=agent_name[0], ms=3, lw=5)
    plt.plot(all_coeff[:, 2], "o-", label=agent_name[1], ms=3, lw=5)
    plt.plot(all_coeff[:, 1], "o-", label=agent_name[2], ms=3, lw=5)
    plt.plot(all_coeff[:, 0], "o-", label=agent_name[3], ms=3, lw=5)
    if reverse_point is not None:
        plt.plot([reverse_point-window, reverse_point-window], [0.0, np.max(all_coeff)+0.1], "k--", lw = 3, alpha = 0.5)
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.05, np.max(all_coeff) + 0.3)
    plt.yticks(
        np.arange(0, 1.3, 0.2),
        ["0.{}".format(each) if each < 10 else "1.0" for each in np.arange(0, 12, 2)],
        fontsize = 20)
    plt.xlim(0, all_coeff.shape[0]-1)
    plt.xlabel("Time Step", fontsize=20)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=20, ncol = 4)
    plt.show()


def plotWeightVariationWithSuicide(all_agent_weight, window, is_success = None, reverse_point = None):
    # Determine agent names
    agent_name = ["Suicide Agent", "Pessimistic Agent", "Optimistic Agent", "Local Agent", "Global Agent"]
    # Plot weight variation
    all_coeff = np.array(all_agent_weight)
    if is_success is not None: # TODO: deal with fail optimization
        for index in range(1, is_success.shape[0]):
            if not is_success[index]:
                all_agent_weight[index] = all_agent_weight[index - 1]
    # plt.style.use("seaborn")
    plt.subplot(2, 1, 1)
    plt.title("Agent Weights Variation", fontsize = 30)
    plt.stackplot(np.arange(all_coeff.shape[0]),
                  all_coeff[:, 4],  # suicide agent
                  all_coeff[:, 3],  # pessimistic agent
                  all_coeff[:, 2],  # optimistic agent
                  all_coeff[:, 1],  # local agent
                  all_coeff[:, 0],  # global agent
                  labels=agent_name
                  )
    plt.ylim(0, 1.0)
    plt.ylabel("Agent Percentage (%)", fontsize=20)
    plt.yticks(
        np.arange(0.1, 1.1, 0.1),
        ["0.{}".format(each)  if each < 10 else "1.0" for each in np.arange(1, 11, 1)],
        fontsize=20)
    plt.xlim(0, all_coeff.shape[0]-1)
    # plt.xlabel("Time Step", fontsize=20)
    x_ticks = list(range(0, all_coeff.shape[0], 10))
    if (all_coeff.shape[0]-1) not in x_ticks:
        x_ticks.append(all_coeff.shape[0]-1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=15, ncol = 5)

    plt.subplot(2, 1, 2)
    plt.plot(all_coeff[:, 4], "o-", label=agent_name[0], ms=3, lw=5)
    plt.plot(all_coeff[:, 3], "o-", label=agent_name[1], ms=3, lw=5)
    plt.plot(all_coeff[:, 2], "o-", label=agent_name[2], ms=3, lw=5)
    plt.plot(all_coeff[:, 1], "o-", label=agent_name[3], ms=3, lw=5)
    plt.plot(all_coeff[:, 0], "o-", label=agent_name[4], ms=3, lw=5)
    if reverse_point is not None:
        plt.plot([reverse_point-window, reverse_point-window], [0.0, np.max(all_coeff)+0.1], "k--", lw = 3, alpha = 0.5)
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.05, np.max(all_coeff) + 0.3)
    plt.yticks(
        np.arange(0, 1.3, 0.2),
        ["0.{}".format(each) if each < 10 else "1.0" for each in np.arange(0, 12, 2)],
        fontsize = 20)
    plt.xlim(0, all_coeff.shape[0]-1)
    plt.xlabel("Time Step", fontsize=20)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=15, ncol = 5)
    plt.show()


def _consecutiveInterval(list):
    list = np.hstack(([0], list, [0]))# avoid not ended consecutive numbers
    start_index = np.where(np.diff(list) == 1)[0] + 1
    end_index = np.where(np.diff(list) == -1)[0] + 1
    consecutive_list = [[start_index[i], end_index[i]] for i in range(len(start_index))]
    return consecutive_list


def plotTrueLabel(trial_name, window):
    #TODO: check the label
    # Read data
    with open("../common_data/labeled_df_toynew.pkl", "rb") as file:
        data = pickle.load(file)
    # Extract labels
    data = data[data.file == trial_name]
    data = data[
        ["label_local_graze",
         "label_hunt1",
         "label_hunt2",
         "label_prehunt",
         "label_global_optimal",
         "label_global_notoptimal",
         "label_evade"]
    ]
    data = data.fillna(0)
    global_graze_label = np.array(np.logical_or(data.label_global_optimal.values, data.label_global_notoptimal.values), dtype = np.int)
    local_graze_label = data.label_local_graze.values
    optimistic_label = data.label_prehunt.values
    pessimistic_label = data.label_evade.values
    hunt_label = np.array(np.logical_or(data.label_hunt1.values, data.label_hunt2.values), dtype = np.int)

    global_graze_index = _consecutiveInterval(global_graze_label)
    local_graze_index = _consecutiveInterval(local_graze_label)
    optimistic_index = _consecutiveInterval(optimistic_label)
    pessimistic_index = _consecutiveInterval(pessimistic_label)
    hunt_index = _consecutiveInterval(hunt_label)

    # Plot true labels for this trial
    # plt.plot(global_graze_label, "o", label="Global Graze", ms=5)
    # plt.plot(local_graze_label, "o", label="Local Graze", ms=5)
    # plt.plot(optimistic_label, "o", label="Optimistic (Pre-Hunt)", ms=5)
    # plt.plot(pessimistic_label, "o", label="Pessimistic (Evade)", ms=5)
    for each in global_graze_index:
        plt.fill_between(each, [1.0, 1.0], color = "red")
    for each in local_graze_index:
        plt.fill_between(each, [1.0, 1.0], color = "#0AA344")
    for each in optimistic_index:
        plt.fill_between(each, [1.0, 1.0], color = "#FF8936")
    for each in pessimistic_index:
        plt.fill_between(each, [1.0, 1.0], color = "blue")
    plt.ylabel("Label Indication", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.yticks([0.0, 1.0], [0, 1], fontsize=20)
    plt.xlim(0, data.shape[0] - 1)
    plt.xlabel("Time Step", fontsize=20)
    x_ticks = list(range(0, data.shape[0], 10))
    if (data.shape[0] - 1) not in x_ticks:
        x_ticks.append(data.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    # plt.legend(fontsize=20, ncol=4)
    plt.show()



if __name__ == '__main__':
    # Data
    map_filename = "extracted_data/adjacent_map.csv"
    loc_distance_filename = "extracted_data/dij_distance_map.csv"
    original_data_filename = "../common_data/df_total_with_reward.pkl"

    # # MLE (maximum likelihood estimation)
    # # Note: The performance of MEE is much better.
    # print("="*10, " MLE ", "="*10)
    # data_filename = "stimulus_data/local-graze/diary.csv"
    # MLE(data_filename, map_filename, loc_distance_filename, useful_num_samples = 50)

    # # MEE (minimum error estimation)
    # print("=" * 10, " MEE ", "=" * 10)
    # original_data_filename = "./extracted_data/all_suicide_data.pkl"
    # MEE(data_filename, map_filename, loc_distance_filename, useful_num_samples = 200)
    # X, Y = constructDatasetFromOriginalLog(original_data_filename, clip=200, trial_name = None)
    # MEEWithData(X, Y, map_filename, loc_distance_filename, useful_num_samples = 30)

    # # Moving Window Analysis with MEE
    # trial_name = "1-1-Omega-15-Jul-2019.csv" # filename for the trial
    # original_data_filename = "./extracted_data/one_trial_suicide_data.pkl"
    #
    # type = "suicide"
    # X, Y = constructDatasetFromOriginalLog(original_data_filename, clip = 50, trial_name = None)
    # print("Data Shape : ", X.shape)
    #
    # if type == "all": # use gloabl/local/optimistic/pessimistic agents for analysis
    #     print("{}--{}".format(type, trial_name))
    #     movingWindowAnalysisAll(X, Y, map_filename, loc_distance_filename, window=20, trial_name=trial_name)
    # elif type == "suicide": # use gloabl/local/optimistic/pessimistic agents for analysis
    #     print("{}--{}".format(type, trial_name))
    #     movingWindowAnalysisAllWithSuicide(X, Y, map_filename, loc_distance_filename, window=10, trial_name=None)
    # else:
    #     need_random_lazy = False # include random and lazy agents?
    #     need_optimism = False # use optimitic & pessimitic or global & local agents?
    #     print("{}--{}--{}--{}".format(
    #         type,
    #         trial_name,
    #         "with_random_lazy" if need_random_lazy else "without_random_lazy",
    #         "optimisim" if need_optimism else "area"))
    #     movingWindowAnalysis(X, Y, map_filename, loc_distance_filename, window = 10,
    #                             trial_name = "1-1-Omega-15-Jul-2019.csv", need_random_lazy = True, optimism_agent = True)


    # Plot agent weights variation
    all_agent_weight = np.load("MEE-with_suicide-agent-weight-real_data-window10-area_and_optimisim.npy")
    is_success = np.load("MEE-with_suicide-is_success-weight-window10-area_and_optimisim.npy")
    is_suicide = np.load("MEE-with_suicide-is_suicide-real_data-window10-area_and_optimisim.npy")
    is_scared = np.load("MEE-with_suicide-is_scared-real_data-window10-area_and_optimisim.npy")
    print("Suicide Index : ", np.where(is_suicide == 1))
    # plotWeightVariation(all_agent_weight, is_success = is_success, window = 20, reverse_point = None,
    #                     with_random_lazy = True, optimism_agent = False)
    # plotWeightVariationAllAgent(all_agent_weight, is_success=is_success, window=20, reverse_point=None)
    plotWeightVariationWithSuicide(all_agent_weight, is_success=is_success, window = 10, reverse_point=None)


    # plotTrueLabel("1-2-Omega-15-Jul-2019.csv", window = 20)