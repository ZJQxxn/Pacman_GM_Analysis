'''
Description:
    MLE parameter estimation for multi-agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    Aug. 11 2020
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

sys.path.append('./')
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from PathTreeConstructor import PathTree, OptimisticAgent, PessimisticAgent
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent
from SuicideAgent import SuicideAgent

sys.path.append('../common_data')
from LabelingData import labeling

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
#             ESTIMATION
# ===========================================================

def negativeLogLikelihood(param, utility_param, all_data, adjacent_data, adjacent_path, locs_df, reward_amount, agents_list, return_trajectory = False):
    '''
    Compute the negative log likelihood given data. 
    :param param: Parameters.
    :param all_data: All the experimental data (pd.DataFrame).
    :param adjacent_data: Adjacent data (pd.DataFrame).
    :param locs_df: Dij distance map data.
    :param reward_amount: Reward value (dict). 
    :param useful_num_samples: Number of samples used in the computation.
    :param return_trajectory: Whether return the estimated probability for each sample.
    :return: 
    '''
    #TODO: need more parameter check
    # Check function variables
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        print("Agent List :", agents_list)
        agent_object_dict = {each : None for each in agents_list}
    # Parameters
    if "global" in agents_list:
        global_depth = utility_param["global_depth"]
        global_ghost_attractive_thr = utility_param["global_ghost_attractive_thr"]
        global_fruit_attractive_thr = utility_param["global_fruit_attractive_thr"]
        global_ghost_repulsive_thr = utility_param["global_ghost_repulsive_thr"]
    if "local" in agents_list:
        local_depth = utility_param["local_depth"]
        local_ghost_attractive_thr = utility_param["local_ghost_attractive_thr"]
        local_fruit_attractive_thr = utility_param["local_fruit_attractive_thr"]
        local_ghost_repulsive_thr = utility_param["local_ghost_repulsive_thr"]
    agent_weight = list(param)
    # Compute log likelihood
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None # TODO: need revise for lazy agent
    loop_count = 0 #TODO: for lazyAgent; revise later
    # useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(num_samples):
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
        if "global" in agents_list:
            global_agent = PathTree( # TODO: parameters change to two parts: constant and game status
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
                depth = global_depth,
                ghost_attractive_thr = global_ghost_attractive_thr,
                fruit_attractive_thr = global_fruit_attractive_thr,
                ghost_repulsive_thr = global_ghost_repulsive_thr
            )
            agent_object_dict["global"] = global_agent
        if "local" in agents_list:
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
        if "optimistic" in agents_list:
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
                depth = optimistic_depth,
                ghost_attractive_thr = optimistic_ghost_attractive_thr,
                fruit_attractive_thr = optimistic_fruit_attractive_thr,
                ghost_repulsive_thr = optimistic_ghost_repulsive_thr
            )
        if "pessimistic" in agents_list:
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
                depth = pessimistic_depth,
                ghost_attractive_thr = pessimistic_ghost_attractive_thr,
                fruit_attractive_thr = pessimistic_fruit_attractive_thr,
                ghost_repulsive_thr = pessimistic_ghost_repulsive_thr
            )
        if "lazy" in agents_list:
            lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir, loop_count, max_loop=5) # TODO: max_loop should be a param
        if "random" in agents_list:
            random_agent = RandomAgent(adjacent_data, cur_pos, last_dir, None)
        if "suicide" in agents_list:
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


def estimationError(param, all_data, true_prob, adjacent_data, locs_df, reward_amount, agents_list, useful_num_samples = None, return_trajectory = False):
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


if __name__ == '__main__':
    config = {
        # Optimization method: "MLE" (maximumn likelihood estimation) or "MEE" (minimum error estimation)
        "method": "MEE",
        # Loss function (required when method = "MEE"): "l2-norm" or "cross-entropy"
        "loss-func": "l2-norm",
        # Agents
        "agents":["global", "local", "random", "lazy", "random", "optimistic", "pessimistic", "suicide"],
        # Parameters for computing the utility
        "utility_param":{
            # for global agent
            "global_depth" : 15,
            "global_ghost_attractive_thr" : 34,
            "global_fruit_attractive_thr" : 34,
            "global_ghost_repulsive_thr" : 12,
            # for local agent
            "local_depth" : 5,
            "local_ghost_attractive_thr" : 5,
            "local_fruit_attractive_thr" : 5,
            "local_ghost_repulsive_thr" : 5
        }
    }

    # ============ TEST =============