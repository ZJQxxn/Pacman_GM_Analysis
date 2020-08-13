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


def readDatasetFromPkl(filename, trial_name = None):
    '''
    Construct dataset from a .pkl file.
    :param filename: Filename.
    :param trial_name: Trial name.
    :return: 
    '''
    # Read data and pre-processing
    with open(filename, "rb") as file:
        # file.seek(0) # deal with the error that "could not find MARK"
        all_data = pickle.load(file)
    if trial_name is not None: # explicitly indicate the trial
        all_data = all_data[all_data.file == trial_name]
    all_data = all_data.reset_index()
    true_prob = all_data.next_pacman_dir_fill
    # true_prob = all_data.pacman_dir
    # TODO: when the pacman stays
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
    X = all_data
    Y = true_prob
    return X, Y


def makeChoice(prob):
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


# ===========================================================
#                      ESTIMATION
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
    # Check function variables
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        # print("Agent List :", agents_list)
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
    if "optimistic" in agents_list:
        print()
        optimistic_depth = utility_param["optimistic_depth"]
        optimistic_ghost_attractive_thr = utility_param["optimistic_ghost_attractive_thr"]
        optimistic_fruit_attractive_thr = utility_param["optimistic_fruit_attractive_thr"]
        optimistic_ghost_repulsive_thr = utility_param["optimistic_ghost_repulsive_thr"]
    if "pessimistic" in agents_list:
        pessimistic_depth = utility_param["pessimistic_depth"]
        pessimistic_ghost_attractive_thr = utility_param["pessimistic_ghost_attractive_thr"]
        pessimistic_fruit_attractive_thr = utility_param["pessimistic_fruit_attractive_thr"]
        pessimistic_ghost_repulsive_thr = utility_param["pessimistic_ghost_repulsive_thr"]
    agent_weight = list(param)
    # Compute log likelihood
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    for index in range(num_samples):
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
            reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
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
            agent_object_dict["local"] = local_agent
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
            agent_object_dict["optimistic"] = optimistic_agent
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
            agent_object_dict["pessimistic"] = pessimistic_agent
        if "lazy" in agents_list:
            lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir[index]) # TODO: max_loop should be a param
            agent_object_dict["lazy"] = lazy_agent
        if "random" in agents_list:
            random_agent = RandomAgent(adjacent_data, cur_pos, last_dir[index], None)
            agent_object_dict["random"] = random_agent
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
                last_dir[index]
            )
            agent_object_dict["suicide"] = suicide_agent
        # Estimation
        agent_estimation = np.zeros((4, len(agents_list))) # (number of directions, number of agents)
        for i, agent in enumerate(agents_list):
            agent_estimation[:, i] = oneHot(agent_object_dict[agent].nextDir())
        dir_prob = agent_estimation @ agent_weight
        # best_dir_index = np.argmax(dir_prob)
        best_dir_index = makeChoice(dir_prob)
        exp_prob = np.exp(dir_prob)
        log_likelihood = dir_prob[best_dir_index] - np.log(np.sum(exp_prob))
        nll += (-log_likelihood)
        estimation_prob_trajectory.append(exp_prob / np.sum(exp_prob)) #TODO: what to append in the estimation
    # print('Finished')
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def estimationError(param, utility_param, all_data, true_prob, adjacent_data, adjacent_path, locs_df, reward_amount, agents_list, return_trajectory = False):
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
    #TODO: two loss function
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        # print("Agent List :", agents_list)
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
    if "optimistic" in agents_list:
        optimistic_depth = utility_param["optimistic_depth"]
        optimistic_ghost_attractive_thr = utility_param["optimistic_ghost_attractive_thr"]
        optimistic_fruit_attractive_thr = utility_param["optimistic_fruit_attractive_thr"]
        optimistic_ghost_repulsive_thr = utility_param["optimistic_ghost_repulsive_thr"]
    if "pessimistic" in agents_list:
        pessimistic_depth = utility_param["pessimistic_depth"]
        pessimistic_ghost_attractive_thr = utility_param["pessimistic_ghost_attractive_thr"]
        pessimistic_fruit_attractive_thr = utility_param["pessimistic_fruit_attractive_thr"]
        pessimistic_ghost_repulsive_thr = utility_param["pessimistic_ghost_repulsive_thr"]
    agent_weight = list(param)
    # Compute estimation error
    ee = 0  # estimation error
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    for index in range(num_samples):
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
            reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
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
            agent_object_dict["local"] = local_agent
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
            agent_object_dict["optimistic"] = optimistic_agent
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
            agent_object_dict["pessimistic"] = pessimistic_agent
        if "lazy" in agents_list:
            lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir[index],) # TODO: max_loop should be a param
            agent_object_dict["lazy"] = lazy_agent
        if "random" in agents_list:
            random_agent = RandomAgent(adjacent_data, cur_pos, last_dir[index], None)
            agent_object_dict["random"] = random_agent
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
                last_dir[index]
            )
            agent_object_dict["suicide"] = suicide_agent
        # Estimation
        agent_estimation = np.zeros((4, len(agents_list))) # (number of directions, number of agents)
        for i, agent in enumerate(agents_list):
            agent_estimation[:, i] = oneHot(agent_object_dict[agent].nextDir())
        dir_prob = agent_estimation @ agent_weight
        error = np.linalg.norm(dir_prob - true_prob.values[index]
                               if isinstance(true_prob, pd.DataFrame) or isinstance(true_prob, pd.Series)
                               else dir_prob - true_prob[index])
        ee += error
        estimation_prob_trajectory.append(dir_prob)
    if not return_trajectory:
        return ee
    else:
        return (ee, estimation_prob_trajectory)


def MLE(config):
    print("=" * 20, " MLE ", "=" * 20)
    print("Agent List :", config["agents"])
    # Load pre-computed data
    adjacent_data = readAdjacentMap(config["map_filename"])
    locs_df = readLocDistance(config["loc_distance_filename"])
    adjacent_path = readAdjacentPath(config["loc_distance_filename"])
    reward_amount = readRewardAmount()
    # Load experiment data
    all_data, true_prob = readDatasetFromPkl(config["data_filename"]) # TODO: trial name
    print("Number of samples : ", all_data.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = all_data.shape[0]
    else:
        num_samples = all_data.shape[0] if config["clip_samples"] > all_data.shape[0] else config["clip_samples"]
    all_data = all_data.iloc[:num_samples]
    true_prob = true_prob.iloc[:num_samples]
    print("Number of used samples : ", all_data.shape[0])
    # Optimization
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})# TODO: summation
    func = lambda parameter: negativeLogLikelihood(
        params,
        config["utility_param"],
        all_data,
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        config['agents'],
        return_trajectory = False
    )
    is_success = False
    retry_num = 0
    while not is_success and retry_num < config["maximum_try"]:
        res = scipy.optimize.minimize(
            func,
            x0 = params,
            method="SLSQP",
            bounds=bounds,
            tol=1e-5,
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
    _, estimated_prob = negativeLogLikelihood(
        res.x,
        config["utility_param"],
        all_data,
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        config['agents'],
        return_trajectory = True
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))


def MEE(config):
    print("=" * 20, " MEE ", "=" * 20)
    print("Agent List :", config["agents"])
    # Load pre-computed data
    adjacent_data = readAdjacentMap(config["map_filename"])
    locs_df = readLocDistance(config["loc_distance_filename"])
    adjacent_path = readAdjacentPath(config["loc_distance_filename"])
    reward_amount = readRewardAmount()
    # Load experiment data
    all_data, true_prob = readDatasetFromPkl(config["data_filename"])  # TODO: trial name
    print("Number of samples : ", all_data.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = all_data.shape[0]
    else:
        num_samples = all_data.shape[0] if config["clip_samples"] > all_data.shape[0] else config["clip_samples"]
    all_data = all_data.iloc[:num_samples]
    true_prob = true_prob.iloc[:num_samples]
    print("Number of used samples : ", all_data.shape[0])
    # Optimization
    # bounds = [[0.0, 1.0]] * len(config["agents"])
    # params = np.array([0.25] * len(config["agents"]))
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    func = lambda parameter: estimationError(
        params,
        config["utility_param"],
        all_data,
        true_prob,
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        config["agents"],
        return_trajectory = False
    )
    is_success = False
    retry_num = 0
    while not is_success and retry_num < config["maximum_try"]:
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
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = estimationError(
        res.x,
        config["utility_param"],
        all_data,
        true_prob,
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        config["agents"],
        return_trajectory = True
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))


def movingWindowAnalysis(config):
    print("=" * 20, " Moving Window ", "=" * 20)
    print("Agent List :", config["agents"])
    window = config["window"]
    # Load pre-computed data
    adjacent_data = readAdjacentMap(config["map_filename"])
    locs_df = readLocDistance(config["loc_distance_filename"])
    adjacent_path = readAdjacentPath(config["loc_distance_filename"])
    reward_amount = readRewardAmount()
    # Load experiment data
    X, Y = readDatasetFromPkl(config["data_filename"])
    print("Number of samples : ", X.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = X.shape[0]
    else:
        num_samples = X.shape[0] if config["clip_samples"] > X.shape[0] else config["clip_samples"]
    X = X.iloc[:num_samples]
    Y = Y.iloc[:num_samples]
    print("Number of used samples : ", X.shape[0])
    # Construct optimizer
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})
    if "MEE" == config["method"]:
        func = lambda parameter: estimationError(
            params,
            config["utility_param"],
            sub_X,
            sub_Y,
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            config["agents"],
            return_trajectory=False
        )
    elif "MLE" == config["method"]:
        func = lambda parameter: negativeLogLikelihood(
            params,
            config["utility_param"],
            sub_X,
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            config['agents'],
            return_trajectory=False
        )
    else:
        raise ValueError('Undefined optimizer {}! Should be "MLE" or "MEE".'.format(config["method"]))
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
        # optimization in the window
        is_success = False
        retry_num = 0
        while not is_success and retry_num < config["maximum_try"]:
            res = scipy.optimize.minimize(
                func,
                x0 = params,
                method="SLSQP",
                bounds=bounds,
                tol=1e-5,
                constraints = cons
            )
            is_success = res.success
            if not is_success:
                print("Fail, retrying...")
                retry_num += 1
        all_success.append(is_success)
        # Correct rate
        if "MEE" == config["method"]:
            _, estimated_prob = estimationError(
                res.x,
                config["utility_param"],
                sub_X,
                sub_Y,
                adjacent_data,
                adjacent_path,
                locs_df,
                reward_amount,
                config["agents"],
                return_trajectory = True
            )
        elif "MLE" == config["method"]:
            _, estimated_prob = negativeLogLikelihood(
                res.x,
                config["utility_param"],
                sub_X,
                adjacent_data,
                adjacent_path,
                locs_df,
                reward_amount,
                config['agents'],
                return_trajectory = True
            )
        estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # Save estimated agent weights
    type = "_".join(config['agents'])
    np.save("MEE-agent_weight-window{}-{}.npy".format(window, type), all_coeff)
    np.save("MEE-is_success-window{}-{}.npy".format(window, type), all_success)


# ===========================================================
#                       PLOTTING
# ===========================================================
def plotWeightVariation(all_agent_weight, agent_list, window, is_success = None, plot_label = False, filename = None):
    #TODO: specify colors for all agents
    # Determine agent names
    agent_name = [each + " agent" for each in agent_list]
    # Plot weight variation
    all_coeff = np.array(all_agent_weight)
    # Deal with failed optimization
    if is_success is not None:
        for index in range(1, is_success.shape[0]):
            if not is_success[index]:
                all_agent_weight[index] = all_agent_weight[index - 1]
    # plt.style.use("seaborn")
    # plt.plot(all_coeff[:, 4], "o-", label=agent_name[0], ms=3, lw=5)
    # plt.plot(all_coeff[:, 3], "o-", label=agent_name[1], ms=3, lw=5)
    # plt.plot(all_coeff[:, 2], "o-", label=agent_name[2], ms=3, lw=5)
    # plt.plot(all_coeff[:, 1], "o-", label=agent_name[3], ms=3, lw=5)
    # plt.plot(all_coeff[:, 0], "o-", label=agent_name[4], ms=3, lw=5)
    plt.plot(all_coeff, ms=3, lw=5)
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize=20)
    with open(filename, "rb") as file:
        trial_data = pickle.load(file)
    if plot_label:
        plotTrueLabel(trial_data)
        plt.ylim(-0.1, np.max(all_coeff) + 0.3)
    else:
        plt.ylim(-0.05, np.max(all_coeff) + 0.3)
    plt.yticks(
        np.arange(0, 1.3, 0.2),
        ["0.{}".format(each) if each < 10 else "1.0" for each in np.arange(0, 12, 2)],
        fontsize = 20)
    plt.xlim(0, all_coeff.shape[0]-1)
    plt.xlabel("Time Step", fontsize=20)
    x_ticks = list(range(0, all_coeff.shape[0], 10))
    if (all_coeff.shape[0] - 1) not in x_ticks:
        x_ticks.append(all_coeff.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.legend(fontsize=15, ncol = 5)
    plt.show()


def plotTrueLabel(trial_data):
    # TODO: specify colors for all labels
    # Lebeling data
    is_local, is_global, is_evade, is_suicide, is_optimistic, is_pessimistic = labeling(trial_data)
    # Plot labels
    local_label_index = np.where(is_local)
    global_label_index = np.where(is_global)
    suicide_label_index = np.where(is_suicide)
    # TODO: the color
    for each in local_label_index[0]:
        plt.fill_between(x=[each, each + 1], y1=0, y2=-0.1, color="green")
    for each in global_label_index[0]:
        plt.fill_between(x=[each, each + 1], y1=0, y2=-0.1, color="red")
    for each in suicide_label_index[0]:
        plt.fill_between(x=[each, each + 1], y1=0, y2=-0.1, color="black")
    # plt.ylim(-0.05, 0.2)
    # plt.yticks(np.arange(0, 0.21, 0.1), np.arange(0, 0.21, 0.1))



if __name__ == '__main__':
    # Configurations
    config = {
        # Filename
        "data_filename" : "../common_data/1-1-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl",
        "map_filename" : "extracted_data/adjacent_map.csv",
        "loc_distance_filename" : "extracted_data/dij_distance_map.csv",
        # The number of samples used for estimation: None for using all the data
        "clip_samples" : 50,
        # The window size
        "window" : 10,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try" : 5,
        # Optimization method: "MLE" (maximumn likelihood estimation) or "MEE" (minimum error estimation)
        "method": "MEE",
        # Loss function (required when method = "MEE"): "l2-norm" or "cross-entropy"
        "loss-func": "l2-norm",
        # Initial guess of parameters
        "params": [0.0, 0.0, 0.0, 0.0],
        # Bounds for optimization
        "bounds": [[0, 1], [0, 1], [0, 1], [0, 1]],
        # Agents: at least one of "global", "local", "lazy", "random", "optimistic", "pessimistic", "suicide".
        # "agents":["global", "local", "random", "lazy", "random", "optimistic", "pessimistic", "suicide"],
        "agents":["global", "local", "lazy", "random"],
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
            "local_ghost_repulsive_thr" : 5,
            # for optimistic agent
            "optimistic_depth" : 10,
            "optimistic_ghost_attractive_thr" : 34,
            "optimistic_fruit_attractive_thr" : 34,
            "optimistic_ghost_repulsive_thr" : 12,
            # for pessimistic agent
            "pessimistic_depth": 10,
            "pessimistic_ghost_attractive_thr": 34,
            "pessimistic_fruit_attractive_thr": 34,
            "pessimistic_ghost_repulsive_thr": 12,
        }
    }

    # ============ ESTIMATION =============
    MLE(config)
    MEE(config)

    # ============ MOVING WINDOW =============
    # movingWindowAnalysis(config)

    # ============ PLOTTING =============
    # Load the log of moving window analysis; log files are created in the analysis
    # agent_weight = np.load("MEE-agent_weight-window10-global_local_lazy_random.npy")
    # is_success = np.load("MEE-is_success-window10-global_local_lazy_random.npy")
    # plotWeightVariation(agent_weight, config["agents"], config["window"], is_success,
    #                     plot_label = True, filename = config["data_filename"])