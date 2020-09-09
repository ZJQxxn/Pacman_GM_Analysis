'''
Description:
    Fast MLE analysis with pre-computed estimations of every agent.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    17 Aug. 2020
'''

import pickle
import pandas as pd
import numpy as np
import scipy.optimize
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber, makeChoice
from PathTreeAgent import PathTree
from SuicideAgent import SuicideAgent
from PlannedHuntingAgent import PlannedHuntingAgent


# ===================================
#         UTILITY FUNCTION
# ===================================
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


def readDatasetFromPkl(filename, trial_name = None, only_necessary = False):
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
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index()
    # Exclude the (0, 18) position in data
    normal_data_index = []
    for index in range(all_data.shape[0]):
        if not isinstance(all_data.global_Q[index], list): # what if no normal Q?
            normal_data_index.append(index)
    all_data = all_data.iloc[normal_data_index]
    true_prob = all_data.next_pacman_dir_fill
    # Fill nan direction for optimization use
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
    if only_necessary and "at_cross" in all_data.columns.values:
        print("--- Only Necessary ---")
        at_cross_index = np.where(all_data.at_cross)
        X = all_data.iloc[at_cross_index]
        Y = true_prob.iloc[at_cross_index]
        # print("--- Data Shape {} ---".format(len(at_cross_index[0])))
    else:
        X = all_data
        Y = true_prob
    return X, Y


def readTestingDatasetFromPkl(filename, trial_name = None, only_necessary = False):
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
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index()
    # true_prob = all_data.next_pacman_dir_fill
    true_prob = all_data.next_pacman_dir
    # The indeices of data with a direction rather than nan
    if only_necessary:
        not_nan_indication = lambda x: not isinstance(x.next_pacman_dir, float) and x.at_cross
    else:
        not_nan_indication = lambda x: not isinstance(x.next_pacman_dir, float)
    not_nan_index = np.where(all_data.apply(lambda x: not_nan_indication(x), axis = 1))[0]
    all_data = all_data.iloc[not_nan_index]
    true_prob = true_prob.iloc[not_nan_index]
    true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    # Construct the dataset
    if only_necessary and "at_cross" in all_data.columns.values:
        print("--- Only Necessary ---")
        at_cross_index = np.where(all_data.at_cross)
        X = all_data.iloc[at_cross_index]
        Y = true_prob.iloc[at_cross_index]
        # print("--- Data Shape {} ---".format(len(at_cross_index[0])))
    else:
        X = all_data
        Y = true_prob
    return X, Y


# ===================================
#       INDIVIDUAL ESTIMATION
# ===================================
def _readData(filename):
    with open(filename, "rb") as file:
        # file.seek(0) # deal with the error that "could not find MARK"
        all_data = pickle.load(file)
    all_data = all_data.reset_index()
    print()
    return all_data


def _readAuxiliaryData():
    # Load pre-computed data
    adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    adjacent_path = readAdjacentPath("extracted_data/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    return adjacent_data, locs_df, adjacent_path, reward_amount


def _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount):
    # Randomness and laziness
    randomness_coeff = 0.0
    laziness_coeff = 0.0
    # Configuration (for global agent)
    global_depth = 15
    ignore_depth = 5
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 34
    # Configuration (for local agent)
    local_depth = 5
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    # Configuration (for optimistic agent)
    optimistic_depth = 5
    optimistic_ghost_attractive_thr = 5
    optimistic_fruit_attractive_thr = 5
    optimistic_ghost_repulsive_thr = 5
    # Configuration (for pessimistic agent)
    pessimistic_depth = 5
    pessimistic_ghost_attractive_thr = 5
    pessimistic_fruit_attractive_thr = 5
    pessimistic_ghost_repulsive_thr = 5
    # Configuration (for suicide agent)
    suicide_depth = 5
    suicide_ghost_attractive_thr = 5
    suicide_fruit_attractive_thr = 5
    suicide_ghost_repulsive_thr = 5
    # Configuration (flast direction)
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Direction sstimation
    global_estimation = []
    local_estimation = []
    optimistic_estimation = []
    pessimistic_estimation = []
    suicide_estimation = []
    planned_hunting_estimation = []
    # Q-value (utility)
    global_Q = []
    local_Q = []
    optimistic_Q = []
    pessimistic_Q = []
    suicide_Q = []
    planned_hunting_Q = []
    num_samples = all_data.shape[0]
    print("Sample Num : ", num_samples)
    estimated_index = []
    for index in range(num_samples):
        if 0 == (index + 1) % 20:
            print("Finished estimation at {}".format(index + 1))
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        # In case the Pacman position does not exists, e.g. (0, 18)
        if cur_pos not in adjacent_data:
            global_Q.append([0.0, 0.0, 0.0, 0.0])
            local_Q.append([0.0, 0.0, 0.0, 0.0])
            optimistic_Q.append([0.0, 0.0, 0.0, 0.0])
            pessimistic_Q.append([0.0, 0.0, 0.0, 0.0])
            suicide_Q.append([0.0, 0.0, 0.0, 0.0])
            planned_hunting_Q.append([0.0, 0.0, 0.0, 0.0])
            continue
        else:
            estimated_index.append(index)
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
        # Global agents
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
            last_dir[index],
            depth=global_depth,
            ignore_depth=ignore_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff
        )
        global_result = global_agent.nextDir(return_Q=True)
        global_estimation.append(global_result[0])
        global_Q.append(global_result[1])
        # Local estimation
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
            last_dir[index],
            depth=local_depth,
            ghost_attractive_thr = local_ghost_attractive_thr,
            fruit_attractive_thr = local_fruit_attractive_thr,
            ghost_repulsive_thr = local_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff
        )
        local_result = local_agent.nextDir(return_Q=True)
        local_estimation.append(local_result[0])
        local_Q.append(local_result[1])
        # Optimistic agent
        optimistic_agent = PathTree(
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
            last_dir[index],
            depth = optimistic_depth,
            ghost_attractive_thr = optimistic_ghost_attractive_thr,
            fruit_attractive_thr = optimistic_fruit_attractive_thr,
            ghost_repulsive_thr = optimistic_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 1.0,
            risk_coeff = 0.0
        )
        optimistic_result = optimistic_agent.nextDir(return_Q=True)
        optimistic_estimation.append(optimistic_result[0])
        optimistic_Q.append(optimistic_result[1])
        # Pessimistic agent
        pessimistic_agent = PathTree(
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
            last_dir[index],
            depth=pessimistic_depth,
            ghost_attractive_thr=pessimistic_ghost_attractive_thr,
            fruit_attractive_thr=pessimistic_fruit_attractive_thr,
            ghost_repulsive_thr=pessimistic_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 0.0,
            risk_coeff = 1.0
        )
        pessimistic_result = pessimistic_agent.nextDir(return_Q=True)
        pessimistic_estimation.append(pessimistic_result[0])
        pessimistic_Q.append(pessimistic_result[1])
        # Suicide agent
        suicide_agent = SuicideAgent(
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            depth = suicide_depth,
            ghost_attractive_thr = suicide_ghost_attractive_thr,
            ghost_repulsive_thr = suicide_fruit_attractive_thr,
            fruit_attractive_thr = suicide_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff
        )
        suicide_result = suicide_agent.nextDir(return_Q=True)
        suicide_estimation.append(suicide_result[0])
        suicide_Q.append(suicide_result[1])
        # Planned hunting agent
        planned_hunting_agent = PlannedHuntingAgent(
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            ghost_data,
            ghost_status,
            last_dir[index],
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff
        )
        planned_hunting_result = planned_hunting_agent.nextDir(return_Q=True)
        planned_hunting_estimation.append(planned_hunting_result[0])
        planned_hunting_Q.append(planned_hunting_result[1])
    # Assign new columns
    print("Estimation length : ", len(global_estimation))
    print("Data Shape : ", all_data.shape)
    all_data["global_Q"] = np.tile(np.nan, num_samples)
    all_data["global_Q"] = all_data["global_Q"].apply(np.array)
    all_data["global_Q"] = global_Q
    all_data["local_Q"] = np.tile(np.nan, num_samples)
    all_data["local_Q"] = all_data["local_Q"].apply(np.array)
    all_data["local_Q"] = local_Q
    all_data["optimistic_Q"] = np.tile(np.nan, num_samples)
    all_data["optimistic_Q"] = all_data["optimistic_Q"].apply(np.array)
    all_data["optimistic_Q"] = optimistic_Q
    all_data["pessimistic_Q"] = np.tile(np.nan, num_samples)
    all_data["pessimistic_Q"] = all_data["pessimistic_Q"].apply(np.array)
    all_data["pessimistic_Q"] = pessimistic_Q
    all_data["suicide_Q"] = np.tile(np.nan, num_samples)
    all_data["suicide_Q"] = all_data["suicide_Q"].apply(np.array)
    all_data["suicide_Q"] = suicide_Q
    all_data["planned_hunting_Q"] = np.tile(np.nan, num_samples)
    all_data["planned_hunting_Q"] = all_data["planned_hunting_Q"].apply(np.array)
    all_data["planned_hunting_Q"] = planned_hunting_Q
    print("\n")
    print("Direction Estimation :")
    print("\n")
    print("Q value :")
    print(all_data[["global_Q", "local_Q", "optimistic_Q",
                    "pessimistic_Q", "suicide_Q", "planned_hunting_Q"]].iloc[:5])
    return all_data


def preEstimation():
    pd.options.mode.chained_assignment = None
    # Individual Estimation
    print("=" * 15, " Individual Estimation ", "=" * 15)
    adjacent_data, locs_df, adjacent_path, reward_amount = _readAuxiliaryData()
    print("Finished reading auxiliary data.")
    filename_list = [
        "../common_data/1-1-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl",
        "../common_data/1-2-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl",
        "../common_data/global_data.pkl",
        "../common_data/local_data.pkl",
        "../common_data/global_testing_data.pkl",
        "../common_data/local_testing_data.pkl"
    ]
    for filename in filename_list:
        print("-" * 50)
        print(filename)
        all_data = _readData(filename)
        print("Finished reading data.")
        print("Start estimating...")
        all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount)
        with open("{}-new_agent.pkl".format(filename), "wb") as file:
            pickle.dump(all_data, file)
    pd.options.mode.chained_assignment = "warn"


# ===================================
#         FAST OPTIMIZATION
# ===================================
def _preProcessingQ(Q_value, last_dir, randomness_coeff = 1.0):
    '''
    Preprocessing for Q-value, including convert negative utility to non-negative, set utilities of unavailable 
    directions to -inf, and normalize utilities.
    :param Q_value: 
    :return: 
    '''
    num_samples = Q_value.shape[0]
    temp_Q = []
    unavailable_index = []
    available_index = []
    for index in range(num_samples):
        cur_Q = Q_value.iloc[index]
        unavailable_index.append(np.where(cur_Q == 0))
        available_index.append(np.where(cur_Q != 0))
        # Add randomness and laziness
        Q_scale = scaleOfNumber(np.max(np.abs(cur_Q)))
        randomness = np.random.normal(loc=0, scale=0.1, size=len(available_index[index][0])) * Q_scale
        Q_value.iloc[index][available_index[index]] = (
            Q_value.iloc[index][available_index[index]]
            + randomness_coeff * randomness
        )# randomness
        if last_dir[index] is not None and dir_list.index(last_dir[index]) in available_index[index][0]:
            Q_value.iloc[index][dir_list.index(last_dir[index])] = (
                Q_value.iloc[index][dir_list.index(last_dir[index])]
                + Q_scale
            )  # laziness
        temp_Q.extend(Q_value.iloc[index])
    # Convert  negative to non-negative
    offset = 0.0
    if np.any(np.array(temp_Q) < 0):
        offset = np.min(temp_Q)
    for index in range(num_samples):
        Q_value.iloc[index][available_index[index]] = Q_value.iloc[index][available_index[index]] - offset + 1
    # Normalizing
    normalizing_factor = np.nanmax(temp_Q)
    normalizing_factor = 1 if 0 == normalizing_factor else normalizing_factor
    # Set unavailable directions
    for index in range(num_samples):
        # cur_Q = Q_value.iloc[index]
        # unavailable_index = np.where(cur_Q == 0)
        Q_value.iloc[index] = Q_value.iloc[index] / normalizing_factor
        # Q_value.iloc[index_][unavailable_index[index_]] = -999
        Q_value.iloc[index][unavailable_index[index]] = 0
    return (offset, normalizing_factor, Q_value)


def negativeLikelihood(param, all_data, true_prob, agents_list, return_trajectory = False):
    '''
    Estimate agent weights with utility (Q-value).
    :param param: 
    :param all_data: 
    :param agent_list: 
    :param return_trajectory: 
    :return: 
    '''
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute estimation error
    nll = 0  # negative log likelihood
    num_samples = all_data.shape[0]
    agents_list = ["{}_Q".format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][each_agent]
    dir_Q_value = agent_Q_value @ agent_weight
    true_dir = true_prob.apply(lambda x: makeChoice(x)).values
    # true_dir = np.array([makeChoice(dir_Q_value[each]) if not np.isnan(dir_Q_value[each][0]) else -1 for each in range(num_samples)])
    exp_prob = np.exp(dir_Q_value)
    for each_sample in range(num_samples):
        # In computing the Q-value, divided-by-zero might exists when normalizing the Q
        # TODO: fix this in  Q-value computing
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
        nll = nll -log_likelihood
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def MLE(config):
    print("=" * 20, " MLE ", "=" * 20)
    print("Agent List :", config["agents"])
    # Load experiment data
    all_data, true_prob = readDatasetFromPkl(config["data_filename"], only_necessary=config["only_necessary"])
    feasible_data_index = np.where(
        all_data["{}_Q".format(config["agents"][0])].apply(lambda x: not isinstance(x, float))
    )[0]
    all_data = all_data.iloc[feasible_data_index]
    true_prob = true_prob.iloc[feasible_data_index]
    print("Number of samples : ", all_data.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = all_data.shape[0]
    else:
        num_samples = all_data.shape[0] if config["clip_samples"] > all_data.shape[0] else config["clip_samples"]
    all_data = all_data.iloc[:num_samples]
    true_prob = true_prob.iloc[:num_samples]
    # pre-processing of Q-value
    agent_normalizing_factors = []
    agent_offset = []
    for agent_name in ["{}_Q".format(each) for each in config["agents"]]:
        preprocessing_res = _preProcessingQ(all_data[agent_name], last_dir = all_data.pacman_dir.values, randomness_coeff = 1.0)
        agent_offset.append(preprocessing_res[0])
        agent_normalizing_factors.append(preprocessing_res[1])
        all_data[agent_name] = preprocessing_res[2]
    print("Number of used samples : ", all_data.shape[0])
    print("Agent Normalizing Factors : ", agent_normalizing_factors)
    print("Agent Offset : ", agent_offset)
    # Optimization
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)

    # Notes [Jiaqi Aug. 13]: -- about the lambda function --
    # params = [0, 0, 0, 0]
    # func = lambda parameter: func() [WRONG]
    # func = lambda params: func() [CORRECT]
    func = lambda params: negativeLikelihood(
        params,
        all_data,
        true_prob,
        config["agents"],
        return_trajectory = False
    )
    is_success = False
    retry_num = 0
    while not is_success and retry_num < config["maximum_try"]:
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds, # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints = cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print("Normalized Parameter (res / sum(res)): ", res.x / np.sum(res.x))
    print("Message : ", res.message)
    # Estimation
    testing_data, testing_true_prob = readTestingDatasetFromPkl(
        config["testing_data_filename"],
        only_necessary=config["only_necessary"])
    # not_nan_index = [each for each in not_nan_index if ]
    print("Testing data num : ", testing_data.shape[0])
    _, estimated_prob = negativeLikelihood(
        res.x,
        testing_data,
        testing_true_prob,
        config["agents"],
        return_trajectory = True
    )
    true_dir = np.array([np.argmax(each) for each in testing_true_prob])
    # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate on testing data: ", correct_rate / len(testing_true_prob))


def movingWindowAnalysis(config, save_res = True):
    print("=" * 20, " Moving Window ", "=" * 20)
    print("Agent List :", config["agents"])
    window = config["window"]
    # Load experiment data
    X, Y = readDatasetFromPkl(config["data_filename"], only_necessary = config["only_necessary"])
    print("Number of samples : ", X.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = X.shape[0]
    else:
        num_samples = X.shape[0] if config["clip_samples"] > X.shape[0] else config["clip_samples"]
    X = X.iloc[:num_samples]
    Y = Y.iloc[:num_samples]
    # pre-processing of Q-value
    agent_normalizing_factors = []
    agent_offset = []
    for agent_name in ["{}_Q".format(each) for each in config["agents"]]:
        preprocessing_res = _preProcessingQ(X[agent_name], X.pacman_dir.values, randomness_coeff=1.0)
        agent_offset.append(preprocessing_res[0])
        agent_normalizing_factors.append(preprocessing_res[1])
        X[agent_name] = preprocessing_res[2]
    print("Number of used samples : ", X.shape[0])
    print("Agent Normalizing Factors : ", agent_normalizing_factors)
    print("Agent Offset : ", agent_offset)
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
    func = lambda params: negativeLikelihood(
        params,
        sub_X,
        sub_Y,
        config["agents"],
        return_trajectory=False
    )
    subset_index = np.arange(window, len(Y) - window)
    all_coeff = []
    all_correct_rate = []
    all_success = []
    at_cross_index = np.where(X.at_cross.values)[0]
    at_cross_accuracy = []
    # Moving the window
    for index in subset_index:
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
        _, estimated_prob = negativeLikelihood(
            res.x,
            sub_X,
            sub_Y,
            config["agents"],
            return_trajectory = True
        )
        # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
        true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
        # Coefficient and accuracy for decision point
        if index in at_cross_index:
            at_cross_accuracy.append((index, res.x, correct_rate))
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    print("Average Correct Rate (Decision Position): {}".format(np.mean([each[2] for each in at_cross_accuracy])))
    # Save estimated agent weights
    if save_res:
        type = "_".join(config['agents'])
        np.save("{}-agent_weight-window{}-{}-new_agent.npy".format(config["method"], window, type), all_coeff)
        np.save("{}-is_success-window{}-{}-new_agent.npy".format(config["method"], window, type), all_success)
        np.save("{}-at_cross_accuracy-window{}-{}-new_agent.npy".format(config["method"], window, type), at_cross_accuracy)


# ===================================
#         VISUALIZATION
# ===================================
def plotWeightVariation(all_agent_weight, window, is_success = None):
    # Determine agent names
    agent_name = ["Global", "Local", "Optimistic", "Pessimistic", "Suicide", "Planned Hunting"]
    agent_color = ["red", "blue", "green", "cyan", "magenta", "black"]
    # Plot weight variation
    all_coeff = np.array(all_agent_weight)
    if is_success is not None:
        for index in range(1, is_success.shape[0]):
            if not is_success[index]:
                all_agent_weight[index] = all_agent_weight[index - 1]
    # Noamalize
    # all_coeff = all_coeff / np.max(all_coeff)
    for index in range(all_coeff.shape[0]):
        all_coeff[index] = all_coeff[index] / np.sum(all_coeff[index])
        # all_coeff[index] = all_coeff[index] / np.linalg.norm(all_coeff[index])
    for index in range(6):
        plt.plot(all_coeff[:, index], color = agent_color[index], ms = 3, lw = 5,label = agent_name[index])
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize = 15)
    plt.xlim(0, all_coeff.shape[0] - 1)
    x_ticks = list(range(0, all_coeff.shape[0], 10))
    if (all_coeff.shape[0] - 1) not in x_ticks:
        x_ticks.append(all_coeff.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.xlabel("Time Step", fontsize = 20)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, ncol=6)
    plt.show()





if __name__ == '__main__':
    # # Pre-estimation
    preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None
    config = {
        # Filename
        "data_filename": "../common_data/1-1-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl-new_agent.pkl",
        # Testing data filename
        "testing_data_filename": "../common_data/1-1-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl-new_agent.pkl",
        # Method: "MLE" or "MEE"
        "method": "MLE",
        # Only making decisions when necessary
        "only_necessary": False,
        # The number of samples used for estimation: None for using all the data
        "clip_samples": None,
        # The window size
        "window": 10,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try": 5,
        # Loss function (required when method = "MEE"): "l2-norm" or "cross-entropy"
        "loss-func": "l2-norm",
        # Initial guess of parameters
        "params": [1, 1, 1, 1, 1, 1],
        # Bounds for optimization
        "bounds": [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]], # TODO: the bound...
        # Agents: at least one of "global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting".
        "agents": ["global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting"],
    }

    # ============ ESTIMATION =============
    # MLE(config)

    # ============ MOVING WINDOW =============
    # movingWindowAnalysis(config, save_res = True)

    # ============ PLOTTING =============
    # Load the log of moving window analysis; log files are created in the analysis
    # agent_weight = np.load("MLE-agent_weight-window10-global_local_optimistic_pessimistic_suicide_planned_hunting-new_agent.npy")
    # is_success = np.load("MLE-is_success-window10-global_local_optimistic_pessimistic_suicide_planned_hunting-new_agent.npy")
    # plotWeightVariation(agent_weight, config["window"], is_success)