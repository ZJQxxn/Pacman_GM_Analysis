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

import sys
sys.path.append("./")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath
from PathTreeConstructor import PathTree, OptimisticAgent, PessimisticAgent
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent
from SuicideAgent import SuicideAgent
from PlannedHuntingAgent import PlannedHuntingAgent
from MultiAgentMLEAnalysis import oneHot, makeChoice, readDatasetFromPkl, readTestingDatasetFromPkl
from MultiAgentMLEAnalysis import plotWeightVariation

# ===================================
#       INDIVIDUAL ESTIMATION
# ===================================
def _readData(filename):
    with open(filename, "rb") as file:
        # file.seek(0) # deal with the error that "could not find MARK"
        all_data = pickle.load(file)
    all_data = all_data.reset_index()

    return all_data


def _readAuxiliaryData():
    # Load pre-computed data
    adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    adjacent_path = readAdjacentPath("extracted_data/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    return adjacent_data, locs_df, adjacent_path, reward_amount


def _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount):
    # Configuration (for global agent)
    global_depth = 15
    ignore_depth = 5
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    # Configuration (for local agent)
    local_depth = 5
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    # Configuration (for optimistic agent)
    optimistic_depth = 10
    optimistic_ghost_attractive_thr = 34
    optimistic_fruit_attractive_thr = 34
    optimistic_ghost_repulsive_thr = 12
    # Configuration (for pessimistic agent)
    pessimistic_depth = 10
    pessimistic_ghost_attractive_thr = 34
    pessimistic_fruit_attractive_thr = 34
    pessimistic_ghost_repulsive_thr = 12
    # Configuration (for suicide agent)
    suicide_depth = 15
    suicide_ghost_attractive_thr = 34
    suicide_fruit_attractive_thr = 34
    suicide_ghost_repulsive_thr = 12
    # Configuration (for lazy, random)
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Direction sstimation
    global_estimation = []
    local_estimation = []
    lazy_estimation = []
    random_estimation = []
    optimistic_estimation = []
    pessimistic_estimation = []
    suicide_estimation = []
    planned_hunting_estimation = []
    # Q-value (utility)
    global_Q = []
    local_Q = []
    lazy_Q = []
    random_Q = []
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
            random_Q.append([0.0, 0.0, 0.0, 0.0])
            lazy_Q.append([0.0, 0.0, 0.0, 0.0])
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
            depth=global_depth,
            ignore_depth=ignore_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr
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
            depth=local_depth,
            ghost_attractive_thr=local_ghost_attractive_thr,
            fruit_attractive_thr=local_fruit_attractive_thr,
            ghost_repulsive_thr=local_ghost_repulsive_thr
        )
        local_result = local_agent.nextDir(return_Q=True)
        local_estimation.append(local_result[0])
        local_Q.append(local_result[1])
        # Optimistic agent
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
            depth=optimistic_depth,
            ghost_attractive_thr=optimistic_ghost_attractive_thr,
            fruit_attractive_thr=optimistic_fruit_attractive_thr,
            ghost_repulsive_thr=optimistic_ghost_repulsive_thr
        )
        optimistic_result = optimistic_agent.nextDir(return_Q=True)
        optimistic_estimation.append(optimistic_result[0])
        optimistic_Q.append(optimistic_result[1])
        # Pessimistic agent
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
            depth=pessimistic_depth,
            ghost_attractive_thr=pessimistic_ghost_attractive_thr,
            fruit_attractive_thr=pessimistic_fruit_attractive_thr,
            ghost_repulsive_thr=pessimistic_ghost_repulsive_thr
        )
        pessimistic_result = pessimistic_agent.nextDir(return_Q=True)
        pessimistic_estimation.append(pessimistic_result[0])
        pessimistic_Q.append(pessimistic_result[1])
        # Lazy agent
        lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir[index])
        lazy_result = lazy_agent.nextDir(return_Q=True)
        lazy_estimation.append(lazy_result[0])
        lazy_Q.append(lazy_result[1])
        # Random agent
        random_agent = RandomAgent(adjacent_data, cur_pos, last_dir[index], None)
        random_result = random_agent.nextDir(return_Q=True)
        random_estimation.append(random_result[0])
        random_Q.append(random_result[1])
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
            fruit_attractive_thr = suicide_ghost_repulsive_thr
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
            ghost_status
        )
        planned_hunting_result = planned_hunting_agent.nextDir(return_Q=True)
        planned_hunting_estimation.append(planned_hunting_result[0])
        planned_hunting_Q.append(planned_hunting_result[1])
    # Assign new columns
    # TODO: simplify; directly assign a vector to cells of the pd.DataFrame
    print("Estimation length : ", len(global_estimation))
    print("Data Shape : ", all_data.shape)
    all_data["global_estimation"] = np.tile(np.nan, num_samples)
    all_data["global_estimation"][estimated_index] = global_estimation
    all_data["global_Q"] = np.tile(np.nan, num_samples)
    all_data["global_Q"] = all_data["global_Q"].apply(np.array)
    all_data["global_Q"] = global_Q
    all_data["local_estimation"] = np.tile(np.nan, num_samples)
    all_data["local_estimation"][estimated_index] = local_estimation
    all_data["local_Q"] = np.tile(np.nan, num_samples)
    all_data["local_Q"] = all_data["local_Q"].apply(np.array)
    all_data["local_Q"] = local_Q
    all_data["optimistic_estimation"] = np.tile(np.nan, num_samples)
    all_data["optimistic_estimation"][estimated_index] = optimistic_estimation
    all_data["optimistic_Q"] = np.tile(np.nan, num_samples)
    all_data["optimistic_Q"] = all_data["optimistic_Q"].apply(np.array)
    all_data["optimistic_Q"] = optimistic_Q
    all_data["pessimistic_estimation"] = np.tile(np.nan, num_samples)
    all_data["pessimistic_estimation"][estimated_index] = pessimistic_estimation
    all_data["pessimistic_Q"] = np.tile(np.nan, num_samples)
    all_data["pessimistic_Q"] = all_data["pessimistic_Q"].apply(np.array)
    all_data["pessimistic_Q"] = pessimistic_Q
    all_data["lazy_estimation"] = np.tile(np.nan, num_samples)
    all_data["lazy_estimation"][estimated_index] = lazy_estimation
    all_data["lazy_Q"] = np.tile(np.nan, num_samples)
    all_data["lazy_Q"] = all_data["lazy_Q"].apply(np.array)
    all_data["lazy_Q"] = lazy_Q
    all_data["random_estimation"] = np.tile(np.nan, num_samples)
    all_data["random_estimation"][estimated_index] = random_estimation
    all_data["random_Q"] = np.tile(np.nan, num_samples)
    all_data["random_Q"] = all_data["random_Q"].apply(np.array)
    all_data["random_Q"] = random_Q
    all_data["suicide_estimation"] = np.tile(np.nan, num_samples)
    all_data["suicide_estimation"][estimated_index] = suicide_estimation
    all_data["suicide_Q"] = np.tile(np.nan, num_samples)
    all_data["suicide_Q"] = all_data["suicide_Q"].apply(np.array)
    all_data["suicide_Q"] = suicide_Q
    all_data["planned_hunting_estimation"] = np.tile(np.nan, num_samples)
    all_data["planned_hunting_estimation"][estimated_index] = planned_hunting_estimation
    all_data["planned_hunting_Q"] = np.tile(np.nan, num_samples)
    all_data["planned_hunting_Q"] = all_data["planned_hunting_Q"].apply(np.array)
    all_data["planned_hunting_Q"] = planned_hunting_Q
    print("\n")
    print("Direction Estimation :")
    print(all_data[["global_estimation", "local_estimation", "optimistic_estimation",
                    "pessimistic_estimation", "lazy_estimation", "random_estimation",
                    "suicide_estimation", "planned_hunting_estimation"]].iloc[:5])
    print("\n")
    print("Q value :")
    print(all_data[["global_Q", "local_Q", "optimistic_Q",
                    "pessimistic_Q", "lazy_Q", "random_Q",
                    "suicide_Q", "planned_hunting_Q"]].iloc[:5])
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
        with open("{}-with_estimation.pkl".format(filename), "wb") as file:
            pickle.dump(all_data, file)
    pd.options.mode.chained_assignment = "warn"


# ===================================
#         FAST OPTIMIZATION
# ===================================
def _preProcessingQ(Q_value):
    '''
    Preprocessing for Q-value, including convert negative utility to non-negative, set utilities of unavailable 
    directions to -inf, and normalize utilities.
    :param Q_value: 
    :return: 
    '''
    num_samples = Q_value.shape[0]
    temp_Q = []
    unavailable_index = []
    # Convert negative to non-negative
    for index in range(num_samples):
        cur_Q = Q_value.iloc[index]
        unavailable_index.append(np.where(cur_Q == 0))
        if np.any(cur_Q < 0):
            available_index = np.where(cur_Q != 0)
            Q_value.iloc[index][available_index] = Q_value.iloc[index][available_index] - np.min(Q_value.iloc[index][available_index]) + 1
        temp_Q.extend(Q_value.iloc[index])
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
    return (normalizing_factor, Q_value)


def estimationError(param, loss_func, all_data, true_prob, agents_list, return_trajectory = False):
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    true_prob = np.array([[i for i in each]
                          for each in (true_prob.values
                                       if isinstance(true_prob, pd.DataFrame) or isinstance(true_prob, pd.Series)
                                       else true_prob)
                          ])
    # Compute estimation error
    ee = 0  # estimation error
    num_samples = all_data.shape[0]
    agents_list = ["{}_estimation".format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    one_hot_estimation = np.zeros((num_samples, 4, len(agents_list)))
    for index in range(num_samples):
        for i in range(len(agents_list)):
            one_hot_estimation[index, :, i] = oneHot(pre_estimation[index, i])
    estimation_prob_trajectory = one_hot_estimation @ agent_weight
    if "l2-norm" == loss_func:
        error = np.linalg.norm(estimation_prob_trajectory - true_prob)
    elif "cross-entropy" == loss_func:
        error = log_loss(true_prob, estimation_prob_trajectory)
    else:
        raise ValueError("Undefined loss function {}!".format(loss_func))
    ee += error
    if not return_trajectory:
        return ee
    else:
        return (ee, estimation_prob_trajectory)


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
    # true_dir = true_prob.apply(lambda x: np.argmax(x)).values
    true_dir = np.array([makeChoice(dir_Q_value[each]) if not np.isnan(dir_Q_value[each][0]) else -1 for each in range(num_samples)])
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


def MEE(config):
    print("=" * 20, " MEE ", "=" * 20)
    print("Agent List :", config["agents"])
    # Load experiment data
    all_data, true_prob = readDatasetFromPkl(config["data_filename"], only_necessary = config["only_necessary"])
    feasible_data_index = np.where(
        all_data["{}_estimation".format(config["agents"][0])].apply(lambda x: not isinstance(x, float))
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
    # Notes [Jiaqi Aug. 13]: -- about the lambda function --
    # params = [0, 0, 0, 0]
    # func = lambda parameter: func() [WRONG]
    # func = lambda params: func() [CORRECT]
    func = lambda params: estimationError(
        params,
        config["loss-func"],
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
    print("Message : ", res.message)
    # Estimation
    testing_data, testing_true_prob = readTestingDatasetFromPkl(
        config["testing_data_filename"],
        only_necessary = config["only_necessary"])
    # not_nan_index = [each for each in not_nan_index if ]
    print("Testing data num : ", testing_data.shape[0])
    _, estimated_prob = estimationError(
        res.x,
        config["loss-func"],
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
    for agent_name in ["{}_Q".format(each) for each in config["agents"]]:
        preprocessing_res = _preProcessingQ(all_data[agent_name])
        agent_normalizing_factors.append(preprocessing_res[0])
        all_data[agent_name] = preprocessing_res[1]
    print("Number of used samples : ", all_data.shape[0])
    print("Agent Normalizing Factors : ", agent_normalizing_factors)
    # TODO: some bad data; np.nan or np.inf in the Q-value vector
    # Optimization
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: sum(x) - 1})

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
            # bounds=bounds, # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            # constraints=cons
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


def movingWindowAnalysis(config):
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
        func = lambda params: estimationError(
            params,
            config["loss-func"],
            sub_X,
            sub_Y,
            config["agents"],
            return_trajectory=False
        )
    elif "MLE" == config["method"]:
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
        if "MEE" == config["method"]:
            _, estimated_prob = estimationError(
                res.x,
                config["loss-func"],
                sub_X,
                sub_Y,
                config["agents"],
                return_trajectory = True
            )
        elif "MLE" == config["method"]:
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
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # Save estimated agent weights
    type = "_".join(config['agents'])
    np.save("{}-agent_weight-window{}-{}.npy".format(config["method"], window, type), all_coeff)
    np.save("{}-is_success-window{}-{}.npy".format(config["method"], window, type), all_success)


if __name__ == '__main__':
    # # Pre-estimation
    # preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None
    config = {
        # Filename
        "data_filename": "../common_data/global_data.pkl-with_estimation.pkl",
        # Testing data filename
        "testing_data_filename": "../common_data/global_testing_data.pkl-with_estimation.pkl",
        # Method: "MLE" or "MEE"
        "method": "MEE",
        # Only making decisions when necessary
        "only_necessary": True,
        # The number of samples used for estimation: None for using all the data
        "clip_samples": None,
        # The window size
        "window": 10,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try": 5,
        # Loss function (required when method = "MEE"): "l2-norm" or "cross-entropy"
        "loss-func": "l2-norm",
        # Initial guess of parameters
        "params": [0.0, 0.0, 0.0],
        # Bounds for optimization
        "bounds": [[0, 1], [0, 1], [0, 1]],
        # Agents: at least one of "global", "local", "lazy", "random", "optimistic", "pessimistic", "suicide", "planned_hunting".
        "agents": ["global", "local", "random"],
    }

    # ============ ESTIMATION =============
    # MEE(config)
    MLE(config)

    # ============ MOVING WINDOW =============
    # movingWindowAnalysis(config)

    # ============ PLOTTING =============
    # # Load the log of moving window analysis; log files are created in the analysis
    # agent_weight = np.load("MEE-agent_weight-window10-global_local_lazy_random.npy")
    # is_success = np.load("MEE-is_success-window10-global_local_lazy_random.npy")
    # plotWeightVariation(agent_weight, config["agents"], config["window"], is_success,
    #                     plot_label = True, filename = config["data_filename"])