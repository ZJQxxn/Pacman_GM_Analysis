'''
Description:
    Fast MLE analysis with pre-computed estimations of every agent on all th trials.
    
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
from sklearn.model_selection import KFold
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


#TODO: split training and testing

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
    # Randomness and laziness
    randomness_coeff = 1.0
    laziness_coeff = 1.0
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
    # Configuration (last direction)
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Direction estimation
    global_estimation = []
    local_estimation = []
    pessimistic_estimation = []
    suicide_estimation = []
    planned_hunting_estimation = []
    # Q-value (utility)
    global_Q = []
    local_Q = []
    pessimistic_Q = []
    suicide_Q = []
    planned_hunting_Q = []
    num_samples = all_data.shape[0]
    print("Sample Num : ", num_samples)
    estimated_index = []
    for index in range(num_samples):
        if 0 == (index + 1) % 100:
            print("Finished estimation at {}".format(index + 1))
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        # In case the Pacman position does not exists, e.g. (0, 18)
        if cur_pos not in adjacent_data:
            global_Q.append([0.0, 0.0, 0.0, 0.0])
            local_Q.append([0.0, 0.0, 0.0, 0.0])
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
            laziness_coeff = laziness_coeff,
            reward_coeff=1.0,
            risk_coeff=0.0
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
            laziness_coeff = laziness_coeff,
            reward_coeff=1.0,
            risk_coeff=0.0
        )
        local_result = local_agent.nextDir(return_Q=True)
        local_estimation.append(local_result[0])
        local_Q.append(local_result[1])
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
    print(all_data[["global_Q", "local_Q", "pessimistic_Q", "suicide_Q", "planned_hunting_Q"]].iloc[:5])
    return all_data


def preEstimation():
    pd.options.mode.chained_assignment = None
    # Individual Estimation
    print("=" * 15, " Individual Estimation ", "=" * 15)
    adjacent_data, locs_df, adjacent_path, reward_amount = _readAuxiliaryData()
    print("Finished reading auxiliary data.")
    filename_list = [
        "../common_data/partial_data_with_reward_label_cross.pkl",
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
    available_index = []
    for index in range(num_samples):
        cur_Q = Q_value.iloc[index]
        unavailable_index.append(np.where(cur_Q == 0))
        available_index.append(np.where(cur_Q != 0))
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
        Q_value.iloc[index] = Q_value.iloc[index] / normalizing_factor
        Q_value.iloc[index][unavailable_index[index]] = 0
    return (offset, normalizing_factor, Q_value)

def _beanNumVSCorrectRate(data, true_dir, estimation_dir):
    data = data[["file", "origin_index", "beans"]]
    data["beans_num"] = data.beans.apply(lambda  x: len(x) if not isinstance(x, float) else 0)
    # data["true_dir"] = true_prob.apply(lambda x: makeChoice(x))
    # data["estimated_dir"] = estimation.apply(lambda x: makeChoice(x))
    data["true_dir"] = true_dir
    data["estimated_dir"] = estimation_dir
    data["is_correct"] = data[["true_dir", "estimated_dir"]].apply(lambda x: x.true_dir == x.estimated_dir, axis = 1)
    return data


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
    exp_prob = np.exp(dir_Q_value)
    for each_sample in range(num_samples):
        # In computing the Q-value, divided-by-zero might exists when normalizing the Q
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
    agent_type = "_".join(config["agents"])
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
        preprocessing_res = _preProcessingQ(all_data[agent_name])
        agent_offset.append(preprocessing_res[0])
        agent_normalizing_factors.append(preprocessing_res[1])
        all_data[agent_name] = preprocessing_res[2]
    print("Finished preprocessing!")
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
    # 5-fold cross-validation
    fold_weight = []
    fold_accuracy = []
    # Split into training set and testing set (1:1)
    if "level_0" in all_data.columns.values:
        all_data = all_data.drop(columns = ["level_0"])
    all_data = all_data.reset_index()
    training_index, testing_index = list(KFold(n_splits=2, shuffle=True).split(all_data))[0]
    training_data = all_data.iloc[training_index]
    training_true_prob = true_prob.iloc[training_index]
    testing_data = all_data.iloc[testing_index]
    testing_true_prob = true_prob.iloc[testing_index]
    if "level_0" in training_data.columns.values:
        training_data = training_data.drop(columns = ["level_0"])
    # if "level_0" in all_data.columns.values:
    #     training_true_prob = training_true_prob.drop(columns = ["level_0"])
    if "level_0" in testing_data.columns.values:
        testing_data = testing_data.drop(columns = ["level_0"])
    # if "level_0" in testing_true_prob.columns.values:
    #     testing_true_prob = testing_true_prob.drop(columns = ["level_0"])
    training_data = training_data.reset_index()
    training_true_prob = training_true_prob.reset_index().next_pacman_dir_fill
    testing_data = testing_data.reset_index()
    testing_true_prob = testing_true_prob.reset_index().next_pacman_dir_fill

    # 5-fold cross-validation
    for training_index, validation_index in KFold(n_splits=5).split(training_data):
        print("-" * 30)
        # Notes [Jiaqi Aug. 13]: -- about the lambda function --
        # params = [0, 0, 0, 0]
        # func = lambda parameter: func() [WRONG]
        # func = lambda params: func() [CORRECT]
        func = lambda params: negativeLikelihood(
            params,
            training_data.iloc[training_index],
            training_true_prob.iloc[training_index],
            config["agents"],
            return_trajectory=False
        )
        is_success = False
        retry_num = 0
        while not is_success and retry_num < config["maximum_try"]:
            res = scipy.optimize.minimize(
                func,
                x0=params,
                method="SLSQP",
                bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
                tol=1e-5,
                constraints=cons
            )
            is_success = res.success
            if not is_success:
                retry_num += 1
                print("Failed, retrying...")
        print("Initial guess : ", params)
        print("Estimated Parameter : ", res.x)
        print("Normalized Parameter (res / sum(res)): ", res.x / np.sum(res.x))
        print("Message : ", res.message)
        # Estimation on validation
        _, estimated_prob = negativeLikelihood(
            res.x,
            training_data.iloc[validation_index],
            training_true_prob.iloc[validation_index],
            config["agents"],
            return_trajectory=True
        )
        true_dir = np.array([makeChoice(each) for each in training_true_prob.iloc[validation_index]])
        # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == true_dir)
        correct_rate = correct_rate / len(true_dir)
        print("Correct rate on validation data: ", correct_rate)
        fold_weight.append(res.x)
        fold_accuracy.append(correct_rate)
    # The weight with the highest correct rate
    best = np.argmax(fold_accuracy).item()
    print("="*25)
    print("Best Weight : ", fold_weight[best])
    print("Best accuracy : ", fold_accuracy[best])
    # =========================================================
    # Estimation on the whole training data
    _, estimated_prob = negativeLikelihood(
        fold_weight[best],
        training_data,
        training_true_prob,
        config["agents"],
        return_trajectory=True
    )
    true_dir = np.array([makeChoice(each) for each in training_true_prob])
    estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    correct_rate = correct_rate / len(true_dir)
    print("Correct rate on the whole training data: ", correct_rate)
    with open("training_result-{}.pkl".format(agent_type), "wb") as file:
        pickle.dump(_beanNumVSCorrectRate(training_data, true_dir, estimated_dir), file)
    # Estimation on testing data
    _, estimated_prob = negativeLikelihood(
        fold_weight[best],
        testing_data,
        testing_true_prob,
        config["agents"],
        return_trajectory=True
    )
    true_dir = np.array([makeChoice(each) for each in testing_true_prob])
    # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    correct_rate = correct_rate / len(true_dir)
    print("Correct rate on the whole testing data: ", correct_rate)
    with open("testing_result-{}.pkl".format(agent_type), "wb") as file:
        pickle.dump(_beanNumVSCorrectRate(testing_data, true_dir, estimated_dir), file)
    np.save("weight-{}.npy".format(agent_type), fold_weight[best])



# ===================================
#         VISUALIZATION
# ===================================
def plotBeanVSAccuracy(training_result_file, testing_result_file):
    # Read data
    with open(training_result_file, "rb") as file:
        training_result = pickle.load(file)
    with open(testing_result_file, "rb") as file:
        testing_result = pickle.load(file)
    # Plot correct rate vs. number of beans
    # for training result
    plt.subplot(1, 2, 1)
    plt.title("Training Set")
    max_bean_num = np.max(training_result.beans_num)
    min_bean_num = np.min(training_result.beans_num)
    print("Min Bean Num : ", min_bean_num)
    print("Max Bean Num : ", max_bean_num)
    bean_nums = np.arange(min_bean_num, max_bean_num + 1, 1)
    # bean_nums = np.arange(0, 91, 10)
    step_nums = np.zeros_like(bean_nums)
    correct_nums = np.zeros_like(bean_nums)
    for index in range(training_result.shape[0]):
        cur_data = training_result.iloc[index]
        bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        # bean_index = cur_data.beans_num // 10
        step_nums[bean_index] += 1
        if cur_data.is_correct:
            correct_nums[bean_index] += 1
    plt.bar(bean_nums, np.divide(correct_nums, step_nums))
    plt.xlim((min_bean_num-1, max_bean_num+1))
    # for testing result
    plt.subplot(1, 2, 2)
    plt.title("Testing Set")
    max_bean_num = np.max(testing_result.beans_num)
    min_bean_num = np.min(testing_result.beans_num)
    print("Min Bean Num : ", min_bean_num)
    print("Max Bean Num : ", max_bean_num)
    bean_nums = np.arange(min_bean_num, max_bean_num + 1, 1)
    # bean_nums = np.arange(0, 91, 10)
    step_nums = np.zeros_like(bean_nums)
    correct_nums = np.zeros_like(bean_nums)
    for index in range(testing_result.shape[0]):
        cur_data = testing_result.iloc[index]
        bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        # bean_index = cur_data.beans_num // 10
        step_nums[bean_index] += 1
        if cur_data.is_correct:
            correct_nums[bean_index] += 1
    plt.bar(bean_nums, np.divide(correct_nums, step_nums))
    plt.xlim((min_bean_num - 1, max_bean_num + 1))

    plt.show()









if __name__ == '__main__':
    # # Pre-estimation
    # preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None
    config = {
        # Filename
        "data_filename": "../common_data/partial_data_with_reward_label_cross.pkl-new_agent.pkl",
        # Testing data filename
        "testing_data_filename": "../common_data/partial_data_with_reward_label_cross.pkl-new_agent.pkl",
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
        # "params": [1, 1, 1, 1, 1],
        "params": [1, 1, 1],
        # Bounds for optimization
        # "bounds": [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]],
        "bounds": [[0, 1000], [0, 1000], [0, 1000]],
        # Agents: at least one of "global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting".
        # "agents": ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        "agents": ["global", "local", "pessimistic"],
    }

    # ============ ESTIMATION =============
    # MLE(config)

    print("Only Path Tree Agents : ")
    plotBeanVSAccuracy(
        "./altogether_analysis/training_result-global_local_pessimistic.pkl",
        "./altogether_analysis/testing_result-global_local_pessimistic.pkl"
    )

    print("All the Agents : ")
    plotBeanVSAccuracy(
        "./altogether_analysis/training_result-global_local_pessimistic_suicide_planned_hunting.pkl",
        "./altogether_analysis/testing_result-global_local_pessimistic_suicide_planned_hunting.pkl"
    )