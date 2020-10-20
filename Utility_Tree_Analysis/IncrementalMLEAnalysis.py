'''
Description:
    Fast MLE analysis with pre-computed estimations of every agent.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    23 Sep. 2020
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
from sklearn.model_selection import KFold


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


def readDatasetFromPkl(filename, trial_num = None, only_necessary = False):
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
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index()
    # Exclude the (0, 18) position in data
    normal_data_index = []
    for index in range(all_data.shape[0]):
        if not isinstance(all_data.global_Q[index], list): # what if no normal Q?
            normal_data_index.append(index)
    all_data = all_data.iloc[normal_data_index]
    # Select the first 100 trials
    trial_name_list = np.unique(all_data.file.values)
    print("The number of trials : ", len(trial_name_list))
    if trial_num is None:
        trial_num = len(trial_name_list)
    if len(trial_name_list) > trial_num:
        trial_name_list = np.random.choice(trial_name_list, trial_num, replace = False)
    need_index = all_data.file.apply(lambda x: x in trial_name_list)
    need_index = np.where(need_index)[0]
    all_data = all_data.iloc[need_index]
    true_prob = all_data.next_pacman_dir_fill.reset_index(drop = True)
    print("Selected number of trials : ", len(trial_name_list))
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
        if "at_cross" not in all_data.columns.values:
            adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
            all_data.at_cross = all_data.pacmanPos.apply(
                lambda x: (
                    False if x not in adjacent_data else np.sum(
                        [1 if not isinstance(each, float) else 0
                        for each in list(adjacent_data[x].values())]
                    ) > 2
                )
            )
        at_cross_index = np.where(all_data.at_cross)
        X = all_data.iloc[at_cross_index]
        Y = true_prob.iloc[at_cross_index]
        # print("--- Data Shape {} ---".format(len(at_cross_index[0])))
    else:
        X = all_data
        Y = true_prob
    return X, Y


def readTrialDataFromPkl(filename, trial_num = None):
    # Read data and pre-processing
    with open(filename, "rb") as file:
        # file.seek(0) # deal with the error that "could not find MARK"
        all_data = pickle.load(file)
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index(drop=True)
    adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
    # Exclude the (0, 18) position in data
    normal_data_index = []
    for index in range(all_data.shape[0]):
        if not isinstance(all_data.global_Q[index], list): # what if no normal Q?
            normal_data_index.append(index)
    all_data = all_data.iloc[normal_data_index]
    # Select the required number of trials
    trial_name_list = np.unique(all_data.file.values)
    print("The number of trials : ", len(trial_name_list))
    if trial_num is None:
        trial_num = len(trial_name_list)
    if len(trial_name_list) > trial_num:
        trial_name_list = np.random.choice(trial_name_list, trial_num, replace = False)
    X_list = []
    Y_list = []
    for each in trial_name_list:
        X = all_data[all_data.file == each]
        if "at_cross" not in X.columns.values:
            X.at_cross = X.pacmanPos.apply(
                lambda x: (
                    False if x not in adjacent_data else np.sum(
                        [1 if not isinstance(each, float) else 0
                         for each in list(adjacent_data[x].values())]
                    ) > 2
                )
            )
        # Fill nan direction for optimization use
        true_prob = X.next_pacman_dir_fill.reset_index(drop=True)
        start_index = 0
        while pd.isna(true_prob[start_index]):
            start_index += 1
        true_prob = true_prob[start_index:].reset_index(drop=True)
        all_data = all_data[start_index:].reset_index(drop=True)
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
        Y = true_prob
        # Append
        X_list.append(X)
        Y_list.append(Y)
    return X_list, Y_list, trial_name_list

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
    # Must be 0.0, because the normalizing is completed in the pre-processing
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
        if 0 == (index + 1) % 20:
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
            reward_coeff = 1.0,
            risk_coeff = 0.0
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
            depth = local_depth,
            ignore_depth = 0,
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
            ignore_depth = 0,
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
        # "../common_data/before_last_life_data.pkl",
        # "../common_data/last_life_data.pkl",
        # "../common_data/first_life_data.pkl",
        # "../common_data/partial_data_with_reward_label_cross.pkl",
        # "../common_data/agent_data/suicide_data.pkl",
        # "../common_data/agent_data/global_data.pkl",
        # "../common_data/agent_data/planned_hunting_data.pkl",
        "../common_data/agent_data/most_planned_hunting_data.pkl",
        "../common_data/agent_data/really_suicide_data.pkl",
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
def _preProcessingQ(Q_value, pacmanPos, adjacent_data):
    '''
    Preprocessing for Q-value, including convert negative utility to non-negative, set utilities of unavailable
    directions to -inf, and normalize utilities.
    :param Q_value:
    :return:
    '''
    dir_list = ["left", "right", "up", "down"]
    num_samples = Q_value.shape[0]
    temp_Q = []
    unavailable_index = []
    available_index = []
    for index in range(num_samples):
        # cur_Q = Q_value.iloc[index]
        temp_available = []
        temp_unavailable = []
        for dir_index, each in enumerate(dir_list):
            if not isinstance(adjacent_data[pacmanPos.iloc[index]][each], float):
                temp_available.append(dir_index)
            else:
                temp_unavailable.append(dir_index)
        unavailable_index.append(temp_unavailable)
        available_index.append(temp_available)
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


def _NumOfLocalBeans(pacman_pos, beans, dist_data):
    nearby_beans_list = []
    if beans is not None and not isinstance(beans, float):
        for each in beans:
            if tuple(each) == tuple(pacman_pos) or dist_data[tuple(pacman_pos)][tuple(each)] <= 5:
                nearby_beans_list.append(each)
    return len(nearby_beans_list)


def _rebornBeans(pacman_pos, beans, dist_data):
    # The number of beans nearby the Pacman
    nearby_beans_list = []
    if beans is not None and not isinstance(beans, float):
        for each in beans:
            if tuple(each) == tuple(pacman_pos) or dist_data[tuple(pacman_pos)][tuple(each)] <= 5:
                nearby_beans_list.append(each)
    nearby_beans_num = len(nearby_beans_list)
    # The number of beans nearby the reborn position
    reborn_beans_list = []
    reborn_pos = (14, 27)
    if beans is not None and not isinstance(beans, float):
        for each in beans:
            if tuple(each) == reborn_pos or dist_data[reborn_pos][tuple(each)] <= 5:
                reborn_beans_list.append(each)
    reborn_beans_num = len(reborn_beans_list)
    return reborn_beans_num - nearby_beans_num


def _planningReward(pacman_pos, ghosts_pos, ghost_status, energizers, dist_data):
    if energizers is None or isinstance(energizers, float) or np.all(np.array(ghost_status) >= 3) or np.all(np.array(ghost_status) == np.nan):
        return -np.inf
    P_E_distance = [dist_data[pacman_pos][each_energizer] if pacman_pos != each_energizer else 0 for each_energizer in energizers]
    PE = np.min(P_E_distance)
    closest_energizer = energizers[np.argmin(P_E_distance).item()]
    E_G_distance = [dist_data[closest_energizer][each_ghost] if closest_energizer != each_ghost else 0 for each_ghost in ghosts_pos]
    EG = np.min(E_G_distance)
    return EG-PE


def _addGameStatus(data, true_dir, estimation_dir, dist_data):
    # data = data[["file", "origin_index", "beans", "energizers", "pacmanPos", "ghost1Pos", "ghost2Pos", "energizers", "ifscared1", "ifscared2"]]
    data["beans_num"] = data.beans.apply(lambda  x: len(x) if not isinstance(x, float) else 0)
    data["beans_num_within_10"] = data[["beans", "pacmanPos"]].apply(
        lambda x: _NumOfLocalBeans(x.pacmanPos, x.beans, dist_data), axis = 1
    )
    data["reborn_beans_num"] = data[["beans", "pacmanPos"]].apply(
        lambda x: _rebornBeans(x.pacmanPos, x.beans, dist_data), axis = 1
    )
    data["planning_reward"] = data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2", "energizers"]].apply(
        lambda x: _planningReward(x.pacmanPos, [x.ghost1Pos, x.ghost2Pos], [x.ifscared1, x.ifscared2], x.energizers, dist_data), axis=1
    )
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
    # Load experiment data
    all_data, true_prob = readDatasetFromPkl(config["data_filename"], trial_num=config["trial_num"], only_necessary=config["only_necessary"])
    print("Number of samples : ", all_data.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = all_data.shape[0]
    else:
        num_samples = all_data.shape[0] if config["clip_samples"] > all_data.shape[0] else config["clip_samples"]
    all_data = all_data.iloc[:num_samples]
    true_prob = true_prob.iloc[:num_samples]
    # pre-processing of Q-value
    adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
    dist_data = readLocDistance("extracted_data/dij_distance_map.csv")
    agent_normalizing_factors = []
    agent_offset = []
    for agent_name in ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]:
        preprocessing_res = _preProcessingQ(all_data[agent_name], all_data.pacmanPos, adjacent_data)
        agent_offset.append(preprocessing_res[0])
        agent_normalizing_factors.append(preprocessing_res[1])
        all_data[agent_name] = preprocessing_res[2]
    print("Finished preprocessing!")
    print("Number of used samples : ", all_data.shape[0])
    print("Agent Normalizing Factors : ", agent_normalizing_factors)
    print("Agent Offset : ", agent_offset)
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
    diff_agent_list = [
        ["local", "pessimistic"],
        ["local", "pessimistic", "global"],
        ["local", "pessimistic", "global", "planned_hunting"],
        ["local", "pessimistic", "global", "suicide"] #TODO: !
    ]
    accuracy_list = []
    for agent_list in diff_agent_list:
        print("Agent List :", agent_list)
        agent_type = "_".join(agent_list)
        # Optimization
        params = [1 for _ in range(len(agent_list))]
        bounds = [[0, 1000] for _ in range(len(agent_list))]
        cons = []  # construct the bounds in the form of constraints
        for par in range(len(bounds)):
            l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
            u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
            cons.append(l)
            cons.append(u)
        # 5-fold cross-validation
        fold_weight = []
        fold_accuracy = []
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
                agent_list,
                return_trajectory=False
            )
            is_success = False
            retry_num = 0
            while not is_success and retry_num < config["maximum_try"]:
                res = scipy.optimize.minimize(
                    func,
                    x0=params,
                    method="SLSQP",
                    bounds=bounds,
                    # exclude bounds and cons because the Q-value has different scales for different agents
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
                agent_list,
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
        print("=" * 25)
        print("Best Weight : ", fold_weight[best])
        print("Best accuracy : ", fold_accuracy[best])
        # =========================================================
        # Estimation on the whole training data
        _, estimated_prob = negativeLikelihood(
            fold_weight[best],
            training_data,
            training_true_prob,
            agent_list,
            return_trajectory=True
        )
        true_dir = np.array([makeChoice(each) for each in training_true_prob])
        estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == true_dir)
        correct_rate = correct_rate / len(true_dir)
        print("Correct rate on the whole training data: ", correct_rate)
        print("-"*25)
        print("Start adding game status for training result...")
        with open("altogether_analysis/training_result-{}.pkl".format(agent_type), "wb") as file:
            pickle.dump(_addGameStatus(training_data, true_dir, estimated_dir, dist_data), file)
        print("Finished adding game status.")
        # Estimation on testing data
        _, estimated_prob = negativeLikelihood(
            fold_weight[best],
            testing_data,
            testing_true_prob,
            agent_list,
            return_trajectory=True
        )
        true_dir = np.array([makeChoice(each) for each in testing_true_prob])
        # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == true_dir)
        correct_rate = correct_rate / len(true_dir)
        print("Correct rate on the whole testing data: ", correct_rate)
        accuracy_list.append(correct_rate)
        print("-" * 25)
        print("Start adding game status for training result...")
        with open("altogether_analysis/testing_result-{}.pkl".format(agent_type), "wb") as file:
            pickle.dump(_addGameStatus(testing_data, true_dir, estimated_dir, dist_data), file)
        print("Finished adding game status.")
        np.save("altogether_analysis/weight-{}.npy".format(agent_type), fold_weight[best])
    # Print results for different conditions
    print("="*30)
    print("Accuracy:\n")
    for index in range(len(diff_agent_list)):
        print(diff_agent_list[index], accuracy_list[index])
    np.save("altogether_analysis/500-trials/accuracy.npy", accuracy_list)
    return diff_agent_list


def singleTrialMovingWindowAnalysis(config):
    print("=" * 20, " Moving Window ", "=" * 20)
    window = config["window"]
    agent_list = ["global", "local", "pessimistic", "planned_hunting", "suicide"]
    adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
    dist_data = readLocDistance("extracted_data/dij_distance_map.csv")
    # Load experiment data
    # X is data for a list of trials and Y is the corresponding responses
    X_list, Y_list, trial_name_list = readTrialDataFromPkl(config["data_filename"], trial_num=config["moving_window_trial_num"])
    print("Number of trials : ", len(trial_name_list))
    # for every trial
    trial_correct_rate = []
    trial_cross_correct_rate = []
    trial_sample_num = []
    for trial_index in range(len(trial_name_list)):
        print("-"*30)
        print(trial_name_list[trial_index])
        X = X_list[trial_index]
        Y = Y_list[trial_index]
        # pre-processing of Q-value
        agent_normalizing_factors = []
        agent_offset = []
        for agent_name in ["{}_Q".format(each) for each in agent_list]:
            preprocessing_res = _preProcessingQ(X[agent_name], X.pacmanPos, adjacent_data)
            agent_offset.append(preprocessing_res[0])
            agent_normalizing_factors.append(preprocessing_res[1])
            X[agent_name] = preprocessing_res[2]
        print("Number of samples : ", X.shape[0])
        trial_sample_num.append((trial_name_list[trial_index], X.shape[0]))
        print("Agent Normalizing Factors : ", agent_normalizing_factors)
        print("Agent Offset : ", agent_offset)
        # Construct optimizer
        params = [1 for _ in range(len(agent_list))]
        bounds = [[0, 1000] for _ in range(len(agent_list))]
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
            agent_list,
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
                    x0=params,
                    method="SLSQP",
                    bounds=bounds,
                    tol=1e-5,
                    constraints=cons
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
                agent_list,
                return_trajectory=True
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
        trial_correct_rate.append(np.mean(all_correct_rate))
        trial_cross_correct_rate.append(np.mean([each[2] for each in at_cross_accuracy]))
    # Overall results print
    print("="*30)
    print("Trial sample num : ", trial_sample_num)
    print("Trial correct rate summary : ", trial_correct_rate)
    print("Trial correct rate (decision point) summary : ", trial_cross_correct_rate)
    print("Overall correct rate summary : ", np.mean(trial_correct_rate))
    print("Overall correct rate (decision point) summary : ", np.mean(trial_cross_correct_rate))



# ===================================
#         VISUALIZATION
# ===================================
def plotCorrectRate(data_name = None):
    accuracy_list = np.load("altogether_analysis/500-trials/accuracy.npy")
    plt.title("Testing Set Correct Rate ({})".format(data_name), fontsize = 20)
    width = 0.8
    plt.bar(np.arange(len(accuracy_list)), accuracy_list, width = width, align = "center")
    x = np.arange(len(accuracy_list))
    plt.xticks(x, ["local + pessimistic", "+ global", "+ planned hunting", "+ suicide"], fontsize = 20)
    plt.yticks(np.arange(0, 1.1, 0.2), [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
    plt.ylabel("Correct Rate", fontsize = 20)
    for index in range(len(accuracy_list)):
        plt.text(x = x[index]  , y = accuracy_list[index] + 0.01, s = "{s:.2f}".format(s = accuracy_list[index]),
                 fontsize = 20,  horizontalalignment='center')
    plt.show()


def plotGlobalIncremental(data_name = None, bin_size = 5):
    # Read data
    wo_global = "_".join(diff_agent_list[0]) # without global agents
    w_global = "_".join(diff_agent_list[1]) # with global agents
    with open("./altogether_analysis/testing_result-{}.pkl".format(wo_global), "rb") as file:
        wo_global_testing_result = pickle.load(file)
    with open("./altogether_analysis/testing_result-{}.pkl".format(w_global), "rb") as file:
        w_global_testing_result = pickle.load(file)
    max_bean_num = max(max(wo_global_testing_result.beans_num_within_10), max(w_global_testing_result.beans_num_within_10))
    min_bean_num = 0
    # bin_size = 3
    ind = np.arange(min_bean_num, max_bean_num+1, bin_size)
    wo_step_nums = np.zeros_like(ind)
    wo_correct_nums = np.zeros_like(ind)
    w_step_nums = np.zeros_like(ind)
    w_correct_nums = np.zeros_like(ind)
    # every bin
    for index in range(wo_global_testing_result.shape[0]):
        cur_data = wo_global_testing_result.iloc[index]
        bean_index = cur_data.beans_num_within_10 // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        wo_step_nums[bean_index] += 1
        if cur_data.is_correct:
            wo_correct_nums[bean_index] += 1
        else:
            continue
    for index in range(w_global_testing_result.shape[0]):
        cur_data = w_global_testing_result.iloc[index]
        bean_index = cur_data.beans_num_within_10 // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        w_step_nums[bean_index] += 1
        if cur_data.is_correct:
            w_correct_nums[bean_index] += 1
        else:
            continue
    wo_cr = np.divide(wo_correct_nums, wo_step_nums)[::-1]
    w_cr = np.divide(w_correct_nums, w_step_nums)[::-1]
    plt.bar(np.arange(len(ind)) - 0.45, wo_cr, label="local + pessimistic", width=0.45, align="edge")
    plt.bar(np.arange(len(ind)), w_cr, label="local + pessimistic + global", width=0.45, align="edge")
    difference = w_cr - wo_cr
    for index, each in enumerate(difference):
        if each < 0:
            color = "green"
        elif each > 0:
            color = "red"
        else:
            continue
        plt.text(x=np.arange(len(ind))[index], y=max(w_cr[index], wo_cr[index]) + 0.01,
                 s="{s:+.4f}".format(s=each),
                 fontsize=15, horizontalalignment='center', color=color)
    plt.legend(loc="upper left", fontsize = 20)
    plt.xlabel("# of beans within 5 steps", fontsize = 20)
    x_ticks = ["[{}, {})".format(ind[i], ind[i+1]) for i in range(len(ind)-1)]
    x_ticks.append(f'[{ind[-1]}, $\infty$)')
    plt.xticks(np.arange(len(ind)), x_ticks[::-1], fontsize = 20)
    plt.ylabel("Correct Rate", fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0, 1.1)
    plt.show()


def plotPlannedHuntingIncremental(data_name = None, bin_size = 5):
    # Read data
    wo_planned = "_".join(diff_agent_list[1]) # without planned hunting agents
    w_planned = "_".join(diff_agent_list[2]) # with planned hunting agents
    with open("./altogether_analysis/testing_result-{}.pkl".format(wo_planned), "rb") as file:
        wo_planned_testing_result = pickle.load(file)
        need_planned_index = np.where(wo_planned_testing_result.planning_reward.values != -np.inf)
        wo_planned_testing_result = wo_planned_testing_result.iloc[need_planned_index].reset_index(drop = True)
    with open("./altogether_analysis/testing_result-{}.pkl".format(w_planned), "rb") as file:
        w_planned_testing_result = pickle.load(file)
        need_planned_index = np.where(w_planned_testing_result.planning_reward.values != -np.inf)
        w_planned_testing_result = w_planned_testing_result.iloc[need_planned_index].reset_index(drop=True)
    max_bean_num = max(max(wo_planned_testing_result.planning_reward), max(w_planned_testing_result.planning_reward))
    min_bean_num = min(min(wo_planned_testing_result.planning_reward), min(w_planned_testing_result.planning_reward))
    # bin_size = 10
    ind = np.arange(min_bean_num, max_bean_num+1, bin_size)
    wo_step_nums = np.zeros_like(ind)
    wo_correct_nums = np.zeros_like(ind)
    w_step_nums = np.zeros_like(ind)
    w_correct_nums = np.zeros_like(ind)
    # every bin
    for index in range(wo_planned_testing_result.shape[0]):
        cur_data = wo_planned_testing_result.iloc[index]
        bean_index = int((cur_data.planning_reward - min_bean_num)) // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        wo_step_nums[bean_index] += 1
        if cur_data.is_correct:
            wo_correct_nums[bean_index] += 1
        else:
            continue
    for index in range(w_planned_testing_result.shape[0]):
        cur_data = w_planned_testing_result.iloc[index]
        bean_index = int((cur_data.planning_reward - min_bean_num)) // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        w_step_nums[bean_index] += 1
        if cur_data.is_correct:
            w_correct_nums[bean_index] += 1
        else:
            continue
    wo_cr = np.divide(wo_correct_nums, wo_step_nums)[::-1]
    w_cr = np.divide(w_correct_nums, w_step_nums)[::-1]
    plt.bar(np.arange(len(ind)) - 0.45, wo_cr,
            label="local + pessimistic + global", width=0.45, align="edge")
    plt.bar(np.arange(len(ind)), w_cr,
            label="local + pessimistic + global + planned hunting", width=0.45, align="edge")
    difference = w_cr - wo_cr
    for index, each in enumerate(difference):
        if each < 0:
            color = "green"
        elif each > 0:
            color = "red"
        else:
            continue
        plt.text(x=np.arange(len(ind))[index], y=max(w_cr[index], wo_cr[index]) + 0.01,
                 s="{s:+.4f}".format(s=each),
                 fontsize=15, horizontalalignment='center', color=color)
    plt.xlabel("EG - PE", fontsize = 20)
    x_ticks = ["[{}, {})".format(ind[i], ind[i+1]) for i in range(len(ind)-1)]
    x_ticks.append(f'[{ind[-1]}, $\infty$)')
    plt.xticks(np.arange(len(ind)), x_ticks[::-1], fontsize = 10)
    plt.ylabel("Correct Rate", fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend(loc="upper center", fontsize = 20, ncol = 2)
    plt.ylim(0, 1.1)
    plt.show()


def plotSuicideIncremental(data_name = None, bin_size = 5):
    # Read data
    wo_suicide = "_".join(diff_agent_list[2]) # without suicide agents
    w_suicide = "_".join(diff_agent_list[3]) # with planned hunting agents
    with open("./altogether_analysis/testing_result-{}.pkl".format(wo_suicide), "rb") as file:
        wo_suicide_testing_result = pickle.load(file)
    with open("./altogether_analysis/testing_result-{}.pkl".format(w_suicide), "rb") as file:
        w_suicide_testing_result = pickle.load(file)
    max_bean_num = max(max(wo_suicide_testing_result.reborn_beans_num), max(w_suicide_testing_result.reborn_beans_num))
    min_bean_num = min(min(wo_suicide_testing_result.reborn_beans_num), min(w_suicide_testing_result.reborn_beans_num))
    # bin_size = 2
    ind = np.arange(min_bean_num, max_bean_num + 1, bin_size)
    wo_step_nums = np.zeros_like(ind)
    wo_correct_nums = np.zeros_like(ind)
    w_step_nums = np.zeros_like(ind)
    w_correct_nums = np.zeros_like(ind)
    # every bin
    for index in range(wo_suicide_testing_result.shape[0]):
        cur_data = wo_suicide_testing_result.iloc[index]
        bean_index = int((cur_data.reborn_beans_num - min_bean_num)) // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        wo_step_nums[bean_index] += 1
        if cur_data.is_correct:
            wo_correct_nums[bean_index] += 1
        else:
            continue
    for index in range(w_suicide_testing_result.shape[0]):
        cur_data = w_suicide_testing_result.iloc[index]
        bean_index = int((cur_data.reborn_beans_num - min_bean_num)) // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        w_step_nums[bean_index] += 1
        if cur_data.is_correct:
            w_correct_nums[bean_index] += 1
        else:
            continue
    wo_cr = np.divide(wo_correct_nums, wo_step_nums)[::-1]
    w_cr = np.divide(w_correct_nums, w_step_nums)[::-1]
    plt.bar(np.arange(len(ind)) - 0.45, wo_cr,
            label="local + pessimistic + global", width=0.45, align="edge")
    plt.bar(np.arange(len(ind)), w_cr,
            label="local + pessimistic + global + suicide", width=0.45, align="edge")
    difference = w_cr - wo_cr
    for index, each in enumerate(difference):
        if each < 0:
            color = "green"
        elif each > 0:
            color = "red"
        else:
            continue
        plt.text(x=np.arange(len(ind))[index], y=max(w_cr[index], wo_cr[index]) + 0.01,
                 s="{s:+.4f}".format(s=each),
                 fontsize=15, horizontalalignment='center', color=color)
    plt.xlabel("# of beans nearby reborn pos - # of beans nearby current pos", fontsize = 20)
    x_ticks = ["[{}, {})".format(ind[i], ind[i+1]) for i in range(len(ind)-1)]
    x_ticks.append(f'[{ind[-1]}, $\infty$)')
    plt.xticks(np.arange(len(ind)), x_ticks[::-1], fontsize = 20)
    plt.ylabel("Correct Rate", fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylim(0, 1.1)
    plt.legend(loc="upper center", fontsize = 17, ncol = 2)
    plt.show()


def correctRateVSBeanNum(agent_name = None, bin_size = 10):
    agent_list = ["local", "pessimistic", "global", "planned hunting", "suicide"]
    if agent_name not in agent_list:
        raise ValueError("Undefined agent {}!".format(agent_name))
    agent_index = agent_list.index(agent_name)
    wo_label = " + ".join(agent_list[:agent_index])
    w_label = " + ".join(agent_list[:agent_index+1])
    agent_index = agent_index - 2 + 1
    # Read data
    wo_agent = "_".join(diff_agent_list[agent_index-1])  # without a certain agent
    w_agent = "_".join(diff_agent_list[agent_index])  # with a certain agent
    with open("./altogether_analysis/testing_result-{}.pkl".format(wo_agent), "rb") as file:
        wo_agent_testing_result = pickle.load(file)
    with open("./altogether_analysis/testing_result-{}.pkl".format(w_agent), "rb") as file:
        w_agent_testing_result = pickle.load(file)
    max_bean_num = max(max(wo_agent_testing_result.beans_num), max(w_agent_testing_result.beans_num))
    min_bean_num = 0
    ind = np.arange(min_bean_num, max_bean_num, bin_size)
    wo_step_nums = np.zeros_like(ind)
    wo_correct_nums = np.zeros_like(ind)
    w_step_nums = np.zeros_like(ind)
    w_correct_nums = np.zeros_like(ind)
    # every bin
    for index in range(wo_agent_testing_result.shape[0]):
        cur_data = wo_agent_testing_result.iloc[index]
        bean_index = cur_data.beans_num // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        wo_step_nums[bean_index] += 1
        if cur_data.is_correct:
            wo_correct_nums[bean_index] += 1
        else:
            continue
    for index in range(w_agent_testing_result.shape[0]):
        cur_data = w_agent_testing_result.iloc[index]
        bean_index = cur_data.beans_num // bin_size
        # if min_bean_num <= cur_data.beans_num <= max_bean_num:
        #     bean_index = bean_nums[cur_data.beans_num - min_bean_num]
        w_step_nums[bean_index] += 1
        if cur_data.is_correct:
            w_correct_nums[bean_index] += 1
        else:
            continue
    wo_cr = np.divide(wo_correct_nums, wo_step_nums)[::-1]
    w_cr = np.divide(w_correct_nums, w_step_nums)[::-1]
    plt.bar(np.arange(len(ind)) - 0.45, wo_cr, label = wo_label, width=0.45, align="edge")
    plt.bar(np.arange(len(ind)), w_cr, label = w_label, width=0.45, align="edge")
    difference = w_cr - wo_cr
    for index, each in enumerate(difference):
        if each < 0:
            color = "green"
        elif each > 0:
            color = "red"
        else:
            continue
        plt.text(x = np.arange(len(ind))[index], y = max(w_cr[index], wo_cr[index])+0.01, s = "{s:+.4f}".format(s=each),
                 fontsize = 15,  horizontalalignment='center', color = color)
    plt.xlabel("# of beans", fontsize=20)
    x_ticks = ["[{}, {})".format(ind[i], ind[i + 1]) for i in range(len(ind) - 1)]
    x_ticks.append(f'[{ind[-1]}, $\infty$)')
    plt.xticks(np.arange(len(ind)), x_ticks[::-1], fontsize=20)
    plt.ylabel("Correct Rate", fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0, 1.1)
    plt.legend(loc="upper center", fontsize = 17, ncol = 2)
    plt.show()


if __name__ == '__main__':
    # # Pre-estimation
    # preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None
    config = {
        # Filename
        # "data_filename": "../common_data/partial_data_with_reward_label_cross.pkl-new_agent.pkl",
        "data_filename": "../common_data/agent_data/planned_hunting_data.pkl-new_agent.pkl",
        # Only making decisions when necessary
        "only_necessary": True,
        # The number of samples used for estimation: None for using all the data
        "clip_samples": None,
        # The window size
        "window": 10,
        # Number of trials used for moving window analysis
        "moving_window_trial_num": 10,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try": 5,
        # The number of trials used for analysis; None for using all the trials
        "trial_num": None
    }

    # ============ ESTIMATION =============
    diff_agent_list = MLE(config)

    # singleTrialMovingWindowAnalysis(config)


    # # ============ VISUALIZATION =============
    diff_agent_list = [
        ["local", "pessimistic"],
        ["local", "pessimistic", "global"],
        ["local", "pessimistic", "global", "planned_hunting"],
        ["local", "pessimistic", "global", "suicide"]
    ]

    # plotCorrectRate(data_name="planned hunting")
    # plotGlobalIncremental(data_name="planned hunting", bin_size = 3)
    # plotPlannedHuntingIncremental(data_name="planned hunting", bin_size = 8)
    # plotSuicideIncremental(data_name="planned hunting", bin_size = 2)

    # correctRateVSBeanNum(agent_name="global", bin_size = 10)
    # correctRateVSBeanNum(agent_name="planned hunting", bin_size = 10)
    # correctRateVSBeanNum(agent_name="suicide", bin_size = 10)

    # weight = np.load("./altogether_analysis/weight-global_local_pessimistic_suicide_planned_hunting.npy")
    # print(weight)

    # with open("./altogether_analysis/planned-hunting-data/testing_result-local_pessimistic_global.pkl", "rb") as file:
    #     wo_plan = pickle.load(file)[["true_dir", "estimated_dir", "is_correct"]]
    # with open("./altogether_analysis/planned-hunting-data/testing_result-local_pessimistic_global_planned_hunting.pkl", "rb") as file:
    #     w_plan = pickle.load(file)[["true_dir", "estimated_dir", "is_correct"]]
    # is_same = 0
    # is_correct = 0
    # wo_is_correct = 0
    # rest_is_same = 0
    # for index in range(wo_plan.shape[0]):
    #     if wo_plan.iloc[index].estimated_dir == w_plan.iloc[index].estimated_dir:
    #         is_same += 1
    #     else:
    #         if w_plan.iloc[index].is_correct :
    #             is_correct += 1
    #         if wo_plan.iloc[index].is_correct :
    #             wo_is_correct += 1
    #         if wo_plan.iloc[index].is_correct == w_plan.iloc[index].is_correct:
    #             rest_is_same += 1
    # print(is_same)
    # print(is_same/wo_plan.shape[0])
    # print(is_correct / (wo_plan.shape[0] - is_same))
    # print(wo_is_correct / (wo_plan.shape[0] - is_same))
    # print(rest_is_same / (wo_plan.shape[0] - is_same))

    # print(np.sum(wo_plan.at_cross))