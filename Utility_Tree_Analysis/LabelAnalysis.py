'''
Description:
    Compare simulated labels with hand-crafted labels.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    28 Oct. 2020
'''

import pickle
import pandas as pd
import numpy as np
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import copy
import seaborn

import sys
sys.path.append("./")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber
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


def readTransitionData(filename):
    '''
    Read data for MLE analysis.
    :param filename: Filename.
    '''
    # Read data and pre-processing
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index(drop = True)
    # Exclude the (0, 18) position in data
    normal_data_index = []
    for index in range(all_data.shape[0]):
        if not isinstance(all_data.global_Q[index], list): # what if no normal Q?
            normal_data_index.append(index)
    all_data = all_data.iloc[normal_data_index]
    # Split into trajectories
    trajectory_data = []
    grouped_data = all_data.groupby(["file", "trajectory_index"])
    for name, group in grouped_data:
        group = group.reset_index(drop = True)
        # True moving directions
        true_prob = group.next_pacman_dir_fill
        # Fill nan direction for optimization use
        start_index = 0
        while pd.isna(true_prob[start_index]):
            start_index += 1
            if start_index == len(true_prob):
                break
        if start_index == len(true_prob):
            print("Moving direction of trajectory {} is all nan.".format(name))
            continue
        if start_index > 0:
            true_prob[:start_index+1] = true_prob[start_index+1]
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
        trajectory_data.append([name, group, true_prob, group.iloc[0]["trajectory_shape"]])
    temp = trajectory_data[0]
    return trajectory_data


def readTrialData(filename):
    '''
        Read data for MLE analysis.
        :param filename: Filename.
        '''
    # Read data and pre-processing
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index(drop=True)
    # Exclude the (0, 18) position in data
    normal_data_index = []
    for index in range(all_data.shape[0]):
        if not isinstance(all_data.global_Q[index], list):  # what if no normal Q?
            normal_data_index.append(index)
    all_data = all_data.iloc[normal_data_index]
    # Split into trials
    trial_data = []
    trial_name_list = np.unique(all_data.file.values)
    for each in trial_name_list:
        each_trial = all_data[all_data.file == each].reset_index(drop = True)
        # True moving directions
        true_prob = each_trial.next_pacman_dir_fill
        # Fill nan direction for optimization use
        start_index = 0
        while pd.isna(true_prob[start_index]):
            start_index += 1
            if start_index == len(true_prob):
                break
        if start_index == len(true_prob):
            print("Moving direciton of trial {} is all nan.".format(each))
            continue
        if start_index > 0:
            true_prob[:start_index + 1] = true_prob[start_index + 1]
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
        trial_data.append([each, each_trial, true_prob])
    return trial_data



# ===================================
#       INDIVIDUAL ESTIMATION
# ===================================
def _readData(filename):
    '''
    Read data for pre-estimation.
    '''
    with open(filename, "rb") as file:
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
    suicide_depth = 10
    suicide_ghost_attractive_thr = 10
    suicide_fruit_attractive_thr = 10
    suicide_ghost_repulsive_thr = 10
    # Configuration (flast direction)
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Direction sstimation
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
            ghost_attractive_thr = local_ghost_attractive_thr,
            fruit_attractive_thr = local_fruit_attractive_thr,
            ghost_repulsive_thr = local_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 1.0,
            risk_coeff = 0.0
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
            depth = pessimistic_depth,
            ghost_attractive_thr = pessimistic_ghost_attractive_thr,
            fruit_attractive_thr = pessimistic_fruit_attractive_thr,
            ghost_repulsive_thr = pessimistic_ghost_repulsive_thr,
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
        "../common_data/transition/global_to_local.pkl",
        "../common_data/transition/local_to_global.pkl",
        "../common_data/transition/local_to_evade.pkl",
    ]
    for filename in filename_list:
        print("-" * 50)
        print(filename)
        all_data = _readData(filename)
        print("Finished reading data.")
        print("Start estimating...")
        all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount)
        with open("../common_data/transition/{}-with_Q.pkl".format(filename.split("/")[-1].split(".")[0]), "wb") as file:
            pickle.dump(all_data, file)
        print("Save to ", "../common_data/transition/{}-with_Q.pkl".format(filename.split("/")[-1].split(".")[0]))
    pd.options.mode.chained_assignment = "warn"


# ===================================
#         FAST OPTIMIZATION
# ===================================
def _makeChoice(prob):
    copy_estimated = copy.deepcopy(prob)
    if np.any(prob) < 0:
        available_dir_index = np.where(prob != 0)
        copy_estimated[available_dir_index] = copy_estimated[available_dir_index] - np.min(copy_estimated[available_dir_index]) + 1
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


def _estimationLabeling(Q_value, agent_list):
    return agent_list[np.argmax(Q_value)]


def _handcraftLabeling(labels):
    labels = labels.fillna(0)
    # local
    if labels.label_local_graze or labels.label_local_graze_noghost:
        return "local"
    # evade (pessmistic)
    elif labels.label_evade:
        return "pessimistic"
    # global
    elif labels.label_global_optimal or labels.label_global_notoptimal or labels.label_global:
        return "global"
    # suicide
    elif labels.label_suicide:
        return "suicide"
    # planned hunting
    elif labels.label_true_planned_hunting:
        return "planned_hunting"
    else:
        return None


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
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
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


def movingWindowAnalysis(config):
    print("=" * 20, " Moving Window ", "=" * 20)
    transition_type = config["trajectory_data_filename"].split("/")[-1].split(".")[0].split("-")[0]
    print(transition_type)
    print("Agent List :", config["agents"])
    window = config["window"]
    # Construct optimizer
    params = [1 for _ in range(len(config["agents"]))]
    bounds = [[0, 1000] for _ in range(len(config["agents"]))]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    # Load trajectory data
    trajectory_data = readTransitionData(config["trajectory_data_filename"])
    trajectory_shapes = [each[3] if isinstance(each[3], list) else each[3][-1] for each in trajectory_data] # Unknown BUG:
    trajectory_length = [min([each[1] - each[0] + 1, each[2] - each[1]]) for each in trajectory_shapes]
    trajectory_length = min(trajectory_length)
    print("Num of trajectories : ", len(trajectory_shapes))
    print("Trajectory length : ", trajectory_length)
    window_index = np.arange(1, 2*trajectory_length)
    # (num of trajectories, num of windows, num of agents)
    trajectory_weight = np.zeros((len(trajectory_data), len(window_index), len(config["agents"])))
    # (num of trajectories, num of windows)
    trajectory_cr = np.zeros((len(trajectory_data), len(window_index)))
    # For each trajectory, estimate agent weights through sliding windows
    for trajectory_index, trajectory in enumerate(trajectory_data):
        start_index = trajectory_shapes[trajectory_index][1] - trajectory_length - trajectory_shapes[trajectory_index][0]
        end_index = trajectory_shapes[trajectory_index][1] + trajectory_length - trajectory_shapes[trajectory_index][0]
        X = trajectory[1].iloc[start_index:end_index]
        Y = trajectory[2].iloc[start_index:end_index]
        num_samples = len(Y)
        print("-"*15)
        print("Trajectory {} : ".format(trajectory_index), trajectory[0])
        # for each window
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[
                    centering_point - window if centering_point - window >= 0 else 0
                    :
                    centering_point + window if centering_point + window < num_samples else num_samples
                    ]
            sub_Y = Y[
                    centering_point - window if centering_point - window >= 0 else 0
                    :
                    centering_point + window if centering_point + window < num_samples else num_samples
                    ]
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
                config["agents"],
                return_trajectory=False
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
                    print("Fail, retrying...")
                    retry_num += 1
            trajectory_weight[trajectory_index, centering_index, :] = res.x
            # correct rate in the window
            _, estimated_prob = negativeLikelihood(
                res.x,
                sub_X,
                sub_Y,
                config["agents"],
                return_trajectory = True
            )
            # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            trajectory_cr[trajectory_index, centering_index] = correct_rate
    # Print out results and save data
    print("Average Correct Rate: {}".format(np.nanmean(trajectory_cr, axis=0)))
    avg_agent_weight = np.nanmean(trajectory_weight, axis=0)
    print("Estimated label : ", [_estimationLabeling(each, config["agents"]) for each in avg_agent_weight])
    # Save estimated agent weights
    np.save("../common_data/transition/{}-agent_weight.npy".format(transition_type), trajectory_weight)
    np.save("../common_data/transition/{}-cr.npy".format(transition_type), trajectory_cr)


def correlationAnalysis(config):
    # Read trial data
    temp_trial_data = readTrialData(config["trial_data_filename"])
    trial_num = len(temp_trial_data)
    print("Num of trials : ", trial_num)
    trial_index = range(trial_num)
    if config["trial_num"] is not None:
        if config["trial_num"] < trial_num:
            trial_index = np.random.choice(range(trial_num), config["trial_num"], replace = False)
    trial_data = [temp_trial_data[each] for each in trial_index]
    label_list = ["label_local_graze", "label_local_graze_noghost",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]
    trial_weight = []
    trial_cr = []
    handcrafted_labels = []
    estimated_labels = []
    # Construct optimizer
    params = [1 for _ in range(len(config["agents"]))]
    bounds = [[0, 1000] for _ in range(len(config["agents"]))]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    for trial_index, each in enumerate(trial_data):
        print("-"*15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print(trial_name)
        # Hand-crafted label
        temp_handcrafted_label = [_handcraftLabeling(X[label_list].iloc[index]) for index in range(X.shape[0])]
        handcrafted_labels.append(temp_handcrafted_label)
        # Estimating label through moving window analysis
        window = config["trial_window"]
        print("Trial length : ", trial_length)
        window_index = np.arange(1, trial_length - 1)
        # (num of windows, num of agents)
        temp_weight = np.zeros((len(window_index), len(config["agents"])))
        # (1, num of windows)
        temp_cr = np.zeros((len(window_index), ))
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[
                    centering_point - window if centering_point - window >= 0 else 0
                    :
                    centering_point + window if centering_point + window < trial_length else trial_length
                    ]
            sub_Y = Y[
                    centering_point - window if centering_point - window >= 0 else 0
                    :
                    centering_point + window if centering_point + window < trial_length else trial_length
                    ]
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
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
                    bounds=bounds,
                    tol=1e-5,
                    constraints=cons
                )
                is_success = res.success
                if not is_success:
                    print("Fail, retrying...")
                    retry_num += 1
            temp_weight[centering_index, :] = res.x
            trial_weight.append(temp_weight)
            # correct rate in the window
            _, estimated_prob = negativeLikelihood(
                res.x,
                sub_X,
                sub_Y,
                config["agents"],
                return_trajectory=True
            )
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            temp_cr[centering_index] = correct_rate
        trial_cr.append(temp_cr)
        temp_estimated_label = [_estimationLabeling(each, config["agents"]) for each in temp_weight]
        estimated_labels.append(temp_estimated_label)
        print("Average correct rate for trial : ", np.nanmean(temp_cr))
    print("Average correct rate for all : ", np.nanmean([np.nanmean(each) for each in trial_cr]))
    # Save data
    save_base = config["trial_data_filename"].split("/")[-1].split(".")[0]
    np.save("../common_data/trial/{}-estimated_labels.npy".format(save_base), estimated_labels)
    np.save("../common_data/trial/{}-handcrafted_labels.npy".format(save_base), handcrafted_labels)
    np.save("../common_data/trial/{}-trial_cr.npy".format(save_base), trial_cr)



# ===================================
#         VISUALIZATION
# ===================================
def plotWeightVariation(config, plot_sem = False, need_normalization = False):
    # Determine agent names
    agent_list = config["agent_list"]
    agent_color = {
        "local":"red",
        "global":"blue",
        "pessimistic":"green",
        "suicide":"cyan",
        "planned_hunting":"magenta"
    }
    # Read data
    local2global_weight = np.load(config["local_to_global_agent_weight"])
    local2global_cr = np.load(config["local_to_global_cr"])
    local2evade_weight = np.load(config["local_to_evade_agent_weight"])
    local2evade_cr = np.load(config["local_to_evade_cr"])
    global2local_weight = np.load(config["global_to_local_agent_weight"])
    global2local_cr = np.load(config["global_to_local_cr"])

    # Plot weight variation
    plt.subplot(1 ,3, 1)
    agent_name = agent_list[0]
    plt.title("Local $\\rightarrow$ Global (avg cr = {avg:.3f})".format(avg = np.nanmean(local2global_cr)), fontsize = 20)
    avg_local2global_weight = np.nanmean(local2global_weight, axis = 0)
    # normalization
    if need_normalization:
        for index in range(avg_local2global_weight.shape[0]):
            avg_local2global_weight[index, :]  = avg_local2global_weight[index, :] / np.max(avg_local2global_weight[index, :])
            local2global_weight[:, index, :] = local2global_weight[:, index, :] / np.max(local2global_weight[:, index, :])
    sem_local2global_weight  = scipy.stats.sem(local2global_weight, axis=0, nan_policy = "omit")
    for index in range(len(agent_name)):
        plt.plot(avg_local2global_weight[:, index], color = agent_color[agent_name[index]], ms = 3, lw = 5,label = agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, len(avg_local2global_weight)),
                avg_local2global_weight[:, index] - sem_local2global_weight[:, index],
                avg_local2global_weight[:, index] + sem_local2global_weight[:, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.ylabel("Normalized Agent Weight", fontsize=20)
    plt.xlim(0, avg_local2global_weight.shape[0] - 1)
    centering_point = (len(avg_local2global_weight) - 1) / 2
    x_ticks = [str(int(each)) for each in np.arange(0-centering_point, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, len(avg_local2global_weight)-centering_point, 1)])
    if (avg_local2global_weight.shape[0] - 1) not in x_ticks:
        x_ticks.append(avg_local2global_weight.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(np.arange(len(avg_local2global_weight)), x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize = 15)
    plt.yticks(fontsize=15)
    plt.ylim(0.1, 1.1)
    plt.legend(loc = "lower center", fontsize=15, ncol=len(agent_name))
    # plt.show()

    plt.subplot(1 ,3, 2)
    agent_name = agent_list[1]
    plt.title("Local $\\rightarrow$ Evade  (avg cr = {avg:.3f})".format(avg = np.nanmean(local2evade_cr)), fontsize = 20)
    avg_local2evade_weight = np.nanmean(local2evade_weight, axis=0)
    # normalization
    if need_normalization:
        for index in range(avg_local2evade_weight.shape[0]):
            avg_local2evade_weight[index, :] = avg_local2evade_weight[index, :] / np.max(avg_local2evade_weight[index, :])
            local2evade_weight[:, index, :] = local2evade_weight[:, index, :] / np.max(local2evade_weight[:, index, :])
    sem_local2evade_weight = scipy.stats.sem(local2evade_weight, axis=0, nan_policy = "omit")
    for index in range(len(agent_name)):
        plt.plot(avg_local2evade_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5, label=agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, len(avg_local2evade_weight)),
                avg_local2evade_weight[:, index] - sem_local2evade_weight[:, index],
                avg_local2evade_weight[:, index] + sem_local2evade_weight[:, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, avg_local2evade_weight.shape[0] - 1)
    centering_point = (len(avg_local2evade_weight) - 1) / 2
    x_ticks = [str(int(each)) for each in np.arange(0 - centering_point, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, len(avg_local2evade_weight) - centering_point, 1)])
    if (avg_local2evade_weight.shape[0] - 1) not in x_ticks:
        x_ticks.append(avg_local2evade_weight.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(np.arange(len(avg_local2evade_weight)), x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.1, 1.1)
    plt.legend(loc = "lower center", fontsize=15, ncol=len(agent_name))
    # plt.show()

    plt.subplot(1, 3, 3)
    agent_name = agent_list[2]
    plt.title("Global $\\rightarrow$ Local  (avg cr = {avg:.3f})".format(avg = np.nanmean(global2local_cr)), fontsize = 20)
    avg_global2local_weight = np.nanmean(global2local_weight, axis=0)
    # normalization
    if need_normalization:
        for index in range(avg_global2local_weight.shape[0]):
            avg_global2local_weight[index, :] = avg_global2local_weight[index, :] / np.max(avg_global2local_weight[index, :])
            global2local_weight[:, index, :] = global2local_weight[:, index, :] / np.max(global2local_weight[:, index, :])
    sem_global2local_weight = scipy.stats.sem(global2local_weight, axis=0, nan_policy = "omit")
    for index in range(len(agent_name)):
        plt.plot(avg_global2local_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5, label=agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, len(avg_global2local_weight)),
                avg_global2local_weight[:, index] - sem_global2local_weight[:, index],
                avg_global2local_weight[:, index] + sem_global2local_weight[:, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, avg_global2local_weight.shape[0] - 1)
    centering_point = (len(avg_global2local_weight) - 1) / 2
    x_ticks = [str(int(each)) for each in np.arange(0 - centering_point, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, len(avg_global2local_weight) - centering_point, 1)])
    if (avg_global2local_weight.shape[0] - 1) not in x_ticks:
        x_ticks.append(avg_global2local_weight.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(np.arange(len(avg_global2local_weight)), x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.1, 1.1)
    plt.legend(loc = "lower center", fontsize=15, ncol=len(agent_name))
    plt.show()


def _label2Index(labels):
    label_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    label_val = copy.deepcopy(labels)
    for index, each in enumerate(label_val):
        if each is not None:
            label_val[index] = label_list.index(each)
        else:
            label_val[index] = None
    return label_val


def computeCorrelation(config):
    estimated_labels = np.load(config["estimated_label_filename"], allow_pickle=True)
    handcrafted_labels = np.load(config["handcrafted_label_filename"], allow_pickle=True)
    trial_cr = np.load(config["trial_cr_filename"], allow_pickle=True)
    trial_num = len(estimated_labels)
    trial_matching_rate = []
    # trial_correlation = []
    for index in range(trial_num):
        estimated = np.array(_label2Index(estimated_labels[index]))
        handcrafted = np.array(_label2Index(handcrafted_labels[index]))
        handcrafted = handcrafted[1:len(handcrafted) - 1]  # TODO: check this; because of the moving window
        # what about None value
        not_none_index = np.where(handcrafted != None)
        if len(not_none_index[0]) != 0:
            estimated = np.array(estimated)[not_none_index]
            handcrafted = np.array(handcrafted)[not_none_index]
            matching_rate = np.sum(estimated == handcrafted) / len(estimated)
            # trial_correlation.append(scipy.stats.pearsonr(estimated, handcrafted))
            trial_matching_rate.append(matching_rate)
    print("-"*15)
    print("Matching rate : ")
    print("Max : ", np.nanmax(trial_matching_rate))
    print("Min : ", np.nanmin(trial_matching_rate))
    print("Median : ", np.nanmedian(trial_matching_rate))
    print("Average : ", np.nanmean(trial_matching_rate))
    # print("-" * 15)
    # print("Correlation : ")
    # print("Max : ", np.nanmax(trial_correlation))
    # print("Min : ", np.nanmin(trial_correlation))
    # print("Median : ", np.nanmedian(trial_correlation))
    # print("Average : ", np.nanmean(trial_correlation))
    # histogram
    plt.title("Label Matching on 500 Trials", fontsize = 20)
    plt.hist(trial_matching_rate)
    plt.xlabel("Correct Rate (estimated label = hand-crafted label)", fontsize = 20)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.ylabel("# of Trials", fontsize=20)
    plt.yticks(fontsize=20)

    # plt.subplot(1, 2, 2)
    # plt.hist([each[0] for each in trial_correlation])

    plt.show()






if __name__ == '__main__':
    # # Pre-estimation
    # preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None
    config = {
        # Agents: at least one of "global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting".
        # "agents": ["local", "global", "pessimistic", "suicide", "planned_hunting"],
        "agents": ["local", "global"],
        # ==================================================================================
        #                       For Sliding Window Analysis
        # Filename
        "trajectory_data_filename": "../common_data/transition/global_to_local-with_Q.pkl",
        # The window size
        "window": 5,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try": 5,
        # ==================================================================================

        # ==================================================================================
        #                       For Correlation Analysis
        # Filename
        "trial_data_filename": "../common_data/trial/500_trial_data-with_Q.pkl",
        # The number of trials used for analysis
        "trial_num" : None,
        # Window size for correlation analysis
        "trial_window" : 5,
        # ==================================================================================

        # ==================================================================================
        #                       For Experimental Results Visualization
        "estimated_label_filename" : "../common_data/trial/500_trial_data-with_Q-estimated_labels.npy",
        "handcrafted_label_filename": "../common_data/trial/500_trial_data-with_Q-handcrafted_labels.npy",
        "trial_cr_filename": "../common_data/trial/500_trial_data-with_Q-trial_cr.npy",

        "local_to_global_agent_weight" : "../common_data/transition/relevant_agents/local_to_global-agent_weight.npy",
        "local_to_global_cr": "../common_data/transition/relevant_agents/local_to_global-cr.npy",
        "local_to_evade_agent_weight": "../common_data/transition/relevant_agents/local_to_evade-agent_weight.npy",
        "local_to_evade_cr": "../common_data/transition/relevant_agents/local_to_evade-cr.npy",
        "global_to_local_agent_weight": "../common_data/transition/relevant_agents/global_to_local-agent_weight.npy",
        "global_to_local_cr": "../common_data/transition/relevant_agents/global_to_local-cr.npy",
        "agent_list" : [["local", "global"], ["local", "pessimistic"], ["local", "global"]]
    }

    # ============ MOVING WINDOW =============
    # movingWindowAnalysis(config)

    # ============ Correlation =============
    # correlationAnalysis(config)

    # ============ VISUALIZATION =============
    # computeCorrelation(config)
    plotWeightVariation(config, plot_sem = True, need_normalization = True)