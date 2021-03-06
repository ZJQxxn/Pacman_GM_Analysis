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
import os
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
    for index in range(all_data.shape[0]):
        if isinstance(all_data.global_Q[index], list): # what if no normal Q?
            if index == all_data.shape[0] - 1:
                if isinstance(all_data.global_Q[index - 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index - 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index - 1]
            else:
                if isinstance(all_data.global_Q[index+1], list):
                    all_data.global_Q[index] = all_data.global_Q[index + 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index + 1]
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
    # temp = trajectory_data[0]
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
    for index in range(all_data.shape[0]):
        if isinstance(all_data.global_Q[index], list):  # what if no normal Q?
            if index == all_data.shape[0] - 1:
                if isinstance(all_data.global_Q[index - 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index - 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index - 1]
            else:
                if isinstance(all_data.global_Q[index + 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index + 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index + 1]
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


def readAllData(filename, trial_num):
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index(drop=True)
    for index in range(all_data.shape[0]):
        if isinstance(all_data.global_Q[index], list):  # what if no normal Q?
            if index == all_data.shape[0] - 1:
                if isinstance(all_data.global_Q[index - 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index - 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index - 1]
            else:
                if isinstance(all_data.global_Q[index + 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index + 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index + 1]
    # Split into trials
    trial_data = []
    trial_name_list = np.unique(all_data.file.values)
    trial_index = np.arange(len(trial_name_list))
    if trial_num is not None and trial_num < len(trial_index):
        trial_index = np.random.choice(trial_index, trial_num, replace = False)
    trial_name_list = trial_name_list[trial_index]
    is_used = np.where(all_data.file.apply(lambda x: x in trial_name_list).values == 1)
    all_data = all_data.iloc[is_used].reset_index(drop=True)
    # True moving directions
    true_prob = all_data.next_pacman_dir_fill
    # Fill nan direction for optimization use
    start_index = 0
    while pd.isna(true_prob[start_index]):
        start_index += 1
        if start_index == len(true_prob):
            break
    if start_index > 0:
        true_prob[:start_index + 1] = true_prob[start_index + 1]
    for index in range(1, true_prob.shape[0]):
        if pd.isna(true_prob[index]):
            true_prob[index] = true_prob[index - 1]
    true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    return all_data, true_prob


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
    indicies = np.argsort(Q_value)
    # estimated_label = [agent_list[each] for each in indicies[-2:]]
    estimated_label = agent_list[indicies[-1]]
    # if Q_value[indicies[-2]] - Q_value[indicies[-3]] <= 0.1:
    #     estimated_label.append(agent_list[indicies[-3]])
    return estimated_label


def _handcraftLabeling(labels):
    hand_crafted_label = []
    labels = labels.fillna(0)
    # local
    if labels.label_local_graze or labels.label_local_graze_noghost or labels.label_true_accidental_hunting or labels.label_global_ending:
        hand_crafted_label.append("local")
    # evade (pessmistic)
    if labels.label_evade:
        hand_crafted_label.append("pessimistic")
    # global
    if labels.label_global_optimal or labels.label_global_notoptimal or labels.label_global:
        if labels.label_global_ending:
            pass
        else:
            hand_crafted_label.append("global")
    # suicide
    if labels.label_suicide:
        hand_crafted_label.append("suicide")
    # planned hunting
    if labels.label_true_planned_hunting:
        hand_crafted_label.append("planned_hunting")
    if len(hand_crafted_label) == 0:
        hand_crafted_label = None
    return hand_crafted_label


def _label2Index(labels):
    label_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    label_val = copy.deepcopy(labels)
    for index, each in enumerate(label_val):
        if each is not None:
            label_val[index] = label_list.index(each)
        else:
            label_val[index] = None
    return label_val


def negativeLikelihood(param, all_data, true_prob, agents_list, return_trajectory = False, need_intercept = False):
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
        if need_intercept:
            if len(agents_list)+1 != len(param):
                raise ValueError("Specify intercept!")
            agent_weight = [param[i] for i in range(len(param)-1)]
            intercept = param[-1]
        else:
            agent_weight = [param[i] for i in range(len(param))]
            intercept = 0
    # Compute estimation error
    nll = 0  # negative log likelihood
    num_samples = all_data.shape[0]
    agents_list = ["{}_Q".format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][each_agent]
    dir_Q_value = agent_Q_value @ agent_weight + intercept # add intercept
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
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    window = config["window"]
    # Construct optimizer
    params = [1 for _ in range(len(config["agents"]))]
    bounds = [[0, 1000] for _ in range(len(config["agents"]))]
    if config["need_intercept"]:
        params.append(1)
        bounds.append([-1000, 1000])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    # Load trajectory data
    trajectory_data = readTransitionData(config["trajectory_data_filename"])
    trajectory_shapes = [each[3] if isinstance(each[3], list) else each[3][-1] for each in trajectory_data] # Unknown BUG:
    trajectory_length = [min([each[1] - each[0], each[2] - each[1] - 1]) for each in trajectory_shapes]
    trajectory_length = min(trajectory_length)
    print("Num of trajectories : ", len(trajectory_shapes))
    print("Trajectory length : ", trajectory_length)
    window_index = np.arange(window, 2*trajectory_length - window+ 1)
    # (num of trajectories, num of windows, num of agents)
    trajectory_weight = np.zeros(
        (len(trajectory_data), len(window_index), len(config["agents"]) if not config["need_intercept"] else len(config["agents"]) + 1)
    )
    # (num of trajectories, num of windows)
    trajectory_cr = np.zeros((len(trajectory_data), len(window_index)))
    # (num of trajectories, num of windows, num of samples in each window, num of agents)
    trajectory_Q = np.zeros((len(trajectory_data), len(window_index), window*2+1, 5, 4))
    # For each trajectory, estimate agent weights through sliding windows
    for trajectory_index, trajectory in enumerate(trajectory_data):
        start_index = trajectory_shapes[trajectory_index][1] - trajectory_length - trajectory_shapes[trajectory_index][0]
        end_index = trajectory_shapes[trajectory_index][1] + trajectory_length - trajectory_shapes[trajectory_index][0] + 1
        X = trajectory[1].iloc[start_index:end_index]
        Y = trajectory[2].iloc[start_index:end_index]
        num_samples = len(Y)
        print("-"*15)
        print("Trajectory {} : ".format(trajectory_index), trajectory[0])
        # for each window
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
                config["agents"],
                return_trajectory=False,
                need_intercept=config["need_intercept"]
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
            # correct rate in the window
            _, estimated_prob = negativeLikelihood(
                res.x,
                sub_X,
                sub_Y,
                config["agents"],
                return_trajectory = True,
                need_intercept=config["need_intercept"]
            )
            # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            trajectory_cr[trajectory_index, centering_index] = correct_rate
            trajectory_weight[trajectory_index, centering_index, :] = res.x
            Q_value = sub_X[agents_list].values
            for i in range(window*2+1): # num of samples in a window
                for j in range(5): # number of agents
                    trajectory_Q[trajectory_index, centering_index, i, j, :] = Q_value[i][j]
    # Print out results and save data
    print("Average Correct Rate: {}".format(np.nanmean(trajectory_cr, axis=0)))
    if config["need_intercept"]:
        avg_agent_weight = np.nanmean(trajectory_weight[:, :, :-1], axis=0)
    else:
        avg_agent_weight = np.nanmean(trajectory_weight, axis=0)
    print("Estimated label : ", [_estimationLabeling(each, config["agents"]) for each in avg_agent_weight])
    # Save estimated agent weights
    np.save("../common_data/transition/{}-window{}-agent_weight-{}_intercept.npy".format(
        transition_type, window, "w" if config["need_intercept"] else "wo"), trajectory_weight)
    np.save("../common_data/transition/{}-window{}-cr-{}_intercept.npy".format(
        transition_type, window,"w" if config["need_intercept"] else "wo"), trajectory_cr)
    np.save("../common_data/transition/{}-window{}-Q-{}_intercept.npy".format(
        transition_type, window, "w" if config["need_intercept"] else "wo"), trajectory_Q)


def integrationAnalysis(config):
    print("=" * 20, " Moving Window ", "=" * 20)
    transition_type = config["trajectory_data_filename"].split("/")[-1].split(".")[0].split("-")[0]
    print(transition_type)
    print("Agent List :", config["agents"])
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    # Construct optimizer
    params = [1 for _ in range(len(config["agents"]))]
    bounds = [[0, 1000] for _ in range(len(config["agents"]))]
    if config["need_intercept"]:
        params.append(1)
        bounds.append([-1000, 1000])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    # Load trajectory data
    trajectory_data = readTransitionData(config["trajectory_data_filename"])
    trajectory_shapes = [each[3] if isinstance(each[3], list) else each[3][-1] for each in trajectory_data]
    trajectory_length = [min([each[1] - each[0], each[2] - each[1] - 1]) for each in trajectory_shapes]
    trajectory_length = min(trajectory_length)
    print("Num of trajectories : ", len(trajectory_shapes))
    print("Trajectory length : ", trajectory_length)
    # trial data --> step data
    X_column_names = trajectory_data[0][1].columns.values
    # Y_column_names = trajectory_data[0][2].columns.values
    step_data = []
    for index in range(trajectory_length*2+1):
        temp_step_X = []
        temp_step_Y = []
        for each_trajectory in trajectory_data:
            temp_step_X.append(each_trajectory[1].iloc[index].values)
            temp_step_Y.append(each_trajectory[2][index])
        step_data.append([pd.DataFrame(temp_step_X, columns = X_column_names), pd.Series(temp_step_Y)])
    # (num of trajectories, num of windows, num of agents)
    trajectory_weight = np.zeros(
        (len(step_data), len(config["agents"]) if not config["need_intercept"] else len(config["agents"]) + 1)
    )
    # (num of trajectories, num of windows)
    trajectory_cr = np.zeros((len(step_data),))
    # (num of trajectories, num of windows, num of samples in each window, num of agents)
    trajectory_Q = np.zeros((len(step_data), len(trajectory_data), 5, 4))
    # For each time step, estimate agent weights
    for step_index, each_step in enumerate(step_data):
        print("Step at {}...".format(step_index))
        X = each_step[0]
        Y = each_step[1]
        # estimation in the window
        func = lambda params: negativeLikelihood(
            params,
            X,
            Y,
            config["agents"],
            return_trajectory=False,
            need_intercept=config["need_intercept"]
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
        # correct rate in the window
        _, estimated_prob = negativeLikelihood(
            res.x,
            X,
            Y,
            config["agents"],
            return_trajectory=True,
            need_intercept=config["need_intercept"]
        )
        # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
        true_dir = Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        trajectory_cr[step_index] = correct_rate
        trajectory_weight[step_index, :] = res.x
        Q_value = X[agents_list].values
        for i in range(Q_value.shape[0]):  # num of samples in a window
            for j in range(5):  # number of agents
                trajectory_Q[step_index, i, j, :] = Q_value[i][j]

    # Print out results and save data
    print("Average Correct Rate: {}".format(np.nanmean(trajectory_cr)))
    print("Estimated label : ", [_estimationLabeling(each, config["agents"]) for each in trajectory_weight[:, :-1]])
    # Save estimated agent weights
    np.save("../common_data/integration_transition/{}-agent_weight-{}_intercept.npy".format(
        transition_type, "w" if config["need_intercept"] else "wo"), trajectory_weight)
    np.save("../common_data/integration_transition/{}-cr-{}_intercept.npy".format(
        transition_type,"w" if config["need_intercept"] else "wo"), trajectory_cr)
    np.save("../common_data/integration_transition/{}-Q-{}_intercept.npy".format(
        transition_type, "w" if config["need_intercept"] else "wo"), trajectory_Q)


def correlationAnalysis(config):
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in config["correlation_agents"]]
    window = config["trial_window"]
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
                  "label_true_planned_hunting"] #TODO: change this
    trial_weight = []
    trial_cr = []
    handcrafted_labels = []
    estimated_labels = []
    trial_bean_vs_cr = []
    trial_Q = []
    # Construct optimizer
    params = [1 for _ in range(len(config["correlation_agents"]))]
    bounds = [[0, 1000] for _ in range(len(config["correlation_agents"]))]
    if config["need_intercept"]:
        params.append(1)
        bounds.append([-1000, 1000])
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
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight = np.zeros((len(window_index), len(config["correlation_agents"]) if not config["need_intercept"] else len(config["correlation_agents"]) + 1))
        temp_cr = np.zeros((len(window_index), ))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
                config["correlation_agents"],
                return_trajectory = False,
                need_intercept = config["need_intercept"]
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
            # correct rate in the window
            _, estimated_prob = negativeLikelihood(
                res.x,
                sub_X,
                sub_Y,
                config["correlation_agents"],
                return_trajectory = True,
                need_intercept = config["need_intercept"]
            )
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            # trial name, pacman pos, beans, energizers, fruit pos, true dir, estimated dir, is correct, window cr
            trial_bean_vs_cr.append(
                [
                    trial_name,
                    cur_step.pacmanPos,
                    cur_step.beans,
                    cur_step.energizers,
                    cur_step.fruitPos,
                    true_dir[len(sub_Y) // 2],
                    estimated_dir[len(sub_Y) // 2],
                    true_dir[len(sub_Y) // 2] == estimated_dir[len(sub_Y) // 2],
                    correct_rate
                ]
            )
            temp_cr[centering_index] = correct_rate
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
        trial_cr.append(temp_cr)
        trial_weight.append(temp_weight)
        trial_Q.append(temp_trial_Q)
        if config["need_intercept"]:
            temp_estimated_label = [_estimationLabeling(each, config["correlation_agents"]) for each in temp_weight[:,:-1]]
        else:
            temp_estimated_label = [_estimationLabeling(each, config["correlation_agents"]) for each in temp_weight]
        estimated_labels.append(temp_estimated_label)
        print("Average correct rate for trial : ", np.nanmean(temp_cr))
    print("Average correct rate for all : ", np.nanmean([np.nanmean(each) for each in trial_cr]))
    # Save data
    save_base = config["trial_data_filename"].split("/")[-1].split(".")[0]
    np.save("../common_data/trial/{}-window{}-{}_intercept-estimated_labels.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), estimated_labels)
    np.save("../common_data/trial/{}-window{}-{}_intercept-handcrafted_labels.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), handcrafted_labels)
    np.save("../common_data/trial/{}-window{}-{}_intercept-trial_cr.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_cr)
    np.save("../common_data/trial/{}-window{}-{}_intercept-trial_weight.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_weight)
    np.save("../common_data/trial/{}-window{}-{}_intercept-bean_vs_cr.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_bean_vs_cr)
    np.save("../common_data/trial/{}-window{}-{}_intercept-Q.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_Q)


def multipleLabelAnalysis(config):
    print("== Multi Label Aalysis ==")
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    window = config["trial_window"]
    temp_trial_data = readTrialData(config["trial_data_filename"])
    trial_num = len(temp_trial_data)
    print("Num of trials : ", trial_num)
    trial_index = range(trial_num)
    if config["trial_num"] is not None:
        if config["trial_num"] < trial_num:
            trial_index = np.random.choice(range(trial_num), config["trial_num"], replace = False)
    trial_data = [temp_trial_data[each] for each in trial_index]
    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]

    trial_weight_main = []
    trial_weight_rest = []
    trial_Q = []
    handcrafted_labels = []
    trial_matching_rate = []
    all_estimated_label = []
    for trial_index, each in enumerate(trial_data):
        print("-"*15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print(trial_index, " : ", trial_name)
        # Hand-crafted label
        temp_handcrafted_label = [_handcraftLabeling(X[label_list].iloc[index]) for index in range(X.shape[0])]
        temp_handcrafted_label = temp_handcrafted_label[window:-window]
        handcrafted_labels.append(temp_handcrafted_label)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight_main = np.zeros((len(window_index), 2 if not config["need_intercept"] else 3))
        temp_weight_rest = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        trial_estimated_label = []
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
            # estimation in the window
            window_estimated_label = []
            for agent_index, agent_name in enumerate([["global", "local"], ["pessimistic", "suicide", "planned_hunting"]]):
                # Construct optimizer
                params = [1 for _ in range(len(agent_name))]
                bounds = [[0, 1000] for _ in range(len(agent_name))]
                if config["need_intercept"]:
                    params.append(1)
                    bounds.append([-1000, 1000])
                cons = []  # construct the bounds in the form of constraints
                for par in range(len(bounds)):
                    l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
                    u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
                    cons.append(l)
                    cons.append(u)
                # estimation in the window
                func = lambda params: negativeLikelihood(
                    params,
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=False,
                    need_intercept=config["need_intercept"]
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
                if agent_index == 0:
                    temp_weight_main[centering_index, :] = res.x
                    contribution = temp_weight_main[centering_index, :-1] * \
                                   [scaleOfNumber(each) for each in np.max(np.abs(temp_trial_Q[centering_index, :, [0, 1], :]), axis=(1, 2))]
                    # contribution = temp_weight_main[centering_index, :-1] * \
                    #                [each for each in
                    #                 np.nanmean(temp_trial_Q[centering_index, :, [0, 1], :], axis=(1, 2))]
                else:
                    temp_weight_rest[centering_index, :] = res.x
                    contribution = temp_weight_rest[centering_index, :-1] * \
                                   [scaleOfNumber(each) for each in np.max(np.abs(temp_trial_Q[centering_index, :, [2, 3, 4], :]), axis=(1, 2))]
                    # contribution = temp_weight_rest[centering_index, :-1] * \
                    #                [each for each in
                    #                 np.nanmean(temp_trial_Q[centering_index, :, [2, 3, 4], :], axis=(1, 2))]
                window_estimated_label.append(_estimationLabeling(contribution, agent_name))
            trial_estimated_label.append(window_estimated_label)
        matched_num = 0
        not_nan_num = 0
        for i in range(len(temp_handcrafted_label)):
            if temp_handcrafted_label[i] is not None:
                not_nan_num += 1
                if len(np.intersect1d(temp_handcrafted_label[i], trial_estimated_label[i])) > 0:
                    matched_num += 1
        print(" Trial label matching rate : ", matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")
        trial_matching_rate.append(matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")
        trial_weight_main.append(temp_weight_main)
        trial_weight_rest.append(temp_weight_rest)
        trial_Q.append(temp_trial_Q)
        all_estimated_label.append(trial_estimated_label)
    # Save data
    save_base = config["trial_data_filename"].split("/")[-1].split(".")[0]
    np.save("../common_data/multi_label/{}-window{}-{}_intercept-multi_labels.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), all_estimated_label)
    np.save("../common_data/multi_label/{}-window{}-{}_intercept-handcrafted_labels.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), handcrafted_labels)
    np.save("../common_data/multi_label/{}-window{}-{}_intercept-matching_rate.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_matching_rate)
    np.save("../common_data/multi_label/{}-window{}-{}_intercept-trial_weight_main.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_weight_main)
    np.save("../common_data/multi_label/{}-window{}-{}_intercept-trial_weight_rest.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_weight_rest)
    np.save("../common_data/multi_label/{}-window{}-{}_intercept-Q.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_Q)
    # Report
    print("Average matching rate : ", np.mean(trial_matching_rate))
    print("Min matching rate : ", np.min(trial_matching_rate))
    print("Max matching rate : ", np.max(trial_matching_rate))


def incrementalAnalysis(config):
    # Read trial data
    # agent_name = config["incremental_data_filename"]
    # agents_list = ["{}_Q".format(each) for each in agent_name]
    window = config["incremental_window"]
    trial_data = readTrialData(config["incremental_data_filename"])
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)
    trial_index = np.arange(trial_num)
    if config["incremental_num_trial"] is not None:
        if config["incremental_num_trial"] < trial_num:
            trial_index = np.random.choice(trial_index, config["incremental_num_trial"], replace=False)
    trial_data = [trial_data[each] for each in trial_index]
    trial_num = len(trial_data)
    print("Num of used trials : ", trial_num)
    # Incremental analysis
    incremental_agents_list = [
        ["local"],
        ["local", "pessimistic"],
        ["local", "global"],
        ["local", "pessimistic", "global"],
        ["local", "pessimistic", "global", "planned_hunting"],
        ["local", "pessimistic", "global", "planned_hunting", "suicide"]
    ]
    all_cr = []
    for trial_index, each in enumerate(trial_data):
        print("-"*15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("Trial name : ", trial_name)
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        trial_cr = []
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            agent_cr = []
            for agent_name in incremental_agents_list:
                # Construct optimizer
                params = [1 for _ in range(len(agent_name))]
                bounds = [[0, 1000] for _ in range(len(agent_name))]
                if config["need_intercept"]:
                    params.append(1)
                    bounds.append([-1000, 1000])
                cons = []  # construct the bounds in the form of constraints
                for par in range(len(bounds)):
                    l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
                    u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
                    cons.append(l)
                    cons.append(u)
                # estimation in the window
                func = lambda params: negativeLikelihood(
                    params,
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=False,
                    need_intercept=config["need_intercept"]
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
                # correct rate in the window
                _, estimated_prob = negativeLikelihood(
                    res.x,
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=True,
                    need_intercept=config["need_intercept"]
                )
                estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
                true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
                correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
                agent_cr.append(correct_rate)
            trial_cr.append([cur_step.file, cur_step.pacmanPos, cur_step.beans, agent_cr]) #TODO: save cur_step for later use
            print(agent_cr)
        print("Average correct rate for trial : ", np.nanmean([temp[-1] for temp in trial_cr]))
        all_cr.append(trial_cr)
    # save correct rate data
    if "incremental" not in os.listdir("../common_data"):
        os.mkdir("../common_data/incremental")
    np.save("../common_data/incremental/{}trial-window{}-incremental_cr-{}_intercept.npy".format(
        config["incremental_num_trial"], window, "w" if config["need_intercept"] else "wo"), all_cr)


def singleTrialFitting(config):
    # Read trial data
    agent_name = config["single_trial_agents"]
    agents_list = ["{}_Q".format(each) for each in agent_name]
    window = config["single_trial_window"]
    trial_data = readTrialData(config["single_trial_data_filename"])
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)

    trial_name_list = None
    # Old data
    # trial_name_list = ["10-1-Omega-02-Aug-2019-1.csv", "1-1-Omega-19-Aug-2019-1.csv",
    #                    "1-1-Omega-22-Jul-2019-1.csv", "1-4-Omega-21-Jun-2019-1.csv"]

    # trial_name_list = ["1-4-Omega-04-Jul-2019-1.csv", "10-2-Patamon-07-Jul-2019-1.csv", "10-3-Omega-09-Jul-2019-1.csv",
    # "10-7-Patamon-10-Aug-2019-1.csv", "11-1-Patamon-11-Jun-2019-1.csv", "13-2-Patamon-10-Sep-2019-1.csv"]

    # Best trials
    # "12-2-Patamon-13-Aug-2019-1.csv",
    trial_name_list = ["14-2-Patamon-10-Jul-2019-1.csv", "13-5-Patamon-21-Aug-2019-1.csv",
                       "13-3-Patamon-28-Jun-2019-1.csv", "14-1-Patamon-14-Jun-2019-1.csv"]



    record = []
    # trial_name_list = None
    if trial_name_list is not None and len(trial_name_list) > 0:
        temp_trial_Data = []
        for each in trial_data:
            if each[0] in trial_name_list:
                temp_trial_Data.append(each)
        trial_data = temp_trial_Data

    with open("../common_data/single_trial/trial_data.pkl", "wb") as file:
        pickle.dump(trial_data, file)

    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]

    all_hand_crafted = []
    all_estimated = []
    all_weight_main = []
    all_weight_rest = []
    all_Q = []

    multi_agent_list = [["global", "local"], ["pessimistic", "suicide", "planned_hunting"]]
    # Construct optimizer
    params = [1 for _ in range(len(agent_name))]
    bounds = [[0, 1000] for _ in range(len(agent_name))]
    if config["need_intercept"]:
        params.append(1)
        bounds.append([-1000, 1000])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    for trial_index, each in enumerate(trial_data):
        temp_record = []
        print("-"*15)
        trial_name = each[0]
        temp_record.append(trial_name)
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("Trial name : ", trial_name)
        # Hand-crafted label
        handcrafted_label = [_handcraftLabeling(X[label_list].iloc[index]) for index in range(X.shape[0])]
        handcrafted_label = handcrafted_label[window : -window]
        all_hand_crafted.append(handcrafted_label)
        label_not_nan_index = []
        for i, each in enumerate(handcrafted_label):
            if each is not None:
                label_not_nan_index.append(i)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight_main = np.zeros((len(window_index), 2 if not config["need_intercept"] else 3))
        temp_weight_rest = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))

        temp_contribution_main = np.zeros((len(window_index), 2))
        temp_contribution_rest = np.zeros((len(window_index), 3))
        cr = np.zeros((len(window_index), ))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        # For each trial, estimate agent weights through sliding windows

        trial_estimated_label = []
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
            # estimation in the window
            window_estimated_label = []
            for agent_index, agent_name in enumerate(multi_agent_list):
                # Construct optimizer
                params = [1 for _ in range(len(agent_name))]
                bounds = [[0, 1000] for _ in range(len(agent_name))]
                if config["need_intercept"]:
                    params.append(1)
                    bounds.append([-1000, 1000])
                cons = []  # construct the bounds in the form of constraints
                for par in range(len(bounds)):
                    l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
                    u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
                    cons.append(l)
                    cons.append(u)
                # estimation in the window
                func = lambda params: negativeLikelihood(
                    params,
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=False,
                    need_intercept=config["need_intercept"]
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
                if agent_index == 0:
                    temp_weight_main[centering_index, :] = res.x
                    contribution = temp_weight_main[centering_index, :-1] * \
                                   [scaleOfNumber(each) for each in
                                    np.max(np.abs(temp_trial_Q[centering_index, :, [0, 1], :]), axis=(1, 2))]
                    temp_contribution_main[centering_index, :] = contribution
                    # contribution = temp_weight_main[centering_index, :-1] * \
                    #                [each for each in
                    #                 np.nanmean(temp_trial_Q[centering_index, :, [0, 1], :], axis=(1, 2))]
                else:
                    temp_weight_rest[centering_index, :] = res.x
                    contribution = temp_weight_rest[centering_index, :-1] * \
                                   [scaleOfNumber(each) for each in
                                    np.max(np.abs(temp_trial_Q[centering_index, :, [2, 3, 4], :]), axis=(1, 2))]
                    temp_contribution_rest[centering_index, :] = contribution
                    # contribution = temp_weight_rest[centering_index, :-1] * \
                    #                [each for each in
                    #                 np.nanmean(temp_trial_Q[centering_index, :, [2, 3, 4], :], axis=(1, 2))]
                window_estimated_label.append(_estimationLabeling(contribution, agent_name))
            trial_estimated_label.append(window_estimated_label)
        matched_num = 0
        not_nan_num = 0
        for i in range(len(handcrafted_label)):
            if handcrafted_label[i] is not None:
                not_nan_num += 1
                if len(np.intersect1d(handcrafted_label[i], trial_estimated_label[i])) > 0:
                    matched_num += 1
        print(" Trial label matching rate : ", matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")

        temp_record.append(temp_weight_main)
        temp_record.append(temp_weight_rest)
        temp_record.append(temp_contribution_main)
        temp_record.append(temp_contribution_rest)
        temp_record.append(trial_estimated_label)
        temp_record.append(handcrafted_label)
        record.append(temp_record)

        all_weight_main.append(temp_weight_main)
        all_weight_rest.append(temp_weight_rest)
        all_estimated.append(trial_estimated_label)

        # # Estimated labels
        # #TODO: W*Q, normalization
        # for i in range(len(weight)):
        #         weight[i, :-1] = weight[i, :-1] * [scaleOfNumber(each) for each in np.nanmax(np.abs(trial_Q[i]), axis=(0, 2))]
        # all_Q.append(trial_Q)
        # if config["need_intercept"]:
        #     estimated_label = [_estimationLabeling(each, agent_name) for each in weight[:,:-1]]
        # else:
        #     estimated_label = [_estimationLabeling(each, agent_name) for each in weight]
        # all_estimated.append(estimated_label)
        # # is_matched = [(estimated_label[i] in handcrafted_label[i]) for i in label_not_nan_index]
        # is_matched = [(len(np.intersect1d(estimated_label[i], handcrafted_label[i])) > 0) for i in label_not_nan_index]
        # print("Label matching rate : ", np.sum(is_matched) / len(is_matched))

        estimated_label = [
            [
                _estimationLabeling(temp_contribution_main[index], ["global", "local"]),
                _estimationLabeling(temp_contribution_rest[index], ["pessimistic", "suicide", "planned_hunting"])
            ]
            for index in range(len(temp_contribution_main))
        ]

        # Plot weight variation of this trial
        agent_color = {
            "local": "red",
            "global": "blue",
            "pessimistic": "green",
            "suicide": "cyan",
            "planned_hunting": "magenta"
        }
        # normalization
        for index in range(temp_weight_main.shape[0]):
            if config["need_intercept"]:
                temp_weight_main[index, :-1] = temp_weight_main[index, :-1] / np.linalg.norm(temp_weight_main[index, :-1])
            else:
                temp_weight_main[index, :] = temp_weight_main[index, :] / np.linalg.norm(temp_weight_main[index, :])
        for index in range(temp_weight_rest.shape[0]):
            if config["need_intercept"]:
                temp_weight_rest[index, :-1] = temp_weight_rest[index, :-1] / np.linalg.norm(temp_weight_rest[index, :-1])
            else:
                temp_weight_rest[index, :] = temp_weight_rest[index, :] / np.linalg.norm(temp_weight_rest[index, :])

        plt.title(trial_name, fontsize = 15)
        for index in range(len(multi_agent_list[0])):
            plt.plot(temp_weight_main[:, index], color=agent_color[multi_agent_list[0][index]], ms=3, lw=5,
                     label=multi_agent_list[0][index])
        # for pessimistic agent
        plt.plot(temp_weight_rest[:, 0], color=agent_color["pessimistic"], ms=3, lw=5, label="pessimistic")
        plt.ylabel("Normalized Agent Weight", fontsize=20)
        plt.xlim(0, temp_weight_main.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(0.0, 1.1)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name))
        plt.show()

        # plt.subplot(2, 1, 2)
        # for index in range(len(multi_agent_list[1])):
        #     plt.plot(temp_weight_rest[:, index], color=agent_color[multi_agent_list[1][index]], ms=3, lw=5,
        #              label=multi_agent_list[1][index])
        # plot hand-crafted label
        # TODO: what about None value; multiple hand-crafted labels

        plt.subplot(2, 1, 1)
        # first line for hand-crafted, second line for estimated
        label_ind = {"global": 1, "local": 2}
        label_mat = np.zeros((2, len(handcrafted_label)), dtype = np.int)
        for index in range(label_mat.shape[1]):
            hand = handcrafted_label[index]
            est = estimated_label[index]
            if hand is None:
                label_mat[:, index] = [0, 0]
                continue
            # for hand-crafted
            if "local" in hand or "global" in hand:
                if "local" in hand:
                    label_mat[0, index] = label_ind["local"]
                else:
                    label_mat[0, index] = label_ind["global"]
            # for estimated
            label_mat[1, index] = label_ind[est[0]]
        colors = {"white": 0, "blue": 1, "red": 2}
        # "lightblue": 6, "blue": 7,"lightcoral": 8, "red": 9, "brown": 10,"violet": 11, "blueviolet": 12, "indigo": 13, "khaki": 14, "orange": 15, "pink": 16, "black": 17}
        l_colors = sorted(colors, key=colors.get)
        seaborn.heatmap(label_mat, yticklabels=["hand-crafted", "estimated"], cbar = False, linewidths=.5, cmap=l_colors, vmin=0, vmax=len(colors)-1, square=False)
        plt.xticks(np.arange(label_mat.shape[1]), list(range(label_mat.shape[1])))

        plt.subplot(2, 1, 2)
        # first line for hand-crafted, second line for estimated
        label_ind = {"pessimistic": 3}
        label_mat = np.zeros((2, len(handcrafted_label)), dtype=np.int)
        for index in range(label_mat.shape[1]):
            hand = handcrafted_label[index]
            est = estimated_label[index]
            if hand is None:
                label_mat[:, index] = [0, 0]
                continue
            # for hand-crafted
            if "pessimistic" in hand:
                label_mat[0, index] = label_ind["pessimistic"]
            # for estimated
            if est[1] == "pessimistic":
                label_mat[1, index] = label_ind["pessimistic"]
            else:
                label_mat[1, index] = 0
        colors = {"white": 0, "green": 3}
        # "lightblue": 6, "blue": 7,"lightcoral": 8, "red": 9, "brown": 10,"violet": 11, "blueviolet": 12, "indigo": 13, "khaki": 14, "orange": 15, "pink": 16, "black": 17}
        l_colors = sorted(colors, key=colors.get)
        seaborn.heatmap(label_mat, yticklabels=["hand-crafted", "estimated"], cbar=False, linewidths=.5, cmap=l_colors,
                        vmin=0, vmax=len(colors) - 1, square = False)

        # for i in range(len(handcrafted_label)):
        #     if handcrafted_label[i] is not None:
        #         if len(handcrafted_label[i]) == 2:
        #             plt.fill_between(x=[i, i + 1], y1=0, y2=0.05, color=agent_color[handcrafted_label[i][0]])
        #             plt.fill_between(x=[i, i + 1], y1=0.05, y2=0.1, color=agent_color[handcrafted_label[i][1]])
        #         else:
        #             plt.fill_between(x=[i, i + 1], y1=0, y2=-0.1, color=agent_color[handcrafted_label[i][0]])
        #
        #
        # for i in range(len(estimated_label)):
        #     if handcrafted_label[i] is not None:
        #         plt.fill_between(x=[i, i + 1], y1=0, y2=-0.05, color=agent_color[estimated_label[i][0]])
        #         plt.fill_between(x=[i, i + 1], y1=-0.05, y2=-0.1, color=agent_color[estimated_label[i][1]])

        # plt.ylabel("Normalized Agent Weight", fontsize=20)
        # plt.xlim(0, temp_weight_rest.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=15)
        # plt.xticks(fontsize = 15)
        # plt.yticks(fontsize=15)
        # plt.ylim(-0.1, 1.1)
        # plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name))

        plt.show()

        all_Q.append(temp_trial_Q)
    # # Save data
    np.save("../common_data/single_trial/two_ghosts_records.npy", record)
    # np.save("../common_data/single_trial/estimated_labels.npy", all_estimated)
    # np.save("../common_data/single_trial/agent_weights.npy", all_weight)
    # np.save("../common_data/single_trial/agent_contributions.npy", all_Q)


def singleTrialThreeFitting(config):
    # Read trial data
    agent_name = config["single_trial_agents"]
    agents_list = ["{}_Q".format(each) for each in agent_name]
    window = config["single_trial_window"]
    trial_data = readTrialData(config["single_trial_data_filename"])
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)

    trial_name_list = None
    # Old data
    # trial_name_list = ["10-1-Omega-02-Aug-2019-1.csv", "1-1-Omega-19-Aug-2019-1.csv",
    #                    "1-1-Omega-22-Jul-2019-1.csv", "1-4-Omega-21-Jun-2019-1.csv"]

    # trial_name_list = ["1-4-Omega-04-Jul-2019-1.csv", "10-2-Patamon-07-Jul-2019-1.csv", "10-3-Omega-09-Jul-2019-1.csv",
    # "10-7-Patamon-10-Aug-2019-1.csv", "11-1-Patamon-11-Jun-2019-1.csv", "13-2-Patamon-10-Sep-2019-1.csv"]

    # Best trials
    # "12-2-Patamon-13-Aug-2019-1.csv",
    trial_name_list = ["14-2-Patamon-10-Jul-2019-1.csv", "13-5-Patamon-21-Aug-2019-1.csv",
                       "13-3-Patamon-28-Jun-2019-1.csv", "14-1-Patamon-14-Jun-2019-1.csv"]



    record = []
    # trial_name_list = None
    if trial_name_list is not None and len(trial_name_list) > 0:
        temp_trial_Data = []
        for each in trial_data:
            if each[0] in trial_name_list:
                temp_trial_Data.append(each)
        trial_data = temp_trial_Data

    with open("../common_data/single_trial/trial_data.pkl", "wb") as file:
        pickle.dump(trial_data, file)

    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]

    all_hand_crafted = []
    all_estimated = []
    all_weight_main = []
    all_weight_rest = []
    all_Q = []

    agent_name = ["global", "local", "pessimistic"]
    # Construct optimizer
    params = [1 for _ in range(len(agent_name))]
    bounds = [[0, 1000] for _ in range(len(agent_name))]
    if config["need_intercept"]:
        params.append(1)
        bounds.append([-1000, 1000])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    for trial_index, each in enumerate(trial_data):
        temp_record = []
        print("-"*15)
        trial_name = each[0]
        temp_record.append(trial_name)
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("Trial name : ", trial_name)
        # Hand-crafted label
        handcrafted_label = [_handcraftLabeling(X[label_list].iloc[index]) for index in range(X.shape[0])]
        handcrafted_label = handcrafted_label[window : -window]
        all_hand_crafted.append(handcrafted_label)
        label_not_nan_index = []
        for i, each in enumerate(handcrafted_label):
            if each is not None:
                label_not_nan_index.append(i)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))
        # temp_weight_rest = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))

        temp_contribution = np.zeros((len(window_index), 3))
        # temp_contribution_rest = np.zeros((len(window_index), 3))
        cr = np.zeros((len(window_index), ))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        # For each trial, estimate agent weights through sliding windows

        trial_estimated_label = []
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
            # estimation in the window
            window_estimated_label = []
            # Construct optimizer
            params = [1 for _ in range(len(agent_name))]
            bounds = [[0, 1000] for _ in range(len(agent_name))]
            if config["need_intercept"]:
                params.append(1)
                bounds.append([-1000, 1000])
            cons = []  # construct the bounds in the form of constraints
            for par in range(len(bounds)):
                l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
                u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
                cons.append(l)
                cons.append(u)
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
                agent_name,
                return_trajectory=False,
                need_intercept=config["need_intercept"]
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
            contribution = temp_weight[centering_index, :-1] * \
                           [scaleOfNumber(each) for each in
                            np.max(np.abs(temp_trial_Q[centering_index, :, [0, 1, 2], :]), axis=(1, 2))]
            temp_contribution[centering_index, :] = contribution
            window_estimated_label.append(_estimationLabeling(contribution, agent_name))
            trial_estimated_label.append(window_estimated_label)

        matched_num = 0
        not_nan_num = 0
        for i in range(len(handcrafted_label)):
            if handcrafted_label[i] is not None:
                not_nan_num += 1
                if len(np.intersect1d(handcrafted_label[i], trial_estimated_label[i])) > 0:
                    matched_num += 1
        print(" Trial label matching rate : ", matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")

        temp_record.append(copy.deepcopy(temp_weight))
        temp_record.append(copy.deepcopy(temp_contribution))
        temp_record.append(copy.deepcopy(trial_estimated_label))
        temp_record.append(copy.deepcopy(handcrafted_label))
        record.append(copy.deepcopy(temp_record))

        all_weight_main.append(temp_weight)
        all_estimated.append(trial_estimated_label)

        # # Estimated labels
        # #TODO: W*Q, normalization
        # for i in range(len(weight)):
        #         weight[i, :-1] = weight[i, :-1] * [scaleOfNumber(each) for each in np.nanmax(np.abs(trial_Q[i]), axis=(0, 2))]
        # all_Q.append(trial_Q)
        # if config["need_intercept"]:
        #     estimated_label = [_estimationLabeling(each, agent_name) for each in weight[:,:-1]]
        # else:
        #     estimated_label = [_estimationLabeling(each, agent_name) for each in weight]
        # all_estimated.append(estimated_label)
        # # is_matched = [(estimated_label[i] in handcrafted_label[i]) for i in label_not_nan_index]
        # is_matched = [(len(np.intersect1d(estimated_label[i], handcrafted_label[i])) > 0) for i in label_not_nan_index]
        # print("Label matching rate : ", np.sum(is_matched) / len(is_matched))

        estimated_label = [
            [
                _estimationLabeling(temp_contribution[index], ["global", "local", "pessimistic"])
            ]
            for index in range(len(temp_contribution))
        ]

        # Plot weight variation of this trial
        agent_color = {
            "local": "red",
            "global": "blue",
            "pessimistic": "green",
            "suicide": "cyan",
            "planned_hunting": "magenta"
        }
        # normalization
        for index in range(temp_weight.shape[0]):
            if config["need_intercept"]:
                temp_weight[index, :-1] = temp_weight[index, :-1] / np.linalg.norm(temp_weight[index, :-1])
            else:
                temp_weight[index, :] = temp_weight[index, :] / np.linalg.norm(temp_weight[index, :])

        plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(temp_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=agent_name[index])

        for i in range(len(handcrafted_label)):
            if handcrafted_label[i] is not None:
                if len(handcrafted_label[i]) == 2:
                    plt.fill_between(x=[i, i + 1], y1=0, y2=-0.05, color=agent_color[handcrafted_label[i][0]])
                    plt.fill_between(x=[i, i + 1], y1=-0.05, y2=-0.1, color=agent_color[handcrafted_label[i][1]])
                else:
                    plt.fill_between(x=[i, i + 1], y1=0, y2=-0.1, color=agent_color[handcrafted_label[i][0]])

        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=20)
        plt.xlim(0, temp_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.1, 1.1)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name))
        plt.show()

        # plt.subplot(2, 1, 2)
        # for index in range(len(multi_agent_list[1])):
        #     plt.plot(temp_weight_rest[:, index], color=agent_color[multi_agent_list[1][index]], ms=3, lw=5,
        #              label=multi_agent_list[1][index])
        # plot hand-crafted label
        # TODO: what about None value; multiple hand-crafted labels

        # plt.subplot(2, 1, 1)
        # # first line for hand-crafted, second line for estimated
        # label_ind = {"global": 1, "local": 2}
        # label_mat = np.zeros((2, len(handcrafted_label)), dtype = np.int)
        # for index in range(label_mat.shape[1]):
        #     hand = handcrafted_label[index]
        #     est = estimated_label[index]
        #     if hand is None:
        #         label_mat[:, index] = [0, 0]
        #         continue
        #     # for hand-crafted
        #     if "local" in hand or "global" in hand:
        #         if "local" in hand:
        #             label_mat[0, index] = label_ind["local"]
        #         else:
        #             label_mat[0, index] = label_ind["global"]
        #     # for estimated
        #     label_mat[1, index] = label_ind[est[0]]
        # colors = {"white": 0, "blue": 1, "red": 2}
        # # "lightblue": 6, "blue": 7,"lightcoral": 8, "red": 9, "brown": 10,"violet": 11, "blueviolet": 12, "indigo": 13, "khaki": 14, "orange": 15, "pink": 16, "black": 17}
        # l_colors = sorted(colors, key=colors.get)
        # seaborn.heatmap(label_mat, yticklabels=["hand-crafted", "estimated"], cbar = False, linewidths=.5, cmap=l_colors, vmin=0, vmax=len(colors)-1, square=False)
        # plt.xticks(np.arange(label_mat.shape[1]), list(range(label_mat.shape[1])))

        # plt.subplot(2, 1, 2)
        # # first line for hand-crafted, second line for estimated
        # label_ind = {"pessimistic": 3}
        # label_mat = np.zeros((2, len(handcrafted_label)), dtype=np.int)
        # for index in range(label_mat.shape[1]):
        #     hand = handcrafted_label[index]
        #     est = estimated_label[index]
        #     if hand is None:
        #         label_mat[:, index] = [0, 0]
        #         continue
        #     # for hand-crafted
        #     if "pessimistic" in hand:
        #         label_mat[0, index] = label_ind["pessimistic"]
        #     # for estimated
        #     if est[1] == "pessimistic":
        #         label_mat[1, index] = label_ind["pessimistic"]
        #     else:
        #         label_mat[1, index] = 0
        # colors = {"white": 0, "green": 3}
        # # "lightblue": 6, "blue": 7,"lightcoral": 8, "red": 9, "brown": 10,"violet": 11, "blueviolet": 12, "indigo": 13, "khaki": 14, "orange": 15, "pink": 16, "black": 17}
        # l_colors = sorted(colors, key=colors.get)
        # seaborn.heatmap(label_mat, yticklabels=["hand-crafted", "estimated"], cbar=False, linewidths=.5, cmap=l_colors,
        #                 vmin=0, vmax=len(colors) - 1, square = False)


        #
        #
        # for i in range(len(estimated_label)):
        #     if handcrafted_label[i] is not None:
        #         plt.fill_between(x=[i, i + 1], y1=0, y2=-0.05, color=agent_color[estimated_label[i][0]])
        #         plt.fill_between(x=[i, i + 1], y1=-0.05, y2=-0.1, color=agent_color[estimated_label[i][1]])

        # plt.ylabel("Normalized Agent Weight", fontsize=20)
        # plt.xlim(0, temp_weight_rest.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=15)
        # plt.xticks(fontsize = 15)
        # plt.yticks(fontsize=15)
        # plt.ylim(-0.1, 1.1)
        # plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name))

        # plt.show()

        all_Q.append(temp_trial_Q)
    # # Save data
    np.save("../common_data/single_trial/two_ghosts_records.npy", record)
    print()
    # np.save("../common_data/single_trial/estimated_labels.npy", all_estimated)
    # np.save("../common_data/single_trial/agent_weights.npy", all_weight)
    # np.save("../common_data/single_trial/agent_contributions.npy", all_Q)


def simpleMLE(config):
    # Read trial data
    agent_name = config["MLE_agents"]
    agents_list = ["{}_Q".format(each) for each in agent_name]
    X, Y = readAllData(config["MLE_data_filename"], config["MLE_num_trial"])
    trial_num = len(np.unique(X.file.values))
    print("Num of trials : ", trial_num)
    print("Data shape : ", X.shape)
    # Construct optimizer
    params = [1 for _ in range(len(agent_name))]
    bounds = [[0, 1000] for _ in range(len(agent_name))]
    if config["need_intercept"]:
        params.append(1)
        bounds.append([-1000, 1000])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    func = lambda params: negativeLikelihood(
        params,
        X,
        Y,
        agent_name,
        return_trajectory=False,
        need_intercept=config["need_intercept"]
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
    # correct rate in the window
    _, estimated_prob = negativeLikelihood(
        res.x,
        X,
        Y,
        agent_name,
        return_trajectory=True,
        need_intercept=config["need_intercept"]
    )
    estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
    true_dir = Y.apply(lambda x: np.argmax(x)).values
    correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
    print("Weight : ", res.x)
    print("Correct rate : ", correct_rate)



# ===================================
#         VISUALIZATION
# ===================================
def plotWeightVariation(config, plot_sem = False, contribution = True, need_normalization = False, normalizing_type = None):
    # Determine agent names
    agent_list = config["agent_list"]
    all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    agent_color = {
        "local":"red",
        "global":"blue",
        "pessimistic":"green",
        "suicide":"cyan",
        "planned_hunting":"magenta"
    }
    # Read data
    # Weight shape : (num of trajectory, num of windows, num of used agents + intercept)
    # Correct rate shape : (num of trajectory, num of windows)
    # Q value shape : (num of trajectory, num of windows, whole window size, 5 agents, 4 directions)
    local2global_weight = np.load(config["local_to_global_agent_weight"])
    local2global_cr = np.load(config["local_to_global_cr"])
    local2global_Q = np.load(config["local_to_global_Q"])
    local2global_Q = local2global_Q[:, :, :, [all_agent_list.index(each) for each in agent_list[0]], :]

    global2local_weight = np.load(config["global_to_local_agent_weight"])
    global2local_cr = np.load(config["global_to_local_cr"])
    global2local_Q = np.load(config["global_to_local_Q"])
    global2local_Q = global2local_Q[:, :, :, [all_agent_list.index(each) for each in agent_list[2]], :]

    local2evade_weight = np.load(config["local_to_evade_agent_weight"])
    local2evade_cr = np.load(config["local_to_evade_cr"])
    local2evade_Q = np.load(config["local_to_evade_Q"])
    local2evade_Q = local2evade_Q[:, :, :, [all_agent_list.index(each) for each in agent_list[1]], :]

    evade2local_weight = np.load(config["evade_to_local_agent_weight"])
    evade2local_cr = np.load(config["evade_to_local_cr"])
    evade2local_Q = np.load(config["evade_to_local_Q"])
    evade2local_Q = evade2local_Q[:, :, :, [all_agent_list.index(each) for each in agent_list[3]], :]

    if contribution:
        # TODO: W*Q, normalization
        for i in range(local2global_weight.shape[0]):
            for j in range(local2global_weight.shape[1]):
                local2global_weight[i, j, :-1] = local2global_weight[i, j, :-1] \
                                                 * [scaleOfNumber(each) for each in
                                                    np.max(np.abs(local2global_Q[i, j, :, :, :]), axis=(0, 2))]


        # TODO: W*Q, normalization
        for i in range(global2local_weight.shape[0]):
            for j in range(global2local_weight.shape[1]):
                global2local_weight[i, j, :-1] = global2local_weight[i, j, :-1] \
                                                 * [scaleOfNumber(each) for each in
                                                    np.max(np.abs(global2local_Q[i, j, :, :, :]), axis=(0, 2))]


        # TODO: W*Q, normalization
        for i in range(local2evade_weight.shape[0]):
            for j in range(local2evade_weight.shape[1]):
                local2evade_weight[i, j, :-1] = local2evade_weight[i, j, :-1] \
                                                * [scaleOfNumber(each) for each in
                                                   np.max(np.abs(local2evade_Q[i, j, :, :, :]), axis=(0, 2))]


        # TODO: W*Q, normalization
        for i in range(evade2local_weight.shape[0]):
            for j in range(evade2local_weight.shape[1]):
                evade2local_weight[i, j, :-1] = evade2local_weight[i, j, :-1] \
                                                * [scaleOfNumber(each) for each in
                                                   np.max(np.abs(evade2local_Q[i, j, :, :, :]), axis=(0, 2))]

    # Plot weight variation
    plt.subplot(1 ,4, 1)
    agent_name = agent_list[0]
    plt.title("Local $\\rightarrow$ Global \n (avg cr = {avg:.3f})".format(avg = np.nanmean(local2global_cr)), fontsize = 20)
    avg_local2global_weight = np.nanmean(local2global_weight, axis = 0)
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_local2global_weight.shape[0]):
                #TODO: what if no intercept !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
                avg_local2global_weight[index, :-1]  = avg_local2global_weight[index, :-1] / np.max(avg_local2global_weight[index, :-1])
                local2global_weight[:, index, :-1] = local2global_weight[:, index, :-1] / np.max(local2global_weight[:, index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_local2global_weight.shape[0]):
                avg_local2global_weight[index, :-1]  = avg_local2global_weight[index, :-1] / np.linalg.norm(avg_local2global_weight[index, :-1])
                local2global_weight[:, index, :-1] = local2global_weight[:, index, :-1] / np.linalg.norm(local2global_weight[:, index, :-1])
        elif "all" == normalizing_type:
            avg_local2global_weight = avg_local2global_weight / np.max(avg_local2global_weight)
            local2global_weight = local2global_weight / np.max(local2global_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    # sem_local2global_weight  = scipy.stats.sem(local2global_weight, axis=0, nan_policy = "omit")
    sem_local2global_weight = np.std(local2global_weight, axis=0)
    centering_index = (len(avg_local2global_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_local2global_weight[centering_index - 4:centering_index + 4 + 1, index], color = agent_color[agent_name[index]], ms = 3, lw = 5,label = agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, 9),
                avg_local2global_weight[centering_index - 4:centering_index + 4 + 1, index] - sem_local2global_weight[centering_index - 4:centering_index + 4 + 1, index],
                avg_local2global_weight[centering_index - 4:centering_index + 4 + 1, index] + sem_local2global_weight[centering_index - 4:centering_index + 4 + 1, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.ylabel("Normalized Agent Weight", fontsize=20)
    plt.xlim(0, 8)
    x_ticks = [str(int(each)) for each in np.arange(0 - 4, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, 5, 1)])
    x_ticks_index = np.arange(len(x_ticks))
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize = 15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc = "lower center", fontsize=13, ncol=2)
    # plt.show()

    plt.subplot(1, 4, 2)
    agent_name = agent_list[2]
    plt.title("Global $\\rightarrow$ Local \n (avg cr = {avg:.3f})".format(avg = np.nanmean(global2local_cr)), fontsize = 20)
    avg_global2local_weight = np.nanmean(global2local_weight, axis=0)
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_global2local_weight.shape[0]):
                avg_global2local_weight[index, :-1]  = avg_global2local_weight[index, :-1] / np.max(avg_global2local_weight[index, :-1])
                global2local_weight[:, index, :-1] = global2local_weight[:, index, :-1] / np.max(global2local_weight[:, index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_global2local_weight.shape[0]):
                avg_global2local_weight[index, :-1]  = avg_global2local_weight[index, :-1] / np.linalg.norm(avg_global2local_weight[index, :-1])
                global2local_weight[:, index, :-1] = global2local_weight[:, index, :-1] / np.linalg.norm(global2local_weight[:, index, :-1])
        elif "all" == normalizing_type:
            avg_global2local_weight = avg_global2local_weight / np.max(avg_global2local_weight)
            global2local_weight = global2local_weight / np.max(global2local_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    # sem_global2local_weight = scipy.stats.sem(global2local_weight, axis=0, nan_policy = "omit")
    sem_global2local_weight = np.std(global2local_weight, axis=0)
    centering_index = (len(avg_global2local_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_global2local_weight[centering_index - 4: centering_index + 4 + 1, index], color=agent_color[agent_name[index]], ms=3, lw=5, label=agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, 9),
                avg_global2local_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_global2local_weight[centering_index - 4: centering_index + 4 + 1, index],
                avg_global2local_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_global2local_weight[centering_index - 4: centering_index + 4 + 1, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, 8)
    x_ticks = [str(int(each)) for each in np.arange(-4, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, 5, 1)])
    x_ticks_index = np.arange(len(x_ticks))
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc = "lower center", fontsize=13, ncol=2)

    plt.subplot(1, 4, 3)
    agent_name = agent_list[1]
    plt.title("Local $\\rightarrow$ Evade \n (avg cr = {avg:.3f})".format(avg=np.nanmean(local2evade_cr)), fontsize=20)
    avg_local2evade_weight = np.nanmean(local2evade_weight, axis=0)
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_local2evade_weight.shape[0]):
                avg_local2evade_weight[index, :-1] = avg_local2evade_weight[index, :-1] / np.max(
                    avg_local2evade_weight[index, :-1])
                local2evade_weight[:, index, :-1] = local2evade_weight[:, index, :-1] / np.max(
                    local2evade_weight[:, index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_local2evade_weight.shape[0]):
                avg_local2evade_weight[index, :-1] = avg_local2evade_weight[index, :-1] / np.linalg.norm(
                    avg_local2evade_weight[index, :-1])
                local2evade_weight[:, index, :-1] = local2evade_weight[:, index, :-1] / np.linalg.norm(
                    local2evade_weight[:, index, :-1])
        elif "all" == normalizing_type:
            avg_local2evade_weight = avg_local2evade_weight / np.max(avg_local2evade_weight)
            local2evade_weight = local2evade_weight / np.max(local2global_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    # sem_local2evade_weight = scipy.stats.sem(local2evade_weight, axis=0, nan_policy = "omit")
    sem_local2evade_weight = np.std(local2evade_weight, axis=0)
    centering_index = (len(avg_local2evade_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, 9),
                avg_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index],
                avg_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, 8)
    centering_point = (len(avg_local2evade_weight) - 1) / 2
    x_ticks = [str(int(each)) for each in np.arange(-4, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, 5, 1)])
    x_ticks = np.array(x_ticks)
    plt.xticks(np.arange(len(x_ticks)), x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=13, ncol=2)

    plt.subplot(1, 4, 4)
    agent_name = agent_list[1]
    plt.title("Evade $\\rightarrow$ Local \n (avg cr = {avg:.3f})".format(avg=np.nanmean(evade2local_cr)), fontsize=20)
    avg_evade2local_weight = np.nanmean(evade2local_weight, axis=0)
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_evade2local_weight.shape[0]):
                avg_evade2local_weight[index, :-1] = avg_evade2local_weight[index, :-1] / np.max(
                    avg_evade2local_weight[index, :-1])
                evade2local_weight[:, index, :-1] = evade2local_weight[:, index, :-1] / np.max(
                    evade2local_weight[:, index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_evade2local_weight.shape[0]):
                avg_evade2local_weight[index, :-1] = avg_evade2local_weight[index, :-1] / np.linalg.norm(
                    avg_evade2local_weight[index, :-1])
                evade2local_weight[:, index, :-1] = evade2local_weight[:, index, :-1] / np.linalg.norm(
                    evade2local_weight[:, index, :-1])
        elif "all" == normalizing_type:
            avg_evade2local_weight = avg_evade2local_weight / np.max(avg_evade2local_weight)
            evade2local_weight = evade2local_weight / np.max(evade2local_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    # sem_local2evade_weight = scipy.stats.sem(local2evade_weight, axis=0, nan_policy = "omit")
    sem_evade2local_weight = np.std(evade2local_weight, axis=0)
    centering_index = (len(avg_evade2local_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=agent_name[index])
        if plot_sem:
            plt.fill_between(
                np.arange(0, 9),
                avg_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index],
                avg_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index],
                # color="#dcb2ed",
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, 8)
    x_ticks = [str(int(each)) for each in np.arange(-4, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, 5, 1)])
    x_ticks = np.array(x_ticks)
    plt.xticks(np.arange(len(x_ticks)), x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=13, ncol=2)

    plt.show()


def plotIntegrationVariation(config, plot_sem = False, contribution = True, need_normalization = False, normalizing_type = None):
    # Determine agent names
    agent_list = config["agent_list"]
    all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    agent_color = {
        "local":"red",
        "global":"blue",
        "pessimistic":"green",
        "suicide":"cyan",
        "planned_hunting":"magenta"
    }
    # Read data
    # Weight shape : (num of trajectory, num of windows, num of used agents + intercept)
    # Correct rate shape : (num of trajectory, num of windows)
    # Q value shape : (num of trajectory, num of windows, whole window size, 5 agents, 4 directions)
    local2global_weight = np.load(config["integration_local_to_global_agent_weight"])
    local2global_cr = np.load(config["integration_local_to_global_cr"])
    local2global_Q = np.load(config["integration_local_to_global_Q"])
    local2global_Q = local2global_Q[:, :, [all_agent_list.index(each) for each in agent_list[0]], :]

    global2local_weight = np.load(config["integration_global_to_local_agent_weight"])
    global2local_cr = np.load(config["integration_global_to_local_cr"])
    global2local_Q = np.load(config["integration_global_to_local_Q"])
    global2local_Q = global2local_Q[:, :, [all_agent_list.index(each) for each in agent_list[2]], :]

    local2evade_weight = np.load(config["integration_local_to_evade_agent_weight"])
    local2evade_cr = np.load(config["integration_local_to_evade_cr"])
    local2evade_Q = np.load(config["integration_local_to_evade_Q"])
    local2evade_Q = local2evade_Q[:, :, [all_agent_list.index(each) for each in agent_list[1]], :]

    evade2local_weight = np.load(config["integration_evade_to_local_agent_weight"])
    evade2local_cr = np.load(config["integration_evade_to_local_cr"])
    evade2local_Q = np.load(config["integration_evade_to_local_Q"])
    evade2local_Q = evade2local_Q[:, :, [all_agent_list.index(each) for each in agent_list[3]], :]

    if contribution:
        # TODO: W*Q, normalization
        for i in range(local2global_weight.shape[0]):
            local2global_weight[i, :-1] = local2global_weight[i, :-1] * [scaleOfNumber(each) for each in np.max(np.abs(local2global_Q[i, :, :, :]), axis=(0, 2))]


        # TODO: W*Q, normalization
        for i in range(global2local_weight.shape[0]):
            for j in range(global2local_weight.shape[1]):
                global2local_weight[i, :-1] = global2local_weight[i, :-1] * [scaleOfNumber(each) for each in np.max(np.abs(global2local_Q[i, :, :, :]), axis=(0, 2))]


        # TODO: W*Q, normalization
        for i in range(local2evade_weight.shape[0]):
            for j in range(local2evade_weight.shape[1]):
                local2evade_weight[i, :-1] = local2evade_weight[i, :-1] * [scaleOfNumber(each) for each in np.max(np.abs(local2evade_Q[i, :, :, :]), axis=(0, 2))]


        # TODO: W*Q, normalization
        for i in range(evade2local_weight.shape[0]):
            for j in range(evade2local_weight.shape[1]):
                evade2local_weight[i, :-1] = evade2local_weight[i, :-1] * [scaleOfNumber(each) for each in np.max(np.abs(evade2local_Q[i, :, :, :]), axis=(0, 2))]

    # Plot weight variation
    plt.subplot(1 ,4, 1)
    agent_name = agent_list[0]
    plt.title("Local $\\rightarrow$ Global (avg cr = {avg:.3f})".format(avg = np.nanmean(local2global_cr)), fontsize = 20)
    avg_local2global_weight = local2global_weight
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_local2global_weight.shape[0]):
                #TODO: what if no intercept !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
                avg_local2global_weight[index, :-1]  = avg_local2global_weight[index, :-1] / np.max(avg_local2global_weight[index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_local2global_weight.shape[0]):
                avg_local2global_weight[index, :-1]  = avg_local2global_weight[index, :-1] / np.linalg.norm(avg_local2global_weight[index, :-1])
        elif "avg" == normalizing_type:
            for index in range(avg_local2global_weight.shape[0]):
                avg_local2global_weight[index, :-1]  = avg_local2global_weight[index, :-1] / np.sum(avg_local2global_weight[index, :-1])
        elif "all" == normalizing_type:
            avg_local2global_weight = avg_local2global_weight / np.max(avg_local2global_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    for index in range(len(agent_name)):
        plt.plot(avg_local2global_weight[:, index], color = agent_color[agent_name[index]], ms = 3, lw = 5,label = agent_name[index])
        # if plot_sem:
        #     plt.fill_between(
        #         np.arange(0, len(avg_local2global_weight)),
        #         avg_local2global_weight[:, index] - sem_local2global_weight[:, index],
        #         avg_local2global_weight[:, index] + sem_local2global_weight[:, index],
        #         # color="#dcb2ed",
        #         color=agent_color[agent_name[index]],
        #         alpha=0.3,
        #         linewidth=4
        #     )
    plt.ylabel("Normalized Agent Weight", fontsize=20)
    plt.xlim(0, avg_local2global_weight.shape[0] - 1)
    centering_point = (len(avg_local2global_weight) - 1) / 2
    if "window1" in config["local_to_global_agent_weight"].split("-"):
        x_ticks = [-13, -10, -7, -4, -1, "$\mathbf{c}$", 1, 4, 7, 10, 13]
        x_ticks_index = [1, 4, 7, 10, 13, 14, 15, 18, 21, 24, 27]
    elif "window3" in config["local_to_global_agent_weight"].split("-"):
        x_ticks = [-10, -7, -4, -1, "$\mathbf{c}$", 1, 4, 7, 10]
        x_ticks_index = [2,5,8,11,12,13,16,19,22]
    else:
        x_ticks = [str(int(each)) for each in np.arange(0 - centering_point, 0, 1)]
        x_ticks.append("$\\mathbf{c}$")
        x_ticks.extend([str(int(each)) for each in np.arange(1, len(avg_local2global_weight) - centering_point, 1)])
        if (avg_local2global_weight.shape[0] - 1) not in x_ticks:
            x_ticks.append(avg_local2global_weight.shape[0] - 1)
        x_ticks_index = np.arange(len(avg_local2global_weight))
    x_ticks = np.array(x_ticks)
    x_ticks_index = np.array(x_ticks_index)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize = 15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc = "lower center", fontsize=15, ncol=len(agent_name))
    # plt.show()

    plt.subplot(1, 4, 2)
    agent_name = agent_list[2]
    plt.title("Global $\\rightarrow$ Local  (avg cr = {avg:.3f})".format(avg = np.nanmean(global2local_cr)), fontsize = 20)
    avg_global2local_weight = global2local_weight
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_global2local_weight.shape[0]):
                avg_global2local_weight[index, :-1]  = avg_global2local_weight[index, :-1] / np.max(avg_global2local_weight[index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_global2local_weight.shape[0]):
                avg_global2local_weight[index, :-1]  = avg_global2local_weight[index, :-1] / np.linalg.norm(avg_global2local_weight[index, :-1])
        elif "avg" == normalizing_type:
            for index in range(avg_global2local_weight.shape[0]):
                avg_global2local_weight[index, :-1]  = avg_global2local_weight[index, :-1] / np.sum(avg_global2local_weight[index, :-1])
        elif "all" == normalizing_type:
            avg_global2local_weight = avg_global2local_weight / np.max(avg_global2local_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    for index in range(len(agent_name)):
        plt.plot(avg_global2local_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5, label=agent_name[index])
        # if plot_sem:
        #     plt.fill_between(
        #         np.arange(0, len(avg_global2local_weight)),
        #         avg_global2local_weight[:, index] - sem_global2local_weight[:, index],
        #         avg_global2local_weight[:, index] + sem_global2local_weight[:, index],
        #         # color="#dcb2ed",
        #         color=agent_color[agent_name[index]],
        #         alpha=0.3,
        #         linewidth=4
        #     )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, avg_global2local_weight.shape[0] - 1)
    centering_point = (len(avg_global2local_weight) - 1) / 2
    if "window1" in config["global_to_local_agent_weight"].split("-"):
        x_ticks = [-13, -10, -7, -4, -1, "$\mathbf{c}$", 1, 4, 7, 10, 13]
        x_ticks_index = [1, 4, 7, 10, 13, 14, 15, 18, 21, 24, 27]
    elif "window3" in config["global_to_local_agent_weight"].split("-"):
        x_ticks = [-10, -7, -4, -1, "$\mathbf{c}$", 1, 4, 7, 10]
        x_ticks_index = [2, 5, 8, 11, 12, 13, 16, 19, 22]
    else:
        x_ticks = [str(int(each)) for each in np.arange(0 - centering_point, 0, 1)]
        x_ticks.append("$\\mathbf{c}$")
        x_ticks.extend([str(int(each)) for each in np.arange(1, len(avg_global2local_weight) - centering_point, 1)])
        if (avg_local2global_weight.shape[0] - 1) not in x_ticks:
            x_ticks.append(avg_global2local_weight.shape[0] - 1)
        x_ticks_index = np.arange(len(avg_global2local_weight))
    x_ticks = np.array(x_ticks)
    x_ticks_index = np.array(x_ticks_index)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc = "lower center", fontsize=15, ncol=len(agent_name))

    plt.subplot(1, 4, 3)
    agent_name = agent_list[1]
    plt.title("Local $\\rightarrow$ Evade  (avg cr = {avg:.3f})".format(avg=np.nanmean(local2evade_cr)), fontsize=20)
    avg_local2evade_weight = local2evade_weight
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_local2evade_weight.shape[0]):
                avg_local2evade_weight[index, :-1] = avg_local2evade_weight[index, :-1] / np.max(
                    avg_local2evade_weight[index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_local2evade_weight.shape[0]):
                avg_local2evade_weight[index, :-1] = avg_local2evade_weight[index, :-1] / np.linalg.norm(
                    avg_local2evade_weight[index, :-1])
        elif "avg" == normalizing_type:
            for index in range(avg_local2evade_weight.shape[0]):
                avg_local2evade_weight[index, :-1]  = avg_local2evade_weight[index, :-1] / np.sum(avg_local2evade_weight[index, :-1])
        elif "all" == normalizing_type:
            avg_local2evade_weight = avg_local2evade_weight / np.max(avg_local2evade_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    for index in range(len(agent_name)):
        plt.plot(avg_local2evade_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=agent_name[index])
        # if plot_sem:
        #     plt.fill_between(
        #         np.arange(0, len(avg_local2evade_weight)),
        #         avg_local2evade_weight[:, index] - sem_local2evade_weight[:, index],
        #         avg_local2evade_weight[:, index] + sem_local2evade_weight[:, index],
        #         # color="#dcb2ed",
        #         color=agent_color[agent_name[index]],
        #         alpha=0.3,
        #         linewidth=4
        #     )
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
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=len(agent_name))

    plt.subplot(1, 4, 4)
    agent_name = agent_list[1]
    plt.title("Evade $\\rightarrow$ Local  (avg cr = {avg:.3f})".format(avg=np.nanmean(evade2local_cr)), fontsize=20)
    avg_evade2local_weight = evade2local_weight
    # normalization
    if need_normalization:
        if normalizing_type is None:
            raise ValueError("The type of normalizing should be specified!")
        elif "step" == normalizing_type:
            for index in range(avg_evade2local_weight.shape[0]):
                avg_evade2local_weight[index, :-1] = avg_evade2local_weight[index, :-1] / np.max(
                    avg_evade2local_weight[index, :-1])
        elif "sum" == normalizing_type:
            for index in range(avg_evade2local_weight.shape[0]):
                avg_evade2local_weight[index, :-1] = avg_evade2local_weight[index, :-1] / np.linalg.norm(
                    avg_evade2local_weight[index, :-1])
        elif "avg" == normalizing_type:
            for index in range(avg_evade2local_weight.shape[0]):
                avg_evade2local_weight[index, :-1]  = avg_evade2local_weight[index, :-1] / np.sum(avg_evade2local_weight[index, :-1])
        elif "all" == normalizing_type:
            avg_evade2local_weight = avg_evade2local_weight / np.max(avg_evade2local_weight)
        else:
            raise NotImplementedError("Undefined normalizing type {}!".format(normalizing_type))
    for index in range(len(agent_name)):
        plt.plot(avg_evade2local_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=agent_name[index])
        # if plot_sem:
            # plt.fill_between(
            #     np.arange(0, len(avg_evade2local_weight)),
            #     avg_evade2local_weight[:, index] - sem_evade2local_weight[:, index],
            #     avg_evade2local_weight[:, index] + sem_evade2local_weight[:, index],
            #     # color="#dcb2ed",
            #     color=agent_color[agent_name[index]],
            #     alpha=0.3,
            #     linewidth=4
            # )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, avg_evade2local_weight.shape[0] - 1)
    centering_point = (len(avg_evade2local_weight) - 1) / 2
    x_ticks = [str(int(each)) for each in np.arange(0 - centering_point, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, len(avg_evade2local_weight) - centering_point, 1)])
    if (avg_evade2local_weight.shape[0] - 1) not in x_ticks:
        x_ticks.append(avg_evade2local_weight.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(np.arange(len(avg_evade2local_weight)), x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=len(agent_name))

    plt.show()


def plotMultiLabelMatching(config):
    window = config["trial_window"]
    # Read data
    # trial_weight : (num of trials, num of windows, num of agents + 1)
    # trial_Q : (num of trials, num of windows, num of agents + 1, num of directions)
    estimated_labels = np.load(config["estimated_label_filename"], allow_pickle=True)
    handcrafted_labels = np.load(config["handcrafted_label_filename"], allow_pickle=True)
    trial_matching_rate = np.load(config["trial_matching_rate_filename"], allow_pickle=True)
    not_nan_trial_matching_rate = []
    for each in trial_matching_rate:
        if "Nan trial" != each:
            not_nan_trial_matching_rate.append(float(each))
    trial_matching_rate = not_nan_trial_matching_rate
    # # trial_cr = np.load(config["trial_cr_filename"], allow_pickle=True)
    # trial_weight = np.load(config["trial_weight_filename"], allow_pickle=True)
    # trial_weight = [trial_weight[index][:, :5] for index in range(len(trial_weight))] #TODO: what about wo_intercept
    # trial_Q = np.load(config["trial_Q_filename"], allow_pickle=True)
    # # TODO: W*Q, normalization
    # if contribution:
    #     for i in range(len(trial_weight)):
    #         for j in range(len(trial_weight[i])):
    #             trial_weight[i][j, :] = trial_weight[i][j, :] * [scaleOfNumber(each) for each in np.nanmax(np.abs(trial_Q[i][j]), axis = (0, 2))]
    #             # trial_weight[i][j, :] = trial_weight[i][j, :] * [each for each in
    #             #                                                  np.nanmax(np.abs(trial_Q[i][j]), axis=(0, 2))]
    # estimated_labels = []
    # for index in range(len(trial_weight)):
    #     temp_estimated_labels = [_estimationLabeling(each, config["correlation_agents"]) for each in trial_weight[index]]
    #     estimated_labels.append(temp_estimated_labels)
    #
    # trial_num = len(estimated_labels)
    # trial_matching_rate = []
    # # trial_correlation = []
    # is_matched = []
    # for index in range(trial_num):
    #     # estimated = np.array(_label2Index(estimated_labels[index]))
    #     # handcrafted = np.array(_label2Index(handcrafted_labels[index]))
    #     estimated = np.array(estimated_labels[index])
    #     handcrafted = np.array(handcrafted_labels[index])
    #     handcrafted = handcrafted[window:len(handcrafted) - window]
    #     # if len(estimated) != len(handcrafted):
    #     if len(estimated) != len(handcrafted):
    #         raise IndexError("len(estimated labels) != len(hand-crafted labels)")
    #     # what about None value
    #     not_none_index = np.where(handcrafted != None)
    #     if isinstance(not_none_index, tuple):
    #         not_none_index = not_none_index[0]
    #     if len(not_none_index) != 0:
    #         estimated = np.array(estimated)[not_none_index]
    #         handcrafted = np.array(handcrafted)[not_none_index]
    #         for i in range(len(estimated)):
    #             if len(np.intersect1d(estimated[i], handcrafted[i])) > 0:
    #                 is_matched.append(1)
    #             else:
    #                 is_matched.append(0)
    #         # matching_rate = np.sum(estimated == handcrafted) / len(estimated)
    #         matching_rate = np.sum(is_matched) / len(is_matched)
    #         # trial_correlation.append(scipy.stats.pearsonr(estimated, handcrafted))
    #         trial_matching_rate.append(matching_rate)

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
    # plt.title("Label Matching on 500 Trials (avg cr = {cr:.4f})".format(cr=np.mean([np.mean(each) for each in trial_cr])), fontsize = 20)

    plt.subplot(1, 2, 1)
    plt.title("Label Matching on {} Trials".format(len(trial_matching_rate)), fontsize = 20)
    plt.hist(trial_matching_rate)
    plt.xlabel("Correct Rate (estimated label = hand-crafted label)", fontsize = 20)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.ylabel("# of Trials", fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()

    # Plot confusion matrix
    # _________________________
    # |______|_local_|_global_| loca + evade | global + evade
    # | local|       |        |
    # |global|       |        |
    # | evade|
    # |-----------------------
    temp_handcrafted = []
    temp_estimated = []
    for i in handcrafted_labels:
        for j in i:
            temp_handcrafted.append(j)
    for i in estimated_labels:
        for j in i:
            temp_estimated.append(j)
    handcrafted_labels = temp_handcrafted
    estimated_labels = temp_estimated
    confusion_matrix = np.zeros((2, 2), dtype = np.int)
    used_index = []
    # for index in range(len(handcrafted_labels)):
    #     if handcrafted_labels[index] is not None and ("local" in handcrafted_labels[index] or "global" in handcrafted_labels[index] or "pessimistic" in handcrafted_labels[index]):
    #         if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
    #             continue
    #         used_index.append(index)
    for index in range(len(handcrafted_labels)):
        if handcrafted_labels[index] is not None and ("local" in handcrafted_labels[index] or "pessimistic" in handcrafted_labels[index]):
            if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
                continue
            used_index.append(index)
    estimated_labels = np.array(estimated_labels)[used_index]
    handcrafted_labels = np.array(handcrafted_labels)[used_index]
    weird_index = []
    local_num = 0
    local_evade_num = 0
    for index in range(len(used_index)):
        est = [each for each in estimated_labels[index]]
        hand = [each for each in handcrafted_labels[index]]

        if "local" in hand and "pessimistic" not in hand:
            local_num += 1
        if "local" in hand and "pessimistic" in hand:
            local_evade_num += 1

        if ("local" in est and "pessimistic" not in est) and ("local" in hand and "pessimistic" not in hand):
            confusion_matrix[0, 0] += 1
        if ("local" in est and "pessimistic" not in est) and ("local" in hand and "pessimistic" in hand):
            confusion_matrix[0, 1] += 1

        if ("local" in est and "pessimistic" in est)and ("local" in hand and "pessimistic" not in hand):
            confusion_matrix[1, 0] += 1
        if ("local" in est and "pessimistic" in est) and ("local" in hand and "pessimistic" in hand):
            confusion_matrix[1, 1] += 1

        # if ["local"] == est and ["local"] == hand:
        #     confusion_matrix[0, 0] += 1
        # elif ["local"] == est and ["global"] == hand:
        #     confusion_matrix[0, 1] += 1
        # elif ["local"] == est and ["local", "pessimistic"] == hand:
        #     confusion_matrix[0, 2] += 1
        # elif ["local"] == est and ["global", "pessimistic"] == hand:
        #     confusion_matrix[0, 3] += 1
        #
        # elif ["global"] == est and ["local"] == hand:
        #     confusion_matrix[1, 0] += 1
        # elif ["global"] == est and ["global"] == hand:
        #     confusion_matrix[1, 1] += 1
        # elif ["global"] == est and ["local", "pessimistic"] == hand:
        #     confusion_matrix[1, 2] += 1
        # elif["global"] == est and ["global", "pessimistic"] == hand:
        #     confusion_matrix[1, 3] += 1
        #
        # elif ["local", "pessimistic"] == est and ["local"] == hand:
        #     confusion_matrix[2, 0] += 1
        # elif ["local", "pessimistic"] == est and ["global"] == hand:
        #     confusion_matrix[2, 1] += 1
        # elif ["local", "pessimistic"] == est and ["local", "pessimistic"] == hand:
        #     confusion_matrix[2, 2] += 1
        # elif ["local", "pessimistic"] == est and ["global", "pessimistic"] == hand:
        #     confusion_matrix[2, 3] += 1
        #
        # elif ["global", "pessimistic"] == est and ["local"] == hand:
        #     confusion_matrix[3, 0] += 1
        # elif ["global", "pessimistic"] == est and ["global"] == hand:
        #     confusion_matrix[3, 1] += 1
        # elif ["global", "pessimistic"] == est and ["local", "pessimistic"] == hand:
        #     confusion_matrix[3, 2] += 1
        # elif ["global", "pessimistic"] == est and ["global", "pessimistic"] == hand:
        #     confusion_matrix[3, 3] += 1
        #
        # else:
        #     weird_index.append(index)

        # if "local" in est and "local" in hand:
        #     confusion_matrix[0, 0] += 1
        # if "local" in est and "global" in hand:
        #     confusion_matrix[0, 1] += 1
        # if "local" in est and "pessimistic" in hand:
        #     confusion_matrix[0, 2] += 1
        #
        # if "global" in est and "local" in hand:
        #     confusion_matrix[1, 0] += 1
        # if "global" in est and "global" in hand:
        #     confusion_matrix[1, 1] += 1
        # if "global" in est and "pessimistic" in hand:
        #     confusion_matrix[1, 2] += 1
        #
        # if "pessimistic" in est and "local" in hand:
        #     confusion_matrix[2, 0] += 1
        # if "pessimistic" in est and "global" in hand:
        #     confusion_matrix[2, 1] += 1
        # if "pessimistic" in est and "pessimistic" in hand:
        #     confusion_matrix[2, 2] += 1

    # F1 = 2TP / (2TP + FP + FN)
    # F1_score = 2 * confusion_matrix[0, 0] / (2 * confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])

    plt.subplot(1, 2, 2)
    # plt.title("F1 Score = {a:.3f}".format(a = F1_score), fontsize = 15)
    seaborn.heatmap(confusion_matrix,
                    annot = True, cmap='Blues',
                    # xticklabels = ["local", "global", "local + evade", "global + evade"],
                    # yticklabels = ["local", "global", "local + evade", "global + evade"],
                    xticklabels=["local \n ({a:.2%})".format(a = np.sum(confusion_matrix[:,0]) / local_num),
                                 "local + evade \n ({a:.2%})".format(a = np.sum(confusion_matrix[:,1]) / local_evade_num)],
                    yticklabels=["local", "local + evade"],
                    cbar = False, square = True, annot_kws = {"fontsize" : 15})
    plt.xlabel("Hand-Crafted Label", fontsize = 15)
    plt.ylabel("Estimated Label", fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()


def plotAllAgentMatching(config, contribution = True):
    window = config["all_agent_window"]
    # Read data
    # trial_weight : (num of trials, num of windows, num of agents + 1)
    # trial_Q : (num of trials, num of windows, num of agents + 1, num of directions)
    # estimated_labels = np.load(config["estimated_label_filename"], allow_pickle=True)
    handcrafted_labels = np.load(config["all_agent_handcrafted_label_filename"], allow_pickle=True)
    # trial_cr = np.load(config["trial_cr_filename"], allow_pickle=True)
    trial_weight = np.load(config["all_agent_trial_weight_filename"], allow_pickle=True)
    trial_weight = [trial_weight[index][:, :5] for index in range(len(trial_weight))] #TODO: what about wo_intercept
    trial_Q = np.load(config["all_agent_trial_Q_filename"], allow_pickle=True)
    # # TODO: W*Q, normalization
    if contribution:
        for i in range(len(trial_weight)):
            for j in range(len(trial_weight[i])):
                trial_weight[i][j, :] = trial_weight[i][j, :] * [scaleOfNumber(each) for each in np.nanmax(np.abs(trial_Q[i][j]), axis = (0, 2))]
                # trial_weight[i][j, :] = trial_weight[i][j, :] * [each for each in
                #                                                  np.nanmax(np.abs(trial_Q[i][j]), axis=(0, 2))]
    estimated_labels = []
    for index in range(len(trial_weight)):
        temp_estimated_labels = [_estimationLabeling(each, config["correlation_agents"]) for each in trial_weight[index]]
        estimated_labels.append(temp_estimated_labels)

    trial_num = len(estimated_labels)
    trial_matching_rate = []
    # trial_correlation = []
    is_matched = []
    for index in range(trial_num):
        # estimated = np.array(_label2Index(estimated_labels[index]))
        # handcrafted = np.array(_label2Index(handcrafted_labels[index]))
        estimated = np.array(estimated_labels[index])
        handcrafted = np.array(handcrafted_labels[index])
        handcrafted = handcrafted[window:- window]
        # if len(estimated) != len(handcrafted):
        if len(estimated) != len(handcrafted):
            raise IndexError("len(estimated labels) != len(hand-crafted labels)")
        # what about None value
        not_none_index = np.where(handcrafted != None)
        if isinstance(not_none_index, tuple):
            not_none_index = not_none_index[0]
        if len(not_none_index) != 0:
            estimated = np.array(estimated)[not_none_index]
            handcrafted = np.array(handcrafted)[not_none_index]
            for i in range(len(estimated)):
                if len(np.intersect1d(estimated[i], handcrafted[i])) > 0:
                    is_matched.append(1)
                else:
                    is_matched.append(0)
            # matching_rate = np.sum(estimated == handcrafted) / len(estimated)
            matching_rate = np.sum(is_matched) / len(is_matched)
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
    # plt.title("Label Matching on 500 Trials (avg cr = {cr:.4f})".format(cr=np.mean([np.mean(each) for each in trial_cr])), fontsize = 20)

    plt.subplot(1, 2, 1)
    plt.title("Label Matching on {} Trials".format(len(trial_matching_rate)), fontsize = 20)
    plt.hist(trial_matching_rate)
    plt.xlabel("Correct Rate (estimated label = hand-crafted label)", fontsize = 20)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.ylabel("# of Trials", fontsize=20)
    plt.yticks(fontsize=20)
    # plt.show()

    # Plot confusion matrix
    # _________________________
    # |______|_local_|_global_|
    # | local|       |        |
    # |global|       |        |
    # |-----------------------
    temp_handcrafted = []
    temp_estimated = []
    for i in handcrafted_labels:
        for j in i[window:-window]:
            temp_handcrafted.append(j)
    for i in estimated_labels:
        for j in i:
            temp_estimated.append(j)
    handcrafted_labels = temp_handcrafted
    estimated_labels = temp_estimated
    confusion_matrix = np.zeros((2, 2), dtype = np.int)
    used_index = []
    for index in range(len(handcrafted_labels)):
        if handcrafted_labels[index] is not None and ("local" in handcrafted_labels[index] or "global" in handcrafted_labels[index]):
            if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
                continue
            used_index.append(index)
    estimated_labels = np.array(estimated_labels)[used_index]
    handcrafted_labels = np.array(handcrafted_labels)[used_index]
    weird_index = []
    for index in range(len(used_index)):
        est = estimated_labels[index]
        hand = handcrafted_labels[index]
        if "local" in est and "local" in hand:
            confusion_matrix[0, 0] += 1
        elif "local" in est and "global" in hand:
            confusion_matrix[0, 1] += 1
        elif "global" in est and "local" in hand:
            confusion_matrix[1, 0] += 1
        elif "global" in est and "global" in hand:
            confusion_matrix[1, 1] += 1
        else:
            weird_index.append(index)
    # F1 = 2TP / (2TP + FP + FN)
    F1_score = 2 * confusion_matrix[0, 0] / (2 * confusion_matrix[0, 0] + confusion_matrix[0, 1] + confusion_matrix[1, 0])

    plt.subplot(1, 2, 2)
    plt.title("F1 Score = {a:.3f}".format(a = F1_score), fontsize = 15)
    seaborn.heatmap(confusion_matrix,
                    annot = True, cmap='Blues',
                    xticklabels = ["local", "global"], yticklabels = ["local", "global"],
                    cbar = False, square = True, annot_kws = {"fontsize" : 15})
    plt.xlabel("Hand-Crafted Label", fontsize = 15)
    plt.ylabel("Estimated Label", fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    plt.show()


def _checkError(config):
    window = config["trial_window"]
    # Read data
    # trial_weight : (num of trials, num of windows, num of agents + 1)
    # trial_Q : (num of trials, num of windows, num of agents + 1, num of directions)
    handcrafted_labels = np.load(config["handcrafted_label_filename"], allow_pickle=True)
    trial_cr = np.load(config["trial_cr_filename"], allow_pickle=True)
    trial_weight = np.load(config["trial_weight_filename"], allow_pickle=True)
    trial_weight = [trial_weight[index][:, :5] for index in range(len(trial_weight))]
    normalized_weight = copy.deepcopy(trial_weight)
    trial_Q = np.load(config["trial_Q_filename"], allow_pickle=True)
    # TODO: W*Q, normalization
    for i in range(len(trial_weight)):
        for j in range(len(trial_weight[i])):
            normalized_weight[i][j, :] = normalized_weight[i][j, :] * [scaleOfNumber(each) for each in np.max(np.abs(trial_Q[i][j]), axis = (0, 2))]
    estimated_labels = []
    for index in range(len(trial_weight)):
        temp_estimated_labels = [_estimationLabeling(each, config["correlation_agents"]) for each in normalized_weight[index]]
        estimated_labels.append(temp_estimated_labels)

    trial_num = len(estimated_labels)
    trial_matching_rate = []
    # trial_correlation = []
    is_matched = []
    wrong_matched = {"global": [], "local":[], "pessimistic":[], "planned_hunting":[], "suicide":[]}
    for index in range(trial_num):
        # estimated = np.array(_label2Index(estimated_labels[index]))
        # handcrafted = np.array(_label2Index(handcrafted_labels[index]))
        estimated = np.array(estimated_labels[index])
        handcrafted = np.array(handcrafted_labels[index])
        handcrafted = handcrafted[window:len(handcrafted) - window]
        temp_normalized_weight = normalized_weight[index]
        temp_weight = trial_weight[index]
        # if len(estimated) != len(handcrafted):
        if len(estimated) != len(handcrafted):
            raise IndexError("len(estimated labels) != len(hand-crafted labels)")
        # what about None value
        not_none_index = np.where(handcrafted != None)
        if isinstance(not_none_index, tuple):
            not_none_index = not_none_index[0]
        if len(not_none_index) != 0:
            estimated = np.array(estimated)[not_none_index]
            handcrafted = np.array(handcrafted)[not_none_index]
            temp_weight = temp_weight[not_none_index]
            temp_normalized_weight = temp_normalized_weight[not_none_index]
            for i in range(len(estimated)):
                if len(np.intersect1d(estimated[i], handcrafted[i])) > 0:
                    is_matched.append(1)
                else:
                    for k in handcrafted[i]:
                        wrong_matched[k].append([handcrafted[i], estimated[i], temp_normalized_weight[i], temp_weight[i]])
                    is_matched.append(0)
            # matching_rate = np.sum(estimated == handcrafted) / len(estimated)
            matching_rate = np.sum(is_matched) / len(is_matched)
            # trial_correlation.append(scipy.stats.pearsonr(estimated, handcrafted))
            trial_matching_rate.append(matching_rate)
    print("-"*15)
    print("Matching rate : ")
    print("Max : ", np.nanmax(trial_matching_rate))
    print("Min : ", np.nanmin(trial_matching_rate))
    print("Median : ", np.nanmedian(trial_matching_rate))
    print("Average : ", np.nanmean(trial_matching_rate))
    print("-"*15)
    local = wrong_matched["local"]
    global_data = wrong_matched["global"]
    planned = wrong_matched["planned_hunting"]
    suicide = wrong_matched["suicide"]
    pessimistic = wrong_matched["pessimistic"]
    print()


def plotBeanNumVSCr(config):
    print("-"*15)
    # trial name, pacman pos, beans, window cr for different agents
    bean_vs_cr = np.load(config["bean_vs_cr_filename"], allow_pickle = True)
    bean_num = []
    agent_cr = []
    for i in bean_vs_cr:
        for j in i:
            bean_num.append(len(j[2]))
            agent_cr.append(j[3])
    # bean_num = [len(each[2]) if isinstance(each[2], list) else 0 for each in bean_vs_cr]
    # agent_cr  = [each[3] for each in bean_vs_cr]
    max_bean_num = max(bean_num)
    min_bean_num = min(bean_num)
    print("Max bean num : ", max_bean_num)
    print("Min bean num : ", min_bean_num)
    agent_index = [0, 2, 3, 4, 5] # (local, + global, + pessimistic, + planned hunting, +suicide)
    first_phase_agent_cr = [] # num of beans <= 10
    second_phase_agent_cr = [] # 10 < num of beans < 80
    third_phase_agent_cr = [] # num of beans > 80
    # every bin
    for index, each in enumerate(bean_num):
        if each <= 10:
            first_phase_agent_cr.append(np.array(agent_cr[index])[agent_index])
        elif 10 < each < 80:
            second_phase_agent_cr.append(np.array(agent_cr[index])[agent_index])
        else:
            third_phase_agent_cr.append(np.array(agent_cr[index])[agent_index])

    # plotting
    x_ticks = ["local", "+ global", "+ pessi.", "+ plan.", "+suicide"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)

    plt.subplot(1, 3, 1)
    plt.title("# of Beans $\leqslant$ 10", fontsize  =20)
    avg_cr = np.mean(first_phase_agent_cr, axis = 0)
    var_cr = np.var(first_phase_agent_cr, axis = 0)
    plt.errorbar(x_index, avg_cr, yerr = var_cr, fmt = "k", mfc = "r", marker = "o", linestyle = "", ms = 15, elinewidth = 5)
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.ylabel("Direction Estimation Correct Rate", fontsize = 20)
    plt.yticks(fontsize=15)
    plt.ylim(0.7, 1.02)

    plt.subplot(1, 3, 2)
    plt.title("10 < # of Beans < 80", fontsize=20)
    avg_cr = np.mean(second_phase_agent_cr, axis=0)
    var_cr = np.var(second_phase_agent_cr, axis=0)
    plt.errorbar(x_index, avg_cr, yerr=var_cr, fmt="k", mfc="r", marker="o", linestyle="", ms=15, elinewidth=5)
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([])
    plt.ylim(0.7, 1.02)

    plt.subplot(1, 3, 3)
    plt.title("80 $\leqslant$ # of Beans", fontsize=20)
    avg_cr = np.mean(third_phase_agent_cr, axis=0)
    var_cr = np.var(third_phase_agent_cr, axis=0)
    plt.errorbar(x_index, avg_cr, yerr=var_cr, fmt="k", mfc="r", marker="o", linestyle="", ms=15, elinewidth=5)
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([])
    plt.ylim(0.7, 1)
    plt.show()



if __name__ == '__main__':
    # # Pre-estimation
    # preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None

    print("Args : ", sys.argv)
    if len(sys.argv) > 1:
        type = sys.argv[1]
        if "local_to_global" == type or "global_to_local" == type:
            agents = ["local", "global"]
        elif "local_to_evade" == type or "evade_to_local":
            agents = ["local", "pessimistic"]
        else:
            raise ValueError("Undefined transition type {}!".format(type))
    else:
        type = "local_to_evade"
        agents = ["local", "pessimistic"]

    config = {
        # TODO: ===================================
        # TODO:       Always set to True
        # TODO: ===================================
        "need_intercept" : True,

        # ==================================================================================
        #                       For Sliding Window Analysis
        # Filename
        "trajectory_data_filename": "../common_data/transition/{}-with_Q.pkl".format(type),
        # The window size
        "window": 1,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try": 5,
        # Agents: at least one of "global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting".
        # "agents": ["local", "global", "pessimistic", "suicide", "planned_hunting"],
        "agents": agents,
        # ==================================================================================

        # ==================================================================================
        #                       For Correlation Analysis and Multiple Label Analysis
        # Filename
        "trial_data_filename": "../common_data/trial/global15-local5-100_trial_data_new-with_Q.pkl",
        # The number of trials used for analysis
        "trial_num" :None,
        # Window size for correlation analysis
        "trial_window" : 3,
        "correlation_agents": ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        # ==================================================================================

        # ==================================================================================
        #                       For Single Trial Analysis
        # Filename
        "single_trial_data_filename": "../common_data/trial/global15-local10-100_trial_data_new-with_Q.pkl",
        # Window size for correlation analysis
        "single_trial_window": 3,
        "single_trial_agents": ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        # ==================================================================================

        # ==================================================================================
        #                       For Incremental Analysis
        # Filename
        "incremental_data_filename": "../common_data/trial/500_trial_data-with_Q.pkl",
        # Window size for correlation analysis
        "incremental_window": 3,
        "incremental_num_trial" : None,
        # ==================================================================================

        # ==================================================================================
        #                       For Simple MLE Analysis
        # Filename
        "MLE_data_filename": "../common_data/trial/500_trial_data-with_Q.pkl",
        # Window size for MLE analysis
        "MLE_num_trial": None,
        "MLE_agents": ["local", "pessimistic"],
        # ==================================================================================

        # ==================================================================================
        #                       For Experimental Results Visualization
        # this multi-label data is the true estimated label
        "estimated_label_filename" : "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-multi_labels.npy",
        "handcrafted_label_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-handcrafted_labels.npy",
        # "trial_cr_filename": "../common_data/multi_label/500_trial_data-with_Q-window3-w_intercept-trial_cr.npy",
        "trial_weight_main_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-trial_weight_main.npy",
        "trial_weight_rest_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-trial_weight_rest.npy",
        "trial_Q_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-Q.npy",
        "trial_matching_rate_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-matching_rate.npy",

        # ------------------------------------------------------------------------------------
        # "estimated_label_filename": "../common_data/multi_label/100_trial_data-with_Q-window3-w_intercept-multi_labels.npy",
        "all_agent_handcrafted_label_filename": "../common_data/trial/global16-local8/100_trial_data-with_Q-window3-w_intercept-handcrafted_labels.npy",
        # "trial_cr_filename": "../common_data/multi_label/500_trial_data-with_Q-window3-w_intercept-trial_cr.npy",
        "all_agent_trial_weight_filename": "../common_data/trial/global16-local8/100_trial_data-with_Q-window3-w_intercept-trial_weight.npy",
        "all_agent_trial_Q_filename": "../common_data/trial/global16-local8/100_trial_data-with_Q-window3-w_intercept-Q.npy",
        "all_agent_window" : 3,

        # ------------------------------------------------------------------------------------

        "local_to_global_agent_weight" : "../common_data/transition/local_to_global-window1-agent_weight-w_intercept.npy",
        "local_to_global_cr": "../common_data/transition/local_to_global-window1-cr-w_intercept.npy",
        "local_to_global_Q": "../common_data/transition/local_to_global-window1-Q-w_intercept.npy",

        "local_to_evade_agent_weight": "../common_data/transition/local_to_evade-window1-agent_weight-w_intercept.npy",
        "local_to_evade_cr": "../common_data/transition/local_to_evade-window1-cr-w_intercept.npy",
        "local_to_evade_Q": "../common_data/transition/local_to_evade-window1-Q-w_intercept.npy",

        "global_to_local_agent_weight": "../common_data/transition/global_to_local-window1-agent_weight-w_intercept.npy",
        "global_to_local_cr": "../common_data/transition/global_to_local-window1-cr-w_intercept.npy",
        "global_to_local_Q": "../common_data/transition/global_to_local-window1-Q-w_intercept.npy",

        "evade_to_local_agent_weight": "../common_data/transition/evade_to_local-window1-agent_weight-w_intercept.npy",
        "evade_to_local_cr": "../common_data/transition/evade_to_local-window1-cr-w_intercept.npy",
        "evade_to_local_Q": "../common_data/transition/evade_to_local-window1-Q-w_intercept.npy",

        "agent_list" : [["local", "global"], ["local", "pessimistic"], ["local", "global"], ["local", "pessimistic"]],

        "integration_local_to_global_agent_weight": "../common_data/integration_transition/local_to_global-agent_weight-w_intercept.npy",
        "integration_local_to_global_cr": "../common_data/integration_transition/local_to_global-cr-w_intercept.npy",
        "integration_local_to_global_Q": "../common_data/integration_transition/local_to_global-Q-w_intercept.npy",

        "integration_local_to_evade_agent_weight": "../common_data/integration_transition/local_to_evade-agent_weight-w_intercept.npy",
        "integration_local_to_evade_cr": "../common_data/integration_transition/local_to_evade-cr-w_intercept.npy",
        "integration_local_to_evade_Q": "../common_data/integration_transition/local_to_evade-Q-w_intercept.npy",

        "integration_global_to_local_agent_weight": "../common_data/integration_transition/global_to_local-agent_weight-w_intercept.npy",
        "integration_global_to_local_cr": "../common_data/integration_transition/global_to_local-cr-w_intercept.npy",
        "integration_global_to_local_Q": "../common_data/integration_transition/global_to_local-Q-w_intercept.npy",

        "integration_evade_to_local_agent_weight": "../common_data/integration_transition/evade_to_local-agent_weight-w_intercept.npy",
        "integration_evade_to_local_cr": "../common_data/integration_transition/evade_to_local-cr-w_intercept.npy",
        "integration_evade_to_local_Q": "../common_data/integration_transition/evade_to_local-Q-w_intercept.npy",

        # ------------------------------------------------------------------------------------

        "bean_vs_cr_filename" : "../common_data/incremental/window3-incremental_cr-wo_intercept.npy",
        "bin_size" : 10,
    }
    print("Window size for moving window analysis : ", config["window"])
    print("Window size for trial analysis : ", config["trial_window"])
    # ============ MOVING WINDOW =============
    movingWindowAnalysis(config)

    # singleTrialFitting(config)
    # singleTrialThreeFitting(config) # global, local, pessimistic

    # simpleMLE(config)


    # integrationAnalysis(config)

    # ============ Correlation =============
    # correlationAnalysis(config)
    # multipleLabelAnalysis(config)

    # incrementalAnalysis(config)

    # ============ VISUALIZATION =============
    # plotMultiLabelMatching(config)
    # plotAllAgentMatching(config, contribution = True)

    # plotWeightVariation(config, plot_sem = True, contribution = True, need_normalization = True, normalizing_type="sum") # step / sum / all
    # plotIntegrationVariation(config, plot_sem = True, contribution = True, need_normalization = True, normalizing_type="sum")

    # plotBeanNumVSCr(config)

    # _checkError(config)