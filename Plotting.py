'''
Description:
    Plotting all the figures.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    10 Nov. 2020
'''

import pickle
import pandas as pd
import numpy as np
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import copy
import seaborn
import os
import sys

from palettable.colorbrewer.qualitative import Dark2_7
from palettable.colorbrewer.diverging import RdBu_8
from palettable.scientific.sequential import Davos_5
from palettable.tableau import BlueRed_6
from palettable.cartocolors.qualitative import Vivid_5


sys.path.append("./Utility_Tree_Analysis")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber
from PathTreeAgent import PathTree
from SuicideAgent import SuicideAgent
from PlannedHuntingAgent import PlannedHuntingAgent



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


def _pessimisticProcesing(global_Q, local_Q, pess_Q):
    offset = np.max(np.abs(np.concatenate(pess_Q)))
    temp_global_Q = copy.deepcopy(global_Q)
    temp_local_Q = copy.deepcopy(local_Q)
    temp_pess_Q = copy.deepcopy(pess_Q)
    for index in range(len(temp_pess_Q)):
        if np.any(temp_pess_Q[index] < -5):
            non_zero = np.where(temp_pess_Q[index] != 0)
            temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
    # for index in range(len(temp_pess_Q)):
    #     non_zero = np.where(temp_pess_Q[index] != 0)
    #     temp_global_Q[index][non_zero] = temp_global_Q[index][non_zero] + offset
    #     temp_local_Q[index][non_zero] = temp_local_Q[index][non_zero] + offset
    #     temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
    return temp_global_Q, temp_local_Q, temp_pess_Q


def _makeChoice(prob):
    copy_estimated = copy.deepcopy(prob)
    if np.any(prob) < 0:
        available_dir_index = np.where(prob != 0)
        copy_estimated[available_dir_index] = copy_estimated[available_dir_index] - np.min(copy_estimated[available_dir_index]) + 1
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


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
                    all_data.local_Q[index] = all_data.local_Q[index - 2]
                    all_data.pessimistic_Q[index] = all_data.pessimistic_Q[index - 2]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index - 2]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index - 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index - 1]
                    all_data.local_Q[index] = all_data.local_Q[index - 1]
                    all_data.pessimistic_Q[index] = all_data.pessimistic_Q[index - 1]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index - 1]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index - 1]
            else:
                if isinstance(all_data.global_Q[index + 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index + 2]
                    all_data.local_Q[index] = all_data.local_Q[index + 2]
                    all_data.pessimistic_Q[index] = all_data.pessimistic_Q[index + 2]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index + 2]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index + 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index + 1]
                    all_data.local_Q[index] = all_data.local_Q[index + 1]
                    all_data.pessimistic_Q[index] = all_data.pessimistic_Q[index + 1]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index + 1]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index + 1]
    # Pre-processng pessimistic Q
    # TODO: check this
    all_data.global_Q, all_data.local_Q, all_data.pessimistic_Q = _pessimisticProcesing(all_data.global_Q, all_data.local_Q, all_data.pessimistic_Q)
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


def singleTrialThreeFitting(config):
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
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
    # ,
    trial_name_list = ["14-2-Patamon-10-Jul-2019-1.csv", "13-5-Patamon-21-Aug-2019-1.csv",
                       "13-3-Patamon-28-Jun-2019-1.csv", "14-1-Patamon-14-Jun-2019-1.csv", "12-2-Patamon-13-Aug-2019-1.csv"]

    # trial_name_list = ["13-5-Patamon-21-Aug-2019-1.csv",
    #                    "12-2-Patamon-13-Aug-2019-1.csv"]

    record = []
    # trial_name_list = None
    if trial_name_list is not None and len(trial_name_list) > 0:
        temp_trial_Data = []
        for each in trial_data:
            if each[0] in trial_name_list:
                temp_trial_Data.append(each)
        trial_data = temp_trial_Data

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
    bounds = [[0, 1] for _ in range(len(agent_name))]
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
        # temp_Q = []
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
        temp_record.append(copy.deepcopy(temp_trial_Q[:,:,[0, 1, 2], :]))
        record.append(copy.deepcopy(temp_record))

        all_weight_main.append(temp_weight)
        all_estimated.append(trial_estimated_label)
        all_Q.append(temp_trial_Q)


        estimated_label = [
            [
                _estimationLabeling(temp_contribution[index], ["global", "local", "pessimistic"])
            ]
            for index in range(len(temp_contribution))
        ]

        # Plot weight variation of this trial
        colors = RdBu_8.mpl_colors
        agent_color = {
            "local": colors[0],
            "pessimistic": colors[1],
            "global": colors[-3],
            "suicide": colors[-2],
            "planned_hunting": colors[-1]
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
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
                     label=label_name[agent_name[index]])

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
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon = False)
        plt.show()

    print()
    print()
    # # Save data
    # np.save("../common_data/single_trial/records.npy", record)
    # np.save("../common_data/single_trial/estimated_labels.npy", all_estimated)
    # np.save("../common_data/single_trial/agent_weights.npy", all_weight)
    # np.save("../common_data/single_trial/agent_contributions.npy", all_Q)

# ===================================
#         VISUALIZATION
# ===================================
def plotWeightVariation(config):
    # Determine agent names
    agent_list = config["agent_list"]
    all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    colors = RdBu_8.mpl_colors
    agent_color = {
        "local" : colors[0],
        "pessimistic" : colors[1],
        "global" : colors[-3],
        "suicide" : colors[-2],
        "planned_hunting" : colors[-1]
    }
    label_name = {
        "local": "local",
        "pessimistic": "evade",
        "global": "global",
        "suicide": "suicide",
        "planned_hunting": "attack"
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

    # Compute contributions: weight * Q value scale
    for i in range(local2global_weight.shape[0]):
        for j in range(local2global_weight.shape[1]):
            local2global_weight[i, j, :-1] = local2global_weight[i, j, :-1] \
                                             * [scaleOfNumber(each) for each in
                                                np.max(np.abs(local2global_Q[i, j, :, :, :]), axis=(0, 2))]
    for i in range(global2local_weight.shape[0]):
        for j in range(global2local_weight.shape[1]):
            global2local_weight[i, j, :-1] = global2local_weight[i, j, :-1] \
                                             * [scaleOfNumber(each) for each in
                                                np.max(np.abs(global2local_Q[i, j, :, :, :]), axis=(0, 2))]
    for i in range(local2evade_weight.shape[0]):
        for j in range(local2evade_weight.shape[1]):
            local2evade_weight[i, j, :-1] = local2evade_weight[i, j, :-1] \
                                            * [scaleOfNumber(each) for each in
                                               np.max(np.abs(local2evade_Q[i, j, :, :, :]), axis=(0, 2))]
    for i in range(evade2local_weight.shape[0]):
        for j in range(evade2local_weight.shape[1]):
            evade2local_weight[i, j, :-1] = evade2local_weight[i, j, :-1] \
                                            * [scaleOfNumber(each) for each in
                                               np.max(np.abs(evade2local_Q[i, j, :, :, :]), axis=(0, 2))]

    x_ticks = [int(each) for each in np.arange(0 - 4, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, 5, 1)])
    x_ticks_index = np.arange(len(x_ticks))

    # Plot weight variation
    plt.subplot(1 ,4, 1)
    agent_name = agent_list[0]
    plt.title("Local $\\rightarrow$ Global \n (avg cr = {avg:.3f})".format(avg = np.nanmean(local2global_cr)), fontsize = 20)
    avg_local2global_weight = np.nanmean(local2global_weight, axis = 0)
    # normalization
    for index in range(avg_local2global_weight.shape[0]):
        avg_local2global_weight[index, :-1] = avg_local2global_weight[index, :-1] / np.linalg.norm(avg_local2global_weight[index, :-1])
        local2global_weight[:, index, :-1] = local2global_weight[:, index, :-1] / np.linalg.norm(local2global_weight[:, index, :-1])
    sem_local2global_weight = np.std(local2global_weight, axis=0)
    centering_index = (len(avg_local2global_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_local2global_weight[centering_index - 4:centering_index + 4 + 1, index], color = agent_color[agent_name[index]], ms = 3, lw = 5,label = label_name[agent_name[index]])
        plt.fill_between(
            np.arange(0, 9),
            avg_local2global_weight[centering_index - 4:centering_index + 4 + 1, index] - sem_local2global_weight[centering_index - 4:centering_index + 4 + 1,index],
            avg_local2global_weight[centering_index - 4:centering_index + 4 + 1, index] + sem_local2global_weight[centering_index - 4:centering_index + 4 + 1,index],
            color=agent_color[agent_name[index]],
            alpha=0.3,
            linewidth=4
        )
    plt.ylabel("Normalized Agent Weight", fontsize=20)
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize = 15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc = "lower center", fontsize=13, ncol=2, frameon = False)
    # plt.show()

    plt.subplot(1, 4, 2)
    agent_name = agent_list[2]
    plt.title("Global $\\rightarrow$ Local \n (avg cr = {avg:.3f})".format(avg = np.nanmean(global2local_cr)), fontsize = 20)
    avg_global2local_weight = np.nanmean(global2local_weight, axis=0)
    # normalization
    for index in range(avg_global2local_weight.shape[0]):
        avg_global2local_weight[index, :-1] = avg_global2local_weight[index, :-1] / np.linalg.norm(
            avg_global2local_weight[index, :-1])
        global2local_weight[:, index, :-1] = global2local_weight[:, index, :-1] / np.linalg.norm(
            global2local_weight[:, index, :-1])
    sem_global2local_weight = np.std(global2local_weight, axis=0)
    centering_index = (len(avg_global2local_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_global2local_weight[centering_index - 4: centering_index + 4 + 1, index], color=agent_color[agent_name[index]], ms=3, lw=5, label = label_name[agent_name[index]])
        plt.fill_between(
            np.arange(0, 9),
            avg_global2local_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_global2local_weight[centering_index - 4: centering_index + 4 + 1,index],
            avg_global2local_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_global2local_weight[centering_index - 4: centering_index + 4 + 1,index],
            color=agent_color[agent_name[index]],
            alpha=0.3,
            linewidth=4
        )
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc = "lower center", fontsize=13, ncol=2, frameon = False)

    plt.subplot(1, 4, 3)
    agent_name = agent_list[1]
    plt.title("Local $\\rightarrow$ Evade \n (avg cr = {avg:.3f})".format(avg=np.nanmean(local2evade_cr)), fontsize=20)
    avg_local2evade_weight = np.nanmean(local2evade_weight, axis=0)
    # normalization
    for index in range(avg_local2evade_weight.shape[0]):
        avg_local2evade_weight[index, :-1] = avg_local2evade_weight[index, :-1] / np.linalg.norm(avg_local2evade_weight[index, :-1])
        local2evade_weight[:, index, :-1] = local2evade_weight[:, index, :-1] / np.linalg.norm(local2evade_weight[:, index, :-1])
    sem_local2evade_weight = np.std(local2evade_weight, axis=0)
    centering_index = (len(avg_local2evade_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label = label_name[agent_name[index]])
        plt.fill_between(
            np.arange(0, 9),
            avg_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_local2evade_weight[centering_index - 4: centering_index + 4 + 1,index],
            avg_local2evade_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_local2evade_weight[centering_index - 4: centering_index + 4 + 1,index],
            color=agent_color[agent_name[index]],
            alpha=0.3,
            linewidth=4
        )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=13, ncol=2, frameon = False)

    plt.subplot(1, 4, 4)
    agent_name = agent_list[1]
    plt.title("Evade $\\rightarrow$ Local \n (avg cr = {avg:.3f})".format(avg=np.nanmean(evade2local_cr)), fontsize=20)
    avg_evade2local_weight = np.nanmean(evade2local_weight, axis=0)
    # normalization
    for index in range(avg_evade2local_weight.shape[0]):
        avg_evade2local_weight[index, :-1] = avg_evade2local_weight[index, :-1] / np.linalg.norm(avg_evade2local_weight[index, :-1])
        evade2local_weight[:, index, :-1] = evade2local_weight[:, index, :-1] / np.linalg.norm(evade2local_weight[:, index, :-1])
    sem_evade2local_weight = np.std(evade2local_weight, axis=0)
    centering_index = (len(avg_evade2local_weight) -1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label = label_name[agent_name[index]])
        plt.fill_between(
            np.arange(0, 9),
            avg_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_evade2local_weight[centering_index - 4: centering_index + 4 + 1,index],
            avg_evade2local_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_evade2local_weight[centering_index - 4: centering_index + 4 + 1,index],
            color=agent_color[agent_name[index]],
            alpha=0.3,
            linewidth=4
        )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=13, ncol=2, frameon = False)
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


def plotThreeAgentMatching(config):
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

    print("-"*15)
    print("Matching rate : ")
    print("Max : ", np.nanmax(trial_matching_rate))
    print("Min : ", np.nanmin(trial_matching_rate))
    print("Median : ", np.nanmedian(trial_matching_rate))
    print("Average : ", np.nanmean(trial_matching_rate))

    colors = Davos_5.mpl_colors[1]
    plt.subplot(1, 2, 1)
    plt.title("Label Matching on {} Trials".format(len(trial_matching_rate)), fontsize = 20)
    plt.hist(trial_matching_rate, color=colors, rwidth = 0.9)
    plt.xlabel("Label Matching Rate", fontsize = 20)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.ylabel("# of Trials", fontsize=20)
    plt.yticks([], fontsize=20)
    # plt.show()

    # Plot confusion matrix
    # _________________________
    # |______|_local_|_global_| evade
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
    confusion_matrix = np.zeros((3, 3), dtype = np.int)
    used_index = []
    for index in range(len(handcrafted_labels)):
        if handcrafted_labels[index] is not None and \
                ("local" in handcrafted_labels[index] or
                 "global" in handcrafted_labels[index] or
                 "pessimistic" in handcrafted_labels[index]):
            if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
                continue
            used_index.append(index)
    estimated_labels = np.array(estimated_labels)[used_index]
    handcrafted_labels = np.array(handcrafted_labels)[used_index]

    weird_index = []
    for index in range(len(used_index)):
        est = [each for each in estimated_labels[index]]
        hand = [each for each in handcrafted_labels[index]]

        if "local" in est and "local" in hand:
            confusion_matrix[0, 0] += 1
        if "local" in est and "global" in hand:
            confusion_matrix[0, 1] += 1
        if "local" in est and "pessimistic" in hand:
            confusion_matrix[0, 2] += 1

        if "global" in est and "local" in hand:
            confusion_matrix[1, 0] += 1
        if "global" in est and "global" in hand:
            confusion_matrix[1, 1] += 1
        if "global" in est and "pessimistic" in hand:
            confusion_matrix[1, 2] += 1

        if "pessimistic" in est and "local" in hand:
            confusion_matrix[2, 0] += 1
        if "pessimistic" in est and "global" in hand:
            confusion_matrix[2, 1] += 1
        if "pessimistic" in est and "pessimistic" in hand:
            confusion_matrix[2, 2] += 1

    confusion_matrix = np.array(confusion_matrix, dtype = np.float)
    for col in range(3):
        confusion_matrix[:, col] = confusion_matrix[:, col] / np.sum(confusion_matrix[:, col])


    plt.subplot(1, 2, 2)
    plt.title("Confusion Matrix", fontsize = 20)
    seaborn.heatmap(confusion_matrix,
                    annot = True, cmap = "Blues", fmt = ".1%",
                    # xticklabels = ["local", "global", "local + evade", "global + evade"],
                    # yticklabels = ["local", "global", "local + evade", "global + evade"],
                    xticklabels=["local", "global", "evade"],
                    yticklabels=["local", "global", "evade"],
                    cbar = False, square = True, annot_kws = {"fontsize" : 20})
    plt.xlabel("Hand-Crafted Label", fontsize = 20)
    plt.ylabel("Estimated Label", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()


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

    colors = Vivid_5.mpl_colors

    plt.subplot(1, 3, 1)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("# of Beans $\leqslant$ 10", fontsize  =20)
    avg_cr = np.mean(first_phase_agent_cr, axis = 0)
    var_cr = np.var(first_phase_agent_cr, axis = 0)
    # plt.errorbar(x_index, avg_cr, yerr = var_cr, fmt = "k", mfc = "r", marker = "o", linestyle = "", ms = 15, elinewidth = 5)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.ylabel("Direction Correct Rate", fontsize = 20)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], [0.8, 0.85, 0.9, 0.95, 1.0], fontsize=15)
    plt.ylim(0.8, 1.0)

    plt.subplot(1, 3, 2)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("10 < # of Beans < 80", fontsize=20)
    avg_cr = np.mean(second_phase_agent_cr, axis=0)
    var_cr = np.var(second_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], [], fontsize=15)
    plt.ylim(0.8, 1.0)

    plt.subplot(1, 3, 3)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("80 $\leqslant$ # of Beans", fontsize=20)
    avg_cr = np.mean(third_phase_agent_cr, axis=0)
    var_cr = np.var(third_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([])
    plt.ylim(0.8, 1.0)
    plt.show()



if __name__ == '__main__':
    # Configurations
    pd.options.mode.chained_assignment = None

    config = {
        # TODO: ===================================
        # TODO:       Always set to True
        # TODO: ===================================
        "need_intercept" : True,
        "maximum_try": 5,

        "single_trial_data_filename": "./common_data/trial/100_trial_data_new-one_ghost-with_Q.pkl",
        # The number of trials used for analysis
        "trial_num": None,
        # Window size for correlation analysis
        "single_trial_window": 10,
        "single_trial_agents": ["global", "local", "pessimistic"],

        # ==================================================================================
        #                       For Experimental Results Visualization
        "estimated_label_filename": "./common_data/diff_pessimistic/non_negative_pess/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-multi_labels.npy",
        "handcrafted_label_filename": "./common_data/diff_pessimistic/non_negative_pess/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-handcrafted_labels.npy",
        "trial_weight_filename": "./common_data/diff_pessimistic/non_negative_pess/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-trial_weight.npy",
        "trial_Q_filename": "./common_data/diff_pessimistic/1non_negative_pess/00_trial_data_new-one_ghost-with_Q-window3-w_intercept-Q.npy",
        "trial_matching_rate_filename": "./common_data/diff_pessimistic/non_negative_pess/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-matching_rate.npy",
        "trial_window": 3,

        # ------------------------------------------------------------------------------------

        "local_to_global_agent_weight" : "./common_data/transition/local_to_global-window1-agent_weight-w_intercept.npy",
        "local_to_global_cr": "./common_data/transition/local_to_global-window1-cr-w_intercept.npy",
        "local_to_global_Q": "./common_data/transition/local_to_global-window1-Q-w_intercept.npy",

        "local_to_evade_agent_weight": "./common_data/transition/local_to_evade-window1-agent_weight-w_intercept.npy",
        "local_to_evade_cr": "./common_data/transition/local_to_evade-window1-cr-w_intercept.npy",
        "local_to_evade_Q": "./common_data/transition/local_to_evade-window1-Q-w_intercept.npy",

        "global_to_local_agent_weight": "./common_data/transition/global_to_local-window1-agent_weight-w_intercept.npy",
        "global_to_local_cr": "./common_data/transition/global_to_local-window1-cr-w_intercept.npy",
        "global_to_local_Q": "./common_data/transition/global_to_local-window1-Q-w_intercept.npy",

        "evade_to_local_agent_weight": "./common_data/transition/evade_to_local-window1-agent_weight-w_intercept.npy",
        "evade_to_local_cr": "./common_data/transition/evade_to_local-window1-cr-w_intercept.npy",
        "evade_to_local_Q": "./common_data/transition/evade_to_local-window1-Q-w_intercept.npy",

        "agent_list" : [["local", "global"], ["local", "pessimistic"], ["local", "global"], ["local", "pessimistic"]],

        # ------------------------------------------------------------------------------------

        "bean_vs_cr_filename" : "./common_data/incremental/500trial-window3-incremental_cr-w_intercept.npy",
    }

    # ============ VISUALIZATION =============
    # plotMultiLabelMatching(config)

    # plotThreeAgentMatching(config)

    # plotWeightVariation(config)

    # plotBeanNumVSCr(config)

    singleTrialThreeFitting(config)