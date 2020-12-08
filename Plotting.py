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

# plt.rc('font', family='CMU Serif', weight="roman")
# plt.rc('font', family='Myriad Pro')
#
# from matplotlib import rcParams
# rcParams['mathtext.default'] = 'regular'

params = {
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
}
plt.rcParams.update(params)


import copy
import seaborn
import os
import sys

from palettable.cmocean.diverging import Balance_6
from palettable.colorbrewer.diverging import RdBu_8, RdYlBu_5
from palettable.scientific.sequential import Davos_5
from palettable.scientific.diverging import Roma_5, Vik_5, Roma_3
from palettable.tableau import BlueRed_6
from palettable.cartocolors.qualitative import Vivid_5
from palettable.lightbartlein.diverging import BlueDarkRed18_18, BlueOrange12_5, BlueDarkRed18_4


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


def _PG(x, locs_df):
    PG = [0, 0]
    if x.ifscared1 < 3:
        if tuple(x.pacmanPos) != tuple(x.ghost1Pos):
            PG[0] = locs_df[tuple(x.pacmanPos)][tuple(x.ghost1Pos)]
        else:
            pass
    else:
        PG[0] = 100
    if x.ifscared2 < 3:
        if tuple(x.pacmanPos) != tuple(x.ghost2Pos):
            PG[1] = locs_df[tuple(x.pacmanPos)][tuple(x.ghost2Pos)]
        else:
            pass
    else:
        PG[1] = 100
    return PG


def _pessimisticProcesing(pess_Q, PG):
    offset = np.max(np.abs(np.concatenate(pess_Q)))
    temp_pess_Q = copy.deepcopy(pess_Q)
    for index in range(len(temp_pess_Q)):
        non_zero = np.where(temp_pess_Q[index] != 0)
        # if np.any(temp_pess_Q[index] < -5):
        if np.any(np.array(PG[index]) <= 10):
            temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
        else:
            temp_pess_Q[index][non_zero] = 0.0
    # for index in range(len(temp_pess_Q)):
    #     non_zero = np.where(temp_pess_Q[index] != 0)
    #     temp_global_Q[index][non_zero] = temp_global_Q[index][non_zero] + offset
    #     temp_local_Q[index][non_zero] = temp_local_Q[index][non_zero] + offset
    #     temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
    return temp_pess_Q


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
    locs_df = readLocDistance("./Utility_Tree_Analysis/extracted_data/dij_distance_map.csv")
    PG = all_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PG(x, locs_df),
        axis = 1
    )
    all_data.pessimistic_Q = _pessimisticProcesing(all_data.pessimistic_Q, PG)
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
    # print("Num of trials : ", trial_num)

    trial_name_list = None
    # Old data
    # trial_name_list = ["10-1-Omega-02-Aug-2019-1.csv", "1-1-Omega-19-Aug-2019-1.csv",
    #                    "1-1-Omega-22-Jul-2019-1.csv", "1-4-Omega-21-Jun-2019-1.csv"]

    # trial_name_list = ["1-4-Omega-04-Jul-2019-1.csv", "10-2-Patamon-07-Jul-2019-1.csv", "10-3-Omega-09-Jul-2019-1.csv",
    # "10-7-Patamon-10-Aug-2019-1.csv", "11-1-Patamon-11-Jun-2019-1.csv", "13-2-Patamon-10-Sep-2019-1.csv"]

    # Best trials
    # "13-2-Patamon-10-Sep-2019-1.csv", "10-3-Omega-09-Jul-2019-1.csv", "10-2-Patamon-07-Jul-2019-1.csv",
    # "10-7-Patamon-10-Aug-2019-1.csv", "11-1-Patamon-11-Jun-2019-1.csv", "13-5-Patamon-21-Aug-2019-1.csv",
    # "14-1-Patamon-14-Jun-2019-1.csv", "14-2-Patamon-10-Jul-2019-1.csv"

    # trial_name_list = ["14-2-Patamon-10-Jul-2019-1.csv"] # 111

    # trial_name_list = ["13-5-Patamon-21-Aug-2019-1.csv",
    #                    "12-2-Patamon-13-Aug-2019-1.csv"]
    all_trial_names = np.array([each[0] for each in trial_data])
    trial_name_list = np.random.choice(all_trial_names, 10, replace = True)
    trial_name_list = all_trial_names[np.where(np.array([each[1].shape[0] for each in trial_data]) == 80)]

    # trial_name_list = ["26-6-Omega-21-Aug-2019-1.csv"]
    record = []
    # trial_name_list = None
    if trial_name_list is not None and len(trial_name_list) > 0:
        temp_trial_Data = []
        for each in trial_data:
            if each[0] in trial_name_list:
                temp_trial_Data.append(each)
        trial_data = temp_trial_Data
    print("Num of trials : ", len(trial_data))
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
    for trial_index, each in enumerate(trial_data):
        temp_record = []
        print("-"*15)
        trial_name = each[0]
        temp_record.append(trial_name)
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("Index ", trial_index, " Trial name : ", trial_name)
        # Hand-crafted label
        handcrafted_label = [_handcraftLabeling(X[label_list].iloc[index]) for index in range(X.shape[0])]
        handcrafted_label = handcrafted_label[window : -window]
        all_hand_crafted.append(handcrafted_label)
        # label_not_nan_index = []
        # for i, each in enumerate(handcrafted_label):
        #     if each is not None:
        #         label_not_nan_index.append(i)
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
            params = [0 for _ in range(len(agent_name))]
            bounds = [[0, 10] for _ in range(len(agent_name))]
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
            window_estimated_label.append(_estimationThreeLabeling(contribution, agent_name))
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
            _estimationThreeLabeling(temp_contribution[index] / np.linalg.norm(temp_contribution[index]), agent_name)
            for index in range(len(temp_contribution))
        ]

        # Plot weight variation of this trial
        colors = RdYlBu_5.mpl_colors
        agent_color = {
            "local": colors[0],
            "pessimistic": colors[1],
            "global": colors[-1],
            "suicide": Balance_6.mpl_colors[2],
            "planned_hunting": colors[3]
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

        plt.figure(figsize = (18,13))
        plt.subplot(2, 1, 1)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(temp_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=20)
        plt.xlim(0, temp_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize = 20)
        x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=20, ncol = len(agent_name), frameon = False, bbox_to_anchor = (0.5, 1.2))
        # plt.show()

        # plt.figure(figsize=(13,5))
        plt.subplot(2, 1, 2)
        for i in range(len(handcrafted_label)):
            if handcrafted_label[i] is not None and (
                    "pessimistic" in handcrafted_label[i] or
                    "local" in handcrafted_label[i] or
                    "global" in handcrafted_label[i]
            ):
                handcrafted_label[i] = sorted(handcrafted_label[i])
                estimated_label[i] = sorted(estimated_label[i])

                # Hand-crafted labels
                if len(handcrafted_label[i]) > 2:
                    handcrafted_label[i] = handcrafted_label[i][:2]
                if len(handcrafted_label[i]) == 2:
                    if "local" in handcrafted_label[i] and "global" in handcrafted_label[i]:
                        handcrafted_label[i] = ["global", "global"]
                    # if "pessimistic" in handcrafted_label[i]:
                    #     plt.fill_between(x=[i, i + 1], y1=0.2, y2=0.3, facecolor=agent_color["pessimistic"])
                    # else:
                    plt.fill_between(x=[i, i + 1], y1=0.2, y2=0.25, facecolor=agent_color[handcrafted_label[i][0]])
                    plt.fill_between(x=[i, i + 1], y1=0.25, y2=0.3, facecolor=agent_color[handcrafted_label[i][1]])
                else:
                    plt.fill_between(x=[i, i + 1], y1=0.2, y2=0.3, facecolor=agent_color[handcrafted_label[i][0]])
                # Estimated labels
                if len(estimated_label[i]) == 1:
                    plt.fill_between(x=[i, i + 1], y1=0.0, y2=0.1, facecolor=agent_color[estimated_label[i][0]])
                else:
                    plt.fill_between(x=[i, i + 1], y1=0.0, y2=0.05, facecolor=agent_color[estimated_label[i][0]])
                    plt.fill_between(x=[i, i + 1], y1=0.05, y2=0.1, facecolor=agent_color[estimated_label[i][1]])
        plt.xlim(0, temp_weight.shape[0])
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([0.25, 0.05], ["Fitted Label", "Rule-Based Label"], fontsize=20)
        plt.ylim(-0.05, 0.35)
        plt.axis('off')
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
def _estimationThreeLabeling(contributions, all_agent_name):
    # global, local, pessimistic
    labels = []
    agent_name = ["global", "local"]
    if np.any(contributions[:2] > 0):
        labels.append(agent_name[np.argmax(contributions[:2])])
    # Threshold for different labels
    if "pessimistic" == all_agent_name[-1]:
        threshold = 0.0
    elif "planned_hunting" == all_agent_name[-1]:
        threshold = 0.0
    elif "suicide" == all_agent_name[-1]:
        threshold = 0.1
    else:
        raise NotImplementedError("Unknown agent {}!".format(all_agent_name[-1]))
    if contributions[-1] > threshold:
        labels.append(all_agent_name[-1])
    return labels


def _estimationMultipleLabeling(contributions, all_agent_name):
    # global, local, pessimistic
    labels = []
    agent_name = ["global", "local"]
    if np.any(contributions[:2] > 0):
        labels.append(agent_name[np.argmax(contributions[:2])])
    # Threshold for different labels
    pess_threshold = 0.1
    planned_threshold = 0.1
    suicide_threshold = 0.1
    if "pessimistic" in all_agent_name:
        if contributions[all_agent_name.index("pessimistic")] > pess_threshold:
            labels.append("pessimistic")
    if "suicide" in all_agent_name:
        if contributions[all_agent_name.index("suicide")] > suicide_threshold:
            labels.append("suicide")
    if "planned_hunting" in all_agent_name:
        if contributions[all_agent_name.index("planned_hunting")] > planned_threshold:
            labels.append("planned_hunting")
    return labels


def _estimationLocalEvadeSuicideLabeling(contributions):
    labels = []
    local_threshold = 0.0
    pess_threshold = 0.1
    suicide_threshold = 0.1
    if contributions[0] > local_threshold:
        labels.append("local")
    if contributions[1] > pess_threshold:
        labels.append("pessimistic")
    if contributions[2] > suicide_threshold:
        labels.append("suicide")
    # agent_name = ["pessimistic", "suicide"]
    # if np.any(contributions[1:] > 0):
    #     labels.append(agent_name[np.argmax(contributions[1:])])
    return labels


def plotWeightVariation(config):
    # Determine agent names
    agent_list = config["agent_list"]
    all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    colors = RdYlBu_5.mpl_colors
    agent_color = {
        "local" : colors[0],
        "pessimistic" : colors[1],
        "global" : colors[-1],
        "suicide" : Balance_6.mpl_colors[2],
        "planned_hunting" : colors[3]
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

    local2planned_weight = np.load(config["local_to_planned_agent_weight"])
    local2planned_cr = np.load(config["local_to_planned_cr"])
    local2planned_Q = np.load(config["local_to_planned_Q"])
    local2planned_Q = local2planned_Q[:, :, :, [all_agent_list.index(each) for each in agent_list[4]], :]

    local2suicide_weight = np.load(config["local_to_suicide_agent_weight"])
    local2suicide_cr = np.load(config["local_to_suicide_cr"])
    local2suicide_Q = np.load(config["local_to_suicide_Q"])
    local2suicide_Q = local2suicide_Q[:, :, :, [all_agent_list.index(each) for each in agent_list[5]], :]

    print("Local - Global : ", local2global_weight.shape[0])
    print("Global - Local : ", global2local_weight.shape[0])
    print("Local - Evade : ", local2evade_weight.shape[0])
    print("Evade - Local : ", evade2local_weight.shape[0])
    print("Local - Attack : ", local2planned_weight.shape[0])
    print("Local - Suicide : ", local2suicide_weight.shape[0])

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
    for i in range(local2planned_weight.shape[0]):
        for j in range(local2planned_weight.shape[1]):
            local2planned_weight[i, j, :-1] = local2planned_weight[i, j, :-1] \
                                              * [scaleOfNumber(each) for each in
                                                 np.max(np.abs(local2planned_Q[i, j, :, :, :]), axis=(0, 2))]
    for i in range(local2suicide_weight.shape[0]):
        for j in range(local2suicide_weight.shape[1]):
            local2suicide_weight[i, j, :-1] = local2suicide_weight[i, j, :-1] \
                                              * [scaleOfNumber(each) for each in
                                                 np.max(np.abs(local2suicide_Q[i, j, :, :, :]), axis=(0, 2))]

    x_ticks = [int(each) for each in np.arange(0 - 4, 0, 1)]
    x_ticks.append("$\\mathbf{c}$")
    x_ticks.extend([str(int(each)) for each in np.arange(1, 5, 1)])
    x_ticks_index = np.arange(len(x_ticks))

    plt.figure(figsize=(18, 19))
    # Plot weight variation
    plt.subplot(2 ,3, 1)
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
    plt.legend(loc = "lower center", fontsize=15, ncol=2, frameon = False)

    plt.subplot(2, 3, 4)
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
    plt.ylabel("Normalized Agent Weight", fontsize=20)
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=2, frameon=False)

    plt.subplot(2, 3, 2)
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
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [],fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=2, frameon = False)

    plt.subplot(2, 3, 5)
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
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [],fontsize=20)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=2, frameon = False)

    plt.subplot(2, 3, 3)
    agent_name = agent_list[4]
    plt.title("Local $\\rightarrow$ Attack \n (avg cr = {avg:.3f})".format(avg=np.nanmean(local2planned_cr)),
              fontsize=20)
    avg_local2planned_weight = np.nanmean(local2planned_weight, axis=0)
    # normalization
    for index in range(avg_local2planned_weight.shape[0]):
        avg_local2planned_weight[index, :-1] = avg_local2planned_weight[index, :-1] / np.linalg.norm(
            avg_local2planned_weight[index, :-1])
        local2planned_weight[:, index, :-1] = local2planned_weight[:, index, :-1] / np.linalg.norm(
            local2planned_weight[:, index, :-1])
    sem_local2planned_weight = np.std(local2planned_weight, axis=0)
    centering_index = (len(avg_local2planned_weight) - 1) // 2
    # centering_index += 5 # TODO: shift the centering
    for index in range(len(agent_name)):
        plt.plot(avg_local2planned_weight[centering_index - 4:centering_index + 4 + 1, index],
                 color=agent_color[agent_name[index]], ms=3, lw=5, label=label_name[agent_name[index]])
        plt.fill_between(
            np.arange(0, 9),
            avg_local2planned_weight[centering_index - 4:centering_index + 4 + 1, index] - sem_local2planned_weight[
                                                                                           centering_index - 4:centering_index + 4 + 1,
                                                                                           index],
            avg_local2planned_weight[centering_index - 4:centering_index + 4 + 1, index] + sem_local2planned_weight[
                                                                                           centering_index - 4:centering_index + 4 + 1,
                                                                                           index],
            color=agent_color[agent_name[index]],
            alpha=0.3,
            linewidth=4
        )
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [],fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=2, frameon=False)

    plt.subplot(2, 3, 6)
    agent_name = agent_list[5]
    plt.title("Local $\\rightarrow$ Suicide \n (avg cr = {avg:.3f})".format(avg=np.nanmean(local2suicide_cr)),
              fontsize=20)
    avg_local2suicide_weight = np.nanmean(local2suicide_weight, axis=0)
    # normalization
    for index in range(avg_local2suicide_weight.shape[0]):
        avg_local2suicide_weight[index, :-1] = avg_local2suicide_weight[index, :-1] / np.linalg.norm(
            avg_local2suicide_weight[index, :-1])
        local2suicide_weight[:, index, :-1] = local2suicide_weight[:, index, :-1] / np.linalg.norm(
            local2suicide_weight[:, index, :-1])
    sem_local2suicide_weight = np.std(local2suicide_weight, axis=0)
    centering_index = (len(avg_local2suicide_weight) - 1) // 2
    for index in range(len(agent_name)):
        plt.plot(avg_local2suicide_weight[centering_index - 4: centering_index + 4 + 1, index],
                 color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        plt.fill_between(
            np.arange(0, 9),
            avg_local2suicide_weight[centering_index - 4: centering_index + 4 + 1, index] - sem_local2suicide_weight[
                                                                                            centering_index - 4: centering_index + 4 + 1,
                                                                                            index],
            avg_local2suicide_weight[centering_index - 4: centering_index + 4 + 1, index] + sem_local2suicide_weight[
                                                                                            centering_index - 4: centering_index + 4 + 1,
                                                                                            index],
            color=agent_color[agent_name[index]],
            alpha=0.3,
            linewidth=4
        )
    # plt.ylabel("Agent Weight ($\\beta$)", fontsize=15)
    plt.xlim(0, 8)
    plt.xticks(x_ticks_index, x_ticks, fontsize=15)
    plt.xlabel("Time Step", fontsize=15)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [], fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=2, frameon=False)

    # plt.show()
    plt.savefig("./common_data/transition/transition_weight_dynamics.pdf")

    #TODO: =========================================================================================
    # centering_index += 5 # TODO: shift the centering
    plt.clf()
    agent_name = ["local", "suicide"]
    for index in range(len(agent_name)):
        plt.plot(avg_local2suicide_weight[:,index],
                 color=agent_color[agent_name[index]], ms=3, lw=5, label=label_name[agent_name[index]])

    plt.xlabel("Time Step", fontsize=15)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], [],fontsize=15)
    plt.ylim(0.0, 1.1)
    plt.legend(loc="lower center", fontsize=15, ncol=2, frameon=False)
    plt.show()


def plotThreeAgentMatching(config):
    # agent_name = config["trial_weight_filename"].split("/")[-2].split("_")
    # if "planned" in agent_name and "hunting" in agent_name:
    #     agent_name = agent_name[:-2]
    #     agent_name.append("planned_hunting")
    agent_name = config["trial_agent_name"]
    if len(agent_name) != 3:
        raise NotImplementedError("The agent list is {}!".format(agent_name))
    print("Agent name : ", agent_name)
    # Read data
    # trial_weight : (num of trials, num of windows, num of agents + 1)
    # trial_Q : (num of trials, num of windows, num of agents + 1, num of directions)
    handcrafted_labels = np.load(config["handcrafted_label_filename"].format("_".join(agent_name)), allow_pickle=True)
    trial_weight = np.load(config["trial_weight_filename"].format("_".join(agent_name)), allow_pickle = True)
    trial_Q = np.load(config["trial_Q_filename"].format("_".join(agent_name)), allow_pickle = True)
    trial_contributions = []
    trial_matching_rate = []
    estimated_labels = []
    for trial_index in range(len(trial_weight)):
        temp_contribution = []
        temp_labels = []
        is_same = []
        for centering_index in range(len(trial_weight[trial_index])):
            contribution = trial_weight[trial_index][centering_index, :-1] * \
                           [scaleOfNumber(each) for each in np.max(
                               np.abs(trial_Q[trial_index][centering_index, :, [0, 1, 2], :]),axis=(1, 2)
                           )]
            # normalization
            contribution = contribution / np.linalg.norm(contribution)
            temp_contribution.append(copy.deepcopy(contribution))
            # Labeling
            est = _estimationThreeLabeling(contribution, agent_name)
            temp_labels.append(copy.deepcopy(est))
            # Matching
            if handcrafted_labels[trial_index][centering_index] is not None:
                if len(np.intersect1d(est, handcrafted_labels[trial_index][centering_index])) > 0:
                    is_same.append(1)
                else:
                    is_same.append(0)
        trial_contributions.append(copy.deepcopy(temp_contribution))
        estimated_labels.append(copy.deepcopy(temp_labels))
        trial_matching_rate.append(np.sum(is_same)/len(is_same) if len(is_same) > 0 else None)

    # trial_matching_rate = np.load(config["trial_matching_rate_filename"], allow_pickle=True)
    not_nan_trial_matching_rate = []
    for each in trial_matching_rate:
        if each is not None:
            not_nan_trial_matching_rate.append(float(each))
    trial_matching_rate = not_nan_trial_matching_rate

    print("-"*15)
    print("Matching rate : ")
    print("Max : ", np.nanmax(trial_matching_rate))
    print("Min : ", np.nanmin(trial_matching_rate))
    print("Median : ", np.nanmedian(trial_matching_rate))
    print("Average : ", np.nanmean(trial_matching_rate))

    colors = Davos_5.mpl_colors[1]
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    # plt.title("Label Matching on {} Trials".format(len(trial_matching_rate)), fontsize = 20)
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
                 agent_name[-1] in handcrafted_labels[index]):
            if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
                continue
            used_index.append(index)
    estimated_labels = np.array(estimated_labels)[used_index]
    handcrafted_labels = np.array(handcrafted_labels)[used_index]

    weird_index = []
    for index in range(len(used_index)):
        est = [each for each in estimated_labels[index]]
        hand = [each for each in handcrafted_labels[index]]

        if ["local"] == est and ["local"] == hand:
            confusion_matrix[0, 0] += 1
        if ["local"] == est and ["global"] == hand:
            confusion_matrix[0, 1] += 1
        if ["local"] == est and [agent_name[-1]] == hand:
            confusion_matrix[0, 2] += 1

        if ["global"] == est and ["local" ]== hand:
            confusion_matrix[1, 0] += 1
        if ["global"] == est and ["global"] == hand:
            confusion_matrix[1, 1] += 1
        if ["global"] == est and [agent_name[-1]] == hand:
            confusion_matrix[1, 2] += 1

        if (agent_name[-1] in est and "local" not in est) and ["local"] == hand:
            confusion_matrix[2, 0] += 1
        if (agent_name[-1] in est and "global" not in est) and ["global"] == hand:
            confusion_matrix[2, 1] += 1
        if agent_name[-1] in est and [agent_name[-1]] == hand:
            confusion_matrix[2, 2] += 1

    confusion_matrix = np.array(confusion_matrix, dtype = np.float)
    for col in range(3):
        confusion_matrix[:, col] = confusion_matrix[:, col] / np.sum(confusion_matrix[:, col])


    plt.subplot(1, 2, 2)
    if "planned_hunting" in agent_name:
        agent_name[agent_name.index("planned_hunting")] = "attack"
    if "pessimistic" in agent_name:
        agent_name[agent_name.index("pessimistic")] = "evade"
    seaborn.heatmap(confusion_matrix,
                    annot = True, cmap = "binary", fmt = ".1%",
                    xticklabels = ["local", "global", agent_name[-1]],
                    yticklabels = ["local", "global", agent_name[-1]],
                    cbar = False, square = True, annot_kws = {"fontsize" : 20})
    plt.xlabel("Rule-Based Label", fontsize = 20)
    plt.ylabel("Fitted Label", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()


def plotAllAgentMatching(config):
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    agent_index = [["global", "local", "pessimistic", "suicide", "planned_hunting"].index(each) for each in agent_name]
    print("Agent name : ", agent_name)
    # Read data
    # trial_weight : (num of trials, num of windows, num of agents + 1)
    # trial_Q : (num of trials, num of windows, num of agents + 1, num of directions)
    handcrafted_labels = np.load(config["handcrafted_label_filename"].format("_".join(agent_name)), allow_pickle=True)
    trial_weight = np.load(config["trial_weight_filename"].format("_".join(agent_name)), allow_pickle = True)
    trial_Q = np.load(config["trial_Q_filename"].format("_".join(agent_name)), allow_pickle = True)
    trial_contributions = []
    trial_matching_rate = []
    estimated_labels = []
    for trial_index in range(len(trial_weight)):
        temp_contribution = []
        temp_labels = []
        is_same = []
        for centering_index in range(len(trial_weight[trial_index])):
            contribution = trial_weight[trial_index][centering_index, :-1] * \
                           [scaleOfNumber(each) for each in np.max(
                               np.abs(trial_Q[trial_index][centering_index, :, agent_index, :]),axis=(1, 2)
                           )]
            # normalization
            contribution = contribution / np.linalg.norm(contribution)
            temp_contribution.append(copy.deepcopy(contribution))
            # Labeling
            est = _estimationMultipleLabeling(contribution, agent_name)
            temp_labels.append(copy.deepcopy(est))
            # Matching
            if handcrafted_labels[trial_index][centering_index] is not None:
                if len(np.intersect1d(est, handcrafted_labels[trial_index][centering_index])) > 0:
                    is_same.append(1)
                else:
                    is_same.append(0)
        trial_contributions.append(copy.deepcopy(temp_contribution))
        estimated_labels.append(copy.deepcopy(temp_labels))
        trial_matching_rate.append(np.sum(is_same)/len(is_same) if len(is_same) > 0 else None)

    # trial_matching_rate = np.load(config["trial_matching_rate_filename"], allow_pickle=True)
    not_nan_trial_matching_rate = []
    for each in trial_matching_rate:
        if each is not None:
            not_nan_trial_matching_rate.append(float(each))
    trial_matching_rate = not_nan_trial_matching_rate

    print("-"*15)
    print("Matching rate : ")
    print("Max : ", np.nanmax(trial_matching_rate))
    print("Min : ", np.nanmin(trial_matching_rate))
    print("Median : ", np.nanmedian(trial_matching_rate))
    print("Average : ", np.nanmean(trial_matching_rate))

    colors = Davos_5.mpl_colors[1]
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    # plt.title("Label Matching on {} Trials".format(len(trial_matching_rate)), fontsize = 20)
    plt.hist(trial_matching_rate, color=colors, rwidth = 0.9)
    plt.xlabel("Label Matching Rate", fontsize = 20)
    plt.xlim(0, 1.0)
    plt.xticks(np.arange(0, 1.1, 0.1), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.ylabel("# of Trials", fontsize=20)
    plt.yticks([], fontsize=20)
    # plt.show()

    # Plot confusion matrix
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

    confusion_matrix = np.zeros((len(agent_name), len(agent_name)), dtype = np.int)
    used_index = []
    for index in range(len(handcrafted_labels)):
        if handcrafted_labels[index] is not None:
            if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
                continue
            used_index.append(index)
    estimated_labels = np.array(estimated_labels)[used_index]
    handcrafted_labels = np.array(handcrafted_labels)[used_index]

    weird_index = []
    for index in range(len(used_index)):
        est = [each for each in estimated_labels[index]]
        hand = [each for each in handcrafted_labels[index]]

        if ["local"] == est and ["local"] == hand:
            confusion_matrix[0, 0] += 1
        if ["local"] == est and ["global"] == hand:
            confusion_matrix[0, 1] += 1
        if ["local"] == est and ["pessimistic"] == hand:
            confusion_matrix[0, 2] += 1
        if ["local"] == est and ["suicide"] == hand:
            confusion_matrix[0, 3] += 1
        if ["local"] == est and ["planned_hunting"] == hand:
            confusion_matrix[0, 4] += 1

        if ["global"] == est and ["local" ]== hand:
            confusion_matrix[1, 0] += 1
        if ["global"] == est and ["global"] == hand:
            confusion_matrix[1, 1] += 1
        if ["global"] == est and ["pessimistic"] == hand:
            confusion_matrix[1, 2] += 1
        if ["global"] == est and ["suicide"] == hand:
            confusion_matrix[1, 3] += 1
        if ["global"] == est and ["planned_hunting"] == hand:
            confusion_matrix[1, 4] += 1

        if ("pessimistic" in est and "local" not in est) and ["local"] == hand:
            confusion_matrix[2, 0] += 1
        if ("pessimistic" in est and "global" not in est) and ["global"] == hand:
            confusion_matrix[2, 1] += 1
        if ("pessimistic" in est and "suicide" not in est) and ["pessimistic"] == hand:
            confusion_matrix[2, 2] += 1
        if ("pessimistic" in est and "suicide" not in est) and ["suicide"] == hand:
            confusion_matrix[2, 3] += 1
        if ("pessimistic" in est and "suicide" not in est) and ["planned_hunting"] == hand:
            confusion_matrix[2, 4] += 1

        if ("suicide" in est and "local" not in est) and ["local"] == hand:
            confusion_matrix[3, 0] += 1
        if ("suicide" in est and "global" not in est) and ["global"] == hand:
            confusion_matrix[3, 1] += 1
        if ("suicide" in est and "pessimistic" not in est) and ["pessimistic"] == hand:
            confusion_matrix[3, 2] += 1
        if ("suicide" in est and "pessimistic" not in est) and ["suicide"] == hand:
            confusion_matrix[3, 3] += 1
        if ("suicide" in est and "pessimistic" not in est) and ["planned_hunting"] == hand:
            confusion_matrix[3, 4] += 1

        # if ["suicide"] == est  and ["local"] == hand:
        #     confusion_matrix[3, 0] += 1
        # if ["suicide"] == est and ["global"] == hand:
        #     confusion_matrix[3, 1] += 1
        # if ["suicide"] == est and ["pessimistic"] == hand:
        #     confusion_matrix[3, 2] += 1
        # if ["suicide"] == est and ["suicide"] == hand:
        #     confusion_matrix[3, 3] += 1
        # if ["suicide"] == est and ["planned_hunting"] == hand:
        #     confusion_matrix[3, 4] += 1

        if ("planned_hunting" in est and "local" not in est) and ["local"] == hand:
            confusion_matrix[4, 0] += 1
        if ("planned_hunting" in est and "global" not in est) and ["global"] == hand:
            confusion_matrix[4, 1] += 1
        if ("planned_hunting" in est) and ["pessimistic"] == hand:
            confusion_matrix[4, 2] += 1
        if ("planned_hunting" in est) and ["suicide"] == hand:
            confusion_matrix[4, 3] += 1
        if ("planned_hunting" in est) and ["planned_hunting"] == hand:
            confusion_matrix[4, 4] += 1

    confusion_matrix = np.array(confusion_matrix, dtype = np.float)
    for col in range(len(agent_name)):
        confusion_matrix[:, col] = confusion_matrix[:, col] / np.sum(confusion_matrix[:, col])


    plt.subplot(1, 2, 2)
    if "planned_hunting" in agent_name:
        agent_name[agent_name.index("planned_hunting")] = "attack"
    if "pessimistic" in agent_name:
        agent_name[agent_name.index("pessimistic")] = "evade"
    ticks = ["local", "global"]
    ticks.extend(agent_name[2:])
    seaborn.heatmap(confusion_matrix,
                    annot = True, cmap = "binary", fmt = ".1%",
                    xticklabels = ticks,
                    yticklabels = ticks,
                    cbar = False, square = True, annot_kws = {"fontsize" : 20})
    plt.xlabel("Rule-Based Label", fontsize = 20)
    plt.ylabel("Fitted Label", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.show()


def plotLocalEvadeSuicideMatching(config):
    # agent_name = config["trial_weight_filename"].split("/")[-2].split("_")
    # if "planned" in agent_name and "hunting" in agent_name:
    #     agent_name = agent_name[:-2]
    #     agent_name.append("planned_hunting")
    agent_name = ["local", "pessimistic", "suicide"]
    if len(agent_name) != 3:
        raise NotImplementedError("The agent list is {}!".format(agent_name))
    print("Agent name : ", agent_name)
    # Read data
    # trial_weight : (num of trials, num of windows, num of agents + 1)
    # trial_Q : (num of trials, num of windows, num of agents + 1, num of directions)
    handcrafted_labels = np.load(config["handcrafted_label_filename"].format("_".join(agent_name)), allow_pickle=True)
    trial_weight = np.load(config["trial_weight_filename"].format("_".join(agent_name)), allow_pickle = True)
    trial_Q = np.load(config["trial_Q_filename"].format("_".join(agent_name)), allow_pickle = True)
    trial_contributions = []
    trial_matching_rate = []
    estimated_labels = []
    for trial_index in range(len(trial_weight)):
        temp_contribution = []
        temp_labels = []
        is_same = []
        for centering_index in range(len(trial_weight[trial_index])):
            contribution = trial_weight[trial_index][centering_index, :-1] * \
                           [scaleOfNumber(each) for each in np.max(
                               np.abs(trial_Q[trial_index][centering_index, :, [0, 1, 2], :]),axis=(1, 2)
                           )]
            # normalization
            contribution = contribution / np.linalg.norm(contribution)
            temp_contribution.append(copy.deepcopy(contribution))
            # Labeling
            est = _estimationLocalEvadeSuicideLabeling(contribution)
            temp_labels.append(copy.deepcopy(est))
            # Matching
            if handcrafted_labels[trial_index][centering_index] is not None:
                if len(np.intersect1d(est, handcrafted_labels[trial_index][centering_index])) > 0:
                    is_same.append(1)
                else:
                    is_same.append(0)
        trial_contributions.append(copy.deepcopy(temp_contribution))
        estimated_labels.append(copy.deepcopy(temp_labels))
        trial_matching_rate.append(np.sum(is_same)/len(is_same) if len(is_same) > 0 else None)

    # trial_matching_rate = np.load(config["trial_matching_rate_filename"], allow_pickle=True)
    not_nan_trial_matching_rate = []
    for each in trial_matching_rate:
        if each is not None:
            not_nan_trial_matching_rate.append(float(each))
    trial_matching_rate = not_nan_trial_matching_rate

    print("-"*15)
    print("Matching rate : ")
    print("Max : ", np.nanmax(trial_matching_rate))
    print("Min : ", np.nanmin(trial_matching_rate))
    print("Median : ", np.nanmedian(trial_matching_rate))
    print("Average : ", np.nanmean(trial_matching_rate))

    colors = Davos_5.mpl_colors[1]
    plt.figure(figsize=(18, 8))
    plt.subplot(1, 2, 1)
    # plt.title("Label Matching on {} Trials".format(len(trial_matching_rate)), fontsize = 20)
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
        if handcrafted_labels[index] is not None:
            if "local" in handcrafted_labels[index] and "global" in handcrafted_labels[index]:
                continue
            used_index.append(index)
    estimated_labels = np.array(estimated_labels)[used_index]
    handcrafted_labels = np.array(handcrafted_labels)[used_index]

    weird_index = []
    for index in range(len(used_index)):
        est = [each for each in estimated_labels[index]]
        hand = [each for each in handcrafted_labels[index]]

        if ["pessimistic"] == hand:
            print()

        if ["local"] == est and ["local"] == hand:
            confusion_matrix[0, 0] += 1
        if ["local"] == est and ["pessimistic"] == hand:
            confusion_matrix[0, 1] += 1
        if ["local"] == est and ["suicide"] == hand:
            confusion_matrix[0, 2] += 1

        if ("pessimistic" in est and "suicide" not in est) and ["local"] == hand:
            confusion_matrix[1, 0] += 1
        if ("pessimistic" in est and "suicide" not in est) and ["pessimistic"] == hand:
            confusion_matrix[1, 1] += 1
        if ("pessimistic" in est and "suicide" not in est) and ["suicide"] == hand:
            confusion_matrix[1, 2] += 1

        if ("suicide" in est and "pessimistic" not in est) and ["local"] == hand:
            confusion_matrix[2, 0] += 1
        if ("suicide" in est and "pessimistic" not in est) and ["pessimistic"] == hand:
            confusion_matrix[2, 1] += 1
        if ("suicide" in est and "pessimistic" not in est) and ["suicide"] == hand:
            confusion_matrix[2, 2] += 1

    confusion_matrix = np.array(confusion_matrix, dtype = np.float)
    for col in range(3):
        confusion_matrix[:, col] = confusion_matrix[:, col] / np.sum(confusion_matrix[:, col])


    plt.subplot(1, 2, 2)
    if "planned_hunting" in agent_name:
        agent_name[agent_name.index("planned_hunting")] = "attack"
    if "pessimistic" in agent_name:
        agent_name[agent_name.index("pessimistic")] = "evade"
    seaborn.heatmap(confusion_matrix,
                    annot = True, cmap = "binary", fmt = ".1%",
                    xticklabels = ["local", "evade", "suicide"],
                    yticklabels = ["local", "evade", "suicide"],
                    cbar = False, square = True, annot_kws = {"fontsize" : 20})
    plt.xlabel("Rule-Based Label", fontsize = 20)
    plt.ylabel("Fitted Label", fontsize = 20)
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
    x_ticks = ["local", "+global", "+evade", "+attack", "+suicide"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)
    colors = RdYlBu_5.mpl_colors
    colors[2] = Balance_6.mpl_colors[2]
    colors = [colors[0], colors[-1], colors[1], colors[3], colors[2]]

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Early Stage (Pellets >= 80)", fontsize=20)
    avg_cr = np.mean(third_phase_agent_cr, axis=0)
    var_cr = np.var(third_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], [0.8, 0.85, 0.9, 0.95, 1.0], fontsize = 15)
    plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)
    plt.ylim(0.8, 1.0)

    plt.subplot(1, 3, 2)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Middle Stage (10 < Pellets < 80)", fontsize=20)
    avg_cr = np.mean(second_phase_agent_cr, axis=0)
    var_cr = np.var(second_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00], [], fontsize=15)
    plt.ylim(0.8, 1.0)

    plt.subplot(1, 3, 3)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Ending Stage (Pellets <= 10)", fontsize=20)
    avg_cr = np.mean(first_phase_agent_cr, axis=0)
    var_cr = np.var(first_phase_agent_cr, axis=0)
    # plt.errorbar(x_index, avg_cr, yerr = var_cr, fmt = "k", mfc = "r", marker = "o", linestyle = "", ms = 15, elinewidth = 5)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.8, 0.85, 0.9, 0.95, 1.0], [], fontsize=15)
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
        "single_trial_window": 2,
        "single_trial_agents": ["global", "local", "pessimistic"],

        # ==================================================================================
        #                       For Experimental Results Visualization
        "estimated_label_filename": "./common_data/{}/100_trial_data_Omega-with_Q-window3-w_intercept-multi_labels.npy",
        "handcrafted_label_filename": "./common_data/{}/100_trial_data_Omega-with_Q-window3-w_intercept-handcrafted_labels.npy",
        "trial_weight_filename": "./common_data/{}/100_trial_data_Omega-with_Q-window3-w_intercept-trial_weight.npy",
        "trial_Q_filename": "./common_data/{}/100_trial_data_Omega-with_Q-window3-w_intercept-Q.npy",
        "trial_matching_rate_filename": "./common_data/{}/100_trial_data_Omega-with_Q-window3-w_intercept-matching_rate.npy",
        "trial_agent_name" : ["global", "local", "pessimistic", "suicide", "planned_hunting"],
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

        "local_to_planned_agent_weight": "./common_data/transition/local_to_planned-window1-agent_weight-w_intercept.npy",
        "local_to_planned_cr": "./common_data/transition/local_to_planned-window1-cr-w_intercept.npy",
        "local_to_planned_Q": "./common_data/transition/local_to_planned-window1-Q-w_intercept.npy",

        "local_to_suicide_agent_weight": "./common_data/transition/local_to_suicide-window1-agent_weight-w_intercept.npy",
        "local_to_suicide_cr": "./common_data/transition/local_to_suicide-window1-cr-w_intercept.npy",
        "local_to_suicide_Q": "./common_data/transition/local_to_suicide-window1-Q-w_intercept.npy",

        "agent_list" : [["local", "global"], ["local", "pessimistic"], ["local", "global"],
                        ["local", "pessimistic"], ["local", "planned_hunting"], ["local", "suicide"]],

        # ------------------------------------------------------------------------------------

        "bean_vs_cr_filename" : "./common_data/incremental/100trial-Omega-window3-incremental_cr-w_intercept.npy",
    }

    # ============ VISUALIZATION =============
    # plotThreeAgentMatching(config) # For three agent
    # plotLocalEvadeSuicideMatching(config) # For local, evade, and suicide

    # plotAllAgentMatching(config)

    plotWeightVariation(config)

    # plotBeanNumVSCr(config)

    # singleTrialThreeFitting(config)

    # Best trials:
    # "13-2-Patamon-10-Sep-2019-1.csv", "10-3-Omega-09-Jul-2019-1.csv", "10-2-Patamon-07-Jul-2019-1.csv",
    # "10-7-Patamon-10-Aug-2019-1.csv", "11-1-Patamon-11-Jun-2019-1.csv", "13-5-Patamon-21-Aug-2019-1.csv",
    # "14-1-Patamon-14-Jun-2019-1.csv", "14-2-Patamon-10-Jul-2019-1.csv"