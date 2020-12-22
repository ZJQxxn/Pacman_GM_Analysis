'''
Description:
    Fitting weight dynamics.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    1 Dec. 2020
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
from DescriptiveAnalysis import _pessimisticProcesing, _plannedHuntingProcesing,_suicideProcesing, _makeChoice, _label2Index, negativeLikelihood
from DescriptiveAnalysis import _localProcesing, _globalProcesing
from DescriptiveAnalysis import _PG, _PE, _ghostStatus, _energizerNum, _PR, _RR, _PGWODead, _PRDistGlobal

dir_list = ['left', 'right', 'up', 'down']
locs_df = readLocDistance("../common_data/dij_distance_map.csv")
print("Finished reading distance file!")
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
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    PG = all_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PG(x, locs_df),
        axis=1
    )
    PG_wo_dead = all_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PGWODead(x, locs_df),
        axis=1
    )
    PE = all_data[["pacmanPos", "energizers"]].apply(
        lambda x: _PE(x, locs_df),
        axis=1
    )
    ghost_status = all_data[["ifscared1", "ifscared2"]].apply(
        lambda x: _ghostStatus(x),
        axis=1
    )
    energizer_num = all_data[["energizers"]].apply(
        lambda x: _energizerNum(x),
        axis=1
    )
    PR = all_data[
        ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PR(x, locs_df),
        axis=1
    )
    PR_global = all_data[
        ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PRDistGlobal(x, locs_df),
        axis=1
    )
    RR = all_data[
        ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _RR(x, locs_df),
        axis=1
    )
    print("Finished extracting features.")
    # TODO: planned hunting and suicide Q value
    all_data.pessimistic_Q = _pessimisticProcesing(all_data.pessimistic_Q, PG, ghost_status)
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num,
                                                          PE, PG_wo_dead)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG_wo_dead)
    all_data.local_Q = _localProcesing(all_data.local_Q, PR)
    all_data.global_Q = _globalProcesing(all_data.global_Q, PR_global)
    print("Finished Q-value pre-processing.")
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


def _readOneTrialData():
    '''
           Read data for MLE analysis.
           :param filename: Filename.
           '''
    # Read data and pre-processing
    with open("../common_data/trial/500_trial_data_Omega-with_Q.pkl", "rb") as file:
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
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    PG = all_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PG(x, locs_df),
        axis=1
    )
    PE = all_data[["pacmanPos", "energizers"]].apply(
        lambda x: _PE(x, locs_df),
        axis=1
    )
    ghost_status = all_data[["ifscared1", "ifscared2"]].apply(
        lambda x: _ghostStatus(x),
        axis=1
    )
    energizer_num = all_data[["energizers"]].apply(
        lambda x: _energizerNum(x),
        axis=1
    )
    PR = all_data[
        ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PR(x, locs_df),
        axis=1
    )
    RR = all_data[
        ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _RR(x, locs_df),
        axis=1
    )
    print("Finished extracting features.")
    # TODO: planned hunting and suicide Q value
    all_data.pessimistic_Q = _pessimisticProcesing(all_data.pessimistic_Q, PG)
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG)
    print("Finished Q-value pre-processing.")
    # Split into trials
    trial_data = []
    trial_name_list = ["7-3-Omega-11-Jun-2019-1.csv", "9-3-Omega-19-Aug-2019-1.csv"]
    for each in trial_name_list:
        each_trial = all_data[all_data.file == each].reset_index(drop=True)
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


def multiAgentAnalysis(trial_num = None):
    print("== Omega Data Analysis with All the Agents ==")
    trial_data_filename = "../common_data/trial/7000_trial_data_Patamon-with_Q-descriptive.pkl"
    # trial_data_filename = "../common_data/single_trial/5_trial-data_for_comparison-one_ghost-with_Q.pkl"
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    print(trial_data_filename)
    print(agent_name)
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    window = 3
    print("window size : ", window)
    temp_trial_data = readTrialData(trial_data_filename)

    #TODO: !!!!!
    # temp_trial_data = _readOneTrialData()

    all_trial_num = len(temp_trial_data)
    print("Num of trials : ", all_trial_num)
    trial_index = range(all_trial_num)
    if trial_num is not None:
        if trial_num < all_trial_num:
            trial_index = np.random.choice(range(all_trial_num), trial_num, replace = False)
    trial_data = [temp_trial_data[each] for each in trial_index]
    agent_index = [0, 1, 2, 3, 4]
    # For every trial
    for trial_index, each in enumerate(trial_data):
        trial_window_Q = [np.nan for _ in range(window)]
        trial_window_weight = [np.nan for _ in range(window)]
        trial_window_contribution = [np.nan for _ in range(window)]
        print("-"*15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print(trial_index, " : ", trial_name)
        print("Trial length : ", trial_length)
        #TODO: !!!!!!
        # Preprocess suicide Q in the beginning of a trial
        cur_index = 0
        while ((14,27)==X.pacmanPos[cur_index] or locs_df[(14, 27)][X.pacmanPos[cur_index]] < 10) and cur_index < trial_length:
            non_zero = np.where(X.suicide_Q[cur_index] != 0)
            X.suicide_Q[cur_index][non_zero] = 0.0
            cur_index += 1
            if cur_index >= trial_length:
                break
        #
        window_index = np.arange(window, trial_length - window)
        # (num of windows, window size, num of agents, num of directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
            # estimation in the window
            window_estimated_label = []
            # Construct optimizer
            params = [0 for _ in range(len(agent_name) + 1)]
            bounds = [[0, 10] for _ in range(len(agent_name))]
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
                need_intercept=True
            )
            is_success = False
            retry_num = 0
            while not is_success and retry_num < 5:
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

            # temp_weight[centering_index, :] = res.x

            contribution = res.x[:-1] * [scaleOfNumber(each) for each in
                                np.max(np.abs(temp_trial_Q[centering_index, :, agent_index, :]), axis=(1, 2))]
            trial_window_weight.append(res.x)
            trial_window_Q.append(temp_trial_Q[centering_index, :, agent_index, :])
            trial_window_contribution.append(contribution)
            # temp_contribution.append(copy.deepcopy(contribution))
        trial_window_weight.extend([np.nan for _ in range(window)])
        trial_window_Q.extend([np.nan for _ in range(window)])
        trial_window_contribution.extend([np.nan for _ in range(window)])
        trial_data[trial_index][1]["weight"] = copy.deepcopy(trial_window_weight)
        trial_data[trial_index][1]["contribution"] = copy.deepcopy(trial_window_contribution)
        trial_data[trial_index][1]["window_Q"] = copy.deepcopy(trial_window_Q)

    # # Save data
    processed_trial_data = pd.concat([each[1] for each in trial_data])
    with open("../common_data/trial/{}-with_weight-window{}-new_agents.pkl".format(trial_data_filename.split("/")[-1].split(".")[-2], window), "wb") as file:
        pickle.dump(processed_trial_data, file)
    # with open("../common_data/trial/sample-window{}.pkl".format(window), "wb") as file:
    #     pickle.dump(processed_trial_data, file)



if __name__ == '__main__':
    multiAgentAnalysis(trial_num=None)

    # with open("../common_data/trial/sample-window3.pkl", "rb") as file:
    #     data = pickle.load(file)

