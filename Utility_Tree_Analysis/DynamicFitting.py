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
from PathTreeAgent import PathTree
from SuicideAgent import SuicideAgent
from PlannedHuntingAgent import PlannedHuntingAgent


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
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status)
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


def _ghostStatus(x):
    ghost_status =[int(x.ifscared1), int(x.ifscared2)]
    return ghost_status


def _energizerNum(x):
    if x.energizers is None or isinstance(x.energizers, float):
        num= 0
    else:
        num = len(x.energizers)
    return num


def _PR(x, locs_df):
    # beans, energizers, fruits, scared ghosts
    PR_dist = []
    if x.beans is not None and x.beans != [] and not isinstance(x.beans, float):
        PR_dist.extend(x.beans)
    if x.energizers is not None and x.energizers != [] and not isinstance(x.energizers, float):
        PR_dist.extend(x.energizers)
    if x.fruitPos is not None and x.fruitPos != [] and not  isinstance(x.fruitPos, float):
        PR_dist.append(x.fruitPos)
    if x.ifscared1 > 3:
        PR_dist.append(x.ghost1Pos)
    if x.ifscared2 > 3:
        PR_dist.append(x.ghost2Pos)
    # Compute distance
    PR_dist = [locs_df[x.pacmanPos][each] if each != x.pacmanPos else 0 for each in PR_dist]
    if len(PR_dist) > 0:
        return np.min(PR_dist)
    else:
        return 100


def _RR(x, locs_df):
    # beans, energizers, fruits, scared ghosts
    RR_dist = []
    if x.beans is not None and x.beans != [] and not  isinstance(x.beans, float):
        RR_dist.extend(x.beans)
    if x.energizers is not None and x.energizers != [] and not  isinstance(x.energizers, float):
        RR_dist.extend(x.energizers)
    if x.fruitPos is not None and x.fruitPos != [] and not  isinstance(x.fruitPos, float):
        RR_dist.append(x.fruitPos)
    if x.ifscared1 > 3:
        RR_dist.append(x.ghost1Pos)
    if x.ifscared2 > 3:
        RR_dist.append(x.ghost2Pos)
    # Compute distance
    reborn_pos = (14, 27)
    RR_dist = [locs_df[reborn_pos][each] if each != reborn_pos else 0 for each in RR_dist]
    if len(RR_dist) > 0:
        return np.min(RR_dist)
    else:
        return 100


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


def _plannedHuntingProcesing(planned_Q, ghost_status, energizer_num):
    if np.any(np.concatenate(planned_Q) < 0):
        offset = np.max(np.abs(np.concatenate(planned_Q))) # TODO: max absolte value of negative values
    else:
        offset = 0.0
    temp_planned_Q = copy.deepcopy(planned_Q)
    for index in range(len(temp_planned_Q)):
        non_zero = np.where(temp_planned_Q[index] != 0)
        if np.all(np.array(ghost_status[index]) >= 3) or energizer_num[index] == 0:
            temp_planned_Q[index][non_zero] = 0.0
        else:
            temp_planned_Q[index][non_zero] = temp_planned_Q[index][non_zero] + offset
    return temp_planned_Q


def _suicideProcesing(suicide_Q, PR, RR, ghost_status):
    # PR: minimum distance between Pacman position and reward entities
    # RR: minimum distance between reborn position and reward entities
    if np.any(np.concatenate(suicide_Q) < 0):
        offset = np.max(np.abs(np.concatenate(suicide_Q)))  # TODO: max absolte value of negative values
    else:
        offset = 0.0
    temp_suicide_Q = copy.deepcopy(suicide_Q)
    for index in range(len(temp_suicide_Q)):
        non_zero = np.where(temp_suicide_Q[index] != 0)
        # if np.all(np.array(ghost_status[index]) >= 3) or (PR[index] > 10 and RR[index] > 10):
        if np.all(np.array(ghost_status[index]) >= 3) or RR[index] > 10:
            temp_suicide_Q[index][non_zero] = 0.0
        else:
            temp_suicide_Q[index][non_zero] = temp_suicide_Q[index][non_zero] + offset
    return temp_suicide_Q


def _label2Index(labels):
    label_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    label_val = copy.deepcopy(labels)
    for index, each in enumerate(label_val):
        if each is not None:
            label_val[index] = label_list.index(each)
        else:
            label_val[index] = None
    return label_val


def _makeChoice(prob):
    copy_estimated = copy.deepcopy(prob)
    if np.any(prob) < 0:
        available_dir_index = np.where(prob != 0)
        copy_estimated[available_dir_index] = copy_estimated[available_dir_index] - np.min(copy_estimated[available_dir_index]) + 1
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


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


def multiAgentAnalysis(trial_num = None):
    print("== Omega Data Analysis with All the Agents ==")
    trial_data_filename = "../common_data/trial/500_trial_data_Omega-with_Q.pkl"
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
    with open("../common_data/trial/{}-with_weight-window{}-new_suicide.pkl".format(trial_data_filename.split("/")[-1].split(".")[-2], window), "wb") as file:
        pickle.dump(processed_trial_data, file)
    # with open("../common_data/trial/sample-window{}.pkl".format(window), "wb") as file:
    #     pickle.dump(processed_trial_data, file)


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
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status)
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


if __name__ == '__main__':
    multiAgentAnalysis(trial_num=None)

    # with open("../common_data/trial/sample-window3.pkl", "rb") as file:
    #     data = pickle.load(file)
    print()

    pass