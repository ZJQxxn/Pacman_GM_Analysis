'''
Description:
    Compare simulated labels with hand-crafted labels.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    17 Dec. 2020
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
from EnergizerAgent import EnergizerAgent
from ApproachingAgent import ApproachingAgent


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


def _PGWODead(x, locs_df):
    PG = [0, 0]
    if x.ifscared1 != 3:
        if tuple(x.pacmanPos) != tuple(x.ghost1Pos):
            PG[0] = locs_df[tuple(x.pacmanPos)][tuple(x.ghost1Pos)]
        else:
            pass
    else:
        PG[0] = 100
    if x.ifscared2 != 3:
        if tuple(x.pacmanPos) != tuple(x.ghost2Pos):
            PG[1] = locs_df[tuple(x.pacmanPos)][tuple(x.ghost2Pos)]
        else:
            pass
    else:
        PG[1] = 100
    return PG


def _PE(x, locs_df):
    PE = []
    if isinstance(x.energizers, float) or len(x.energizers) == 0:
        PE = [100]
    else:
        for each in x.energizers:
            if x.pacmanPos == each:
                PE.append(0)
            else:
                PE.append(locs_df[tuple(x.pacmanPos)][tuple(each)])
    return np.min(PE)


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


def _mixStatus(ghost_status, PG):
    if ghost_status[0] < 3 and ghost_status[1] > 3:
        if PG[1] >= 15:
            return True
        else:
            return False
    if ghost_status[0] > 3 and ghost_status[1] < 3:
        if PG[0] >= 15:
            return True
        else:
            return False
    return False


def _pessimisticProcesing(pess_Q, PG, ghost_status):
    if np.any(np.concatenate(pess_Q) < 0):
        temp_pess_Q = np.concatenate(pess_Q)
        temp_pess_Q[temp_pess_Q > 0] = 0.0
        offset = np.max(np.abs(temp_pess_Q))  # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_pess_Q = copy.deepcopy(pess_Q)
    for index in range(len(temp_pess_Q)):
        non_zero = np.where(temp_pess_Q[index] != 0)
        # if np.any(temp_pess_Q[index] < -5):
        if np.any(np.array(PG[index]) <= 10) and np.all(np.array(ghost_status[index]) < 3):
            temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
        else:
            temp_pess_Q[index][non_zero] = 0.0
    # for index in range(len(temp_pess_Q)):
    #     non_zero = np.where(temp_pess_Q[index] != 0)
    #     temp_global_Q[index][non_zero] = temp_global_Q[index][non_zero] + offset
    #     temp_local_Q[index][non_zero] = temp_local_Q[index][non_zero] + offset
    #     temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
    return temp_pess_Q


def _plannedHuntingProcesing(planned_Q, ghost_status, energizer_num, PE, PG):
    eat_index = []
    if np.any(np.concatenate(planned_Q) < 0):
        temp_planned_Q = np.concatenate(planned_Q)
        temp_planned_Q[temp_planned_Q > 0] = 0.0
        offset = np.max(np.abs(temp_planned_Q)) # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_planned_Q = copy.deepcopy(planned_Q)
    for index in range(len(temp_planned_Q)):
        non_zero = np.where(temp_planned_Q[index] != 0)
        # if np.all(np.array(ghost_status[index]) >= 3) or energizer_num[index] == 0 or PE[index] > 15:
        if (np.all(np.array(ghost_status[index]) <= 3) and energizer_num[index] == 0) \
                or (np.all(np.array(ghost_status[index]) < 3) and PE[index] >= 15) \
                or np.all(np.array(ghost_status[index]) == 3) or np.all(np.array(PG[index]) >= 15) or np.any(np.array(ghost_status[index]) > 3):
        # _mixStatus(ghost_status[index], PG[index])
            temp_planned_Q[index][non_zero] = 0.0
        else:
            temp_planned_Q[index][non_zero] = temp_planned_Q[index][non_zero] + offset
        # if index > 0 and np.all(np.array(ghost_status[index]) > 3) and np.all(np.array(ghost_status[index-1]) < 3):
        #     eat_index.append(index)
    # if len(eat_index) > 0:
    #     for i in eat_index:
    #         for j in range(i, i+1):
    #             non_zero = np.where(temp_planned_Q[i-1] != 0)
    #             temp_planned_Q[j] = temp_planned_Q[i-1]
    #             temp_planned_Q[j][non_zero] += np.random.normal(1, 0.5, len(non_zero[0]))
    return temp_planned_Q


def _suicideProcesing(suicide_Q, PR, RR, ghost_status, PG):
    # PR: minimum distance between Pacman position and reward entities
    # RR: minimum distance between reborn position and reward entities
    if np.any(np.concatenate(suicide_Q) < 0):
        temp_suicide_Q = np.concatenate(suicide_Q)
        temp_suicide_Q[temp_suicide_Q > 0] = 0.0
        offset = np.max(np.abs(temp_suicide_Q))  # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_suicide_Q = copy.deepcopy(suicide_Q)
    for index in range(len(temp_suicide_Q)):
        non_zero = np.where(temp_suicide_Q[index] != 0)
        # if np.all(np.array(ghost_status[index]) >= 3) or (PR[index] > 10 and RR[index] > 10):
        if np.all(np.array(ghost_status[index]) == 3) or not np.any(np.array(PG[index]) < 10):
            temp_suicide_Q[index][non_zero] = 0.0
        else:
            temp_suicide_Q[index][non_zero] = temp_suicide_Q[index][non_zero] + offset
    return temp_suicide_Q


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
    RR = all_data[
        ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _RR(x, locs_df),
        axis=1
    )
    print("Finished extracting features.")
    # TODO: planned hunting and suicide Q value
    all_data.pessimistic_Q = _pessimisticProcesing(all_data.pessimistic_Q, PG, ghost_status)
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE, PG_wo_dead)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG_wo_dead)
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

def _estimationVagueLabeling(contributions, all_agent_name):
    sorted_contributions = np.sort(contributions)[::-1]
    if sorted_contributions[0] - sorted_contributions[1] < 0.2 :
        return ["vague"]
    else:
        label = all_agent_name[np.argmax(contributions)]
        return [label]

def _estimationThreeLabeling(contributions):
    # global, local, pessimistic
    labels = []
    agent_name = ["global", "local"]
    if np.any(contributions[:2] > 0):
        labels.append(agent_name[np.argmax(contributions[:2])])
    if contributions[-1] > 0.5:
        labels.append("pessimistic")
    return labels


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


def multiAgentAnalysis(config):
    print("== Multi Label Analysis with Multiple Agents ==")
    print(config["trial_data_filename"])
    print(config["multi_agent_list"])
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

    trial_weight = []
    trial_Q = []
    trial_contribution = []
    handcrafted_labels = []
    trial_matching_rate = []
    all_estimated_label = []
    record = []
    # agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    agent_name = config["multi_agent_list"]
    agent_index = [["global", "local", "pessimistic", "suicide", "planned_hunting"].index(each) for each in agent_name]

    for trial_index, each in enumerate(trial_data):
        trial_record = []
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
        temp_weight = np.zeros((len(window_index), len(agent_name) if not config["need_intercept"] else len(agent_name) + 1))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        trial_estimated_label = []
        temp_contribution = []
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

            temp_weight[centering_index, :] = res.x
            contribution = temp_weight[centering_index, :-1] * [scaleOfNumber(each) for each in
                                np.max(np.abs(temp_trial_Q[centering_index, :, agent_index, :]), axis=(1, 2))]
            temp_contribution.append(copy.deepcopy(contribution))
            window_estimated_label.append(_estimationVagueLabeling(contribution, agent_name))
            trial_estimated_label.append(copy.deepcopy(window_estimated_label))
        trial_contribution.append(copy.deepcopy(temp_contribution))
        matched_num = 0
        not_nan_num = 0
        for i in range(len(temp_handcrafted_label)):
            if temp_handcrafted_label[i] is not None:
                not_nan_num += 1
                if len(np.intersect1d(temp_handcrafted_label[i], trial_estimated_label[i])) > 0:
                    matched_num += 1
        print(" Trial label matching rate : ", matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")
        trial_matching_rate.append(matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")
        trial_weight.append(copy.deepcopy(temp_weight))
        trial_Q.append(copy.deepcopy(temp_trial_Q))
        all_estimated_label.append(copy.deepcopy(trial_estimated_label))
        # records
        trial_record.append(copy.deepcopy(temp_weight))
        trial_record.append(copy.deepcopy(temp_contribution))
        trial_record.append(copy.deepcopy(trial_estimated_label))
        trial_record.append(copy.deepcopy(temp_handcrafted_label))
        trial_record.append(copy.deepcopy(temp_trial_Q))
        record.append(copy.deepcopy(trial_record))

    # Save data
    dir_names = "-".join(agent_name)
    if dir_names not in os.listdir("../common_data"):
        os.mkdir("../common_data/{}".format(dir_names))
    save_base = config["trial_data_filename"].split("/")[-1].split(".")[0]
    np.save("../common_data/{}/{}-window{}-{}_intercept-multi_labels.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), all_estimated_label)
    np.save("../common_data/{}/{}-window{}-{}_intercept-handcrafted_labels.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), handcrafted_labels)
    np.save("../common_data/{}/{}-window{}-{}_intercept-matching_rate.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_matching_rate)
    np.save("../common_data/{}/{}-window{}-{}_intercept-trial_weight.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_weight)
    np.save("../common_data/{}/{}-window{}-{}_intercept-Q.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_Q)
    np.save("../common_data/{}/{}-window{}-{}_intercept-contribution.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_contribution)
    # Save Records
    data_type = None
    if "new" in config["trial_data_filename"]:
        data_type = "planned"
    elif "accidental" in config["trial_data_filename"]:
        data_type = "accidental"
    elif "suicide" in config["trial_data_filename"]:
        data_type = "suicide"
    elif "global" in config["trial_data_filename"]:
        data_type = "global"
    else:
        data_type = None
    np.save("../common_data/{}/{}_descriptive_records.npy".format(dir_names, data_type), record)


def incrementalAnalysis(config):
    # Read trial data
    # agent_name = config["incremental_data_filename"]
    # agents_list = ["{}_Q".format(each) for each in agent_name]
    print("=== Incremental Analysis ====")
    print(config["incremental_data_filename"])
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
    np.save("../common_data/incremental/descriptive-window{}-incremental_cr-{}_intercept.npy".format(
        window, "w" if config["need_intercept"] else "wo"), all_cr)


def singleTrialAllFitting(config):
    orginal_weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/new_100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    # Read trial data
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    agents_list = ["{}_Q".format(each) for each in agent_name]
    window = config["single_trial_window"]
    trial_data = readTrialData(config["single_trial_data_filename"])
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)

    trial_name_list = [
        "1-1-Omega-12-Aug-2019-1.csv",
        "1-1-Omega-31-Jul-2019-1.csv",
        "1-2-Omega-19-Aug-2019-1.csv",
        "10-1-Omega-13-Jun-2019-1.csv"
    ]
    trial_indices = [4, 10, 13, 29]
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
        label_not_nan_index = []
        for i, each in enumerate(handcrafted_label):
            if each is not None:
                label_not_nan_index.append(i)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight = np.zeros((len(window_index), 5 if not config["need_intercept"] else 6))
        # temp_weight_rest = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))
        # temp_Q = []
        temp_contribution = np.zeros((len(window_index), 5))
        # temp_contribution_rest = np.zeros((len(window_index), 3))
        cr = np.zeros((len(window_index), ))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        # For each trial, estimate agent weights through sliding windows

        trial_estimated_label = []
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

            temp_weight[centering_index, :] = res.x
            contribution = temp_weight[centering_index, :-1] * \
                           [scaleOfNumber(each) for each in
                            np.max(np.abs(temp_trial_Q[centering_index, :, :, :]), axis=(0, 2))]
            temp_contribution[centering_index, :] = contribution
            window_estimated_label.append(_estimationLabeling(contribution, agent_name))
            trial_estimated_label.append(window_estimated_label)
        # ============================================================
        # Visualization
        # Fin planned hunting trajectory
        cur_trial_data = trial_data[trial_index][1]
        start_index = 0
        end_index = cur_trial_data.shape[0]
        i = 0
        while i < cur_trial_data.shape[0]:
            if cur_trial_data.label_true_planned_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == cur_trial_data.shape[0]:
            continue
        while i < cur_trial_data.shape[0]:
            if cur_trial_data.ifscared1.values[i] == 3 or cur_trial_data.ifscared2.values[i] == 3 or (
                    cur_trial_data.ifscared1.values[i] < 3 and cur_trial_data.ifscared1.values[i - 1] > 3):
                end_index = i
                break
            i += 1
        if i == cur_trial_data.shape[0]:
            continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        trial_original_weight = orginal_weight[trial_indices[trial_index]]
        # Hand-crafted label and normalize contribution
        trial_original_weight = np.array(trial_original_weight)[start_index:end_index]
        trial_hand_crafted = np.array(handcrafted_label)[start_index:end_index]
        trial_descriptive_weight = np.array(temp_contribution)[start_index:end_index]
        # normalization
        for index in range(trial_original_weight.shape[0]):
            trial_original_weight[index, :] = trial_original_weight[index, :] / np.linalg.norm(trial_original_weight[index, :])
            trial_descriptive_weight[index, :] = trial_descriptive_weight[index, :] / np.linalg.norm(trial_descriptive_weight[index, :])
        # Plot weight variation of this trial
        # agent_color = {
        #     "local": colors[0],
        #     "pessimistic": colors[1],
        #     "global": colors[-1],
        #     "suicide": Balance_6.mpl_colors[2],
        #     "planned_hunting": colors[3],
        #     "vague": "black"
        # }
        agent_color = {
            "local": "#D7181C",
            "pessimistic": "#FDAE61",
            "global": "#44B53C",
            "suicide": "#836BB7",
            "planned_hunting": "#81B3FF",
            "vague": "black"
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
        }
        plt.figure(figsize=(18, 13))
        plt.subplot(3, 1, 1)
        for index in range(len(agent_name)):
            plt.plot(trial_original_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_original_weight.shape[0] - 1)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [start_index + window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))

        plt.subplot(3, 1, 2)
        for i in range(len(trial_hand_crafted)):
            if trial_hand_crafted[i] is not None:
                seq = np.linspace(-0.05, 0.0, len(trial_hand_crafted[i]) + 1)
                for j, h in enumerate(trial_hand_crafted[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, trial_original_weight.shape[0])
        plt.xticks([], [], fontsize=15)
        plt.yticks([-0.025], ["Rule-Based Label"], fontsize=15)

        plt.subplot(3, 1, 3)
        plt.title("Descriptive Agents ({})".format(trial_name_list[trial_index]), fontsize=10)
        for index in range(len(agent_name)):
            plt.plot(trial_descriptive_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_descriptive_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.show()
        print()

# ===================================
#         COMPARISON
# ===================================
def plotComparison():
    from palettable.cmocean.diverging import Balance_6
    from palettable.colorbrewer.diverging import RdBu_8, RdYlBu_5
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    data = readTrialData("../common_data/trial/100_trial_data_Omega-with_Q.pkl")
    filename = [each[0] for each in data]
    del data
    weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    hand_crafted_label = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/100_trial_data_Omega-with_Q-window3-w_intercept-handcrafted_labels.npy",
        allow_pickle=True
    )
    descriptive_weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/100_trial_data_Omega-with_Q-descriptive-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    trial_num = len(weight)
    random_indices = list(range(trial_num))
    np.random.shuffle(random_indices)
    cnt = 1
    for trial_index in random_indices:
        print("|{}| Trial Name : {}".format(cnt, filename[trial_index]))
        cnt += 1
        # Hand-crafted label and normalize contribution
        trial_weight = np.array(weight[trial_index])
        trial_hand_crafted = np.array(hand_crafted_label[trial_index])
        trial_descriptive_weight = np.array(descriptive_weight[trial_index])
        window = 3
        # normalization
        for index in range(trial_weight.shape[0]):
            trial_weight[index, :] = trial_weight[index, :] / np.linalg.norm(trial_weight[index, :])
            trial_descriptive_weight[index, :] = trial_descriptive_weight[index, :] / np.linalg.norm(
                trial_descriptive_weight[index, :])
        # Plot weight variation of this trial
        colors = RdYlBu_5.mpl_colors
        agent_color = {
            "local": colors[0],
            "pessimistic": colors[1],
            "global": colors[-1],
            "suicide": Balance_6.mpl_colors[2],
            "planned_hunting": colors[3],
            "vague": "black"
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
        }

        plt.figure(figsize=(18, 13))
        plt.subplot(3, 1, 1)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(trial_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_weight.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=20)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))

        # plt.figure(figsize=(13,5))
        plt.subplot(3, 1, 2)
        for i in range(len(trial_hand_crafted)):
            if trial_hand_crafted[i] is not None:
                seq = np.linspace(-0.05, 0.0, len(trial_hand_crafted[i]) + 1)
                for j, h in enumerate(trial_hand_crafted[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
                # seq = np.linspace(-0.2, -0.1, len(estimated_label[i]) + 1)
                # for j, h in enumerate(estimated_label[i]):
                #     plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, trial_weight.shape[0])
        plt.xticks([], [])
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([-0.05, -0.15], ["Rule-Based Label", "Fitted Label"], fontsize=10)
        plt.yticks([-0.025], ["Rule-Based Label"], fontsize=15)
        # plt.ylim(-0.05, 0.35)
        # plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.title("Descriptive Agents ({})".format(filename[trial_index]), fontsize = 10)
        for index in range(len(agent_name)):
            plt.plot(trial_descriptive_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_descriptive_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        # plt.legend(loc="upper center", fontsize=20, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))
        plt.savefig("../common_data/global_local_pessimistic_suicide_planned_hunting/images/{}.jpg".format(filename[trial_index]))
        # plt.clf()
        # plt.show()


def plotPlannedHunting():
    from palettable.cmocean.diverging import Balance_6
    from palettable.colorbrewer.diverging import RdBu_8, RdYlBu_5
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    data = readTrialData("../common_data/trial/new_100_trial_data_Omega-with_Q.pkl")
    filename = [each[0] for each in data]
    # del data
    weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/new_100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    hand_crafted_label = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/new_100_trial_data_Omega-with_Q-window3-w_intercept-handcrafted_labels.npy",
        allow_pickle=True
    )
    descriptive_weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/new_100_trial_data_Omega-with_Q-descriptive-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    trial_num = len(weight)
    random_indices = list(range(trial_num))
    np.random.shuffle(random_indices)
    cnt = 1
    for trial_index in random_indices:
        print("|{}| Trial Name : {}".format(cnt, filename[trial_index]))
        trial_data = data[trial_index][1]
        cnt += 1
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_true_planned_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            continue
        while i < trial_data.shape[0]:
            if trial_data.ifscared1.values[i] == 3 or trial_data.ifscared2.values[i] == 3 or (trial_data.ifscared1.values[i] < 3 and trial_data.ifscared1.values[i-1] > 3):
                end_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        # Hand-crafted label and normalize contribution
        trial_weight = np.array(weight[trial_index])[start_index:end_index]
        trial_hand_crafted = np.array(hand_crafted_label[trial_index])[start_index:end_index]
        trial_descriptive_weight = np.array(descriptive_weight[trial_index])[start_index:end_index]
        window = 3
        # normalization
        for index in range(trial_weight.shape[0]):
            trial_weight[index, :] = trial_weight[index, :] / np.linalg.norm(trial_weight[index, :])
            trial_descriptive_weight[index, :] = trial_descriptive_weight[index, :] / np.linalg.norm(
                trial_descriptive_weight[index, :])
        # Plot weight variation of this trial
        colors = RdYlBu_5.mpl_colors
        agent_color = {
            "local": "#D7181C",
            "pessimistic": "#FDAE61",
            "global": "#44B53C",
            "suicide": "#836BB7",
            "planned_hunting": "#81B3FF",
            "vague": "black"
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
        }

        plt.figure(figsize=(18, 13))
        plt.subplot(3, 1, 1)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(trial_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_weight.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=20)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [start_index + window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))

        # plt.figure(figsize=(13,5))
        plt.subplot(3, 1, 2)
        for i in range(len(trial_hand_crafted)):
            if trial_hand_crafted[i] is not None:
                seq = np.linspace(-0.05, 0.0, len(trial_hand_crafted[i]) + 1)
                for j, h in enumerate(trial_hand_crafted[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
                # seq = np.linspace(-0.2, -0.1, len(estimated_label[i]) + 1)
                # for j, h in enumerate(estimated_label[i]):
                #     plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, trial_weight.shape[0])
        plt.xticks([], [])
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([-0.05, -0.15], ["Rule-Based Label", "Fitted Label"], fontsize=10)
        plt.yticks([-0.025], ["Rule-Based Label"], fontsize=15)
        # plt.ylim(-0.05, 0.35)
        # plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.title("Descriptive Agents ({})".format(filename[trial_index]), fontsize=10)
        for index in range(len(agent_name)):
            plt.plot(trial_descriptive_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_descriptive_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        # plt.legend(loc="upper center", fontsize=20, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))
        plt.savefig("../common_data/global_local_pessimistic_suicide_planned_hunting/planned/{}.jpg".format(filename[trial_index]))
        # plt.show()


def plotAccidentalHunting():
    from palettable.cmocean.diverging import Balance_6
    from palettable.colorbrewer.diverging import RdBu_8, RdYlBu_5
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    data = readTrialData("../common_data/trial/accidental_100_trial_data_Omega-with_Q.pkl")
    filename = [each[0] for each in data]
    # del data
    weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/accidental_100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    hand_crafted_label = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/accidental_100_trial_data_Omega-with_Q-window3-w_intercept-handcrafted_labels.npy",
        allow_pickle=True
    )
    descriptive_weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/accidental_100_trial_data_Omega-with_Q-descriptive-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    trial_num = len(weight)
    random_indices = list(range(trial_num))
    np.random.shuffle(random_indices)
    cnt = 1
    for trial_index in random_indices:
        print("|{}| Trial Name : {}".format(cnt, filename[trial_index]))
        trial_data = data[trial_index][1]
        cnt += 1
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_true_accidental_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            continue
        while i < trial_data.shape[0]:
            if (trial_data.ifscared1.values[i] < 3 and trial_data.ifscared1.values[i-1] > 3):
                end_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        # Hand-crafted label and normalize contribution
        trial_weight = np.array(weight[trial_index])[start_index:end_index]
        trial_hand_crafted = np.array(hand_crafted_label[trial_index])[start_index:end_index]
        trial_descriptive_weight = np.array(descriptive_weight[trial_index])[start_index:end_index]
        window = 3
        # normalization
        for index in range(trial_weight.shape[0]):
            trial_weight[index, :] = trial_weight[index, :] / np.linalg.norm(trial_weight[index, :])
            trial_descriptive_weight[index, :] = trial_descriptive_weight[index, :] / np.linalg.norm(
                trial_descriptive_weight[index, :])
        # Plot weight variation of this trial
        colors = RdYlBu_5.mpl_colors
        agent_color = {
            "local": "#D7181C",
            "pessimistic": "#FDAE61",
            "global": "#44B53C",
            "suicide": "#836BB7",
            "planned_hunting": "#81B3FF",
            "vague": "black"
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
        }

        plt.figure(figsize=(18, 13))
        plt.subplot(3, 1, 1)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(trial_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_weight.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=20)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [start_index + window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))

        # plt.figure(figsize=(13,5))
        plt.subplot(3, 1, 2)
        for i in range(len(trial_hand_crafted)):
            if trial_hand_crafted[i] is not None:
                seq = np.linspace(-0.05, 0.0, len(trial_hand_crafted[i]) + 1)
                for j, h in enumerate(trial_hand_crafted[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
                # seq = np.linspace(-0.2, -0.1, len(estimated_label[i]) + 1)
                # for j, h in enumerate(estimated_label[i]):
                #     plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, trial_weight.shape[0])
        plt.xticks([], [])
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([-0.05, -0.15], ["Rule-Based Label", "Fitted Label"], fontsize=10)
        plt.yticks([-0.025], ["Rule-Based Label"], fontsize=15)
        # plt.ylim(-0.05, 0.35)
        # plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.title("Descriptive Agents ({})".format(filename[trial_index]), fontsize=10)
        for index in range(len(agent_name)):
            plt.plot(trial_descriptive_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_descriptive_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        # plt.legend(loc="upper center", fontsize=20, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))
        plt.savefig("../common_data/global_local_pessimistic_suicide_planned_hunting/accidental/{}.jpg".format(filename[trial_index]))
        # plt.show()


def plotSuicide():
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    data = readTrialData("../common_data/trial/suicide_100_trial_data_Omega-with_Q.pkl")
    filename = [each[0] for each in data]
    # del data
    weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/suicide_100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    hand_crafted_label = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/suicide_100_trial_data_Omega-with_Q-window3-w_intercept-handcrafted_labels.npy",
        allow_pickle=True
    )
    descriptive_weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/suicide_100_trial_data_Omega-with_Q-descriptive-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    trial_num = len(weight)
    random_indices = list(range(trial_num))
    np.random.shuffle(random_indices)
    cnt = 1
    for trial_index in random_indices:
        print("|{}| Trial Name : {}".format(cnt, filename[trial_index]))
        trial_data = data[trial_index][1]
        cnt += 1
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_suicide.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            continue
        start_index = max(0, start_index - 20)
        print("Star index : ", start_index)
        print("End index : ", end_index)
        # Hand-crafted label and normalize contribution
        trial_weight = np.array(weight[trial_index])[start_index:end_index]
        trial_hand_crafted = np.array(hand_crafted_label[trial_index])[start_index:end_index]
        trial_descriptive_weight = np.array(descriptive_weight[trial_index])[start_index:end_index]
        window = 3
        # normalization
        for index in range(trial_weight.shape[0]):
            trial_weight[index, :] = trial_weight[index, :] / np.linalg.norm(trial_weight[index, :])
            trial_descriptive_weight[index, :] = trial_descriptive_weight[index, :] / np.linalg.norm(
                trial_descriptive_weight[index, :])
        # Plot weight variation of this trial
        agent_color = {
            "local": "#D7181C",
            "pessimistic": "#FDAE61",
            "global": "#44B53C",
            "suicide": "#836BB7",
            "planned_hunting": "#81B3FF",
            "vague": "black"
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
        }

        plt.figure(figsize=(18, 13))
        plt.subplot(3, 1, 1)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(trial_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_weight.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=20)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [start_index + window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))

        # plt.figure(figsize=(13,5))
        plt.subplot(3, 1, 2)
        for i in range(len(trial_hand_crafted)):
            if trial_hand_crafted[i] is not None:
                seq = np.linspace(-0.05, 0.0, len(trial_hand_crafted[i]) + 1)
                for j, h in enumerate(trial_hand_crafted[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
                # seq = np.linspace(-0.2, -0.1, len(estimated_label[i]) + 1)
                # for j, h in enumerate(estimated_label[i]):
                #     plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, trial_weight.shape[0])
        plt.xticks([], [])
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([-0.05, -0.15], ["Rule-Based Label", "Fitted Label"], fontsize=10)
        plt.yticks([-0.025], ["Rule-Based Label"], fontsize=15)
        # plt.ylim(-0.05, 0.35)
        # plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.title("Descriptive Agents ({})".format(filename[trial_index]), fontsize=10)
        for index in range(len(agent_name)):
            plt.plot(trial_descriptive_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_descriptive_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        # plt.legend(loc="upper center", fontsize=20, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))
        plt.savefig("../common_data/global_local_pessimistic_suicide_planned_hunting/suicide/{}.jpg".format(filename[trial_index]))
        # plt.show()


def plotGlobal():
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    data = readTrialData("../common_data/trial/global_100_trial_data_Omega-with_Q.pkl")
    filename = [each[0] for each in data]
    # del data
    weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/global_100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    hand_crafted_label = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/global_100_trial_data_Omega-with_Q-window3-w_intercept-handcrafted_labels.npy",
        allow_pickle=True
    )
    descriptive_weight = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/global_100_trial_data_Omega-with_Q-descriptive-window3-w_intercept-contribution.npy",
        allow_pickle=True
    )
    trial_num = len(weight)
    random_indices = list(range(trial_num))
    np.random.shuffle(random_indices)
    cnt = 1
    for trial_index in random_indices:
        print("|{}| Trial Name : {}".format(cnt, filename[trial_index]))
        trial_data = data[trial_index][1]
        cnt += 1
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_global_optimal.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            continue
        start_index = max(0, start_index - 20)
        print("Star index : ", start_index)
        print("End index : ", end_index)
        # Hand-crafted label and normalize contribution
        trial_weight = np.array(weight[trial_index])[start_index:end_index]
        trial_hand_crafted = np.array(hand_crafted_label[trial_index])[start_index:end_index]
        trial_descriptive_weight = np.array(descriptive_weight[trial_index])[start_index:end_index]
        window = 3
        # normalization
        for index in range(trial_weight.shape[0]):
            trial_weight[index, :] = trial_weight[index, :] / np.linalg.norm(trial_weight[index, :])
            trial_descriptive_weight[index, :] = trial_descriptive_weight[index, :] / np.linalg.norm(
                trial_descriptive_weight[index, :])
        # Plot weight variation of this trial
        agent_color = {
            "local": "#D7181C",
            "pessimistic": "#FDAE61",
            "global": "#44B53C",
            "suicide": "#836BB7",
            "planned_hunting": "#81B3FF",
            "vague": "black"
        }
        label_name = {
            "local": "local",
            "pessimistic": "evade",
            "global": "global",
            "suicide": "suicide",
            "planned_hunting": "attack"
        }

        plt.figure(figsize=(18, 13))
        plt.subplot(3, 1, 1)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(trial_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_weight.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=20)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [start_index + window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))

        # plt.figure(figsize=(13,5))
        plt.subplot(3, 1, 2)
        for i in range(len(trial_hand_crafted)):
            if trial_hand_crafted[i] is not None:
                seq = np.linspace(-0.05, 0.0, len(trial_hand_crafted[i]) + 1)
                for j, h in enumerate(trial_hand_crafted[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
                # seq = np.linspace(-0.2, -0.1, len(estimated_label[i]) + 1)
                # for j, h in enumerate(estimated_label[i]):
                #     plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, trial_weight.shape[0])
        plt.xticks([], [])
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([-0.05, -0.15], ["Rule-Based Label", "Fitted Label"], fontsize=10)
        plt.yticks([-0.025], ["Rule-Based Label"], fontsize=15)
        # plt.ylim(-0.05, 0.35)
        # plt.axis('off')

        plt.subplot(3, 1, 3)
        plt.title("Descriptive Agents ({})".format(filename[trial_index]), fontsize=10)
        for index in range(len(agent_name)):
            plt.plot(trial_descriptive_weight[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=15)
        plt.xlim(0, trial_descriptive_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        x_ticks_index = np.linspace(0, len(trial_hand_crafted), 5)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.01, 1.02)
        # plt.legend(loc="upper center", fontsize=20, ncol=len(agent_name), frameon=False, bbox_to_anchor=(0.5, 1.2))
        plt.savefig("../common_data/global_local_pessimistic_suicide_planned_hunting/global/{}.jpg".format(filename[trial_index]))
        # plt.show()


def extractIndex():
    # ================================================================================================
    # Planned Data
    planned_data = readTrialData("../common_data/trial/new_100_trial_data_Omega-with_Q.pkl")
    # For detailed record
    detailed_record = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/planned_detailed_records.npy",
        allow_pickle=True
    )
    window = 3
    temp_detailed_record = []
    trial_num = len(planned_data)
    for trial_index in range(trial_num):
        trial_data = planned_data[trial_index][1]
        trial_record = detailed_record[trial_index]
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_true_planned_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            print("!!!!! IGNORE 1 !!!!!")
            continue
        while i < trial_data.shape[0]:
            if trial_data.ifscared1.values[i] == 3 or trial_data.ifscared2.values[i] == 3 or (
                    trial_data.ifscared1.values[i] < 3 and trial_data.ifscared1.values[i - 1] > 3):
                end_index = i
                break
            i += 1
        # if i == trial_data.shape[0]:
        #     print("!!!!! IGNORE 2 !!!!!")
        #     continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        temp_detailed_record.append(
            [
                trial_data.iloc[0].file,
                trial_record[0][start_index-window:end_index-window],
                trial_record[1][start_index-window:end_index-window],
                trial_record[2][start_index-window:end_index-window],
                trial_record[3][start_index-window:end_index-window],
                trial_record[4][start_index-window:end_index-window],
                [start_index, end_index]
            ]
        )
    np.save("../common_data/global_local_pessimistic_suicide_planned_hunting/sub_planned_detailed_records.npy", temp_detailed_record)
    # For descriptive record
    descriptive_record = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/planned_descriptive_records.npy",
        allow_pickle=True
    )
    temp_descriptive_record = []
    trial_num = len(planned_data)
    for trial_index in range(trial_num):
        trial_data = planned_data[trial_index][1]
        trial_record = descriptive_record[trial_index]
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_true_planned_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            print("!!!!! IGNORE 1 !!!!!")
            continue
        while i < trial_data.shape[0]:
            if trial_data.ifscared1.values[i] == 3 or trial_data.ifscared2.values[i] == 3 or (
                    trial_data.ifscared1.values[i] < 3 and trial_data.ifscared1.values[i - 1] > 3):
                end_index = i
                break
            i += 1
        # if i == trial_data.shape[0]:
        #     print("!!!!! IGNORE 2 !!!!!")
        #     continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        temp_descriptive_record.append(
            [
                trial_data.iloc[0].file,
                trial_record[0][start_index - window:end_index - window],
                trial_record[1][start_index - window:end_index - window],
                trial_record[2][start_index - window:end_index - window],
                trial_record[3][start_index - window:end_index - window],
                trial_record[4][start_index - window:end_index - window],
                [start_index, end_index]
            ]
        )
    np.save("../common_data/global_local_pessimistic_suicide_planned_hunting/sub_planned_descriptive_records.npy",
            temp_descriptive_record)

    # ================================================================================================
    # Accidental Data
    accidental_data = readTrialData("../common_data/trial/accidental_100_trial_data_Omega-with_Q.pkl")
    # For detailed record
    detailed_record = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/accidental_detailed_records.npy",
        allow_pickle=True
    )
    window = 3
    temp_detailed_record = []
    trial_num = len(planned_data)
    for trial_index in range(trial_num):
        trial_data = accidental_data[trial_index][1]
        trial_record = detailed_record[trial_index]
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_true_accidental_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            print("!!!!! IGNORE 1 !!!!!")
            continue
        while i < trial_data.shape[0]:
            if (trial_data.ifscared1.values[i] < 3 and trial_data.ifscared1.values[i - 1] > 3):
                end_index = i
                break
            i += 1
        # if i == trial_data.shape[0]:
        #     print("!!!!! IGNORE 2 !!!!!")
        #     continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        temp_detailed_record.append(
            [
                trial_data.iloc[0].file,
                trial_record[0][start_index - window:end_index - window],
                trial_record[1][start_index - window:end_index - window],
                trial_record[2][start_index - window:end_index - window],
                trial_record[3][start_index - window:end_index - window],
                trial_record[4][start_index - window:end_index - window],
                [start_index, end_index]
            ]
        )
    np.save("../common_data/global_local_pessimistic_suicide_planned_hunting/sub_accidental_detailed_records.npy",
            temp_detailed_record)
    # For descriptive record
    descriptive_record = np.load(
        "../common_data/global_local_pessimistic_suicide_planned_hunting/accidental_descriptive_records.npy",
        allow_pickle=True
    )
    window = 3
    temp_descriptive_record = []
    trial_num = len(accidental_data)
    for trial_index in range(trial_num):
        trial_data = accidental_data[trial_index][1]
        trial_record = descriptive_record[trial_index]
        # Fin planned hunting trajectory
        start_index = 0
        end_index = trial_data.shape[0]
        i = 0
        while i < trial_data.shape[0]:
            if trial_data.label_true_accidental_hunting.values[i] == 1:
                start_index = i
                break
            i += 1
        if i == trial_data.shape[0]:
            print("!!!!! IGNORE 1 !!!!!")
            continue
        while i < trial_data.shape[0]:
            if (trial_data.ifscared1.values[i] < 3 and trial_data.ifscared1.values[i - 1] > 3):
                end_index = i
                break
            i += 1
        # if i == trial_data.shape[0]:
        #     print("!!!!! IGNORE 2 !!!!!")
        #     continue
        end_index += 1
        print("Star index : ", start_index)
        print("End index : ", end_index)
        temp_descriptive_record.append(
            [
                trial_data.iloc[0].file,
                trial_record[0][start_index - window:end_index - window],
                trial_record[1][start_index - window:end_index - window],
                trial_record[2][start_index - window:end_index - window],
                trial_record[3][start_index - window:end_index - window],
                trial_record[4][start_index - window:end_index - window],
                [start_index, end_index]
            ]
        )
    np.save("../common_data/global_local_pessimistic_suicide_planned_hunting/sub_accidental_descriptive_records.npy",
            temp_descriptive_record)



if __name__ == '__main__':
    # # Pre-estimation
    # preEstimation()


    # Configurations
    pd.options.mode.chained_assignment = None

    config = {
        # TODO: ===================================
        # TODO:       Always set to True
        # TODO: ===================================
        "need_intercept" : True,

        # ==================================================================================
        #                       For Correlation Analysis and Multiple Label Analysis
        # Filename
        "trial_data_filename": "../common_data/trial/global_100_trial_data_Omega-with_Q-descriptive.pkl",
        # The number of trials used for analysis
        "trial_num" : None,
        # Window size for correlation analysis
        "trial_window" : 3,
        "multi_agent_list" : ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        # ==================================================================================
        "incremental_window" : 3,
        "incremental_data_filename" : "../common_data/trial/100_trial_data_Omega-with_Q-descriptive.pkl",
        "incremental_num_trial" : None,

        "single_trial_data_filename" : "../common_data/trial/test_planned_trial_data_Omega-with_Q-descriptive.pkl",
        "single_trial_window" : 3,

    }


    # ============ MOVING WINDOW =============

    # multiAgentAnalysis(config)

    # plotComparison()
    # plotPlannedHunting()
    # plotAccidentalHunting()
    plotSuicide()
    plotGlobal()

    # incrementalAnalysis(config)

    # singleTrialAllFitting(config)

    # data = np.load("../common_data/global_local_pessimistic_suicide_planned_hunting/detailed_records.npy", allow_pickle=True)
    # print()

    # data = readTrialData("../common_data/trial/accidental_100_trial_data_Omega-with_Q.pkl")
    # filename = [each[0] for each in data]
    # np.save("../common_data/trial/accidental_trial_name.npy", filename)

    # extractIndex()

    # data = np.load("../common_data/global_local_pessimistic_suicide_planned_hunting/sub_planned_descriptive_records.npy", allow_pickle=True)
    # print()