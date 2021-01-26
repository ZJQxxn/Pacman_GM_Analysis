'''
Description:
    Compare simulated labels with hand-crafted labels.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    4 Jan. 2021
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


def _PRDistGlobal(x, locs_df):
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
    cnt = 0
    for each in PR_dist:
        if 10<=each<=15:
            cnt+=1
    return cnt


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
        # if np.any(np.array(PG[index]) <= 10) and np.all(np.array(ghost_status[index]) < 3):
        if np.all(np.array(ghost_status[index]) < 3) and np.any(np.array(PG[index]) <= 10):
            temp_pess_Q[index][non_zero] = temp_pess_Q[index][non_zero] + offset
        else:
            temp_pess_Q[index][non_zero] = 0.0
    return temp_pess_Q


def _plannedHuntingProcesing(planned_Q, ghost_status, energizer_num, PE, PG):
    if np.any(np.concatenate(planned_Q) < 0):
        temp_planned_Q = np.concatenate(planned_Q)
        temp_planned_Q[temp_planned_Q > 0] = 0.0
        offset = np.max(np.abs(temp_planned_Q)) # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_planned_Q = copy.deepcopy(planned_Q)
    for index in range(len(temp_planned_Q)):
        non_zero = np.where(temp_planned_Q[index] != 0)
        # if (np.all(np.array(ghost_status[index]) <= 3) and energizer_num[index] == 0) \
        #         or (np.any(np.array(ghost_status[index]) < 3) and PE[index] >= 15) \
        #         or np.all(np.array(ghost_status[index]) == 3) or np.all(np.array(PG[index]) >= 15) or np.any(np.array(ghost_status[index]) > 3):
        if (np.all(np.array(ghost_status[index]) <= 3) and energizer_num[index] == 0) \
                or np.all(np.array(ghost_status[index]) == 3) or np.any(np.array(ghost_status[index]) > 3)\
                or PE[index] > 10:
            temp_planned_Q[index][non_zero] = 0.0
        else:
            temp_planned_Q[index][non_zero] = temp_planned_Q[index][non_zero] + offset
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
        # if np.all(np.array(ghost_status[index]) == 3) or np.all(np.array(PG[index]) > 10):
        if np.all(np.array(ghost_status[index]) == 3) or np.all(np.array(PG[index]) > 10):
            temp_suicide_Q[index][non_zero] = 0.0
        else:
            temp_suicide_Q[index][non_zero] = temp_suicide_Q[index][non_zero] + offset
    return temp_suicide_Q


def _globalProcesing(global_Q, PR_global):
    if np.any(np.concatenate(global_Q) < 0):
        temp_global_Q = np.concatenate(global_Q)
        temp_global_Q[temp_global_Q > 0] = 0.0
        offset = np.max(np.abs(temp_global_Q))  # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_global_Q = copy.deepcopy(global_Q)
    for index in range(len(temp_global_Q)):
        non_zero = np.where(temp_global_Q[index] != 0)
        if PR_global[index] == 0: # no reward in the range of 10-15
            temp_global_Q[index][non_zero] = 0.0
        else:
            temp_global_Q[index][non_zero] = temp_global_Q[index][non_zero] + offset
    return temp_global_Q


def _localProcesing(local_Q, PR):
    if np.any(np.concatenate(local_Q) < 0):
        temp_local_Q = np.concatenate(local_Q)
        temp_local_Q[temp_local_Q > 0] = 0.0
        offset = np.max(np.abs(temp_local_Q))  # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_local_Q = copy.deepcopy(local_Q)
    for index in range(len(temp_local_Q)):
        non_zero = np.where(temp_local_Q[index] != 0)
        if PR[index] > 5:
            temp_local_Q[index][non_zero] = 0.0
        else:
            temp_local_Q[index][non_zero] = temp_local_Q[index][non_zero] + offset
    return temp_local_Q


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
                    all_data.pessimistic_blinky_Q[index] = all_data.pessimistic_blinky_Q[index - 2]
                    all_data.pessimistic_clyde_Q[index] = all_data.pessimistic_clyde_Q[index - 2]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index - 2]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index - 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index - 1]
                    all_data.local_Q[index] = all_data.local_Q[index - 1]
                    all_data.pessimistic_blinky_Q[index] = all_data.pessimistic_blinky_Q[index - 1]
                    all_data.pessimistic_clyde_Q[index] = all_data.pessimistic_clyde_Q[index - 1]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index - 1]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index - 1]
            else:
                if isinstance(all_data.global_Q[index + 1], list):
                    all_data.global_Q[index] = all_data.global_Q[index + 2]
                    all_data.local_Q[index] = all_data.local_Q[index + 2]
                    all_data.pessimistic_blinky_Q[index] = all_data.pessimistic_blinky_Q[index + 2]
                    all_data.pessimistic_clyde_Q[index] = all_data.pessimistic_clyde_Q[index + 2]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index + 2]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index + 2]
                else:
                    all_data.global_Q[index] = all_data.global_Q[index + 1]
                    all_data.local_Q[index] = all_data.local_Q[index + 1]
                    all_data.pessimistic_blinky_Q[index] = all_data.pessimistic_blinky_Q[index + 1]
                    all_data.pessimistic_clyde_Q[index] = all_data.pessimistic_clyde_Q[index + 1]
                    all_data.suicide_Q[index] = all_data.suicide_Q[index + 1]
                    all_data.planned_hunting_Q[index] = all_data.planned_hunting_Q[index + 1]
    # Pre-processng pessimistic Q
    # TODO: check this
    try:
        locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    except:
        locs_df = readLocDistance("./Utility_Tree_Analysis/extracted_data/dij_distance_map.csv")
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
    all_data.pessimistic_blinky_Q = _pessimisticProcesing(all_data.pessimistic_blinky_Q, PG, ghost_status)
    all_data.pessimistic_clyde_Q = _pessimisticProcesing(all_data.pessimistic_clyde_Q, PG, ghost_status)
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE, PG_wo_dead)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG_wo_dead)
    # all_data.local_Q = _localProcesing(all_data.local_Q, PR)
    # all_data.global_Q = _globalProcesing(all_data.global_Q, PR_global)
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
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]]
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
    agent_index = [["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"].index(each) for each in agent_name]

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
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, len(agent_name), 4))
        trial_estimated_label = []
        temp_contribution = []
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window+1]
            sub_Y = Y[centering_point - window:centering_point + window+1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(len(agent_name)):  # number of agents
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
        trial_record.append(trial_name)
        record.append(copy.deepcopy(trial_record))

    # Save data
    dir_names = "-".join(agent_name)
    if dir_names not in os.listdir("../common_data"):
        os.mkdir("../common_data/{}".format(dir_names))
    save_base = config["trial_data_filename"].split("/")[-1].split(".")[0]
    np.save("../common_data/{}/equal-{}-window{}-{}_intercept-multi_labels.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), all_estimated_label)
    np.save("../common_data/{}/equal-{}-window{}-{}_intercept-handcrafted_labels.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), handcrafted_labels)
    np.save("../common_data/{}/equal-{}-window{}-{}_intercept-matching_rate.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_matching_rate)
    np.save("../common_data/{}/equal-{}-window{}-{}_intercept-trial_weight.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_weight)
    np.save("../common_data/{}/equal-{}-window{}-{}_intercept-Q.npy".format(
        dir_names, save_base, window, "w" if config["need_intercept"] else "wo"), trial_Q)
    np.save("../common_data/{}/equal-{}-window{}-{}_intercept-contribution.npy".format(
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
    if data_type is not None:
        np.save("../common_data/{}/{}_path10_records.npy".format(dir_names, data_type), record)
    else:
        np.save("../common_data/uniform_path10_records.npy".format(dir_names, data_type), record)

# ===============================================
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
        ["local", "global"],
        ["local", "pessimistic_blinky", "global"],
        ["local", "pessimistic_clyde", "global"],
        ["local", "pessimistic_blinky", "pessimistic_clyde", "global"],
        ["local", "pessimistic_blinky", "pessimistic_clyde", "global", "planned_hunting"],
        ["local", "pessimistic_blinky", "pessimistic_clyde", "global", "planned_hunting", "suicide"]
    ]
    all_cr = []
    all_contribution = []
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
    np.save("../common_data/incremental/path10-window{}-incremental_cr-{}_intercept.npy".format(
        window, "w" if config["need_intercept"] else "wo"), all_cr)


def decrementalAnalysis(config):
    # Read trial data
    # agent_name = config["incremental_data_filename"]
    # agents_list = ["{}_Q".format(each) for each in agent_name]
    print("=== Decremental Analysis ====")
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
    # Decremental analysis
    incremental_agents_list = [
        ["local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"], # w/o global
        ["global", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"], # w/o local
        ["global", "local", "pessimistic_clyde", "suicide", "planned_hunting"], # w/o blinky
        ["global", "local", "pessimistic_blinky", "suicide", "planned_hunting"],  # w/o clyde
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "planned_hunting"],# w/o suicide
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide"], # w/o planned hunting
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"] # all the agents
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
    if "decremental" not in os.listdir("../common_data"):
        os.mkdir("../common_data/decremental")
    np.save("../common_data/decremental/path10-{}trial-window{}-incremental_cr-{}_intercept.npy".format(
        config["incremental_num_trial"], window, "w" if config["need_intercept"] else "wo"), all_cr)


def oneAgentAnalysis(config):
    # Read trial data
    # agent_name = config["incremental_data_filename"]
    # agents_list = ["{}_Q".format(each) for each in agent_name]
    print("=== One Agent Analysis ====")
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
        ["global"],
        ["local"],
        ["pessimistic_blinky"],
        ["pessimistic_clyde"],
        ["suicide"],
        ["planned_hunting"]
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
    if "one_agent" not in os.listdir("../common_data"):
        os.mkdir("../common_data/one_agent")
    np.save("../common_data/one_agent/path10-{}trial-window{}-incremental_cr-{}_intercept.npy".format(
        config["incremental_num_trial"], window, "w" if config["need_intercept"] else "wo"), all_cr)


def stageAnalysis(config):
    # Read trial data
    print("=== Stage Together Analysis ====")
    print(config["incremental_data_filename"])
    data = readTrialData(config["incremental_data_filename"])
    all_X = pd.concat([each[1] for each in data])
    all_Y = pd.concat([each[2] for each in data])
    print("Shape of data : ", all_X.shape)
    beans_num = all_X.beans.apply(lambda x: len(x) if not isinstance(x, float) else 0)
    early_index = np.where(beans_num >= 80)[0]
    medium_index = np.intersect1d(np.where(10 < beans_num)[0], np.where(beans_num < 80)[0])
    end_index = np.where(beans_num <= 10)[0]
    stage_index = [early_index, medium_index, end_index]
    stage_name = ["early", "medium", "end"]
    # Incremental analysis
    incremental_agents_list = [
        ["local"],
        ["local", "global"],
        ["local", "pessimistic_blinky", "global"],
        ["local", "pessimistic_clyde", "global"],
        ["local", "pessimistic_blinky", "pessimistic_clyde", "global"],
        ["local", "pessimistic_blinky", "pessimistic_clyde", "global", "planned_hunting"],
        ["local", "pessimistic_blinky", "pessimistic_clyde", "global", "planned_hunting", "suicide"]
    ]
    all_cr = {"early":[], "medium":[], "end":[]}
    for i, index in enumerate(stage_index):
        print("-" * 15)
        print(stage_name[i])
        X = all_X.iloc[index]
        Y = all_Y.iloc[index]
        stage_length = X.shape[0]
        print("Stage length : ", stage_length)
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
            print("{} | {}".format(agent_name, correct_rate))
            all_cr[stage_name[i]].append(correct_rate)
    # save correct rate data
    if "stage_together" not in os.listdir("../common_data"):
        os.mkdir("../common_data/stage_together")
    filename = "../common_data/stage_together/path10-{}trial-cr.npy".format(config["incremental_num_trial"])
    np.save(filename, all_cr)


def stageCombineAnalysis(config):
    # Read trial data
    print("=== Stage Combine Analysis (use all for MLE) ====")
    print(config["incremental_data_filename"])
    data = readTrialData(config["incremental_data_filename"])
    all_X = pd.concat([each[1] for each in data])
    all_Y = pd.concat([each[2] for each in data])
    print("Shape of data : ", len(data))
    # Incremental analysis
    incremental_agents_list = [
        ["global"],
        ["local"],
        ["pessimistic_blinky"],
        ["pessimistic_clyde"],
        ["suicide"],
        ["planned_hunting"],
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]
    ]
    all_cr = []
    weight = []
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
            all_X,
            all_Y,
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
        # compute contribution
        Q_list = ["{}_Q".format(each) for each in agent_name]
        cur_weight = res.x[:-1]
        for i, j in enumerate(Q_list):
            Q_value = all_X[j].values
            Q_scale = scaleOfNumber(np.concatenate(np.abs(Q_value)).max())
            cur_weight[i] *= Q_scale
        weight.append(cur_weight)
        # correct rate for each trial
        trial_cr = []
        for each_trial in data:
            trial_X = each_trial[1]
            trial_Y = each_trial[2]
            _, estimated_prob = negativeLikelihood(
                res.x,
                trial_X,
                trial_Y,
                agent_name,
                return_trajectory=True,
                need_intercept=config["need_intercept"]
            )
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = trial_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            trial_cr.append(correct_rate)
        all_cr.append(copy.deepcopy(trial_cr))
        print("{} | Avg Cr : {}".format(agent_name, np.nanmean(trial_cr)))

    # save correct rate data
    if "stage_together" not in os.listdir("../common_data"):
        os.mkdir("../common_data/stage_together")
    filename = "../common_data/stage_together/path10-all-100trial-cr.npy"
    np.save(filename, all_cr)
    np.save("../common_data/stage_together/path10-all-100trial-weight.npy", weight)


def specialCaseAnalysis(config):
    # Read trial data
    print("=== Special Case Analysis ====")
    print(config["incremental_data_filename"])
    data = readTrialData(config["incremental_data_filename"])
    all_X = pd.concat([each[1] for each in data])
    all_Y = pd.concat([each[2] for each in data])
    print("Shape of data : ", all_X.shape)
    # Incremental analysis
    incremental_agents_list = [
        ["global"],
        ["local"],
        ["pessimistic_blinky"],
        ["pessimistic_clyde"],
        ["suicide"],
        ["planned_hunting"],
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]
    ]
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading distance file")
    # cr_dict = {"end":[], "close-normal":[], "close-scared":[]}
    end_index = all_X.beans.apply(lambda x: len(x) <= 10 if not isinstance(x, float) else True)
    end_index = np.where(end_index== True)[0]
    early_index = all_X.beans.apply(lambda x: len(x) >= 80 if not isinstance(x, float) else False)
    early_index = np.where(early_index == True)[0]
    middle_index = all_X.beans.apply(lambda x: 10 < len(x) < 80 if not isinstance(x, float) else False)
    middle_index = np.where(middle_index == True)[0]
    scared_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 > 3 or x.ifscared2 > 3, axis = 1)
    scared_index = np.where(scared_index== True)[0]
    normal_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 or x.ifscared2 < 3, axis = 1)
    normal_index = np.where(normal_index == True)[0]
    close_index = all_X[["pacmanPos", "ghost1Pos"]].apply(
        lambda x: True if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 10,
        axis = 1
    )
    close_index = np.where(close_index== True)[0]
    cr_index = {
        "early":early_index,
        "middle":middle_index,
        "end":end_index,
        "close-normal":np.intersect1d(close_index, normal_index),
        "close-scared":np.intersect1d(close_index, scared_index)
    }
    cr_weight = {
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": []
    }
    cr_contribuion = {
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": []
    }
    cr_trial = {
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": []
    }
    for case in cr_index:
        print("-"*20)
        print("Case : ", case)
        X = all_X.iloc[cr_index[case]]
        Y = all_Y.iloc[cr_index[case]]
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
            cr_weight[case].append(copy.deepcopy(res.x))
            # compute contribution
            Q_list = ["{}_Q".format(each) for each in agent_name]
            cur_weight = res.x[:-1]
            for i,j in enumerate(Q_list):
                Q_value = all_X[j].values
                Q_scale = scaleOfNumber(np.concatenate(np.abs(Q_value)).max())
                cur_weight[i] *= Q_scale
            cr_contribuion[case].append(copy.deepcopy(cur_weight))
    # correct rate in the window
    for each_trial in data:
        trial_X = each_trial[1]
        trial_Y = each_trial[2]
        end_index = trial_X.beans.apply(lambda x: len(x) <= 10 if not isinstance(x, float) else True)
        end_index = np.where(end_index == True)[0]
        early_index = trial_X.beans.apply(lambda x: len(x) >= 80 if not isinstance(x, float) else True)
        early_index = np.where(early_index == True)[0]
        middle_index = trial_X.beans.apply(lambda x: 10 < len(x) < 80 if not isinstance(x, float) else True)
        middle_index = np.where(middle_index == True)[0]
        scared_index = trial_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 > 3 or x.ifscared2 > 3, axis=1)
        scared_index = np.where(scared_index == True)[0]
        normal_index = trial_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 or x.ifscared2 < 3, axis=1)
        normal_index = np.where(normal_index == True)[0]
        close_index = trial_X[["pacmanPos", "ghost1Pos"]].apply(
            lambda x: True if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 5,
            axis=1
        )
        close_index = np.where(close_index == True)[0]
        trial_index = {
            "early": early_index,
            "middle": middle_index,
            "end": end_index,
            "close-normal": np.intersect1d(close_index, normal_index),
            "close-scared": np.intersect1d(close_index, scared_index)
        }
        for case in trial_index:
            sub_X = trial_X.iloc[trial_index[case]]
            sub_Y = trial_Y.iloc[trial_index[case]]
            trial_cr = []
            for agent_index, agent_name in enumerate(incremental_agents_list):
                _, estimated_prob = negativeLikelihood(
                    cr_weight[case][agent_index],
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=True,
                    need_intercept=config["need_intercept"]
                )
                estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
                true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
                correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
                trial_cr.append(correct_rate)
            cr_trial[case].append(copy.deepcopy(trial_cr))
    # save correct rate data
    if "special_case" not in os.listdir("../common_data"):
        os.mkdir("../common_data/special_case")
    filename = "../common_data/special_case/path10-100trial-cr.npy"
    np.save(filename, cr_trial)
    np.save("../common_data/special_case/path10-100trial-contribution.npy", cr_contribuion)


def closedGhostAnalysis(config):
    # Read trial data
    print("=== Closed Ghost Analysis ====")
    print(config["incremental_data_filename"])
    data = readTrialData(config["incremental_data_filename"])
    all_X = pd.concat([each[1] for each in data])
    all_Y = pd.concat([each[2] for each in data])
    print("Shape of data : ", all_X.shape)
    # Incremental analysis
    incremental_agents_list = [
        ["global"],
        ["local"],
        ["pessimistic_blinky"],
        ["pessimistic_clyde"],
        ["suicide"],
        ["planned_hunting"],
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]
    ]
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading distance file")

    blinky_scared_index = all_X.ifscared1.apply(lambda x: x > 3)
    blinky_scared_index = np.where(blinky_scared_index == True)[0]

    clyde_scared_index = all_X.ifscared2.apply(lambda x: x > 3)
    clyde_scared_index = np.where(clyde_scared_index == True)[0]

    blinky_normal_index = all_X.ifscared1.apply(lambda x: x < 3)
    blinky_normal_index = np.where(blinky_normal_index == True)[0]

    clyde_normal_index = all_X.ifscared2.apply(lambda x: x < 3)
    clyde_normal_index = np.where(clyde_normal_index == True)[0]

    blinky_close_index = all_X[["pacmanPos", "ghost1Pos", "label_evade1"]].apply(
        lambda x: False if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 5 and x.label_evade1,
        axis=1
    )
    blinky_close_index = np.where(blinky_close_index == True)[0]

    clyde_close_index = all_X[["pacmanPos", "ghost2Pos", "label_evade2"]].apply(
        lambda x: False if x.pacmanPos == x.ghost2Pos else locs_df[x.pacmanPos][x.ghost2Pos] <= 5 and x.label_evade2,
        axis=1
    )
    clyde_close_index = np.where(clyde_close_index == True)[0]


    cr_index = {
        "blinky-close-normal": np.intersect1d(blinky_close_index, blinky_normal_index),
        "clyde-close-normal": np.intersect1d(clyde_close_index, clyde_normal_index),
        "blinky-close-scared": np.intersect1d(blinky_close_index, blinky_scared_index),
        "clyde-close-scared": np.intersect1d(clyde_close_index, clyde_scared_index),
    }
    cr_weight = {
        "blinky-close-normal": [],
        "clyde-close-normal": [],
        "blinky-close-scared": [],
        "clyde-close-scared": [],
    }
    cr_contribuion = {
        "blinky-close-normal": [],
        "clyde-close-normal": [],
        "blinky-close-scared": [],
        "clyde-close-scared": [],
    }
    cr_trial = {
        "blinky-close-normal": [],
        "clyde-close-normal": [],
        "blinky-close-scared": [],
        "clyde-close-scared": [],
    }
    for case in cr_index:
        print("-" * 20)
        print("Case : ", case)
        X = all_X.iloc[cr_index[case]]
        Y = all_Y.iloc[cr_index[case]]
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
            cr_weight[case].append(copy.deepcopy(res.x))
            # compute contribution
            Q_list = ["{}_Q".format(each) for each in agent_name]
            cur_weight = res.x[:-1]
            for i, j in enumerate(Q_list):
                Q_value = all_X[j].values
                Q_scale = scaleOfNumber(np.concatenate(np.abs(Q_value)).max())
                cur_weight[i] *= Q_scale
            cr_contribuion[case].append(copy.deepcopy(cur_weight))
    # correct rate in the window
    for each_trial in data:
        trial_X = each_trial[1]
        trial_Y = each_trial[2]

        blinky_scared_index = trial_X.ifscared1.apply(lambda x: x > 3)
        blinky_scared_index = np.where(blinky_scared_index == True)[0]
        clyde_scared_index = trial_X.ifscared2.apply(lambda x: x > 3)
        clyde_scared_index = np.where(clyde_scared_index == True)[0]
        blinky_normal_index = trial_X.ifscared1.apply(lambda x: x < 3)
        blinky_normal_index = np.where(blinky_normal_index == True)[0]
        clyde_normal_index = trial_X.ifscared2.apply(lambda x: x < 3)
        clyde_normal_index = np.where(clyde_normal_index == True)[0]
        blinky_close_index = trial_X[["pacmanPos", "ghost1Pos", "label_evade1"]].apply(
            lambda x: False if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 5 and x.label_evade1,
            axis=1
        )
        blinky_close_index = np.where(blinky_close_index == True)[0]
        clyde_close_index = trial_X[["pacmanPos", "ghost2Pos", "label_evade2"]].apply(
            lambda x: False if x.pacmanPos == x.ghost2Pos else locs_df[x.pacmanPos][x.ghost2Pos] <= 5 and x.label_evade2,
            axis=1
        )
        clyde_close_index = np.where(clyde_close_index == True)[0]

        trial_index = {
            "blinky-close-normal": np.intersect1d(blinky_close_index, blinky_normal_index),
            "clyde-close-normal": np.intersect1d(clyde_close_index, clyde_normal_index),
            "blinky-close-scared": np.intersect1d(blinky_close_index, blinky_scared_index),
            "clyde-close-scared": np.intersect1d(clyde_close_index, clyde_scared_index),
        }

        for case in trial_index:
            sub_X = trial_X.iloc[trial_index[case]]
            sub_Y = trial_Y.iloc[trial_index[case]]
            trial_cr = []
            for agent_index, agent_name in enumerate(incremental_agents_list):
                _, estimated_prob = negativeLikelihood(
                    cr_weight[case][agent_index],
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=True,
                    need_intercept=config["need_intercept"]
                )
                estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
                true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
                correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
                trial_cr.append(correct_rate)
            cr_trial[case].append(copy.deepcopy(trial_cr))
    # save correct rate data
    if "closed_ghost" not in os.listdir("../common_data"):
        os.mkdir("../common_data/closed_ghost")
    filename = "../common_data/closed_ghost/path10-100trial-cr.npy"
    np.save(filename, cr_trial)
    np.save("../common_data/closed_ghost/path10-100trial-contribution.npy", cr_contribuion)


def specialCaseMovingAnalysis(config):
    # Read trial data
    print("=== Special Case (Moving Window) ====")
    print(config["incremental_data_filename"])
    data = readTrialData(config["incremental_data_filename"])
    window = 3
    # Incremental analysis
    incremental_agents_list = [
        ["global"],
        ["local"],
        ["pessimistic_blinky"],
        ["pessimistic_clyde"],
        ["suicide"],
        ["planned_hunting"],
        ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]
    ]
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading distance file")
    cr_trial = {
        "early": [],
        "middle": [],
        "end": [],
        "close-normal": [],
        "close-scared": []
    }

    # for each trial
    for trial_index, each_trial in enumerate(data):
        print("|{}| Trial Name : {}".format(trial_index, each_trial[0]))
        all_X = each_trial[1]
        all_Y = each_trial[2]
        trial_length = all_X.shape[0]
        end_index = all_X.beans.apply(lambda x: len(x) <= 10 if not isinstance(x, float) else True)
        end_index = np.where(end_index == True)[0]
        early_index = all_X.beans.apply(lambda x: len(x) >= 80 if not isinstance(x, float) else False)
        early_index = np.where(early_index == True)[0]
        middle_index = all_X.beans.apply(lambda x: 10 < len(x) < 80 if not isinstance(x, float) else False)
        middle_index = np.where(middle_index == True)[0]
        scared_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 > 3 or x.ifscared2 > 3, axis=1)
        scared_index = np.where(scared_index == True)[0]
        normal_index = all_X[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 or x.ifscared2 < 3, axis=1)
        normal_index = np.where(normal_index == True)[0]
        close_index = all_X[["pacmanPos", "ghost1Pos"]].apply(
            lambda x: True if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos] <= 5,
            axis=1
        )
        close_index = np.where(close_index == True)[0]
        cr_index = {
            "early": early_index,
            "middle": middle_index,
            "end": end_index,
            "close-normal": np.intersect1d(close_index, normal_index),
            "close-scared": np.intersect1d(close_index, scared_index)
        }
        print("Trial length : ", trial_length)
        trial_early_cr = []
        trial_middle_cr = []
        trial_end_cr = []
        trial_close_normal_cr = []
        trial_close_scared_cr = []
        window_index = np.arange(window, trial_length - window)
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = all_X[centering_point - window:centering_point + window + 1]
            sub_Y = all_Y[centering_point - window:centering_point + window + 1]
            agent_cr = []
            for agent_index, agent_name in enumerate(incremental_agents_list):
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
                # estimation
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
            # Check the type of this data
            if centering_point in cr_index["early"]:
                trial_early_cr.append(copy.deepcopy(agent_cr))
            if centering_point in cr_index["middle"]:
                trial_middle_cr.append(copy.deepcopy(agent_cr))
            if centering_point in cr_index["end"]:
                trial_end_cr.append(copy.deepcopy(agent_cr))
            if centering_point in cr_index["close-normal"]:
                trial_close_normal_cr.append(copy.deepcopy(agent_cr))
            if centering_point in cr_index["close-scared"]:
                trial_close_scared_cr.append(copy.deepcopy(agent_cr))
        # Assign types
        if len(trial_end_cr) > 0:
            cr_trial["end"].append(copy.deepcopy(trial_end_cr))
        if len(trial_middle_cr) > 0:
            cr_trial["middle"].append(copy.deepcopy(trial_middle_cr))
        if len(trial_early_cr) > 0:
            cr_trial["early"].append(copy.deepcopy(trial_early_cr))
        if len(trial_close_scared_cr) > 0:
            cr_trial["close-scared"].append(copy.deepcopy(trial_close_scared_cr))
        if len(trial_close_normal_cr) > 0:
            cr_trial["close-normal"].append(copy.deepcopy(trial_close_normal_cr))
    # save correct rate data
    if "special_case" not in os.listdir("../common_data"):
        os.mkdir("../common_data/special_case")
    filename = "../common_data/special_case/path10-100trial-moving_window-cr.npy"
    np.save(filename, cr_trial)


def diffLabelAnalysis():
    print("="*20, " Diff State Analysis ", "="*20)
    filename = "../common_data/trial/100_trial_data_Omega-with_Q-path10.pkl"
    print(filename)
    data = readTrialData(filename)
    # data = [data[i] for i in range(10)]
    print("Num of trials : ", len(data))
    window = 3
    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade", "label_evade1",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]]
    agent_name_list = [["local"], ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]]
    local_cr = []
    global_cr = []
    evade_cr = []
    suicide_cr = []
    attack_cr = []
    vague_cr = []
    for index, each in enumerate(data):
        trial_name = each[0]
        print("-"*40)
        print("|{}| Trial Name : {}".format(index + 1, trial_name))
        trial_X = each[1]
        trial_Y = each[2]
        handcrafted_label = [_handcraftLabeling(trial_X[label_list].iloc[index]) for index in range(trial_X.shape[0])]
        # handcrafted_label = handcrafted_label[window:-window]
        # Moving window analysis
        trial_local_cr = []
        trial_global_cr = []
        trial_evade_cr = []
        trial_suicide_cr = []
        trial_attack_cr = []
        trial_vague_cr = []
        trial_length = trial_X.shape[0]
        print("Length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            temp_cr = []
            print("Window at {}...".format(centering_point))
            sub_X = trial_X[centering_point - window:centering_point + window + 1]
            sub_Y = trial_Y[centering_point - window:centering_point + window + 1]
            for agent_name in agent_name_list:
                # Construct optimizer
                params = [0 for _ in range(len(agent_name))]
                bounds = [[0, 10] for _ in range(len(agent_name))]
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
                # correct rate in the window
                _, estimated_prob = negativeLikelihood(
                    res.x,
                    sub_X,
                    sub_Y,
                    agent_name,
                    return_trajectory=True,
                    need_intercept=True
                )
                # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
                estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
                true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
                correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
                temp_cr.append(correct_rate)
            # Assign label
            if handcrafted_label[centering_point] is None:
                continue
            else:
                if len(handcrafted_label[centering_point]) > 1:
                    trial_vague_cr.append(copy.deepcopy(temp_cr))
                elif handcrafted_label[centering_point] == ["local"]:
                    trial_local_cr.append(copy.deepcopy(temp_cr))
                elif handcrafted_label[centering_point] == ["global"]:
                    trial_global_cr.append(copy.deepcopy(temp_cr))
                elif handcrafted_label[centering_point] == ["pessimistic"]:
                    trial_evade_cr.append(copy.deepcopy(temp_cr))
                elif handcrafted_label[centering_point] == ["suicide"]:
                    trial_suicide_cr.append(copy.deepcopy(temp_cr))
                elif handcrafted_label[centering_point] == ["planned_hunting"]:
                    trial_attack_cr.append(copy.deepcopy(temp_cr))
                else:
                    continue
        # Trial
        if len(trial_local_cr) > 0:
            local_cr.append(copy.deepcopy(trial_local_cr))
        if len(trial_global_cr) > 0:
            global_cr.append(copy.deepcopy(trial_global_cr))
        if len(trial_evade_cr) > 0:
            evade_cr.append(copy.deepcopy(trial_evade_cr))
        if len(trial_attack_cr) > 0:
            attack_cr.append(copy.deepcopy(trial_attack_cr))
        if len(trial_suicide_cr) > 0:
            suicide_cr.append(copy.deepcopy(trial_suicide_cr))
        if len(trial_vague_cr) > 0:
            vague_cr.append(copy.deepcopy(trial_vague_cr))
    # Summary & Save
    print("-"*40)
    print("Summary : ")
    print("Local num : ", len(local_cr))
    print("Global num : ", len(global_cr))
    print("Evade num : ", len(evade_cr))
    print("Suicide num : ", len(suicide_cr))
    print("Attack num : ", len(attack_cr))
    print("Vague num : ", len(vague_cr))
    state_cr = [global_cr, local_cr, evade_cr, suicide_cr, attack_cr, vague_cr]
    if "state_comparison" not in os.listdir("../common_data"):
        os.mkdir("../common_data/state_comparison")
    np.save("../common_data/state_comparison/path10-100trial_Omega_diff_state_agent_cr.npy", state_cr)

# ===============================================
def readSimpleTrialData(filename):
    '''
        Read data for MLE analysis.
        :param filename: Filename.
        '''
    # Read data and pre-processing
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index(drop=True)
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


# ===============================================

def pathLabelAnalysis(config):
    print("="*20, " Path Label Analysis ", "="*20)
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]]
    window = config["descriptive_window"]
    print(config["descriptive_filename"])
    trial_data = readTrialData(config["descriptive_filename"])
    # trial_data = [trial_data[i] for i in range(2)]
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)

    record = []
    agent_name = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]
    agent_index = [["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"].index(i) for i in agent_name]
    # Construct optimizer
    for trial_index, each in enumerate(trial_data):
        print("-"*15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("| {} | ".format(trial_index), " Trial name : ", trial_name)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        temp_contribution = np.zeros((len(window_index), len(agent_name)))
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 6, 4))
        # For each trial, estimate agent weights through sliding windows
        temp_estimated_label = []
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
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

            cur_weight = res.x[:-1]
            contribution = cur_weight * \
                           [scaleOfNumber(each) for each in
                            np.max(np.abs(temp_trial_Q[centering_index, :, agent_index, :]), axis=(1, 2))]
            temp_contribution[centering_index, :] = contribution / np.linalg.norm(contribution)
            temp_estimated_label.append(_estimationVagueLabeling(temp_contribution[centering_index, :], agent_name))
        # save to record
        record.append([
            trial_name,
            copy.deepcopy(temp_contribution),
            copy.deepcopy(temp_estimated_label),
            X.energizers.apply(lambda x: len(x) if not isinstance(x, float) else 0)[window:-window],
            X[["ifscared1", "ifscared2"]][window:-window]
        ])
    # Save data
    if "path_label_analysis" not in os.listdir("../common_data"):
        os.mkdir("../common_data/path_label_analysis")
    np.save(
        "../common_data/path_label_analysis/{}-record.npy".format(
            config["descriptive_filename"].split("/")[-1].split(".")[-2]
        ), record)


if __name__ == '__main__':
    # Configurations
    pd.options.mode.chained_assignment = None

    config = {
        # TODO: ===================================
        # TODO:       Always set to True
        # TODO: ===================================
        "need_intercept" : True,
        "maximum_try" : 5,
        # ==================================================================================
        #                       For Correlation Analysis and Multiple Label Analysis
        # Filename
        # "trial_data_filename": "../common_data/trial/{}_100_trial_data_Omega-with_Q-equal.pkl".format(type),
        "trial_data_filename": "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl",
        # The number of trials used for analysis
        "trial_num" : None,
        # Window size for correlation analysis
        "trial_window" : 3,
        "multi_agent_list" : ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"],
        # ==================================================================================
        "incremental_window" : 3,
        "incremental_data_filename" : "../common_data/trial/100_trial_data_Omega-with_Q-path10.pkl",
        "incremental_num_trial" : None,

        "single_trial_data_filename" : "../common_data/trial/test_suicide_trial_data_Omega-with_Q-path10.pkl",
        "single_trial_window" : 3,

        # ==================================================================================
        # "descriptive_filename" : "../common_data/trial/accidental_200_trial_data_Omega-with_Q-descriptive.pkl",
        "descriptive_filename" : "../common_data/trial/100_trial_data_Omega-with_Q-path10.pkl",
        "descriptive_window" : 3,
    }


    # ============ MOVING WINDOW =============

    # multiAgentAnalysis(config)

    # multiAgentAnalysis(config)

    # plotComparison()
    # plotPlannedHunting()
    # plotAccidentalHunting()
    # plotSuicide()
    # plotGlobal()
    # plotCentering()

    incrementalAnalysis(config)
    # decrementalAnalysis(config)
    # oneAgentAnalysis(config)
    # stageAnalysis(config)
    # stageCombineAnalysis(config)
    # specialCaseAnalysis(config)
    # specialCaseMovingAnalysis(config)
    #
    # closedGhostAnalysis(config)
    #
    # diffLabelAnalysis()


    # pathLabelAnalysis(config)

    # data = np.load("../common_data/equal_records.npy", allow_pickle=True)
    # print()
