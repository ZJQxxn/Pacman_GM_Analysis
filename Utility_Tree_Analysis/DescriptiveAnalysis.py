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
    offset = np.max(np.abs(np.concatenate(pess_Q)))
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
    if np.any(np.concatenate(planned_Q) < 0):
        offset = np.max(np.abs(np.concatenate(planned_Q))) # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_planned_Q = copy.deepcopy(planned_Q)
    for index in range(len(temp_planned_Q)):
        non_zero = np.where(temp_planned_Q[index] != 0)
        # if np.all(np.array(ghost_status[index]) >= 3) or energizer_num[index] == 0 or PE[index] > 15:
        if (np.all(np.array(ghost_status[index]) <= 3) and energizer_num[index] == 0) \
                or (np.all(np.array(ghost_status[index]) < 3) and PE[index] >= 15) \
                or np.all(np.array(ghost_status[index]) == 3) or np.all(np.array(PG[index]) >= 15) or _mixStatus(ghost_status[index], PG[index]):
            temp_planned_Q[index][non_zero] = 0.0
        else:
            temp_planned_Q[index][non_zero] = temp_planned_Q[index][non_zero] + offset
    return temp_planned_Q


def _suicideProcesing(suicide_Q, PR, RR, ghost_status, PG):
    # PR: minimum distance between Pacman position and reward entities
    # RR: minimum distance between reborn position and reward entities
    if np.any(np.concatenate(suicide_Q) < 0):
        offset = np.max(np.abs(np.concatenate(suicide_Q)))  # TODO: max absolute value of negative values
    else:
        offset = 0.0
    temp_suicide_Q = copy.deepcopy(suicide_Q)
    for index in range(len(temp_suicide_Q)):
        non_zero = np.where(temp_suicide_Q[index] != 0)
        # if np.all(np.array(ghost_status[index]) >= 3) or (PR[index] > 10 and RR[index] > 10):
        if np.all(np.array(ghost_status[index]) >= 3) or RR[index] > 10 or PR[index] <= 10 or not np.any(np.array(PG[index]) < 10):
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
    # # Pre-processng pessimistic Q
    # # TODO: check this
    # locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    # PG = all_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
    #     lambda x: _PG(x, locs_df),
    #     axis=1
    # )
    # PG_wo_dead = all_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
    #     lambda x: _PGWODead(x, locs_df),
    #     axis=1
    # )
    # PE = all_data[["pacmanPos", "energizers"]].apply(
    #     lambda x: _PE(x, locs_df),
    #     axis=1
    # )
    # ghost_status = all_data[["ifscared1", "ifscared2"]].apply(
    #     lambda x: _ghostStatus(x),
    #     axis=1
    # )
    # energizer_num = all_data[["energizers"]].apply(
    #     lambda x: _energizerNum(x),
    #     axis=1
    # )
    # PR = all_data[
    #     ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
    #     lambda x: _PR(x, locs_df),
    #     axis=1
    # )
    # RR = all_data[
    #     ["pacmanPos", "energizers", "beans", "fruitPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
    #     lambda x: _RR(x, locs_df),
    #     axis=1
    # )
    # print("Finished extracting features.")
    # # TODO: planned hunting and suicide Q value
    # all_data.pessimistic_Q = _pessimisticProcesing(all_data.pessimistic_Q, PG, ghost_status)
    # all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE, PG_wo_dead)
    # all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG)
    # print("Finished Q-value pre-processing.")
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
    np.save("../common_data/{}/descriptive_records.npy".format(dir_names), record)


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
        "trial_data_filename": "../common_data/trial/100_trial_data_Omega-with_Q-descriptive.pkl",
        # The number of trials used for analysis
        "trial_num" : None,
        # Window size for correlation analysis
        "trial_window" : 3,
        "multi_agent_list" : ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        # ==================================================================================
    }


    # ============ MOVING WINDOW =============

    # multiAgentAnalysis(config)
    plotComparison()

    # data = np.load("../common_data/global_local_pessimistic_suicide_planned_hunting/detailed_records.npy", allow_pickle=True)
    # print()