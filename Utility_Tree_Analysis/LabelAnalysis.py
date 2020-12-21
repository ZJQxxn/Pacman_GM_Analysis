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
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE,
                                                          PG_wo_dead)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG)
    print("Finished Q-value pre-processing.")
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
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG)
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


def readTrajectoryData(filename):
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
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE,
                                                          PG_wo_dead)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG)
    print("Finished Q-value pre-processing.")
    # Split into trajectories
    trial_data = []
    trial_name_list = np.unique(all_data.file.values)
    for each in trial_name_list:
        each_trial = all_data[all_data.file == each].reset_index(drop = True)
        trajectory_list = np.unique(each_trial.trajectory_index.values)
        for trajectory in trajectory_list:
            each_trajectory = each_trial[each_trial.trajectory_index == trajectory].reset_index(drop=True)
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
            trial_data.append(["{}-{}".format(each,trajectory), each_trajectory, true_prob])
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
    all_data.planned_hunting_Q = _plannedHuntingProcesing(all_data.planned_hunting_Q, ghost_status, energizer_num, PE,
                                                          PG_wo_dead)
    all_data.suicide_Q = _suicideProcesing(all_data.suicide_Q, PR, RR, ghost_status, PG)
    print("Finished Q-value pre-processing.")
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
    # laziness_coeff = 1.0
    # Configuration (for global agent)
    global_depth = 15
    ignore_depth = 10
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 34
    # Configuration (for local agent)
    local_depth = 10
    local_ghost_attractive_thr = 10
    local_fruit_attractive_thr = 10
    local_ghost_repulsive_thr = 10
    # Configuration (for optimistic agent)
    optimistic_depth = 5
    optimistic_ghost_attractive_thr = 5
    optimistic_fruit_attractive_thr = 5
    optimistic_ghost_repulsive_thr = 5
    # Configuration (for pessimistic agent)
    pessimistic_depth = 10
    pessimistic_ghost_attractive_thr = 10
    pessimistic_fruit_attractive_thr = 10
    pessimistic_ghost_repulsive_thr = 10
    # Configuration (fpr planne hunting agent)
    ghost_attractive_thr = 15
    energizer_attractive_thr = 15
    # Configuration (for suicide agent)
    suicide_depth = 10
    suicide_ghost_attractive_thr = 10
    suicide_fruit_attractive_thr = 10
    suicide_ghost_repulsive_thr = 10
    # Configuration (flast direction)
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
        # The tunnel
        if cur_pos == (0, 18):
            cur_pos = (1, 18)
        if cur_pos == (29, 18):
            cur_pos = (28, 18)
        adj_num = sum([isinstance(adjacent_data[cur_pos][each], tuple) for each in adjacent_data[cur_pos]])
        if adj_num > 2:
            laziness_coeff = 0.1
        else:
            laziness_coeff = 0.5
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
            ghost_attractive_thr=ghost_attractive_thr,
            energizer_attractive_thr = energizer_attractive_thr,
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
        # "../common_data/trial/5_trial_data.pkl",
        # "../common_data/transition/global_to_local.pkl",
        # "../common_data/transition/local_to_global.pkl",
        # "../common_data/transition/local_to_evade.pkl",
        # "../common_data/transition/evade_to_local.pkl",
        # "../common_data/transition/local_to_planned.pkl",
        # "../common_data/transition/local_to_suicide.pkl",
        # "../common_data/trial/500_trial_data.pkl",
    ]
    for filename in filename_list:
        print("-" * 50)
        print(filename)
        all_data = _readData(filename)
        print("Finished reading data.")
        print("Start estimating...")
        all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount)
        with open("{}/{}-with_Q.pkl".format(
                "../common_data/transition" if "transition" in filename.split("/") else "../common_data/trial",
                filename.split("/")[-1].split(".")[0]
        ), "wb") as file:
            pickle.dump(all_data, file)
        print("{}-with_Q.pkl saved!".format(filename.split("/")[-1].split(".")[0]))
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


def movingWindowAnalysis(config):
    print("=" * 20, " Moving Window ", "=" * 20)
    transition_type = config["trajectory_data_filename"].split("/")[-1].split(".")[0].split("-")[0]
    print(transition_type)
    print("Agent List :", config["agents"])
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    window = config["window"]
    # Construct optimizer
    params = [0 for _ in range(len(config["agents"]))]
    bounds = [[0, 10] for _ in range(len(config["agents"]))]
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


def threeAgentAnalysis(config):
    print("== Multi Label Aaalysis with Three Agents ==")
    print(config["trial_data_filename"])
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
        temp_weight = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        trial_estimated_label = []
        temp_contribution = []
        # For each trial, estimate agent weights through sliding windows
        agent_name = ["global", "local", "pessimistic"]
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
            contribution = temp_weight[centering_index, :-1] * [scaleOfNumber(each) for each in
                                np.max(np.abs(temp_trial_Q[centering_index, :, [0, 1, 2], :]), axis=(1, 2))]
            temp_contribution.append(copy.deepcopy(contribution))
            window_estimated_label.append(_estimationLabeling(contribution, agent_name))
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

    # Save data
    print()
    if "global_local_pessimistic" not in os.listdir("../common_data"):
        os.mkdir("../common_data/global_local_pessimistic")
    save_base = config["trial_data_filename"].split("/")[-1].split(".")[0]
    np.save("../common_data/global_local_pessimistic/{}-window{}-{}_intercept-multi_labels.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), all_estimated_label)
    np.save("../common_data/global_local_pessimistic/{}-window{}-{}_intercept-handcrafted_labels.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), handcrafted_labels)
    np.save("../common_data/global_local_pessimistic/{}-window{}-{}_intercept-matching_rate.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_matching_rate)
    np.save("../common_data/global_local_pessimistic/{}-window{}-{}_intercept-trial_weight.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_weight)
    np.save("../common_data/global_local_pessimistic/{}-window{}-{}_intercept-Q.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_Q)
    np.save("../common_data/global_local_pessimistic/{}-window{}-{}_intercept-contribution.npy".format(
        save_base, window, "w" if config["need_intercept"] else "wo"), trial_contribution)
    # Report
    # print("Average matching rate : ", np.mean(trial_matching_rate))
    # print("Min matching rate : ", np.min(trial_matching_rate))
    # print("Max matching rate : ", np.max(trial_matching_rate))


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
    if data_type is not None:
        np.save("../common_data/{}/{}_detailed_records.npy".format(dir_names, data_type), record)
    else:
        np.save("../common_data/{}/records.npy".format(dir_names), record)


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
        ["local", "pessimistic", "suicide", "planned_hunting"],
        ["global", "pessimistic", "suicide", "planned_hunting"],
        ["global", "local", "suicide", "planned_hunting"],
        ["global", "local", "pessimistic", "planned_hunting"],
        ["global", "local", "pessimistic", "suicide"],
        ["global", "local", "pessimistic", "suicide", "planned_hunting"]
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
    np.save("../common_data/decremental/{}trial-window{}-incremental_cr-{}_intercept.npy".format(
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
        ["pessimistic"],
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
    np.save("../common_data/one_agent/{}trial-window{}-incremental_cr-{}_intercept.npy".format(
        config["incremental_num_trial"], window, "w" if config["need_intercept"] else "wo"), all_cr)


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
    # params = [1 for _ in range(len(agent_name))]
    # bounds = [[0, 1] for _ in range(len(agent_name))]
    # if config["need_intercept"]:
    #     params.append(1)
    #     bounds.append([-1000, 1000])
    # cons = []  # construct the bounds in the form of constraints
    # for par in range(len(bounds)):
    #     l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
    #     u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
    #     cons.append(l)
    #     cons.append(u)
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
        print()

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
    np.save("../common_data/single_trial/records.npy", record)
    print()
    # np.save("../common_data/single_trial/estimated_labels.npy", all_estimated)
    # np.save("../common_data/single_trial/agent_weights.npy", all_weight)
    # np.save("../common_data/single_trial/agent_contributions.npy", all_Q)


def singleTrialAllFitting(config):
    locs_df = readLocDistance("../common_data/dij_distance_map.csv")
    # Read trial data
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
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
    # ,
    # trial_name_list = ["14-2-Patamon-10-Jul-2019-1.csv", "13-5-Patamon-21-Aug-2019-1.csv",
    #                    "13-3-Patamon-28-Jun-2019-1.csv", "14-1-Patamon-14-Jun-2019-1.csv", "12-2-Patamon-13-Aug-2019-1.csv"]

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

    # with open("../common_data/single_trial/trial_data.pkl", "wb") as file:
    #     pickle.dump(trial_data, file)

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

    random_index = np.arange(len(trial_data))
    np.random.shuffle(random_index)
    temp_trial_data = []
    for each in random_index:
        temp_trial_data.append(trial_data[each])
    trial_data = temp_trial_data
    for trial_index, each in enumerate(trial_data):
        temp_record = []
        print("-"*15)
        trial_name = each[0]
        temp_record.append(trial_name)
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("Trial name : ", trial_name)
        # TODO: !!!!!!
        # Preprocess suicide Q in the beginning of a trial
        cur_index = 0
        while ((14, 27) == X.pacmanPos[cur_index] or locs_df[(14, 27)][
            X.pacmanPos[cur_index]] < 10) and cur_index < trial_length:
            non_zero = np.where(X.suicide_Q[cur_index] != 0)
            X.suicide_Q[cur_index][non_zero] = 0.0
            cur_index += 1
            if cur_index >= trial_length:
                break
        #
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
                            np.max(np.abs(temp_trial_Q[centering_index, :, :, :]), axis=(0, 2))]
            temp_contribution[centering_index, :] = contribution
            window_estimated_label.append(_estimationLabeling(contribution, agent_name))
            trial_estimated_label.append(window_estimated_label)

        matched_num = 0
        not_nan_num = 0
        # for i in range(len(handcrafted_label)):
        #     if handcrafted_label[i] is not None:
        #         not_nan_num += 1
        #         if len(np.intersect1d(handcrafted_label[i], trial_estimated_label[i])) > 0:
        #             matched_num += 1
        # print(" Trial label matching rate : ", matched_num / not_nan_num if not_nan_num != 0 else "Nan trial")

        temp_record.append(copy.deepcopy(temp_weight))
        temp_record.append(copy.deepcopy(temp_contribution))
        temp_record.append(copy.deepcopy(trial_estimated_label))
        temp_record.append(copy.deepcopy(handcrafted_label))
        temp_record.append(copy.deepcopy(temp_trial_Q[:,:,[0, 1, 2], :]))
        record.append(copy.deepcopy(temp_record))

        all_weight_main.append(temp_weight)
        all_estimated.append(trial_estimated_label)


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

                seq = np.linspace(-0.1, 0.0, len(handcrafted_label[i])+1)
                for j, h in enumerate(handcrafted_label[i]):
                    plt.fill_between(x=[i, i + 1], y1=seq[j+1], y2=seq[j], color=agent_color[h])
                #
                # if len(handcrafted_label[i]) == 2:
                #     plt.fill_between(x=[i, i + 1], y1=0, y2=-0.05, color=agent_color[handcrafted_label[i][0]])
                #     plt.fill_between(x=[i, i + 1], y1=-0.05, y2=-0.1, color=agent_color[handcrafted_label[i][1]])
                # else:
                #     plt.fill_between(x=[i, i + 1], y1=0, y2=-0.1, color=agent_color[handcrafted_label[i][0]])

        # for pessimistic agent
        plt.ylabel("Normalized Agent Weight", fontsize=20)
        plt.xlim(0, temp_weight.shape[0] - 1)
        plt.xlabel("Time Step", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.ylim(-0.1, 1.1)
        plt.legend(loc="upper center", fontsize=15, ncol=len(agent_name))
        plt.show()
        print()


        all_Q.append(temp_trial_Q)
    # # Save data
    # np.save("../common_data/single_trial/records.npy", record)


def trajectoryTransitionFitting(config):
    print("="*15, " Trajectory Transition Analysis ", "="*15)
    print(config["trajectory_transition_data_filename"])
    all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in all_agent_list]
    agent_name = config["trajectory_transition_agents"]
    agent_index = [all_agent_list.index(each) for each in agent_name]
    window = config["trajectory_transition_window"]
    trial_data = readTrajectoryData(config["trajectory_transition_data_filename"])
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)

    trial_name_list = None
    record = []
    # trial_name_list = None
    if trial_name_list is not None and len(trial_name_list) > 0:
        temp_trial_Data = []
        for each in trial_data:
            if each[0] in trial_name_list:
                temp_trial_Data.append(each)
        trial_data = temp_trial_Data

    if config["trajectory_num"] is not None:
        trajectory_index = np.random.choice(np.arange(trial_num), config["trajectory_num"], replace = False)
        temp_trial_data = []
        for index in range(trial_num):
            if index in trajectory_index:
                temp_trial_data.append(trial_data[index])
        trial_data = temp_trial_data

    print("Num of used trials : ", len(trial_data))
    # with open("../common_data/single_trial/trial_data.pkl", "wb") as file:
    #     pickle.dump(trial_data, file)

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

    for trial_index, each in enumerate(trial_data):
        temp_record = []
        print("-"*15)
        trial_name = each[0]
        temp_record.append(trial_name)
        X = each[1]
        Y = each[2]
        indices = X.origin_index.values[window:-window]
        trial_length = X.shape[0]
        print("Trial name : {} ".format(trial_index), trial_name)
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
        temp_weight = np.zeros((len(window_index), len(agent_name) if not config["need_intercept"] else len(agent_name)+1))
        # temp_weight_rest = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))
        # temp_Q = []
        temp_contribution = np.zeros((len(window_index), len(agent_name)))
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
                            np.max(np.abs(temp_trial_Q[centering_index, :, agent_index, :]), axis=(1, 2))]
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
        temp_record.append(copy.deepcopy(temp_trial_Q[:,:,agent_index, :]))
        temp_record.append(copy.deepcopy(indices))
        record.append(copy.deepcopy(temp_record))

        all_weight_main.append(temp_weight)
        all_estimated.append(trial_estimated_label)

        estimated_label = [
            [
                _estimationLabeling(temp_contribution[index], agent_name)
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
        # print()

        all_Q.append(temp_trial_Q)
    # # Save data
    np.save("../common_data/single_trajectory/{}-transition_records.npy".format(
        config["trajectory_transition_data_filename"].split("/")[-1].split("-")[0]), record)
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


def controlledMLE(config):
    print("=" * 20, " Controlled Moving Window ", "=" * 20)
    agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    print("Agent List :", agent_name)
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    window = config["controlled_window"]
    # Load trajectory data
    temp_trial_data = readTrialData(config["trial_data_filename"])
    trial_num = len(temp_trial_data)
    print("Num of trials : ", trial_num)
    trial_index = range(trial_num)
    if config["controlled_trial_num"] is not None:
        if config["controlled_trial_num"] < trial_num:
            trial_index = np.random.choice(range(trial_num), config["controlled_trial_num"], replace=False)
    trial_data = [temp_trial_data[each] for each in trial_index]
    trial_cr = []
    # agent_name = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
    agent_index = [["global", "local", "pessimistic", "suicide", "planned_hunting"].index(each) for each in agent_name]

    for trial_index, each in enumerate(trial_data):
        print("-" * 15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        random_index = np.arange(X.shape[0])
        np.random.shuffle(random_index)
        random_X = X.iloc[random_index]
        random_Y = Y.iloc[random_index]
        trial_length = X.shape[0]
        print(trial_index, " : ", trial_name)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_cr = []
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = random_X[centering_point - window:centering_point + window + 1]
            sub_X.iloc[window] = X.iloc[centering_point]
            sub_Y = random_Y[centering_point - window:centering_point + window + 1]
            sub_Y.iloc[window] = Y.iloc[centering_point]
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
            temp_cr.append(correct_rate)
        trial_cr.append(temp_cr)
    print("Average correct rate for every trial : ", [np.nanmean(each) for each in trial_cr])
    print("Average correct rate for all : ", np.nanmean([np.nanmean(each) for each in trial_cr]))
    if "controlled" not in os.listdir("../common_data"):
        os.mkdir("../common_data/controlled")
    np.save("../common_data/controlled/trial_cr.npy", trial_cr)


def _extractDiffState():
    print("=" * 40)
    print("Extracting Different State Analysis")
    print("=" * 40)
    local_X = []
    local_Y = []
    # Read global data
    local2global_data = readTransitionData("../common_data/transition/local_to_global-with_Q.pkl")
    global2local_data = readTransitionData("../common_data/transition/global_to_local-with_Q.pkl")
    global_X = []
    global_Y = []
    for each in local2global_data:
        length = each[1].shape[0]
        global_X.append(each[1].iloc[length // 2:])
        global_Y.append(each[2][length // 2:])
        local_X.append(each[1].iloc[:length // 2])
        local_Y.append(each[2][:length // 2])
    for each in global2local_data:
        length = each[1].shape[0]
        global_X.append(each[1].iloc[:length // 2])
        global_Y.append(each[2][:length // 2])
        local_X.append(each[1].iloc[length // 2:])
        local_Y.append(each[2][length // 2:])
    global_X = pd.concat(global_X).reset_index(drop=True)
    global_Y = pd.concat(global_Y).reset_index(drop=True)
    print("Finished global.")
    # Read evade data
    local2evade_data = readTransitionData("../common_data/transition/local_to_evade-with_Q.pkl")
    evade2local_data = readTransitionData("../common_data/transition/evade_to_local-with_Q.pkl")
    evade_X = []
    evade_Y = []
    for each in local2evade_data:
        length = each[1].shape[0]
        evade_X.append(each[1].iloc[length // 2:])
        evade_Y.append(each[2][length // 2:])
        local_X.append(each[1].iloc[:length // 2])
        local_Y.append(each[2][:length // 2])
    for each in evade2local_data:
        length = each[1].shape[0]
        evade_X.append(each[1].iloc[:length // 2])
        evade_Y.append(each[2][:length // 2])
        local_X.append(each[1].iloc[length // 2:])
        local_Y.append(each[2][length // 2:])
    evade_X = pd.concat(evade_X).reset_index(drop=True)
    evade_Y = pd.concat(evade_Y).reset_index(drop=True)
    print("Finished evade.")
    # Read attack data
    local2attack_data = readTransitionData("../common_data/transition/local_to_planned-with_Q.pkl")
    attack_X = []
    attack_Y = []
    for each in local2attack_data:
        length = each[1].shape[0]
        attack_X.append(each[1].iloc[length // 2:])
        attack_Y.append(each[2][length // 2:])
        local_X.append(each[1].iloc[:length // 2])
        local_Y.append(each[2][:length // 2])
    attack_X = pd.concat(attack_X).reset_index(drop=True)
    attack_Y = pd.concat(attack_Y).reset_index(drop=True)
    print("Finished attack.")
    # Read suicide data
    local2suicide_data = readTransitionData("../common_data/transition/local_to_suicide-with_Q.pkl")
    suicide_X = []
    suicide_Y = []
    for each in local2suicide_data:
        length = each[1].shape[0]
        suicide_X.append(each[1].iloc[length // 2:])
        suicide_Y.append(each[2][length // 2:])
        local_X.append(each[1].iloc[:length // 2])
        local_Y.append(each[2][:length // 2])
    suicide_X = pd.concat(suicide_X).reset_index(drop=True)
    suicide_Y = pd.concat(suicide_Y).reset_index(drop=True)
    print("Finished suicide.")
    # Read local data
    local_X = pd.concat(local_X).reset_index(drop=True)
    local_Y = pd.concat(local_Y).reset_index(drop=True)
    # Read vague data
    trial_data = readTrialData("../common_data/trial/1000_trial_data_Omega-with_Q.pkl")
    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]
    vague_X = []
    vague_Y = []
    for each in trial_data:
        length = each[1].shape[0]
        handcrafted_label = [_handcraftLabeling(each[1][label_list].iloc[index]) for index in range(length)]
        is_vague = [len(label) > 1 if label is not None and not isinstance(label, float) else False for label in
                    handcrafted_label]
        is_vague = np.where(np.array(is_vague) == 1)[0]
        if len(is_vague) > 0:
            vague_X.append(each[1].iloc[is_vague])
            vague_Y.append(each[2][is_vague])
    if len(vague_X) > 0:
        vague_X = pd.concat(vague_X).reset_index(drop=True)
        vague_Y = pd.concat(vague_Y).reset_index(drop=True)
    else:
        vague_X = None
        vague_Y = None
    print("Finished vague.")
    # Save data
    print("-"*40)
    if "state_comparison" not in os.listdir("../common_data"):
        os.mkdir("../common_data/state_comparison")
    np.save("../common_data/state_comparison/local_data.npy", (local_X, local_Y))
    np.save("../common_data/state_comparison/global_data.npy", (global_X, global_Y))
    np.save("../common_data/state_comparison/evade_data.npy", (evade_X, evade_Y))
    np.save("../common_data/state_comparison/attack_data.npy", (attack_X, attack_Y))
    np.save("../common_data/state_comparison/suicide_data.npy", (suicide_X, suicide_Y))
    np.save("../common_data/state_comparison/vague_data.npy", (vague_X, vague_Y))
    print("Finished saving!")
    print("Local shape : ", local_X.shape)
    print("Global shape : ", global_X.shape)
    print("Evade shape : ", evade_X.shape)
    print("Attack shape : ", attack_X.shape)
    print("Suicide shape : ", suicide_X.shape)
    print("Vague shape : ", vague_X.shape)


def diffStateAnalysis(config):
    print("="*40)
    print("Different State Analysis")
    print("=" * 40)
    local_X, local_Y = np.load("../common_data/state_comparison/local_data.npy", allow_pickle=True)
    global_X, global_Y = np.load("../common_data/state_comparison/global_data.npy", allow_pickle=True)
    evade_X, evade_Y = np.load("../common_data/state_comparison/evade_data.npy", allow_pickle=True)
    attack_X, attack_Y = np.load("../common_data/state_comparison/attack_data.npy", allow_pickle=True)
    suicide_X, suicide_Y = np.load("../common_data/state_comparison/suicide_data.npy", allow_pickle=True)
    vague_X, vague_Y = np.load("../common_data/state_comparison/vague_data.npy", allow_pickle=True)
    print("Finished reading data.")
    # Analysis
    state_cr = {
        "global": None,
        "local":None,
        "pessimistic": None,
        "planned_hunting":None,
        "suicide":None,
        "vague":None
    }
    state_data = {
        "vague": (vague_X, vague_Y),
        "global":(global_X, global_Y),
        "local":(local_X, local_Y),
        "pessimistic": (evade_X, evade_Y),
        "planned_hunting":(attack_X, attack_Y),
        "suicide":(suicide_X, suicide_Y),
    }
    for state in state_data:
        print("="*15, " {} Analysis".format(state), "="*15)
        X, Y = state_data[state]
        if X is None:
            print("No data for state \"{}\"".format(state))
            state_cr[state] = None
            continue
        temp_cr = []
        for agent_name in [["local"], ["local", state], ["global", "local", "pessimistic", "suicide", "planned_hunting"]]:
            if agent_name == ["local", "vague"]:
                temp_cr.append(np.nan)
                continue
            print("Agent name : ", agent_name)
            # agents_list = ["{}_Q".format(each) for each in agent_name]
            print("Data shape : ", X.shape)
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
            func = lambda params: negativeLikelihood(
                params,
                X,
                Y,
                agent_name,
                return_trajectory = False,
                need_intercept = True
            )
            is_success = False
            retry_num = 0
            while not is_success and retry_num < 5:
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
                X,
                Y,
                agent_name,
                return_trajectory=True,
                need_intercept=True
            )
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            print("Weight : ", res.x)
            print("Correct rate : ", correct_rate)
            temp_cr.append(correct_rate)
        state_cr[state] = temp_cr
    print("="*40)
    print("Summary : ")
    print(state_cr)
    if "state_comparison" not in os.listdir("../common_data"):
        os.mkdir("../common_data/state_comparison")
    np.save("../common_data/state_comparison/state_cr.npy", state_cr)





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
        elif "local_to_evade" == type or "evade_to_local" == type:
            agents = ["local", "pessimistic"]
        elif "local_to_planned" == type or "local_to_accidental" == type or "graze_to_hunt" == type:
            agents = ["local", "planned_hunting"]
        elif "local_to_suicide" == type:
            agents = ["local", "suicide"]
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
        "trial_data_filename": "../common_data/trial/suicide_100_trial_data_Omega-with_Q.pkl",
        # The number of trials used for analysis
        "trial_num" : None,
        # Window size for correlation analysis
        "trial_window" : 3,
        "correlation_agents": ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        "multi_agent_list" : ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        # ==================================================================================

        # ==================================================================================
        #                       For Single Trial Analysis
        # Filename
        # "single_trial_data_filename": "../common_data/trial/15-6-Patamon-04-Jul-2019-4-new_suicide-with_Q.pkl",
        "single_trial_data_filename": "../common_data/trial/100_trial_data_Omega-with_Q.pkl",
        # Window size for correlation analysis
        "single_trial_window": 3,
        "single_trial_agents": ["global", "local", "pessimistic", "suicide", "planned_hunting"],
        # ==================================================================================

        # ==================================================================================
        #                       For Trajectory Transtion Analysis
        # Filename
        # "single_trial_data_filename": "../common_data/trial/global15-local10-100_trial_data_new-with_Q.pkl",
        "trajectory_transition_data_filename": "../common_data/transition/local_to_planned-with_Q.pkl",
        # Window size for correlation analysis
        "trajectory_transition_window": 1,
        "trajectory_num": None,
        "trajectory_transition_agents": ["local", "planned_hunting"],
        # ==================================================================================

        # ==================================================================================
        #                       For Incremental Analysis
        # Filename
        "incremental_data_filename": "../common_data/trial/100_trial_data_Omega-with_Q.pkl",
        # Window size for correlation analysis
        "incremental_window": 3,
        "incremental_num_trial" : None,
        # ==================================================================================

        # ==================================================================================
        #                       For Simple MLE Analysis
        # Filename
        # "MLE_data_filename": "../common_data/trial/100_trial_data_new-one_ghost-with_Q.pkl",
        "MLE_data_filename": "../common_data/trial/100_trial_data_Omega-with_Q.pkl",
        # Window size for MLE analysis
        "MLE_num_trial": None,
        "MLE_agents": ["global", "local", "pessimistic"],
        # ==================================================================================

        # ==================================================================================
        #                       For Controlled Analysis
        # Filename
        "controlled_data_filename": "../common_data/trial/100_trial_data_all_new-with_Q.pkl",
        # The window size
        "controlled_window": 1,
        # Maximum try of estimation, in case the optimization will fail
        "controlled_maximum_try": 5,
        "controlled_trial_num": 50,
        # ==================================================================================

        # ==================================================================================
        #                       For Different State Analysis
        # Filename
        # "diff_state": ["global", "local", "evade", "attack", "suicide", "vague"],
        # ==================================================================================


        # ==================================================================================
        #                       For Experimental Results Visualization
        # this multi-label data is the true estimated label
        # "estimated_label_filename" : "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-multi_labels.npy",
        # "handcrafted_label_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-handcrafted_labels.npy",
        # # "trial_cr_filename": "../common_data/multi_label/500_trial_data-with_Q-window3-w_intercept-trial_cr.npy",
        # "trial_weight_main_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-trial_weight_main.npy",
        # "trial_weight_rest_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-trial_weight_rest.npy",
        # "trial_Q_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-Q.npy",
        # "trial_matching_rate_filename": "../common_data/multi_label/global15-local10-100_trial_data_new-with_Q-window3-w_intercept-matching_rate.npy",

        "estimated_label_filename": "../common_data/global_local_pessimistic/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-multi_labels.npy",
        "handcrafted_label_filename": "../common_data/global_local_pessimistic/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-handcrafted_labels.npy",
        "trial_weight_filename": "../common_data/global_local_pessimistic/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-trial_weight.npy",
        "trial_Q_filename": "../common_data/global_local_pessimistic/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-Q.npy",
        "trial_matching_rate_filename": "../common_data/global_local_pessimistic/100_trial_data_new-one_ghost-with_Q-window3-w_intercept-matching_rate.npy",

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

        "bean_vs_cr_filename" : "../common_data/incremental/500trial-window3-incremental_cr-w_intercept.npy",
        "bin_size" : 10,
    }


    # ============ MOVING WINDOW =============
    # movingWindowAnalysis(config)

    # singleTrialThreeFitting(config) # global, local, pessimistic
    # singleTrialAllFitting(config)

    # simpleMLE(config)

    # threeAgentAnalysis(config)

    # incrementalAnalysis(config)
    # decrementalAnalysis(config)
    # oneAgentAnalysis(config)

    # multiAgentAnalysis(config)

    # _extractDiffState()
    # diffStateAnalysis(config)

    # simpleMLE(config)
    # controlledMLE(config)

    # data = np.load("../common_data/controlled/trial_cr.npy", allow_pickle=True)
    # print()
    # trajectoryTransitionFitting(config)

    # ============ VISUALIZATION =============
    # plotThreeAgentMatching(config)

    # plotWeightVariation(config, plot_sem = True, contribution = True, need_normalization = True, normalizing_type="sum") # step / sum / all

    # plotBeanNumVSCr(config)