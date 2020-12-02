'''
Description:
    Extract the transition data of local --> global and local --> evade.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    4 Nov. 2020
'''

import os
import pandas as pd
import numpy as np
import pickle
import copy

import sys

sys.path.append("../Utility_Tree_Analysis")
from TreeAnalysisUtils import readLocDistance

# Global variable
locs_df = readLocDistance("dij_distance_map.csv")
print("Finished reading distance file!")


def _PG(x):
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


def _rewardNum(x):
    num = 0
    if not isinstance(x.beans, float):
        num += len(x.beans)
    if not isinstance(x.energizers, float):
        num += len(x.energizers)
    if not isinstance(x.fruitPos, float):
        num += 1
    return num


def _PB(x):
    # Minimum distance between Pacman and beans
    PB = []
    if not isinstance(x.beans, float):
        for each in x.beans:
            if tuple(x.pacmanPos) != tuple(each):
                PB.append(locs_df[tuple(x.pacmanPos)][tuple(each)])
            else:
                PB.append(0)
    if not isinstance(x.energizers, float):
        for each in x.energizers:
            if tuple(x.pacmanPos) != tuple(each):
                PB.append(locs_df[tuple(x.pacmanPos)][tuple(each)])
            else:
                PB.append(0)
    if not isinstance(x.fruitPos, float):
        if tuple(x.pacmanPos) != tuple(x.fruitPos):
            PB.append(locs_df[tuple(x.pacmanPos)][tuple(x.fruitPos)])
        else:
            PB.append(0)
    # If no reward entities exist
    if len(PB) == 0:
        PB = [100]
    return np.min(PB)


def _extractAllData(trial_num = 20000):
    # Read data
    # data_filename = "partial_data_with_reward_label_cross.pkl"
    # with open(data_filename, "rb") as file:
    #     all_data_with_label = pickle.load(file)

    data_filename = "/home/qlyang/Documents/pacman/constants/all_data_new.pkl"
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    all_data_with_label = data["df_total"]

    all_data_with_label = all_data_with_label.sort_index()
    accident_index = np.concatenate(data["cons_list_accident"])
    plan_index = np.concatenate(data["cons_list_plan"])
    is_accidental = np.zeros((all_data_with_label.shape[0],))
    is_accidental[accident_index] = 1
    is_planned = np.zeros((all_data_with_label.shape[0],))
    is_planned[plan_index] = 1
    all_data_with_label["label_true_accidental_hunting"] = is_accidental
    all_data_with_label["label_true_planned_hunting"] = is_planned
    all_data_with_label = all_data_with_label.reset_index(drop=True)
    label_list = [
        "label_local_graze",
        "label_local_graze_noghost",
        "label_global_optimal",
        "label_global_notoptimal",
        "label_global",
        "label_evade",
        "label_suicide",
        "label_true_accidental_hunting",
        "label_true_planned_hunting",
        "label_global_ending"
    ]
    all_data_with_label[label_list] = all_data_with_label[label_list].fillna(0)
    print("All data shape : ", all_data_with_label.shape)
    trial_name_list = np.unique(all_data_with_label.file.values)
    print("Trial Num : ", len(trial_name_list))
    if len(trial_name_list) > trial_num:
        trial_name_list = trial_name_list[np.random.choice(len(trial_name_list), trial_num, replace=False)]
        print("Too much trials! Use only part of them. Trial Num : ", len(trial_name_list))
    print("Finished reading all data!")
    return all_data_with_label, trial_name_list


def _readLocDistance(filename):
    '''
    Read in the location distance.
    :param filename: File name.
    :return: A pandas.DataFrame denoting the dijkstra distance between every two locations of the map.
    '''
    locs_df = pd.read_csv(filename)[["pos1", "pos2", "dis"]]
    locs_df.pos1, locs_df.pos2= (
        locs_df.pos1.apply(eval),
        locs_df.pos2.apply(eval)
    )
    dict_locs_df = {}
    for each in locs_df.values:
        if each[0] not in dict_locs_df:
            dict_locs_df[each[0]] = {}
        dict_locs_df[each[0]][each[1]] = each[2]
    # correct the distance between two ends of the tunnel
    dict_locs_df[(1, 18)][(27, 18)] = 1
    return dict_locs_df


def _findDiscontinuePeriod(ls):
    '''
    # ls = [1,1,0,0,1,0,1,0,0,0,0]
    # print(_findDiscontinuePeriod(ls))
    #
    # ls = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    # print(_findDiscontinuePeriod(ls))
    # ls = [1,1,1]
    # print(_findDiscontinuePeriod(ls))
    '''

    ls = np.array(ls)
    continue_period = []
    if np.all(ls == 0):
        return [len(ls)]
    # if starts with 0s
    if ls[0] == 0:
        index = 0
        while ls[index] == 0:
            index += 1
        continue_period.append(index)
        ls = ls[index:]
    # end with 0s
    if ls[len(ls)-1] == 0:
        index = 0
        while ls[len(ls)- 1 - index] == 0:
            index += 1
        continue_period.append(index)
        ls = ls[:len(ls) - index]
    # the rest parts starting with 1
    absdiff = np.abs(np.diff(ls))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    continue_period.extend(list(ranges[:,1] - ranges[:,0]))
    return continue_period


def _findTransitionPoint(state1_indication, state2_indication, length):
    state1_indication = [int(each) for each in state1_indication.values]
    state2_indication = [int(each) for each in state2_indication.values]
    nums = len(state1_indication)
    state1_diff = np.diff(state1_indication)
    state2_diff = np.diff(state2_indication)
    transition_point = np.intersect1d(np.where(state1_diff == -1), np.where(state2_diff == 1))
    if len(transition_point) == 0:
        return []
    else:
        trajectories = []
        for index, each in enumerate(transition_point):
            cur = each
            # trajectory is not long nough
            if cur-length < 0 or  cur + 1 + length >= len(state1_indication):
                continue
            first_phase = state1_indication[cur-length:cur]
            second_phase = state2_indication[cur+1:cur + 1 + length]
            first_max_discontinue_period = _findDiscontinuePeriod(first_phase)
            second_max_discontinue_period = _findDiscontinuePeriod(second_phase)
            first_max_discontinue_period = [0] if len(first_max_discontinue_period) == 0 else first_max_discontinue_period
            second_max_discontinue_period = [0] if len(second_max_discontinue_period) == 0 else second_max_discontinue_period
            if max(first_max_discontinue_period) <= 3 and max(second_max_discontinue_period) <= 3:
                trajectories.append([
                    each - length, # starting index
                    each, # centering index
                    each + length + 1 # ending index
                ])
    return trajectories


def _findEvadeTransitionPoint(state1_indication, state2_indication, length, reward_num):
    '''
    Example:

    state1_indication = [1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    state2_indication = [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1]
    reward_num = [5, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1, 1]
    res = _findEvadeTransitionPoint(state1_indication, state2_indication, 3, reward_num)
    print(res)

    '''
    if isinstance(state1_indication, list):
        pass
    else:
        state1_indication = [int(each) for each in state1_indication.values]
        state2_indication = [int(each) for each in state2_indication.values]
    nums = len(state1_indication)
    state1_diff = np.diff(state1_indication)
    state2_diff = np.diff(state2_indication)
    transition_point = np.intersect1d(np.where(state1_diff == -1), np.where(state2_diff == 1))
    # Supplement transition point
    for index in range(len(state1_diff)-2):
        if (state1_diff[index] == -1 and state2_diff[index+1] == 1) or (state1_diff[index] == -1 and state2_diff[index+2] == 1):
            transition_point = np.append(transition_point, index)
    if len(transition_point) == 0:
        return []
    else:
        trajectories = []
        for index, each in enumerate(transition_point):
            cur = each
            # trajectory is not long enough
            if cur-length < 0 or  cur + 1 + length >= len(state1_indication):
                continue
            first_phase = state1_indication[cur-length:cur]
            second_phase = state2_indication[cur+1:cur + 1 + length]
            first_max_discontinue_period = _findDiscontinuePeriod(first_phase)
            second_max_discontinue_period = _findDiscontinuePeriod(second_phase)
            first_max_discontinue_period = [0] if len(first_max_discontinue_period) == 0 else first_max_discontinue_period
            second_max_discontinue_period = [0] if len(second_max_discontinue_period) == 0 else second_max_discontinue_period
            if max(first_max_discontinue_period) <= 0 and max(second_max_discontinue_period) <= 0:
                if np.all(np.array(reward_num[each-length:each]) == reward_num[each-length]):
                    trajectories.append([
                        each - length,  # starting index
                        each,  # centering index
                        each + length + 1  # ending index
                    ])
    return trajectories


def _findTransitionWOBreak(state1_indication, state2_indication, length):
    state1_indication = [int(each) for each in state1_indication.values]
    state2_indication = [int(each) for each in state2_indication.values]
    nums = len(state1_indication)
    state1_diff = np.diff(state1_indication)
    state2_diff = np.diff(state2_indication)
    transition_point = np.intersect1d(np.where(state1_diff == -1), np.where(state2_diff == 1))
    if len(transition_point) == 0:
        return []
    else:
        trajectories = []
        for index, each in enumerate(transition_point):
            cur = each
            # trajectory is not long nough
            if cur-length < 0 or  cur + 1 + length >= len(state1_indication):
                continue
            first_phase = state1_indication[cur-length:cur]
            second_phase = state2_indication[cur+1:cur + 1 + length]
            first_max_discontinue_period = _findDiscontinuePeriod(first_phase)
            second_max_discontinue_period = _findDiscontinuePeriod(second_phase)
            first_max_discontinue_period = [0] if len(first_max_discontinue_period) == 0 else first_max_discontinue_period
            second_max_discontinue_period = [0] if len(second_max_discontinue_period) == 0 else second_max_discontinue_period
            if max(first_max_discontinue_period) <= 0 and max(second_max_discontinue_period) <= 0:
                trajectories.append([
                    each - length, # starting index
                    each, # centering index
                    each + length + 1 # ending index
                ])
    return trajectories


def _local2Global(trial_data):
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
                           "label_global_ending"]].apply(
        lambda
            x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1 or x.label_global_ending == 1,
        axis=1
    )
    is_global = trial_data[["label_global_optimal", "label_global_notoptimal", "label_global"]].apply(
        lambda x: x.label_global_optimal == 1,
        axis = 1
    )
    length = 10
    print("Lcaol_to_global length : ", length)
    trajectory_point = _findTransitionPoint(is_local, is_global, length)
    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _local2Evade(trial_data):
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
                           "label_global_ending"]].apply(
        lambda
            x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1,
        axis=1
    )
    PG = trial_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PG(x),
        axis=1
    )
    trial_data["PG"] = PG
    is_evade = trial_data[["label_evade1","PG"]].apply(
        lambda x: x.label_evade1 == 1,  # and np.any(np.array(x.PG) <= 10),
        axis=1
    )

    # Only evade and only local
    is_pure_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
                           "label_global_ending", "label_evade", "label_evade1"]].apply(
        lambda
            x: (x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1) and
               (x.label_evade1 == 0),
        axis=1
    )
    is_pure_evade = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
                           "label_global_ending", "label_evade", "label_evade1", "PG"]].apply(
        lambda x: x.label_evade1 == 1 and
                  (x.label_local_graze == 0 and x.label_local_graze_noghost == 0 and x.label_true_accidental_hunting == 0),
        axis=1
    )


    trial_data = trial_data.drop(columns = ["PG"])
    length = 10
    print("Local_to_evade length : ", length)
    # trajectory_point = _findTransitionWOBreak(is_local, is_evade, length)
    trajectory_point = _findTransitionWOBreak(is_local, is_pure_evade, length) #TODO: pure local/evade
    # trajectory_point = _findEvadeTransitionPoint(is_pure_local, is_pure_evade, length)  # TODO: pure local/evade
    # trajectory_point = _findTransitionWOBreak(is_local, is_evade, length)
    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _evade2Local(trial_data):
    PB = trial_data[["pacmanPos", "beans", "energizers", "fruitPos"]].apply(
        lambda x: _PB(x),
        axis=1
    )
    is_evade_ending = PB.apply(lambda x: x <= 12)
    trial_data["label_evade_ending"] = is_evade_ending
    PG = trial_data[["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]].apply(
        lambda x: _PG(x),
        axis=1
    )
    reward_num = trial_data[["beans", "energizers", "fruitPos"]].apply(
        lambda x: _rewardNum(x),
        axis=1
    )
    trial_data["PG"] = PG
    trial_data["reward_num"] = reward_num

    is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
                           "label_global_ending", "label_evade_ending"]].apply(
        lambda
            x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1 or x.label_evade_ending == 1,
        axis=1
    )

    is_evade = trial_data[["label_evade", "label_evade1", "PG", "reward_num", "label_evade_ending"]].apply(
        lambda x: x.label_evade1 == 1 and x.label_evade_ending == 0,  # and np.any(np.array(x.PG) <= 10),
        axis=1
    )

    # # Only evade and only local
    # is_pure_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
    #                             "label_global_ending", "label_evade", "label_evade1"]].apply(
    #     lambda
    #         x: (
    #                        x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1) and
    #            (x.label_evade1 == 0 and x.label_evade2 == 0 and x.label_evade == 0),
    #     axis=1
    # )
    # is_pure_evade = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting",
    #                             "label_global_ending", "label_evade", "label_evade1", "PG"]].apply(
    #     lambda x: x.label_evade1 == 1 and
    #               (
    #                           x.label_local_graze == 0 and x.label_local_graze_noghost == 0 and x.label_true_accidental_hunting == 0),
    #     axis=1
    # )
    trial_data = trial_data.drop(columns = ["PG", "reward_num", "label_evade_ending"])

    length = 15
    print("Local_to_evade length : ", length)
    # trajectory_point = _findTransitionWOBreak(is_evade, is_local, length)
    trajectory_point = _findEvadeTransitionPoint(is_evade, is_local, length, reward_num)
    # trajectory_point = _findEvadeTransitionPoint(is_pure_evade, is_pure_local, length)
    # trajectory_point = _findTransitionWOBreak(is_evade, is_local, length)

    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _global2Local(trial_data):
    # label_global_ending 只对于 global -- local 有影响。
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting", "label_global_ending"]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1 or x.label_global_ending == 1,
        axis=1
    )
    is_global = trial_data[["label_global_optimal", "label_global_notoptimal", "label_global", "label_global_ending"]].apply(
        lambda x: x.label_global_optimal and x.label_global_ending != 1,
        axis=1
    )
    length = 10
    print("Global_to_local length : ", length)
    trajectory_point = _findTransitionPoint(is_global, is_local, length)
    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _local2Planned(trial_data):
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting", "label_global_ending"]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1,
        axis = 1
    )
    is_planned = trial_data[["label_true_planned_hunting"]].apply(
        lambda x: x.label_true_planned_hunting == 1,
        axis = 1
    )
    length = 15
    print("Local_to_planned length : ", length)
    trajectory_point = _findTransitionPoint(is_local, is_planned, length)
    # trajectory_point = _findTransitionWOBreak(is_local, is_planned, length)
    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _local2Suicide(trial_data):
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_global_ending", "label_true_accidental_hunting",]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1,
        axis=1
    )

    # is_local = trial_data[["label_local_graze", "label_local_graze_noghost", "label_true_accidental_hunting"]].apply(
    #     lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1 or x.label_true_accidental_hunting == 1,
    #     axis = 1
    # )
    is_suicide = trial_data[["label_suicide"]].apply(
        lambda x: x.label_suicide == 1,
        axis = 1
    )
    length = 5
    print("Local_to_suicide length : ", length)
    # trajectory_point = _findTransitionPoint(is_local, is_suicide, length)

    trajectory_point = _findTransitionWOBreak(is_local, is_suicide, length)

    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _extractTrialData(trial_data, transition_type):
    print("Transition types : ", transition_type)
    # Initialization
    trial_evade_to_local = None
    trial_local_to_evade = None
    trial_local_to_global = None
    trial_global_to_local = None
    trial_local_to_planned = None
    trial_local_to_suicide = None
    # For every transtion type
    if "local_to_global" in transition_type:
        trial_local_to_global = _local2Global(trial_data)
    if "global_to_local" in transition_type:
        trial_global_to_local = _global2Local(trial_data)
    if "local_to_planned" in transition_type:
        trial_local_to_planned = _local2Planned(trial_data)
    if "local_to_suicide" in transition_type:
        trial_local_to_suicide = _local2Suicide(trial_data)
    if "local_to_evade" in transition_type:
        trial_local_to_evade = _local2Evade(trial_data)
    if "evade_to_local" in transition_type:
        trial_evade_to_local = _evade2Local(trial_data)
    return trial_local_to_global, trial_local_to_evade, trial_global_to_local, trial_local_to_planned, trial_local_to_suicide, trial_evade_to_local


def extractTransitionData(transition_type, need_save = True):
    # Initialization
    local_to_global = []
    local_to_evade = []
    global_to_local= []
    local_to_planned = []
    local_to_suicide = []
    evade_to_local = []
    # Read data
    all_data, trial_name_list = _extractAllData()
    columns_values = np.append(all_data.columns.values, ["trajectory_index", "trajectory_shape"])
    print("Used Trial Num : ", len(trial_name_list))
    # Extract data for every trial
    local2global_trial_num = 0
    local2evade_trial_num = 0
    global2local_trial_num = 0
    local2planned_trial_num = 0
    local2suicide_trial_num = 0
    evade2local_trial_num = 0
    for index, trial in enumerate(trial_name_list):
        print("-"*25)
        print("{}-th : ".format(index + 1), trial)
        trial_data = all_data[all_data.file == trial]
        trial_data = trial_data.reset_index(drop=True)
        (trial_local_to_global, trial_local_to_evade, trial_global_to_local,
         trial_local_to_planned, trial_local_to_suicide, trial_evade_to_local) = _extractTrialData(trial_data, transition_type)
        if trial_local_to_global is not None:
            if local2global_trial_num > 1000: #TODO:
                pass
            else:
                local_to_global.extend(copy.deepcopy(trial_local_to_global))
                local2global_trial_num += 1
        if trial_local_to_evade is not None:
            if local2evade_trial_num > 1000: #TODO:
                pass
            else:
                local_to_evade.extend(copy.deepcopy(trial_local_to_evade))
                local2evade_trial_num += 1
        if trial_global_to_local is not None:
            if global2local_trial_num > 1000: #TODO:
                pass
            else:
                global_to_local.extend(copy.deepcopy(trial_global_to_local))
                global2local_trial_num += 1
        if trial_local_to_planned is not None:
            if local2planned_trial_num > 1000: #TODO:
                pass
            else:
                local_to_planned.extend(copy.deepcopy(trial_local_to_planned))
                local2planned_trial_num += 1
        if trial_local_to_suicide is not None:
            local_to_suicide.extend(copy.deepcopy(trial_local_to_suicide))
            local2suicide_trial_num += 1
        if trial_evade_to_local is not None:
            if evade2local_trial_num > 1000: #TODO:
                pass
            else:
                evade_to_local.extend(copy.deepcopy(trial_evade_to_local))
                evade2local_trial_num += 1
        print("-"*25)
    print("Finished extracting!")
    # Write data
    print("Local_to_global trial num : ", local2global_trial_num)
    print("Global_to_local trial num : ", global2local_trial_num)
    print("Local_to_evade trial num : ", local2evade_trial_num)
    print("Evade_to_local trial num : ", evade2local_trial_num)
    print("Local_to_planned trial num : ", local2planned_trial_num)
    print("Local_to_suicide trial num : ", local2suicide_trial_num)
    if need_save:
        if "transition" not in os.listdir():
            os.mkdir("transition")
        if len(local_to_global) > 0:
            local_to_global = pd.DataFrame(data=local_to_global, columns=columns_values)
            with open("transition/local_to_global.pkl", "wb") as file:
                pickle.dump(local_to_global, file)
            print("Finished writing local_to_global {}.".format(local_to_global.shape[0]))
        else:
            print("No local_to_global data!")
        if len(local_to_evade) > 0:
            local_to_evade = pd.DataFrame(data=local_to_evade, columns=columns_values)
            with open("transition/local_to_evade.pkl", "wb") as file:
                pickle.dump(local_to_evade, file)
            print("Finished writing local_to_evade {}.".format(local_to_evade.shape[0]))
        else:
            print("No local_to_evade data!")
        if len(global_to_local) > 0:
            global_to_local = pd.DataFrame(data=global_to_local, columns=columns_values)
            with open("transition/global_to_local.pkl", "wb") as file:
                pickle.dump(global_to_local, file)
            print("Finished writing global_to_local {}.".format(global_to_local.shape[0]))
        else:
            print("No global_to_local data!")
        if len(local_to_planned) > 0:
            local_to_planned = pd.DataFrame(data=local_to_planned, columns=columns_values)
            with open("transition/local_to_planned.pkl", "wb") as file:
                pickle.dump(local_to_planned, file)
            print("Finished writing local_to_planned {}.".format(local_to_planned.shape[0]))
        else:
            print("No local_to_planned data!")
        if len(local_to_suicide) > 0:
            local_to_suicide = pd.DataFrame(data=local_to_suicide, columns=columns_values)
            with open("transition/local_to_suicide.pkl", "wb") as file:
                pickle.dump(local_to_suicide, file)
            print("Finished writing local_to_suicide {}.".format(local_to_suicide.shape[0]))
        else:
            print("No local_to_suicide data!")
        if len(evade_to_local) > 0:
            evade_to_local = pd.DataFrame(data=evade_to_local, columns=columns_values)
            with open("transition/evade_to_local.pkl", "wb") as file:
                pickle.dump(evade_to_local, file)
            print("Finished writing evade_to_local {}.".format(evade_to_local.shape[0]))
        else:
            print("No evade_to_local data!")


if __name__ == '__main__':
    # Extract transition data
    transition_type = ["evade_to_local"]
    extractTransitionData(transition_type, need_save = True)