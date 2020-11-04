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


def _extractAllData(trial_num = 20000):
    # Read data
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    all_data_with_label = data["df_total"]

    # data_filename = "partial_data_with_reward_label_cross.pkl"
    # with open(data_filename, "rb") as file:
    #     all_data_with_label = pickle.load(file)

    all_data_with_label = all_data_with_label.sort_index()
    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_optimal", "label_global_notoptimal", "label_global", "label_evade"]
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


def _local2Global(trial_data):
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost"]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1,
        axis = 1
    )
    is_global = trial_data[["label_global_optimal", "label_global_notoptimal", "label_global"]].apply(
        lambda x: x.label_global_optimal == 1,
        axis = 1
    )
    trajectory_point = _findTransitionPoint(is_local, is_global, 15)
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
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost"]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1,
        axis=1
    )
    is_evade = trial_data[["label_evade"]].apply(
        lambda x: x.label_evade == 1,
        axis=1
    )
    trajectory_point = _findTransitionPoint(is_local, is_evade, 5)
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
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost"]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1,
        axis=1
    )
    is_global = trial_data[["label_global_optimal", "label_global_notoptimal", "label_global"]].apply(
        lambda x: x.label_global_optimal,
        axis=1
    )
    trajectory_point = _findTransitionPoint(is_global, is_local, 15)
    if len(trajectory_point) == 0:
        return None
    else:
        trajectory_data = []
        for trajectory_index, each in enumerate(trajectory_point):
            for index in range(each[0], each[2] + 1):
                each_step = np.append(trial_data.iloc[index].values, [trajectory_index, each])
                trajectory_data.append(each_step)
    return trajectory_data


def _extractTrialData(trial_data):
    trial_local_to_evade = _local2Evade(trial_data)
    trial_local_to_global = _local2Global(trial_data)
    trial_global_to_local = _global2Local(trial_data)
    return trial_local_to_global, trial_local_to_evade, trial_global_to_local


def extractTransitionData():
    # Initialization
    local_to_global = []
    local_to_evade = []
    global_to_local= []
    # Read data
    all_data, trial_name_list = _extractAllData()
    columns_values = np.append(all_data.columns.values, ["trajectory_index", "trajectory_shape"])
    print("Used Trial Num : ", len(trial_name_list))
    # Extract data for every trial
    local2global_trial_num = 0
    local2evade_trial_num = 0
    global2local_trial_num = 0
    for index, trial in enumerate(trial_name_list):
        print("-"*25)
        print("{}-th : ".format(index + 1), trial)
        trial_data = all_data[all_data.file == trial]
        trial_data = trial_data.reset_index(drop=True)
        trial_local_to_global, trial_local_to_evade, trial_global_to_local = _extractTrialData(trial_data)
        if trial_local_to_global is not None:
            local_to_global.extend(copy.deepcopy(trial_local_to_global))
            local2global_trial_num += 1
        if trial_local_to_evade is not None:
            local_to_evade.extend(copy.deepcopy(trial_local_to_evade))
            local2evade_trial_num += 1
        if trial_global_to_local is not None:
            global_to_local.extend(copy.deepcopy(trial_global_to_local))
            global2local_trial_num += 1
        print("-"*25)
    print("Finished extracting!")
    # Write data
    if "transition" not in os.listdir():
        os.mkdir("transition")
    if len(local_to_global) > 0:
        local_to_global = pd.DataFrame(data = local_to_global, columns = columns_values)
        with open("transition/local_to_global.pkl", "wb") as file:
            pickle.dump(local_to_global, file)
        print("Local_to_global trial num : ", local2global_trial_num)
        print("Finished writing local_to_global {}.".format(local_to_global.shape[0]))
    else:
        print("No local_to_global data!")
    if len(local_to_evade) > 0:
        local_to_evade = pd.DataFrame(data = local_to_evade, columns = columns_values)
        with open("transition/local_to_evade.pkl", "wb") as file:
            pickle.dump(local_to_evade, file)
        print("Local_to_evade trial num : ", local2evade_trial_num)
        print("Finished writing local_to_evade {}.".format(local_to_evade.shape[0]))
    else:
        print("No local_to_evade data!")
    if len(global_to_local) > 0:
        global_to_local = pd.DataFrame(data = global_to_local, columns = columns_values)
        with open("transition/global_to_local.pkl", "wb") as file:
            pickle.dump(global_to_local, file)
        print("Global_to_local trial num : ", global2local_trial_num)
        print("Finished writing global_to_local {}.".format(global_to_local.shape[0]))
    else:
        print("No global_to_local data!")


if __name__ == '__main__':
    #
    # ls = [1,1,0,0,1,0,1,0,0,0,0]
    # print(_findDiscontinuePeriod(ls))
    #
    # ls = [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]
    # print(_findDiscontinuePeriod(ls))
    # ls = [1,1,1]
    # print(_findDiscontinuePeriod(ls))


    # Extract transition data
    extractTransitionData()

    # with open("transition/local_to_evade.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     data = data[["file", "trajectory_index", "label_local_graze", "label_local_graze_noghost", "label_evade"]]
    #     print()