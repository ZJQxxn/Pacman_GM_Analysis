'''
Description:
    Extract the transition data of local --> global and local --> evade.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    29 Oct. 2020
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
            # time period for the first state
            first_state_count = 0 # because of the diff == -1 position
            cur = each
            start_index = 0 if index == 0 else transition_point[index - 1]
            while cur >= start_index:
                if state1_indication[cur] == 1:
                    first_state_count += 1
                    cur -= 1
                    continue
                else:
                    break
            # time period for the second state
            second_state_count = 0  # because of the diff == 1 position
            cur = each + 1
            end_index = nums if (index == len(transition_point)-1) else transition_point[index + 1]
            while cur < end_index:
                if state2_indication[cur] == 1:
                    second_state_count += 1
                    cur += 1
                    continue
                else:
                    break
            if second_state_count >= length and first_state_count >= length:
                # [starting step, centering step, ending step]
                trajectories.append([
                    each - first_state_count + 1 if each - first_state_count + 1 >= 0 else 0,
                    each,
                    each + second_state_count if each + second_state_count < nums else nums - 1
                ])
    return trajectories


def _local2Global(trial_data):
    is_local = trial_data[["label_local_graze", "label_local_graze_noghost"]].apply(
        lambda x: x.label_local_graze == 1 or x.label_local_graze_noghost == 1,
        axis = 1
    )
    is_global = trial_data[["label_global_optimal", "label_global_notoptimal", "label_global"]].apply(
        lambda x: x.label_global_optimal == 1 or x.label_global_notoptimal == 1 or x.label_global == 1,
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
        lambda x: x.label_global_optimal == 1 or x.label_global_notoptimal == 1 or x.label_global == 1,
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
        with open("transition/relevant_agents/local_to_global.pkl", "wb") as file:
            pickle.dump(local_to_global, file)
        print("Local_to_global trial num : ", local2global_trial_num)
        print("Finished writing local_to_global {}.".format(local_to_global.shape[0]))
    else:
        print("No local_to_global data!")
    if len(local_to_evade) > 0:
        local_to_evade = pd.DataFrame(data = local_to_evade, columns = columns_values)
        with open("transition/relevant_agents/local_to_evade.pkl", "wb") as file:
            pickle.dump(local_to_evade, file)
        print("Local_to_evade trial num : ", local2evade_trial_num)
        print("Finished writing local_to_evade {}.".format(local_to_evade.shape[0]))
    else:
        print("No local_to_evade data!")
    if len(global_to_local) > 0:
        global_to_local = pd.DataFrame(data = global_to_local, columns = columns_values)
        with open("transition/relevant_agents/global_to_local.pkl", "wb") as file:
            pickle.dump(global_to_local, file)
        print("Global_to_local trial num : ", global2local_trial_num)
        print("Finished writing global_to_local {}.".format(global_to_local.shape[0]))
    else:
        print("No global_to_local data!")


if __name__ == '__main__':
    # Extract transition data
    extractTransitionData()

    # with open("transition/local_to_evade.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     data = data[["file", "trajectory_index", "label_local_graze", "label_local_graze_noghost", "label_evade"]]
    #     print()