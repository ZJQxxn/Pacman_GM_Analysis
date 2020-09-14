'''
Description:
    Extract game status with certain labels for simulation.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    9 Sep. 2020
'''


import pandas as pd
import numpy as np
import pickle
import copy


def _extractAllData():
    # Configurations
    # data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    # reward_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/df_total_with_reward.pkl"
    data_filename = "all_data_with_label.pkl"
    reward_data_filename = "df_total_with_reward.pkl"
    # Read data
    with open(data_filename, "rb") as file:
        all_data_with_label = pickle.load(file)
    # all_data_with_label = all_data["df_total"]
    print(all_data_with_label.shape)
    with open(reward_data_filename, "rb") as file:
        reward_data = pickle.load(file)
    all_data_with_label["Reward"] = reward_data.Reward
    all_data_with_label["fruitPos"] = reward_data.fruitPos
    print("Finished reading all data!")
    return all_data_with_label


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


def _findLocal(trial_data):
    beans = trial_data.iloc[0].beans
    beans = len(beans) if not isinstance(beans, float) else 0
    if beans > 32:
        return trial_data.iloc[0].values
    else:
        return None


def _findGlobal(trial_data):
    nan_index = np.where(np.isnan(trial_data.label_global))
    trial_data.label_global.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(trial_data.label_global_optimal))
    trial_data.label_global_optimal.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(trial_data.label_global_notoptimal))
    trial_data.label_global_notoptimal.iloc[nan_index] = 0
    is_global = trial_data.apply(
        lambda x: np.logical_or(np.logical_or(x.label_global, x.label_global_optimal), x.label_global_notoptimal),
        axis = 1
    )
    global_start = np.where(is_global == 1)
    if len(global_start[0]) == 0:
        return None
    else:
        return trial_data.iloc[global_start[0][0]].values


def _findPessimistic(trial_data):
    # is_scared1 = trial_data.ifscared1
    # is_scared2 = trial_data.ifscared2
    # is_pessimistic = np.logical_or(is_scared1 <= 2, is_scared2 <= 2)
    # pessimistic_start = np.where(is_pessimistic == 1)
    # if len(pessimistic_start[0]) == 0:
    #     return None
    # else:
    #     return trial_data.iloc[pessimistic_start[0][0]].values
    label_evade = trial_data.label_evade
    nan_index = np.where(np.isnan(label_evade))
    label_evade.iloc[nan_index] = 0
    evade_start = np.where(label_evade == 1)
    if len(evade_start[0]) == 0:
        return None
    else:
        return trial_data.iloc[evade_start[0][0]].values


def _findSuicide(trial_data):
    label_suicide = trial_data.label_suicide
    nan_index = np.where(np.isnan(label_suicide))
    label_suicide.iloc[nan_index] = 0
    suicide_start = np.where(label_suicide == 1)
    if len(suicide_start[0]) == 0:
        return None
    else:
        return trial_data.iloc[suicide_start[0][0]].values


def _findPlanned(trial_data, loc_dis):
    # label_planning = trial_data.label_planning
    # nan_index = np.where(np.isnan(label_planning))
    # label_planning.iloc[nan_index] = 0
    PE = trial_data.apply(
        lambda x: [loc_dis[x.pacmanPos][each] if x.pacmanPos != each else 0 for each in x.energizers ]
        if not isinstance(x.energizers, float) else [0],
        axis = 1
    )
    PG = trial_data.apply(lambda x: [
        loc_dis[x.pacmanPos][x.ghost1Pos] if x.pacmanPos != x.ghost1Pos else 0,
        loc_dis[x.pacmanPos][x.ghost1Pos] if x.pacmanPos != x.ghost1Pos else 0
    ], axis = 1)
    planning_index = []
    for index in range(trial_data.shape[0]):
        if np.any(np.array(PE.iloc[index]) < 10):
            planning_index.append(index)
    if len(planning_index) == 0:
        return None
    else:
        return trial_data.iloc[planning_index[0]].values


def _extractTrialData(trial_data, loc_dis):
    temp_global = _findGlobal(trial_data)
    temp_local = _findLocal(trial_data)
    temp_pessimistic = _findPessimistic(trial_data)
    temp_suicide = _findSuicide(trial_data)
    temp_planned = _findPlanned(trial_data, loc_dis)
    return temp_global, temp_local, temp_pessimistic, temp_suicide, temp_planned


def extractStatus():
    # Initialization
    global_status = []
    local_status = []
    pessimistic_satus = []
    suicide_status = []
    planned_hunting_status= []
    # Read data
    # with open("1-1-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl", "rb") as file:
    #     all_data = pickle.load(file)
    all_data = _extractAllData()
    loc_dis = _readLocDistance("dij_distance_map.csv")
    trial_name_list  = np.unique(all_data.file.values)
    print("Trial Num : ", len(trial_name_list))
    if len(trial_name_list) > 2000:
        trial_name_list = trial_name_list[np.random.choice(len(trial_name_list), 2000, replace=False)]
        print("Too much trial! Use only part of them. Trial Num : ", len(trial_name_list))
    for index, trial in enumerate(trial_name_list):
        print("-"*25)
        print("{}-th : ".format(index + 1), trial)
        trial_data = all_data[all_data.file == trial]
        temp_global, temp_local, temp_pessimistic, temp_suicide, temp_planned = _extractTrialData(trial_data, loc_dis)
        if temp_global is not None:
            global_status.append(copy.deepcopy(temp_global))
        if temp_local is not None:
            local_status.append(copy.deepcopy(temp_local))
        if temp_pessimistic is not None:
            pessimistic_satus.append(copy.deepcopy(temp_pessimistic))
        if temp_suicide is not None:
            suicide_status.append(copy.deepcopy(temp_suicide))
        if temp_planned is not None:
            planned_hunting_status.append(copy.deepcopy(temp_planned))
        print("-"*25)
    print("Finished extracting!")
    # Write data
    if len(global_status) > 0:
        global_status = pd.DataFrame(data=global_status, columns=trial_data.columns.values)
        with open("status/global_status.pkl", "wb") as file:
            pickle.dump(global_status, file)
        print("Finished writing global status {}.".format(global_status.shape[0]))
    else:
        print("No global status!")

    if len(local_status) > 0:
        local_status = pd.DataFrame(data=local_status, columns=trial_data.columns.values)
        with open("status/local_status.pkl", "wb") as file:
            pickle.dump(local_status, file)
        print("Finished writing local status {}.".format(local_status.shape[0]))
    else:
        print("No local status!")

    if len(pessimistic_satus) > 0:
        pessimistic_satus = pd.DataFrame(data=pessimistic_satus, columns=trial_data.columns.values)
        with open("status/pessimistic_status.pkl", "wb") as file:
            pickle.dump(pessimistic_satus, file)
        print("Finished writing pessimistic status {}.".format(pessimistic_satus.shape[0]))
    else:
        print("No pessimistic status!")

    if len(suicide_status) > 0:
        suicide_status = pd.DataFrame(data=suicide_status, columns=trial_data.columns.values)
        with open("status/suicide_status.pkl", "wb") as file:
            pickle.dump(suicide_status, file)
        print("Finished writing suicide status {}.".format(suicide_status.shape[0]))
    else:
        print("No suicide status!")

    if len(planned_hunting_status) > 0:
        planned_hunting_status = pd.DataFrame(data=planned_hunting_status, columns=trial_data.columns.values)
        with open("status/planned_hunting_status.pkl", "wb") as file:
            pickle.dump(planned_hunting_status, file)
        print("Finished writing planned hunting status {}.".format(planned_hunting_status.shape[0]))
    else:
        print("No planned hunting status!")




if __name__ == '__main__':
    extractStatus()
    # with open("df_total_with_reward.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     print()