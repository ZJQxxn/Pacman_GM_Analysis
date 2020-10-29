'''
Description:
    Extract the transition data of local --> global and local --> evade.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    29 Oct. 2020
'''


import pandas as pd
import numpy as np
import pickle
import copy


def _extractAllData():
    # Configurations
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    # Read data
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    all_data_with_label = data["df_total"]
    all_data_with_label = all_data_with_label.sort_index()
    accident_index = np.concatenate(data["cons_list_accident"])
    group_list = _consecutiveLengh(accident_index)
    print("Accident group num : ", len(group_list))
    accident_group_list = []
    accident_group_length = []
    for each in group_list:
        if len(each) > 20:
            accident_group_list.append(each)
            accident_group_length.append(len(each))
    accident_index = [each[0] for each in accident_group_list]
    print("Accident group num : ", len(accident_group_list))
    accident_data = all_data_with_label.iloc[accident_index].reset_index(drop=True)
    normal_ghost_index = []
    for index in range(accident_data.shape[0]):
        ghost_status = accident_data.iloc[index][["ifscared1", "ifscared2"]].values
        ghost_loc = [tuple(each) for each in accident_data.iloc[index][["ghost1Pos", "ghost2Pos"]].values]
        if np.all(ghost_status < 3):
            if (14, 18) in ghost_loc or (14, 19) in ghost_loc:
                continue
            normal_ghost_index.append(index)
    accident_data = accident_data.iloc[normal_ghost_index].reset_index(drop=True)
    print("Alive step num : ", len(normal_ghost_index))
    # accident_data.label_planning = accident_data.label_planning.apply(lambda x: 2.0)

    plan_index = np.concatenate(data["cons_list_plan"])
    group_list = _consecutiveLengh(plan_index)
    print("Planned group num : ", len(group_list))
    planned_group_list = []
    planned_group_length = []
    for each in group_list:
        if len(each) > 20:
            planned_group_list.append(each)
            planned_group_length.append(len(each))
    plan_index = [each[0] for each in planned_group_list]
    print("Planned group num : ", len(planned_group_list))
    plan_data = all_data_with_label.iloc[plan_index].reset_index(drop=True)
    normal_ghost_index = []
    for index in range(plan_data.shape[0]):
        ghost_status = plan_data.iloc[index][["ifscared1", "ifscared2"]].values
        ghost_loc = [tuple(each) for each in plan_data.iloc[index][["ghost1Pos", "ghost2Pos"]].values]
        if np.all(ghost_status < 3):
            if (14, 18) in ghost_loc or (14, 19) in ghost_loc:
                continue
            normal_ghost_index.append(index)
    plan_data = plan_data.iloc[normal_ghost_index].reset_index(drop=True)
    print("Alive step num : ", len(normal_ghost_index))

    # accident_data.label_planning = accident_data.label_planning.apply(lambda x: 2.0)

    # accident_data = all_data_with_label.iloc[accident_index]
    print(all_data_with_label.shape)
    trial_name_list = np.unique(all_data_with_label.file.values)
    print("Trial Num : ", len(trial_name_list))
    if len(trial_name_list) > 2000:
        trial_name_list = trial_name_list[np.random.choice(len(trial_name_list), 2000, replace=False)]
        print("Too much trial! Use only part of them. Trial Num : ", len(trial_name_list))
    print("Finished reading all data!")
    return all_data_with_label, accident_data, plan_data, trial_name_list


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
        ghost_status = trial_data.iloc[global_start[0][0]][["ifscared1", "ifscared2"]].values
        ghost_loc = [tuple(each) for each in trial_data.iloc[global_start[0][0]][["ghost1Pos", "ghost2Pos"]].values]
        if (14, 18) in ghost_loc or (14, 19) in ghost_loc:
            return None
        if np.any(ghost_status >= 3):
            return None
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
        ghost_status = trial_data.iloc[evade_start[0][0]][["ifscared1", "ifscared2"]].values
        ghost_loc = [tuple(each) for each in trial_data.iloc[evade_start[0][0]][["ghost1Pos", "ghost2Pos"]].values]
        if (14, 18) in ghost_loc or (14, 19) in ghost_loc:
            return None
        if np.any(ghost_status >=3 ):
            return None
        return trial_data.iloc[evade_start[0][0]].values


def _findSuicide(trial_data):
    label_suicide = trial_data.label_suicide
    nan_index = np.where(np.isnan(label_suicide))
    label_suicide.iloc[nan_index] = 0
    suicide_start = np.where(label_suicide == 1)[0]
    group_list = _consecutiveLengh(suicide_start)
    group_length = [len(each) for each in group_list]
    print("Max suicide length : ", np.max(group_length))
    if len(group_list) == 0 or (np.max(group_length) < 5):
        print("No good suicide data.")
        return None
    else:
        ghost_status = trial_data.reset_index(drop=True).iloc[group_list[np.argmax(group_length)][0]][
            ["ifscared1", "ifscared2"]].values
        ghost_loc = [tuple(each) for each in trial_data.reset_index(drop=True).iloc[group_list[np.argmax(group_length)][0]][["ghost1Pos", "ghost2Pos"]].values]
        if (14, 18) in ghost_loc or (14, 19) in ghost_loc:
            return None
        if np.any(ghost_status >= 3):
            return None
        print(group_list[np.argmax(group_length)])
        return trial_data.reset_index().iloc[group_list[np.argmax(group_length)][0]].values[1:]


def _findPlanned(trial_data, loc_dis):
    label_planning = trial_data.label_planning
    nan_index = np.where(np.isnan(label_planning))
    label_planning.iloc[nan_index] = 0
    planning_index = np.where(label_planning == 1)[0]
    # PE = trial_data.apply(
    #     lambda x: [loc_dis[x.pacmanPos][each] if x.pacmanPos != each else 0 for each in x.energizers ]
    #     if not isinstance(x.energizers, float) else [0],
    #     axis = 1
    # )
    # PG = trial_data.apply(lambda x: [
    #     loc_dis[x.pacmanPos][x.ghost1Pos] if x.pacmanPos != x.ghost1Pos else 0,
    #     loc_dis[x.pacmanPos][x.ghost1Pos] if x.pacmanPos != x.ghost1Pos else 0
    # ], axis = 1)
    # is_normal =trial_data.apply(lambda x: [x.ifscared1, x.ifscared2], axis = 1)
    #
    # planning_index = []
    # for index in range(trial_data.shape[0]):
    #     if np.any(np.array(PE.iloc[index]) < 10) and \
    #             np.any(np.array(PE.iloc[index]) > 0) and \
    #             np.any(np.array(is_normal.iloc[index]) < 3) and \
    #             np.all(5 < np.array(PG.iloc[index])) and \
    #             np.all(np.array(PG.iloc[index]) < 20):
    #         planning_index.append(index)
    # if len(planning_index) == 0:
    #     return None
    # else:
    #     return trial_data.iloc[planning_index[0]].values
    group_list = _consecutiveLengh(planning_index)
    group_length = [len(each) for each in group_list]
    print("Max planned group length : ", np.max(group_length))
    if len(group_list) == 0 or (np.max(group_length) < 20):
        print("No good planned hunting data.")
        return None
    else:
        print(group_list[np.argmax(group_length)])
        return trial_data.reset_index().iloc[group_list[np.argmax(group_length)][0]].values[1:]


def _extractTrialData(trial_data, loc_dis):
    temp_global = _findGlobal(trial_data)
    temp_local = _findLocal(trial_data)
    temp_pessimistic = _findPessimistic(trial_data)
    temp_suicide = _findSuicide(trial_data)
    # temp_planned = _findPlanned(trial_data, loc_dis)
    return temp_global, temp_local, temp_pessimistic, temp_suicide


def _consecutiveLengh(num_list):
    return np.split(num_list, np.where(np.diff(num_list) != 1)[0]+1)


def extractStatus():
    # Initialization
    global_status = []
    local_status = []
    pessimistic_status = []
    suicide_status = []
    # planned_hunting_status= []
    # Read data
    all_data, accident_data, plan_data, trial_name_list = _extractAllData()
    loc_dis = _readLocDistance("dij_distance_map.csv")
    # trial_name_list  = np.unique(all_data.file.values)
    print("Used Trial Num : ", len(trial_name_list))
    # if len(trial_name_list) > 2000:
    #     trial_name_list = trial_name_list[np.random.choice(len(trial_name_list), 2000, replace=False)]
    #     print("Too much trial! Use only part of them. Trial Num : ", len(trial_name_list))
    for index, trial in enumerate(trial_name_list):
        print("-"*25)
        print("{}-th : ".format(index + 1), trial)
        trial_data = all_data[all_data.file == trial]
        temp_global, temp_local, temp_pessimistic, temp_suicide = _extractTrialData(trial_data, loc_dis)
        if temp_global is not None:
            global_status.append(copy.deepcopy(temp_global))
        if temp_local is not None:
            local_status.append(copy.deepcopy(temp_local))
        if temp_pessimistic is not None:
            pessimistic_status.append(copy.deepcopy(temp_pessimistic))
        if temp_suicide is not None:
            suicide_status.append(copy.deepcopy(temp_suicide))
        # if temp_planned is not None:
        #     planned_hunting_status.append(copy.deepcopy(temp_planned))
        print("-"*25)
    # Append accidentally hunting status
    print("Accidentally hunting shape : ", accident_data.shape)
    accident_index = np.random.choice(
        accident_data.shape[0],
        accident_data.shape[0] if accident_data.shape[0] < 2000 else 2000,
        replace = False
    )
    plan_index = np.random.choice(
        plan_data.shape[0],
        plan_data.shape[0] if plan_data.shape[0] < 2000 else 2000,
        replace=False
    )
    accident_data = accident_data.iloc[accident_index]
    plan_data = plan_data.iloc[plan_index]
    # for index in accident_index:
    #     planned_hunting_status.append(copy.deepcopy(accident_data.iloc[index].values[1:]))
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

    if len(pessimistic_status) > 0:
        pessimistic_status = pd.DataFrame(data=pessimistic_status, columns=trial_data.columns.values)
        with open("status/pessimistic_status.pkl", "wb") as file:
            pickle.dump(pessimistic_status, file)
        print("Finished writing pessimistic status {}.".format(pessimistic_status.shape[0]))
    else:
        print("No pessimistic status!")

    if len(suicide_status) > 0:
        suicide_status = pd.DataFrame(data=suicide_status, columns=trial_data.columns.values)
        with open("status/suicide_status.pkl", "wb") as file:
            pickle.dump(suicide_status, file)
        print("Finished writing suicide status {}.".format(suicide_status.shape[0]))
    else:
        print("No suicide status!")

    # if len(planned_hunting_status) > 0:
    #     planned_hunting_status = pd.DataFrame(data=planned_hunting_status, columns=trial_data.columns.values)
    #     with open("status/planned_hunting_status.pkl", "wb") as file:
    #         pickle.dump(planned_hunting_status, file)
    #     print("Finished writing planned hunting status {}.".format(planned_hunting_status.shape[0]))
    # planned_hunting_status = pd.concat([plan_data, accident_data], ignore_index=True)
    if "level_0" in plan_data:
        plan_data = plan_data.drop(columns = ["level_0"])
    with open("status/planned_hunting_status.pkl", "wb") as file:
        pickle.dump(plan_data, file)
    print("Finished writing planned hunting status {}.".format(plan_data.shape[0]))

    if "level_0" in accident_data:
        accident_data = accident_data.drop(columns = ["level_0"])
    with open("status/accidental_hunting_status.pkl", "wb") as file:
        pickle.dump(accident_data, file)
    print("Finished writing accidental hunting status {}.".format(accident_data.shape[0]))




if __name__ == '__main__':
    # print(_consecutiveLengh([1,2,3,5,6,7,9,10,11]))
    # print(_consecutiveLengh([]))

    extractStatus()


    # _extractAllData()
    # with open("/home/qlyang/Documents/pacman/constants/all_data.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     print(list(data.keys()))
    # all_data = data["df_total"]
    # print("Columns : ", all_data.columns.values)
    # print("Data Shape : ", all_data.shape)
    # trial_list = np.unique(all_data.file.values)
    # print("Trial Num : ", len(trial_list))
    # print("Trial Sample : ", trial_list[:3])
    #
    # accident = data["cons_list_accident"]
    # print("Accident length : ", len(accident))
    # print(accident.shape)
    # print(accident[:10])

    # with open("status/planned_hunting_status.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     wo = data.iloc[np.where(data.energizers.apply(lambda x: isinstance(x, float)).values == True)[0]][["file", "origin_index","energizers", "label_planning"]]
    #     w = data.iloc[np.where(data.energizers.apply(lambda x: isinstance(x, float)).values == False)[0]][["file", "origin_index","energizers", "label_planning"]]
    #     print()
    #     print()

    # with open("status/suicide_status.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     a = np.sum(data["ifscared1"].values >= 3)
    #     b = np.sum(data["ifscared2"].values >= 3)
    #
    #     print()

