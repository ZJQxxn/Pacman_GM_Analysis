'''
Description:
    Extract data from all_data file.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
 
Date:
    23 July 2020 
'''

import pickle
import pandas as pd
import numpy as np
import sys
import copy

sys.path.append("./")
from LabelingData import _isGlobal, _isLocal

sys.path.append("../Utility_Tree_Analysis/")
from TreeAnalysisUtils import readAdjacentMap


def _ifAtCross(adjacent_data, pacmanPos):
    return pacmanPos.apply(
        lambda x: (
            False if x not in adjacent_data else
            np.sum(
                [1 if not isinstance(each, float) else 0
                 for each in list(adjacent_data[x].values())]
            ) > 2

        )
    )


def _splitTrial(trial_name_list):
    # 第几次局游戏-第几条命-哪个monkey-日-月-年-重复标识码
    trial_dict = {}
    for each in trial_name_list:
        each = each.strip(".csv") # strip .csv
        each = each.split("-")
        life_num = each[1]
        trial_name = "-".join([each[0], "{}", *each[2:]])
        if trial_name not in trial_dict:
            trial_dict[trial_name] = []
        trial_dict[trial_name].append(int(life_num))

    first_life_list = [each.format(min(trial_dict[each])) + ".csv" for each in trial_dict.keys()]
    last_life_list = [each.format(max(trial_dict[each])) + ".csv" for each in trial_dict.keys()]
    before_last_life_list = []
    for each in trial_dict.keys():
        if len(trial_dict[each]) > 2:
            before_last_life_list.append(each.format(sorted(trial_dict[each])[-2]) + ".csv")
    return first_life_list, last_life_list, before_last_life_list


def extractLifeGame():
    # Configurations
    print("="*40)
    print("EXTRACT THE FIRST LIFE DATA")
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    adjacent_data = readAdjacentMap("/home/qlyang/jiaqi/Pacman-Analysis/Utility_Tree_Analysis/extracted_data/adjacent_map.csv")
    # Read data
    with open(data_filename, "rb") as file:
        all_data = pickle.load(file)
    all_data_with_label = all_data["df_total"].sort_index()
    trial_name_list = np.unique(all_data_with_label.file.values)
    first_life_list, last_life_list, before_last_life_list = _splitTrial(trial_name_list)
    print("Finished processing.")
    print("-"*40)
    print("Length of the first life game : ", len(first_life_list))
    print(first_life_list[:5])
    if len(first_life_list) > 100:
        first_life_list = np.random.choice(first_life_list, 100)
    is_first = all_data_with_label.file.apply(lambda x: x in first_life_list)
    first_game_index = np.where(is_first == 1)
    first_game_data = all_data_with_label.iloc[first_game_index]
    first_game_data.at_cross = _ifAtCross(adjacent_data, first_game_data.pacmanPos)
    print("First game samples number : ", first_game_data.shape)
    print("-" * 40)
    print("Length of the last life game : ", len(last_life_list))
    print(last_life_list[:5])
    if len(last_life_list) > 100:
        last_life_list = np.random.choice(last_life_list, 100)
    is_last = all_data_with_label.file.apply(lambda x: x in last_life_list)
    last_game_index = np.where(is_last == 1)
    last_game_data = all_data_with_label.iloc[last_game_index]
    last_game_data.at_cross = _ifAtCross(adjacent_data, last_game_data.pacmanPos)
    print("Last game samples number : ", last_game_data.shape)
    print("-" * 40)
    print("Length of the before last life game : ", len(before_last_life_list))
    print(before_last_life_list[:5])
    if len(before_last_life_list) > 100:
        before_last_life_list = np.random.choice(before_last_life_list, 100)
    is_before_last = all_data_with_label.file.apply(lambda x: x in before_last_life_list)
    before_last_game_index = np.where(is_before_last == 1)
    before_last_game_data = all_data_with_label.iloc[before_last_game_index]
    before_last_game_data.at_cross = _ifAtCross(adjacent_data, before_last_game_data.pacmanPos)
    print("Before last game samples number : ", before_last_game_data.shape)
    print("-" * 40)
    with open("first_life_data.pkl", "wb") as file:
        pickle.dump(first_game_data, file)
    with open("last_life_data.pkl", "wb") as file:
        pickle.dump(last_game_data, file)
    with open("before_last_life_data.pkl", "wb") as file:
        pickle.dump(before_last_game_data, file)
    print("Finished writing data!")
    print("="*40)

# ==============================================
#       EXTRACT DATA FOR EACH AGENT
# ==============================================
def _extractGlobalData(all_data_with_label, save_res = False):
    nan_index = np.where(np.isnan(all_data_with_label.label_global))
    all_data_with_label.label_global.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(all_data_with_label.label_global_optimal))
    all_data_with_label.label_global_optimal.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(all_data_with_label.label_global_notoptimal))
    all_data_with_label.label_global_notoptimal.iloc[nan_index] = 0
    is_global = all_data_with_label.apply(
        lambda x: np.logical_or(np.logical_or(x.label_global, x.label_global_optimal), x.label_global_notoptimal),
        axis=1
    )
    global_index = np.where(is_global == 1)
    global_data = all_data_with_label.iloc[global_index].reset_index(drop=True)
    print("Global data shape : ", global_data.shape)
    if not save_res:
        return
    with open("agent_data/global_data.pkl", "wb") as file:
        global_data = global_data.iloc[np.random.choice(global_data.shape[0], 20000, replace=False)].reset_index(
            drop=True)
        print("Used global data shape : ", global_data.shape)
        pickle.dump(global_data, file)
    print("Finished writing global data.")


def _extractPlannedData(all_data_with_label, plan_index, save_res = False):
    # Obtain the number of time steps for every trial
    trial_timesteps_num = all_data_with_label.groupby("file").index.count()
    trial_timesteps_num = [(trial_timesteps_num.index[i], trial_timesteps_num.iloc[i]) for i in
                           range(len(trial_timesteps_num))]
    trial_timesteps_num = {each[0]: [each[1], 0] for each in trial_timesteps_num}
    # Extract planned hunting trial
    plan_data = all_data_with_label.iloc[plan_index].reset_index(drop=True)
    print("Planned hunting data shape : ", plan_data.shape)
    for index in range(plan_data.shape[0]):
        trial_timesteps_num[plan_data.iloc[index].file][1] += 1
    trial_planned_ratio = [(each, trial_timesteps_num[each][1] / trial_timesteps_num[each][0]) for each in trial_timesteps_num]
    print(trial_planned_ratio[:5])
    trial_planned_ratio.sort(key=lambda x: x[1], reverse=True)
    print(trial_planned_ratio[:5])
    trial_name_list = [each[0] for each in trial_planned_ratio[:200]] if len(trial_planned_ratio) > 200 \
        else [each[0] for each in trial_planned_ratio]
    is_plan = all_data_with_label.file.apply(lambda x: x in trial_name_list)
    plan_game_index = np.where(is_plan == 1)
    plan_game_data = all_data_with_label.iloc[plan_game_index]
    print("Used plan data shape : ", plan_game_data.shape)
    if not save_res:
        return
    with open("agent_data/most_planned_hunting_data.pkl", "wb") as file:
        # plan_data = plan_data.iloc[np.random.choice(plan_data.shape[0], 20000, replace=False)].reset_index(drop=True)
        pickle.dump(plan_game_data, file)
    print("Finished writing planed hunting data.")


def _extracSuicideDta(all_data_with_label, save_res = False):
    # Obtain the number of time steps for every trial
    trial_timesteps_num = all_data_with_label.groupby("file").index.count()
    trial_timesteps_num = [(trial_timesteps_num.index[i], trial_timesteps_num.iloc[i]) for i in
                           range(len(trial_timesteps_num))]
    trial_timesteps_num = {each[0]: [each[1], 0] for each in trial_timesteps_num}
    # Extract suicide trial
    trial_name_list = np.unique(all_data_with_label.file.values)
    useful_trial_name_list = []
    for trial_name in trial_name_list:
        is_useful = _usefulSuicide(all_data_with_label[all_data_with_label.file == trial_name].label_suicide.values)
        if is_useful:
            useful_trial_name_list.append(trial_name)
    useful_trial_name_list = np.random.choice(useful_trial_name_list, 200, replace=False) if len(useful_trial_name_list) > 200 else useful_trial_name_list
    is_suicide = all_data_with_label.file.apply(lambda x: x in useful_trial_name_list)
    suicide_game_index = np.where(is_suicide == 1)
    suicide_game_data = all_data_with_label.iloc[suicide_game_index]
    print("Used suicide data shape : ", suicide_game_data.shape)
    if not save_res:
        return
    with open("agent_data/really_suicide_data.pkl", "wb") as file:
        pickle.dump(suicide_game_data, file)
    print("Finished writing suicide data.")


def _usefulSuicide(label_suicide):
    suicide_count = 0
    for index in list(range(len(label_suicide)-1, -1, -1)):
        if label_suicide[index] == 1:
            suicide_count += 1
            if suicide_count == 5:
                return True
        else:
            if suicide_count == 5:
                return True
            return False
    return False


def extractAgentData(data_list, save_res = False):
    print("Start extracting data...")
    print(data_list)
    # Configurations
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    # data_filename = "first_life_data.pkl"
    # Read data
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    all_data_with_label = data["df_total"]

    # all_data_with_label = data

    all_data_with_label = all_data_with_label.sort_index()
    print("Shape of all data : ", all_data_with_label.shape)
    # Extract planned hunting data
    if "planned hunting" in data_list:
        plan_index = np.concatenate(data["cons_list_plan"])

        # plan_index = np.arange(300)


        _extractPlannedData(all_data_with_label, plan_index, save_res)
    # Extract suicide data
    if "suicide" in data_list:
        _extracSuicideDta(all_data_with_label, save_res)
    # Extract global data
    if "global" in data_list:
        _extractGlobalData(all_data_with_label, save_res)
    print("Finished extracting all the data!")
    print("-"*30)




if __name__ == '__main__':
    # # Extract the first life and the last life data
    # extractLifeGame()

    # Extract data for every agent
    extractAgentData(data_list=["planned hunting", "suicide"], save_res = True)

    # print(_usefulSuicide([0, 0, 1, 1, 0, 1, 1, 1]))



