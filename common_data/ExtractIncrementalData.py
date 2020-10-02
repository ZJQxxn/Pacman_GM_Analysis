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

sys.path.append("./")
from LabelingData import _isGlobal, _isLocal

sys.path.append("../Utility_Tree_Analysis/")
from TreeAnalysisUtils import readAdjacentMap


# 每一段index都是df_total的index，注意：在对df_total进行处理（尤其是merge）的时候，不要改变其index！！！！
# accidentally hunting就是all_data.pickle中的['cons_list_accident']
# planned hunting就是all_data.pickle中的['cons_list_plan']


# 这个function输出的第一个variable就是suicide_list
def _generate_suicide_normal(df_total):
    select_last_num = 100
    suicide_normal = (
        df_total.reset_index()
        .merge(
            (df_total.groupby("file")["label_suicide"].sum() > 0)
            .rename("suicide_trial")
            .reset_index(),
            on="file",
            how="left",
        )
        .sort_values(by="level_0")
        .groupby(["file", "suicide_trial"])
        .apply(lambda x: x.level_0.tail(select_last_num).tolist())
        .reset_index()
    )
    suicide_lists = suicide_normal[suicide_normal["suicide_trial"] == True][0]
    normal_lists = suicide_normal[suicide_normal["suicide_trial"] == False][0]
    return suicide_lists, normal_lists


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







if __name__ == '__main__':
    extractLifeGame()

    # with open("partial_data_with_reward_label_cross.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     print()



