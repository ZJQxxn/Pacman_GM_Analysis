import os
import pandas as pd
import numpy as np
import pickle
import copy

import sys

def extractData():
    # Read data
    # data_filename = "partial_data_with_reward_label_cross.pkl"
    # with open(data_filename, "rb") as file:
    #     all_data_with_label = pickle.load(file)

    data_filename = "/home/qlyang/Documents/pacman/constants/all_data_new.pkl"
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    all_data_with_label = data["df_total"]
    all_data_with_label = all_data_with_label.sort_index()
    all_data_with_label = all_data_with_label.reset_index(drop = True)
    print(all_data_with_label.shape)
    trial_name_list = [
        'Omega-02-Jun-2019-2.csv',
        'Patamon-12-Jun-2019-1.csv',
        'Patamon-10-Jun-2019-1.csv',
        'Omega-02-Jul-2019-2.csv',
        'Patamon-10-June-2019-1.csv',
        'Omega-02-Jul-2019-1.csv'
    ]
    all_trial_names = np.unique(all_data_with_label.file.values)
    used_trial_name_list = [[],[],[],[],[],[]]
    print(all_trial_names[0][4:])
    for each in all_trial_names:
        if each[4:] in trial_name_list:
            used_trial_name_list[trial_name_list.index(each[4:])].append(each)
    for index in range(6):
        used_trial_name_list[index] = sorted(used_trial_name_list[index])
    trial_name_list = np.concatenate(used_trial_name_list)
    # print(used_trial_name_list)
    # print(np.concatenate(used_trial_name_list))
    print("Number of trials : ", len(trial_name_list))
    is_need = all_data_with_label.file.apply(lambda x: x in trial_name_list)
    is_need = np.where(is_need == 1)
    trial_data = all_data_with_label.iloc[is_need]
    trial_data = trial_data.reset_index(drop=True)
    print("Shape : ", trial_data.shape)
    with open("./all_fig_data.pkl", "wb") as file:
        pickle.dump(trial_data, file)
    print("Finished saving data.")


if __name__ == '__main__':
    # extractData()

    with open("./all_fig_data.pkl", "rb") as file:
        trial_data = pickle.load(file)
    print(trial_data.columns.values)
    print()