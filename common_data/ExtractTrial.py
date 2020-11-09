'''
Description:
    Extract multiple trials of data.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    1 Oct. 2020
'''

import os
import pandas as pd
import numpy as np
import pickle
import copy


def extractTrialData(trial_num = 10):
    # Read data
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data_new.pkl" #TODO: new data
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    all_data_with_label = data["df_total"]
    all_data_with_label = all_data_with_label.sort_index()
    print(all_data_with_label.shape)
    # Add label for planned and accidental hunting
    accident_index = np.concatenate(data["cons_list_accident"])
    plan_index = np.concatenate(data["cons_list_plan"])
    is_accidental = np.zeros((all_data_with_label.shape[0], ))
    is_accidental[accident_index] = 1
    is_planned = np.zeros((all_data_with_label.shape[0],))
    is_planned[plan_index] = 1
    all_data_with_label["label_true_accidental_hunting"] = is_accidental
    all_data_with_label["label_true_planned_hunting"] = is_planned
    all_data_with_label = all_data_with_label.reset_index(drop = True)
    print("Finished labeling.")
    # Extract trial data
    trial_name_list = np.unique(all_data_with_label.file.values)
    if trial_num < len(trial_name_list):
        trial_name_list = np.random.choice(trial_name_list, trial_num)
    is_need = all_data_with_label.file.apply(lambda x: x in trial_name_list)
    trial_index = np.where(is_need == 1)
    # trial_index = []
    # for index in range(all_data_with_label.shape[0]):
    #     if all_data_with_label.iloc[index].file in trial_name_list:
    #         trial_index.append(index)
    trial_data = all_data_with_label.iloc[trial_index]
    if "trial" not in os.listdir():
        os.mkdir("trial")
    with open("trial/{}_trial_data.pkl".format(trial_num), "wb") as file:
        pickle.dump(trial_data, file)
    print("Finished writing {} trial data with the shape of {}.".format(trial_num, trial_data.shape))



if __name__ == '__main__':
    # Extract transition data
    extractTrialData(trial_num = 500)

    # data_filename = "/home/qlyang/Documents/pacman/constants/all_data_new.pkl" #TODO: new data
    # with open(data_filename, "rb") as file:
    #     data = pickle.load(file)
    # all_data_with_label = data["df_total"]
    # print(all_data_with_label.shape)
    # print(all_data_with_label.columns.values)
