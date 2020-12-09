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


def extractSuicideTrialData(trial_num = 10):
    '''
    Extrat the first 100 trial by the length of suicide.
    '''
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
    trial_name_list = np.unique(all_data_with_label.file.values)


    # Extract trial data with many suicide data
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


def extractMonkeyData(trial_num = 10):
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
    temp_list = []
    for each in trial_name_list:
        if "Omega" in each.split("-"):
            temp_list.append(each)
    trial_name_list = np.array(temp_list)
    print("Num of all the Omega trials : ", len(trial_name_list))
    if trial_num < len(trial_name_list):
        trial_name_list = np.random.choice(trial_name_list, trial_num)
    print("Used num of Omega trials : ", len(trial_name_list))
    is_need = all_data_with_label.file.apply(lambda x: x in trial_name_list)
    trial_index = np.where(is_need == 1)
    trial_data = all_data_with_label.iloc[trial_index]
    trial_data = trial_data.reset_index(drop = True)
    print("Data shape : ", trial_data.shape)
    if "Omega_trial" not in os.listdir():
        os.mkdir("Omega_trial")
    with open("trial/{}_trial_data_Omega.pkl".format(trial_num), "wb") as file:
        pickle.dump(trial_data, file)
    print("Finished writing {} trial data with the shape of {}.".format(trial_num, trial_data.shape))


def _extractOneTrial():
    # Read data
    data_filename = "./trial/50_trial_data_Omega-with_Q.pkl"  # TODO: new data
    with open(data_filename, "rb") as file:
        data = pickle.load(file)
    data = data.reset_index(drop = True)
    # trial_name = ["9-3-Omega-19-Aug-2019-1.csv"]
    trial_name = "25-2-Omega-24-Jun-2019-1.csv" # "39-1-Omega-22-Aug-2019-1.csv"
    is_need = data.file.apply(lambda x : x in trial_name)
    need_index = np.where(is_need.values == 1)
    trial_data = data.iloc[need_index]
    trial_data = trial_data.reset_index(drop = True)
    print(trial_data.shape)
    print("Finished extracting trial data.")
    with open("trial/{}.pkl".format(trial_name.split(".")[-2]), "wb") as file:
        pickle.dump(trial_data, file)
    print("Finished saving trial data.")


def _extractMultiTrial(trial_num = 100):
    with open("trial/1000_trial_data_Omega-with_Q.pkl", "rb") as file:
        data = pickle.load(file)
    data = data.sort_index()
    print("All data shape : ", data.shape)
    trial_name_list = np.unique(data.file.values)
    if trial_num < len(trial_name_list):
        trial_name_list = np.random.choice(trial_name_list, trial_num)
    is_need = data.file.apply(lambda x: x in trial_name_list)
    trial_index = np.where(is_need == 1)
    trial_data = data.iloc[trial_index].reset_index(drop = True)
    print("{} trial data shape : ".format(trial_num), trial_data.shape)
    with open("trial/{}_trial_data_Omega-with_Q.pkl".format(trial_num), "wb") as file:
        pickle.dump(trial_data, file)
    print("Finished saving data.")


if __name__ == '__main__':
    # Extract transition data
    # extractTrialData(trial_num = 500)

    # data_filename = "/home/qlyang/Documents/pacman/constants/all_data_new.pkl" #TODO: new data
    # with open(data_filename, "rb") as file:
    #     data = pickle.load(file)
    # all_data_with_label = data["df_total"]
    # print(all_data_with_label.shape)
    # print(all_data_with_label.columns.values)

    # Extract monkey data
    extractMonkeyData(trial_num=8000)

    # _extractOneTrial()

    # _extractMultiTrial(trial_num = 100)