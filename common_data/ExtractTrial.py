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
import sys
sys.path.append("../Utility_Tree_Analysis")
sys.path.append("../")

from LabelAnalysis import _makeChoice, _handcraftLabeling
from Plotting import _estimationVagueLabeling


dir_list = ['left', 'right', 'up', 'down']

label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
              "label_global_optimal", "label_global_notoptimal", "label_global",
              "label_evade",
              "label_suicide",
              "label_true_accidental_hunting",
              "label_true_planned_hunting"]

all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]


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


def extractOmegaData(trial_num = 10):
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


def extractPatamonData(trial_num = 10):
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
        if "Patamon" in each.split("-"):
            temp_list.append(each)
    trial_name_list = np.array(temp_list)
    print("Num of all the Patamon trials : ", len(trial_name_list))
    if trial_num < len(trial_name_list):
        trial_name_list = np.random.choice(trial_name_list, trial_num)
    print("Used num of Patamon trials : ", len(trial_name_list))
    is_need = all_data_with_label.file.apply(lambda x: x in trial_name_list)
    trial_index = np.where(is_need == 1)
    trial_data = all_data_with_label.iloc[trial_index]
    trial_data = trial_data.reset_index(drop = True)
    print("Data shape : ", trial_data.shape)
    # if "Patamon_trial" not in os.listdir():
    #     os.mkdir("Patamon_trial")
    with open("trial/{}_trial_data_Patamon.pkl".format(trial_num), "wb") as file:
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


def _multiAgentDir(weight, window_Q):
    if weight is None or isinstance(weight, float):
        return np.nan
    weight = weight[:-1]
    index = int(window_Q.shape[1] - 1 / 2)
    Q_value = window_Q[:, index, :]
    return dir_list[_makeChoice(weight @ Q_value)]


def transferTrialData():
    # For Omega
    print("="*20, " Omega ", "="*20)
    with open("./trial/8000_trial_data_Omega-with_Q-with_weight-window3-new_agents.pkl", "rb") as file:
        data = pickle.load(file)
    print("Finished reading data.")
    omega_trials = []
    trial_data = data.groupby("file")
    count = 0
    for each_trial in trial_data:
        if count >= 200:
            break
        print("-"*40)
        print("|{}| Trial Name : ".format(count+1), each_trial[0])
        trial = each_trial[1]
        true_dir = trial.next_pacman_dir_fill
        global_dir = trial.global_Q.apply(lambda x: dir_list[_makeChoice(x)])
        local_dir = trial.local_Q.apply(lambda x: dir_list[_makeChoice(x)])
        evade_dir = trial.pessimistic_Q.apply(lambda x: dir_list[_makeChoice(x)])
        attack_dir = trial.planned_hunting_Q.apply(lambda x: dir_list[_makeChoice(x)])
        suicide_dir = trial.suicide_Q.apply(lambda x: dir_list[_makeChoice(x)])
        multi_agent_dir = trial[["weight", "window_Q"]].apply(lambda x: _multiAgentDir(x.weight, x.window_Q), axis = 1)
        hand_crafted_label = trial[label_list].apply(lambda x : _handcraftLabeling(x), axis = 1)
        fitted_label = trial.weight.apply(lambda x: _estimationVagueLabeling(x, all_agent_list) if not isinstance(x, float) else np.nan)
        trial["true_dir"] = true_dir
        trial["global_dir"] = global_dir
        trial["local_dir"] = local_dir
        trial["evade_dir"] = evade_dir
        trial["attack_dir"] = attack_dir
        trial["suicide_dir"] = suicide_dir
        trial["multi_agent_dir"] = multi_agent_dir
        trial["hand_crafted_label"] = hand_crafted_label
        trial["fitted_label"] = fitted_label
        omega_trials.append(copy.deepcopy(trial))
        count += 1
    print("Finished extracting Omega data.")
    with open("./trial/200trial_Omega_videos.pkl", "wb") as file:
        pickle.dump(omega_trials, file)
    print("Finished saving Omega data.")
    # For Patamon
    print("\n")
    print("=" * 20, " Patamon ", "=" * 20)
    with open("./trial/7000_trial_data_Patamon-with_Q-with_weight-window3-new_agents.pkl", "rb") as file:
        data = pickle.load(file)
    print("Finished reading data.")
    patamon_trials = []
    trial_data = data.groupby("file")
    count = 0
    for each_trial in trial_data:
        if count >= 200:
            break
        print("-" * 40)
        print("|{}| Trial Name : ".format(count+1), each_trial[0])
        trial = each_trial[1]
        true_dir = trial.next_pacman_dir_fill
        global_dir = trial.global_Q.apply(lambda x: dir_list[_makeChoice(x)])
        local_dir = trial.local_Q.apply(lambda x: dir_list[_makeChoice(x)])
        evade_dir = trial.pessimistic_Q.apply(lambda x: dir_list[_makeChoice(x)])
        attack_dir = trial.planned_hunting_Q.apply(lambda x: dir_list[_makeChoice(x)])
        suicide_dir = trial.suicide_Q.apply(lambda x: dir_list[_makeChoice(x)])
        multi_agent_dir = trial[["weight", "window_Q"]].apply(lambda x: _multiAgentDir(x.weight, x.window_Q), axis=1)
        hand_crafted_label = trial[label_list].apply(lambda x: _handcraftLabeling(x), axis=1)
        fitted_label = trial.weight.apply(
            lambda x: _estimationVagueLabeling(x, all_agent_list) if not isinstance(x, float) else np.nan)
        trial["true_dir"] = true_dir
        trial["global_dir"] = global_dir
        trial["local_dir"] = local_dir
        trial["evade_dir"] = evade_dir
        trial["attack_dir"] = attack_dir
        trial["suicide_dir"] = suicide_dir
        trial["multi_agent_dir"] = multi_agent_dir
        trial["hand_crafted_label"] = hand_crafted_label
        trial["fitted_label"] = fitted_label
        patamon_trials.append(copy.deepcopy(trial))
        count += 1
    print("Finished extracting Patamon data.")
    with open("./trial/200trial_Patamon_videos.pkl", "wb") as file:
        pickle.dump(patamon_trials, file)
    print("Finished saving Patamon data.")





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
    # extractOmegaData(trial_num=8000)
    # extractPatamonData(trial_num=7000)

    # _extractOneTrial()

    # _extractMultiTrial(trial_num = 100)

    # transferTrialData()
    # with open("trial/200trial_Omega_videos.pkl", "rb") as file:
    #     data = pickle.load(file)
    #     print(len(data))
    #     sample = data[0]
    #     print()

    # data_filename = "/home/qlyang/Documents/pacman/constants/all_data_new.pkl"
    # with open(data_filename, "rb") as file:
    #     data = pickle.load(file)
    # print(data.keys())

    pass