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


def _obtain_suicide_list(df_total):
    index = df_total[df_total.label_suicide == 1].index
    return index


def _obtain_evade_list(df_total):
    index = df_total[df_total.label_evade == 1].index
    return index


def extractSuicideAndEvade():
    # Configurations
    data_filename = "labeled_df_toynew.pkl"
    reward_data_filename = "df_total_with_reward.pkl"
    evade_data_filename = "../Utility_Tree_Analysis/extracted_data/evade_data.pkl"
    suicide_data_filename = "../Utility_Tree_Analysis/extracted_data/one_trial_suicide_data.pkl"
    # Read in the complet data
    with open(data_filename, "rb") as file:
        df_total = pickle.load(file)
    with open(reward_data_filename, "rb") as file:
        reward_data = pickle.load(file)
    df_total["Reward"] = reward_data.Reward
    df_total["fruitPos"] = reward_data.fruitPos
    clip = 1000
    print("=" * 20)
    print("Finished reading.")
    print("Data shape : ", df_total.shape)
    # Extract evade data
    evade_list = _obtain_evade_list(df_total)
    evade_data = df_total.iloc[evade_list[:clip]]
    print("Evade data shape : ", len(evade_list))
    # Extract suicide data
    # suicide_lists, normal_lists = generate_suicide_normal(df_total)
    # print("Suicide data shape : ", len(suicide_lists))
    # trial_suicide_index = suicide_lists.values[0]
    # suicide_data = df_total.iloc[trial_suicide_index[0]-50 : trial_suicide_index[-1]+50]
    # TODO: obtain the trial with suicide time steps
    suicide_list = _obtain_suicide_list(df_total)
    print("Suicide data shape : ", len(suicide_list))
    suicide_data = df_total.iloc[suicide_list]
    # Save extracted data
    with open(evade_data_filename, "wb") as file:
        pickle.dump(evade_data, file)
    with open(suicide_data_filename, "wb") as file:
        pickle.dump(suicide_data, file)
    print("Finished writing.")


def extractTrialData():
    # Configurations
    print("="*20)
    print("EXTRACT TRIAL DATA")
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    trial_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/{}-trial_data_with_label.pkl"
    df_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/all_data_with_label.pkl"
    reward_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/df_total_with_reward.pkl"

    # Read data
    with open(data_filename, "rb") as file:
        all_data = pickle.load(file)
    all_data_with_label = all_data["df_total"]
    print(all_data_with_label.shape)
    with open(reward_data_filename, "rb") as file:
        reward_data = pickle.load(file)
    all_data_with_label["Reward"] = reward_data.Reward
    all_data_with_label["fruitPos"] = reward_data.fruitPos
    print("Finished reading.")
    # Extract trial data
    trial_name_list = ["1-1-Omega-15-Jul-2019-1.csv", "1-2-Omega-15-Jul-2019-1.csv"]
    print("Trial List : ", trial_name_list)
    # print(all_data_with_label.file.values[:5])
    print("-" * 20)
    for trial_name in trial_name_list:
        index = np.where(all_data_with_label.file.values == trial_name)
        trial_data = all_data_with_label.iloc[index]
        print("Data Shape : ", trial_data.shape)
        print("Finished extracting {}.".format(trial_name))
        with open(trial_data_filename.format(trial_name), "wb") as file:
            pickle.dump(trial_data, file)
        print("Finished writing {}.".format(trial_name))
    # # Write all data
    # with open(df_data_filename, "wb") as file:
    #     pickle.dump(all_data_with_label, file)
    # print("Finished writing all data.")
    print("="*20)


def extractGlobalData():
    # Configurations
    print("=" * 20)
    print("EXTRACT GLOBAL DATA")
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    global_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/global_data.pkl"
    df_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/all_data_with_label.pkl"
    reward_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/df_total_with_reward.pkl"

    # Read data
    with open(data_filename, "rb") as file:
        all_data = pickle.load(file)
    all_data_with_label = all_data["df_total"]
    print(all_data_with_label.shape)
    with open(reward_data_filename, "rb") as file:
        reward_data = pickle.load(file)
    all_data_with_label["Reward"] = reward_data.Reward
    all_data_with_label["fruitPos"] = reward_data.fruitPos
    print("Finished reading.")
    # Extract global data
    print("-" * 20)
    global_data = all_data_with_label[
        _isGlobal(all_data_with_label.label_global,
                  all_data_with_label.label_global_optimal,
                  all_data_with_label.label_global_notoptimal)
    ]
    print("Data Shape : ", len(global_data))
    print("Finished extracting global.")
    with open(global_data_filename, "wb") as file:
        pickle.dump(global_data.iloc[:1000], file)
    print("Finished writing first 1000 global data.")
    print("=" * 20)


def extractLocalData():
    # Configurations
    print("=" * 20)
    print("EXTRACT LOCAL DATA")
    data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    local_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/local_data.pkl"
    df_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/all_data_with_label.pkl"
    reward_data_filename = "/home/qlyang/jiaqi/Pacman-Analysis/common_data/df_total_with_reward.pkl"

    # Read data
    with open(data_filename, "rb") as file:
        all_data = pickle.load(file)
    all_data_with_label = all_data["df_total"]
    print(all_data_with_label.shape)
    with open(reward_data_filename, "rb") as file:
        reward_data = pickle.load(file)
    all_data_with_label["Reward"] = reward_data.Reward
    all_data_with_label["fruitPos"] = reward_data.fruitPos
    print("Finished reading.")
    # Extract global data
    print("-" * 20)
    local_data = all_data_with_label[
        _isLocal(all_data_with_label.label_local_graze,
                  all_data_with_label.label_local_graze_noghost)
    ]
    print("Data Shape : ", len(local_data))
    print("Finished extracting local.")
    with open(local_data_filename, "wb") as file:
        pickle.dump(local_data.iloc[:1000], file)
    print("Finished writing first 1000 local data.")
    print("=" * 20)


if __name__ == '__main__':
    # # Extract suicide and evade data
    # extractSuicideAndEvade()

    # # Extract trial data
    # extractTrialData()

    # Extract label data
    extractGlobalData()
    extractLocalData()

    pass

