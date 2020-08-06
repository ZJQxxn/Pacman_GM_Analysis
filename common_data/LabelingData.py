'''
Description:
    Labeling the data.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    25 July 2020
'''


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _ifScared(ifscared):
    ifscared = (ifscared >= 4)
    return ifscared


def _isGlobal(label_global, label_global_optimal, label_global_notoptimal):
    nan_index = np.where(np.isnan(label_global))
    label_global.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(label_global_optimal))
    label_global_optimal.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(label_global_notoptimal))
    label_global_notoptimal.iloc[nan_index] = 0
    return np.logical_or(
        np.logical_or(label_global, label_global_optimal),
        label_global_notoptimal)


def _isLocal(label_local_graze, label_local_graze_noghost):
    nan_index = np.where(np.isnan(label_local_graze))
    label_local_graze.iloc[nan_index] = 0
    nan_index = np.where(np.isnan(label_local_graze_noghost))
    label_local_graze_noghost.iloc[nan_index] = 0
    return np.logical_or(label_local_graze, label_local_graze_noghost)


def _isEvade(label_evade, ifscared1, ifscared2):
    # ifscared1 = _ifScared(ifscared1)
    # ifscared2 = _ifScared(ifscared2)
    # is_evade = ifscared1 and ifscared2 and label_evade
    nan_index = np.where(np.isnan(label_evade))
    label_evade.iloc[nan_index] = 0
    return label_evade


def _isSuicide(label_suicide, ifscared1, ifscared2):
    nan_index = np.where(np.isnan(label_suicide))
    label_suicide.iloc[nan_index] = 0
    return label_suicide


def _isOptimistic(label_hunt1, label_hunt2, label_prehunt, label_huntfailure_hunt, label_huntfailure_graze, ifscared1, ifscared2):
    nan_index = np.where(np.isnan(label_prehunt))
    label_prehunt.iloc[nan_index] = 0
    return label_prehunt


def _isPessimistic(label_evade, ifscared1, ifscared2):
    nan_index = np.where(np.isnan(label_evade))
    label_evade.iloc[nan_index] = 0
    ifscared1 = _ifScared(ifscared1)
    ifscared2 = _ifScared(ifscared2)
    return np.logical_and(
        np.logical_and(1-ifscared1, ifscared2),
        label_evade)


def labeling(df_total):
    is_local = _isLocal(df_total.label_local_graze, df_total.label_local_graze_noghost)
    is_global = _isGlobal(df_total.label_global, df_total.label_global_optimal, df_total.label_global_notoptimal)
    is_evade =  _isEvade(df_total.label_evade, df_total.ifscared1, df_total.ifscared2)
    is_suicide = _isSuicide(df_total.label_suicide, df_total.ifscared1, df_total.ifscared2)
    is_optimistic = _isOptimistic(df_total.label_hunt1, df_total.label_hunt2, df_total.label_prehunt,
                                  df_total.label_huntfailure_hunt, df_total.label_huntfailure_graze,
                                  df_total.ifscared1, df_total.ifscared2)
    is_pessimistic = _isPessimistic(df_total.label_evade, df_total.ifscared1, df_total.ifscared2)
    print("The number of local : ", np.sum(is_local))
    print("The number of global : ", np.sum(is_global))
    print("The number of evade : ", np.sum(is_evade))
    print("The number of suicide : ", np.sum(is_suicide))
    print("The number of optimistic : ", np.sum(is_optimistic))
    print("The number of pessimistic : ", np.sum(is_pessimistic))
    return (is_local, is_global, is_evade, is_suicide, is_optimistic, is_pessimistic)






if __name__ == '__main__':
    # Configurations
    data_filename = "1-1-Omega-15-Jul-2019-1.csv-trial_data_with_label.pkl"
    # Read in the complet data
    with open(data_filename, "rb") as file:
        df_total = pickle.load(file)
    print("=" * 20)
    print("Finished reading.")
    # print(df_total.columns.values)
    is_local, is_global, is_evade, is_suicide, is_optimistic, is_pessimistic = labeling(df_total)

    # Plot for test
    local_label_index = np.where(is_local)
    global_label_index = np.where(is_global)
    suicide_label_index = np.where(is_suicide)
    for each in local_label_index[0]:
        plt.fill_between(x = [each, each+1], y1 = 0, y2 = -0.05, color = "blue")
    for each in global_label_index[0]:
        plt.fill_between(x = [each, each+1], y1 = 0, y2 = -0.05, color = "red")
    for each in suicide_label_index[0]:
        plt.fill_between(x = [each, each+1], y1 = 0, y2 = -0.05, color = "green")
    plt.ylim(-0.05, 0.2)
    plt.yticks(np.arange(0, 0.21, 0.1), np.arange(0, 0.21, 0.1))
    plt.show()