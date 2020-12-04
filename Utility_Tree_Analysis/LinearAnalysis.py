'''
Description:
    Use linear models to fit features.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    28 Oct. 2020
'''

import pickle
import pandas as pd
import numpy as np
import scipy.optimize
import scipy.stats
import matplotlib.pyplot as plt
import copy
import seaborn
import os
import sys
from sklearn.linear_model import Perceptron

sys.path.append("./")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber
from LabelAnalysis import readTrialData, negativeLikelihood, _makeChoice


dir_list = ['left', 'right', 'up', 'down']
locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
reborn_pos = (14, 27)
inf_val = 100


def oneHot(val):
    '''
    Convert the direction into a one-hot vector.
    :param val: The direction. should be the type ``str''.
    :return:
    '''
    # Type check
    if val not in dir_list:
        raise ValueError("Undefined direction {}!".format(val))
    if not isinstance(val, str):
        raise TypeError("Undefined direction type {}!".format(type(val)))
    # One-hot
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[dir_list.index(val)] = 1
    return onehot_vec


def readTrialDataForLR(filename):
    # Read data and pre-processing
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index(drop=True)
    # Split into trials
    trial_data = []
    trial_name_list = np.unique(all_data.file.values)
    for each in trial_name_list:
        each_trial = all_data[all_data.file == each].reset_index(drop = True)
        # True moving directions
        true_prob = each_trial.next_pacman_dir_fill
        # Fill nan direction for optimization use
        start_index = 0
        while pd.isna(true_prob[start_index]):
            start_index += 1
            if start_index == len(true_prob):
                break
        if start_index == len(true_prob):
            print("Moving direciton of trial {} is all nan.".format(each))
            continue
        if start_index > 0:
            true_prob[:start_index + 1] = true_prob[start_index + 1]
        for index in range(1, true_prob.shape[0]):
            if pd.isna(true_prob[index]):
                true_prob[index] = true_prob[index - 1]
        true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
        trial_data.append([each, each_trial, true_prob])
    print("Finished reading data...")
    return trial_data


def extractFeature(trial_data):
    trial_features = []
    trial_labels = []
    trial_names = []
    #
    for trial_index, each in enumerate(trial_data):
        trial_names.append(each[0])
        trial = each[1]
        true_dir = each[2]
        # Features for the estimation
        PG1 = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : 0 if x.pacmanPos == x.ghost1Pos else locs_df[x.pacmanPos][x.ghost1Pos],
            axis = 1
        )
        PG2 = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: 0 if x.pacmanPos == x.ghost2Pos else locs_df[x.pacmanPos][x.ghost2Pos],
            axis=1
        )
        min_PE = trial[["pacmanPos", "energizers"]].apply(
            lambda x : inf_val if isinstance(x.energizers, float)
            else np.min([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.energizers]),
            axis = 1
        )
        PF = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x : inf_val if isinstance(x.fruitPos, float)
            else (0 if x.pacmanPos == x.fruitPos else locs_df[x.pacmanPos][x.fruitPos]),
            axis = 1
        )
        beans_15step = trial[["pacmanPos", "beans"]].apply(
            lambda x : 0 if isinstance(x.beans, float)
            else np.sum(
                np.where(
                    np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) <= 15
                )
            ),
            axis = 1
        )
        beans_diff = trial[["pacmanPos", "beans"]].apply(
            lambda x : 0 if isinstance(x.beans, float)
            else np.sum(
                np.where(
                    np.array([0 if reborn_pos == each else locs_df[reborn_pos][each] for each in x.beans]) <= 15
                )
            ) - np.sum(
                np.where(
                    np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) <= 15
                )
            ),
            axis = 1
        )
        # trial_data["PG1"], trial_data["PG2"], trial_data["min_PE"], trial_data["PF"], trial_data["beans_15step"], trial_data["beans_diff"] \
        #     = [PG1, PG2, min_PE, PF, beans_15step, beans_diff]
        processed_trial_data = pd.DataFrame(
            data=
            {
                "ifscared1" : trial.ifscared1,
                "ifscared2" : trial.ifscared2,
                "PG1" : PG1,
                "PG2" : PG2,
                "min_PE" : min_PE,
                "PF" : PF,
                "beans_15step" : beans_15step,
                "beans_diff" : beans_diff
            }
        )
        # X = trial_data[["ifscared1", "ifscared2", "PG1", "PG2", "min_PE", "PF", "beans_15step", "beans_diff"]]
        X = processed_trial_data
        y = true_dir.apply(lambda x : list(x).index(1))
        trial_features.append(copy.deepcopy(X))
        trial_labels.append(copy.deepcopy(y))
    print("Finished extracing features...")
    return trial_names, trial_features, trial_labels

#TODO: for every adjacent positions

def _adjacentDist(pacmanPos, ghostPos, type, adjacent_data):
    # Pacman in tunnel
    if pacmanPos == (29, 18):
        pacmanPos = (28, 18)
    if pacmanPos == (0, 18):
        pacmanPos = (1, 18)
    if isinstance(adjacent_data[pacmanPos][type], float):
        return inf_val
    # Find adjacent positions
    if type == "left":
        adjacent = (pacmanPos[0] - 1, pacmanPos[1])
    elif type == "right":
        adjacent = (pacmanPos[0] + 1, pacmanPos[1])
    elif type == "up":
        adjacent = (pacmanPos[0], pacmanPos[1] - 1)
    elif type == "down":
        adjacent = (pacmanPos[0], pacmanPos[1] + 1)
    else:
        raise ValueError("Undefined direction {}!".format(type))
    # Adjacent positions in he tunnel
    if adjacent == (29, 18):
        adjacent = (28, 18)
    elif adjacent == (0, 18):
        adjacent = (1, 18)
    else:
        pass
    return 0 if adjacent == ghostPos else locs_df[adjacent][ghostPos]


def extractFeatureWRTDir(trial_data):
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    trial_features = []
    trial_labels = []
    trial_names = []
    #
    for trial_index, each in enumerate(trial_data):
        trial_names.append(each[0])
        trial = each[1]
        true_dir = each[2]
        # Features for the estimation

        PG1_left = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "left", adjacent_data),
            axis = 1
        )
        PG1_right = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "right", adjacent_data),
            axis = 1
        )
        PG1_up = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "up", adjacent_data),
            axis = 1
        )
        PG1_down = trial[["pacmanPos", "ghost1Pos"]].apply(
            lambda x : _adjacentDist(x.pacmanPos, x.ghost1Pos, "down", adjacent_data),
            axis = 1
        )

        PG2_left = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "left", adjacent_data),
            axis=1
        )
        PG2_right = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "right", adjacent_data),
            axis=1
        )
        PG2_up = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "up", adjacent_data),
            axis=1
        )
        PG2_down = trial[["pacmanPos", "ghost2Pos"]].apply(
            lambda x: _adjacentDist(x.pacmanPos, x.ghost2Pos, "down", adjacent_data),
            axis=1
        )

        PE_left = trial[["pacmanPos", "energizers"]].apply(
            lambda x : inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "left", adjacent_data) for each in x.energizers]),
            axis = 1
        )
        PE_right = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "right", adjacent_data) for each in x.energizers]),
            axis=1
        )
        PE_up = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "up", adjacent_data) for each in x.energizers]),
            axis=1
        )
        PE_down = trial[["pacmanPos", "energizers"]].apply(
            lambda x: inf_val if isinstance(x.energizers, float)
            else np.min([_adjacentDist(x.pacmanPos, each, "down", adjacent_data) for each in x.energizers]),
            axis=1
        )

        PF_left = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x : inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "left", adjacent_data),
            axis = 1
        )
        PF_right = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x: inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "right", adjacent_data),
            axis=1
        )
        PF_up = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x: inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "up", adjacent_data),
            axis=1
        )
        PF_down = trial[["pacmanPos", "fruitPos"]].apply(
            lambda x : inf_val if isinstance(x.fruitPos, float)
            else _adjacentDist(x.pacmanPos, x.fruitPos, "down", adjacent_data),
            axis = 1
        )
        beans_15step = trial[["pacmanPos", "beans"]].apply(
            lambda x : 0 if isinstance(x.beans, float)
            else np.sum(
                np.where(
                    np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) <= 15
                )
            ),
            axis = 1
        )
        beans_diff = trial[["pacmanPos", "beans"]].apply(
            lambda x : 0 if isinstance(x.beans, float)
            else np.sum(
                np.where(
                    np.array([0 if reborn_pos == each else locs_df[reborn_pos][each] for each in x.beans]) <= 15
                )
            ) - np.sum(
                np.where(
                    np.array([0 if x.pacmanPos == each else locs_df[x.pacmanPos][each] for each in x.beans]) <= 15
                )
            ),
            axis = 1
        )
        # trial_data["PG1"], trial_data["PG2"], trial_data["min_PE"], trial_data["PF"], trial_data["beans_15step"], trial_data["beans_diff"] \
        #     = [PG1, PG2, min_PE, PF, beans_15step, beans_diff]
        processed_trial_data = pd.DataFrame(
            data=
            {
                "ifscared1" : trial.ifscared1,
                "ifscared2" : trial.ifscared2,

                "PG1_left" : PG1_left,
                "PG1_right": PG1_right,
                "PG1_up": PG1_up,
                "PG1_down": PG1_down,

                "PG2_left" : PG2_left,
                "PG2_right": PG2_right,
                "PG2_up": PG2_up,
                "PG2_down": PG2_down,

                "PE_left" : PE_left,
                "PE_right": PE_right,
                "PE_up": PE_up,
                "PE_down": PE_down,

                "PF_left" : PF_left,
                "PF_right": PF_right,
                "PF_up": PF_up,
                "PF_down": PF_down,

                "beans_15step" : beans_15step,
                "beans_diff" : beans_diff
            }
        )
        X = processed_trial_data
        y = true_dir.apply(lambda x : list(x).index(1))
        trial_features.append(copy.deepcopy(X))
        trial_labels.append(copy.deepcopy(y))
    print("Finished extracing features...")
    return trial_names, trial_features, trial_labels


def perceptron(config, feature_type):
    print("="*30)
    data_filename = config["data_filename"]
    window = config["window"]
    print("Filename : ", data_filename)
    print("Window size : ", window)
    print("Feature Type : ", feature_type)
    if feature_type == "features":
        trial_names, trial_features, trial_labels = extractFeature(readTrialDataForLR(data_filename))
    elif feature_type == "features_wrt_dir":
        trial_names, trial_features, trial_labels = extractFeatureWRTDir(readTrialDataForLR(data_filename))
    else:
        raise ValueError("Undefine feature type \"{}\"!".format(feature_type))
    trial_num = len(trial_names)
    print("Num of trials : ", trial_num)
    trial_index = list(range(trial_num))
    if config["trial_num"] is not None:
        if config["trial_num"] < trial_num:
            trial_index = np.random.choice(range(trial_num), config["trial_num"], replace = False)
    trial_names = [trial_names[each] for each in trial_index]
    trial_features = [trial_features[each] for each in trial_index]
    trial_labels = [trial_labels[each] for each in trial_index]

    all_weight = []
    all_cr = []
    for trial_index, each in enumerate(trial_names):
        trial_name = each
        X = trial_features[trial_index]
        y = trial_labels[trial_index]
        print("-" * 30)
        print("{} : {}".format(trial_index, trial_name))
        trial_length = X.shape[0]
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        trial_weight = np.zeros((len(window_index), 4, X.shape[1] + 1)) # (# of windows, 4 classes, # of features + intercept)
        trial_cr = np.zeros((len(window_index),))
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = y[centering_point - window:centering_point + window + 1]
            origin_sub_X = copy.deepcopy(sub_X)
            origin_sub_Y = copy.deepcopy(sub_Y)
            # Complete four classes
            for label in [0, 1, 2, 3]:
                if label not in sub_Y.values:
                    sub_X = sub_X.append({each : inf_val for each in sub_X.columns.values}, ignore_index = True)
                    sub_Y = sub_Y.append(pd.Series(label), ignore_index = True)
            model = Perceptron()
            model.fit(sub_X, sub_Y)
            cr = model.score(origin_sub_X, origin_sub_Y)
            trial_cr[centering_index] = cr
            trial_weight[centering_index, :, :] = np.concatenate([model.coef_, model.intercept_.reshape((4, 1))], axis = 1)
        print("Trial avg cr : ", np.mean(trial_cr))
        all_weight.append(trial_weight)
        all_cr.append(trial_cr)
    print("="*30)
    print("Overall avg cr : ", np.mean([np.mean(each) for each in all_cr]))
    if "LR_comparison" not in os.listdir("../common_data"):
        os.mkdir("../common_data/{}".format("LR_comparison"))
    save_base = data_filename.split("/")[-1].split(".")[0]
    np.save("../common_data/LR_comparison/{}-window{}-perceptron-{}-weight.npy".format(save_base, window, feature_type), all_weight)
    np.save("../common_data/LR_comparison/{}-window{}-perceptron-{}-cr.npy".format(save_base, window, feature_type), all_cr)
    print("Finished saving for perceptron...")


def multiAgentAnalysis(config):
    print("=" * 30)
    data_filename = config["data_filename"]
    window = config["window"]
    print("Filename : ", data_filename)
    print("Window size : ", window)
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic", "suicide", "planned_hunting"]]
    temp_trial_data = readTrialData(data_filename)
    trial_num = len(temp_trial_data)
    print("Num of trials : ", trial_num)
    trial_index = range(trial_num)
    if config["trial_num"] is not None:
        if config["trial_num"] < trial_num:
            trial_index = np.random.choice(range(trial_num), config["trial_num"], replace = False)
    trial_data = [temp_trial_data[each] for each in trial_index]
    trial_weight = []
    trial_Q = []
    trial_contribution = []
    trial_cr = []
    agent_name = config["agents"]
    print(agent_name)
    agent_index = [["global", "local", "pessimistic", "suicide", "planned_hunting"].index(each) for each in agent_name]
    for trial_index, each in enumerate(trial_data):
        print("-"*15)
        trial_name = each[0]
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print(trial_index, " : ", trial_name)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight = np.zeros((len(window_index), len(agent_name) + 1))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, 5, 4))
        temp_contribution = []
        temp_cr = []
        # For each trial, estimate agent weights through sliding windows
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            sub_X = X[centering_point - window:centering_point + window + 1]
            sub_Y = Y[centering_point - window:centering_point + window + 1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(5):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
            # estimation in the window
            # Construct optimizer
            params = [0 for _ in range(len(agent_name))]
            bounds = [[0, 10] for _ in range(len(agent_name))]
            params.append(1)
            bounds.append([-1000, 1000])
            cons = []  # construct the bounds in the form of constraints
            for par in range(len(bounds)):
                l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
                u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
                cons.append(l)
                cons.append(u)
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
                agent_name,
                return_trajectory = False,
                need_intercept = True
            )
            is_success = False
            retry_num = 0
            while not is_success and retry_num < 5:
                res = scipy.optimize.minimize(
                    func,
                    x0=params,
                    method="SLSQP",
                    bounds=bounds,
                    tol=1e-5,
                    constraints=cons
                )
                is_success = res.success
                if not is_success:
                    print("Fail, retrying...")
                    retry_num += 1
            # correct rate in the window
            _, estimated_prob = negativeLikelihood(
                res.x,
                sub_X,
                sub_Y,
                agent_name,
                return_trajectory=True,
                need_intercept=True
            )
            # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            temp_cr.append(correct_rate)
            temp_weight[centering_index, :] = res.x
            contribution = temp_weight[centering_index, :-1] * [scaleOfNumber(each) for each in
                                np.max(np.abs(temp_trial_Q[centering_index, :, agent_index, :]), axis=(1, 2))]
            temp_contribution.append(copy.deepcopy(contribution))
        print("Trial avg cr : ", np.nanmean(temp_cr))
        trial_contribution.append(copy.deepcopy(temp_contribution))
        trial_weight.append(copy.deepcopy(temp_weight))
        trial_Q.append(copy.deepcopy(temp_trial_Q))
        trial_cr.append(temp_cr)
    print("="*30)
    print("Overall avg cr : ", np.nanmean([np.nanmean(each) for each in trial_cr]))
    # Save data
    if "LR_comparison" not in os.listdir("../common_data"):
        os.mkdir("../common_data/{}".format("LR_comparison"))
    save_base = data_filename.split("/")[-1].split(".")[0]
    np.save("../common_data/LR_comparison/{}-window{}-agent-{}-weight.npy".format(save_base, window, "_".join(agent_name)), trial_weight)
    np.save("../common_data/LR_comparison/{}-window{}-agent-{}-cr.npy".format(save_base, window, "_".join(agent_name)), trial_cr)
    np.save("../common_data/LR_comparison/{}-window{}-agent-{}-Q.npy".format(save_base, window, "_".join(agent_name)), trial_Q)
    np.save("../common_data/LR_comparison/{}-window{}-agent-{}-contribution.npy".format(save_base, window, "_".join(agent_name)), trial_contribution)
    print("Finished saving for multi-agent...")


def comparison(config):
    if "features" in config["analysis"]:
        print("="*14, " Perceptron ", "="*14)
        perceptron(config, "features")
        print("\n\n")
    if "features_wrt_dir" in config["analysis"]:
        print("=" * 14, " Perceptron (dir features) ", "=" * 14)
        perceptron(config, "features_wrt_dir")
        print("\n\n")
    if "multi-agent" in config["analysis"]:
        print("=" * 14, " Multi-Agent ", "=" * 14)
        multiAgentAnalysis(config)


def showResults():
    perceptron_cr = np.load("../common_data/LR_comparison/100_trial_data_all_new-with_Q-window5-perceptron-cr.npy", allow_pickle=True)
    perceptron_dir_cr = np.load("../common_data/LR_comparison/100_trial_data_all_new-with_Q-window5-perceptron-features_wrt_dir-cr.npy",allow_pickle=True)
    multi_agent_cr = np.load("../common_data/LR_comparison/100_trial_data_all_new-with_Q-window5-agent-cr.npy", allow_pickle=True)
    local_cr = np.load("../common_data/LR_comparison/100_trial_data_all_new-with_Q-window5-agent-local-cr.npy", allow_pickle=True)
    perceptron_trial_cr = [np.nanmean(each) for each in perceptron_cr]
    perceptron_dir_trial_cr = [np.nanmean(each) for each in perceptron_dir_cr]
    multi_agent_trial_cr = [np.nanmean(each) for each in multi_agent_cr]
    local_trial_cr = [np.nanmean(each) for each in local_cr]
    print("Perceptron : ", np.mean(perceptron_trial_cr))
    print("Perceptron (dir) : ", np.mean(perceptron_dir_trial_cr))
    print("Multi-Agent : ", np.mean(multi_agent_trial_cr))
    print("Multi-Agent (local) : ", np.mean(local_trial_cr))
    cr = np.concatenate([[perceptron_trial_cr], [perceptron_dir_trial_cr], [local_trial_cr], [multi_agent_trial_cr]]).T

    from palettable.colorbrewer.diverging import RdBu_7
    color = RdBu_7.mpl_colors
    plt.hist(cr, density = False,  histtype='bar', bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
             label = ["Perceptron", "Perceptron (w.r.t. dir)", "Multi-Agent (local)", "Multi-Agent (all)"], align="mid",
             rwidth = 1.0, color=[color[-2], color[-1], color[1], color[0]])
    plt.xlabel("Joystick Movement Estimation Correct Rate", fontsize = 20)
    plt.xticks(np.arange(0.0, 1.1, 0.1), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.xlim(0.3, 1.0)
    plt.ylabel("# of Trials", fontsize=20)
    plt.legend(frameon = False, fontsize = 20, ncol = 2)
    plt.show()



if __name__ == '__main__':
    config = {
        # "data_filename" : "../common_data/trial/500_trial_data_Omega-with_Q.pkl",
        # "data_filename": "../common_data/trial/5_trial-data_for_comparison-one_ghost-with_Q-with_weight.pkl",
        "data_filename": "../common_data/trial/100_trial_data_all_new-with_Q.pkl",
        "window" : 5,
        "trial_num" : None,
        "need_intercept" : True,
        # "analysis" : ["features", "features_wrt_dir", "multi-agent"],
        "analysis": ["multi-agent"],
        "agents" : ["local"]
    }

    # comparison(config)

    showResults()