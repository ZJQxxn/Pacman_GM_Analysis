'''
Description:
    Fast MLE analysis with pre-computed estimations of every agent.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    17 Aug. 2020
'''

import pickle
import pandas as pd
import numpy as np
import scipy.optimize
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

import sys
sys.path.append("./")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber, makeChoice


# ===================================
#         UTILITY FUNCTION
# ===================================
dir_list = ['left', 'right', 'up', 'down']

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


def evalList(list_str):
    list_seq = list_str.strip("[").strip("]").split(" ")
    while "" in list_seq:
        list_seq.remove("")
    return [float(each) for each in list_seq]


def readDatasetFromPkl(filename, trial_name = None, only_necessary = False):
    '''
    Construct dataset from a .pkl file.
    :param filename: Filename.
    :param trial_name: Trial name.
    :return: 
    '''
    # Read data and pre-processing
    with open(filename, "rb") as file:
        # file.seek(0) # deal with the error that "could not find MARK"
        all_data = pickle.load(file)
    if trial_name is not None: # explicitly indicate the trial
        all_data = all_data[all_data.file == trial_name]
    if "level_0" not in all_data.columns.values:
        all_data = all_data.reset_index()
    # Exclude the (0, 18) position in data
    normal_data_index = []
    for index in range(all_data.shape[0]):
        if not isinstance(all_data.global_Q[index], list): # what if no normal Q?
            normal_data_index.append(index)
    all_data = all_data.iloc[normal_data_index]
    true_prob = all_data.next_pacman_dir_fill
    # Fill nan direction for optimization use
    start_index = 0
    while pd.isna(true_prob[start_index]):
        start_index += 1
    true_prob = true_prob[start_index:].reset_index(drop = True)
    all_data = all_data[start_index:].reset_index(drop = True)
    for index in range(1, true_prob.shape[0]):
        if pd.isna(true_prob[index]):
            true_prob[index] = true_prob[index - 1]
    true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    # Construct the dataset
    if only_necessary and "at_cross" in all_data.columns.values:
        print("--- Only Necessary ---")
        at_cross_index = np.where(all_data.at_cross)
        X = all_data.iloc[at_cross_index]
        Y = true_prob.iloc[at_cross_index]
        # print("--- Data Shape {} ---".format(len(at_cross_index[0])))
    else:
        X = all_data
        Y = true_prob
    return X, Y


def readDatasetFomCSV(filename):
    with open(filename, "r") as file:
        data = pd.read_csv(file)
    data.rename(
        columns = {
            "ghost1_status": "ifscared1",
            "ghost2_status": "ifscared2",
            "fruit_pos":"fruitPos",
            "fruit_type":"Reward",
            "ghost1_pos":"ghost1Pos",
            "ghost2_pos": "ghost2Pos",
            "possible_dir": "pacman_dir_fill"
        }, inplace = True)
    for each in [
        "pacmanPos",
        "energizers",
        "beans",
        "ifscared1",
        "ifscared2",
        "ghost1Pos",
        "ghost2Pos",
        "fruitPos",
        "Reward"
    ]:
        data[each] = data[each].apply(lambda x: eval(x) if isinstance(x, str) else np.nan)
    for each in [
        "global_Q",
        "local_Q",
        "optimistic_Q",
        "pessimistic_Q",
        "suicide_Q",
        "planned_hunting_Q"
    ]:
        data[each] = data[each].apply(lambda x: evalList(x) if isinstance(x, str) else np.nan)
    data.pacman_dir = data.pacman_dir_fill.shift(1)
    # True direction
    true_prob = data.pacman_dir_fill
    true_prob = true_prob.apply(lambda x: np.array(oneHot(x)))
    return data, true_prob


# ===================================
#         FAST OPTIMIZATION
# ===================================
def _preProcessingQ(Q_value, last_dir, randomness_coeff = 0.0):
    # TODO: add noise and laziness before non-negativity?
    '''
    Preprocessing for Q-value, including convert negative utility to non-negative, set utilities of unavailable 
    directions to -inf, and normalize utilities.
    :param Q_value: 
    :return: 
    '''
    num_samples = Q_value.shape[0]
    temp_Q = []
    unavailable_index = []
    available_index = []
    for index in range(num_samples):
        cur_Q = np.array(Q_value.iloc[index])
        unavailable_index.append(np.where(cur_Q == 0))
        available_index.append(np.where(cur_Q != 0))
        # # Add randomness and lazines
        # Q_value.iloc[index][available_index[index]] = (
        #     Q_value.iloc[index][available_index[index]]
        #     + randomness_coeff * np.random.normal(size = len(available_index[index][0])) # randomness
        # )
        # if last_dir[index] is not None and dir_list.index(last_dir[index]) in available_index[index][0]:
        #     Q_value.iloc[index][dir_list.index(last_dir[index])] = (
        #         Q_value.iloc[index][dir_list.index(last_dir[index])]
        #         + scaleOfNumber(np.max(np.abs(cur_Q)))
        #     )  # laziness
        temp_Q.extend(Q_value.iloc[index])
    # Convert  negative to non-negative
    offset = 0.0
    if np.any(np.array(temp_Q) < 0):
        offset = np.min(temp_Q)
    for index in range(num_samples):
        for each in available_index[index][0]:
            Q_value.iloc[index][each] = Q_value.iloc[index][each] - offset + 1
    # Normalizing
    normalizing_factor = np.nanmax(temp_Q)
    normalizing_factor = 1 if 0 == normalizing_factor else normalizing_factor
    # Set unavailable directions
    for index in range(num_samples):
        Q_value.iloc[index] = Q_value.iloc[index] / normalizing_factor
        for each in unavailable_index[index][0]:
            Q_value.iloc[index][each] = 0
    return (offset, normalizing_factor, Q_value)


def negativeLikelihood(param, all_data, true_prob, agents_list, return_trajectory = False):
    '''
    Estimate agent weights with utility (Q-value).
    :param param: 
    :param all_data: 
    :param agent_list: 
    :param return_trajectory: 
    :return: 
    '''
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute estimation error
    nll = 0  # negative log likelihood
    num_samples = all_data.shape[0]
    agents_list = ["{}_Q".format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][each_agent]
    dir_Q_value = agent_Q_value @ agent_weight
    true_dir = true_prob.apply(lambda x: makeChoice(x)).values
    # true_dir = np.array([makeChoice(dir_Q_value[each]) if not np.isnan(dir_Q_value[each][0]) else -1 for each in range(num_samples)])
    exp_prob = np.exp(dir_Q_value)
    for each_sample in range(num_samples):
        # In computing the Q-value, divided-by-zero might exists when normalizing the Q
        # TODO: fix this in  Q-value computing
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(np.sum(exp_prob[each_sample]))
        nll = nll -log_likelihood
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def MLE(config):
    print("=" * 20, " MLE ", "=" * 20)
    print("Agent List :", config["agents"])
    # Load experiment data
    all_data, true_prob = readDatasetFomCSV(config["data_filename"])
    feasible_data_index = np.where(
        all_data["{}_Q".format(config["agents"][0])].apply(lambda x: not isinstance(x, float))
    )[0]
    all_data = all_data.iloc[feasible_data_index]
    true_prob = true_prob.iloc[feasible_data_index]
    print("Number of samples : ", all_data.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = all_data.shape[0]
    else:
        num_samples = all_data.shape[0] if config["clip_samples"] > all_data.shape[0] else config["clip_samples"]
    all_data = all_data.iloc[:num_samples]
    true_prob = true_prob.iloc[:num_samples]
    # pre-processing of Q-value
    agent_normalizing_factors = []
    agent_offset = []
    for agent_name in ["{}_Q".format(each) for each in config["agents"]]:
        preprocessing_res = _preProcessingQ(all_data[agent_name], last_dir = all_data.pacman_dir.values)
        agent_offset.append(preprocessing_res[0])
        agent_normalizing_factors.append(preprocessing_res[1])
        all_data[agent_name] = preprocessing_res[2]
    print("Number of used samples : ", all_data.shape[0])
    print("Agent Normalizing Factors : ", agent_normalizing_factors)
    print("Agent Offset : ", agent_offset)
    # Optimization
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)

    # Notes [Jiaqi Aug. 13]: -- about the lambda function --
    # params = [0, 0, 0, 0]
    # func = lambda parameter: func() [WRONG]
    # func = lambda params: func() [CORRECT]
    func = lambda params: negativeLikelihood(
        params,
        all_data,
        true_prob,
        config["agents"],
        return_trajectory = False
    )
    is_success = False
    retry_num = 0
    while not is_success and retry_num < config["maximum_try"]:
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds, # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints = cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print("Normalized Parameter (res / sum(res)): ", res.x / np.sum(res.x))
    print("Message : ", res.message)
    # Estimation
    testing_data, testing_true_prob = readDatasetFomCSV(config["testing_data_filename"])    # not_nan_index = [each for each in not_nan_index if ]
    print("Testing data num : ", testing_data.shape[0])
    _, estimated_prob = negativeLikelihood(
        res.x,
        testing_data,
        testing_true_prob,
        config["agents"],
        return_trajectory = True
    )
    true_dir = np.array([np.argmax(each) for each in testing_true_prob])
    # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate on testing data: ", correct_rate / len(testing_true_prob))


def movingWindowAnalysis(config, save_res = True):
    print("=" * 20, " Moving Window ", "=" * 20)
    print("Agent List :", config["agents"])
    window = config["window"]
    # Load experiment data
    X, Y = readDatasetFomCSV(config["data_filename"])
    print("Number of samples : ", X.shape[0])
    if "clip_samples" not in config or config["clip_samples"] is None:
        num_samples = X.shape[0]
    else:
        num_samples = X.shape[0] if config["clip_samples"] > X.shape[0] else config["clip_samples"]
    X = X.iloc[:num_samples]
    Y = Y.iloc[:num_samples]
    # pre-processing of Q-value
    agent_normalizing_factors = []
    agent_offset = []
    for agent_name in ["{}_Q".format(each) for each in config["agents"]]:
        preprocessing_res = _preProcessingQ(X[agent_name], X.pacman_dir.values, randomness_coeff=1.0)
        agent_offset.append(preprocessing_res[0])
        agent_normalizing_factors.append(preprocessing_res[1])
        X[agent_name] = preprocessing_res[2]
    print("Number of used samples : ", X.shape[0])
    print("Agent Normalizing Factors : ", agent_normalizing_factors)
    print("Agent Offset : ", agent_offset)
    print("Number of used samples : ", X.shape[0])
    # Construct optimizer
    bounds = config["bounds"]
    params = config["params"]
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    func = lambda params: negativeLikelihood(
        params,
        sub_X,
        sub_Y,
        config["agents"],
        return_trajectory=False
    )
    subset_index = np.arange(window, len(Y) - window)
    all_coeff = []
    all_correct_rate = []
    all_success = []
    # at_cross_index = np.where(X.at_cross.values)[0]
    # at_cross_accuracy = []
    # Moving the window
    for index in subset_index:
        print("Window at {}...".format(index))
        sub_X = X[index - window:index + window]
        sub_Y = Y[index - window:index + window]
        # optimization in the window
        is_success = False
        retry_num = 0
        while not is_success and retry_num < config["maximum_try"]:
            res = scipy.optimize.minimize(
                func,
                x0 = params,
                method="SLSQP",
                bounds=bounds,
                tol=1e-5,
                constraints = cons
            )
            is_success = res.success
            if not is_success:
                print("Fail, retrying...")
                retry_num += 1
        all_success.append(is_success)
        # Correct rate
        _, estimated_prob = negativeLikelihood(
            res.x,
            sub_X,
            sub_Y,
            config["agents"],
            return_trajectory = True
        )
        # estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        estimated_dir = np.array([makeChoice(each) for each in estimated_prob])
        true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
        correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
        # Coefficient and accuracy for decision point
        # if index in at_cross_index:
        #     at_cross_accuracy.append((index, res.x, correct_rate))
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # print("Average Correct Rate (Decision Position): {}".format(np.mean([each[2] for each in at_cross_accuracy])))
    # Save estimated agent weights
    if save_res:
        type = "_".join(config['agents'])
        np.save("{}-agent_weight-window{}-{}-new_agent.npy".format(config["method"], window, type), all_coeff)
        np.save("{}-is_success-window{}-{}-new_agent.npy".format(config["method"], window, type), all_success)
        # np.save("{}-at_cross_accuracy-window{}-{}-new_agent.npy".format(config["method"], window, type), at_cross_accuracy)


# ===================================
#         VISUALIZATION
# ===================================
def plotWeightVariation(all_agent_weight, window, is_success = None):
    # Determine agent names
    agent_name = ["Global", "Local", "Optimistic", "Pessimistic", "Suicide", "Planned Hunting"]
    agent_color = ["red", "blue", "green", "cyan", "magenta", "black"]
    # Plot weight variation
    all_coeff = np.array(all_agent_weight)
    if is_success is not None:
        for index in range(1, is_success.shape[0]):
            if not is_success[index]:
                all_agent_weight[index] = all_agent_weight[index - 1]
    # Noamalize
    # all_coeff = all_coeff / np.max(all_coeff)
    for index in range(all_coeff.shape[0]):
        all_coeff[index] = all_coeff[index] / np.sum(all_coeff[index])
        # all_coeff[index] = all_coeff[index] / np.linalg.norm(all_coeff[index])
    for index in range(6):
        plt.plot(all_coeff[:, index], color = agent_color[index], ms = 3, lw = 5,label = agent_name[index])
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize = 15)
    plt.xlim(0, all_coeff.shape[0] - 1)
    x_ticks = list(range(0, all_coeff.shape[0], 10))
    if (all_coeff.shape[0] - 1) not in x_ticks:
        x_ticks.append(all_coeff.shape[0] - 1)
    x_ticks = np.array(x_ticks)
    plt.xticks(x_ticks, x_ticks + window, fontsize=20)
    plt.xlabel("Time Step", fontsize = 20)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=15, ncol=6)
    plt.show()




if __name__ == '__main__':
    # Configurations
    pd.options.mode.chained_assignment = None
    config = {
        # Filename
        "data_filename": "../game_data/multi_agent/mixed/trial2/mixed_record.csv",
        # Testing data filename
        "testing_data_filename": "../game_data/multi_agent/mixed/trial2/mixed_record.csv",
        # Method: "MLE" or "MEE"
        "method": "MLE",
        # Only making decisions when necessary
        "only_necessary": False,
        # The number of samples used for estimation: None for using all the data
        "clip_samples": None,
        # The window size
        "window": 10,
        # Maximum try of estimation, in case the optimization will fail
        "maximum_try": 5,
        # Loss function (required when method = "MEE"): "l2-norm" or "cross-entropy"
        "loss-func": "l2-norm",
        # Initial guess of parameters
        "params": [1, 1, 1, 1, 1, 1],
        # Bounds for optimization
        "bounds": [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]],
        # Agents: at least one of "global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting".
        "agents": ["global", "local", "optimistic", "pessimistic", "suicide", "planned_hunting"],
    }

    # ============ ESTIMATION =============
    # MLE(config)

    # ============ MOVING WINDOW =============
    # movingWindowAnalysis(config, save_res = True) #TODO: read data

    # ============ PLOTTING =============
    # Load the log of moving window analysis; log files are created in the analysis
    agent_weight = np.load("MLE-agent_weight-window10-global_local_optimistic_pessimistic_suicide_planned_hunting-new_agent.npy")
    is_success = np.load("MLE-is_success-window10-global_local_optimistic_pessimistic_suicide_planned_hunting-new_agent.npy")
    plotWeightVariation(agent_weight, config["window"], is_success)