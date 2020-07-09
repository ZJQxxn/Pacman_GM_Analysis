'''
Description:
    MLE parameter estimation for multi-agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    July 1 2020
'''

import pickle
import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
import scipy.optimize
from sklearn.model_selection import train_test_split

import sys
# from MultiAgentInteractor import MultiAgentInteractor
sys.path.append('./')
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount
from PathTreeConstructor import PathTree
from LazyAgent import LazyAgent
from RandomAgent import RandomAgent


# ===========================================================
#               UTILITY FUNCTIONS
# ===========================================================

# Global variables
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


# ===========================================================
#             MAXIMUM LIKELIHOOD ESTIMATION
# ===========================================================
def negativeLogLikelihood(param, all_data, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    # Parameters
    global_depth = 15
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    local_depth = 5
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    agent_weight = [param[0], param[1], param[2], param[3]]
    # Compute log likelihood
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    # for index in range(num_samples):
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):  # TODO: use only a part of samples for efficiency for now
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos)
        energizer_data = eval(each.energizers)
        bean_data = eval(each.beans)
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values # TODO: check whether same as ``ifscared''
        reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        # Construct agents
        global_agent = PathTree(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth=global_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr
        )
        local_agent = PathTree(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth=local_depth,
            ghost_attractive_thr=local_ghost_attractive_thr,
            fruit_attractive_thr=local_fruit_attractive_thr,
            ghost_repulsive_thr=local_ghost_repulsive_thr
        )
        lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir, loop_count, max_loop=5)
        random_agent = RandomAgent(adjacent_data, cur_pos, last_dir, None)
        # Estimation
        agent_estimation = np.zeros((4, 4))
        _, _, global_best_path = global_agent.construct()
        _, _,local_best_path = local_agent.construct()
        lazy_next_dir, not_turn = lazy_agent.nextDir()
        if not_turn:
            loop_count += 1
        random_next_dir = random_agent.nextDir()
        agent_estimation[:, 0] = oneHot(global_best_path[0][1])
        agent_estimation[:, 1] = oneHot(local_best_path[0][1])
        agent_estimation[:, 2] = oneHot(lazy_next_dir)
        agent_estimation[:, 3] = oneHot(random_next_dir)
        dir_prob = agent_estimation @ agent_weight
        best_dir_index = np.argmax(dir_prob)
        last_dir = dir_list[best_dir_index]
        exp_prob = np.exp(dir_prob)
        log_likelihood = dir_prob[best_dir_index] - np.log(np.sum(exp_prob))
        nll += (-log_likelihood)
        estimation_prob_trajectory.append(exp_prob / np.sum(exp_prob))
    # print('Finished')
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def MLE(data_filename, map_filename, loc_distance_filename, useful_num_samples = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    # Load experiment data
    # with open(data_filename, 'rb') as file:
    #     all_data = pickle.load(file)
    with open(data_filename, 'r') as file:
        all_data = pd.read_csv(file)
    print("Number of sanmples : ", all_data.shape[0])
    # Optimization
    print("Number of used samples : ", useful_num_samples)
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1})
    func = lambda parameter: negativeLogLikelihood(parameter, all_data, adjacent_data, locs_df, reward_amount,
                                                   useful_num_samples = useful_num_samples)
    is_success = False
    retry_num = 0
    while not is_success and retry_num < 10:
        res = scipy.optimize.minimize(
            func,
            x0 = params,
            method = "SLSQP",
            bounds = bounds,
            tol = 1e-8,
            constraints = cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = negativeLogLikelihood(res.x, all_data, adjacent_data, locs_df, reward_amount,
                                              useful_num_samples = useful_num_samples, return_trajectory = True)
    true_dir = all_data.pacman_dir.apply(
            lambda x: np.argmax([float(each) for each in x.strip('[]').split(' ')]) if not isinstance(x, float) else -1
        ).values[:useful_num_samples]
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))


# ===========================================================
#               MINIMUM ERROR ESTIMATION
# ===========================================================
def estimationError(param, all_data, adjacent_data, locs_df, reward_amount, useful_num_samples = None, return_trajectory = False):
    # Parameters
    global_depth = 15
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 12
    local_depth = 5
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    agent_weight = [param[0], param[1], param[2], param[3]]
    # True probability
    true_prob = []
    for index in range(all_data.pacman_dir.values.shape[0]):
        each = all_data.pacman_dir.values[index]
        each = each.strip('[]').split(' ')
        while '' in each: # For the weird case that '' might exist in the split list
            each.remove('')
        true_prob.append([float(e) for e in each])
    true_prob = np.array(true_prob)
    # Compute log likelihood
    nll = 0  # negative log likelihood
    estimation_prob_trajectory = []
    num_samples = all_data.shape[0]
    last_dir = None
    loop_count = 0
    # for index in range(num_samples):
    useful_num_samples = useful_num_samples if useful_num_samples is not None else num_samples
    for index in range(useful_num_samples):  # TODO: use only a part of samples for efficiency for now
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos)
        energizer_data = eval(each.energizers)
        bean_data = eval(each.beans)
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values # TODO: check whether same as ``ifscared''
        reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        # Construct agents
        global_agent = PathTree(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth=global_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr
        )
        local_agent = PathTree(
            adjacent_data,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            depth=local_depth,
            ghost_attractive_thr=local_ghost_attractive_thr,
            fruit_attractive_thr=local_fruit_attractive_thr,
            ghost_repulsive_thr=local_ghost_repulsive_thr
        )
        lazy_agent = LazyAgent(adjacent_data, cur_pos, last_dir, loop_count, max_loop=5)
        random_agent = RandomAgent(adjacent_data, cur_pos, last_dir, None)
        # Estimation
        agent_estimation = np.zeros((4, 4))
        _, _, global_best_path = global_agent.construct()
        _, _,local_best_path = local_agent.construct()
        lazy_next_dir, not_turn = lazy_agent.nextDir()
        if not_turn:
            loop_count += 1
        random_next_dir = random_agent.nextDir()
        agent_estimation[:, 0] = oneHot(global_best_path[0][1])
        agent_estimation[:, 1] = oneHot(local_best_path[0][1])
        agent_estimation[:, 2] = oneHot(lazy_next_dir)
        agent_estimation[:, 3] = oneHot(random_next_dir)
        dir_prob = agent_estimation @ agent_weight
        error = np.linalg.norm(dir_prob - true_prob[index])
        nll += error
        # estimation_prob_trajectory.append(exp_prob / np.sum(exp_prob))
        estimation_prob_trajectory.append(dir_prob)
    # print('Finished')
    if not return_trajectory:
        return nll
    else:
        return (nll, estimation_prob_trajectory)


def MEE(data_filename, map_filename, loc_distance_filename, useful_num_samples = None):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    # Load experiment data
    # with open(data_filename, 'rb') as file:
    #     all_data = pickle.load(file)
    with open(data_filename, 'r') as file:
        all_data = pd.read_csv(file)
    print("Number of samples : ", all_data.shape[0])
    # Optimization
    if useful_num_samples is None:
        useful_num_samples = all_data.shape[0]
    print("Number of used samples : ", useful_num_samples)
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1})
    func = lambda parameter: estimationError(parameter, all_data, adjacent_data, locs_df, reward_amount,
                                                   useful_num_samples = useful_num_samples)
    is_success = False
    retry_num = 0
    while not is_success and retry_num < 10:
        res = scipy.optimize.minimize(
            func,
            x0 = params,
            method = "SLSQP",
            bounds = bounds,
            tol = 1e-8,
            constraints = cons
        )
        is_success = res.success
        if not is_success:
            retry_num += 1
            print("Failed, retrying...")
    print("Initial guess : ", params)
    print("Estimated Parameter : ", res.x)
    print(res)
    # Estimation
    _, estimated_prob = estimationError(res.x, all_data, adjacent_data, locs_df, reward_amount,
                                              useful_num_samples = useful_num_samples, return_trajectory = True)
    true_dir = []
    for index in range(all_data.pacman_dir.values.shape[0]):
        each = all_data.pacman_dir.values[index]
        each = each.strip('[]').split(' ')
        while '' in each:  # For the weird case that '' might exist in the split list
            each.remove('')
        true_dir.append(np.argmax([float(e) for e in each]))
    true_dir = np.array(true_dir)
    estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir)
    print("Correct rate : ", correct_rate / len(true_dir))


# ===========================================================
#               AGENT WEIGHT ANALYSIS
# ===========================================================
def constructDatasetFromCSV(filename, clip = None):
    # Read data and pre-processing
    agent_dir = pd.read_csv(filename)
    overall_dir = []
    for index in range(agent_dir.pacman_dir.values.shape[0]):
        each = agent_dir.pacman_dir.values[index]
        each = each.strip('[]').split(' ')
        while '' in each: # For the weird case that '' might exist in the split list
            each.remove('')
        overall_dir.append(np.argmax([float(e) for e in each]))
    overall_dir = np.array(overall_dir)
    # Construct the dataset
    if clip is not None and clip > agent_dir.shape[0]:
        print("Warning: requir more data than you have. Use the entire dataset by default.")
        clip = None
    X = agent_dir if clip is None else agent_dir[:clip]
    Y = overall_dir if clip is None else overall_dir[:clip]
    return X, Y


def movingWindowAnalysis(X, Y, map_filename, loc_distance_filename, window = 100):
    # Load pre-computed data
    adjacent_data = readAdjacentMap(map_filename)
    locs_df = readLocDistance(loc_distance_filename)
    reward_amount = readRewardAmount()
    print("Finished pre-processing!")
    print("Start optimizing...")
    print("="*15)
    # Construct constraints for the optimizer
    bounds = [[0, 1], [0, 1], [0, 1], [0, 1]]
    params = np.array([0.0, 0.0, 0.0, 0.0])
    cons = []  # construct the bounds in the form of constraints
    for par in range(len(bounds)):
        l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
        u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)
    cons.append({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1})
    # The indices
    subset_index = np.arange(window, len(Y) - window)
    all_coeff = []
    all_correct_rate = []
    # Moving the window
    for index in subset_index:
        # if index % 20 == 0:
        print("Window at {}...".format(index))
        sub_X = X[index - window:index + window]
        sub_Y = Y[index - window:index + window]
        # X_train, X_test, Y_train, Y_test = train_test_split(sub_X, sub_Y, test_size=0.2)
        # Optimize with minimum error estimation (MEE)
        func = lambda parameter: estimationError(parameter, sub_X, adjacent_data, locs_df, reward_amount)
        is_success = False
        retry_num = 0
        while not is_success and retry_num < 10:
            res = scipy.optimize.minimize(
                func,
                x0 = params,
                method = "SLSQP",
                bounds = bounds,
                tol = 1e-8,
                constraints = cons
            )
            is_success = res.success
            if not is_success:
                retry_num += 1
        # Make estimations on the testing dataset
        _, estimated_prob = estimationError(res.x, sub_X, adjacent_data, locs_df, reward_amount, return_trajectory=True)
        estimated_dir = np.array([np.argmax(each) for each in estimated_prob])
        correct_rate = np.sum(estimated_dir == sub_Y) / len(sub_Y)
        all_correct_rate.append(correct_rate)
        # The coefficient
        all_coeff.append(res.x)
    print("Average Coefficient: {}".format(np.mean(all_coeff, axis=0)))
    print("Average Correct Rate: {}".format(np.mean(all_correct_rate)))
    # Plot weight variation
    all_coeff = np.array(all_coeff)
    plt.stackplot(np.arange(all_coeff.shape[0]),
                  all_coeff[:, 3],  # random agent
                  all_coeff[:, 2],  # lazy agent
                  all_coeff[:, 1],  # local agent
                  all_coeff[:, 0],  # global agent
                  labels=["Random Agent", "Lazy Agent", "Local Agent", "Global Agent"])
    plt.ylim(0, 1.0)
    plt.ylabel("Agent Percentage (%)", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, all_coeff.shape[0])
    plt.xlabel("Time Step", fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    plt.clf()
    plt.plot(all_coeff[:, 0], "o-", label="Global Agent", ms=2, lw=0.5)
    plt.plot(all_coeff[:, 1], "o-", label="Local Agent", ms=2, lw=0.5)
    plt.plot(all_coeff[:, 2], "o--", label="Lazy Agent", ms=2, lw=0.5)
    plt.plot(all_coeff[:, 3], "o--", label="Random Agent", ms=2, lw=0.5)
    plt.ylabel("Agent Weight ($\\beta$)", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, all_coeff.shape[0])
    plt.xlabel("Time Step", fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.show()

    plt.clf()
    plt.title("Correct Rate vs. Time Step", fontsize=20)
    plt.plot(all_correct_rate, "d-", label="Correct Rate", ms=3, lw=2)
    plt.ylabel("Correct Rate (%)", fontsize=20)
    plt.ylim(0, 1.1)
    plt.yticks(fontsize=20)
    plt.xlim(0, len(all_correct_rate))
    plt.xlabel("Time Step", fontsize=20)
    plt.xticks(fontsize=20)
    # plt.legend(fontsize=20)
    plt.show()

    # Save estimated agent weights
    np.save("MEE-agent-weight.npy", all_coeff)




if __name__ == '__main__':
    # data_filename = "extracted_data/test_data.pkl"
    data_filename = "stimulus_data/stimulus-switch/diary.csv"
    map_filename = "extracted_data/adjacent_map.csv"
    loc_distance_filename = "extracted_data/dij_distance_map.csv"

    # # MLE (maximum likelihood estimation)
    # # Note: The performance of MEE is much better.
    # print("="*10, " MLE ", "="*10)
    # MLE(data_filename, map_filename, loc_distance_filename, useful_num_samples = 100)

    # # MEE (minimum error estimation)
    # print("=" * 10, " MEE ", "=" * 10)
    # MEE(data_filename, map_filename, loc_distance_filename, useful_num_samples = 100)

    # Moving Window Analysis with MEE
    X, Y = constructDatasetFromCSV(data_filename, clip = None)
    movingWindowAnalysis(X, Y, map_filename, loc_distance_filename, window = 50)