'''
Description:
    Compare simulated labels with hand-crafted labels.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    17 Dec. 2020
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

sys.path.append("./")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber
from PathTreeAgent import PathTree
from SimpleGlobalAgent import SimpleGlobal
from SimpleEnergizerAgent import SimpleEnergizer
from GhostAgent import GhostAgent

# ===================================
#         UTILITY FUNCTION
# ===================================
dir_list = ['left', 'right', 'up', 'down']




# ===================================
#       INDIVIDUAL ESTIMATION
# ===================================
def _readData(filename):
    '''
    Read data for pre-estimation.
    '''
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    all_data = all_data.reset_index(drop=True)
    return all_data


def _readSimulationData(filename):
    #TODO: this one!
    '''
    Read data for pre-estimation.
    '''
    with open(filename, "rb") as file:
        all_data = pickle.load(file)
    all_data = all_data.reset_index(drop=True)
    return all_data


def _readAuxiliaryData():
    # Load pre-computed data
    adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    adjacent_path = readAdjacentPath("extracted_data/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    return adjacent_data, locs_df, adjacent_path, reward_amount


def _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount):
    # Randomness and laziness
    randomness_coeff = 1.0
    # laziness_coeff = 1.0
    # Configuration (for global agent)
    global_depth = 15
    ignore_depth = 10
    global_ghost_attractive_thr = 34
    global_fruit_attractive_thr = 34
    global_ghost_repulsive_thr = 34
    # Configuration (for local agent)
    local_depth = 5
    local_ghost_attractive_thr = 5
    local_fruit_attractive_thr = 5
    local_ghost_repulsive_thr = 5
    # Configuration (for optimistic agent)
    optimistic_depth = 5
    optimistic_ghost_attractive_thr = 5
    optimistic_fruit_attractive_thr = 5
    optimistic_ghost_repulsive_thr = 5
    # Configuration (for pessimistic agent)
    pessimistic_depth = 10
    pessimistic_ghost_attractive_thr = 10
    pessimistic_fruit_attractive_thr = 10
    pessimistic_ghost_repulsive_thr = 10
    # Configuration (fpr planne hunting agent)
    ghost_attractive_thr = 15
    energizer_attractive_thr = 15
    beans_attractive_thr = 5
    # Configuration (for suicide agent)
    suicide_depth = 10
    suicide_ghost_attractive_thr = 10
    suicide_fruit_attractive_thr = 10
    suicide_ghost_repulsive_thr = 10
    # Configuration (flast direction)
    last_dir = all_data.pacman_dir.values
    last_dir[np.where(pd.isna(last_dir))] = None
    # Direction estimation
    global_estimation = []
    local_estimation = []
    pessimistic_estimation = []
    suicide_estimation = []
    planned_hunting_estimation = []
    # Q-value (utility)
    global_Q = []
    local_Q = []
    blinky_Q = []
    clyde_Q = []
    planned_hunting_Q = []
    num_samples = all_data.shape[0]
    print("Sample Num : ", num_samples)
    estimated_index = []
    for index in range(num_samples):
        if 0 == (index + 1) % 20:
            print("Finished estimation at {}".format(index + 1))
        # Extract game status and Pacman status
        each = all_data.iloc[index]
        cur_pos = eval(each.pacmanPos) if isinstance(each.pacmanPos, str) else each.pacmanPos
        # The tunnel
        if cur_pos == (0, 18):
            cur_pos = (1, 18)
        if cur_pos == (29, 18):
            cur_pos = (28, 18)
        adj_num = sum([isinstance(adjacent_data[cur_pos][each], tuple) for each in adjacent_data[cur_pos]])
        if adj_num > 2:
            laziness_coeff = 0.1
        else:
            laziness_coeff = 0.5
        energizer_data = eval(each.energizers) if isinstance(each.energizers, str) else each.energizers
        bean_data = eval(each.beans) if isinstance(each.beans, str) else each.beans
        ghost_data = np.array([eval(each.ghost1_pos), eval(each.ghost2_pos)]) \
            if "ghost1_pos" in all_data.columns.values or "ghost2_pos" in all_data.columns.values \
            else np.array([each.ghost1Pos, each.ghost2Pos])
        ghost_status = each[["ghost1_status", "ghost2_status"]].values \
            if "ghost1_status" in all_data.columns.values or "ghost2_status" in all_data.columns.values \
            else np.array([each.ifscared1, each.ifscared1])
        if "fruit_type" in all_data.columns.values:
            reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
        else:
            reward_type = each.Reward
        if "fruit_pos" in all_data.columns.values:
            fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
        else:
            fruit_pos = each.fruitPos
        # Global agents
        global_agent = SimpleGlobal(
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
            last_dir[index],
            depth=global_depth,
            ignore_depth=ignore_depth,
            ghost_attractive_thr=global_ghost_attractive_thr,
            fruit_attractive_thr=global_fruit_attractive_thr,
            ghost_repulsive_thr=global_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 1.0,
            risk_coeff = 0.0
        )
        global_result = global_agent.nextDir(return_Q=True)
        global_estimation.append(global_result[0])
        global_Q.append(global_result[1])
        # Local estimation
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
            last_dir[index],
            depth = local_depth,
            ghost_attractive_thr = local_ghost_attractive_thr,
            fruit_attractive_thr = local_fruit_attractive_thr,
            ghost_repulsive_thr = local_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 1.0,
            risk_coeff = 0.0
        )
        local_result = local_agent.nextDir(return_Q=True)
        local_estimation.append(local_result[0])
        local_Q.append(local_result[1])
        # Pessimistic agent
        blinky_agent = GhostAgent(
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            "red",
            depth = pessimistic_depth,
            ghost_attractive_thr = pessimistic_ghost_attractive_thr,
            fruit_attractive_thr = pessimistic_fruit_attractive_thr,
            ghost_repulsive_thr = pessimistic_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 0.0,
            risk_coeff = 1.0
        )
        blinky_result = blinky_agent.nextDir(return_Q=True)
        blinky_Q.append(blinky_result[1])

        clyde_agent = GhostAgent(
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            bean_data,
            ghost_data,
            reward_type,
            fruit_pos,
            ghost_status,
            last_dir[index],
            "yellow",
            depth = pessimistic_depth,
            ghost_attractive_thr = pessimistic_ghost_attractive_thr,
            fruit_attractive_thr = pessimistic_fruit_attractive_thr,
            ghost_repulsive_thr = pessimistic_ghost_repulsive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff,
            reward_coeff = 0.0,
            risk_coeff = 1.0
        )
        clyde_result = clyde_agent.nextDir(return_Q=True)
        # pessimistic_estimation.append(pessimistic_blinky_result[0])
        clyde_Q.append(clyde_result[1])
        # Planned hunting agent
        planned_hunting_agent = SimpleEnergizer(
            adjacent_data,
            adjacent_path,
            locs_df,
            reward_amount,
            cur_pos,
            energizer_data,
            ghost_data,
            ghost_status,
            bean_data,
            last_dir[index],
            ghost_attractive_thr=ghost_attractive_thr,
            energizer_attractive_thr = energizer_attractive_thr,
            beans_attractive_thr = beans_attractive_thr,
            randomness_coeff = randomness_coeff,
            laziness_coeff = laziness_coeff
        )
        planned_hunting_result = planned_hunting_agent.nextDir(return_Q=True)
        planned_hunting_estimation.append(planned_hunting_result[0])
        planned_hunting_Q.append(planned_hunting_result[1])
    # Assign new columns
    print("Estimation length : ", len(global_estimation))
    print("Data Shape : ", all_data.shape)
    all_data["global_Q"] = np.tile(np.nan, num_samples)
    all_data["global_Q"] = all_data["global_Q"].apply(np.array)
    all_data["global_Q"] = global_Q
    all_data["local_Q"] = np.tile(np.nan, num_samples)
    all_data["local_Q"] = all_data["local_Q"].apply(np.array)
    all_data["local_Q"] = local_Q
    all_data["blinky_Q"] = np.tile(np.nan, num_samples)
    all_data["blinky_Q"] = all_data["blinky_Q"].apply(np.array)
    all_data["blinky_Q"] = blinky_Q
    all_data["clyde_Q"] = np.tile(np.nan, num_samples)
    all_data["clyde_Q"] = all_data["clyde_Q"].apply(np.array)
    all_data["clyde_Q"] = clyde_Q
    all_data["planned_hunting_Q"] = np.tile(np.nan, num_samples)
    all_data["planned_hunting_Q"] = all_data["planned_hunting_Q"].apply(np.array)
    all_data["planned_hunting_Q"] = planned_hunting_Q
    print("\n")
    print("Direction Estimation :")
    print("\n")
    print("Q value :")
    print(all_data[["global_Q", "local_Q", "blinky_Q", "clyde_Q", "planned_hunting_Q"]].iloc[:5])
    return all_data


def preEstimation():
    pd.options.mode.chained_assignment = None
    # Individual Estimation
    print("=" * 15, " Individual Estimation ", "=" * 15)
    adjacent_data, locs_df, adjacent_path, reward_amount = _readAuxiliaryData()
    print("Finished reading auxiliary data.")
    filename_list = [
        # "../common_data/trial/5_trial_data.pkl",
        # "../common_data/transition/global_to_local.pkl",
        # "../common_data/transition/local_to_global.pkl",
        # "../common_data/transition/local_to_evade.pkl",
        # "../common_data/transition/evade_to_local.pkl",
        # "../common_data/transition/local_to_planned.pkl",
        # "../common_data/transition/local_to_suicide.pkl",
        # "../common_data/transition/local_to_accidental.pkl",
        # "../common_data/transition/graze_to_hunt.pkl",
        # "../common_data/trial/500_trial_data.pkl",
        # "../common_data/single_trial/14-1-Patamon-14-Jun-data.pkl",
        # "../common_data/trial/100_trial_data_new.pkl",
        # "../common_data/single_trial/5_trial-data_for_comparison.pkl"
        # "../common_data/simulation/single_trial_record.pkl",
        # "../common_data/trial/500_trial_data_Omega.pkl",
        "../common_data/trial/100_trial_data_Omega.pkl",
        "../common_data/trial/100_trial_data_Patamon.pkl",
        # "../common_data/trial/8000_trial_data_Omega.pkl",
        # "../common_data/trial/7000_trial_data_Patamon.pkl",
        # "../common_data/trial/test_planned_trial_data_Omega.pkl",
        # "../common_data/trial/test_suicide_trial_data_Omega.pkl",

        # "../common_data/trial/new_100_trial_data_Omega.pkl",
        # "../common_data/trial/accidental_100_trial_data_Omega.pkl",
        # "../common_data/trial/suicide_100_trial_data_Omega.pkl",
        # "../common_data/trial/global_100_trial_data_Omega.pkl",

        # "../common_data/trial/PA_AA_data.pkl",

        # For suicide agents
        # "../common_data/trial/9-3-Omega-19-Aug-2019-1.pkl",
        # "../common_data/trial/7-3-Omega-11-Jun-2019-1.pkl",
        # "../common_data/trial/15-6-Patamon-04-Jul-2019-4.pkl",
        # For attack
        # "../common_data/trial/23-1-Omega-05-Aug-2019-1.pkl",
        # "../common_data/trial/27-1-Omega-13-Jun-2019-1.pkl",
        # "../common_data/trial/39-1-Omega-22-Aug-2019-1.pkl",
        # "../common_data/trial/16-2-Omega-16-Jul-2019-1.pkl",
        # "../common_data/trial/25-2-Omega-24-Jun-2019-1.pkl",
    ]
    for filename in filename_list:
        print("-" * 50)
        print(filename)
        all_data = _readData(filename)
        print("Finished reading data.")
        print("Start estimating...")
        all_data = _individualEstimation(all_data, adjacent_data, locs_df, adjacent_path, reward_amount)
        with open("{}/{}-with_Q-ghost.pkl".format(
                "../common_data/transition" if "transition" in filename.split("/") else "../common_data/trial",
                filename.split("/")[-1].split(".")[0]
        ), "wb") as file:
            pickle.dump(all_data, file)
        # with open("../common_data/simulation/single_trial-with_Q.pkl", "wb") as file:
        # # with open("../common_data/trial/{}-one_ghost-with_Q.pkl".format(filename.split("/")[-1].split(".")[0]), "wb") as file:
        #     pickle.dump(all_data, file)
        print("{}-with_Q.pkl saved!".format(filename.split("/")[-1].split(".")[0]))
    pd.options.mode.chained_assignment = "warn"




if __name__ == '__main__':
    # Pre-estimation
    preEstimation()

