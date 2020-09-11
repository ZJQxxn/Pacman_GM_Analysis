'''
Description:
    Take a glance about the data.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    20 July 2020
'''

import pandas as pd
import pickle
import numpy as np

def _pos2String(pos):
    return str((pos[1]-1)*28 + pos[0])

def _convertReward(data):
    fruit_type_list = ['C', 'S', 'O', 'A', 'M']
    beans = data.beans
    energizers = data.energizers
    fruit = data.fruitPos
    fruit_type = data.Reward
    # Convert position to str
    if isinstance(beans, float):
        beans = []
    else:
        beans = [_pos2String(each) for each in beans]
    if isinstance(energizers, float):
        energizers = []
    else:
        energizers = [_pos2String(each) for each in energizers]
    if isinstance(fruit, float):
        fruit = []
    else:
        fruit = [_pos2String(fruit), fruit_type_list[fruit_type-3]]
    return [beans, energizers, fruit]

def _convertGhost(data):
    ghost_pos = [_pos2String(data.ghost1Pos), _pos2String(data.ghost2Pos)]
    ghost_status = [data.ifscared1, data.ifscared1]
    scared = np.array(ghost_status) >= 4
    # TODO: the mode ofghosts
    return (ghost_pos, ghost_status, scared)

def _convertPacman(data):
    return _pos2String(data.pacmanPos)


def get_predefined_states(label, num = 20):
    if label not in ["global", "local", "pessimistic", "suicide", "planned_hunting"]:
        raise ValueError("Unknown label {}".format(label))
    # Read data
    with open("./status/{}_status.pkl".format(label), "rb") as file:
        status_data = pickle.load(file)
    sample_index = np.random.choice(status_data.shape[0], num, replace = False)
    status_data = status_data.iloc[sample_index].reset_index()
    status_list = []
    print(status_data.columns.values)
    for index in range(status_data.shape[0]):
        cur_trial = status_data.iloc[index]
        rewards = _convertReward(cur_trial)
        ghosts = _convertGhost(cur_trial)
        pacman = _convertPacman(cur_trial)
        status_list.append([rewards, ghosts, pacman])
    return status_list



if __name__ == '__main__':
    # status_list = get_predefined_states("global")
    # print()
    with open("/home/qlyang/Documents/pacman/constants/labeled_df_total_omega.pkl", "rb") as file:
        data = pickle.load(file)
    print(data.columns.values)
    print(data.shape)
    # data = data.file.apply(lambda x: "Omega" in x)
    # print("With Label : ", print(np.sum(data.values)))
    #
    # with open("df_total_with_reward.pkl", "rb") as file:
    #     data = pickle.load(file)
    # print("With Reward : ", data.shape)

    # data_filename = "/home/qlyang/Documents/pacman/constants/all_data.pkl"
    # with open(data_filename, "rb") as file:
    #     all_data = pickle.load(file)
    # all_data = all_data["rt"]
    # print(all_data.columns.values)
    # print(all_data.shape)


