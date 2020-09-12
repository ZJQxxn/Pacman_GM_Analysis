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

def readAdjacentMap(filename):
    '''
    Read in the adjacent info of the map.
    :param filename: File name.
    :return: A dictionary denoting adjacency of the map.
    '''
    adjacent_data = pd.read_csv(filename)
    for each in ['pos', 'left', 'right', 'up', 'down']:
        adjacent_data[each] = adjacent_data[each].apply(lambda x : eval(x) if not isinstance(x, float) else np.nan)
    dict_adjacent_data = {}
    for each in adjacent_data.values:
        dict_adjacent_data[each[1]] = {}
        dict_adjacent_data[each[1]]["left"] = each[2] if not isinstance(each[2], float) else np.nan
        dict_adjacent_data[each[1]]["right"] = each[3] if not isinstance(each[3], float) else np.nan
        dict_adjacent_data[each[1]]["up"] = each[4] if not isinstance(each[4], float) else np.nan
        dict_adjacent_data[each[1]]["down"] = each[5] if not isinstance(each[5], float) else np.nan
    return dict_adjacent_data


if __name__ == '__main__':
    # status_list = get_predefined_states("global")
    # print()

    adjacent_data = readAdjacentMap("/home/qlyang/jiaqi/Pacman-Analysis/Utility_Tree_Analysis/extracted_data/adjacent_map.csv")

    with open("/home/qlyang/Documents/pacman/constants/all_data.pkl", "rb") as file:
        data = pickle.load(file)
    data = data["df_total"]
    print("Columns : ", data.columns.values)
    print("Data Shape : ", data.shape)
    trial_list = np.unique(data.file.values)
    print("Trial Num : ", len(trial_list))
    print("Trial Sample : ", trial_list[:3])


    print("="*30)
    trial_list = trial_list[np.random.choice(len(trial_list), 500, replace = False)]
    is_need = np.where(data.file.apply(lambda x: x in trial_list).values)
    data = data.iloc[is_need]
    print("Data Shape : ", data.shape)
    print("Trial Num : ", len(trial_list))
    print("Trial Sample : ", trial_list[:3])
    at_cross = data.pacmanPos.apply(
        lambda x: (
            False if x not in adjacent_data else
            np.sum(
                [1 if not isinstance(each, float) else 0
                 for each in list(adjacent_data[x].values())]
            ) > 2

        )
    )
    data["at_cross"] = at_cross
    print("Finished processing.")
    with open("partial_data_with_reward_label_cross.pkl", "wb") as file:
        pickle.dump(data, file)
    print("Finished writing.")

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


