'''
Description:
    Functions and some initialization values for the utility tree analysis.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    Apr. 21 2020
'''

import pandas as pd
import numpy as np


# =========================================
#         Read Pre-Computed Variable
# =========================================
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

def readAdjacentPath(filename):
    adjacent_data = pd.read_csv(filename)
    adjacent_data.pos1 = adjacent_data.pos1.apply(lambda x: eval(x))
    adjacent_data.pos2 = adjacent_data.pos2.apply(lambda x: eval(x))
    adjacent_data.path = adjacent_data.path.apply(lambda x: eval(x))
    return adjacent_data[["pos1", "pos2", "path"]]


def readLocDistance(filename):
    '''
    Read in the location distance.
    :param filename: File name.
    :return: A pandas.DataFrame denoting the dijkstra distance between every two locations of the map. 
    '''
    locs_df = pd.read_csv(filename)[["pos1", "pos2", "dis"]]
    locs_df.pos1, locs_df.pos2= (
        locs_df.pos1.apply(eval),
        locs_df.pos2.apply(eval)
    )
    dict_locs_df = {}
    for each in locs_df.values:
        if each[0] not in dict_locs_df:
            dict_locs_df[each[0]] = {}
        dict_locs_df[each[0]][each[1]] = each[2]
    # correct the distance between two ends of the tunnel
    dict_locs_df[(1, 18)][(27, 18)] = 1
    return dict_locs_df


def readRewardAmount():
    '''
    Reward amount for every type of reward
    :return: A dictionary denoting the reward amount of each type of reward.
    '''
    reward_amount = {
        1:1, # bean
        2:10, # energizer
        3:3, # cherry
        4:5, # strawberry
        5:8, # orange
        6:12, # apple
        7:17, # melon
        8:10 # ghost
    }
    return reward_amount

# =========================================
#           Tool Functions
# =========================================

def unitStepFunc(x):
    '''
    The unit step function.
    :param x: x
    :return:     If x > 0, return 1; if x < 0, return 0; if x = 0 , return 0.5.
    '''
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return 0.5


def indicateFunc(x, y):
    '''
    The indication function denotes whether two values are the same.
    :param x: x value
    :param y: y value
    :return: If x == y, returns 1, otherwise returns 0.
    '''
    return int(x == y)


