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
#         Pre-Computed Variable
# =========================================

# Read the map adjacent dict
# TODO: recompute for the new map
adjacent_data = pd.read_csv("extracted_data/adjacent_map.csv")
for each in ['pos', 'left', 'right', 'up', 'down']:
    adjacent_data[each] = adjacent_data[each].apply(lambda x : eval(x) if not isinstance(x, float) else np.nan)

# The distance between two positions on the map
# TODO: recompute for the new map
locs_df = pd.read_csv("../common_data/dij_distance_map.csv")[["pos1", "pos2", "dis"]]
locs_df.pos1, locs_df.pos2= (
    locs_df.pos1.apply(eval),
    locs_df.pos2.apply(eval)
)

# Reward amount for every type of reward
reward_amount = {
    1:3,
    2:4,
    3:3,
    4:5,
    5:8,
    6:12,
    7:17
}

print("Finished all the pre-computation and initializations.")

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


