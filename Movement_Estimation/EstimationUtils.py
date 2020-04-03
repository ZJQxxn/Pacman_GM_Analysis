'''
Description:
    Some utility functions for the movement estimation.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date: 
    Mar. 28 2020
'''
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point


# ==================================================
#            All the Util Functions
# ==================================================
def tuple_list(l):
    return [tuple(a) for a in l]

def oneHot(val,val_list):
    # TODO: explanations
    onehot_vec = [0. for each in val_list]
    if isinstance(val, list):
        for each in val:
            if each in val_list:
                onehot_vec[val_list.index(each)] = 1
    else:
        if val in val_list:
            onehot_vec[val_list.index(val)] = 1
    return np.array(onehot_vec)

def relative_dir(destination_pos, pacman_pos):
    '''
    Determine the direction of "destination_pos" with respect to "pacman_pos".
    :param destination_pos: A 2-tuple denoting the position of destination.
    :param pacman_pos: A 2-tuple denoting the Pacman position.
    :return: A list denoting the direction. Specially, for global direction, "same" denotes two positions are 
             located in the same region. 
    '''
    l = []
    dir_array = np.array(destination_pos) - np.array(pacman_pos)
    if 0 == dir_array[0] and 0 == dir_array[1]:
        l.append("same")
        return l
    if dir_array[0] > 0:
        l.append("right")
    elif dir_array[0] < 0:
        l.append("left")
    else:
        pass
    if dir_array[1] > 0:
        l.append("down")
    elif dir_array[1] < 0:
        l.append("up")
    else:
        pass
    return l


# ==================================================
#            All the Initializations
# ==================================================
# Totally 11 regions
polys = [
    [[2, 5], [2, 12], [7, 12], [7, 5]],
    [[8, 5], [8, 14], [14, 14], [14, 5]],
    [[27, 5], [22, 5], [22, 12], [27, 12]],
    [[21, 5], [15, 5], [15, 14], [21, 14]],
    [[2, 13], [7, 13],[7,15],[9, 15], [9, 23], [2, 23]],
    [[20, 15], [20, 23], [29, 23], [29, 13],[22,13],[22,15]],
    [[10, 15], [10, 23], [19, 23], [19, 15]],
    [[2, 24], [2, 34], [7, 34], [7, 24]],
    [[8, 24], [8, 34], [14, 34], [14, 24]],
    [[27, 24], [22, 24], [22, 34], [27, 34]],
    [[21, 24], [15, 24], [15, 34], [21, 34]]
]
poly_ext = [Polygon(p).buffer(0.001) for p in polys]

# The direction relationship between regions
region_relation = {
    1: {"up": np.nan, "down":[5], "left": np.nan, "right": [2]},
    2: {"up": np.nan, "down":[4, 5], "left": [1], "right": [3]},
    3: {"up": np.nan, "down":[5, 6], "left": [2], "right": [4]},
    4: {"up": np.nan, "down": [6], "left": [23], "right": np.nan},
    5: {"up": [1, 2], "down": [8, 9], "left": np.nan, "right": [2, 6, 9]},
    6: {"up": [2, 3], "down": [9, 10], "left": [5], "right": [7]},
    7: {"up": [3, 4], "down": [10, 11], "left": [3, 6, 10], "right": np.nan},
    8: {"up": [5], "down": np.nan, "left": np.nan, "right": [9]},
    9: {"up": [5, 6], "down": np.nan, "left": [8], "right": [10]},
    10: {"up": [6, 7], "down": np.nan, "left": [9], "right": [11]},
    11: {"up": [7], "down": np.nan, "left": [10], "right": np.nan}
}


def determine_centroid(poly_list, pos):
    '''
    Given a position 2-tuple, determine the centroid for eits region.  
    :param poly_list: A list of "shapely.geometry.Polygon" denotes each region. 
    :param pos: The position with shape of a 2-tuple.
    :return: A list contains the centroid point and region index (1-11).
    '''
    for index, p in enumerate(poly_list):
        if p.contains(Point(pos)):
            return (round(p.centroid.xy[0][-1], 1), round(p.centroid.xy[1][-1], 1))

def determine_region(poly_list, pos):
    '''
    Given a position 2-tuple, determine the region index for the position.  
    :param poly_list: A list of "shapely.geometry.Polygon" denotes each region. 
    :param pos: The position with shape of a 2-tuple.
    :return: A list contains the centroid point and region index (1-11).
    '''
    for index, p in enumerate(poly_list):
        if p.contains(Point(pos)):
            return index + 1


map_info = pd.read_csv("../common_data/map_info_brian.csv")
map_info['pos'] = map_info['pos'].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
map_info["pos_global"] = [
    determine_centroid(poly_ext, pos)for pos in map_info.pos.values
]
map_info["region_index"] = [
    determine_region(poly_ext, pos) for pos in map_info.pos.values
]
# map_info = map_info[["pos", "pos_global", "region_index"]]
#TODO: need size of each region 

# ====================================================================
locs_df = pd.read_csv("../common_data/dij_distance_map.csv")
locs_df.pos1, locs_df.pos2, locs_df.path = (
    locs_df.pos1.apply(eval),
    locs_df.pos2.apply(eval),
    locs_df.path.apply(eval)
)

