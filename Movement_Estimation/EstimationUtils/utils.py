'''
Description:
    Some utility functions for the package "FeatureExtractor".
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date: 
    Mar. 28 2020
'''
import numpy as np
import pandas as pd

def tuple_list(l):
    return [tuple(a) for a in l]

def oneHot(val,val_list):
    onehot_vec = [0. for each in val_list]
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

