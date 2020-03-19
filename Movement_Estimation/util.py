''''
Description:
    Utility tools for the movement estimationa. 

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/19
'''
import numpy as np

def oneHot(val,val_list):
    onehot_vec = [0. for each in val_list]
    onehot_vec[val_list.index(val)] = 1
    return np.array(onehot_vec)