'''
Description:
    Extract data from all_data file.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
 
Date:
    23 July 2020 
'''

import pickle
import pandas as pd
import numpy as np


# 每一段index都是df_total的index，注意：在对df_total进行处理（尤其是merge）的时候，不要改变其index！！！！
# accidentally hunting就是all_data.pickle中的['cons_list_accident']
# planned hunting就是all_data.pickle中的['cons_list_plan']


# 这个function输出的第一个variable就是suicide_list
def generate_suicide_normal(df_total):
    select_last_num = 100
    suicide_normal = (
        df_total.reset_index()
        .merge(
            (df_total.groupby("file")["label_suicide"].sum() > 0)
            .rename("suicide_trial")
            .reset_index(),
            on="file",
            how="left",
        )
        .sort_values(by="level_0")
        .groupby(["file", "suicide_trial"])
        .apply(lambda x: x.level_0.tail(select_last_num).tolist())
        .reset_index()
    )
    suicide_lists = suicide_normal[suicide_normal["suicide_trial"] == True][0]
    normal_lists = suicide_normal[suicide_normal["suicide_trial"] == False][0]
    return suicide_lists, normal_lists


def obtain_evade_list(df_total):
    evade_indexes = [
        list(i)[0]
        for i in consecutive_groups(df_total[df_total.label_evade == 1].index) # TODO: just the label of evading
    ]
    return evade_indexes