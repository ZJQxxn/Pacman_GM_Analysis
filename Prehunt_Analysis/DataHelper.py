'''
Description:
    Select out data for prehunt analysis.

uthor:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/4/3
'''

import pandas as pd


def selectPreHuntData(feature_file):
    all_data = pd.read_csv(feature_file)
    pre_hunt_data = all_data[(all_data.status_h1 == "prehunt") |  (all_data.status_h2 == "prehunt")]
    pre_hunt_data.to_csv("extracted_data/prehunt_data.csv")


if __name__ == '__main__':
    new_data_file = '../common_data/df_total_new.csv'
    selectPreHuntData(new_data_file)