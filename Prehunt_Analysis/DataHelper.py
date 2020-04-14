'''
Description:
    Select out data for prehunt analysis.

uthor:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/4/3
'''

import pandas as pd
import numpy as np
import sys

sys.path.append("./")
from AnalysisUtils import determine_region, dijkstra_distance
from AnalysisUtils import poly_ext

def _getEatEnergizerPoint(group):
    eat_energizer_point = []
    pre_index = group[0]
    for index in range(1, len(group)):
        cur_index = group[index]
        if cur_index - pre_index == 1:
            pre_index = cur_index
        else:
            eat_energizer_point.append(pre_index + 1)
            pre_index = cur_index
    eat_energizer_point.append(cur_index + 1)
    return eat_energizer_point


def _getHuntLabel(all_data, time_data):
    res = None
    eat_point = None
    for each in time_data.eat_energizer_point:
        if time_data.time_step < each:
            eat_point = each
            break
    if eat_point is None:
        return np.nan
    eat_point_data = all_data[(all_data.file == time_data.file) & (all_data.time_step == eat_point)]
    if int(float(eat_point_data.status_h1)) == 0 and int(float(eat_point_data.status_h1)) == 0:
        res = "graze"
    else:
        res = "hunt"
    return res


def selectPreHuntData(feature_file):
    all_data = pd.read_csv(feature_file).rename(columns = {"index": "time_step"})
    all_data = (
        all_data.assign(
            shift_pacman_dir = all_data.pacman_dir.shift(-1)
        )
    )
    pre_hunt_data = all_data[(all_data.status_h1 == "prehunt") |  (all_data.status_h2 == "prehunt")]
    energizer_time_step = (
        pre_hunt_data.groupby(["file"]).apply(lambda x: _getEatEnergizerPoint(x.time_step.values))
        .rename("eat_energizer_point")
        .reset_index()
    )[["file", "eat_energizer_point"]]
    print()
    after_energizer_mode = (
        pre_hunt_data.merge(
            energizer_time_step,
            left_on = ["file"],
            right_on = ['file'],
            how = "left")
    )
    after_energizer_mode = (
        after_energizer_mode.assign(
            after_energizer_mode = after_energizer_mode.apply(
                lambda x: _getHuntLabel(all_data, x),axis = 1)
        )
    )
    after_energizer_mode.to_csv("extracted_data/all_prehunt_data.csv")


def _getStartPrehuntPoint(trial_data):
    start_hunt_point = [trial_data.time_step.iloc[0] - 1 if trial_data.time_step.iloc[0] > 0 else 0]
    pre_index = trial_data.time_step.iloc[0]
    for i in range(1, trial_data.shape[0]):
        cur_index = trial_data.time_step.iloc[i]
        if (cur_index - pre_index) > 1:
            start_hunt_point.append(cur_index - 1)
        pre_index = cur_index
    return start_hunt_point


def selectAroundPotentialDecisionPoint(feature_file,  pre_hunt_file):
    # Read data
    all_data = pd.read_csv(feature_file).rename(columns = {"index": "time_step"})
    pre_hunt_data = pd.read_csv(pre_hunt_file)
    for c in ["energizers", "pos"]:
        all_data[c] = all_data[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    for c in ["energizers", "pos", "eat_energizer_point"]:
        pre_hunt_data[c] = pre_hunt_data[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    # The point where the Pacman ate the energizer
    eat_energizer_data = pre_hunt_data.groupby(["file"]).apply(
        lambda x: x.eat_energizer_point.iloc[0]
    ).rename("eat_energizer_point").reset_index()
    # The point where the Pacman started to pre-hunt
    start_prehunt_data = pre_hunt_data.groupby(["file"]).apply(
        lambda x: _getStartPrehuntPoint(x)
    ).rename("start_prehunt_index").reset_index()
    # The region of energizers for each file
    energizer_region_data = pre_hunt_data.groupby(["file"]).apply(
        lambda x: [determine_region(poly_ext, each) for each in x.energizers.iloc[0]]
    ).rename("energizer_region").reset_index()
    # Take out data of 5 time steps after eating an energizer
    after_energizer_data = pd.DataFrame()
    for file in eat_energizer_data.file.values:
        for index, point in enumerate(eat_energizer_data[eat_energizer_data.file == file].eat_energizer_point.values[0]):
            eye_pos =[each for each in all_data[all_data.file == file].pos.iloc[point:point+5].values]
            eye_region = [
                determine_region(poly_ext, each) if not isinstance(each, float) else np.nan for each in eye_pos
            ]
            energizer_pos = pre_hunt_data[pre_hunt_data.file == file].energizers.iloc[0][index]
            after_energizer_data = after_energizer_data.append(
                pd.Series([
                    file,
                    int(point),
                    energizer_pos,
                    int(energizer_region_data[energizer_region_data.file == file].energizer_region.values[0][index]),
                    eye_pos,
                    eye_region]
                ),
                ignore_index=True
            )
    after_energizer_data = after_energizer_data.rename(
        columns ={
            0:"file",
            1:"after_energizer_point",
            2:"energizer_pos",
            3:"energizer_region",
            4:"after_energizer_eye_pos",
            5:"after_energizer_eye_region"}
    )
    # Take out data of 5 time steps before start to pre-hunt
    before_prehunt_data = pd.DataFrame()
    for file in start_prehunt_data.file.values:
        for index, point in enumerate(start_prehunt_data[start_prehunt_data.file == file].start_prehunt_index.values[0]):
            eye_pos =[each for each in all_data[all_data.file == file].pos.iloc[point:point+5].values]
            eye_region = [
                determine_region(poly_ext, each) if not isinstance(each, float) else np.nan for each in eye_pos
            ]
            energizer_pos = pre_hunt_data[pre_hunt_data.file == file].energizers.iloc[0][index]
            before_prehunt_data = before_prehunt_data.append(
                pd.Series([
                    file,
                    int(point),
                    energizer_pos,
                    int(energizer_region_data[energizer_region_data.file == file].energizer_region.values[0][index]),
                    eye_pos,
                    eye_region]
                ),
                ignore_index=True
            )
    before_prehunt_data = before_prehunt_data.rename(
        columns={
            0: "file",
            1: "start_prehunt_point",
            2: "energizer_pos",
            3: "energizer_region",
            4: "start_prehunt_eye_pos",
            5: "start_prehunt_eye_region"}
    )
    # Save data
    after_energizer_data.to_csv("extracted_data/after_energizer_data.csv")
    before_prehunt_data.to_csv("extracted_data/before_prehunt_data.csv")


if __name__ == '__main__':
    new_data_file = '../common_data/df_total_new.csv'
    # selectPreHuntData(new_data_file)
    selectAroundPotentialDecisionPoint(new_data_file, "extracted_data/all_prehunt_data.csv")