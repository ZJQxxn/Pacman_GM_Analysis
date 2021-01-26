import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import seaborn as sns
import sys
import random
import pickle
from more_itertools import consecutive_groups
import copy


np.set_printoptions(suppress=True)
pd.set_option("display.max_rows", 200)
sys.path = sys.path + ["/home/qlyang/Documents/pacman/"]

from helper.utiltools import eval_df
from helper.pacmanutils import (
    rt_df_filter,
    generate_local_4dirs,
    relative_dir,
    largest_2ndlargest_diff,
    to_game,
    if_get_nearbean,
    assign_category,
    combine_pre_post,
    add_stats,
    add_combine_huntdis,
    generate_simulated_local_4dirs,
    get_marker,
    plot_ghost,
    plot_colors_simple,
)

from helper.add_features import add_dis, add_move_dir
from helper.constant_input import (
    OPPOSITE_DIRS,
    MAP_INFO,
    TURNING_POS,
    LOCS_DF,
    POSSIBLE_DIRS,
    SYMBOLS,
    ARRAY,
)
from helper.analysis import status_index

from ipywidgets import interact, fixed
import ipywidgets as widgets



def confu_mat_per_file(x, col):
    return (
        pd.concat([x[col].rename("after"), x[col].shift(1).rename("before")], 1)
        .dropna()
        .groupby(["before", "after"])
        .size()
    )


def add_states(df_reset):
    df_tmp = pd.DataFrame(
        [
            [np.nan] * 6 if isinstance(i, float) else i
            for i in df_reset.contribution.to_list()
        ],
        columns=["global", "local", "evade-blinky", "evade-clyde", "approach", "energizer"],
    )

    vague_mask = (
        np.sort(df_tmp.divide(np.sqrt(df_tmp.sum(1) ** 2), 0).values)[:, -1]
        - np.sort(df_tmp.divide(np.sqrt(df_tmp.sum(1) ** 2), 0).values)[:, -2]
    ) <= 0.1

    nan_mask = df_tmp.fillna(0).sum(1) == 0

    return pd.concat(
        [
            df_reset,
            pd.Series(
                [
                    ["global", "local", "evade-blinky", "evade-clyde", "approach", "energizer"][i]
                    for i in df_tmp.values.argsort()[:, -1]
                ]
            )
            .mask(vague_mask)
            .fillna("vague")
            .mask(nan_mask)
            .rename("labels"),
            df_tmp.divide(np.sqrt(df_tmp.sum(1) ** 2), 0).add_suffix("_weight"),
        ],
        1,
    )


print("Finished configuration.")
print("="*50)

### Read data
monkey = "Patamon"
print("Start reading data for {}...".format(monkey))
df = pd.read_pickle(
    "/home/qlyang/pacman/PacmanAgent/constant/all_trial_data-window3-path10.pkl"
)
df_monkey = df[df.file.str.contains(monkey)]
df_reset_comb = add_states(
    df_monkey.sort_values(by=["file", "index"]).reset_index().drop("level_0", 1)
)
print("Finished reading data...")
print("="*50)



def inTunnel(pos):
    if pos[1] == 18 and (pos[0]<=6 or pos[0]>=23):
        return True
    else:
        return False

df_reset_comb["is_tunnel"] = df_reset_comb.pacmanPos.apply(lambda x : inTunnel(x))
# Fig. 12 (tunnel)
print("For Fig. 12 : ")

tunnel_index = [
            list(i)
            for i in consecutive_groups(
            df_reset_comb[df_reset_comb.is_tunnel==True].index
        )
]
temp_tunnel_list = []
for each in tunnel_index:
    if len(each) == 14:
        temp_tunnel_list.append(each)
tunnel_index = temp_tunnel_list
print("Num of tunnel trajectory : ", len(tunnel_index))

traj_length = [len(each) for each in tunnel_index]
max_length = np.max(traj_length)
print("Max length : ", max_length)
print("Min length : ", np.min(traj_length))
print("Median length : ", np.median(traj_length))
all_weight = np.zeros((len(tunnel_index), 6, max_length))
all_weight[all_weight == 0] = np.nan

for index, each in enumerate(tunnel_index):
    for i, j in enumerate(each):
        contributions = df_reset_comb.contribution[j]
        contributions = np.array(contributions) / np.linalg.norm(np.array(contributions))
        all_weight[index, :, i] = copy.deepcopy(contributions)


print("Fig. 12 data shape : ", all_weight.shape) # 7.3
with open("./plot_data/{}_12.pkl".format(monkey), "wb") as file:
    pickle.dump(all_weight, file)