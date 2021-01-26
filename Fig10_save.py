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

def add_nearby_bean(df):
    df_temp = add_dis(
        df[["file", "index", "pacmanPos"]].merge(
            df[["file", "index", "beans"]].explode("beans"),
            on=["file", "index"],
            how="left",
        ),
        "pacmanPos",
        "beans",
    )
    return df.merge(
        df_temp[df_temp.dis <= 5]
        .groupby(["file", "index"])
        .size()
        .rename("nearby_bean_cnt")
        .reset_index(),
        how="left",
        on=["file", "index"],
    )


# local>0.9是local；global>0.9是global
# pess>0.3 && Not(local or global)
# Attack>0.3 && Not(local or global)
# Suicide>0.3 && Not(local or global)
# Vague: |local&global-0.5|<0.4 && not (Pess, Attack,Suicide)
# global, local, pessimistic, suicide, planned_hunting

def global_local_trans(x, thr):
    if isinstance(x, float):
        return None
    if x[0] > 0.0001 and x[1] > 0.0001:
        if x[0] / np.sqrt(sum(x ** 2)) >= thr:
            return "global_larger"
        elif x[1] / np.sqrt(sum(x ** 2)) >= thr:
            return "local_larger"
        elif x[2] / np.sqrt(sum(x ** 2)) > 0.3:
            return "evade_blinky"
        elif x[3] / np.sqrt(sum(x ** 2)) > 0.3:
            return "evade_clyde"
        elif x[4] / np.sqrt(sum(x ** 2)) > 0.3:
            return "approach"
        elif x[5] / np.sqrt(sum(x ** 2)) > 0.3:
            return "energizer"
        elif (
            x[1] / np.sqrt(sum(x ** 2)) > 1 - thr
            and x[1] / np.sqrt(sum(x ** 2)) < thr
            and x[0] / np.sqrt(sum(x ** 2)) > 1 - thr
            and x[0] / np.sqrt(sum(x ** 2)) < thr
        ):
            return "vague"
        else:
            return None


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
monkey = "Omega"
print("Start reading data for {}...".format(monkey))
df = pd.read_pickle(
    "/home/qlyang/pacman/PacmanAgent/constant/all_trial_data-window3-path10.pkl"
)
df_monkey = df[df.file.str.contains(monkey)]
df_reset_comb = add_states(
    df_monkey.sort_values(by=["file", "index"]).reset_index().drop("level_0", 1)
)
### 调整rt的对位
new_index = (
    df_reset_comb.loc[df_reset_comb[~df_reset_comb.rt.isnull()].index + 1, "pacmanPos"]
    .isin(TURNING_POS)
    .where(lambda x: x == True)
    .dropna()
    .index
)
df_reset_comb.loc[
    new_index, ["rt", "rt_std", "first_pos", "turns_previous", "previous_steps"]
] = df_reset_comb.loc[
    new_index - 1, ["rt", "rt_std", "first_pos", "turns_previous", "previous_steps"]
].values
df_reset_comb.loc[
    new_index - 1, ["rt", "rt_std", "first_pos", "turns_previous", "previous_steps"]
] = [np.nan] * 5
print("Finished reading data...")
print("="*50)


print("Start processing df_reset...")
df_reset = copy.deepcopy(df_reset_comb)
df_reset[df_reset.rwd_cnt.between(11, 80)].groupby("file").apply(
    lambda x: confu_mat_per_file(x, "labels")
).reset_index().groupby(["before", "after"])[0].sum().reset_index().pivot_table(
    index="before", columns="after", values=0, aggfunc="sum"
)
df_reset.assign(
    combine_tag_new=df_reset.apply(
        lambda x: x.labels
        if x.distance1 < 10 and x.ifscared1 < 3 and x.ifscared2 < 3
        else np.nan,
        1,
    )
).groupby("file").apply(
    lambda x: confu_mat_per_file(x, "combine_tag_new")
).reset_index().groupby(
    ["before", "after"]
)[
    0
].sum().reset_index().pivot_table(
    index="before", columns="after", values=0, aggfunc="sum"
)
print("Finished processing df_reset.")
print("="*50)


# Fig. 10
print("For Fig 10 (label_rt) :")

corners = [(2, 5),(2, 24),(2, 12),(2, 33),(2, 9),(2, 30),(27, 5),(27, 24),(27, 12),(27, 33),(27, 9),(27, 30)]
t_junction =  [(2,9),(2,30),(27,9),(27,30),(7,5),(22,5),(7,12),(22,12)]
cross = [(7,9),(7,24),(22,9),(22,24)]
tunnel =  [(7,18),(22,18)]
all_positions = list(set(
    [(2, 5),(2, 24),(2, 12),(2, 33),(2, 9),(2, 30),(27, 5),(27, 24),(27, 12),(27, 33),(27, 9),(27, 30),
     (2,9),(2,30),(27,9),(27,30),(7,5),(22,5),(7,12),(22,12), (7,9),(7,24),(22,9),(22,24),(7,18),(22,18)]
))

pos_index = [corners, t_junction, cross, tunnel, all_positions]
pos_name = ["corners", "t_junctions", "cross", "tunnel", "all"]
for index in range(5):
    sel_points = pos_index[index]
    df_plot = pd.concat(
        [
            df_reset_comb[df_reset_comb.pacmanPos.isin(sel_points)]
                .groupby("labels")
                .rt.apply(lambda x: np.mean(x / 60))
                .rename("mean"),
            df_reset_comb[df_reset_comb.pacmanPos.isin(sel_points)]
                .groupby("labels")
                .rt.apply(lambda x: np.std(x / 60) / np.sqrt(x.shape[0]))
                .rename("std"),
            df_reset_comb[df_reset_comb.pacmanPos.isin(sel_points)]
                .groupby("labels")
                .rt.count()
                .rename("size"),
        ],
        1,
    ).loc[["local", "global", "evade-blinky", "evade-clyde", "energizer", "approach", "vague"], :]
    print("{} data shape : ".format(pos_name[index]), df_plot.shape)  # label_rt
    with open("./plot_data/{}_label_rt_{}.pkl".format(monkey, pos_name[index]), "wb") as file:
        pickle.dump(df_plot, file)
print("=" * 50)
