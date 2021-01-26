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


# Fig. 11.2B
print("For Fig. 11.2C : ")
energizer_start_index = df_reset_comb[
    (df_reset_comb.eat_energizer == True)
    & (df_reset_comb[["ifscared1", "ifscared2"]].min(1).shift() < 3)
][["next_eat_rwd", "energizers", "ifscared1", "ifscared2"]].index
energizer_lists = [
    np.arange(
        i,
        (df_reset_comb.loc[i:, ["ifscared1", "ifscared2"]] <= 3)
        .max(1)
        .where(lambda x: x == True)
        .dropna()
        .index[0],
    )
    for i in energizer_start_index
]

def add_PEG_dis(df_total):
    diss = add_dis(
        add_dis(
            add_dis(
                df_total[
                    ["ghost2Pos", "ghost1Pos", "next_eat_energizer", "pacmanPos"]
                ].reset_index(),
                "pacmanPos",
                "next_eat_energizer",
                "PE_dis",
            ),
            "next_eat_energizer",
            "ghost1Pos",
            "EG1_dis",
        ),
        "next_eat_energizer",
        "ghost2Pos",
        "EG2_dis",
    )

    diss["EG_dis"] = diss[["EG1_dis", "EG2_dis"]].min(1)
    df_total = pd.concat(
        [
            df_total,
            diss.set_index("index")[["PE_dis", "EG_dis", "EG1_dis", "EG2_dis"]],
        ],
        1,
    )
    return df_total

# 找到在每个sub trial之前，开始径直去吃energizer的index
df_temp = (
    pd.Series(energizer_lists)
    .apply(lambda x: x[0] if len(x) > 0 else np.nan)
    .dropna()
    .reset_index()
    .astype(int)
    .set_index(0)
    .reindex(range(417238))
    .rename(columns={"index": "last_index"})
)
df_temp.loc[~df_temp.last_index.isnull(), "last_index"] = df_temp.loc[
    ~df_temp.last_index.isnull()
].index
df_temp = df_temp.fillna(method="bfill")
pre_index = (
    add_dis(
        df_temp.reset_index()
        .rename(columns={0: "prev_index"})
        .merge(
            df_reset_comb["pacmanPos"].reset_index(),
            left_on="prev_index",
            right_on="index",
            how="left",
        )
        .drop(columns="index")
        .merge(
            df_reset_comb["pacmanPos"].reset_index(),
            left_on="last_index",
            right_on="index",
            how="left",
            suffixes=["_prev", "_last"],
        ),
        "pacmanPos_prev",
        "pacmanPos_last",
    )
    .sort_values(by="prev_index")
    .groupby("index")
    .apply(
        lambda x: x.set_index("prev_index")
        .dis.diff()
        .where(lambda x: x > 0)
        .dropna()
        .index.values[-1]
        if len(x.dis.diff().where(lambda x: x > 0).dropna()) > 0
        else x.prev_index.values[0]
    )
)
cons_list_plan = [
    np.arange(pre_index[i[0]], i[0])
    for i in energizer_lists
    if (df_reset_comb.loc[i[:5], "labels"] == "approach").sum() > 0 and len(i) > 0
]
cons_list_accident = [
    np.arange(pre_index[i[0]], i[0])
    for i in energizer_lists
    if (df_reset_comb.loc[i[:5], "labels"] == "approach").sum() == 0 and len(i) > 0
]
df_reset_comb_extend = add_PEG_dis(
    df_reset_comb.drop(columns=["PE_dis", "EG1_dis", "EG2_dis"])
)
mapping = {1: "Accidentally Hunting", 2: "Planned Hunting"}

bin_size = 4
ghost_data = {"1":None, "2":None}
for ghost in ["1", "2"]:
    i = 1
    df_status = pd.DataFrame()
    for sel_list in [cons_list_accident, cons_list_plan]:
        sel_list = pd.Series(sel_list)[
            pd.Series(sel_list)
            .explode()
            .reset_index()
            .set_index(0)
            .reset_index()
            .merge(df_reset_comb_extend.reset_index(), left_on=0, right_on="level_0")
            .groupby("index_x")
            .apply(lambda x: (x.next_eat_rwd.count() / len(x) <= 0.2))
            .values
        ]
        x = df_reset_comb_extend.loc[
            (
                df_reset_comb_extend.index.isin(
                    sel_list.apply(lambda x: x[0]).values.tolist()
                )
            )
            & (
                (df_reset_comb_extend.ifscared1 <= 2)
                | (df_reset_comb_extend.ifscared2 <= 2)
            )
            & (df_reset_comb_extend.PE_dis <= 10)
            & (df_reset_comb_extend["index"] > 0),
            ["distance" + ghost, "EG" + ghost + "_dis", "PE_dis"],
        ]
        df_status = df_status.append(x.reset_index().assign(cate=mapping[i]))
        i += 1
    df_status = df_status.reset_index().drop(columns="level_0")

    # df_status["distance" + ghost] = (
    #     np.ceil(df_status["distance" + ghost])
    # )
    # df_status["EG" + ghost + "_dis"] = (
    #     np.ceil(df_status["EG" + ghost + "_dis"])
    # )
    # df_status["EG" + ghost + "_dis_distance" + ghost] = np.ceil(
    #     (df_status["distance" + ghost] + df_status["EG" + ghost + "_dis"])
    # )
    # df_status["PE_dis"] = np.ceil(df_status["PE_dis"])

    # print(df_status.iloc[:5])


    ghost_data[ghost] = df_status # 11.2b relevant
    print("Ghost {} data shape {}".format(ghost, df_status.shape))
with open("./plot_data/{}_112C.pkl".format(monkey), "wb") as file:
    pickle.dump(ghost_data, file)
print("="*50)