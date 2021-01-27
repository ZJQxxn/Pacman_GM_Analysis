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
print("Finished reading data...")
print("="*50)


# Fig. 11.3
print("For Fig. 11.3 : ")
df_total = copy.deepcopy(df_reset_comb)

def generate_suicide_normal(df_total):
    select_last_num = 10
    suicide_normal = (
        df_total.reset_index()
        .merge(
            (df_total.groupby("file")["label_suicide"].sum() > 0) # TODO: the last 10 steps
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


suicide_lists, normal_lists = generate_suicide_normal(df_total)
# save data

suicide_path = list(
    filter(
        lambda x: len(x) >= 5,
        suicide_lists,
    )
)

normal_path = list(
    filter(
        lambda x: len(x) >= 5,
        normal_lists,
    )
)

# select normal dead after evade
print("Normal data num before filtering: ", len(normal_path))
temp_list = []
for each in normal_path:
    # print(each[0])
    before_normal = df_total.iloc[each[0] - 10:each[0]]
    is_evade = before_normal[["label_evade1", "label_evade2"]].apply(
        lambda x: x.label_evade1 == True or x.label_evade2 == True, axis=1)
    if is_evade.sum() == 10:
        temp_list.append(each)
normal_path = temp_list

print("Suicide data num : ", len(suicide_path))
print("Normal data num : ", len(normal_path))

max_length = np.max([len(each) for each in suicide_path])
print("Max for suicide : ", max_length)
suicide_weight = np.zeros((len(suicide_path), 6, max_length))
suicide_weight[suicide_weight == 0] = np.nan
for index, each in enumerate(suicide_path):
    for i, j in enumerate(each):
        contributions = df_total.contribution[j]
        contributions = np.array(contributions) / np.linalg.norm(np.array(contributions))
        suicide_weight[index, :, i] = copy.deepcopy(contributions)

max_length = np.max([len(each) for each in normal_path])
print("Max for normal : ", max_length)
normal_weight = np.zeros((len(normal_path), 6, max_length))
normal_weight[normal_weight == 0] = np.nan
for index, each in enumerate(normal_path):
    for i, j in enumerate(each):
        contributions = df_total.contribution[j]
        contributions = np.array(contributions) / np.linalg.norm(np.array(contributions))
        normal_weight[index, :, i] = copy.deepcopy(contributions)

print("Suicide weight shape : ", suicide_weight.shape)
print("Normal weight shape : ", normal_weight.shape)

with open("./plot_data/{}_115_suicide_weight.pkl".format(monkey), "wb") as file:
    pickle.dump(suicide_weight, file)
with open("./plot_data/{}_115_normal_weight.pkl".format(monkey), "wb") as file:
    pickle.dump(normal_weight, file)
print("="*50)

suicide_start_index = [each[0] for each in suicide_path]
normal_start_index = [each[0] for each in normal_path]

df_com = df_total[["file", "index", "next_eat_rwd_fill"]].merge(
    df_total.loc[suicide_start_index, "file"]
        .apply(
        lambda x: "-".join(
            [x.split("-")[0]] + [str(int(x.split("-")[1]) + 1)] + x.split("-")[2:]
        ),
    )
        .reset_index()
        .assign(
        index=[0] * len(suicide_start_index),
        reset_pos=[(14, 27)] * len(suicide_start_index),
        suicide_start_index=df_total.loc[suicide_start_index, "pacmanPos"].tolist(),
    ),
    on=["file", "index"],
)
df_com = add_dis(
    add_dis(df_com, "suicide_start_index", "next_eat_rwd_fill").rename(
        columns={"dis": "suicide_dis"}
    ),
    "next_eat_rwd_fill",
    "reset_pos",
).rename(columns={"dis": "reset_dis"})
df_hist_suicide = (
    pd.cut(
        df_com.suicide_dis - df_com.reset_dis,
        bins=range(-38, 30, 2),
        labels=range(-36, 30, 2),
    )
        .value_counts(normalize=True)
        .rename("distance")
        .reset_index()
        .assign(category="suicide")
)
df_hist_suicide.category = "suicide > 0 ratio: " + str(
    round(df_hist_suicide[df_hist_suicide["index"] > 0].sum().distance, 2)
)

df_com_another = df_total[["file", "index", "next_eat_rwd_fill"]].merge(
    df_total.loc[normal_start_index, "file"]
        .apply(
        lambda x: "-".join(
            [x.split("-")[0]] + [str(int(x.split("-")[1]) + 1)] + x.split("-")[2:]
        ),
    )
        .reset_index()
        .assign(
        index=[0] * len(normal_start_index),
        reset_pos=[(14, 27)] * len(normal_start_index),
        normal_start_index=df_total.loc[normal_start_index, "pacmanPos"].tolist(),
    ),
    on=["file", "index"],
)
df_com_another = add_dis(
    add_dis(df_com_another, "normal_start_index", "next_eat_rwd_fill").rename(
        columns={"dis": "normal_dis"}
    ),
    "next_eat_rwd_fill",
    "reset_pos",
).rename(columns={"dis": "reset_dis"})
df_hist_normal = (
    pd.cut(
        df_com_another.normal_dis - df_com_another.reset_dis,
        bins=range(-38, 30, 2),
        labels=range(-36, 30, 2),
    )
        .value_counts(normalize=True)
        .rename("distance")
        .reset_index()
        .assign(category="suicide")
)
df_hist_normal.category = "normal > 0 ratio: " + str(
    round(df_hist_normal[df_hist_normal["index"] > 0].sum().distance, 2)
)

print("df_com_another temp: ", (df_com_another.normal_dis - df_com_another.reset_dis).mean() / 2 + 18)
print("df_com_another df_com: ", (df_com.suicide_dis - df_com.reset_dis).mean() / 2 + 18)

data = {
    "data": pd.concat([df_hist_normal, df_hist_suicide]).sort_values(by="index"),
    "temp": (df_com_another.normal_dis - df_com_another.reset_dis).mean() / 2 + 18,
    "df_com": (df_com.suicide_dis - df_com.reset_dis).mean() / 2 + 18
} # 11.4
print("Data shape : ", data["data"].shape)
print("Temp shape : ", data["temp"].shape)
print("df_com shape : ", data["df_com"].shape)
with open("./plot_data/{}_1141.pkl".format(monkey), "wb") as file:
    pickle.dump(data, file)
print("="*50)

df_total = df_reset_comb
df_plot_all = pd.DataFrame()

for k, target_index in {
    "normal": normal_start_index,
    "suicide": suicide_start_index,
}.items():
    df_plot = (
        pd.Series(target_index)
            .explode()
            .rename("target_index")
            .reset_index()
            .merge(
            df_total["distance1"]
                .reset_index()
                .rename(columns={"index": "level_0"}),
            left_on="target_index",
            right_on="level_0",
        )
            .groupby("index")["distance1"]
            .mean()
            .rename(k)
    )
    df_plot_all = pd.concat(
        [
            df_plot_all,
            pd.cut(
                df_plot,
                bins=range(int(df_plot.min()), int(np.ceil(df_plot.max())) + 1),
                labels=list(
                    range(int(df_plot.min()), int(np.ceil(df_plot.max())) + 1)
                )[1:],
            )
                .value_counts(normalize=True)
                .rename("value")
                .reset_index()
                .assign(category=k),
        ]
    )

df_plot_all = df_plot_all[df_plot_all["index"] < 25] # 11.4.2
print("df_plot_all shape : ", df_plot_all.shape)
with open("./plot_data/{}_1142.pkl".format(monkey), "wb") as file:
    pickle.dump(df_plot_all, file)
print("="*50)