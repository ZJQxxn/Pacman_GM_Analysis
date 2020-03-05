import skimage

import json
import os
import random
import re

import ipywidgets as widgets
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.graph
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import preprocessing
from ipywidgets import fixed, interact
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.textpath import TextToPath

from pprint import pprint
import importlib
import zipfile
import sys
import networkx as nx
import itertools
from itertools import groupby, compress
from more_itertools import consecutive_groups
import pickle
from IPython.core.debugger import set_trace



def dedup_list(num):
    num = sorted(num)
    return list(num for num, _ in groupby(num))



def compare_list(a, b):
    return any([x == y for x, y in zip(a, b)])



def remove_consecutive_dup(l):
    return [x[0] for x in groupby(l)]




def pick_morebean(candidates, beans):
    if len(candidates) == 1:
        return candidates
    else:
        lens = [len(set(p) & set(beans)) for p in candidates]
        return list(compress(candidates, np.equal(lens, max(lens))))



def shortest(l):
    s = min([len(i) for i in l])
    return [i for i in l if len(i) == s]



def truncate_after(aa):
    if len(aa) > 1:
        return [np.array(range(int(aa[i - 1]), int(aa[i]))) for i in range(1, len(aa))]
    elif len(aa) == 1:
        return [np.array(range(int(aa[0]), int(aa[0]) + 10))]
    else:
        return []


def truncate_before(aa):
    if len(aa) > 1:
        return [
            np.array(range(int(aa[i + 1]), int(aa[i]), -1))
            for i in range(0, len(aa) - 1)
        ]
    elif len(aa) == 1:
        return [np.array(range(int(aa[0]), int(aa[0]) - 10, -1))]
    else:
        return []


def tuple_list(l):
    return [tuple(a) for a in l]

    

def is_corner(df_restart, map_info):
    map_info["is_corner"] = (
        map_info.loc[:, ["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"]]
        .astype(bool)
        .apply(
            lambda x: not (
                list(x) == [False, True, False, True]
                or list(x) == [True, False, True, False]
            ),
            1,
        )
    )
    map_info["pacmanPos"] = tuple_list(map_info[["Pos1", "Pos2"]].values)
    df_restart = df_restart.merge(
        map_info[["pacmanPos", "is_corner"]], on="pacmanPos", how="left"
    )
    return df_restart


fp = FontProperties(fname="Font Awesome 5 Pro-Solid-900.otf")
symbols = dict(
    ghost="\uf6e2", cookie="\uf563", cat="\uf6be", eye="\uf06e", monkey="\uf6fb", fix="\uf648"
)

suicide_df = pd.read_csv("suicide_point.csv", delimiter=",")
map_info = pd.read_csv("map_info_brian.csv")
map_info = map_info.assign(pacmanPos=tuple_list(map_info[["Pos1", "Pos2"]].values))
cross_pos = map_info[map_info.NextNum >= 3].pacmanPos.values
cross_pos = list(
    set(cross_pos)
    - set(
        [
            i
            for i in cross_pos
            if i[0] >= 11 and i[0] <= 18 and i[1] >= 16 and i[1] <= 20
        ]
    )
)

T, F = True, False
array = np.asarray(
    map_info.pivot_table(columns="Pos1", index="Pos2")
    .iswall.reindex(range(map_info.Pos2.max() + 1))
    .replace({1: F, np.nan: F, 0: T})
)
array = np.concatenate((array, np.array([[False] * 30])))
costs = np.where(array, 1, 1000)
handler_mapping = {0: "None", 1: "up", 2: "down", 3: "left", 4: "right"}
map_handler_mapping = {1: 1, 2: 3, 3: 2, 4: 4}
cross_road = tuple_list(
    map_info[(map_info.iswall == 0) & (map_info.NextNum >= 3)][["Pos1", "Pos2"]].values
)

locs_df = pd.read_csv("dij_distance_map.csv")
locs_df.pos1, locs_df.pos2 = (
    locs_df.pos1.apply(eval),
    locs_df.pos2.apply(eval),
)
handler_text2num = {"up": 1, "down": 2, "left": 3, "right": 4, "None": 0}


adjacent_cross = pd.read_csv("adjacent_cross.csv")
for c in ["pos1", "pos2", "no_cross_path"]:
    adjacent_cross[c] = adjacent_cross[c].map(eval)


d = dict(
    zip(
        map_info.pacmanPos,
        list(
            zip(
                *[
                    tuple_list(
                        map_info[
                            ["Next" + str(i) + "Pos1", "Next" + str(i) + "Pos2"]
                        ].values
                    )
                    for i in range(1, 5)
                ]
            )
        ),
    )
)

G = nx.DiGraph()
G.add_nodes_from(d.keys())
for k, v in d.items():
    G.add_edges_from(([(k, t) for t in v if t != (0, 0)]))



def update_module(m):
    mod = importlib.reload(sys.modules[m])  # use imp.reload for Python 3
    vars().update(mod.__dict__)


def get_marker(symbol):
    v, codes = TextToPath().get_text_path(fp, symbol)
    v = np.array(v)
    mean = np.mean([np.max(v, axis=0), np.min(v, axis=0)], axis=0)
    return Path(v - mean, codes, closed=False)



def take_record_df(path, Process_dict):
    for filename in sorted(os.listdir(path)):
        if filename.split(".")[-1] == "csv":
            x = pd.read_csv(path + "/" + filename)
            x[["pacMan_1", "ghost1_1", "ghost2_1"]] = x[
                ["pacMan_1", "ghost1_1", "ghost2_1"]
            ].applymap(lambda x: min(x, 29))
            x_rename = pd.DataFrame(
                {
                    "pacmanPos": tuple_list(x[["pacMan_1", "pacMan_2"]].values),
                    "ghost1Pos": tuple_list(x[["ghost1_1", "ghost1_2"]].values),
                    "ghost2Pos": tuple_list(x[["ghost2_1", "ghost2_2"]].values),
                    "ifscared1": x.ghost1_3,
                    "ifscared2": x.ghost2_3,
                    "ifSacc1": x.eyePos_1,
                    "ifSacc2": x.eyePos_2,
                    "handler": x.JoyStick,
                }
            )

            mask = np.mod((x_rename.index.values - 1), 25) == 0
            mask[-1] = True
            Process_dict[filename] = (
                x_rename.loc[mask, :]
                .reset_index()
                .rename(columns={"index": "origin_index"})
            )  # get from index=1, take each 25 indexes

    return Process_dict


def take_reward_df(path, Process_dict, Rewards_dict):
    for filename in sorted(os.listdir(path)):
        if filename.split(".")[-1] == "csv":
            x = pd.read_csv(path + "/" + filename)

            full_df = (
                x[x.Step.isin(Process_dict[filename].origin_index + 1)]
                .reset_index()
                .drop("index", 1)
            )

            replace_dict = (
                (Process_dict[filename].origin_index + 1)
                .reset_index()
                .set_index("origin_index")
                .to_dict()["index"]
            )
            full_df.Step = full_df.Step.replace(replace_dict)
            Rewards_dict[filename] = full_df

    return Rewards_dict


def switch(start, end):
    return start[::-1], end[::-1]


def dijkstra_distance(start, end):
    global costs
    start, end = switch(start, end)
    path, cost = skimage.graph.route_through_array(
        costs, start, end, fully_connected=False
    )
    path = [i[::-1] for i in path]
    return path


def continuous_jump(s, df):
    diff_list = df[s].diff()[:-1]
    if diff_list.values[-1] == -2:
        continuous_start = diff_list.where(lambda x: x != -2).dropna().index.max() + 1
        if s == "distance1":
            has_bean = (
                df["route_bean1"].where(lambda x: x == True).dropna().index.max() + 1
            )
        else:
            has_bean = (
                df["route_bean2"].where(lambda x: x == True).dropna().index.max() + 1
            )
        return max(continuous_start, has_bean)
    else:
        return None


def select_smallest(df, col):
    return df[df[col] == df[col].min()]
    
    

def add_nearest_pt(df, rwd_df):
    rwd_pac = df.pacmanPos.reset_index().merge(
        rwd_df, how="right", left_on="index", right_on="Step"
    )
    rwd_pac["nearrwdPos"] = tuple_list(rwd_pac[["X", "Y"]].values)

    rwd_pac = (
        rwd_pac.merge(
            locs_df,
            left_on=["nearrwdPos", "pacmanPos"],
            right_on=["pos1", "pos2"],
            how="left",
        )
        .rename(columns={"dis": "rwd_pac_distance"})
        .drop(columns=["pos1", "pos2", "X", "Y", "Reward"])
    )

    rwd_pac.rwd_pac_distance = rwd_pac.rwd_pac_distance.fillna(0)

    df = pd.concat(
        [
            rwd_pac.loc[
                rwd_pac.groupby("Step")["rwd_pac_distance"].idxmin(),
                ["Step", "nearrwdPos", "rwd_pac_distance"],
            ].set_index("Step"),
            df,
            rwd_df.groupby("Step").count().X.rename("rwd_cnt"),
        ],
        1,
    )
    eat_big = (
        rwd_df.groupby("Step")
        .sum()["Reward"]
        .diff()
        .where(lambda x: x == -2)
        .dropna()
        .index
    )

    df["scared_time1"] = pd.Series(df.index).apply(
        lambda x: x - eat_big[x - eat_big >= 0].max() + 1, 1
    )
    df["scared_time2"] = pd.Series(df.index).apply(
        lambda x: x - eat_big[x - eat_big >= 0].max() + 1, 1
    )
    df.loc[df.ifscared1 < 4, "scared_time1"] = np.nan
    df.loc[df.ifscared2 < 4, "scared_time2"] = np.nan
    return df


color_d = {
    "Orange": "Graze",
    "Grey": "Evade",
    "Green": "Graze + Evade",
    "Blue": "None",
    "Red": "Suicide",
}


def plot_colors(df_pos, idx):
    xy = np.array(list(zip(*df_pos.loc[max(0, idx - 15) : idx, "pacmanPos"]))).T + 0.5
    i = max(0, idx - 15)
    for start, stop in zip(xy[:-1], xy[1:]):
        x, y = zip(start, stop)
        x1, y1 = start
        x2, y2 = stop
        plt.arrow(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            linewidth=2,
            head_width=0.3,
            head_length=0.3,
            fc=df_pos.color[i],
            ec=df_pos.color[i],
        )
        plt.plot(x, y, color=df_pos.color[i], label=color_d[df_pos.color[i]])
        i += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    
    
def plot_colors_simple(df_pos, idx):
    xy = np.array(list(zip(*df_pos.loc[max(0, idx - 10) : idx, "pacmanPos"]))).T + 0.5
    i = max(0, idx - 15)
    for start, stop in zip(xy[:-1], xy[1:]):
        x, y = zip(start, stop)
        x1, y1 = start
        x2, y2 = stop
        plt.arrow(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            linewidth=2,
            head_width=0.3,
            head_length=0.3,
            fc='green',
            ec='green',
        )
        plt.plot(x, y, color='green', label='pacman path')
        i += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    


def plot_ghost(df_pos, idx):
    color = "red"
    xy = np.array(list(zip(*df_pos.loc[max(0, idx - 15) : idx, "ghost1Pos"]))).T + 0.5
    i = max(0, idx - 15)
    for start, stop in zip(xy[:-1], xy[1:]):
        x, y = zip(start, stop)
        x1, y1 = start
        x2, y2 = stop
        plt.arrow(
            x1,
            y1,
            x2 - x1,
            y2 - y1,
            linewidth=1,
            linestyle="dotted",
            head_width=0.3,
            head_length=0.3,
            fc=color,
            ec=color,
        )
        plt.plot(x, y, linestyle="dotted", color=color, label="Ghost1 Track")
        i += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())


def plot_eating_all(df_pos, k, idx, Rewards_dict):
    reward_sel = Rewards_dict[k + ".csv"]
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(array, ax=ax, linewidth=0.5, annot=False, cbar=False, cmap="bone")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(k)

    df_pos.loc[
        (df_pos["status_g"] == 1)
        & ((df_pos["status_e1"] == 0) | (df_pos["status_e2"] == 0)),
        "color",
    ] = "Orange"
    df_pos.loc[
        (df_pos["status_g"] == 0)
        & ((df_pos["status_e1"] == 1) & (df_pos["status_e2"] == 1)),
        "color",
    ] = "Grey"
    df_pos.loc[
        (df_pos["status_g"] == 1)
        & ((df_pos["status_e1"] == 1) & (df_pos["status_e2"] == 1)),
        "color",
    ] = "Green"
    df_pos.loc[
        (df_pos["status_g"] == 0)
        & ((df_pos["status_e1"] == 0) | (df_pos["status_e2"] == 0)),
        "color",
    ] = "Blue"
    df_pos.loc[df_pos["status_s"] == 1, "color"] = "Red"

    plot_colors(df_pos, idx)
    plot_ghost(df_pos, idx)

    plt.scatter(
        reward_sel[reward_sel.Step == idx].X + 0.5,
        reward_sel[reward_sel.Step == idx].Y + 0.5,
        color="brown",
        s=reward_sel[reward_sel.Step == idx].Reward * 70,
        marker=get_marker(symbols["cookie"]),
    )
    plt.scatter(
        np.array(df_pos.loc[idx, "pacmanPos"])[0] + 0.5,
        np.array(df_pos.loc[idx, "pacmanPos"])[1] + 0.5,
        marker=get_marker(symbols["monkey"]),
        s=300,
        color="green",
    )  # pacman
    
    plt.scatter(
        np.array(df_pos.loc[idx, "pos"])[0] + 0.5,
        np.array(df_pos.loc[idx, "pos"])[1] + 0.5,
        marker=get_marker(symbols["fix"]),
        s=300,
        color="#8E44AD", edgecolor="white"
        
    ) # fixation

    if df_pos.loc[idx, "ifscared1"] < 3:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost1Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost1Pos"])[1] + 0.5,
            marker=get_marker(symbols["ghost"]),
            s=300,
            color="red",
        )  # normal ghost
    elif df_pos.loc[idx, "ifscared1"] >= 4:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost1Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost1Pos"])[1] + 0.5,
            marker=get_marker(symbols["ghost"]),
            s=300,
            color="white",
            edgecolor="red",
        )  # scared + flashing ghost
    else:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost1Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost1Pos"])[1] + 0.5,
            marker=get_marker(symbols["eye"]),
            s=30,
            color="white",
            edgecolor="black",  # dead ghost
        )

    if df_pos.loc[idx, "ifscared2"] < 3:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost2Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost2Pos"])[1] + 0.5,
            marker=get_marker(symbols["ghost"]),
            s=300,
            color="orange",
        )  # ghost 2
    elif df_pos.loc[idx, "ifscared2"] >= 4:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost2Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost2Pos"])[1] + 0.5,
            marker=get_marker(symbols["ghost"]),
            s=300,
            color="white",
            edgecolor="orange",
        )  # scared + flashing ghost
    else:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost2Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost2Pos"])[1] + 0.5,
            marker=get_marker(symbols["eye"]),
            s=30,
            color="white",
            edgecolor="black",
        )  # ghost 1

    if k in suicide_df.file.values:
        predict_tp = suicide_df.loc[suicide_df.file == k, "predict"].values[0]
        actual_tp = suicide_df.loc[suicide_df.file == k, "actual"].values[0]
        ax.text(s="if suicide: " + str(idx >= actual_tp), x=31, y=20, fontsize=12)
        ax.text(
            s="predict if suicide: " + str(idx >= predict_tp), x=31, y=21, fontsize=12
        )

    ax.text(
        s="hunt ghost1: " + str(df_pos.loc[idx, "status_h1"]), x=31, y=13, fontsize=12
    )
    ax.text(
        s="hunt ghost2: " + str(df_pos.loc[idx, "status_h2"]), x=31, y=14, fontsize=12
    )
    ax.text(s="graze: " + str(df_pos.loc[idx, "status_g"]), x=31, y=15, fontsize=12)
    ax.text(
        s="evade ghost1: " + str(df_pos.loc[idx, "status_e1"]), x=31, y=16, fontsize=12
    )
    ax.text(
        s="evade ghost2: " + str(df_pos.loc[idx, "status_e2"]), x=31, y=17, fontsize=12
    )
    ax.text(
        s="handler: " + handler_mapping[df_pos.loc[idx, "handler"]],
        x=31,
        y=19,
        fontsize=15,
    )


    
    
def plot_eating_simple(df_pos, k, idx, Rewards_dict):
    reward_sel = Rewards_dict[k + ".csv"]
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(array, ax=ax, linewidth=0.5, annot=False, cbar=False, cmap="bone")
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title(k)

    plot_colors_simple(df_pos, idx)
    plot_ghost(df_pos, idx)

    plt.scatter(
        reward_sel[reward_sel.Step == idx].X + 0.5,
        reward_sel[reward_sel.Step == idx].Y + 0.5,
        color="brown",
        s=reward_sel[reward_sel.Step == idx].Reward * 70,
        marker=get_marker(symbols["cookie"]),
    )
    plt.scatter(
        np.array(df_pos.loc[idx, "pacmanPos"])[0] + 0.5,
        np.array(df_pos.loc[idx, "pacmanPos"])[1] + 0.5,
        marker=get_marker(symbols["monkey"]),
        s=300,
        color="green",
    )  # pacman
    

    if df_pos.loc[idx, "ifscared1"] < 3:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost1Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost1Pos"])[1] + 0.5,
            marker=get_marker(symbols["ghost"]),
            s=300,
            color="red",
        )  # normal ghost
    elif df_pos.loc[idx, "ifscared1"] >= 4:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost1Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost1Pos"])[1] + 0.5,
            marker=get_marker(symbols["ghost"]),
            s=300,
            color="white",
            edgecolor="red",
        )  # scared + flashing ghost
    else:
        plt.scatter(
            np.array(df_pos.loc[idx, "ghost1Pos"])[0] + 0.5,
            np.array(df_pos.loc[idx, "ghost1Pos"])[1] + 0.5,
            marker=get_marker(symbols["eye"]),
            s=30,
            color="white",
            edgecolor="black",  # dead ghost
        )

#     if df_pos.loc[idx, "ifscared2"] < 3:
#         plt.scatter(
#             np.array(df_pos.loc[idx, "ghost2Pos"])[0] + 0.5,
#             np.array(df_pos.loc[idx, "ghost2Pos"])[1] + 0.5,
#             marker=get_marker(symbols["ghost"]),
#             s=300,
#             color="orange",
#         )  # ghost 2
#     elif df_pos.loc[idx, "ifscared2"] >= 4:
#         plt.scatter(
#             np.array(df_pos.loc[idx, "ghost2Pos"])[0] + 0.5,
#             np.array(df_pos.loc[idx, "ghost2Pos"])[1] + 0.5,
#             marker=get_marker(symbols["ghost"]),
#             s=300,
#             color="white",
#             edgecolor="orange",
#         )  # scared + flashing ghost
#     else:
#         plt.scatter(
#             np.array(df_pos.loc[idx, "ghost2Pos"])[0] + 0.5,
#             np.array(df_pos.loc[idx, "ghost2Pos"])[1] + 0.5,
#             marker=get_marker(symbols["eye"]),
#             s=30,
#             color="white",
#             edgecolor="black",
#         )  # ghost 1

    if k in suicide_df.file.values:
        predict_tp = suicide_df.loc[suicide_df.file == k, "predict"].values[0]
        actual_tp = suicide_df.loc[suicide_df.file == k, "actual"].values[0]
        ax.text(s="if suicide: " + str(idx >= actual_tp), x=31, y=20, fontsize=12)
        ax.text(
            s="predict if suicide: " + str(idx >= predict_tp), x=31, y=21, fontsize=12
        )

    ax.text(
        s="handler: " + handler_mapping[df_pos.loc[idx, "handler"]],
        x=31,
        y=19,
        fontsize=15,
    )

    
    
    
def graze(df):
    df.loc[
        ((df.nearrwdPos == df.nearrwdPos.shift()) & (df.rwd_pac_distance.diff() < 0))
        | (df.rwd_cnt < df.rwd_cnt.shift())
        | (df.nearrwdPos != df.nearrwdPos.shift()),
        "status_g",
    ] = 1
    df["status_g"] = df["status_g"].fillna(0)
    return df


def hunt(df):
    for i in ["1", "2"]:
        df.loc[
            (
                (
                    df[["pacmanPos", "ghost" + i + "Pos"]]
                    != df[["pacmanPos", "ghost" + i + "Pos"]].shift()
                ).sum(1)
                >= 1
            )
            & (
                ((df["ifscared" + i] == 3) & (df["ifscared" + i].diff() != 0))
                | (
                    (df["ifscared" + i] >= 4)
                    & (df["distance" + i].diff() <= 0)
                    & (df["pac_to_ghost" + i] == True)
                )
            ),
            "status_h" + i,
        ] = 1
        df["status_h" + i] = df["status_h" + i].fillna(0)
    return df


def evade(df):
    for i in ["1", "2"]:
        idx = df["distance" + i].diff().where(lambda x: x == 0).dropna().index
        dd = pd.DataFrame(list(enumerate(idx)), columns=["em", "act"])
        dd["gr"] = dd.em - dd.act
        reshape_list = dd.groupby("gr").act.apply(lambda x: list(x)).values
        l = []
        for r in reshape_list:
            r = [r[0] - 1] + r
            if r[0] > 0 and df["distance" + i][r[0]] < df["distance" + i][r[0] - 1]:
                l.extend(r)
        df.loc[l, "status_e" + i] = 1
        df["status_e" + i] = df["status_e" + i].fillna(0)
    return df


def pacman_chase_ghost(df):
    for i in ["1", "2"]:
        df["paths" + i] = df.apply(
            lambda x: dijkstra_distance(x["ghost" + i + "Pos"], x.pacmanPos), 1
        ).shift()
        df.loc[1:, "pac_to_ghost" + i] = df[1:].apply(
            lambda x: x.pacmanPos in x["paths" + i], 1
        )
        df.loc[1:, "toward_each_other" + i] = df[1:].apply(
            lambda x: x["ghost" + i + "Pos"] in x["paths" + i]
            and x.pacmanPos in x["paths" + i],
            1,
        )
    return df.drop(columns=["paths1", "paths2"])


def combine_hunt(x):
    if x.status_h1 + x.status_h2 == 2:
        return min(x.distance1, x.distance2), 1, min(x.scared_time1, x.scared_time1)
    elif x.status_h1 + x.status_h2 == 0:
        return min(x.distance1, x.distance2), 0, min(x.scared_time1, x.scared_time1)
    elif x.status_h1 == 1:
        return x.distance1, 1, x.scared_time1
    else:
        return x.distance2, 1, x.scared_time2


def combine_evade(x):
    if x.status_e1 + x.status_e2 == 2:
        return min(x.distance1, x.distance2), 1
    elif x.status_e1 + x.status_e2 == 0:
        return min(x.distance1, x.distance2), 0
    elif x.status_h1 == 1:
        return x.distance1, 1
    else:
        return x.distance2, 1


def last_two_false(s):
    b = s.replace({True: np.nan}).dropna().index
    if len(b) == 0:
        return np.nan
    c = pd.Series(b)[pd.Series(b).diff() > 1].values
    if len(c) == 0:
        return b[0]
    return c[-1]


def when_switch(df):
    change_index = [df.handler.replace({0: np.nan}).dropna().index.min()] + list(
        df.handler.replace({0: np.nan})
        .fillna(method="ffill")
        .diff()
        .replace({0: np.nan})
        .dropna()
        .index
    )
    df["when_switch"] = df.apply(
        lambda x: max(filter(lambda i: i <= x.name, change_index))
        if len(list(filter(lambda i: i <= x.name, change_index))) > 0
        else 0,
        1,
    )
    return df


def label_s2(df):
    df.loc[
        (
            (df.rwd_pac_distance - df.distance1).shift()
            >= df.rwd_pac_distance - df.distance1
        )
        & (df.rwd_pac_distance.shift() >= df.rwd_pac_distance),
        "label",
    ] = "tobean"
    df.loc[
        (
            (df.rwd_pac_distance - df.distance1).shift()
            < df.rwd_pac_distance - df.distance1
        )
        & (df.rwd_pac_distance.shift() < df.rwd_pac_distance),
        "label",
    ] = "offbean"
    df.loc[
        (
            (df.rwd_pac_distance - df.distance1).shift()
            >= df.rwd_pac_distance - df.distance1
        )
        & (df.rwd_pac_distance.shift() < df.rwd_pac_distance),
        "label",
    ] = "dispatch"
    df.loc[
        (
            (df.rwd_pac_distance - df.distance1).shift()
            < df.rwd_pac_distance - df.distance1
        )
        & (df.rwd_pac_distance.shift() >= df.rwd_pac_distance),
        "label",
    ] = "toward"
    return df


def label_s1(df):
    df.loc[
        (df.rwd_pac_distance.shift() >= df.rwd_pac_distance)
        & (
            (df.distance1 + df.rwd_pac_distance).shift()
            >= df.distance1 + df.rwd_pac_distance
        ),
        "label",
    ] = "tobean"
    df.loc[
        (df.rwd_pac_distance.shift() < df.rwd_pac_distance)
        & (
            (df.distance1 + df.rwd_pac_distance).shift()
            < df.distance1 + df.rwd_pac_distance
        ),
        "label",
    ] = "offbean"
    df.loc[
        (df.rwd_pac_distance.shift() >= df.rwd_pac_distance)
        & (
            (df.distance1 + df.rwd_pac_distance).shift()
            < df.distance1 + df.rwd_pac_distance
        ),
        "label",
    ] = "dispatch"
    df.loc[
        (df.rwd_pac_distance.shift() < df.rwd_pac_distance)
        & (
            (df.distance1 + df.rwd_pac_distance).shift()
            >= df.distance1 + df.rwd_pac_distance
        ),
        "label",
    ] = "toward"
    return df


def when_can_switch(df):
    df_aa = df.merge(map_info, on="pacmanPos", how="left")
    df_aa.handler = df_aa.handler.replace({0: np.nan}).fillna(method="ffill")
    df["when_can_switch"] = df_aa.apply(
        lambda x: last_two_false(
            df_aa.loc[
                : x.name, "Next" + str(map_handler_mapping[x.handler]) + "Pos2"
            ].astype(bool)
        )
        if not np.isnan(x.handler)
        else np.nan,
        1,
    )
    return df


def when_turn(filtered_df):
    filtered_df["when_turn"] = filtered_df.apply(
        lambda x: max(
            filter(
                lambda i: i <= x.name,
                (
                    filtered_df.fillna(0).pacman_dir.shift()
                    != filtered_df.fillna(0).pacman_dir
                )
                .replace({False: np.nan})
                .dropna()
                .index,
            )
        ),
        1,
    )
    return filtered_df


def possible_directions(df_restart):
    df_aa = df_restart.merge(map_info, on="pacmanPos", how="left")
    df_restart["possible_dirs"] = df_aa.apply(
        lambda x: pd.Series(["up", "left", "down", "right"])[
            x[["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"]].astype(bool).values
        ].values,
        1,
    ).values
    return df_restart



def before_possible(df):
    df["before_last"] = df.loc[
        pd.Series(df.index.where(df.is_corner)).fillna(df.index.min()).cummax().values,
        "possible_dirs",
    ].values
    return df


def after_possible(df):
    df["after_first"] = df.loc[
        pd.Series(df.index.where(df.is_corner)).fillna(method="bfill").values,
        "possible_dirs",
    ].values
    return df


def if_miss(df_total):
    if len(
        pd.Series(df_total.index.where(df_total.is_corner)
                 ).fillna(method="bfill").dropna()
    ) >= 1:
        df_total = (
                df_total.groupby("file")
                .apply(before_possible)
                .groupby("file")
                .apply(after_possible)
            )
        df_total.handler = df_total.handler.replace(handler_mapping)
        df_total["if_miss"] = df_total.apply(
            lambda x: x.handler in x.before_last and x.handler not in x.
            after_first if isinstance(x.after_first, list) else False,
            1,
        )
        df_total.loc[
            (
                df_total.loc[df_total[df_total["is_corner"]].index, "pacman_dir"]
                == df_total.loc[df_total[df_total["is_corner"]].index + 1, "pacman_dir"].values
            )
            .where(lambda x: x == True)
            .dropna()
            .index,
            "if_passby",
        ] = True

        df_total.if_passby = df_total.if_passby.fillna(False)
        both_true_index = df_total[(df_total.if_passby) & (df_total.is_corner)].index

        for i in range(1, len(both_true_index)):
            if (
                df_total.loc[both_true_index[i - 1] + 1 : both_true_index[i] - 1][
                    ["if_passby", "is_corner"]
                ]
                .astype(int)
                .sum()
                .sum()
                > 0
            ):
                df_total.loc[both_true_index[i - 1] : both_true_index[i], "if_miss"] = False
            else:
                start_pacman_dir = df_total.loc[both_true_index[i - 1], "pacman_dir"]
                df_total.loc[
                    (df_total.pacman_dir == start_pacman_dir)
                    & (
                        df_total.index.isin(
                            range(both_true_index[i - 1] + 1, both_true_index[i])
                        )
                    ),
                    "if_miss",
                ] = False
        df_total["miss_dis"] = (
            pd.Series(df_total.index.where(df_total.is_corner)
                     ).fillna(0).cummax().values - df_total.index
        ).values
    return df_total


def pacman_dir(df):
    for i in df.index[1:]:
        dir_array = np.array(df.loc[i, "pacmanPos"]) - np.array(
            df.loc[i - 1, "pacmanPos"]
        )
        if np.array_equal(dir_array, np.array([-1, 0])):
            df.loc[i, "pacman_dir"] = "left"
        if np.array_equal(dir_array, np.array([1, 0])):
            df.loc[i, "pacman_dir"] = "right"
        if np.array_equal(dir_array, np.array([0, -1])):
            df.loc[i, "pacman_dir"] = "up"
        if np.array_equal(dir_array, np.array([0, 1])):
            df.loc[i, "pacman_dir"] = "down"
#     df["pacman_dir"] = df["pacman_dir"].fillna(method='ffill')
    return df


def ghost_dir(df):
    for i in df.index[1:]:
        dir_array = np.array(df.loc[i, "ghost2Pos"]) - np.array(
            df.loc[i - 1, "ghost2Pos"]
        )
        if np.array_equal(dir_array, np.array([-1, 0])):
            df.loc[i, "ghost2_dir"] = "left"
        if np.array_equal(dir_array, np.array([1, 0])):
            df.loc[i, "ghost2_dir"] = "right"
        if np.array_equal(dir_array, np.array([0, -1])):
            df.loc[i, "ghost2_dir"] = "up"
        if np.array_equal(dir_array, np.array([0, 1])):
            df.loc[i, "ghost2_dir"] = "down"
#     df["pacman_dir"] = df["pacman_dir"].fillna(method='ffill')
    return df


def check_ghost_twosides(df_model):
    df_temp = (
        df_model.merge(
            locs_df,
            left_on=["next_cross", "ghost1Pos"],
            right_on=["pos1", "pos2"],
            how="left",
        )
        .drop(columns=["pos1", "pos2"])
        .merge(
            locs_df,
            left_on=["next_cross", "ghost2Pos"],
            right_on=["pos1", "pos2"],
            how="left",
        )
        .drop(columns=["pos1", "pos2"])
    )
    df_temp2 = df_temp[(df_temp.dis_x <= 10) | (df_temp.dis_y <= 10)]

    df_temp = (
        df_model.merge(
            locs_df,
            left_on=["next_cross", "ghost1Pos"],
            right_on=["pos1", "pos2"],
            how="left",
        )
        .drop(columns=["pos1", "pos2"])
        .merge(
            locs_df,
            left_on=["next_cross", "ghost2Pos"],
            right_on=["pos1", "pos2"],
            how="left",
        )
        .drop(columns=["pos1", "pos2"])
    )
    df_temp2 = df_temp[(df_temp.dis_x <= 10) | (df_temp.dis_y <= 10)]

    df_temp.loc[(df_temp.dis_x <= 10) | (df_temp.dis_y <= 10), "ghost_in_cross"] = [
        list(itertools.chain.from_iterable(i))
        for i in list(
            zip(
                *[
                    df_temp2.apply(
                        lambda x: list(
                            set(x.next_possible_dir)
                            & set(relative_dir(x.ghost1Pos, x.next_cross))
                        )
                        if x.dis_x <= 10
                        else [],
                        1,
                    ).values,
                    df_temp2.apply(
                        lambda x: list(
                            set(x.next_possible_dir)
                            & set(relative_dir(x.ghost2Pos, x.next_cross))
                        )
                        if x.dis_y <= 10
                        else [],
                        1,
                    ).values,
                ]
            )
        )
    ]
    return df_temp


def relative_dir(pt_pos, pacman_pos):
    l = []
    dir_array = np.array(pt_pos) - np.array(pacman_pos)
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


def if_cause_dead(f, idx):
    if f not in last_trials and abs(Process_dict[f].index.max()) - idx <= 5:
        return True
    elif f in last_trials and abs(Process_dict[f].index.max()) - idx <= 5:
        return "Win"
    else:
        return False


def nearby_beans(filtered_df, Rewards_dict):
    df_t = pd.DataFrame()
    for f in filtered_df.file.unique():
        ddd = (
            filtered_df[filtered_df.file == f]
            .merge(Rewards_dict[f], left_on="index", right_on="Step")
            .drop_duplicates()
        )
        ddd = ddd.assign(rwdPos=tuple_list(ddd[["X", "Y"]].values)).merge(
            locs_df,
            left_on=["rwdPos", "pacmanPos"],
            right_on=["pos1", "pos2"],
            how="left",
        )
        df_t = df_t.append(
            ddd[ddd.dis <= 10]
            .groupby(["Step", "pacmanPos"])
            .count()
            .X.rename("near_beans")
            .reset_index()
            .assign(file=f)
        )
    return filtered_df.merge(
        df_t,
        left_on=["index", "pacmanPos", "file"],
        right_on=["Step", "pacmanPos", "file"],
        how="left",
    )


def add_nearest_pts_dis(df_explore, reward_sel):
    reward_sel = reward_sel.assign(rwd_pos=tuple_list(reward_sel[["X", "Y"]].values))
    df_explore = (
        df_explore.reset_index()
        .merge(reward_sel, left_on="index", right_on="Step")
        .reset_index()
        .merge(locs_df, left_on=["pacmanPos", "rwd_pos"], right_on=["pos1", "pos2"])
        .sort_values(by="dis")
        .groupby("index")
        .first()[list(df_explore.columns) + ["dis", "pos2"]]
        .rename(columns={"dis": "rwd_pac_distance", "pos2": "nearrwdPos"})
    )
    del df_explore.index.name
    return df_explore


def check_dir(x):
    if x > 30 or (x > 9 and x < 18):
        return "down"
    else:
        return "up"


def from_dict2df(ten_points_pac):
    idx_l, item_l = [], []
    for idx, item in ten_points_pac.items():
        idx_l.extend([idx + ".csv"] * len(item))
        item_l.extend(item)

    return pd.DataFrame({"file": idx_l, "index": item_l})


with open("ten_points_pac.json", "r") as f:
    ten_points_pac = json.load(f)
ten_points_df = from_dict2df(ten_points_pac)


def has_cross(aaa):
    for a in aaa:
        if len(set(a[1:-1]) & set(cross_pos)) == 0:
            return a
    return []


def nearest_cross(curr_pos):
    df_temp = locs_df[
        (locs_df.pos1 == curr_pos)
        & (
            locs_df.pos2.isin(
                adjacent_cross[
                    adjacent_cross.no_cross_path.apply(lambda x: curr_pos in x)
                ].pos1.values
            )
        )
    ]
    df_temp.loc[:, "no_cross_path"] = df_temp.apply(
        lambda x: has_cross(list(nx.node_disjoint_paths(G, x.pos1, x.pos2))), 1
    )
    return df_temp[df_temp.no_cross_path.map(len) > 0]



def local_opt_paths(bean_pos, pacman_pos):
    if bean_pos[1] == 14:
        return []
    path_list = []
    if pacman_pos not in cross_pos:
        df_temp = nearest_cross(pacman_pos)
        mask = df_temp.no_cross_path.apply(
            lambda x: abs(np.array(x[1]) - np.array(bean_pos)).sum()
            <= abs(np.array(x[0]) - np.array(bean_pos)).sum()
        )
        path_list_temp = df_temp[mask].no_cross_path.values.tolist()
        if path_list_temp != path_list:
            path_list = path_list_temp
            path_list_temp = []
    else:
        path_list_temp = []
        path_list = [[pacman_pos]]

    signal = True
    while signal:
        for l in path_list:
            mask = adjacent_cross[adjacent_cross.pos1 == l[-1]].apply(
                lambda x: abs(np.array(x.no_cross_path[1]) - np.array(bean_pos)).sum()
                <= abs(np.array(x.no_cross_path[0]) - np.array(bean_pos)).sum()
                and x.no_cross_path[1] != l[-min(2, len(l))]
                and x.no_cross_path[1] != (23, 18)
                and x.no_cross_path[1] != (6, 18)
                and bean_pos not in l
                and len(l) <= 60,
                1,
            )
            if mask.sum() > 0:
                path_list_temp.extend(
                    (
                        l[:-1]
                        + adjacent_cross[adjacent_cross.pos1 == l[-1]][
                            mask
                        ].no_cross_path
                    ).values.tolist()
                )
            else:
                path_list_temp.append(l)
        if path_list_temp != path_list:
            path_list = path_list_temp
            path_list_temp = []
        else:
            signal = False

    path_list = (
        pd.Series(
            [i[: i.index(bean_pos) + 1] if bean_pos in i else np.nan for i in path_list]
        )
        .dropna()
        .tolist()
    )
    return path_list


def approaching_energizer(pacman_pos, energizer_list):
    if not isinstance(energizer_list, float):
        temp = locs_df[
            (locs_df.pos1 == pacman_pos)
            & (locs_df.pos2.isin(energizer_list))
            & (locs_df.dis <= 10)
        ]
        if not temp.empty:
            return (
                temp.sort_values(by="dis")
                .apply(lambda x: relative_dir(x.pos2, x.pos1), 1)
                .values[0]
            )


corner_pos = [(16, 27), (19, 27), (22, 27), (22, 30), (27, 30), (27, 24), (22, 24), (19, 24), (16, 24), (22, 18), (22, 12), (22, 9), (22, 5), (27, 5), (27, 9), (19, 9), (19, 12), (16, 12), (16, 15), (15, 15), (14, 15), (13, 15), (13, 12), (10, 12), (10, 9), (13, 9), (16, 9), (16, 5), (13, 5), (7, 5), (7, 9), (7, 12), (2, 12), (2, 9), (7, 18), (7, 24), (7, 27), (7, 30), (2, 30), (2, 33), (13, 33), (16, 33), (27, 33), (19, 30), (16, 30), (13, 30), (10, 30), (10, 27), (10, 24), (13, 27), (2, 24), (13, 24), (2, 5), (27, 12), (10, 18), (10, 15), (19, 21), (19, 18), (19, 15), (10, 21)]


def next_cross(pacman_pos, pacman_dir):
    if pacman_pos[1] == 18 and (pacman_pos[0] < 7 or pacman_pos[0] > 22):
        return np.nan
    if pacman_dir == "right":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[0] + 1, 35):
                if (i, pacman_pos[1]) in corner_pos:
                    p = (i, pacman_pos[1])
                    break
        else:
            p = pacman_pos
        return p
    
    if pacman_dir == "left":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[0] - 1, 0, -1):
                if (i, pacman_pos[1]) in corner_pos:
                    p = (i, pacman_pos[1])
                    break
        else:
            p = pacman_pos
        return p
    
    if pacman_dir == "up":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[1] - 1, 0, -1):
                if (pacman_pos[0], i) in corner_pos:
                    p = (pacman_pos[0], i)
                    break
        else:
            p = pacman_pos
        return p
    
    if pacman_dir == "down":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[1] + 1, 35):
                if (pacman_pos[0], i) in corner_pos:
                    p = (pacman_pos[0], i)
                    break
        else:
            p = pacman_pos
        return p
    


def dir_choices(pos):
    if not isinstance(pos, float) and not pd.isnull(pos):
        return pd.Series(["up", "left", "down", "right"])[
            map_info.loc[
                map_info.pacmanPos == pos,
                ["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"],
            ]
            .astype(bool)
            .values[0]
        ].values.tolist()
    

def next_cross_dirs(pacman_pos, pacman_dir):
    if pacman_dir == "right":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[0] + 1, 35):
                if (i, pacman_pos[1]) in corner_pos:
                    p = (i, pacman_pos[1])
                    break
        else:
            p = pacman_pos
    try:
        a = pd.Series(["up", "left", "down", "right"])[
                map_info.loc[
                    map_info.pacmanPos == p,
                    ["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"],
                ]
                .astype(bool)
                .values[0]
            ].values.tolist()
        a.remove('left')
        return a
    except:
        pass
    
    if pacman_dir == "left":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[0] - 1, 0, -1):
                if (i, pacman_pos[1]) in corner_pos:
                    p = (i, pacman_pos[1])
                    break
        else:
            p = pacman_pos
    try:
        a = pd.Series(["up", "left", "down", "right"])[
                map_info.loc[
                    map_info.pacmanPos == p,
                    ["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"],
                ]
                .astype(bool)
                .values[0]
            ].values.tolist()
        a.remove('right')
        return a
    except:
        pass
    if pacman_dir == "up":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[1] - 1, 0, -1):
                if (pacman_pos[0], i) in corner_pos:
                    p = (pacman_pos[0], i)
                    break
        else:
            p = pacman_pos
    try:
        a = pd.Series(["up", "left", "down", "right"])[
                map_info.loc[
                    map_info.pacmanPos == p,
                    ["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"],
                ]
                .astype(bool)
                .values[0]
            ].values.tolist()
        a.remove('down')
        return a
    except:
        pass
    if pacman_dir == "down":
        if pacman_pos not in corner_pos:
            for i in range(pacman_pos[1] + 1, 35):
                if (pacman_pos[0], i) in corner_pos:
                    p = (pacman_pos[0], i)
                    break
        else:
            p = pacman_pos
    try:
        a = pd.Series(["up", "left", "down", "right"])[
                map_info.loc[
                    map_info.pacmanPos == p,
                    ["Next1Pos2", "Next2Pos2", "Next3Pos2", "Next4Pos2"],
                ]
                .astype(bool)
                .values[0]
            ].values.tolist()
        a.remove('up')
        return a
    except:
        pass
    
    
    
    
    
def next_cross_nearbean(pos, next_pos, curr_beans):
    try:
        beans_new = [i for i in curr_beans if i not in nx.shortest_path(G, pos, next_pos)]
        tmp_df = locs_df[(locs_df.pos1 == p) & (locs_df.pos2.isin(beans_new))]
        near_pts = tmp_df[tmp_df.dis == tmp_df.dis.min()].pos2.values
        return list(itertools.chain(*[relative_dir(i, pos) for i in near_pts]))
    except:
        pass


def go_straight(pos, curr_dir, beans, iflocal=True):
    try:
        if iflocal:
            return bool(
                set(
                    dijkstra_distance(
                        x.pacmanPos, next_cross(x.pacmanPos, x.pacman_dir)
                    )[:11]
                )
                & set(x.beans)
            )
        else:
            return bool(
                set(
                    dijkstra_distance(
                        x.pacmanPos, next_cross(x.pacmanPos, x.pacman_dir)
                    )[11:]
                )
                & set(x.beans)
            )
    except:
        return False

    
    
dictx = {
    pd.Interval(1, 10, closed="right"): 0,
    pd.Interval(10.0, 18.0, closed="right"): 1,
    pd.Interval(18.0, 27.0, closed="right"): 2,
}
dicty = {
    pd.Interval(4.0, 14.0, closed="right"): 0,
    pd.Interval(14.0, 23.0, closed="right"): 1,
    pd.Interval(23.0, 33.0, closed="right"): 2,
}

xedges = np.linspace(1, 27, 4).round()
yedges = np.linspace(4, 33, 4).round()

forbidden_pos = list(map(lambda x: (x, 18), list(range(7)) + list(range(23, 30))))

def global_pos(pos):
    if not isinstance(pos, float) and pos not in forbidden_pos and pos is not None:
        return (
            pd.Series(pd.cut([pos[0]], xedges)).replace(dictx).values[0][0],
            pd.Series(pd.cut([pos[1]], yedges)).replace(dicty).values[0][0],
        )
    else:
        return np.nan


def dirs_global(pos, beans, next_pos, atcross=False):
    if isinstance(beans, float) or isinstance(pos, float) or pos in forbidden_pos:
        return np.nan
    if atcross:
        try:
            beans = list(set(beans).difference(set(nx.shortest_path(G, pos, next_pos))))
        except:
            return np.nan
    l = pd.value_counts([relative_dir(global_pos(b), global_pos(pos)) for b in beans])
    return list(
        itertools.chain(*l.where(lambda x: x == l.max()).dropna().index.tolist())
    )


                
def extend_has_beans(file, pacman_pos, pacman_dir, Rewards_dict):
    temp = Rewards_dict[file]
    if pacman_dir == "right":
        for i in range(pacman_pos[0] + 1, 35):
            if (i, pacman_pos[1]) in corner_pos:
                end_pos = (i, pacman_pos[1])
                break
        try:
            return (
                temp[
                    (temp.X.isin(list(range(pacman_pos[0], end_pos[0] + 1))))
                    & (temp.Y == pacman_pos[1])
                ].shape[0]
                > 0
            )
        except:
            pass
    if pacman_dir == "left":
        for i in range(pacman_pos[0] - 1, 0, -1):
            if (i, pacman_pos[1]) in corner_pos:
                end_pos = (i, pacman_pos[1])
                break
        try:
            return (
                temp[
                    (temp.X.isin(list(range(end_pos[0], pacman_pos[0]))))
                    & (temp.Y == pacman_pos[1])
                ].shape[0]
                > 0
            )
        except:
            pass
    if pacman_dir == "up":
        for i in range(pacman_pos[1] - 1, 0, -1):
            if (pacman_pos[0], i) in corner_pos:
                end_pos = (pacman_pos[0], i)
                break
        try:
            return (
                temp[
                    (temp.Y.isin(list(range(end_pos[1], pacman_pos[1]))))
                    & (temp.X == pacman_pos[0])
                ].shape[0]
                > 0
            )
        except:
            pass
    if pacman_dir == "down":
        for i in range(pacman_pos[1] + 1, 35):
            if (pacman_pos[0], i) in corner_pos:
                end_pos = (pacman_pos[0], i)
                break
        try:
            return (
                temp[
                    (temp.Y.isin(list(range(pacman_pos[1], end_pos[1] + 1))))
                    & (temp.X == pacman_pos[0])
                ].shape[0]
                > 0
            )
        except:
            pass



map_info_mapping = {
    "up": "Next1Pos",
    "left": "Next2Pos",
    "down": "Next3Pos",
    "right": "Next4Pos",
}


def maptodict(ghost_pos):
    d_dict = {}
    for d in ["up", "down", "right", "left"]:
        pos = tuple(
            map_info.loc[
                map_info.pacmanPos == ghost_pos,
                [map_info_mapping[d] + "1", map_info_mapping[d] + "2"],
            ].values[0]
        )
        if pos != (0, 0):
            d_dict[d] = pos
    return d_dict




def future_position(ghost_pos, ghost_dir, t, pacman_pos):
    if t == 0:
        return ghost_pos
    history = [ghost_pos]
    for i in range(t // 2):
        d_dict = {
            key: val for key, val in maptodict(ghost_pos).items() if val not in history
        }
        if i == 0 and ghost_dir in d_dict.keys():
            ghost_pos = d_dict[ghost_dir]
        else:
            dict_df = pd.DataFrame.from_dict(d_dict, orient="index")
#             if dict_df.empty:
#                 set_trace()
#                 print ('empty')
            dict_df["poss_pos"] = tuple_list(dict_df[[0, 1]].values)
            try:
                ghost_dir, ghost_pos = (
                    locs_df[(locs_df.pos1 == pacman_pos)]
                    .merge(dict_df.reset_index(), left_on="pos2", right_on="poss_pos")
                    .sort_values(by="dis")[["index", "poss_pos"]]
                    .values[-1]
                )
            except:
                return pacman_pos
        history.append(ghost_pos)
    return ghost_pos
        
        
def label_gta(df_model):
    df_model.loc[df_model["toward_each_other" + w] == 1, "label"] = "toward"
    df_model.loc[
        (df_model["toward_each_other" + w] == 0) & (df_model["pac_to_ghost" + w] == 1),
        "label",
    ] = "pafterg"
    df_model.loc[
        (df_model["toward_each_other" + w] == 0)
        & (df_model["pac_to_ghost" + w] == 0)
        & (df_model["distance" + w].shift() == df_model["distance" + w]),
        "label",
    ] = "gafterp"
    df_model.label = df_model.label.fillna("dispatch")
    return df_model

def upzip(path_to_zip_file, directory_to_extract_to):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)