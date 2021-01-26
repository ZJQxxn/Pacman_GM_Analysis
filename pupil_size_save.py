import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import pickle
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
            diss.set_index("level_0")[["PE_dis", "EG_dis", "EG1_dis", "EG2_dis"]],
        ],
        1,
    )
    return df_total



dall = pickle.load(open("../constants/all_data_new.pkl", "rb"))
map_indexes_accident = dall["map_indexes_accident"]
map_indexes_plan = dall["map_indexes_plan"]
del dall


df_total_small = pd.read_pickle("/home/qlyang/pacman/PacmanAgent/constant/all_trial_data-window3-path10.pkl")
df_total = add_PEG_dis(df_total_small)
df_total.next_pacman_dir_fill = df_total.next_pacman_dir_fill.explode()
df_total.pacman_dir_fill = df_total.pacman_dir_fill.explode()

energizer_start_index = df_total[
    (df_total.eat_energizer == True)
    & (df_total[["ifscared1", "ifscared2"]].min(1).shift() < 3)
][["next_eat_rwd", "energizers", "ifscared1", "ifscared2"]].index
energizer_lists = [
    np.arange(
        i,
        (df_total.loc[i:, ["ifscared1", "ifscared2"]] <= 3)
        .max(1)
        .where(lambda x: x == True)
        .dropna()
        .index[0],
    )
    for i in energizer_start_index
]

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
            df_total["pacmanPos"].reset_index(),
            left_on="prev_index",
            right_on="index",
            how="left",
        )
        .drop(columns="index")
        .merge(
            df_total["pacmanPos"].reset_index(),
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
    if (df_total.loc[i[:5], "labels"] == "approach").sum() > 0 and len(i) > 0
]
cons_list_accident = [
    np.arange(pre_index[i[0]], i[0])
    for i in energizer_lists
    if (df_total.loc[i[:5], "labels"] == "approach").sum() == 0 and len(i) > 0
]


# Fig. 16.1 (sacc freq)

print("="*50)
print("For Fig. 16.1 (sacc freq): ")
mapping = {
        "Planned Attack": cons_list_plan.values,
        "Accidental Attack": cons_list_accident.values,
}

l = []
for i in [
    "pacman_sacc",
    "ghost1Pos_sacc",
    "ghost2Pos_sacc",
]:
    for key, sel_index in mapping.items():
        sacc_per = [df_total.loc[s[-7:], i].sum() / 3 for s in sel_index]
        l.append(
            [
                i.split("_")[0],
                np.mean(sacc_per),
                np.std(sacc_per) / np.sqrt(len(sacc_per)),
            ]
        )

df_plot = pd.DataFrame(l, columns=["agent", "mean", "std"]).assign(
    status=list(mapping.keys()) * 3
)
data = df_plot.pivot(index="agent", columns="status", values="mean").loc[:, ::-1]
# save
print("Data shape : ", data.shape)
print("="*50)

# ax = (
#     df_plot.pivot(index="agent", columns="status", values="mean")
#         .loc[:, ::-1]
#         .plot(
#         kind="bar",
#         yerr=df_plot.pivot(index="agent", columns="status", values="std").values.T,
#         color=plt.cm.Set1.colors[:2][::-1],
#     )
# )
# plt.ylabel("Average saccade frequency")
# plt.xlabel("saccade subject")
# plt.legend(title=None)
# ax.set_xticklabels(["Ghost Blinky", "Ghost Clyde", "PacMan Self"], rotation=0)
# plt.savefig("../" + name + "pics/" + save_name + ".pdf", bbox_inches="tight")





# Fig. 16.1 (pupil size)
print("For Fig. 16.1 (pupil size): ")

def rt_before_after_eye(last_index_list, df_total, rt, cutoff, col, cond_col=None):
    after_df, before_df = pd.DataFrame(), pd.DataFrame()
    for i in last_index_list:
        file, index = df_total.loc[i, "file"], df_total.loc[i, "index"]
        before = rt[(rt.file == file) & (rt["index"] < index)].iloc[-cutoff:][
            [col, cond_col]
        ]
        before = before[col].mask(before[cond_col] == 0)
        before = pd.Series(before.values, index=range(before.shape[0], 0, -1))
        before_df = pd.concat([before_df, before], 1)

        after = rt[(rt.file == file) & (rt["index"] > index)].iloc[:cutoff][
            [col, cond_col]
        ]
        after = after[col].mask(after[cond_col] == 0)
        after = pd.Series(after.values, index=range(1, after.shape[0] + 1))
        after_df = pd.concat([after_df, after], 1)
    return after_df, before_df

temp_total = copy.deepcopy(df_total[df_total.file.str.contains("-")])
cons_list_accident = cons_list_accident.apply(lambda x: x if x[-1] < 10e10 else np.nan).dropna().values,
map_indexes_accident = map_indexes_accident.apply(lambda x: x if x[-1] < 10e10 else np.nan).dropna().values
cons_list_plan = cons_list_plan.apply(lambda x: x if x[-1] < 10e10 else np.nan).dropna().values
map_indexes_plan = map_indexes_plan.apply(lambda x: x if x[-1] < 10e10 else np.nan).dropna().values

accident_all = combine_pre_post(cons_list_accident, map_indexes_accident)
prehunt_all = combine_pre_post(cons_list_plan, map_indexes_plan)
cutoff = 10
data = {"prehunt":None, "accident":None}
name = ["prehunt", "accident"]
for index, compute_list in enumerate([prehunt_all, accident_all]):
    after_df, before_df = rt_before_after_eye(
        [
            i[-1]
            for i in compute_list
            if max(temp_total.loc[i[-1] + 1, ["ifscared1", "ifscared2"]] == 3)
        ],
        temp_total,
        temp_total,
        cutoff,
        "eye_size_std2",
        cond_col="eye_size",
    )
    after_sts = pd.DataFrame(
        {
            "mean": after_df.mean(1).values,
            "std": after_df.std(1).values,
            "count": after_df.count(1).values,
        },
        index=range(1, cutoff + 1),
    )
    before_sts = pd.DataFrame(
        {
            "mean": before_df.mean(1).values,
            "std": before_df.std(1).values,
            "count": before_df.count(1).values,
        },
        index=range(-1, -cutoff - 1, -1),
    )
    df_plot = before_sts.append(after_sts).sort_index()
    print("Data shape : ", df_plot.shape)
    data[name[index]] = df_plot
# Save data
print("="*50)

#     plt.errorbar(
#         df_plot.index,
#         df_plot["mean"],
#         yerr=df_plot["std"] / np.sqrt(df_plot["count"]),
#         marker=None,
#         capsize=3,
#         barsabove=True,
#     )
# #     plt.xticks(df_plot.index[::2], [round(i * 25 / 60, 2) for i in df_plot.index[::2]])
# plt.ylabel("relative pupil size")
# plt.xlabel("seconds before or after eat ghost")
# plt.legend(["planned attack", "accidental attack"])
# plt.savefig("../" + name + "pics/4E.pdf", bbox_inches="tight")

# Fig. 16.2 (sacc freq)
print("For Fig. 16.2 (sacc freq):")

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

suicide_lists, normal_lists = generate_suicide_normal(df_total)

all_data = {"suicide":None, "normal":None}
name = ["suicide", "normal"]
for index, compute_list in enumerate([suicide_lists, normal_lists]):
    i = 2
    data = pd.Series(
        [
            df_total.loc[j[-i], "eye_size_std2"]
            for j in compute_list
            if i <= len(j) and df_total.loc[j[-i], "eye_size"] != 0
        ]
    )
    print("Data shape : ", data.shape)
    all_data[name[index]] = data
# save data
print("="*50)

#     data.hist(
#         bins=range(-2000, 2000, 100),
#         alpha=0.7,
#         weights=np.ones(data.shape[0]) / data.shape[0],
#     )
#     plt.grid(False)
#     plt.legend(["suicide", "normal die"])
#     plt.xlabel("pupil size - average pupil size per trial")
#     plt.ylabel("normalized count")
#     plt.xlim(-2000, 2000)
# plt.savefig("../" + name + "pics/5A1histogram.pdf", bbox_inches="tight")
# xaxis = range(10, 1, -1)
# sns.set_palette(plt.cm.Set1.colors[:2][::-1])
# plt.subplots()

xaxis = range(10, 1, -1)
all_data = {"suicide":None, "normal":None}
name = ["suicide", "normal"]
for compute_list in [suicide_lists, normal_lists]:
    data = [
        [
            df_total.loc[j[-i], "eye_size_std2"]
            for j in compute_list
            if i <= len(j) and df_total.loc[j[-i], "eye_size"] != 0
        ]
        for i in xaxis
    ]
    gpd = pd.DataFrame(
        [[np.mean(i), np.std(i), len(i)] for i in data],
        columns=["mean", "std", "count"],
    )
    gpd.index = [round(-(i - 1) * 25 / 60, 2) for i in xaxis]
    print("Data shape : ", gpd.shape)
    all_data[name[index]] = gpd
# save data
print("="*50)

#
#     plt.errorbar(
#         gpd.index,
#         gpd["mean"],
#         yerr=gpd["std"] / np.sqrt(gpd["count"]),
#         marker=None,
#         capsize=3,
#         barsabove=True,
#     )
# plt.legend(["suicide", "normal die"])
# plt.ylabel("relative pupil size")
# plt.xlabel("time before getting caught")
# plt.savefig("../" + name + "pics/5A1line.pdf", bbox_inches="tight")





# plot_4C5C(
#     df_total[df_total.file.str.contains("Patamon")],
#     {
#         "Suicide": suicide_lists.apply(lambda x: x if x[0] >= 686225 else np.nan)
#         .dropna()
#         .values,
#         "Normal Die": normal_lists.apply(lambda x: x if x[0] >= 686225 else np.nan)
#         .dropna()
#         .values,
#     },
#     "5C",
#     "patamon",
# )

temp_total = df_total[df_total.file.str.contains("Patamon")]
maping = {
        "Suicide": suicide_lists.apply(lambda x: x if x[0] >= 686225 else np.nan)
        .dropna()
        .values,
        "Normal Die": normal_lists.apply(lambda x: x if x[0] >= 686225 else np.nan)
        .dropna()
        .values,
    }
l = []
for i in [
    "pacman_sacc",
    "ghost1Pos_sacc",
    "ghost2Pos_sacc",
]:

    for key, sel_index in mapping.items():
        sacc_per = [temp_total.loc[s[-7:], i].sum() / 3 for s in sel_index]
        l.append(
            [
                i.split("_")[0],
                np.mean(sacc_per),
                np.std(sacc_per) / np.sqrt(len(sacc_per)),
            ]
        )

df_plot = pd.DataFrame(l, columns=["agent", "mean", "std"]).assign(
    status=list(mapping.keys()) * 3
)
df_plot = df_plot.pivot(index="agent", columns="status", values="mean").loc[:, ::-1]
print("Data shape : ", df_plot.shape)

# ax = (
#     df_plot.pivot(index="agent", columns="status", values="mean")
#         .loc[:, ::-1]
#         .plot(
#         kind="bar",
#         yerr=df_plot.pivot(index="agent", columns="status", values="std").values.T,
#         color=plt.cm.Set1.colors[:2][::-1],
#     )
# )
# plt.ylabel("Average saccade frequency")
# plt.xlabel("saccade subject")
# plt.legend(title=None)
# ax.set_xticklabels(["Ghost Blinky", "Ghost Clyde", "PacMan Self"], rotation=0)
# plt.savefig("../" + name + "pics/" + save_name + ".pdf", bbox_inches="tight")




