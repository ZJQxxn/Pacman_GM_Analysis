import numpy as np
import pandas as pd
import sys
import pickle
import copy
import warnings

warnings.filterwarnings('ignore')

np.set_printoptions(suppress=True)
pd.set_option("display.max_rows", 200)
sys.path = sys.path + ["/home/qlyang/Documents/pacman/"]

from helper.pacmanutils import combine_pre_post




monkey = "Patamon"

dall = pickle.load(open("../constants/all_data_new.pkl", "rb"))
map_indexes_accident = dall["map_indexes_accident"]
map_indexes_plan = dall["map_indexes_plan"]
cons_list_plan = dall["cons_list_plan"]
cons_list_accident = dall["cons_list_accident"]
del dall


df_total_small = pd.read_pickle("/home/qlyang/pacman/PacmanAgent/constant/all_trial_data-window3-path10.pkl")
df_total = copy.deepcopy(df_total_small[df_total_small.file.str.contains(monkey)]).sort_index()

print("All data shape : ", df_total.shape)
print("Finsihed processing.")
print("="*50)

cons_list_plan = [list(each) for each in cons_list_plan]
cons_list_accident = [list(each) for each in cons_list_accident]
map_indexes_plan = [list(each) for each in map_indexes_plan]
map_indexes_accident = [list(each) for each in map_indexes_accident]
print("Shape of planned hunting list : ", len(cons_list_plan))
print("Shape of accidental list : ", len(cons_list_accident))
print("Shape of map planned : ", len(map_indexes_plan))
print("Shape of map accident : ", len(map_indexes_accident))


print("Finished configuring.")
print("="*50)


# Fig. 16.1 (sacc freq)
print("="*50)
print("For Fig. 16.1 (sacc freq): ")
mapping = {
        "Planned Attack": cons_list_plan, #.values,
        "Accidental Attack": cons_list_accident # .values,
}

l = []
cnt = 0
for i in [
    "pacman_sacc",
    "ghost1Pos_sacc",
    "ghost2Pos_sacc",
]:
    for key, sel_index in mapping.items():
        temp = []
        for s in sel_index:
            try:
                temp.append(df_total.iloc[s[-7:]][i].sum() / 3)
            except:
                cnt+=1
                continue
        # sacc_per = [df_total.iloc[s[-7:]][i].sum() / 3 for s in sel_index]
        sacc_per = temp
        l.append(
            [
                i.split("_")[0],
                np.mean(sacc_per),
                np.std(sacc_per) / np.sqrt(len(sacc_per)),
            ]
        )
    print("Cnt : ", cnt)
df_plot = pd.DataFrame(l, columns=["agent", "mean", "std"]).assign(
    status=list(mapping.keys()) * 3
)
# df_plot = df_plot.pivot(index="agent", columns="status", values="mean").loc[:, ::-1]
print("Data shape : ", df_plot.shape)
# save
print("Data shape : ", df_plot.shape)
with open("./plot_data/{}_11.6.1.saccade.pkl".format(monkey), "wb") as file:
    pickle.dump(df_plot, file)
print("="*50)
# plt.savefig("../" + name + "pics/" + save_name + ".pdf", bbox_inches="tight")


# Fig. 16.1 (pupil size)
print("For Fig. 16.1 (pupil size): ")

def rt_before_after_eye(last_index_list, df_total, rt, cutoff, col, cond_col=None):
    after_df, before_df = pd.DataFrame(), pd.DataFrame()
    for i in last_index_list:
        file, index = df_total.iloc[i]["file"], df_total.iloc[i]["index"]
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

accident_all = combine_pre_post(cons_list_accident, map_indexes_accident)
prehunt_all = combine_pre_post(cons_list_plan, map_indexes_plan)
cutoff = 10
all_data = {"accidental":None, "planned":None}
name = ["accidental", "planned"]
cnt = 0
for index, compute_list in enumerate([prehunt_all, accident_all]):
    temp = []
    for i in compute_list:
        try:
            if max(df_total.iloc[i[-1] + 1][["ifscared1", "ifscared2"]] == 3):
                temp.append(i[-1])
        except:
            cnt+=1
            continue
    print("Cnt :", cnt)

    after_df, before_df = rt_before_after_eye(
        temp,
        df_total,
        df_total,
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
    print("Data shape of {} is {}".format(name[index], df_plot.shape))
    all_data[name[index]] = copy.deepcopy(df_plot)
# Save data
with open("./plot_data/{}_11.6.1.pupil.pkl".format(monkey), "wb") as file:
    pickle.dump(all_data, file)
print("="*50)
# plt.savefig("../" + name + "pics/4E.pdf", bbox_inches="tight")


# =============================================================================
#               FOR FIG 16.2 (SUICIDE AND NORMAL DEAD)
# =============================================================================

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

if "level_0" in df_total.columns.values:
    df_total = df_total.drop(columns="level_0")
suicide_lists, normal_lists = generate_suicide_normal(df_total)
print("Num of suicide data : ", len(suicide_lists))
print("Num of suicidenormala : ", len(normal_lists))


# Fig. 16.2 (sacc freq)
print("For Fig. 16.2 (pupil):")
all_data = {"suicide":None, "normal":None}
name = ["suicide", "normal"]
xaxis = range(10, 1, -1)
for index, compute_list in enumerate([suicide_lists, normal_lists]):
    data = [
        [
            df_total.iloc[j[-i]]["eye_size_std2"]
            for j in compute_list
            if i <= len(j) and df_total.iloc[j[-i]]["eye_size"] != 0
        ]
        for i in xaxis
    ]
    gpd = pd.DataFrame(
        [[np.mean(i), np.std(i), len(i)] for i in data],
        columns=["mean", "std", "count"],
    )
    gpd.index = [round(-(i - 1) * 25 / 60, 2) for i in xaxis]
    all_data[name[index]] = gpd
    print("Data shape for {} : {}".format(name[index], gpd.shape))
# save data
with open("./plot_data/{}_11.6.2.pupil.pkl".format(monkey), "wb") as file:
    pickle.dump(all_data, file)
print("="*50)


print("For Fig. 16.2 (saccade):")
mapping = {
        "Suicide": suicide_lists.values,
        "Normal Die": normal_lists.values,
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
# df_plot = df_plot.pivot(index="agent", columns="status", values="mean").loc[:, ::-1]
print("Data shape : ", df_plot.shape)
# save data
with open("./plot_data/{}_11.6.2.saccade.pkl".format(monkey), "wb") as file:
    pickle.dump(df_plot, file)
# plt.savefig("../" + name + "pics/" + save_name + ".pdf", bbox_inches="tight")
