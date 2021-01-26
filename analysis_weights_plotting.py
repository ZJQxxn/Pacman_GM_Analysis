import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import pickle
import copy
import scipy.stats

from palettable.colorbrewer.diverging import RdBu_7
from palettable.tableau import Tableau_10


# Configurations
params = {
    "legend.fontsize": 14,
    "legend.frameon": False,
    "ytick.labelsize": 14,
    "xtick.labelsize": 14,
    # "figure.dpi": 200,
    "axes.prop_cycle": plt.cycler("color", plt.cm.Accent(np.linspace(0, 1, 5))),
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    # "patch.force_edgecolor": False,
}
plt.rcParams.update(params)
pd.set_option("display.float_format", "{:.5f}".format)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

status_color_mapping = {
    "approach": "#836bb7",
    "energizer": "#81b3ff",
    "global": "#44b53c",
    "evade": "#fdae61",
    "evade_blinky": "#fdae61",
    "evade_clyde": "#c78444",
    "evade-blinky": "#fdae61",
    "evade-clyde": "#c78444",
    "local": "#d7181c",
    "vague": "#929292",
}
agent_name = ["global", "local", "evade_blinky", "evade_clyde", "approach", "energizer"]

print("Finished configuration.")
print("="*50)

# Monkey
monkey = "Omega"


def plotSaccByWeight():
    with open("./plot_data/{}_2A2_sacc_by_weight.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    for agent in [
        "global_weight",
        "local_weight",
        "evade_blinky_weight",
        "evade_clyde_weight",
        "approach_weight",
        "energizer_weight",
    ]:
        plt.figure()
        plt.title(agent)
        for index, sacc in enumerate([
            "pacman_sacc",
            "ghost",
            "for_sacc",
            "beans_sacc",
        ]):
            each_data = data[agent][index]
            each_data.plot(
                    marker="o"
                )
        plt.xticks([1, 3, 5, 7, 9], [0.1, 0.3, 0.5, 0.7, 0.9])
        plt.legend(
            ["pacman_sacc", "ghost", "for_sacc", "beans_sacc", ]
        )
        plt.savefig("./plot_data/" + monkey + "/2A2_sacc_by_weight_" + agent + ".pdf")
        plt.show()


def plotAvgSaccByWeight():
    with open("./plot_data/{}_2A2_avg_sacc_by_weight.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    for agent in [
        "global_weight",
        "local_weight",
        "evade_blinky_weight",
        "evade_clyde_weight",
        "approach_weight",
        "energizer_weight",
    ]:
        # plt.figure()
        plt.title(agent)
        each_data = data[agent]
        each_data.pivot(index="upper", columns="sacc_type", values="mean").plot(
            yerr=each_data.pivot(index="upper", columns="sacc_type", values="std").values.T,
            color=plt.cm.Set1.colors[:4][::-1],
            title=agent,
        )
        plt.ylabel("Average saccade frequency")
        plt.xlabel("weight")
        plt.legend(title=None, ncol=2)
        plt.savefig("./plot_data/" + monkey + "/2A2_avg_sacc_by_weight_" + agent + ".pdf")
        plt.show()
        plt.clf()


def plotFig8Freq():
    with open("./plot_data/{}_8_avg_sacc_freq.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    plt.figure(figsize=(13, 5))
    plt.bar(
        data.status,
        data["mean"],
        yerr=data["std"],
        color=[status_color_mapping[c] for c in data.status],
    )
    plt.ylim(0, 3)
    plt.ylabel("Average saccade frequency")
    plt.xlabel("saccade subject")
    plt.legend(title=None, ncol=3)
    plt.savefig("./plot_data/" + monkey + "/sacc_based_on_weights_combine.pdf")
    plt.show()


def plotFig8AvgStd():
    with open("./plot_data/{}_8_avg_std.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    ax = data["mean"].plot(
        kind="bar",
        yerr=data["std"].values.T,
        color=[status_color_mapping[c] for c in data["mean"].columns],
        legend=None,
    )
    plt.ylim(0, 2)

    # plt.text(-0.3, 1.8, s= "Local", fontdict={"fontsize":10, "color":status_color_mapping["local"], "weight":"bold"})
    # plt.text(0.2, 1.8, s= "Global", fontdict={"fontsize":10, "color":status_color_mapping["global"], "weight":"bold"})
    # plt.text(0.5, 1.8, s= "Evade(Blinky)", fontdict={"fontsize":10, "color":status_color_mapping["evade_blinky"], "weight":"bold"})
    # plt.text(1.5, 1.8, s= "Evade(Clyde)", fontdict={"fontsize":10, "color":status_color_mapping["evade_clyde"], "weight":"bold"})
    # plt.text(2.0, 1.8, s= "Energizer", fontdict={"fontsize":10, "color":status_color_mapping["energizer"], "weight":"bold"})
    # plt.text(2.5, 1.8, s= "Approach", fontdict={"fontsize":10, "color":status_color_mapping["approach"], "weight":"bold"})
    # plt.text(3.0, 1.8, s= "Vague", fontdict={"fontsize":10, "color":status_color_mapping["vague"], "weight":"bold"})




    plt.xticks(np.arange(4), ["Pellets", "Forward", "Ghosts", "PacMan"], rotation=0, fontsize = 15)
    plt.ylabel("Average Saccade Frequency", fontsize = 20)
    plt.xlabel("Saccade Identity", fontsize = 20)
    plt.tight_layout()
    plt.savefig("./plot_data/" + monkey + "/8.pdf")
    plt.show()


def plotFig9():
    with open("./plot_data/{}_9_avg_pupil_size.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    plt.figure(figsize=(17, 8))
    plt.bar(
        data.status,
        data["mean"],
        yerr=data["std"],
        color=[status_color_mapping[c] for c in data.status],
    )
    #     plt.ylim(0, 3)
    plt.ylabel("Average Pupil Size", fontsize = 20)
    plt.yticks(fontsize = 15)
    plt.xticks(np.arange(7), ["local", "global", "evade(Blinky)", "evade(Clye)", "energizer", "approach", "vague"], fontsize = 20)
    plt.legend(title=None, ncol=3)
    plt.savefig("./plot_data/" + monkey + "/9.pdf") # _avg_pupil_size_z
    plt.show()


def plotFig10():
    type_name = ["Corner", "T-Junction", "Cross", "Tunnel", "All"]
    for index, type in enumerate(["corners", "t_junctions", "cross", "tunnel", "all"]):
        with open("./plot_data/{}_label_rt_{}.pkl".format(monkey, type), "rb") as file:
            data = pickle.load(file)
        plt.figure(figsize=(15, 8))
        plt.title(type_name[index], fontsize = 20)
        plt.bar(
            data.index,
            data["mean"],
            yerr=data["std"] / np.sqrt(data["size"]),
            color=[status_color_mapping[c] for c in data.index],
        )
        for i in range(7):
            if i in [0, 1]:
                plt.text(i - 0.15, data["mean"].values[i] + 0.02, data["size"].values[i], fontdict={"fontsize": 20})
            else:
                plt.text(i-0.11, data["mean"].values[i]+0.02, data["size"].values[i], fontdict={"fontsize":20})
        plt.ylabel("Joystick Lead Time", fontsize=20)
        plt.xticks(np.arange(7), ["local", "global", "evade(Blinky)", "evade(Clye)", "energizer", "approach", "vague"],
                   fontsize=20)
        plt.ylim(0, np.max(data["mean"]+0.1))
        plt.yticks(fontsize = 20)
        plt.tight_layout()
        plt.savefig("./plot_data/" + monkey + "/10_{}.pdf".format(type))  # _label_rt
        plt.show()


def plotFig74():
    with open("./plot_data/{}_74.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
        indices = copy.deepcopy(data.index.values)
        indices[-3] = "evade(Blinky)"
        indices[-1] = "evade(Clyde)"
    # data.plot(kind="bar", color=[status_color_mapping[c] for c in data.index])
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(7), data.values, color = [status_color_mapping[c] for c in data.index])
    plt.xticks(np.arange(7), indices, fontsize = 15)
    plt.xlabel("Dominating Strategy", fontsize = 20)
    plt.ylabel("Probability", fontsize=20)
    plt.tight_layout()
    plt.savefig("./plot_data/" + monkey + "/7.4.pdf")
    plt.show()


def plotFig71():
    with open("./plot_data/{}_71.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    cols = data.columns.values

    color = Tableau_10.mpl_colors
    plt.figure(figsize=(15, 10))
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    r_data = []
    for index, c in enumerate(cols):
        temp_data = data[c].values
        for i,j in enumerate(temp_data):
            if j < 0.01:
                continue
            else:
                r_data.extend([i/10+0.05 for _ in range(int(100*j))])
        sns.distplot(r_data, kde=False, bins=bins, label=cols[index], color=color[2-index],hist_kws={"edgecolor": color[2-index]}, norm_hist=True)
        r_data = []

    # sns.distplot(data[[cols[1]]], kde=False, bins=bins, label=cols[1], color=color[1], hist_kws={"edgecolor": color[1]}, norm_hist=True)
    # sns.distplot(data[[cols[2]]], kde=False, bins=bins, label=cols[2], color=color[0], hist_kws={"edgecolor": color[0]}, norm_hist=True)
    plt.ylabel("Probability", fontsize = 20)
    plt.xlabel("Normalized Weight", fontsize=20)
    plt.yticks([0, 2, 4, 6, 8, 10], [0.0, .2, .4, .6, .8, 1.0], fontsize = 20)
    plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 20)
    plt.xlim(0, 1.0)
    plt.legend(fontsize = 20)
    plt.savefig("./plot_data/" + monkey + "/7.1.pdf")
    plt.show()


def plotFig75():
    # in vague
    with open("./plot_data/{}_75.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
        indices = []
        for each in data.index.values[:6]:
            temp = list(each)
            if "evade-blinky" == temp[0]:
                temp[0] = "evade(B)"
            if "evade-clyde" == temp[0]:
                temp[0] = "evade(C)"
            if "evade-blinky" == temp[1]:
                temp[1] = "evade(B)"
            if "evade-clyde" == temp[1]:
                temp[1] = "evade(C)"
            indices.append("{}-{}".format(temp[0], temp[1]))
        data = data.iloc[:6].values

    plt.figure(figsize = (13, 5))
    plt.bar(np.arange(6), data)
    plt.xticks(np.arange(6), indices, fontsize=15)
    plt.xlabel("Transition strategies in Vague", fontsize=20)
    plt.ylabel("Probability", fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    plt.savefig("./plot_data/" + monkey + "/7.5.pdf")
    plt.show()


def plotFig73():
    # around vague
    with open("./plot_data/{}_73.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
        indices = []
        for each in data.index.values[:6]:
            temp = list(each)
            if "evade-blinky" == temp[0]:
                temp[0] = "evade(B)"
            if "evade-clyde" == temp[0]:
                temp[0] = "evade(C)"
            if "evade-blinky" == temp[1]:
                temp[1] = "evade(B)"
            if "evade-clyde" == temp[1]:
                temp[1] = "evade(C)"
            indices.append("{}-{}".format(temp[0], temp[1]))
        data = data.iloc[:6].values

    plt.figure(figsize = (13, 5))
    plt.bar(np.arange(6), data)
    plt.xticks(np.arange(6), indices, fontsize=15)
    plt.xlabel("Transition strategies around Vague", fontsize=20)
    plt.ylabel("Probability", fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.tight_layout()
    plt.savefig("./plot_data/" + monkey + "/7.3.pdf")
    plt.show()


def plotFig72():
    with open("./plot_data/{}_72.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    data["series"].hist(
        grid=False, weights=np.ones_like(data["values"][:, 0]) / data["values"].shape[0], figsize=(8,6)
    )
    plt.xlabel("Largest strategy weight - 2nd largest strategy weight", fontsize = 20)
    plt.ylabel("Probability", fontsize = 20)
    plt.savefig("./plot_data/" + monkey + "/7.2.pdf")
    plt.show()


def plotFig11AB():
    with open("./plot_data/{}_11AB.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    # plt.figure(dpi=300)
    # ax = sns.scatterplot(
    #     data=data,
    #     x="local_4dirs_diff",
    #     y="mean",
    #     #             size=result_df["count"],
    #     #                 ax=ax,
    #     hue="category",
    #     hue_order=["straight", "L-shape", "fork", "cross"],
    #     sizes=(20, 200),
    # )
    for index, c in enumerate(["straight", "L-shape", "fork", "cross"]):
        gpd = data[index]
        plt.errorbar(
            gpd["local_4dirs_diff"],
            gpd["mean"],
            yerr=gpd["std"] / np.sqrt(gpd["count"]),
            marker=None,
            capsize=3,
        )
    plt.xticks([0, 1, 2, 3, 4], [0, 1, 2, 3, ">=4"])
    plt.xlabel("local reward max - 2nd max")
    plt.ylabel("% of toward the most valuable direction")
    plt.title("errorbar = traditional std")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()


def plotFig111C1():
    with open("./plot_data/{}_111C.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    ylim = 10
    ax = sns.barplot(data=data, x="index", y=data[0] / data[0].sum(), palette="Blues_r", )
    ax.set_xticks(range(ylim + 1))
    ax.set_xticklabels(range(ylim + 1))
    ax.set_xlabel("actual trajectory length-shortest trajectory length")
    ax.set_ylabel("probability")
    plt.tight_layout()
    plt.savefig("./plot_data/" + monkey + "/11.1C1.pdf")
    plt.show()


def plotFig111C2():
    with open("./plot_data/{}_111C2.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    ax = sns.barplot(data=data, x="index", y=data[0] / data[0].sum(), palette="Blues_r")
    ax.set_xticks(range(6))
    ax.set_xticklabels(list(range(5)) + [">=5"])
    ax.set_xlabel("actual trajectory turns-fewest trajectory turns")
    ax.set_ylabel("normalized trial count")
    plt.savefig("./plot_data/" + monkey + "/11.1C2.pdf")
    plt.show()


def plotFig112B():
    with open("./plot_data/{}_112B.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    data = {"1":data["1"], "2":data["2"]}
    ghost_name = ["Blinky", "Clyde"]
    for each in data:
        df_plot = data[each]
        plt.figure()
        ax = sns.heatmap(df_plot, square=True, cmap="RdBu_r", vmin=0.5, vmax=1)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.invert_yaxis()
        plt.xlabel("EG distance")
        plt.ylabel("PG distance")
        plt.title(each)
        ax.set_xticklabels(
            [i.get_text().split(".")[0] for i in ax.get_xticklabels()], rotation=0
        )
        ax.set_yticklabels([i.get_text().split(".")[0] for i in ax.get_yticklabels()])
        plt.savefig("./plot_data/" + monkey + "/11.2B_" + ghost_name[int(each)-1] + "_relevent.pdf")
        plt.show()


def plotFig112Hist():
    colors = RdBu_7.mpl_colors
    with open("./plot_data/{}_112C.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    data = {"1":data["1"], "2":data["2"]}
    ghost_name = ["Blinky", "Clyde"]
    all_data = data["1"]
    all_data["distance2"] = data["2"].distance2
    all_data["EG2_dis"] = data["2"].EG2_dis
    planned_indices = np.where(all_data.cate == "Planned Hunting")[0]
    accidental_indices = np.where(all_data.cate == "Accidentally Hunting")[0]

    # 11.2C - Blinky
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    planned_dis = all_data.iloc[planned_indices].EG1_dis + all_data.iloc[planned_indices].distance1
    accidentald_dis = all_data.iloc[accidental_indices].EG1_dis + all_data.iloc[accidental_indices].distance1
    plt.figure(figsize=(10,7))

    sns.histplot(planned_dis, kde=False, bins=bins, label="Planned Hunting", color=colors[-1], stat="probability")
    sns.histplot(accidentald_dis, kde=False, bins=bins, label="Accidentally Hunting", color=colors[0], stat="probability")
    # sns.histplot(planned_dis, stat="probability")
    # sns.histplot(accidentald_dis, stat="probability")

    plt.xlabel("(Energizer-Blinky Distance) + (PacMan-Blinky Distance)", fontsize = 20)
    plt.xticks(fontsize = 20)
    plt.xlim(0, 90)
    plt.ylabel("Probability", fontsize = 20)
    plt.yticks(fontsize=20)
    plt.legend(frameon = False, fontsize = 20)
    plt.savefig("./plot_data/" + monkey + "/11.2C.pdf")
    plt.show()

    # 11.2D - Clyde
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    planned_dis = all_data.iloc[planned_indices].EG2_dis + all_data.iloc[planned_indices].distance2
    accidentald_dis = all_data.iloc[accidental_indices].EG2_dis + all_data.iloc[accidental_indices].distance2
    plt.figure(figsize=(10,7))
    # sns.distplot(planned_dis, kde=False, bins=bins, label="Planned Hunting", color=colors[-1],
    #              hist_kws={"edgecolor": colors[-1]}, norm_hist=True)
    # sns.distplot(accidentald_dis, kde=False, bins=bins, label="Accidentally Hunting", color=colors[0],
    #              hist_kws={"edgecolor": colors[0]})
    sns.histplot(planned_dis, kde=False, bins=bins, label="Planned Hunting", color=colors[-1], stat="probability")
    sns.histplot(accidentald_dis, kde=False, bins=bins, label="Accidentally Hunting", color=colors[0],
                 stat="probability")
    plt.xlabel("(Energizer-Clyde Distance) + (PacMan-Clyde Distance)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlim(0, 90)
    plt.ylabel("Probability", fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.savefig("./plot_data/" + monkey + "/11.2D.pdf")
    plt.show()

    # 11.2E - Ghost close to energizer
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    planned_dis = []
    for each in planned_indices:
        temp_data = all_data.iloc[each]
        if temp_data.EG1_dis < temp_data.EG2_dis:
            temp_dis = temp_data.EG1_dis + temp_data.distance1
        else:
            temp_dis = temp_data.EG2_dis + temp_data.distance2
        planned_dis.append(temp_dis)
    accidentald_dis = []
    for each in accidental_indices:
        temp_data = all_data.iloc[each]
        if temp_data.EG1_dis < temp_data.EG2_dis:
            temp_dis = temp_data.EG1_dis + temp_data.distance1
        else:
            temp_dis = temp_data.EG2_dis + temp_data.distance2
        accidentald_dis.append(temp_dis)
    plt.figure(figsize=(10, 7))
    # sns.distplot(planned_dis, kde=False, bins=bins, label="Planned Hunting", color=colors[-1],
    #              hist_kws={"edgecolor": colors[-1]}, norm_hist=True)
    # sns.distplot(accidentald_dis, kde=False, bins=bins, label="Accidentally Hunting", color=colors[0],
    #              hist_kws={"edgecolor": colors[0]}, norm_hist=True)
    sns.histplot(planned_dis, kde=False, bins=bins, label="Planned Hunting", color=colors[-1], stat="probability")
    sns.histplot(accidentald_dis, kde=False, bins=bins, label="Accidentally Hunting", color=colors[0],
                 stat="probability")
    plt.xlabel("(Energizer-ClosestGhost Distance) + (PacMan-ClosestGhost Distance)", fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlim(0, 90)
    plt.ylabel("Probability", fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.savefig("./plot_data/" + monkey + "/11.2E.pdf")
    plt.show()


def plotFig1131():
    with open("./plot_data/{}_1131.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)

    sns.set_palette([status_color_mapping["approach"], status_color_mapping["evade"]])
    ax = sns.barplot(
        data=data["data"],
        x="index",
        y="distance",
        hue="category",
    )
    ax.vlines(
        x=data["temp"],
        ymin=0,
        ymax=0.15,
        linestyle="--",
        color=status_color_mapping["evade"],
    )
    ax.vlines(
        x=data["df_com"],
        ymin=0,
        ymax=0.15,
        linestyle="--",
        color=status_color_mapping["approach"],
    )
    plt.legend(title=None)
    plt.ylim(0, 0.2)
    tks, labels = ax.get_xticks(), [i.get_text() for i in ax.get_xticklabels()]
    ax.set_xticks(tks[::6])
    ax.set_xticklabels(labels[::6])
    ax.set_xlabel("(Pacman-Pellet Distance) - (Reset-Pellet Distance)")
    ax.set_ylabel("Probability")
    plt.savefig("./plot_data/" + monkey + "/11.3.1.pdf")
    plt.show()


def plotFig1132():
    with open("./plot_data/{}_1132.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    sns.set_palette([status_color_mapping["evade"], status_color_mapping["approach"]])
    df_plot_all = data[data["index"] < 25]
    ax = sns.barplot(
        data=df_plot_all.sort_values(by="index"), x="index", y="value", hue="category",
    )
    tks, labels = ax.get_xticks(), [i.get_text() for i in ax.get_xticklabels()]
    ax.set_xticks(tks[::6])
    ax.set_xticklabels(labels[::6])
    plt.legend(title=None)
    plt.xlabel("PacMan-Ghost Distance")
    plt.ylabel("Probability")
    plt.savefig("./plot_data/" + monkey + "/11.3.2.pdf")
    plt.show()


def plotFig1141():
    with open("./plot_data/{}_1141.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)

    sns.set_palette([status_color_mapping["evade"], status_color_mapping["approach"]])
    ax = sns.barplot(
        data=data["data"],
        x="index",
        y="distance",
        hue="category",

    )
    # handles, labels = ax.get_legend_handles_labels()
    # labels = [
    #     "{} > 0 ratio: {}".format(labels[0].split(" ")[0], labels[0][-4:]),
    #     "{} > 0 ratio: {}".format(labels[1].split(" ")[0], labels[1][-4:])
    # ]
    # ax.legend(handles, labels)
    ax.vlines(
        x=data["temp"],
        ymin=0,
        ymax=0.15,
        linestyle="--",
        color=status_color_mapping["evade"],
    )
    ax.vlines(
        x=data["df_com"],
        ymin=0,
        ymax=0.15,
        linestyle="--",
        color=status_color_mapping["approach"],
    )
    plt.legend()
    plt.ylim(0, 0.2)
    tks, labels = ax.get_xticks(), [i.get_text() for i in ax.get_xticklabels()]
    ax.set_xticks(tks[::6])
    ax.set_xticklabels(labels[::6])
    ax.set_xlabel("(Pacman-Pellet Distance) - (Reset-Pellet Distance)")
    ax.set_ylabel("Probability")
    plt.savefig("./plot_data/" + monkey + "/11.4.1.pdf")
    plt.show()


def plotFig1142():
    with open("./plot_data/{}_1142.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    sns.set_palette([status_color_mapping["evade"], status_color_mapping["approach"]])
    df_plot_all = data[data["index"] < 25]
    ax = sns.barplot(
        data=df_plot_all.sort_values(by="index"), x="index", y="value", hue="category",
    )
    tks, labels = ax.get_xticks(), [i.get_text() for i in ax.get_xticklabels()]
    ax.set_xticks(tks[::6])
    ax.set_xticklabels(labels[::6])
    plt.legend(title=None)
    plt.xlabel("PacMan-Ghost Distance")
    plt.ylabel("Probability")
    plt.savefig("./plot_data/" + monkey + "/11.4.2.pdf")
    plt.show()


def plotFig115():
    plt.figure(figsize=(15, 5))

    plt.subplot(1,2,1)
    plt.title("Suicide", fontsize = 20)
    with open("./plot_data/{}_115_suicide_weight.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    mean_data = np.nanmean(data, axis = 0)
    std_data = np.nanstd(data, axis = 0)
    sem_data = scipy.stats.sem(data, axis=0, nan_policy="omit")
    tmp_agent_name = copy.deepcopy(agent_name)
    tmp_agent_name[2] = "evade(Blinky)"
    tmp_agent_name[3] = "evade(Clyde)"
    for i in range(6):
        plt.plot(mean_data[i,:7], "-", lw = 5, color = status_color_mapping[agent_name[i]], label = tmp_agent_name[i])
        plt.fill_between(
            np.arange(0, 7),
            mean_data[i,:7] - sem_data[i,:7],
            mean_data[i,:7] + sem_data[i,:7],
            color=status_color_mapping[agent_name[i]],
            alpha=0.3,
            linewidth=4
        )
    plt.legend(fontsize = 15, ncol = 6)
    plt.xticks([0, 1, 6], ["", -5, "Dead"], fontsize=20)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
    plt.ylim(0, 1.05)
    plt.xlim(0, 6)
    plt.ylabel("Strategy Weight", fontsize = 20)
    plt.legend(fontsize = 13, ncol = 3)

    plt.subplot(1, 2, 2)
    plt.title("Normal Death", fontsize = 20)
    with open("./plot_data/{}_115_normal_weight.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    mean_data = np.nanmean(data, axis=0)
    std_data = np.nanstd(data, axis=0)
    sem_data = scipy.stats.sem(data, axis=0, nan_policy="omit")
    tmp_agent_name = copy.deepcopy(agent_name)
    tmp_agent_name[2] = "evade(Blinky)"
    tmp_agent_name[3] = "evade(Clyde)"
    for i in range(6):
        plt.plot(mean_data[i, :7], "-", lw=5, color=status_color_mapping[agent_name[i]], label=tmp_agent_name[i])
        plt.fill_between(
            np.arange(0, 7),
            mean_data[i, :7] - sem_data[i, :7],
            mean_data[i, :7] + sem_data[i, :7],
            color=status_color_mapping[agent_name[i]],
            alpha=0.3,
            linewidth=4
        )
    plt.legend(fontsize=15, ncol=6)
    plt.xticks([0, 1, 6], ["", -5, "Dead"], fontsize=20)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.ylim(0, 1.05)
    plt.xlim(0, 6)
    plt.ylabel("Strategy Weight", fontsize=20)
    plt.legend(fontsize=13, ncol=3)

    plt.savefig("./plot_data/" + monkey + "/11.5.pdf")
    plt.show()


def plotFig12():
    with open("./plot_data/{}_12.pkl".format(monkey), "rb") as file:
        data = pickle.load(file)
    mean_data = np.nanmean(data, axis = 0)
    std_data = np.nanstd(data, axis = 0)
    sem_data = scipy.stats.sem(data, axis=0, nan_policy="omit")
    tmp_agent_name = copy.deepcopy(agent_name)
    tmp_agent_name[2] = "evade(Blinky)"
    tmp_agent_name[3] = "evade(Clyde)"

    plt.figure(figsize=(15, 5))
    for i in range(6):
        plt.plot(mean_data[i,:], "-", lw = 5, color = status_color_mapping[agent_name[i]], label = tmp_agent_name[i])
        plt.fill_between(
            np.arange(0, mean_data.shape[1]),
            mean_data[i,:] - sem_data[i,:],
            mean_data[i,:] + sem_data[i,:],
            color=status_color_mapping[agent_name[i]],
            alpha=0.3,
            linewidth=4
        )
    plt.legend(fontsize = 15, ncol = 6)
    plt.xticks([0, 6, 13], ["Move In", "", "Move Out"], fontsize = 20)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
    plt.ylim(0, 1.1)
    plt.xlim(0, 13)
    plt.ylabel("Strategy Weight", fontsize = 20)
    plt.savefig("./plot_data/" + monkey + "/12.pdf")
    plt.show()



if __name__ == '__main__':
    # ==========================
    # Plot saccade by weight

    # plotSaccByWeight()
    # plotAvgSaccByWeight() #TODO: bugs for Patamon

    # ==========================
    # Plot Fig. 7

    # plotFig71()
    # plotFig72()
    # plotFig73()
    # plotFig74()
    # plotFig75()

    # ==========================
    # Plot Fig. 8

    # plotFig8Freq()
    # plotFig8AvgStd() # Fig. 8

    # ==========================
    # Plot Fig. 9

    # plotFig9()

    # ==========================
    # Plot Fig. 10

    # plotFig10()

    # ==========================
    # Plot Fig. 11

    # plotFig11AB()
    # plotFig111C1()
    # plotFig111C2()
    # plotFig112B()
    # plotFig112Hist()

    # plotFig1131()
    # plotFig1132()
    # plotFig1141()
    # plotFig1142()
    #
    # plotFig115()

    # ==========================
    # Plot Fig. 12

    # plotFig12()

    pass