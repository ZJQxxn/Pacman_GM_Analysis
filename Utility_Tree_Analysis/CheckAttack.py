import numpy as np
import pickle
import pandas as pd
import copy
import sys
import matplotlib.pyplot as plt
params = {
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
}
plt.rcParams.update(params)

sys.path.append("./")
from LabelAnalysis import readTrialData, _handcraftLabeling
sys.path.append("../")
from Plotting import  _estimationVagueLabeling, _estimationMultipleLabeling

all_agent_list = ["global", "local", "pessimistic", "suicide", "planned_hunting"]
window = 3

def _readData():
    contribution_filename = "../common_data/global-local-pessimistic-suicide-planned_hunting/1000_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy"
    # contribution_filename = "../common_data/global_local_pessimistic_suicide_planned_hunting/100_trial_data_Omega-with_Q-window3-w_intercept-contribution.npy"
    trial_filename = "../common_data/trial/1000_trial_data_Omega-with_Q.pkl"
    trial_contribution  =np.load(contribution_filename, allow_pickle=True)
    trial_data = readTrialData(trial_filename)
    return [(trial_data[index][0], trial_data[index][1], trial_contribution[index]) for index in range(len(trial_contribution))]


def _eatEnergizerIndex(energizers):
    energizer_num = energizers.apply(lambda x : len(x) if not isinstance(x, float) else 0)
    energizer_diff = np.diff(energizer_num)
    eat_index = np.where(energizer_diff == -1)[0]-1
    return eat_index


def _eatGhostIndex(ghost_status):
    eat_index = []
    for index in range(1, ghost_status.shape[0]):
        if (ghost_status.ifscared1.values[index] == 3 and ghost_status.ifscared1.values[index-1] != 3) \
            or (ghost_status.ifscared2.values[index] == 3 and ghost_status.ifscared2.values[index-1] != 3):
            eat_index.append(index)
    return eat_index

def _preprocessData():
    data = _readData()
    processed_data = []
    for trial in data:
        trial_data = trial[1]
        trial_contribution = trial[2]
        length = len(trial_contribution)
        fitted_label = [np.nan for _ in range(window)]
        # fitted_label.extend([_estimationVagueLabeling(trial_contribution[index], all_agent_list) for index in range(length)])
        fitted_label.extend([_estimationMultipleLabeling(trial_contribution[index], all_agent_list) for index in range(length)])
        fitted_label.extend([np.nan for _ in range(window)])
        is_normal = trial_data[["ifscared1", "ifscared2"]].apply(lambda x: x.ifscared1 < 3 and x.ifscared2 < 3, axis = 1)
        temp_eat_index = np.zeros((length+2*window,), dtype=np.int)
        eat_index = _eatEnergizerIndex(trial_data.energizers)
        temp_eat_index[eat_index] = 1
        eat_index = temp_eat_index
        temp_ghost_index = np.zeros((length + 2 * window,), dtype=np.int)
        eat_ghost_index = _eatGhostIndex(trial_data[["ifscared1", "ifscared2"]])
        temp_ghost_index[eat_ghost_index] = 1
        eat_ghost_index = temp_ghost_index
        trial_data["fitted_label"] = fitted_label
        trial_data["is_normal"] = is_normal
        trial_data["eat_index"] = eat_index
        trial_data["eat_ghost_index"] = eat_ghost_index
        temp_contribution = [np.nan for _ in range(window)]
        temp_contribution.extend(trial_contribution)
        temp_contribution.extend([np.nan for _ in range(window)])
        trial_data["contribution"] = temp_contribution
        processed_data.append(copy.deepcopy(trial_data))
    return processed_data


def computing():
    data= _preprocessData()
    print("Finished reading and pre-processing.")
    all_matching_rate = []


    handcrafted_planned_cnt = 0
    handcrafted_accidental_cnt = 0
    hunting_cnt = 0
    fitted_attack_cnt = 0
    fitted_accidental_cnt = 0

    for index, trial in enumerate(data):
        print("|{}| Trial Name : {}".format(index + 1, trial.file[0]))
        length = trial.shape[0]

        is_planned_accidental = trial[["label_true_planned_hunting", "label_true_accidental_hunting"]].apply(
            lambda x : x.label_true_planned_hunting == 1 or x.label_true_accidental_hunting == 1,
            axis = 1
        )
        if np.all(is_planned_accidental == 0):
            print("No hunting data!")
            continue

        eat_index = np.where(trial.eat_index == 1)[0]
        if len(eat_index) == 0:
            print("No energizer is eaten!")
            continue
        else:
            for index in eat_index:
                # if is_planned_accidental.iloc[index] == 1:
                if trial.label_true_planned_hunting.iloc[index] == 1:
                    hunting_cnt += 1
                    for i in range(index,index+11):
                        if i >= len(trial.fitted_label):
                            break
                        if not isinstance(trial.fitted_label.iloc[i], float) and "planned_hunting" in trial.fitted_label.iloc[i]:
                            fitted_attack_cnt += 1
                            break
        #         # Five steps bef and after eating an energizer
        #         fitted_attack_sub = fitted_attack[max(0, index-5) : min(length, index + 6)]
        #         planned_hunting_sub = trial.label_true_planned_hunting[max(0, index-5) : min(length, index + 6)]
        #         matching = np.logical_and(planned_hunting_sub, fitted_attack_sub)
        #         matching_rate = np.sum(matching) / np.sum(planned_hunting_sub)
        #
        #         if np.any(planned_hunting_sub == 1):
        #             planned_cnt += 1
        #             if np.any(fitted_attack_sub == 1):
        #                 fitted_attack_cnt +=1
        #
        #
        #         temp["planned"].append(matching_rate)
        #
        #
        #         # Five steps after eating an energizer
        #         fitted_not_attack_sub = trial[["fitted_label", "is_normal"]].apply(
        #                 lambda x : not isinstance(x.fitted_label, float) and "planned_hunting" not in x.fitted_label and x.is_normal == 1,
        #                 axis = 1
        #         )[index: min(length, index + 6)]
        #         accidental_hunting_sub = trial.label_true_accidental_hunting[max(0, index - 5): min(length, index + 6)]
        #         matching = np.logical_and(accidental_hunting_sub, fitted_not_attack_sub)
        #         matching_rate = np.sum(matching) / np.sum(accidental_hunting_sub)
        #
        #         if np.any(accidental_hunting_sub == 1):
        #             accidental_cnt += 1
        #             if np.any(fitted_not_attack_sub == 1):
        #                 fitted_accidental_cnt +=1
        #
        #         temp["accidental"].append(matching_rate)
        # all_matching_rate.append(temp)
    print(hunting_cnt)
    print("Hunting Recovery Rate : ", fitted_attack_cnt / hunting_cnt)
    return all_matching_rate


def pltEnergizerWeight():
    data = _preprocessData()
    print("Finished reading and pre-processing.")
    all_planned_weight = []
    all_accidental_weight = []

    for index, trial in enumerate(data):
        print("|{}| Trial Name : {}".format(index + 1, trial.file[0]))
        length = trial.shape[0]
        planned_weight = np.zeros((5, 21))  # (local + attack agent, 20 step from energizer is eaten)
        accidental_weight = np.zeros((5, 21))
        planned_weight[planned_weight == 0] = np.nan
        accidental_weight[accidental_weight == 0] = np.nan
        eat_index = np.where(trial.eat_index == 1)[0]
        if len(eat_index) == 0:
            print("No energizer is eaten!")
            continue
        else:
            for index in eat_index:
                sub_data = trial.iloc[index:min(index+21, length)]
                sub_length = sub_data.shape[0]
                if sub_data.label_true_planned_hunting.values[0] == 1:
                    for i in range(sub_length):
                        if not isinstance(sub_data.contribution.values[i], float):
                            planned_weight[:, i] = sub_data.contribution.values[i] / np.linalg.norm(sub_data.contribution.values[i])
                if sub_data.label_true_accidental_hunting.values[0] == 1:
                    for i in range(sub_length):
                        if not isinstance(sub_data.contribution.values[i], float):
                            accidental_weight[:, i] = sub_data.contribution.values[i] / np.linalg.norm(sub_data.contribution.values[i])
            all_planned_weight.append(planned_weight)
            all_accidental_weight.append(accidental_weight)
    print("Finished getting weight")
    # save weight dynamics
    np.save("energizer_weight_dynamic.npy", {"planned":all_planned_weight, "accidental":all_accidental_weight})
    print("Finished saving data")
    # # plot agent weight
    # all_accidental_weight = np.array(all_accidental_weight)
    # all_planned_weight = np.array(all_planned_weight)
    # plt.subplot(2, 1, 1)
    # plt.title("Planned Hunting", fontsize = 20)
    # plt.plot(np.nanmean(all_planned_weight[:, 1, :], axis = 0), label = "local")
    # plt.plot(np.nanmean(all_planned_weight[:, 4, :], axis = 0), label = "attack")
    # plt.ylim(0.0, 1.0)
    # plt.legend(frameon = False, fontsize = 20)
    # plt.subplot(2, 1, 2)
    # plt.title("Accidental Hunting", fontsize=20)
    # plt.plot(np.nanmean(all_accidental_weight[:, 1, :], axis=0), label="local")
    # plt.plot(np.nanmean(all_accidental_weight[:, 4, :], axis=0), label="attack")
    # plt.legend(frameon=False, fontsize=20)
    # plt.ylim(0.0, 1.0)
    # plt.show()


def pltGhostWeight():
    data = _preprocessData()
    print("Finished reading and pre-processing.")
    all_planned_weight = []
    all_accidental_weight = []

    for index, trial in enumerate(data):
        print("|{}| Trial Name : {}".format(index + 1, trial.file[0]))
        length = trial.shape[0]
        planned_weight = np.zeros((5, 21))  # (local + attack agent, 20 step from energizer is eaten)
        accidental_weight = np.zeros((5, 21))
        planned_weight[planned_weight == 0] = np.nan
        accidental_weight[accidental_weight == 0] = np.nan
        eat_ghost_index = np.where(trial.eat_ghost_index == 1)[0]
        if len(eat_ghost_index) == 0:
            print("No ghost is eaten!")
            continue
        else:
            for index in eat_ghost_index:
                sub_data = trial.iloc[max(0, index-20):index+1]
                sub_length = sub_data.shape[0]
                if sub_data.label_true_planned_hunting.values[0] == 1:
                    for i in range(0, sub_length):
                        if not isinstance(sub_data.contribution.values[sub_length-1-i], float):
                            planned_weight[:, 20-i] = sub_data.contribution.values[sub_length-1-i] / np.linalg.norm(sub_data.contribution.values[sub_length-1-i])
                if sub_data.label_true_accidental_hunting.values[0] == 1:
                    for i in range(0, sub_length):
                        if not isinstance(sub_data.contribution.values[sub_length - 1 - i], float):
                            accidental_weight[:, 20 - i] = sub_data.contribution.values[sub_length - 1 - i] / np.linalg.norm(sub_data.contribution.values[sub_length - 1 - i])
            all_planned_weight.append(planned_weight)
            all_accidental_weight.append(accidental_weight)
    print("Finished getting weight")
    # save weight dynamics
    np.save("ghost_weight_dynamic.npy", {"planned":all_planned_weight, "accidental":all_accidental_weight})
    print("Finished saving data")
    # # plot agent weight
    # all_accidental_weight = np.array(all_accidental_weight)
    # all_planned_weight = np.array(all_planned_weight)
    # plt.subplot(2, 1, 1)
    # plt.title("Planned Hunting", fontsize = 20)
    # plt.plot(np.nanmean(all_planned_weight[:, 1, :], axis = 0), label = "local")
    # plt.plot(np.nanmean(all_planned_weight[:, 4, :], axis = 0), label = "attack")
    # plt.ylim(0.0, 1.0)
    # plt.legend(frameon = False, fontsize = 20)
    # plt.subplot(2, 1, 2)
    # plt.title("Accidental Hunting", fontsize=20)
    # plt.plot(np.nanmean(all_accidental_weight[:, 1, :], axis=0), label="local")
    # plt.plot(np.nanmean(all_accidental_weight[:, 4, :], axis=0), label="attack")
    # plt.legend(frameon=False, fontsize=20)
    # plt.ylim(0.0, 1.0)
    # plt.show()


def readAndPlot():
    from palettable.cmocean.diverging import Balance_6
    from palettable.colorbrewer.diverging import RdYlBu_5
    colors = RdYlBu_5.mpl_colors
    agent_color = {
        "local": "#D7181C",
        "pessimistic": "#FDAE61",
        "global": "#44B53C",
        "suicide": "#836BB7",
        "planned_hunting": "#81B3FF",
        "vague": "black"
    }

    data = np.load("energizer_weight_dynamic.npy", allow_pickle=True).item()
    # plot agent weight
    all_accidental_weight = np.array(data["accidental"])
    all_planned_weight = np.array(data["planned"])
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.title("Planned Hunting", fontsize = 20)
    plt.plot(np.nanmean(all_planned_weight[:, 1, :], axis = 0), label = "local", color=agent_color["local"], ms = 3, lw = 5)
    plt.plot(np.nanmean(all_planned_weight[:, 4, :], axis = 0), label = "attack", color=agent_color["planned_hunting"], ms = 3, lw = 5)
    # ses_weight = np.nanstd(all_planned_weight, axis  = 0)
    # plt.fill_between(
    #     np.arange(21),
    #     np.nanmean(all_planned_weight[:, 1, :], axis = 0) - np.nanstd(all_planned_weight[:, 1, :], axis = 0),
    #     np.nanmean(all_planned_weight[:, 1, :], axis = 0) + np.nanstd(all_planned_weight[:, 1, :], axis = 0),
    #     color=agent_color["local"],
    #     alpha=0.3,
    #     linewidth=4
    # )
    # plt.fill_between(
    #     np.arange(21),
    #     np.nanmean(all_planned_weight[:, 4, :], axis=0) - np.nanstd(all_planned_weight[:, 4, :], axis=0),
    #     np.nanmean(all_planned_weight[:, 4, :], axis=0) + np.nanstd(all_planned_weight[:, 4, :], axis=0),
    #     color=agent_color["planned_hunting"],
    #     alpha=0.3,
    #     linewidth=4
    # )
    plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.ylabel("Normalized Strategy Weight", fontsize = 20)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon = False, fontsize = 20)
    plt.subplot(1, 2, 2)
    plt.title("Accidental Hunting", fontsize=20)
    plt.plot(np.nanmean(all_accidental_weight[:, 1, :], axis=0), label="local", color=agent_color["local"], ms = 3, lw = 5)
    plt.plot(np.nanmean(all_accidental_weight[:, 4, :], axis=0), label="attack", color=agent_color["planned_hunting"], ms = 3, lw = 5)
    # plt.fill_between(
    #     np.arange(21),
    #     np.nanmean(all_accidental_weight[:, 1, :], axis=0) - np.nanstd(all_accidental_weight[:, 1, :], axis=0),
    #     np.nanmean(all_accidental_weight[:, 1, :], axis=0) + np.nanstd(all_accidental_weight[:, 1, :], axis=0),
    #     color=agent_color["local"],
    #     alpha=0.3,
    #     linewidth=4
    # )
    # plt.fill_between(
    #     np.arange(21),
    #     np.nanmean(all_accidental_weight[:, 4, :], axis=0) - np.nanstd(all_accidental_weight[:, 4, :], axis=0),
    #     np.nanmean(all_accidental_weight[:, 4, :], axis=0) + np.nanstd(all_accidental_weight[:, 4, :], axis=0),
    #     color=agent_color["planned_hunting"],
    #     alpha=0.3,
    #     linewidth=4
    # )
    plt.xticks([0, 5, 10, 15, 20], [0, 5, 10, 15, 20], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.show()

    data = np.load("ghost_weight_dynamic.npy", allow_pickle=True).item()
    # plot agent weight
    all_accidental_weight = np.array(data["accidental"])
    all_planned_weight = np.array(data["planned"])
    plt.figure(figsize=(10,10))
    plt.subplot(1, 2, 1)
    plt.title("Planned Hunting", fontsize=20)
    plt.plot(np.nanmean(all_planned_weight[:, 1, :], axis=0), label="local", color=agent_color["local"], ms=3, lw=5)
    plt.plot(np.nanmean(all_planned_weight[:, 4, :], axis=0), label="attack", color=agent_color["planned_hunting"],
             ms=3, lw=5)
    plt.xticks([0, 5, 10, 15, 20], [-20, -15, -10, -5, 0], fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel("Normalized Strategy Weight", fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.legend(frameon=False, fontsize=20)
    plt.subplot(1, 2, 2)
    plt.title("Accidental Hunting", fontsize=20)
    plt.plot(np.nanmean(all_accidental_weight[:, 1, :], axis=0), label="local", color=agent_color["local"], ms=3, lw=5)
    plt.plot(np.nanmean(all_accidental_weight[:, 4, :], axis=0), label="attack", color=agent_color["planned_hunting"],
             ms=3, lw=5)
    plt.xticks([0, 5, 10, 15, 20], [-20, -15, -10, -5, 0], fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, fontsize=20)
    plt.ylim(0.0, 1.0)
    plt.show()






if __name__ == '__main__':
    import pprint
    # matching_rate = computing()
    # np.save("hunting_matching_rate.npy", matching_rate)
    # pprint.pprint(matching_rate)

    # pltEnergizerWeight()
    # pltGhostWeight()
    readAndPlot()


    # import matplotlib.pyplot as plt
    # plt.title("p = 1 million, 32 CPU Threads", fontsize = 20)
    # plt.bar(x = [0], height = [11.2], color = "red")
    # plt.bar(x=[1], height=[24], color="blue")
    # plt.bar(x=[2], height=[114], color="purple")
    # plt.xticks([0, 1, 2], ["FST", "EE", "BIGQUIC"], fontsize = 20)
    # plt.yticks(fontsize = 20)
    # plt.ylabel("Time Cost (hours)", fontsize = 20)
    # plt.show()

    # data =np.load("energizer_weight_dynamic.npy", allow_pickle=True)
    # print()