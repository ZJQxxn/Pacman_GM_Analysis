'''
Description:
    Plotting all the figures.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    10 Nov. 2020
'''

import pickle
import pandas as pd
import numpy as np
import scipy.optimize
import scipy.stats
import scipy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import pprint

# plt.rc('font', family='CMU Serif', weight="roman")
# plt.rc('font', family='Myriad Pro')
#
# from matplotlib import rcParams
# rcParams['mathtext.default'] = 'regular'

params = {
    "pdf.fonttype": 42,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
}
plt.rcParams.update(params)


import copy
import seaborn
import os
import sys

from palettable.cmocean.diverging import Balance_6
from palettable.colorbrewer.diverging import RdBu_8, RdYlBu_5
from palettable.scientific.sequential import Davos_5
from palettable.scientific.diverging import Roma_5, Vik_5, Roma_3
from palettable.tableau import BlueRed_6
from palettable.cartocolors.qualitative import Vivid_5
from palettable.lightbartlein.diverging import BlueDarkRed18_18, BlueOrange12_5, BlueDarkRed18_4


sys.path.append("./Utility_Tree_Analysis")
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath, scaleOfNumber, gini
from LabelAnalysis import _makeChoice, _label2Index, negativeLikelihood
# from LabelAnalysis import _PG, _PE, _ghostStatus, _energizerNum, _PR, _RR, _PGWODead

from PathAnalysis import readTrialData as pathRead

# colors = RdYlBu_5.mpl_colors
# agent_color = {
#         "local" : colors[0],
#         "pessimistic" : colors[1],
#         "global" : colors[-1],
#         "suicide" : Balance_6.mpl_colors[2],
#         "planned_hunting" : colors[3]
#     }

agent_color = {
        "local": "#D7181C",
        "pessimistic": "#FDAE61",
        "pessimistic_blinky": "#FDAE61",
        "pessimistic_clyde": "#c78444",
        "global": "#44B53C",
        "suicide": "#836BB7",
        "planned_hunting": "#81B3FF",
        "vague": "black"
    }
    
label_name = {
        "local": "local",
        "pessimistic": "evade",
        "pessimistic_blinky": "evade(Blinky)",
        "pessimistic_clyde": "evade(Clyde)",
        "global": "global",
        "suicide": "suicide",
        "planned_hunting": "attack"
}


dir_list = ['left', 'right', 'up', 'down']


def oneHot(val):
    '''
    Convert the direction into a one-hot vector.
    :param val: The direction. should be the type ``str''.
    :return:
    '''
    # Type check
    if val not in dir_list:
        raise ValueError("Undefined direction {}!".format(val))
    if not isinstance(val, str):
        raise TypeError("Undefined direction type {}!".format(type(val)))
    # One-hot
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[dir_list.index(val)] = 1
    return onehot_vec


def _handcraftLabeling(labels):
    hand_crafted_label = []
    labels = labels.fillna(0)
    # local
    if labels.label_local_graze or labels.label_local_graze_noghost or labels.label_true_accidental_hunting or labels.label_global_ending:
        hand_crafted_label.append("local")
    # evade (pessmistic)
    if labels.label_evade:
        hand_crafted_label.append("pessimistic")
    # global
    if labels.label_global_optimal or labels.label_global_notoptimal or labels.label_global:
        if labels.label_global_ending:
            pass
        else:
            hand_crafted_label.append("global")
    # suicide
    if labels.label_suicide:
        hand_crafted_label.append("suicide")
    # planned hunting
    if labels.label_true_planned_hunting:
        hand_crafted_label.append("planned_hunting")
    if len(hand_crafted_label) == 0:
        hand_crafted_label = None
    elif len(hand_crafted_label) > 1:
        hand_crafted_label = ["vague"]
    else:
        pass
    return hand_crafted_label


# ===================================
#         VISUALIZATION
# ===================================
def _estimationThreeLabeling(contributions, all_agent_name):
    # global, local, pessimistic
    labels = []
    agent_name = ["global", "local"]
    if np.any(contributions[:2] > 0):
        labels.append(agent_name[np.argmax(contributions[:2])])
    # Threshold for different labels
    if "pessimistic" == all_agent_name[-1]:
        threshold = 0.0
    elif "planned_hunting" == all_agent_name[-1]:
        threshold = 0.0
    elif "suicide" == all_agent_name[-1]:
        threshold = 0.1
    else:
        raise NotImplementedError("Unknown agent {}!".format(all_agent_name[-1]))
    if contributions[-1] > threshold:
        labels.append(all_agent_name[-1])
    return labels


def _estimationMultipleLabeling(contributions, all_agent_name):
    # global, local, pessimistic
    labels = []
    agent_name = ["global", "local"]
    if np.any(contributions[:2] > 0):
        labels.append(agent_name[np.argmax(contributions[:2])])
    # Threshold for different labels
    pess_threshold = 0.1
    planned_threshold = 0.1
    suicide_threshold = 0.1
    if "pessimistic" in all_agent_name:
        if contributions[all_agent_name.index("pessimistic")] > pess_threshold:
            labels.append("pessimistic")
    if "suicide" in all_agent_name:
        if contributions[all_agent_name.index("suicide")] > suicide_threshold:
            labels.append("suicide")
    if "planned_hunting" in all_agent_name:
        if contributions[all_agent_name.index("planned_hunting")] > planned_threshold:
            labels.append("planned_hunting")
    return labels


def _estimationLocalEvadeSuicideLabeling(contributions):
    labels = []
    local_threshold = 0.0
    pess_threshold = 0.1
    suicide_threshold = 0.1
    if contributions[0] > local_threshold:
        labels.append("local")
    if contributions[1] > pess_threshold:
        labels.append("pessimistic")
    if contributions[2] > suicide_threshold:
        labels.append("suicide")
    # agent_name = ["pessimistic", "suicide"]
    # if np.any(contributions[1:] > 0):
    #     labels.append(agent_name[np.argmax(contributions[1:])])
    return labels


def _estimationVagueLabeling(contributions, all_agent_name):
    sorted_contributions = np.sort(contributions)[::-1]
    if sorted_contributions[0] - sorted_contributions[1] < 0.2 :
        return ["vague"]
    else:
        label = all_agent_name[np.argmax(contributions)]
        return [label]


def _estimationThresholdLabeling(contributions, all_agent_name):
    # global, local, pessimistic
    labels = []
    agent_name = ["global", "local"]
    if np.any(contributions[:2] > 0):
        labels.append(agent_name[np.argmax(contributions[:2])])
    # Threshold for different labels
    pess_threshold = 0.1
    planned_threshold = 0.1
    suicide_threshold = 0.1
    if "pessimistic" in all_agent_name:
        if contributions[all_agent_name.index("pessimistic")] > pess_threshold:
            labels.append("pessimistic")
    if "suicide" in all_agent_name:
        if contributions[all_agent_name.index("suicide")] > suicide_threshold:
            labels.append("suicide")
    if "planned_hunting" in all_agent_name:
        if contributions[all_agent_name.index("planned_hunting")] > planned_threshold:
            labels.append("planned_hunting")
    if len(labels) >= 2:
        return ["vague"]
    return labels

# ===================================================

def singleTrialMultiFitting(config):
    print("="*20, " Single Trial ", "="*20)
    # Read trial data
    agents_list = ["{}_Q".format(each) for each in ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]]
    window = config["single_trial_window"]
    trial_data = pathRead(config["single_trial_data_filename"])
    trial_num = len(trial_data)
    print("Num of trials : ", trial_num)

    trial_name_list = None
    all_trial_names = np.array([each[0] for each in trial_data])
    # trial_name_list = np.random.choice(all_trial_names, trial_num, replace = True)
    trial_name_list = all_trial_names
    record = []
    if trial_name_list is not None and len(trial_name_list) > 0:
        temp_trial_Data = []
        for each in trial_data:
            if each[0] in trial_name_list:
                temp_trial_Data.append(each)
        trial_data = temp_trial_Data
    print("Num of trials : ", len(trial_data))
    label_list = ["label_local_graze", "label_local_graze_noghost", "label_global_ending",
                  "label_global_optimal", "label_global_notoptimal", "label_global",
                  "label_evade",
                  "label_suicide",
                  "label_true_accidental_hunting",
                  "label_true_planned_hunting"]

    all_hand_crafted = []
    all_estimated = []
    all_weight_main = []
    all_weight_rest = []
    all_Q = []

    # agent_name = ["global", "local", "pessimistic"]
    agent_name = config["single_trial_agents"]
    agent_index = [["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"].index(i) for i in agent_name]
    # Construct optimizer
    for trial_index, each in enumerate(trial_data):
        # if trial_index > 50:
        #     break
        temp_record = []
        print("-"*15)
        trial_name = each[0]
        temp_record.append(trial_name)
        X = each[1]
        Y = each[2]
        trial_length = X.shape[0]
        print("Index ", trial_index, " Trial name : ", trial_name)
        # Hand-crafted label
        handcrafted_label = [_handcraftLabeling(X[label_list].iloc[index]) for index in range(X.shape[0])]
        handcrafted_label = handcrafted_label[window : -window]
        all_hand_crafted.append(handcrafted_label)
        # Estimating label through moving window analysis
        print("Trial length : ", trial_length)
        window_index = np.arange(window, trial_length - window)
        # (num of windows, num of agents)
        temp_weight = np.zeros((len(window_index), len(agent_name) if not config["need_intercept"] else len(agent_name)))
        # temp_weight_rest = np.zeros((len(window_index), 3 if not config["need_intercept"] else 4))
        # temp_Q = []
        temp_contribution = np.zeros((len(window_index), len(agent_name)))
        # temp_contribution_rest = np.zeros((len(window_index), 3))
        cr = np.zeros((len(window_index), ))
        # (num of windows, window size, num of agents, num pf directions)
        temp_trial_Q = np.zeros((len(window_index), window * 2 + 1, len(agent_name), 4))
        # For each trial, estimate agent weights through sliding windows
        trial_fitted_label = []
        trial_estimated_label = []
        for centering_index, centering_point in enumerate(window_index):
            print("Window at {}...".format(centering_point))
            cur_step = X.iloc[centering_point]
            sub_X = X[centering_point - window:centering_point + window+1]
            sub_Y = Y[centering_point - window:centering_point + window+1]
            Q_value = sub_X[agents_list].values
            for i in range(window * 2 + 1):  # num of samples in a window
                for j in range(len(agent_name)):  # number of agents
                    temp_trial_Q[centering_index, i, j, :] = Q_value[i][j]
            # estimation in the window
            window_estimated_label = []
            # Construct optimizer
            params = [0 for _ in range(len(agent_name))]
            bounds = [[0, 10] for _ in range(len(agent_name))]
            if config["need_intercept"]:
                params.append(1)
                bounds.append([-1000, 1000])
            cons = []  # construct the bounds in the form of constraints
            for par in range(len(bounds)):
                l = {'type': 'ineq', 'fun': lambda x: x[par] - bounds[par][0]}
                u = {'type': 'ineq', 'fun': lambda x: bounds[par][1] - x[par]}
                cons.append(l)
                cons.append(u)
            # estimation in the window
            func = lambda params: negativeLikelihood(
                params,
                sub_X,
                sub_Y,
                agent_name,
                return_trajectory=False,
                need_intercept=config["need_intercept"]
            )
            is_success = False
            retry_num = 0
            while not is_success and retry_num < config["maximum_try"]:
                res = scipy.optimize.minimize(
                    func,
                    x0=params,
                    method="SLSQP",
                    bounds=bounds,
                    tol=1e-5,
                    constraints=cons
                )
                is_success = res.success
                if not is_success:
                    print("Fail, retrying...")
                    retry_num += 1

            temp_weight[centering_index, :] = res.x[:-1]
            contribution = temp_weight[centering_index, :] * \
                           [scaleOfNumber(each) for each in
                            np.max(np.abs(temp_trial_Q[centering_index, :, agent_index, :]), axis=(1, 2))]
            temp_contribution[centering_index, :] = contribution

            # correct rate in the window
            _, estimated_prob = negativeLikelihood(
                res.x,
                sub_X,
                sub_Y,
                agent_name,
                return_trajectory=True,
                need_intercept=config["need_intercept"]
            )
            estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
            true_dir = sub_Y.apply(lambda x: np.argmax(x)).values
            correct_rate = np.sum(estimated_dir == true_dir) / len(true_dir)
            cr[centering_index] = correct_rate


        for index in range(temp_contribution.shape[0]):
            temp_contribution[index, :] = temp_contribution[index, :] / np.linalg.norm(temp_contribution[index, :])
        trial_gini = [gini(temp_contribution[i,:]) for i in range(temp_contribution.shape[0])]

        estimated_label = [
            _estimationVagueLabeling(temp_contribution[index, :], agent_name)
            for index in range(len(temp_contribution))
        ]

        # normalization
        label_name["suicide"] = "approach"
        label_name["planned_hunting"] = "energizer"

        fig = plt.figure(figsize = (18,12), constrained_layout = True)
        spec = fig.add_gridspec(5, 1)
        # plt.subplot(2, 1, 1)
        ax1 = fig.add_subplot(spec[:2,:])
        # plt.title(trial_name, fontsize = 10)
        # plt.title(trial_name, fontsize = 15)
        for index in range(len(agent_name)):
            plt.plot(temp_contribution[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                     label=label_name[agent_name[index]])
        # for pessimistic agent
        plt.ylabel("Normalized Strategy Weight", fontsize=20)
        plt.xlim(0, temp_contribution.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize = 20)
        # x_ticks_index = np.linspace(0, len(temp_contribution), 5)
        x_ticks_index = np.arange(0, len(temp_contribution), 10)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(-0.01, 1.02)
        plt.legend(loc="upper center", fontsize=20, ncol = len(agent_name), frameon = False, bbox_to_anchor = (0.5, 1.2))
        # plt.show()

        # plt.figure(figsize=(13,5))
        # plt.subplot(2, 1, 2)
        ax2 = fig.add_subplot(spec[2, :])
        for i in range(len(estimated_label)):
                # seq = np.linspace(-0.1, 0.0, len(handcrafted_label[i]) + 1)
                # for j, h in enumerate(handcrafted_label[i]):
                #     plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
            seq = np.linspace(-0.02, 0.0, len(estimated_label[i]) + 1)
            for j, h in enumerate(estimated_label[i]):
                plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
        plt.xlim(0, len(estimated_label))
        # x_ticks_index = np.linspace(0, len(handcrafted_label), 5)
        # x_ticks = [window + int(each) for each in x_ticks_index]
        # plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        # plt.yticks([-0.05, -0.15], ["Rule-Based Label", "Fitted Label"], fontsize=10)
        # plt.ylim(-0.05, 0.35)
        plt.axis('off')

        ax3 = fig.add_subplot(spec[3, :])
        plt.title("{a} (Cr : {b:.3f})".format(a=trial_name, b=np.nanmean(cr) if not np.all(np.isnan(cr)) else 0.0),
                  fontsize=10)
        plt.plot(np.arange(len(cr)), cr, "bo-", lw=4, ms=10)
        # for pessimistic agent
        plt.ylabel("Correct Rate", fontsize=20)
        plt.xlim(0, cr.shape[0] - 1)
        # plt.xlabel("Time Step", fontsize=20)
        # x_ticks_index = np.linspace(0, len(temp_contribution), 5)
        x_ticks_index = np.arange(0, len(cr), 10)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylim(0.5, 1.05)

        ax4 = fig.add_subplot(spec[4, :])
        plt.title("Gini Coefficient : ({b:.3f})".format(b=np.nanmean(trial_gini) if not np.all(np.isnan(trial_gini)) else np.nan),
                  fontsize=10)
        plt.plot(np.arange(len(trial_gini)), trial_gini, "bo-", lw=4, ms=10)
        # for pessimistic agent
        plt.ylabel("Gini", fontsize=20)
        plt.xlim(0, len(trial_gini) - 1)
        plt.xlabel("Time Step", fontsize=20)
        # x_ticks_index = np.linspace(0, len(temp_contribution), 5)
        x_ticks_index = np.arange(0, len(trial_gini), 10)
        x_ticks = [window + int(each) for each in x_ticks_index]
        plt.xticks(x_ticks_index, x_ticks, fontsize=20)
        plt.yticks([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], fontsize=20)
        plt.ylim(0.0, 1.0)


        plt.savefig("./common_data/single_trial/uniform_depth10/{}.pdf".format(trial_name))
        # plt.show()
        plt.close()
        plt.clf()


# ================================================

def plotIncremental(config):
    print("-"*15)
    # filename index
    trial_names = np.load("./common_data/trial/100_trial_name.npy", allow_pickle=True)
    trial_indices = []
    for index, each in enumerate(trial_names):
        if each.split(".")[-2][-1]=="1":
            trial_indices.append(index)
    # random correct rate
    random_cr = np.load("./common_data/incremental/100trial-window3-random_is_correct.npy", allow_pickle=True).item()
    # avg_random_cr = np.nanmean([np.nanmean(each) for each in random_cr])
    avg_random_cr = {each:np.nanmean(random_cr[each]) for each in random_cr}
    print(avg_random_cr)
    # trial name, pacman pos, beans, window cr for different agents
    bean_vs_cr = np.load(config["bean_vs_cr_filename"], allow_pickle = True)[trial_indices]
    bean_num = []
    agent_cr = []
    for i in bean_vs_cr:
        temp_cr = []
        temp_bean_num = []
        for j in i:
            temp_bean_num.append(len(j[2]))
            temp_cr.append(j[3])
        bean_num.append(copy.deepcopy(temp_bean_num))
        agent_cr.append(copy.deepcopy(temp_cr))
    # bean_num = [len(each[2]) if isinstance(each[2], list) else 0 for each in bean_vs_cr]
    # agent_cr  = [each[3] for each in bean_vs_cr]
    max_bean_num = max(max(bean_num))
    min_bean_num = min(min(bean_num))
    print("Max bean num : ", max_bean_num)
    print("Min bean num : ", min_bean_num)
    agent_index = [0, 1, 2, 4, 5, 6] # (local, + global, + pessimistic blinky, +pessimistic clyde, + planned hunting, +suicide)
    first_phase_agent_cr = [] # num of beans <= 10
    second_phase_agent_cr = [] # 10 < num of beans < 80
    third_phase_agent_cr = [] # num of beans > 80
    # every bin
    for i in range(len(bean_num)):
        trial_bean_num = bean_num[i]
        trial_cr = agent_cr[i]
        trial_early = []
        trial_middle = []
        trial_end = []
        for j in range(len(trial_bean_num)):
            if trial_bean_num[j] <= 10:
                trial_end.append(trial_cr[j])
            elif 10 < trial_bean_num[j] < 80:
                trial_middle.append(trial_cr[j])
            else:
                trial_early.append(trial_cr[j])
        if len(trial_early) > 0:
            third_phase_agent_cr.append(np.mean(trial_early, axis = 0)[agent_index])
        if len(trial_middle) > 0:
            second_phase_agent_cr.append(np.mean(trial_middle, axis = 0)[agent_index])
        if len(trial_end) > 0:
            first_phase_agent_cr.append(np.mean(trial_end, axis = 0)[agent_index])

    # plotting
    x_ticks = ["local", "+global", "+evade\n(Blinky)", "+evade\n(Clyde)", "+energizer", "+approach"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)
    # colors = RdYlBu_5.mpl_colors
    # colors[2] = Balance_6.mpl_colors[2]
    # colors = [colors[0], colors[-1], colors[1], colors[3], colors[2]]
    colors = [
        agent_color["local"],
        agent_color["global"],
        agent_color["pessimistic_blinky"],
        agent_color["pessimistic_clyde"],
        agent_color["planned_hunting"],
        agent_color["suicide"]
    ]

    plt.figure(figsize=(27, 5))

    plt.subplot(1, 3, 1)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Early Stage (Pellets >= 80)", fontsize=20)
    avg_cr = np.mean(third_phase_agent_cr, axis=0)
    # var_cr = scipy.stats.sem(third_phase_agent_cr, axis=0, nan_policy="omit")
    var_cr = np.nanstd(third_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    # plt.plot([-0.5, 3.5], [avg_random_cr["early"], avg_random_cr["early"]], "--", lw = 5, color = "grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    if "simple" in config["bean_vs_cr_filename"]:
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.3, 1.05)
    else:
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize = 15)
        plt.ylim(0.7, 1.05)
    plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)

    plt.subplot(1, 3, 2)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Middle Stage (10 < Pellets < 80)", fontsize=20)
    avg_cr = np.mean(second_phase_agent_cr, axis=0)
    var_cr = np.nanstd(second_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    # plt.plot([-0.5, 3.5], [avg_random_cr["middle"], avg_random_cr["middle"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    if "simple" in config["bean_vs_cr_filename"]:
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.3, 1.05)
    else:
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.7, 1.05)

    plt.subplot(1, 3, 3)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Ending Stage (Pellets <= 10)", fontsize=20)
    avg_cr = np.mean(first_phase_agent_cr, axis=0)
    var_cr = np.nanstd(first_phase_agent_cr, axis=0)
    # plt.errorbar(x_index, avg_cr, yerr = var_cr, fmt = "k", mfc = "r", marker = "o", linestyle = "", ms = 15, elinewidth = 5)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    # plt.plot([-0.5, 3.5], [avg_random_cr["end"], avg_random_cr["end"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    if "simple" in config["bean_vs_cr_filename"]:
        plt.yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.7, 1.05)
    else:
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.7, 1.05)

    plt.show()


def plotDecremental(config):
    # random correct rate
    random_cr = np.load("./common_data/incremental/100trial-window3-random_is_correct.npy", allow_pickle=True).item()
    # avg_random_cr = np.nanmean([np.nanmean(each) for each in random_cr])
    avg_random_cr = {each: np.nanmean(random_cr[each]) for each in random_cr}
    print("-"*15)
    # trial name, pacman pos, beans, window cr for different agents
    bean_vs_cr = np.load(config["decremental_filename"], allow_pickle = True)
    bean_num = []
    agent_cr = []
    bean_num = []
    agent_cr = []
    for i in bean_vs_cr:
        temp_cr = []
        temp_bean_num = []
        for j in i:
            temp_bean_num.append(len(j[2]))
            temp_cr.append(j[3])
        bean_num.append(copy.deepcopy(temp_bean_num))
        agent_cr.append(copy.deepcopy(temp_cr))
    # bean_num = [len(each[2]) if isinstance(each[2], list) else 0 for each in bean_vs_cr]
    # agent_cr  = [each[3] for each in bean_vs_cr]
    max_bean_num = max(max(bean_num))
    min_bean_num = min(min(bean_num))
    print("Max bean num : ", max_bean_num)
    print("Min bean num : ", min_bean_num)
    agent_index = [1, 0 ,2, 3, 4, 5]
    first_phase_agent_cr = []  # num of beans <= 10
    second_phase_agent_cr = []  # 10 < num of beans < 80
    third_phase_agent_cr = []  # num of beans > 80
    # every bin
    for i in range(len(bean_num)):
        trial_bean_num = bean_num[i]
        trial_cr = agent_cr[i]
        trial_early = []
        trial_middle = []
        trial_end = []
        for j in range(len(trial_bean_num)):
            if trial_bean_num[j] <= 10:
                trial_end.append(trial_cr[j])
            elif 10 < trial_bean_num[j] < 80:
                trial_middle.append(trial_cr[j])
            else:
                trial_early.append(trial_cr[j])
        if len(trial_early) > 0:
            third_phase_agent_cr.append(np.mean(trial_early, axis=0)[agent_index])
        if len(trial_middle) > 0:
            second_phase_agent_cr.append(np.mean(trial_middle, axis=0)[agent_index])
        if len(trial_end) > 0:
            first_phase_agent_cr.append(np.mean(trial_end, axis=0)[agent_index])

    # plotting
    x_ticks = ["-local", "-global", "-evade\n(Blinky)", "-evade\n(Clyde)", "-approach", "-energizer"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)
    # colors = RdYlBu_5.mpl_colors
    # colors[2] = Balance_6.mpl_colors[2]
    # colors = [colors[0], colors[-1], colors[1], colors[3], colors[2]]
    colors = [
        agent_color["local"],
        agent_color["global"],
        agent_color["pessimistic_blinky"],
        agent_color["pessimistic_clyde"],
        agent_color["suicide"],
        agent_color["planned_hunting"]
    ]

    plt.figure(figsize=(27, 5))

    plt.subplot(1, 3, 1)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Early Stage (Pellets >= 80)", fontsize=20)
    avg_cr = np.mean(third_phase_agent_cr, axis=0)
    var_cr = np.nanstd(third_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["early"], avg_random_cr["early"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    if "simple" in config["decremental_filename"]:
        plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.2, 1.05)
    else:
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.4, 1.05)
    plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)

    plt.subplot(1, 3, 2)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Middle Stage (10 < Pellets < 80)", fontsize=20)
    avg_cr = np.mean(second_phase_agent_cr, axis=0)
    var_cr = np.nanstd(second_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["middle"], avg_random_cr["middle"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    if "simple" in config["decremental_filename"]:
        plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   fontsize=15)
        plt.ylim(0.2, 1.05)
    else:
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.4, 1.05)

    plt.subplot(1, 3, 3)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Ending Stage (Pellets <= 10)", fontsize=20)
    avg_cr = np.mean(first_phase_agent_cr, axis=0)
    var_cr = np.nanstd(first_phase_agent_cr, axis=0)
    # plt.errorbar(x_index, avg_cr, yerr = var_cr, fmt = "k", mfc = "r", marker = "o", linestyle = "", ms = 15, elinewidth = 5)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["end"], avg_random_cr["end"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    if "simple" in config["decremental_filename"]:
        plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   fontsize=15)
        plt.ylim(0.2, 1.05)
    else:
        plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
        plt.ylim(0.4, 1.05)

    plt.show()


def plotOneAgent(config):
    # random correct rate
    random_cr = np.load("./common_data/incremental/100trial-window3-random_is_correct.npy", allow_pickle=True).item()
    # avg_random_cr = np.nanmean([np.nanmean(each) for each in random_cr])
    avg_random_cr = {each: np.nanmean(random_cr[each]) for each in random_cr}
    print("-"*15)
    # trial name, pacman pos, beans, window cr for different agents
    bean_vs_cr = np.load(config["one_agent_filename"], allow_pickle = True)
    bean_num = []
    agent_cr = []
    for i in bean_vs_cr:
        temp_cr = []
        temp_bean_num = []
        for j in i:
            temp_bean_num.append(len(j[2]))
            temp_cr.append(j[3])
        bean_num.append(copy.deepcopy(temp_bean_num))
        agent_cr.append(copy.deepcopy(temp_cr))
    # bean_num = [len(each[2]) if isinstance(each[2], list) else 0 for each in bean_vs_cr]
    # agent_cr  = [each[3] for each in bean_vs_cr]
    max_bean_num = max(max(bean_num))
    min_bean_num = min(min(bean_num))
    print("Max bean num : ", max_bean_num)
    print("Min bean num : ", min_bean_num)
    agent_index = [1, 0, 2, 3, 4, 5]
    first_phase_agent_cr = []  # num of beans <= 10
    second_phase_agent_cr = []  # 10 < num of beans < 80
    third_phase_agent_cr = []  # num of beans > 80
    # every bin
    for i in range(len(bean_num)):
        trial_bean_num = bean_num[i]
        trial_cr = agent_cr[i]
        trial_early = []
        trial_middle = []
        trial_end = []
        for j in range(len(trial_bean_num)):
            if trial_bean_num[j] <= 10:
                trial_end.append(trial_cr[j])
            elif 10 < trial_bean_num[j] < 80:
                trial_middle.append(trial_cr[j])
            else:
                trial_early.append(trial_cr[j])
        if len(trial_early) > 0:
            third_phase_agent_cr.append(np.mean(trial_early, axis=0)[agent_index])
        if len(trial_middle) > 0:
            second_phase_agent_cr.append(np.mean(trial_middle, axis=0)[agent_index])
        if len(trial_end) > 0:
            first_phase_agent_cr.append(np.mean(trial_end, axis=0)[agent_index])

    # plotting
    x_ticks = ["local", "global", "evade\n(Blinky)", "evade\n(Clyde)", "approach", "energizer"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)
    # colors = RdYlBu_5.mpl_colors
    # colors[2] = Balance_6.mpl_colors[2]
    # colors = [colors[0], colors[-1], colors[1], colors[3], colors[2]]
    colors = [
        agent_color["local"],
        agent_color["global"],
        agent_color["pessimistic_blinky"],
        agent_color["pessimistic_clyde"],
        agent_color["suicide"],
        agent_color["planned_hunting"]
    ]

    plt.figure(figsize=(25, 5))

    plt.subplot(1, 3, 1)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Early Stage (Pellets >= 80)", fontsize=20)
    avg_cr = np.mean(third_phase_agent_cr, axis=0)
    var_cr = np.nanstd(third_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["early"], avg_random_cr["early"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=15)
    plt.ylim(0.0, 1.05)
    plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)

    plt.subplot(1, 3, 2)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Middle Stage (10 < Pellets < 80)", fontsize=20)
    avg_cr = np.mean(second_phase_agent_cr, axis=0)
    var_cr = np.nanstd(second_phase_agent_cr, axis=0)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec = colors[index], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["middle"], avg_random_cr["middle"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               fontsize=15)
    plt.ylim(0.0, 1.05)

    plt.subplot(1, 3, 3)
    # plt.subplots_adjust(top=0.88,bottom=0.11,left=0.11,right=0.9,hspace=0.2,wspace=0.2)
    plt.title("Ending Stage (Pellets <= 10)", fontsize=20)
    avg_cr = np.mean(first_phase_agent_cr, axis=0)
    var_cr = np.nanstd(first_phase_agent_cr, axis=0)
    # plt.errorbar(x_index, avg_cr, yerr = var_cr, fmt = "k", mfc = "r", marker = "o", linestyle = "", ms = 15, elinewidth = 5)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], avg_cr[index], yerr=var_cr[index],
                     color=colors[index], linestyle="", ms=20, elinewidth=4,
                     mfc=colors[index], mec=colors[index], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["end"], avg_random_cr["end"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 2.75)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               fontsize=15)
    plt.ylim(0.0, 1.05)
    plt.show()


def specialANDComparison(config):
    fig = plt.figure(figsize=(27, 15), constrained_layout=False)
    spec = fig.add_gridspec(2, 3)
    # Plot State Comparison
    # random correct rate
    random_cr = np.load("./common_data/incremental/100trial-window3-random_is_correct.npy", allow_pickle=True).item()
    # avg_random_cr = np.nanmean([np.nanmean(each) for each in random_cr])
    avg_random_cr = np.nanmean([np.nanmean(random_cr[each]) for each in random_cr])
    print("-" * 15)
    # trial name, pacman pos, beans, window cr for different agents
    bean_vs_cr = np.load(config["stage_combine_filename"], allow_pickle=True)
    multi_agent_weight = np.load(config["stage_combine_weight_filename"], allow_pickle=True)[-1]
    multi_agent_weight = multi_agent_weight / np.linalg.norm(multi_agent_weight)

    agent_name = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting", "multi"]
    temp_agent_color = copy.deepcopy(agent_color)
    temp_agent_color["multi"] = "black"
    agent_index = [1, 0, 2, 3, 4, 5, 6]
    agent_name = np.array(agent_name)[agent_index]
    bean_vs_cr = bean_vs_cr[agent_index]
    # plt.figure(figsize=(16, 5))
    x_ticks = ["local", "global", "evade\n(Blinky)", "evade\n(Clyde)", "approach", "energizer", "multi"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)

    ax1 = fig.add_subplot(spec[0, 0])
    plt.title("All Scenarios", fontsize=15)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], np.nanmean(bean_vs_cr[index, :]), yerr = np.nanstd(bean_vs_cr[index, :]),
                     color=temp_agent_color[agent_name[index]], linestyle="", ms=20, elinewidth=4,
                     mfc=temp_agent_color[agent_name[index]], mec=temp_agent_color[agent_name[index]], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr, avg_random_cr], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 3.25)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    plt.ylim(0.0, 1.05)
    # plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)
    plt.ylabel("Joystick Movement Prediction Correct Rate", y=-0.2, fontsize = 15, x = -2)

    # plt.subplot(1, 2, 2)
    ax2 = fig.add_subplot(spec[0, 2])
    x_index = x_index[:-1]
    x_ticks = x_ticks[:-1]
    multi_agent_weight = multi_agent_weight[[1, 0, 2, 3, 4, 5]]
    for i, each in enumerate(multi_agent_weight):
        plt.bar(x_index[i], height=multi_agent_weight[i], width=0.4, color=agent_color[agent_name[i]])
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               fontsize=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Normalized Strategy Weight", fontsize=15)

    # For special cases
    # random correct rate
    random_cr = np.load("./common_data/special_case/100trial-random_is_correct.npy", allow_pickle=True).item()
    # avg_random_cr = np.nanmean([np.nanmean(each) for each in random_cr])
    avg_random_cr = {each: np.nanmean(random_cr[each]) for each in random_cr}
    print("-" * 15)
    # trial name, pacman pos, beans, window cr for different agents
    bean_vs_cr = np.load(config["special_case_filename"], allow_pickle=True).item()
    closed_ghost_cr = np.load(config["closed_ghost_filename"], allow_pickle=True).item()


    agent_index = [1, 0, 2, 3, 4, 5, 6]
    end_agent_cr = np.nanmean(np.array(bean_vs_cr["end"]), axis = 0)[agent_index]  # num of beans <= 10
    scared_agent_cr = np.nanmean(np.array(bean_vs_cr["close-scared"]), axis = 0)[agent_index]  # 10 < num of beans < 80
    normal_agent_cr = np.nanmean(np.array(bean_vs_cr["close-normal"]), axis = 0)[agent_index]  # num of beans > 80
    closed_blinky_normal_cr = np.nanmean(np.array(closed_ghost_cr["blinky-close-normal"]), axis = 0)[agent_index]
    closed_clyde_normal_cr = np.nanmean(np.array(closed_ghost_cr["clyde-close-normal"]), axis=0)[agent_index]

    end_agent_cr_std = np.nanstd(np.array(bean_vs_cr["end"]), axis=0)[agent_index]  # num of beans <= 10
    scared_agent_cr_std = np.nanstd(np.array(bean_vs_cr["close-scared"]), axis=0)[agent_index]  # 10 < num of beans < 80
    normal_agent_cr_std = np.nanstd(np.array(bean_vs_cr["close-normal"]), axis=0)[agent_index]  # num of beans > 80
    closed_blinky_normal_cr_std = np.nanstd(np.array(closed_ghost_cr["blinky-close-normal"]), axis=0)[agent_index]
    closed_clyde_normal_cr_std = np.nanstd(np.array(closed_ghost_cr["clyde-close-normal"]), axis=0)[agent_index]

    # plotting
    agent_name = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting", "multi"]
    temp_agent_color = copy.deepcopy(agent_color)
    temp_agent_color["multi"] = "black"
    agent_index = [1, 0, 2, 3, 4, 5, 6]
    agent_name = np.array(agent_name)[agent_index]
    # bean_vs_cr = bean_vs_cr[agent_index]
    # plt.figure(figsize=(23, 5))
    x_ticks = ["local", "global", "evade\n(Blinky)", "evade\n(Clyde)", "approach", "energizer", "multi"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)

    # plt.subplot(1, 3, 1)
    ax3 = fig.add_subplot(spec[0, 1])
    plt.title("Ending Game", fontsize=15)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], end_agent_cr[index], yerr=end_agent_cr_std[index],
                     color=temp_agent_color[agent_name[index]], linestyle="", ms=20, elinewidth=4,
                     mfc=temp_agent_color[agent_name[index]], mec=temp_agent_color[agent_name[index]], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["end"], avg_random_cr["end"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 3.25)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    plt.ylim(0.0, 1.05)
    # plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)

    # plt.subplot(1, 3, 2)
    ax4 = fig.add_subplot(spec[1, 0])
    plt.title("Close Normal Blinky", fontsize=15)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], closed_blinky_normal_cr[index], yerr=closed_blinky_normal_cr_std[index],
                     color=temp_agent_color[agent_name[index]], linestyle="", ms=20, elinewidth=4,
                     mfc=temp_agent_color[agent_name[index]], mec=temp_agent_color[agent_name[index]], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["close-normal"], avg_random_cr["close-normal"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 3.25)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    plt.ylim(0.0, 1.05)
    # plt.ylabel("Joystick Movement Prediction Correct Rate", fontsize=15)

    # plt.subplot(1, 3, 3)
    ax5 = fig.add_subplot(spec[1, 2])
    plt.title("Close Scared Ghosts", fontsize=15)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], scared_agent_cr[index], yerr=scared_agent_cr_std[index],
                     color=temp_agent_color[agent_name[index]], linestyle="", ms=20, elinewidth=4,
                     mfc=temp_agent_color[agent_name[index]], mec=temp_agent_color[agent_name[index]], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["close-scared"], avg_random_cr["close-scared"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 3.25)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    plt.ylim(0.0, 1.05)

    # TODO: This !!!!!
    ax6 = fig.add_subplot(spec[1, 1])
    plt.title("Close Normal Clyde", fontsize=15)
    for index, each in enumerate(x_index):
        plt.errorbar(x_index[index], closed_clyde_normal_cr[index], yerr=closed_clyde_normal_cr_std[index],
                     color=temp_agent_color[agent_name[index]], linestyle="", ms=20, elinewidth=4,
                     mfc=temp_agent_color[agent_name[index]], mec=temp_agent_color[agent_name[index]], marker="o")
    plt.plot([-0.5, 3.5], [avg_random_cr["close-normal"], avg_random_cr["close-normal"]], "--", lw=5, color="grey")
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.xlim(-0.25, 3.25)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
               [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=15)
    plt.ylim(0.0, 1.05)


    plt.savefig("./common_data/special_case/special_and_comparison.pdf")
    plt.show()


def specialWeight(config):

    multi_agent_weight = np.load(config["special_weight_filename"], allow_pickle=True).item()
    multi_agent_weight = {
        each:multi_agent_weight[each][-1] / np.linalg.norm(multi_agent_weight[each][-1]) for each in multi_agent_weight
    }

    # plt.subplot(1, 2, 2)
    agent_name = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting", "multi"]
    agent_index = [1, 0, 2, 3, 4, 5, 6]
    agent_name = np.array(agent_name)[agent_index]

    x_ticks = ["local", "global", "evade\n(Blinky)", "evade\n(Clyde)", "approach", "energizer", "multi"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)
    x_index = x_index[:-1]
    x_ticks = x_ticks[:-1]

    case_name = ["Early Stage (Pellets >= 80)", "Middle Stage (10 < Pellets < 80)", "End Stage (Pellets <= 10)"]
    plt.figure(figsize=(23, 5))
    for index, case in enumerate(["early", "middle", "end"]):
        plt.subplot(1, 3, index +1)
        plt.title(case_name[index], fontsize = 15)
        temp_multi_agent_weight = multi_agent_weight[case][[1, 0, 2, 3, 4, 5]]
        for i, each in enumerate(multi_agent_weight):
            plt.bar(x_index[i], height=temp_multi_agent_weight[i], width=0.4, color=agent_color[agent_name[i]])
        plt.xticks(x_index, x_ticks, fontsize=15)
        plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                   fontsize=15)
        plt.ylim(0.0, 1.0)
        plt.ylabel("Normalized Strategy Weight", fontsize=15)

    # plt.savefig("./common_data/special_case/special_case_weight.pdf")
    plt.show()


def plotCloseGhost(config):
    # Plot State Comparison
    # blinky-close-normal / clyde-close-normal / blinky-close-scared / clyde-close-scared
    weight = np.load(config["closed_ghost_weight"], allow_pickle=True).item()
    multi_agent_weight = {each:weight[each][-1] for each in weight}
    for each in multi_agent_weight:
        multi_agent_weight[each] = multi_agent_weight[each] / np.linalg.norm(multi_agent_weight[each])
        multi_agent_weight[each] = multi_agent_weight[each][[1, 0, 2, 3, 4, 5]]
    agent_name = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting", "multi"]
    agent_name = np.array(agent_name)[[1,0,2,3,4,5]]
    temp_agent_color = copy.deepcopy(agent_color)
    temp_agent_color["multi"] = "black"

    x_ticks = ["local", "global", "evade\n(Blinky)", "evade\n(Clyde)", "approach", "energizer", "multi"]
    x_index = np.arange(0, len(x_ticks) / 2, 0.5)

    x_index = x_index[:-1]
    x_ticks = x_ticks[:-1]
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 2, 1)
    case_weight = multi_agent_weight["blinky-close-normal"]
    plt.title("Closed Normal Blinky", fontsize = 15)
    for i, each in enumerate(case_weight):
        plt.bar(x_index[i], height=case_weight[i], width=0.4, color=agent_color[agent_name[i]])
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               fontsize=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Normalized Strategy Weight", fontsize=15)
    plt.subplot(1, 2, 2)
    case_weight = multi_agent_weight["clyde-close-normal"]
    plt.title("Closed Normal Clyde", fontsize=15)
    for i, each in enumerate(case_weight):
        plt.bar(x_index[i], height=case_weight[i], width=0.4, color=agent_color[agent_name[i]])
    plt.xticks(x_index, x_ticks, fontsize=15)
    plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
               fontsize=15)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Normalized Strategy Weight", fontsize=15)
    plt.show()


def plotStateComparison(config):
    width = 0.4
    color = RdBu_8.mpl_colors
    random_data = np.load("./common_data/state_comparison/1000trial-random_is_correct.npy", allow_pickle=True).item()
    avg_random_cr = {each:np.nanmean(random_data[each]) for each in random_data}
    filename = "common_data/state_comparison/equal-100trial_Omega_diff_state_agent_cr.npy"
    # filename = "common_data/state_comparison/1000trial_Patamon_diff_state_agent_cr.npy"

    state_cr = np.load(filename, allow_pickle=True)
    state_names = ["global", "local", "evade", "approach", "energizer", "vague"]

    only_local = []
    all_agents = []
    for i in range(6):
        only_local.append([np.nanmean([j[0] for j in each]) for each in state_cr[i]])
        all_agents.append([np.nanmean([j[1] for j in each]) for each in state_cr[i]])
    avg_only_local = [np.nanmean(each) for each in only_local]
    std_only_local = [np.nanstd(each) for each in only_local]
    sem_only_local = [scipy.stats.sem(each) for each in only_local]
    avg_all_agents = [np.nanmean(each) for each in all_agents]
    std_all_agents = [np.nanstd(each) for each in all_agents]
    sem_all_agents = [scipy.stats.sem(each) for each in all_agents]

    plt.figure(figsize=(10,7))
    plt.bar(x = np.arange(0, 6) - width, height = avg_only_local, width = width, label = "Local Agent",
            color = color[0], yerr = sem_only_local, capsize = 7, error_kw = {"capthick":3, "elinewidth":3})
    plt.bar(x = np.arange(0, 6), height=avg_all_agents, width = 0.4, label = "All Agents",
            color = color[-1], yerr = sem_all_agents, capsize = 7, error_kw = {"capthick":3, "elinewidth":3})
    # plt.bar(x=np.arange(0, 6) - width, height=avg_only_local, width=width, label="Local Agent", color=color[0])
    # plt.bar(x=np.arange(0, 6), height=avg_all_agents, width=0.4, label="All Agents", color=color[-1])
    x_index = [[i-3*width/2, i+width/2] for i in range(6)]
    label_list = ["global", "local", "evade", "suicide", "attack", "vague"]
    for i in range(6):
        plt.plot(x_index[i], [avg_random_cr[label_list[i]], avg_random_cr[label_list[i]]], "--", lw = 4, color="k")
    plt.xticks(np.arange(0, 6)-width/2, state_names, fontsize = 20)
    # plt.ylim(0.0, 1.2)
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
    plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
    plt.ylabel("Joystick Movement Estimation Correct Rate", fontsize = 20)
    plt.ylim(0.4, 1.1)
    plt.legend(frameon = False, fontsize = 20, ncol = 2)
    plt.show()

# ===================================================

def plotOptionComparison(config):
    # read data
    width = 0.4
    color = RdBu_8.mpl_colors
    random_data = np.load("./common_data/special_case/100trial-all_random_is_correct.npy", allow_pickle=True).item()
    avg_random_cr = {each: np.nanmean(random_data[each]) for each in random_data}
    # Correct rate data
    hybrid_cr = np.load(config["option_hybrid_filename"], allow_pickle=True).item()
    moving_cr = np.load(config["option_moving_filename"], allow_pickle=True).item()
    all_types = ["early", "middle", "end", "close-normal", "close-scared"]
    type_name = {"early":"Early Stage", "middle":"Middle Stage", "end":"Ending Stage",
                 "close-normal":"Close Normal Ghost", "close-scared":"Close Scared Ghost"}
    # avg and std
    avg_hybrid_cr = {each:np.nanmean(np.array(hybrid_cr[each]), axis = 0) for each in hybrid_cr}
    std_hybrid_cr = {each: scipy.stats.sem(np.array(hybrid_cr[each]), axis = 0, nan_policy="omit") for each in hybrid_cr}
    avg_moving_cr = {each: np.nanmean([np.nanmean(each, axis = 0) for each in moving_cr[each]], axis = 0) for each in moving_cr}
    std_moving_cr = {each: scipy.stats.sem([np.nanmean(each, axis = 0) for each in moving_cr[each]], axis = 0, nan_policy="omit") for each in moving_cr}

    plt.figure(figsize=(19, 7))
    plt.bar(x=np.arange(0, 5) - width, height=[avg_hybrid_cr[each][-1] for each in all_types], width=width, label="Static Hybrid Strategy",
            color=color[0], yerr=[std_hybrid_cr[each][-1] for each in all_types], capsize=7, error_kw={"capthick": 3, "elinewidth": 3})
    plt.bar(x=np.arange(0, 5), height=[avg_moving_cr[each][-1] for each in all_types], width=0.4, label="Dynamic Hybrid Strategy",
            color=color[-1], yerr=[std_moving_cr[each][-1] for each in all_types], capsize=7, error_kw={"capthick": 3, "elinewidth": 3})
    x_index = [[i - 3 * width / 2, i + width / 2] for i in range(5)]
    for i in range(5):
        plt.plot(x_index[i], [avg_random_cr[all_types[i]], avg_random_cr[all_types[i]]], "--", lw=4, color="k")
    plt.xticks(np.arange(0, 5) - width / 2, [type_name[each] for each in all_types], fontsize=20)
    # plt.ylim(0.0, 1.2)
    # plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], [0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 20)
    plt.ylim(0.4, 1.1)
    plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fontsize=20)
    plt.ylabel("Prediction Accuracy", fontsize=20)
    plt.legend(frameon=False, fontsize=20, ncol=2)
    plt.show()


# ===================================================
def _extractPathLabel(config):
    print("="*20, " Simple Label Analysis ", "="*20)
    record = np.load(config["equal_label_filename"], allow_pickle=True)
    print("Num of trials : ", len(record))
    # every type of data
    after_step = 5
    PA = []
    AA = []
    AAG = []
    AANG = []
    # assign type
    for trial_index, each in enumerate(record):
        trial_name = each[0]
        normalized_contribution = each[1]
        estimated_label = each[2]
        estimated_label = [a[0] for a in estimated_label]
        energizers_num = each[3]
        ghost_status = each[4]
        trial_length = len(normalized_contribution)
        # index where eat an energizer
        eat_index = np.where(np.diff(energizers_num) == -1)[0] + 1
        if len(eat_index) == 0:
            continue
        for e_i in eat_index:
            if (e_i + after_step) >= trial_length:
                continue
            # find where a ghost is eaten
            first_dead_index = None
            for j in range(e_i + 1, trial_length):
                # Until a ghost is eaten
                if np.all(np.array(ghost_status.values[j]) <= 3):
                    break
                if (ghost_status.values[j - 1][0] != 3 and ghost_status.values[j][0] == 3) or \
                        (ghost_status.values[j - 1][1] != 3 and ghost_status.values[j][1] == 3):
                    first_dead_index = j
                    break
            # find where the scared time ends
            scared_end_index = None
            for j in range(e_i+1, trial_length):
                # Until scared time ends
                if np.all(np.array(ghost_status.values[j]) <= 3) and np.any(np.array(ghost_status.values[j]) > 3):
                    scared_end_index = j
                    break
            scared_end_index = trial_length-1 if scared_end_index is None else scared_end_index
            # Planned Attack
            # if np.sum(np.array(estimated_label[e_i:e_i+after_step]) == "suicide")/len(estimated_label[e_i:e_i+after_step]) > 0.8:
            # if np.all(np.array(estimated_label[e_i:e_i+after_step]) == "suicide"):
            if np.any(np.array(estimated_label[e_i:e_i+after_step]) == "suicide"):
                # No ghost is eaten in PA
                if first_dead_index is None:
                    continue
                PA.append([
                    normalized_contribution,
                    e_i,
                    first_dead_index,
                    scared_end_index,
                    trial_length
                ])
            # Accidental Attack
            else:
                AA.append([
                    normalized_contribution,
                    e_i,
                    first_dead_index,
                    scared_end_index,
                    trial_length
                ])
                if first_dead_index is not None:
                    AAG.append([
                        normalized_contribution,
                        e_i,
                        first_dead_index,
                        scared_end_index,
                        trial_length
                    ])
                else:
                    AANG.append([
                        normalized_contribution,
                        e_i,
                        first_dead_index,
                        scared_end_index,
                        trial_length
                    ])
    # Summary of different type
    print("Planned Attack (PA) | {} trajectories |".format(len(PA)))
    print("Accidental Attack (AA) | {} trajectories |".format(len(AA)))
    print("Accidental Attack w/ Ghost Eaten (AAG) | {} trajectories |".format(len(AAG)))
    print("Accidental Attack w/o Ghost Eaten (AANG) | {} trajectories |".format(len(AANG)))
    # Save for plotting
    np.save("./common_data/path_label_analysis/{}-extracted_label.npy".format(
        config["equal_label_filename"].split("/")[-1].split(".")[-2]),
        {"PA":PA, "AA":AA, "AAG":AAG, "AANG":AANG}
    )


def plotPathLabel(config, need_se = True):
    data = np.load(config["equal_extracted_filename"], allow_pickle=True).item()
    PA = data["PA"]
    AA= data["AA"]
    AAG = data["AAG"]
    AANG = data["AANG"]
    # |A| PA sort at energizer
    PA_at_energizer = np.zeros((len(PA), 11, 6))
    PA_at_energizer[PA_at_energizer == 0] = np.nan
    for traj_index, trajectory in enumerate(PA):
        contribution = trajectory[0]
        center_index = trajectory[1]
        start_index = max(0, center_index - 5)
        end_index = min(center_index + 5 + 1, trajectory[4])
        before_num = center_index - start_index
        after_num = end_index - center_index - 1
        for i in range(before_num):
            PA_at_energizer[traj_index, 4-i, :] = copy.deepcopy(contribution[center_index-1-i])
        PA_at_energizer[traj_index, 5, :] = copy.deepcopy(contribution[center_index])
        for i in range(after_num):
            PA_at_energizer[traj_index, 6+i, :] = copy.deepcopy(contribution[center_index+1+i])
    # |B| AA sort at energizer
    AA_at_energizer = np.zeros((len(AA), 11, 6))
    AA_at_energizer[AA_at_energizer == 0] = np.nan
    for traj_index, trajectory in enumerate(AA):
        contribution = trajectory[0]
        center_index = trajectory[1]
        start_index = max(0, center_index - 5)
        end_index = min(center_index + 5 + 1, trajectory[4])
        before_num = center_index - start_index
        after_num = end_index - center_index - 1
        for i in range(before_num):
            AA_at_energizer[traj_index, 4 - i, :] = copy.deepcopy(contribution[center_index - 1 - i])
        AA_at_energizer[traj_index, 5, :] = copy.deepcopy(contribution[center_index])
        for i in range(after_num):
            AA_at_energizer[traj_index, 6 + i, :] = copy.deepcopy(contribution[center_index + 1 + i])
    # |C| PA sort at ghost
    PA_at_ghost = np.zeros((len(PA), 11, 6))
    PA_at_ghost[PA_at_ghost == 0] = np.nan
    for traj_index, trajectory in enumerate(PA):
        contribution = trajectory[0]
        center_index = trajectory[2]
        start_index = max(trajectory[1], center_index - 10)
        before_num = center_index - start_index
        for i in range(before_num):
            PA_at_ghost[traj_index, 9 - i, :] = copy.deepcopy(contribution[center_index - 1 - i])
        PA_at_ghost[traj_index, 10, :] = copy.deepcopy(contribution[center_index])
    # |D| AAG sort at ghost
    AAG_at_ghost = np.zeros((len(AAG), 11, 6))
    AAG_at_ghost[AAG_at_ghost == 0] = np.nan
    for traj_index, trajectory in enumerate(AAG):
        contribution = trajectory[0]
        center_index = trajectory[2]
        start_index = max(trajectory[1], center_index - 10)
        before_num = center_index - start_index
        for i in range(before_num):
            AAG_at_ghost[traj_index, 9 - i, :] = copy.deepcopy(contribution[center_index - 1 - i])
        AAG_at_ghost[traj_index, 10, :] = copy.deepcopy(contribution[center_index])
    # |E| AANG sort at normal
    AANG_at_normal = np.zeros((len(AANG), 11, 6))
    AANG_at_normal[AANG_at_normal == 0] = np.nan
    for traj_index, trajectory in enumerate(AANG):
        contribution = trajectory[0]
        center_index = trajectory[3]
        start_index = max(0, center_index - 10)
        before_num = center_index - start_index
        for i in range(before_num):
            AANG_at_normal[traj_index, 9 - i, :] = copy.deepcopy(contribution[center_index - 1 - i])
        AANG_at_normal[traj_index, 10, :] = copy.deepcopy(contribution[center_index])
    # |F| AA sort at ends (AAG ghost / AANG normal)
    AA_at_end = np.zeros((len(AA), 11, 6))
    AA_at_end[AA_at_end == 0] = np.nan
    for traj_index, trajectory in enumerate(AA):
        contribution = trajectory[0]
        center_index = trajectory[2] if trajectory[2] is not None else trajectory[3]
        start_index = max(0, center_index - 10)
        before_num = center_index - start_index
        for i in range(before_num):
            AA_at_end[traj_index, 9 - i, :] = copy.deepcopy(contribution[center_index - 1 - i])
        AA_at_end[traj_index, 10, :] = copy.deepcopy(contribution[center_index])
    # ===============================================
    # Plotting
    label_name = {
        "local": "local",
        "pessimistic": "evade",
        "pessimistic_blinky": "evade (Blinky)",
        "pessimistic_clyde": "evade (Clyde)",
        "global": "global",
        "suicide": "approach",
        "planned_hunting": "energizer"
    }
    agent_name = ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"]

    plt.figure(figsize=(26, 15))
    plt.subplot(2, 3, 1)
    plt.title("Planned Attack at Energizer", fontsize=15)
    PA_at_energizer = np.array(PA_at_energizer)
    avg_PA_at_energizer = np.nanmean(PA_at_energizer, axis=0)
    sem_weight = scipy.stats.sem(PA_at_energizer, axis=0, nan_policy="omit")

    for index in range(len(agent_name)):
        plt.plot(avg_PA_at_energizer[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        if need_se:
            plt.fill_between(
                np.arange(0, avg_PA_at_energizer.shape[0]),
                avg_PA_at_energizer[:, index] - sem_weight[:, index],
                avg_PA_at_energizer[:, index] + sem_weight[:, index],
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.ylabel("Normalized Strategy Weight", fontsize=15)
    plt.xlim(0, avg_PA_at_energizer.shape[1]-1)
    plt.xticks([0, 5, 10], ["-5", "Energizer \n Consumption", "5"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.01, 1.02)
    plt.legend(frameon=False, fontsize=10, loc = "upper center", ncol = 3)

    plt.subplot(2, 3, 2)
    plt.title("Accidental Attack at Energizer", fontsize=15)
    AA_at_energizer = np.array(AA_at_energizer)
    avg_AA_at_energizer = np.nanmean(AA_at_energizer, axis=0)
    # std_planned_redundant_weight = np.nanstd(planned_redundant_weight, axis = 0)
    sem_weight = scipy.stats.sem(AA_at_energizer, axis=0, nan_policy="omit")
    for index in range(len(agent_name)):
        plt.plot(avg_AA_at_energizer[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        if need_se:
            plt.fill_between(
                np.arange(0, avg_AA_at_energizer.shape[0]),
                avg_AA_at_energizer[:, index] - sem_weight[:, index],
                avg_AA_at_energizer[:, index] + sem_weight[:, index],
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.ylabel("Normalized Strategy Weight", fontsize=15)
    plt.xlim(0, avg_AA_at_energizer.shape[1]-1)
    plt.xticks([0, 5, 10], ["-5", "Energizer \n Consumption", "5"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.01, 1.02)
    plt.legend(frameon=False, fontsize=10, loc = "upper center", ncol = 3)

    plt.subplot(2, 3, 4)
    plt.title("Planned Attack at Ghost", fontsize=15)
    PA_at_ghost = np.array(PA_at_ghost)
    avg_PA_at_ghost = np.nanmean(PA_at_ghost, axis=0)
    sem_weight = scipy.stats.sem(PA_at_ghost, axis=0, nan_policy="omit")
    for index in range(len(agent_name)):
        plt.plot(avg_PA_at_ghost[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        if need_se:
            plt.fill_between(
                np.arange(0, avg_PA_at_ghost.shape[0]),
                avg_PA_at_ghost[:, index] - sem_weight[:, index],
                avg_PA_at_ghost[:, index] + sem_weight[:, index],
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.xlim(0, avg_PA_at_ghost.shape[1]-1)
    plt.xticks([0, 5, 10], ["-10", "-5", "Ghost \n Consumption"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.01, 1.02)
    plt.legend(frameon=False, fontsize=10, loc = "upper center", ncol = 3)

    plt.subplot(2, 3, 5)
    plt.title("Accidental Attack at Ghost", fontsize=15)
    AAG_at_ghost = np.array(AAG_at_ghost)
    avg_AAG_at_ghost = np.nanmean(AAG_at_ghost, axis=0)
    sem_weight = scipy.stats.sem(AAG_at_ghost, axis=0, nan_policy="omit")
    for index in range(len(agent_name)):
        plt.plot(avg_AAG_at_ghost[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        if need_se:
            plt.fill_between(
                np.arange(0, avg_AAG_at_ghost.shape[0]),
                avg_AAG_at_ghost[:, index] - sem_weight[:, index],
                avg_AAG_at_ghost[:, index] + sem_weight[:, index],
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.xlim(0, avg_AAG_at_ghost.shape[1]-1)
    plt.xticks([0, 5, 10], ["-10", "-5", "Ghost \n Consumption"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.01, 1.02)
    plt.legend(frameon=False, fontsize=10, loc = "upper center", ncol = 3)

    plt.subplot(2, 3, 3)
    plt.title("Accidental Attack at Normal", fontsize=15)
    AANG_at_normal = np.array(AANG_at_normal)
    avg_AANG_at_normal = np.nanmean(AANG_at_normal, axis=0)
    sem_weight = scipy.stats.sem(AANG_at_normal, axis=0, nan_policy="omit")
    for index in range(len(agent_name)):
        plt.plot(avg_AANG_at_normal[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        if need_se:
            plt.fill_between(
                np.arange(0, avg_AANG_at_normal.shape[0]),
                avg_AANG_at_normal[:, index] - sem_weight[:, index],
                avg_AANG_at_normal[:, index] + sem_weight[:, index],
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.xlim(0, avg_AANG_at_normal.shape[1]-1)
    plt.xticks([0, 5, 10], ["-10", "-5", "Return \n Normal"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.01, 1.02)
    plt.legend(frameon=False, fontsize=10, loc = "upper center", ncol = 3)

    plt.subplot(2, 3, 6)
    plt.title("Accidental Attack at End", fontsize=15)
    AA_at_end = np.array(AA_at_end)
    avg_AA_at_end = np.nanmean(AA_at_end, axis=0)
    # std_planned_redundant_weight = np.nanstd(planned_redundant_weight, axis = 0)
    sem_weight = scipy.stats.sem(AA_at_end, axis=0, nan_policy="omit")
    for index in range(len(agent_name)):
        plt.plot(avg_AA_at_end[:, index], color=agent_color[agent_name[index]], ms=3, lw=5,
                 label=label_name[agent_name[index]])
        if need_se:
            plt.fill_between(
                np.arange(0, avg_AA_at_end.shape[0]),
                avg_AA_at_end[:, index] - sem_weight[:, index],
                avg_AA_at_end[:, index] + sem_weight[:, index],
                color=agent_color[agent_name[index]],
                alpha=0.3,
                linewidth=4
            )
    plt.xlim(0, avg_AA_at_end.shape[1]-1)
    plt.xticks([0, 5, 10], ["-10", "-5", "Return \n Normal"], fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-0.01, 1.02)
    plt.legend(frameon=False, fontsize=10, loc = "upper center", ncol = 3)
    plt.savefig("./common_data/path_label_analysis/3.4_PA_AA.pdf")
    plt.show()




if __name__ == '__main__':
    # Configurations
    pd.options.mode.chained_assignment = None

    config = {
        # TODO: ===================================
        # TODO:       Always set to True
        # TODO: ===================================
        "need_intercept" : True,
        "maximum_try": 5,

        "single_trial_data_filename": "./common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl",
        # The number of trials used for analysis
        "trial_num": None,
        # Window size for correlation analysis
        "single_trial_window": 3,
        "single_trial_agents": ["global", "local", "pessimistic_blinky", "pessimistic_clyde", "suicide", "planned_hunting"],

        # ==================================================================================
        #                       For Experimental Results Visualization
        "estimated_label_filename": "./common_data/{}/trajectory-with_Q-window3-w_intercept-multi_labels.npy",
        "handcrafted_label_filename": "./common_data/{}/trajectory-with_Q-window3-w_intercept-handcrafted_labels.npy",
        "trial_weight_filename": "./common_data/{}/trajectory-with_Q-window3-w_intercept-trial_weight.npy",
        "trial_Q_filename": "./common_data/{}/trajectory-with_Q-window3-w_intercept-Q.npy",
        "trial_matching_rate_filename": "./common_data/{}/trajectory-with_Q-window3-w_intercept-matching_rate.npy",
        # "trial_agent_name" : ["global", "local", "planned_hunting"],
        "trial_window": 3,

        "agent_list" : [["local", "global"], ["local", "pessimistic"], ["local", "global"],
                        ["local", "pessimistic"], ["local", "planned_hunting"], ["local", "suicide"]],

        # ------------------------------------------------------------------------------------

        "bean_vs_cr_filename": "./common_data/incremental/path10-window3-incremental_cr-w_intercept.npy",
        "one_agent_filename": "./common_data/one_agent/path10-100trial-window3-incremental_cr-w_intercept.npy",
        "decremental_filename": "./common_data/decremental/path10-100trial-window3-incremental_cr-w_intercept.npy",
        "stage_together_filename": "./common_data/stage_together/path10-all-100trial-cr.npy",
        "closed_ghost_filename": "./common_data/closed_ghost/path10-100trial-cr.npy",

        "stage_combine_filename": "./common_data/stage_together/path10-all-100trial-cr.npy",
        "stage_combine_weight_filename": "./common_data/stage_together/path10-all-100trial-weight.npy",

        "special_case_filename": "./common_data/special_case/path10-100trial-cr.npy",
        "special_weight_filename": "./common_data/special_case/path10-100trial-contribution.npy",

        "equal_label_filename": "./common_data/path_label_analysis/100_trial_data_Omega-with_Q-path10-record.npy",
        "equal_extracted_filename": "./common_data/path_label_analysis/100_trial_data_Omega-with_Q-path10-record-extracted_label.npy",

        # "option_hybrid_filename": "./common_data/special_case/descriptive-100trial-cr.npy",
        # "option_moving_filename": "./common_data/special_case/descriptive-100trial-moving_window-cr.npy",
        # "option_hybrid_filename": "./common_data/special_case/100trial-cr.npy",
        # "option_moving_filename": "./common_data/special_case/100trial-moving_window-cr.npy",
        "option_hybrid_filename": "./common_data/special_case/path10-100trial-cr.npy",
        "option_moving_filename": "./common_data/special_case/path10-100trial-moving_window-cr.npy",

        "closed_ghost_cr":"./common_data/closed_ghost/path10-100trial-cr.npy",
        "closed_ghost_weight":"./common_data/closed_ghost/path10-100trial-contribution.npy",
    }

    # ============ VISUALIZATION =============
    # Do not use these two functions
    # plotThreeAgentMatching(config) # For three agent
    # plotLocalEvadeSuicideMatching(config) # For local, evade, and suicide

    # plotGlobalLocalAttackMatching(config)
    # plotLocalEvadeSuicideMatching(config)
    # plotAllAgentMatching(config)

    # plotWeightVariation(config)
    # plotTestWeight()

    plotIncremental(config)
    # plotOneAgent(config)
    # plotDecremental(config)
    # plotStateComparison(config)
    #
    # plotCloseGhost(config)
    #
    # specialANDComparison(config)

    # specialWeight(config)

    # plotOptionComparison(config)

    # _extractPathLabel(config)
    # plotPathLabel(config, need_se=False)


    # singleTrialMultiFitting(config)


