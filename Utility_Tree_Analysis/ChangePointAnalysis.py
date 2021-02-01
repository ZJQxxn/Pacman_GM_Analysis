import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle
import scipy
import copy
import ruptures as rpt
import warnings

warnings.filterwarnings('ignore')

sys.path.append("./")

from PathAnalysis import (
    _PG,
    _PGWODead,
    _PE,
    _ghostStatus,
    _energizerNum,
    _PR,
    _RR,
    _pessimisticProcesing,
    _plannedHuntingProcesing,
    _suicideProcesing,
    _globalProcesing,
    _localProcesing,
)

params = {
    "pdf.fonttype": 42,
    "legend.frameon": False,
    "font.sans-serif": "CMU Serif",
    "font.family": "sans-serif",
}
plt.rcParams.update(params)

plt.rcParams.update(params)
pd.set_option("display.float_format", "{:.5f}".format)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 200)

# =================================================
agents = [
        "global",
        "local",
        "pessimistic_blinky",
        "pessimistic_clyde",
        "suicide",
        "planned_hunting",
]

def _estimationVagueLabeling(contributions, all_agent_name):
    sorted_contributions = np.sort(contributions)[::-1]
    if sorted_contributions[0] - sorted_contributions[1] <= 0.1 :
        return ["vague"]
    else:
        label = all_agent_name[np.argmax(contributions)]
        return [label]


def _makeChoice(prob):
    copy_estimated = copy.deepcopy(prob)
    if np.any(prob) < 0:
        available_dir_index = np.where(prob != 0)
        copy_estimated[available_dir_index] = (
            copy_estimated[available_dir_index]
            - np.min(copy_estimated[available_dir_index])
            + 1
        )
    return np.random.choice([idx for idx, i in enumerate(prob) if i == max(prob)])


def negativeLikelihood(
    param, all_data, true_prob, agents_list, return_trajectory=False, suffix="_Q"
):
    """
    Estimate agent weights with utility (Q-value).
    :param param:
    :param all_data:
    :param agent_list: ®
    :param return_trajectory:
    :return:
    """
    if 0 == len(agents_list) or None == agents_list:
        raise ValueError("Undefined agents list!")
    else:
        agent_weight = [param[i] for i in range(len(param))]
    # Compute estimation error
    nll = 0  # negative log likelihood
    num_samples = all_data.shape[0]
    agents_list = [("{}" + suffix).format(each) for each in agents_list]
    pre_estimation = all_data[agents_list].values
    agent_Q_value = np.zeros((num_samples, 4, len(agents_list)))
    for each_sample in range(num_samples):
        for each_agent in range(len(agents_list)):
            agent_Q_value[each_sample, :, each_agent] = pre_estimation[each_sample][
                each_agent
            ]
    dir_Q_value = agent_Q_value @ agent_weight
    true_dir = true_prob.apply(lambda x: _makeChoice(x)).values
    exp_prob = np.exp(dir_Q_value)
    for each_sample in range(num_samples):
        if np.isnan(dir_Q_value[each_sample][0]):
            continue
        log_likelihood = dir_Q_value[each_sample, true_dir[each_sample]] - np.log(
            np.sum(exp_prob[each_sample])
        )
        nll = nll - log_likelihood
    if not return_trajectory:
        return nll
    else:
        return (nll, dir_Q_value)


def oneHot(val):
    """
    Convert the direction into a one-hot vector.
    :param val: The direction. should be the type ``str''.
    :return:
    """
    dir_list = ["left", "right", "up", "down"]
    # Type check
    if val not in dir_list:
        raise ValueError("Undefined direction {}!".format(val))
    if not isinstance(val, str):
        raise TypeError("Undefined direction type {}!".format(type(val)))
    # One-hot
    onehot_vec = [0, 0, 0, 0]
    onehot_vec[dir_list.index(val)] = 1
    return onehot_vec


def normalize(x):
    return (x) / (x).sum()


def caculate_correct_rate(result_x, all_data, true_prob, agents, suffix="_Q"):
    _, estimated_prob = negativeLikelihood(
        result_x, all_data, true_prob, agents, return_trajectory=True, suffix=suffix
    )
    true_dir = np.array([np.argmax(each) for each in true_prob])
    estimated_dir = np.array([_makeChoice(each) for each in estimated_prob])
    correct_rate = np.sum(estimated_dir == true_dir) / len(estimated_dir)
    return correct_rate


def scaleOfNumber(num):
    """
    Obtain the scale of a number.
    :param num: The number
    :return:
    """
    if num >= 1:
        order = len(str(num).split(".")[0])
        return 10 ** (order - 1)
    elif num == 0:
        return 0
    else:
        order = str(num).split(".")[1]
        temp = 0
        for each in order:
            if each == "0":
                temp += 1
            else:
                break
        return 10 ** (-temp - 1)


def check_change(curr_w, prev_w):
    return (abs(normalize(curr_w) - normalize(prev_w))).sum()


def readLocDistance(filename):
    """
    Read in the location distance.
    :param filename: File name.
    :return: A pandas.DataFrame denoting the dijkstra distance between every two locations of the map.
    """
    locs_df = pd.read_csv(filename)[["pos1", "pos2", "dis"]]
    locs_df.pos1, locs_df.pos2 = (locs_df.pos1.apply(eval), locs_df.pos2.apply(eval))
    dict_locs_df = {}
    for each in locs_df.values:
        if each[0] not in dict_locs_df:
            dict_locs_df[each[0]] = {}
        dict_locs_df[each[0]][each[1]] = each[2]
    # correct the distance between two ends of the tunnel
    dict_locs_df[(0, 18)][(29, 18)] = 1
    dict_locs_df[(0, 18)][(1, 18)] = 1
    dict_locs_df[(29, 18)][(0, 18)] = 1
    dict_locs_df[(29, 18)][(28, 18)] = 1
    return dict_locs_df


def revise_q(df_monkey, locs_df):
    PG = df_monkey[
        ["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]
    ].apply(lambda x: _PG(x, locs_df), axis=1)
    PG_wo_dead = df_monkey[
        ["pacmanPos", "ghost1Pos", "ghost2Pos", "ifscared1", "ifscared2"]
    ].apply(lambda x: _PGWODead(x, locs_df), axis=1)
    PE = df_monkey[["pacmanPos", "energizers"]].apply(lambda x: _PE(x, locs_df), axis=1)
    ghost_status = df_monkey[["ifscared1", "ifscared2"]].apply(
        lambda x: _ghostStatus(x), axis=1
    )
    energizer_num = df_monkey[["energizers"]].apply(lambda x: _energizerNum(x), axis=1)
    PR = df_monkey[
        [
            "pacmanPos",
            "energizers",
            "beans",
            "fruitPos",
            "ghost1Pos",
            "ghost2Pos",
            "ifscared1",
            "ifscared2",
        ]
    ].apply(lambda x: _PR(x, locs_df), axis=1)

    RR = df_monkey[
        [
            "pacmanPos",
            "energizers",
            "beans",
            "fruitPos",
            "ghost1Pos",
            "ghost2Pos",
            "ifscared1",
            "ifscared2",
        ]
    ].apply(lambda x: _RR(x, locs_df), axis=1)

    df_monkey.pessimistic_blinky_Q = _pessimisticProcesing(
        df_monkey.pessimistic_blinky_Q, PG, ghost_status
    )
    df_monkey.pessimistic_clyde_Q = _pessimisticProcesing(
        df_monkey.pessimistic_clyde_Q, PG, ghost_status
    )
    df_monkey.planned_hunting_Q = _plannedHuntingProcesing(
        df_monkey.planned_hunting_Q, ghost_status, energizer_num, PE, PG_wo_dead
    )
    df_monkey.suicide_Q = _suicideProcesing(
        df_monkey.suicide_Q, PR, RR, ghost_status, PG_wo_dead
    )
    return df_monkey


def change_dir_index(x):
    temp = pd.Series((x != x.shift()).where(lambda x: x == True).dropna().index)
    return temp[(temp - temp.shift()) > 1].values


def fit_func(df_monkey, cutoff_pts, suffix="_Q"):
    result_list = []
    bounds = [[0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000], [0, 1000]]
    params = [0] * 6
    cons = []  # construct the bounds in the form of constraints
    agents = [
        "global",
        "local",
        "pessimistic_blinky",
        "pessimistic_clyde",
        "suicide",
        "planned_hunting",
    ]

    for par in range(len(bounds)):
        l = {"type": "ineq", "fun": lambda x: x[par] - bounds[par][0]}
        u = {"type": "ineq", "fun": lambda x: bounds[par][1] - x[par]}
        cons.append(l)
        cons.append(u)

    prev = 0
    total_loss = 0
    for prev, end in cutoff_pts:
        all_data = df_monkey[prev:end]
        true_prob = all_data.next_pacman_dir_fill.fillna(method="ffill").apply(oneHot)
        func = lambda params: negativeLikelihood(
            params, all_data, true_prob, agents, return_trajectory=False, suffix=suffix
        )
        res = scipy.optimize.minimize(
            func,
            x0=params,
            method="SLSQP",
            bounds=bounds,  # exclude bounds and cons because the Q-value has different scales for different agents
            tol=1e-5,
            constraints=cons,
        )
        total_loss += negativeLikelihood(
            res.x / res.x.sum(),
            all_data,
            true_prob,
            agents,
            return_trajectory=False,
            suffix=suffix,
        )
        cr = caculate_correct_rate(res.x, all_data, true_prob, agents, suffix=suffix)
        result_list.append(res.x.tolist() + [cr] + [prev] + [end])
    return result_list, total_loss


def normalize_weights(result_list, df_monkey):
    agents = [
        "global",
        "local",
        "pessimistic_blinky",
        "pessimistic_clyde",
        "suicide",
        "planned_hunting",
    ]
    df_result = (
        pd.DataFrame(
            result_list,
            columns=[i + "_w" for i in agents] + ["accuracy", "start", "end"],
        )
        .set_index("start")
        .reindex(range(df_monkey.shape[0]))
        .fillna(method="ffill")
    )
    df_plot = df_result.filter(regex="_w").divide(
        df_result.filter(regex="_w").sum(1), 0
    )
    return df_plot, df_result


def normalize_weights_slide(result_list, df_monkey):
    agents = [
        "global",
        "local",
        "pessimistic_blinky",
        "pessimistic_clyde",
        "suicide",
        "planned_hunting",
    ]
    window = (result_list[0][-1] - result_list[0][-2]) // 2
    for i in result_list:
        i[-2] = i[-2] + window
    df_result = (
        pd.DataFrame(
            result_list,
            columns=[i + "_w" for i in agents] + ["accuracy", "start", "end"],
        )
        .set_index("start")
        .reindex(range(df_monkey.shape[0]))
    )
    df_plot = df_result.filter(regex="_w").divide(
        df_result.filter(regex="_w").sum(1), 0
    )
    return df_plot, df_result


def normalize_Q(Q_val):
    non_zero_indices = np.where(np.array(Q_val) != 0)
    Q_val[non_zero_indices] = Q_val[non_zero_indices] / np.linalg.norm(Q_val[non_zero_indices])
    return Q_val


def add_cutoff_pts(cutoff_pts, df_monkey):
    eat_ghost = (
        (
            ((df_monkey.ifscared1 == 3) & (df_monkey.ifscared1.diff() < 0))
            | ((df_monkey.ifscared2 == 3) & (df_monkey.ifscared2.diff() < 0))
        )
        .where(lambda x: x == True)
        .dropna()
        .index.tolist()
    )

    eat_energizers = (
        (
            df_monkey.energizers.apply(
                lambda x: len(x) if not isinstance(x, float) else 0
            ).diff()
            < 0
        )
        .where(lambda x: x == True)
        .dropna()
        .index.tolist()
    )

    cutoff_pts = sorted(list(cutoff_pts) + eat_ghost + eat_energizers)
    return cutoff_pts

# =================================================

def rawChangePointFitting():
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading trial data.")
    print("-"*50)
    # select data and get fitted weights (based on turning points)
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        df_monkey = revise_q(
            df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        )
        print("| ({}) {} |Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points
        cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill)
        result_list, _ = fit_func(df_monkey, cutoff_pts)
        plot_weight_accuracy(df_monkey, result_list, trial_name, "raw_change_point")


def AICChangePointFitting():
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        # trial_name = "7-1-Omega-03-Sep-2019-1.csv"
        df_monkey = revise_q(
            df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        )
        print("| ({}) {} |Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points
        cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill)
        result_list, _ = fit_func(df_monkey, cutoff_pts)
        print("-" * 50)
        # ============= select best # of breakpoints ================
        # breakpoints detection
        df_plot, _ = normalize_weights(df_monkey, result_list)
        signal = df_plot.filter(regex="_w").fillna(0).values
        algo = rpt.Dynp(model="l2").fit(signal)
        AIC_list = []
        bkpt_list = list(range(2, 101))
        this_bkpt_list = []
        for index, n_bkpt in enumerate(bkpt_list):
            try:
                result = algo.predict(n_bkpt)
                result_list, total_loss = fit_func(df_monkey, result[:-1])
                print(
                    "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',total_loss + 0.5 * n_bkpt * 5,
                )
                AIC_list.append(total_loss + 0.5 * n_bkpt * 5)
                this_bkpt_list.append(n_bkpt)
            except:
                print("No admissible last breakpoints found.")
                break
        # print("-" * 50)
        if len(AIC_list) == 0:
            continue
        best_arg = np.argmin(AIC_list)
        best_num_of_bkpt = this_bkpt_list[best_arg]
        best_AIC = AIC_list[best_arg]
        best_bkpt_list.append((trial_name, best_num_of_bkpt))
        print("Least AIC value : {}, Best # of breakpoints {}".format(best_AIC, best_num_of_bkpt))
        # =============use best # of breakpoints to get weights and accuracy================
        result = algo.predict(best_num_of_bkpt)
        result_list, total_loss = fit_func(df_monkey, result[:-1])
        plot_weight_accuracy(df_monkey, result_list, trial_name, "AIC_change_point")
        print("="*50)
    # save data
    np.save("../common_data/single_trial/AIC_change_point/best_kpt_list.npy", best_bkpt_list)


def logChangePointFitting():
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        # trial_name = "7-1-Omega-03-Sep-2019-1.csv"
        df_monkey = revise_q(
            df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        )
        print("| ({}) {} |Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points
        cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill)
        result_list, _ = fit_func(df_monkey, cutoff_pts)
        print("-" * 50)
        # ============= select best # of breakpoints ================
        # breakpoints detection
        df_plot, _ = normalize_weights(df_monkey, result_list)
        signal = df_plot.filter(regex="_w").fillna(0).values
        algo = rpt.Dynp(model="l2").fit(signal)
        nll_list = []
        bkpt_list = list(range(2, 101))
        this_bkpt_list = []
        for index, n_bkpt in enumerate(bkpt_list):
            try:
                result = algo.predict(n_bkpt)
                result_list, total_loss = fit_func(df_monkey, result[:-1])
                print(
                    "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',total_loss + 0.5 * n_bkpt * 5,
                )
                nll_list.append(total_loss)
                this_bkpt_list.append(n_bkpt)
            except:
                print("No admissible last breakpoints found.")
                break
        # print("-" * 50)
        if len(nll_list) == 0:
            continue
        best_arg = np.argmin(nll_list)
        best_num_of_bkpt = this_bkpt_list[best_arg]
        best_log = nll_list[best_arg]
        best_bkpt_list.append((trial_name, best_num_of_bkpt))
        print("Least Log Likelihood value : {}, Best # of breakpoints {}".format(best_log, best_num_of_bkpt))
        # =============use best # of breakpoints to get weights and accuracy================
        result = algo.predict(best_num_of_bkpt)
        result_list, total_loss = fit_func(df_monkey, result[:-1])
        plot_weight_accuracy(df_monkey, result_list, trial_name, "nll_change_point")
        print("="*50)
    # save data
    np.save("../common_data/single_trial/nll_change_point/likelihood_best_kpt_list.npy", best_bkpt_list)


def woConstraintChangePointFitting():
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        # trial_name = "7-1-Omega-03-Sep-2019-1.csv"
        # df_monkey = revise_q(
        #     df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        # )
        df_monkey = df[df.file == trial_name].reset_index().drop(columns="level_0")
        print("| ({}) {} |Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points
        cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill)
        result_list, _ = fit_func(df_monkey, cutoff_pts)
        print("-" * 50)
        # ============= select best # of breakpoints ================
        # breakpoints detection
        df_plot, _ = normalize_weights(df_monkey, result_list)
        signal = df_plot.filter(regex="_w").fillna(0).values
        algo = rpt.Dynp(model="l2").fit(signal)
        AIC_list = []
        bkpt_list = list(range(2, 10))
        this_bkpt_list = []
        for index, n_bkpt in enumerate(bkpt_list):
            try:
                result = algo.predict(n_bkpt)
                result_list, total_loss = fit_func(df_monkey, result[:-1])
                print(
                    "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',total_loss + 0.5 * n_bkpt * 5,
                )
                AIC_list.append(total_loss + 0.5 * n_bkpt * 5)
                this_bkpt_list.append(n_bkpt)
            except:
                print("No admissible last breakpoints found.")
                break
        # print("-" * 50)
        if len(AIC_list) == 0:
            continue
        best_arg = np.argmin(AIC_list)
        best_num_of_bkpt = this_bkpt_list[best_arg]
        best_AIC = AIC_list[best_arg]
        best_bkpt_list.append((trial_name, best_num_of_bkpt))
        print("Least AIC value : {}, Best # of breakpoints {}".format(best_AIC, best_num_of_bkpt))
        # =============use best # of breakpoints to get weights and accuracy================
        result = algo.predict(best_num_of_bkpt)
        result_list, total_loss = fit_func(df_monkey, result[:-1])
        plot_weight_accuracy(df_monkey, result_list, trial_name, "AIC_change_point")
        print("="*50)
    # save data
    np.save("../common_data/single_trial/AIC_change_point/best_kpt_list.npy", best_bkpt_list)


def singlTrielFitting(need_constraint):
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    trial_name = "12-1-Omega-05-Jul-2019-1.csv"
    trial_name = "3-1-Omega-24-Jun-2019-1.csv"
    # trial_name = "1-8-Omega-11-Jun-2019-1.csv"

    if need_constraint:
        df_monkey = revise_q(
            df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        )
    else:
        df_monkey = df[df.file == trial_name].reset_index().drop(columns="level_0")
    for a in ["global_Q", "local_Q", "pessimistic_blinky_Q", "pessimistic_clyde_Q", "suicide_Q", "planned_hunting_Q"]:
        df_monkey[a] = df_monkey[a].apply(lambda x: normalize_Q(x))
    print("| {} |Data shape {}".format(trial_name, df_monkey.shape))
    ## fit based on turning points
    cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill)
    result_list, _ = fit_func(df_monkey, cutoff_pts)
    print("-" * 50)
    # ============= select best # of breakpoints ================
    # breakpoints detection
    df_plot, _ = normalize_weights(df_monkey, result_list)
    signal = df_plot.filter(regex="_w").fillna(0).values
    algo = rpt.Dynp(model="l2").fit(signal)
    nll_list = []
    bkpt_list = list(range(2, 50))
    this_bkpt_list = []
    for index, n_bkpt in enumerate(bkpt_list):
        try:
            result = algo.predict(n_bkpt)
            result_list, total_loss = fit_func(df_monkey, result[:-1])
            print(
                "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',
                                                                                total_loss + 0.5 * n_bkpt * 5,
            )
            # nll_list.append(total_loss)
            nll_list.append(total_loss)
            this_bkpt_list.append(n_bkpt)

        except:
            print("No admissible last breakpoints found.")
            break
    best_arg = np.argmin(nll_list)
    best_num_of_bkpt = this_bkpt_list[best_arg]
    best_log = nll_list[best_arg]
    best_bkpt_list.append((trial_name, best_num_of_bkpt))
    print("Least Log Likelihood value : {}, Best # of breakpoints {}".format(best_log, best_num_of_bkpt))
    # =============use best # of breakpoints to get weights and accuracy================
    result = algo.predict(best_num_of_bkpt)
    result_list, total_loss = fit_func(df_monkey, result[:-1])
    plot_weight_accuracy(df_monkey, result_list, trial_name, "single_test")
    print("=" * 50)

    rpt.display(signal, result, result)
    plt.show()


def qNormalizePointFitting():
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-uniform_path10.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        # trial_name = "7-1-Omega-03-Sep-2019-1.csv"
        df_monkey = revise_q(df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df)
        for a in ["global_Q", "local_Q", "pessimistic_blinky_Q", "pessimistic_clyde_Q", "suicide_Q", "planned_hunting_Q"]:
            df_monkey[a] = df_monkey[a].apply(lambda x: normalize_Q(x))
        print("| ({}) {} |Data shape {}".format(t, trial_name, df_monkey.shape))
        ## fit based on turning points
        cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill)
        result_list, _ = fit_func(df_monkey, cutoff_pts)
        print("-" * 50)
        # ============= select best # of breakpoints ================
        # breakpoints detection
        df_plot, _ = normalize_weights(df_monkey, result_list)
        signal = df_plot.filter(regex="_w").fillna(0).values
        algo = rpt.Dynp(model="l2").fit(signal)
        nll_list = []
        bkpt_list = list(range(2, 101))
        this_bkpt_list = []
        for index, n_bkpt in enumerate(bkpt_list):
            try:
                result = algo.predict(n_bkpt)
                result_list, total_loss = fit_func(df_monkey, result[:-1])
                print(
                    "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',total_loss + 0.5 * n_bkpt * 5,
                )
                nll_list.append(total_loss)
                this_bkpt_list.append(n_bkpt)
            except:
                print("No admissible last breakpoints found.")
                break
        # print("-" * 50)
        if len(nll_list) == 0:
            continue
        best_arg = np.argmin(nll_list)
        best_num_of_bkpt = this_bkpt_list[best_arg]
        best_log = nll_list[best_arg]
        best_bkpt_list.append((trial_name, best_num_of_bkpt))
        print("Least Log Likelihood value : {}, Best # of breakpoints {}".format(best_log, best_num_of_bkpt))
        # =============use best # of breakpoints to get weights and accuracy================
        result = algo.predict(best_num_of_bkpt)
        result_list, total_loss = fit_func(df_monkey, result[:-1])
        plot_weight_accuracy(df_monkey, result_list, trial_name, "normalize_q_nll")
        print("="*50)
    # save data
    np.save("../common_data/single_trial/normalize_q_nll/likelihood_best_kpt_list.npy", best_bkpt_list)


def cleanChangePointFitting():
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-clean.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    for c in df.filter(regex="_Q").columns:
        if "pess" not in c:
            df[c + "_norm"] = df[c].apply(
                lambda x: x / max(x) if set(x) != {0} else [0, 0, 0, 0]
            )
        else:
            offset_num = df[c].explode().min()
            df[c + "_norm"] = df[c].apply(
                lambda x: (x + offset_num) / max(x + offset_num)
                if set(x) != {0}
                else [0, 0, 0, 0]
            )
    suffix = "_Q_norm"
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        # trial_name = "7-1-Omega-03-Sep-2019-1.csv"
        df_monkey = revise_q(
            df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        )
        # df_monkey = df[df.file == trial_name].reset_index().drop(columns="level_0")
        print("| ({}) {} | Data shape {}".format(t, trial_name, df_monkey.shape))

        ## fit based on turning points
        # cutoff_pts = change_dir_index(df_monkey.next_pacman_dir_fill) # No add eating ghost and eating energizer points
        cutoff_pts = add_cutoff_pts(change_dir_index(df_monkey.next_pacman_dir_fill), df_monkey) # Add eating ghost and eating energizer points
        cutoff_pts = list(zip([0] + list(cutoff_pts[:-1]), cutoff_pts))
        result_list, _ = fit_func(df_monkey, cutoff_pts, suffix=suffix)
        print("-" * 50)
        # ============= select best # of breakpoints ================
        # breakpoints detection
        try:
            df_plot, df_result = normalize_weights(result_list, df_monkey)
        except:
            print("Error occurs in weight normalization.")
            continue
        signal = df_plot.filter(regex="_w").fillna(0).values
        algo = rpt.Dynp(model="l2", jump=2).fit(signal)
        nll_list = []
        bkpt_list = list(range(2, 21))
        this_bkpt_list = []
        for index, n_bkpt in enumerate(bkpt_list):
            try:
                result = algo.predict(n_bkpt)
                result = list(zip([0] + result[:-1], result))
                result_list, total_loss = fit_func(df_monkey, result, suffix=suffix)
                print(
                    "| {} |".format(n_bkpt), 'total loss:', total_loss, 'penalty:', 0.5 * n_bkpt * 5, 'AIC:',total_loss + 0.5 * n_bkpt * 5,
                )
                nll_list.append(total_loss)
                this_bkpt_list.append(n_bkpt)
            except:
                print("No admissible last breakpoints found.")
                break
        # print("-" * 50)
        if len(nll_list) == 0:
            continue
        best_arg = np.argmin(nll_list)
        best_num_of_bkpt = this_bkpt_list[best_arg]
        best_log = nll_list[best_arg]
        best_bkpt_list.append((trial_name, best_num_of_bkpt))
        print("Least Log Likelihood value : {}, Best # of breakpoints {}".format(best_log, best_num_of_bkpt))
        # =============use best # of breakpoints to get weights and accuracy================
        result = algo.predict(best_num_of_bkpt)
        result = list(zip([0] + result[:-1], result))
        result_list, total_loss = fit_func(df_monkey, result, suffix=suffix)
        plot_weight_accuracy(df_monkey, result_list, trial_name, "clean_agents_change")
        print("="*50)
    # save data
    np.save("../common_data/single_trial/clean_agents_change/likelihood_best_kpt_list.npy", best_bkpt_list)


def cleanSlideWindowFitting(window):
    print("="*0)
    print("Window Size = {}".format(window))
    print("=" * 0)
    print("Start reading data...")
    df = pd.read_pickle(
        "../common_data/trial/100_trial_data_Omega-with_Q-clean.pkl"
    )
    locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
    for c in df.filter(regex="_Q").columns:
        if "pess" not in c:
            df[c + "_norm"] = df[c].apply(
                lambda x: x / max(x) if set(x) != {0} else [0, 0, 0, 0]
            )
        else:
            offset_num = df[c].explode().min()
            df[c + "_norm"] = df[c].apply(
                lambda x: (x + offset_num) / max(x + offset_num)
                if set(x) != {0}
                else [0, 0, 0, 0]
            )
    suffix = "_Q_norm"
    print("Finished reading trial data.")
    trial_name_list = np.unique(df.file.values)
    print("The num of trials : ", len(trial_name_list))
    print("-" * 50)
    best_bkpt_list = []
    for t, trial_name in enumerate(trial_name_list):
        # trial_name = "7-1-Omega-03-Sep-2019-1.csv"
        df_monkey = revise_q(
            df[df.file == trial_name].reset_index().drop(columns="level_0"), locs_df
        )
        # df_monkey = df[df.file == trial_name].reset_index().drop(columns="level_0")
        print("| ({}) {} | Data shape {}".format(t, trial_name, df_monkey.shape))

        window = window
        cutoff_pts = list(
            zip(
                list(range(df_monkey.shape[0] - window + 1)),
                list(range(window, df_monkey.shape[0] + 1)),
            )
        )
        result_list, _ = fit_func(df_monkey, cutoff_pts, suffix=suffix)
        plot_weight_accuracy(df_monkey, result_list, trial_name, "clean_agents_slide_window{}".format(window))
        print("="*50)
    # save data
    np.save("../common_data/single_trial/clean_agents_slide_window{}/likelihood_best_kpt_list.npy".format(window), best_bkpt_list)

# =================================================

def plot_weight_accuracy(df_monkey, result_list, trial_name, base):
    agent_color = {
        "local": "#D7181C",
        "pessimistic_blinky": "#FDAE61",
        "pessimistic_clyde": "#c78444",
        "global": "#44B53C",
        "suicide": "#836BB7",
        "planned_hunting": "#81B3FF",
        "vague": "black",
    }
    agent_name = {
        "global":"global",
        "local":"local",
        "pessimistic_blinky":"evade(Blinky)",
        "pessimistic_clyde":"evade(Clyde)",
        "suicide":"approach",
        "planned_hunting":"energizer",
    }

    df_plot, df_result = normalize_weights(result_list, df_monkey)

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    spec = fig.add_gridspec(4, 1)

    ax1 = fig.add_subplot(spec[:2,:])
    for c in df_result.filter(regex="_w").columns:
        plt.plot(df_plot[c], color=agent_color[c[:-2]], ms=3, lw=5,label=agent_name[c[:-2]])
    plt.title(trial_name, fontsize = 13)
    plt.ylabel("Normalized Strategy Weight", fontsize=20)
    plt.xlim(0, df_plot.shape[0] - 1)
    x_ticks_index = np.arange(0, len(df_plot), 10)
    x_ticks = [1+int(each) for each in x_ticks_index]
    plt.xticks(x_ticks_index, x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(-0.01, 1.02)
    plt.legend(loc="upper center", fontsize=20, ncol = len(agent_name), frameon = False, bbox_to_anchor = (0.5, 1.2))

    contributions = df_result[df_result.filter(regex="_w").columns]
    estimated_label = [
        _estimationVagueLabeling(contributions.iloc[i].values/np.linalg.norm(contributions.iloc[i].values), agents)
        for i in range(contributions.shape[0])
    ]
    ax2 = fig.add_subplot(spec[2, :])
    for i in range(len(estimated_label)):
        seq = np.linspace(-0.02, 0.0, len(estimated_label[i]) + 1)
        for j, h in enumerate(estimated_label[i]):
            plt.fill_between(x=[i, i + 1], y1=seq[j + 1], y2=seq[j], color=agent_color[h])
    plt.xlim(0, len(estimated_label))
    plt.axis('off')

    ax3 = fig.add_subplot(spec[3,:])
    plt.title("Cr : {b:.3f}".format(b=np.nanmean(df_result.accuracy) if not np.all(np.isnan(df_result.accuracy)) else 0.0),
              fontsize=13)
    plt.plot(np.arange(len(df_result)), df_result.accuracy, "bo-", lw=4, ms=10)
    plt.ylabel("Correct Rate", fontsize=20)
    plt.xlim(0, df_result.shape[0] - 1)
    x_ticks_index = np.arange(0, len(df_result), 10)
    x_ticks = [1+int(each) for each in x_ticks_index]
    plt.xticks(x_ticks_index, x_ticks, fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylim(0.5, 1.05)

    plt.savefig("../common_data/single_trial/{}/{}.pdf".format(base, trial_name))
    # plt.show()



if __name__ == '__main__':
    # rawChangePointFitting()
    # AICChangePointFitting()
    # logChangePointFitting()

    # qNormalizePointFitting()

    # cleanChangePointFitting()
    # cleanSlideWindowFitting(window = 11)
    cleanSlideWindowFitting(window=7)

    # singlTrielFitting(need_constraint=True)

    # woConstraintChangePointFitting()