'''
Description:
    MLE parameter estimation for multi-agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    July 1 2020
'''

import pandas as pd
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import h5py
from scipy.io import loadmat
import scipy.optimize


def negativeLogLikelihood(param, choices, rewards):
    '''
    Compute the negative log likelihood of data.
    :param param: Parameters (alpha, beta, gamma, omega)
    :param choices: Choices of validation.
    :param rewards: Rewards of validation.
    :return: Negative log  likelihood.
    '''
    # TODO: check correctness; log 0 smoothing
    alpha = param[0]
    beta = param[1]
    gamma = param[2]
    omega = param[3]
    choices_num = 3  # TODO: for three-armed-bandit task
    trials_num = len(rewards)

    reward_value = np.zeros((choices_num,))
    choice_value = np.zeros((choices_num,))
    overall_value = np.zeros((choices_num,))
    value_trajectory = []
    prob_trajectory = []

    nll = 0  # negative log likelihood
    # param_val = param.valuesdict()
    for trial in range(1, trials_num):
        choice = int(choices[trial] - 1)
        reward = int(rewards[trial])
        # reward-dependent value Q_{t+1}(x) = (1-alpha) * Q_{t}(x) + alpha * Rew_t
        reward_value[choice] = (1 - alpha) * reward_value[choice] + alpha * reward
        # choice-dependent value C_{t+1}(x) = (1-beta) * C_{t}(x) + beta * Cho_t
        choice_value[choice] = (1 - beta) * choice_value[choice] + beta * choice
        # overall value V_{t}(x) = gamma * Q_{t}(x) + (1-gamma) * C_{t}(x)
        overall_value[choice] = gamma * reward_value[choice] + (1 - gamma) * choice_value[choice]
        value_trajectory.append(overall_value)
        # negative log likelihood
        weighted_overall_value = omega * overall_value
        exp_overall_value = np.sum(np.exp(weighted_overall_value))
        log_prob = weighted_overall_value[choice] - np.sum(exp_overall_value)
        nll += (- log_prob)
        prob_trajectory.append(exp_overall_value / np.sum(exp_overall_value))
    return nll