''''
Description:
    Model for analyzing the transfer probability from hunting model to the grazing model.

Author:
    Jiaqi Zhang <zjqseu@gmail.com >

Date:
    2020/3/5
'''

import torch
import torch.nn as nn
import numpy as np
import copy

from util import np2tensor, tensor2np
from evaluation import correctRate


class Hunt2Graze:

    def __init__(self):
        pass

    def _estimateGhostLocation(self, ghosts_loc, move_dir):
        '''
        Estimate the moving path of two ghosts.
        :param ghosts_loc: The current location of two ghosts, with shape of (2,2).
        :param move_dir: Moving direction of two ghosts. The direction should be selected from up/down/left/right.
        :return: The moving path of two ghosts.
        '''
        pass

    def _computePursuingTime(self, pacman_loc, ghosts_path, remained_time):
        pass

    def train(self, training_data,training_label ):
        # TODO: train with MLP (input are computed features, output is the label prob)
        pass


    def testing(self, testing_data, testing_label):
        pass

    def deterministicModel(self, data, label):
        '''
        The deterministic model for analyzing the transferation.
        :param data: All the data features, with shape of (number of samples, number of features).
        :param label: All the labels, with shape of (number of samples,)
        :return: Correct rate and predictions.
            - correct_rate (float): The correct rate of estimations.
            - pred_label: Prediction of labels.
        '''
        pred_label = []
        for index, each in enumerate(data):
            # TODO: features
            ghosts_loc = each[0:2]
            move_dir = each[5:7]
            pacman_loc = each[2]
            remained_time = each[3:5]
            ghosts_path = self._estimateGhostLocation(ghosts_loc, move_dir)
            pursing_time = self._computePursuingTime(pacman_loc, ghosts_path, remained_time)
            if pursing_time > remained_time:
                cur_label = [0, 1]  # Switch to grazing mode
            else:
                cur_label = [1, 0]  # Remained hunting mode
            pred_label.append(cur_label)
        correct_rate = correctRate(pred_label, label)
        return (correct_rate, pred_label)