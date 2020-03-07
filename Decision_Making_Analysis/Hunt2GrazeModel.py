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

    def _move(self,cur_loc, cur_dir):
        '''
        Compute the location after taking an action.
        :param cur_loc: Current location.
        :param dir: Moving direction ().
        :return: 
        '''
        next_loc = cur_loc
        # 0 for up, 1 for down, 2 for left, 3 for  right, 4 for stay
        cur_dir = cur_dir.index(1)
        if 0 == cur_dir:
            next_loc[1] = next_loc[1] -1 if next_loc[1] - 1 >= 0 else next_loc[1]
        elif 1 == cur_dir:
            next_loc[1] = next_loc[1] + 1 #TODO: out of boundary? How to determine?
        elif 2 == cur_dir:
            next_loc[0] = next_loc[0] - 1 if next_loc[0] - 1 >= 0 else next_loc[0]
        elif 3 == cur_dir:
            next_loc[0] = next_loc[0] + 1 #TODO: out of boundary? How to determine?
        elif 4 == cur_dir:
            next_loc = next_loc
        else:
            raise ValueError('Unknown direction {}!'.format(cur_dir))
        return next_loc

    def _estimateGhostLocation(self, ghosts_loc, move_dir):
        #TODO: estimate the location? estimat the path? probablistic model?
        '''
        Estimate the location two ghosts.
        :param ghosts_loc: The current location of two ghosts, with shape of (2,2).
        :param move_dir: Moving direction of two ghosts. The direction should be selected from up/down/left/right.
        :return: The locations of two ghosts.
        '''
        return [
            self._move(ghosts_loc[0], move_dir[0]),
            self._move(ghosts_loc[1], move_dir[1])
        ]


    def _computePursuingTime(self, pacman_loc, ghosts_path):
        #TODO: change the computing algorithm
        # The pursuing time is naively computed by the L1 norm of distance between Pacman and ghosts
        distance = np.array(ghosts_path) - np.array(pacman_loc)
        pursuing_time = np.linalg.norm(distance, ord = 1, axis = 1)
        min_time = np.min(pursuing_time)
        return (min_time, np.where(pursuing_time == min_time)[0][0])

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
            ghosts_loc = each[0:2]
            move_dir = each[5:7]
            pacman_loc = each[2]
            remained_time = each[3:5]
            future_loc = self._estimateGhostLocation(ghosts_loc, move_dir)
            pursing_time, catch_step = self._computePursuingTime(pacman_loc, future_loc)
            if np.all(pursing_time > remained_time):
                cur_label = [0, 1]  # Switch to grazing mode
            else:
                cur_label = [1, 0]  # Remained hunting mode
            pred_label.append(cur_label)
        return np.array(pred_label, dtype = np.int)