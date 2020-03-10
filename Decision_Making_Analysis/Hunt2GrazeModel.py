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
import pandas as pd

from util import np2tensor, tensor2np, maptodict, tuple_list
from evaluation import correctRate


class Hunt2Graze:

    def __init__(self):
        self.locs_df = pd.read_csv("data/dij_distance_map.csv")
        self.locs_df.pos1, self.locs_df.pos2 = (
            self.locs_df.pos1.apply(eval),
            self.locs_df.pos2.apply(eval),
        )
        self.map_distance = {}
        # Construct the dict for distance between two points on the map
        for each in self.locs_df.values:
            start = each[0]
            end = each[1]
            distance = each[2]
            if start not in self.map_distance:
                self.map_distance[start] = {}
            if end not in self.map_distance:
                self.map_distance[end] = {}
            self.map_distance[start][start] = 0
            self.map_distance[end][end] = 0
            self.map_distance[start][end] = distance
            self.map_distance[end][start] = distance

    # def _move(self,cur_loc, cur_dir):
    #     '''
    #     Compute the location after taking an action.
    #     :param cur_loc: Current location.
    #     :param dir: Moving direction ().
    #     :return:
    #     '''
    #     next_loc = cur_loc
    #     # 0 for up, 1 for down, 2 for left, 3 for  right, 4 for stay
    #     cur_dir = cur_dir.index(1)
    #     if 0 == cur_dir:
    #         next_loc[1] = next_loc[1] -1 if next_loc[1] - 1 >= 0 else next_loc[1]
    #     elif 1 == cur_dir:
    #         next_loc[1] = next_loc[1] + 1 #TODO: out of boundary? How to determine?
    #     elif 2 == cur_dir:
    #         next_loc[0] = next_loc[0] - 1 if next_loc[0] - 1 >= 0 else next_loc[0]
    #     elif 3 == cur_dir:
    #         next_loc[0] = next_loc[0] + 1 #TODO: out of boundary? How to determine?
    #     elif 4 == cur_dir:
    #         next_loc = next_loc
    #     else:
    #         raise ValueError('Unknown direction {}!'.format(cur_dir))
    #     return next_loc

    def _estimateGhostLocation(self, ghosts_loc, move_dir, remained_time, pacman_loc):
        '''
        Estimate the location two ghosts.
        :param ghosts_loc: The current location of two ghosts, with shape of (2,2).
        :param move_dir: Moving direction of two ghosts. The direction should be selected from up/down/left/right.
        :param remained_time: Remained scaringtime for two ghosts.
        :param pacman_loc: The current location of Pacman.
        :return: The locations of two ghosts.
        '''
        # TODO: Combine this with future_position
        return [
            self._future_position(ghosts_loc[0], move_dir[0], remained_time[0], pacman_loc),
            self._future_position(ghosts_loc[1], move_dir[1], remained_time[1], pacman_loc),
        ]

    def _future_position(self, ghost_pos, ghost_dir, t, pacman_pos):
        # pacman_pos = pd.DataFrame(pacman_pos) # for comparing with dataframe
        if t == 0:
            return ghost_pos
        history = [ghost_pos]
        for i in range(int(t // 2)):
            d_dict = {}
            for key, val in maptodict(ghost_pos).items():
                val = list(val)
                if val not in history:
                    d_dict[key] = val
            if i == 0 and ghost_dir in d_dict.keys():
                ghost_pos = d_dict[ghost_dir]
            else:
                dict_df = pd.DataFrame.from_dict(d_dict, orient="index")
                #             if dict_df.empty:
                #                 set_trace()
                #                 print ('empty')
                dict_df["poss_pos"] = tuple_list(dict_df[[0, 1]].values)
                try: # TODO: not enough memory space
                    ghost_dir, ghost_pos = (
                        self.locs_df[(self.locs_df.pos1 == tuple(pacman_pos))]
                            .merge(dict_df.reset_index(), left_on="pos2", right_on="poss_pos")
                            .sort_values(by="dis")[["index", "poss_pos"]]
                            .values[-1]
                    )
                except:
                    return pacman_pos
            history.append(ghost_pos)
        return ghost_pos

    def _computePursuingTime(self, pacman_loc, ghosts_path):
        pacman_loc = tuple(pacman_loc)
        ghosts_path = [
            tuple(ghosts_path[0]),
            tuple(ghosts_path[1])
        ]
        # # TODO: some location is not in the map
        # try:
        #     distance_1 = self.map_distance[pacman_loc][ghosts_path[0]]
        # except:
        #     print('Pacman Loc {} and Ghost 1 Loc {}'.format(pacman_loc, ghosts_path[0]))
        #     distance_1 = 0
        # try:
        #     distance_2 = self.map_distance[pacman_loc][ghosts_path[1]]
        # except:
        #     print('Pacman Loc {} and Ghost 2 Loc {}'.format(pacman_loc, ghosts_path[1]))
        #     distance_2 = 0
        distance = [
            self.map_distance[pacman_loc][ghosts_path[0]],
            self.map_distance[pacman_loc][ghosts_path[1]]
            # distance_1,
            # distance_2
        ]
        pursuing_time = distance # assume Pacman moves one unit in one time step
        min_time = np.min(pursuing_time)
        return (min_time, np.where(pursuing_time == min_time)[0][0])

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
            if index % 500 == 0 and index is not 0:
                print('Finished {} samples...'.format(index))
                correct_rate = correctRate(pred_label, label[:index])
                print('Classification correct rate {}'.format(correct_rate))
            ghosts_loc = each[0:2]
            move_dir = each[5:7]
            pacman_loc = each[2]
            remained_time = each[3:5]
            future_loc = self._estimateGhostLocation(ghosts_loc, move_dir, remained_time, pacman_loc)
            future_loc = np.array([np.array(each).squeeze() for each in future_loc])
            pursing_time, catch_step = self._computePursuingTime(pacman_loc, future_loc)
            if np.all(pursing_time > remained_time):
                cur_label = [0, 1]  # Switch to grazing mode
            else:
                cur_label = [1, 0]  # Remained hunting mode
            pred_label.append(cur_label)
        return np.array(pred_label, dtype = np.int)


    def train(self, training_data,training_label ):
        # TODO: train with MLP (input are computed features, output is the label prob)
        pass


    def testing(self, testing_data, testing_label):
        pass

