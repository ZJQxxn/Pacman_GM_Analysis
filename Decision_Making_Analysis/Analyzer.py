'''
Description:
    Analyze the transitive probability between hunting and grazing mode.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/6
'''
import torch
import torch.nn as nn
import numpy as np
import copy
import csv
import scipy.io as sio

from Hunt2GrazeModel import Hunt2Graze
from Graze2HuntModel import Graze2Hunt
from evaluation import binaryClassError
from util import tensor2np


class Analyzer:

    def __init__(self, feature_file, label_file, mode_file):
        '''
        Initialization.
        :param feature_file: Filename for features. 
        :param label_file: Filename for labels.
        :param mode_file: Filename for modes.
        '''
        self.data = []
        self.labels = []
        self.modes = []
        # Read features
        with open(feature_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.data.append(row)
        # Read labels
        with open(label_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.labels.append(row)
        # Read modes
        with open(mode_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.modes.append(row)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels) 
        self.modes = np.array(self.modes)
        # Split into hunting mode and grazing mode based on modes
        self.hunting_index = []
        self.grazing_index = []
        for index in range(len(self.data)-1):
            data = self.data[index]
            if 'hunting' == self.modes[index][2]:
                self.hunting_index.append(index)
            elif 'grazing' == self.modes[index][2]:
                self.grazing_index.append(index)
            else:
                raise ValueError('Undefined Pacman mode! Check you modes file.')
        self.hunting_data = self.data[self.hunting_index, :]
        self.grazing_data = self.data[self.grazing_index, :]
        # TODO: Only use first two of labels, because the thrid is a string
        self.hunting_label = np.array(self.labels[self.hunting_index, :2], dtype = np.float)
        self.grazing_label = np.array(self.labels[self.grazing_index, :2], dtype = np.float)


    def _G2HPreprocess(self):
        '''
        Preprocessing data for analyzing grazing to hunting.
        Processed data have following features:
            - distance between Pacman and ghosts (data[4], data[5])
            - distance between Pacman and the closest dot (data[6])
            - distance between Pacman and the closest  big dots, computed with Pacman (data[10]) and dot position (data[7]).
        :return: Preprocessed data matrix.
            - g2h_training_data: Features in training set for graze2hunt analyzing.
            - g2h_training_label: Labels in training set for graze2hunt analyzing.
            - g2h_testing_data: Features in testing set for graze2hunt analyzing
            - g2h_testing_label: Labels in testing set for graze2hunt analyzing
        '''
        # Extract useful features
        preprocessed_data = []
        for index, each in enumerate(self.grazing_data):
            temp = np.zeros((4,))
            temp[0] = each[4]
            temp[1] = each[5]
            temp[2] = each[6]
            temp[3] = self._computeDotDis(each[10], each[7])
            # TODO: For now, we only use 3 features and excluding distance between Pacman and big dots.
            preprocessed_data.append(temp[:3])
        preprocessed_data = np.array(preprocessed_data)
        # Normalization
        for col_index in range(preprocessed_data.shape[1]):
            preprocessed_data[:, col_index] = (preprocessed_data[:, col_index] - np.nanmean(
                preprocessed_data[:, col_index])) / np.nanstd(preprocessed_data[:, col_index])
        # Split into training and testing sets(60% for training and 40% for testing) for each mode
        training_ratio = 0.6
        sample_num = preprocessed_data.shape[0] - 1
        training_num = int(training_ratio * sample_num)
        shuffled_index = np.arange(0, sample_num)
        np.random.shuffle(shuffled_index)
        training_index = shuffled_index[:training_num]
        testing_index = shuffled_index[training_num:]
        g2h_training_data = preprocessed_data[training_index, :]
        g2h_training_label = self.grazing_label[training_index, :]
        g2h_testing_data = preprocessed_data[testing_index, :]
        g2h_testing_label = self.grazing_label[testing_index, :]
        return (g2h_training_data, g2h_training_label, g2h_testing_data, g2h_testing_label)


    def _H2GPreprocess(self):
        #TODO:not finished
        '''
        Preprocessing data for analyzing hunting to grazing.
        Processed data have following features:
            
        :return: Preprocessed data matrix.
            - h2g_training_data: Features in training set for hunt2graze analyzing.
            - h2g_training_label: Labels in training set for hunt2graze analyzing.
            - h2g_testing_data: Features in testing set for hunt2graze analyzing
            - h2g_testing_label: Labels in testing set for hunt2graze analyzing
        '''
        # Extract useful features
        preprocessed_data = []
        for index, each in enumerate(self.hunting_data):
            # TODO: specify features
            temp = np.zeros((4,))
            temp[0] = each[4]
            temp[1] = each[5]
            temp[2] = each[6]
            temp[3] = self._computeDotDis(each[10], each[7])
            preprocessed_data.append(temp)
        preprocessed_data = np.array(preprocessed_data)
        # Normalization
        for col_index in range(preprocessed_data.shape[1]):
            preprocessed_data[:, col_index] = (preprocessed_data[:, col_index] - np.nanmean(preprocessed_data[:, col_index])) / np.nanstd(preprocessed_data[:, col_index])
        # Split into training and testing sets(60% for training and 40% for testing) for each mode
        training_ratio = 0.6
        sample_num = preprocessed_data.shape[0] - 1
        training_num = int(training_ratio * sample_num)
        shuffled__index = np.random.shuffle(np.arange(0, sample_num))
        training_index = shuffled__index[:training_num]
        testing_index = shuffled__index[training_num:]
        h2g_training_data = preprocessed_data[training_index, :]
        h2g_training_label = self.hunting_label[training_index, :]
        h2g_testing_data = preprocessed_data[testing_index, :]
        h2g_testing_label = self.hunting_label[testing_index, :]
        return (h2g_training_data, h2g_training_label, h2g_testing_data, h2g_testing_label)


    def _computeDotDis(self, pacman_pos, bigdot_pos):
        if '' != bigdot_pos:
            pacman_pos = [float(each) for each in pacman_pos.strip("()").split(',')]
            bigdot_pos = np.array(
                [float(each)
                for each in bigdot_pos.strip('[]').replace('(','').replace(')','').split(',')]
            ).reshape((-1,2))
            distance  = np.linalg.norm(bigdot_pos - pacman_pos, ord=1, axis = 1)
            return np.min(distance)
        else:
            #TODO: return what if there is no big dots.
            return np.nan


    def G2HAnalyze(self, in_dim, batch_size = 5, lr = 1e-2, enable_cuda = False):
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2HPreprocess()
        print('...Finished preprocessing!\n')
        model = Graze2Hunt(in_dim, batch_size, lr, enable_cuda)
        # Loss before training
        pred_label = model.test(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        print('Loss before training {}'.format(estimation_loss))
        # Train the model
        print('=' * 30)
        print('Train model for analyzing Graze2Hunt:\n', model.network)
        batch_loss = model.train(training_data, training_label)
        sio.savemat('train_batch_los.mat',
                    {'batch_loss': np.array([tensor2np(each) for each in batch_loss]), 'batch_size': batch_size})
        model.saveModel('save_m/G2H_model.pt')
        # model = Graze2Hunt(in_dim, batch_size, lr, enable_cuda)
        # model.loadModel('save_m/G2H_model.pt')
        # Testing the model after training
        print('\n', '=' * 30)
        print('Testing...')
        pred_label = model.test(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        print('Loss after training {}'.format(estimation_loss))


    def H2GAnalyze(self):
        #TODO: not finished
        training_data, training_label, testing_data, testing_label = self._H2GPreprocess()




if __name__ == '__main__':
    torch.set_num_threads(3)
    torch.set_num_interop_threads(3)

    feature_filename = 'data/extract_feature.csv'
    label_filename = 'data/all_labels.csv'
    mode_filename = 'data/all_modes.csv'
    a = Analyzer(feature_filename, label_filename, mode_filename)
    # print(a.data[0])
    a.G2HAnalyze(3, batch_size = 1) # With MLP: f(D)