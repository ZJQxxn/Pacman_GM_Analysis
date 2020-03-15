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
import pandas as pd
import copy
import csv
import scipy.io as sio
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree.export import export_text, export_graphviz
import matplotlib.pyplot as plt
import graphviz
import random

from Hunt2GrazeModel import Hunt2Graze
from Graze2HuntModel import Graze2Hunt
from evaluation import binaryClassError, correctRate, AUC
from util import tensor2np, oneHot
from util import estimateGhostLocation, future_position, computeLocDis



class Analyzer:
    '''
    Description:
    
    Variables:
    
    Functions:
    '''

    def __init__(self, feature_file, label_file, mode_file):
        '''
        Initialization.
        :param feature_file: Filename for features. 
        :param label_file: Filename for labels.
        :param mode_file: Filename for modes.
        '''
        random.seed(a = None)
        self.data = []
        self.labels = []
        self.modes = []
        # Map info
        self.locs_df = pd.read_csv("data/dij_distance_map.csv")
        self.locs_df.pos1, self.locs_df.pos2 = (
            self.locs_df.pos1.apply(eval),
            self.locs_df.pos2.apply(eval),
        )
        self.map_distance = {}
        # Construct the dict of distances between evewry two points on the map
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
            # data = self.data[index]the last
            # Discard the data of escaping mode, with escaping label, and at the last time step of a trial
            if 'escaping' != self.modes[index][2] \
                    and 'escaping' != self.modes[index + 1][2] \
                    and self.data[index + 1][18] != "0":
                if 'hunting' == self.modes[index][2]:
                    self.hunting_index.append(index)
                elif 'grazing' == self.modes[index][2]:
                    self.grazing_index.append(index)
                else:
                    raise ValueError('Undefined Pacman mode! Check you modes file.')
        self.hunting_data = self.data[self.hunting_index, :]
        self.grazing_data = self.data[self.grazing_index, :]
        # Only use first two of labels, because the third is a string
        self.hunting_label = np.array(self.labels[self.hunting_index, :2], dtype = np.float)
        self.grazing_label = np.array(self.labels[self.grazing_index, :2], dtype = np.float)

    # ======================================
    #           PREPROCESSING
    # ======================================
    def _G2HPreprocess(self, need_stan = True):
        '''
        Preprocessing data for analyzing grazing to hunting.
        Processed data have following features:
            - distance between Pacman and ghosts (data[4], data[5])
            - distance between Pacman and the closest dot (data[6])
            - distance between Pacman and the closest big dots, computed with Pacman (data[10]) and dot position (data[7]).
        :var: Input variables
            - need_stan: bool. Determine whether need to do standardization.
        :return: Preprocessed data matrix.
            - g2h_training_data: Features in training set for graze2hunt analyzing.
            - g2h_training_label: Labels in training set for graze2hunt analyzing.
            - g2h_testing_data: Features in testing set for graze2hunt analyzing
            - g2h_testing_label: Labels in testing set for graze2hunt analyzing
        '''
        # Extract useful features
        preprocessed_data = []
        for index, each in enumerate(self.grazing_data):
            pacman_loc = [int(i) for i in each[10].strip('()').split(',')]
            ghost_loc = each[7]
            if ghost_loc != '':
                ghost_loc = ghost_loc.replace('[(', '').replace('(', '').replace(')]', '').replace(')', '').split(',')
                ghost_loc = [[int(float(ghost_loc[index])), int(float(ghost_loc[index + 1]))] for index in range(len(ghost_loc) - 1)]
            else:
                ghost_loc = []
            temp = [
                float(each[4]), # Distance betweeen Pacman and ghost 1
                float(each[5]), # Distance between Pacman and ghost 2
                float(each[6]) # Distance between Pacman and the closest dot
                # np.min(computeLocDis(self.map_distance, pacman_loc, ghost_loc))  # Distance between Pacman and the closest big dot
            ]
            preprocessed_data.append(temp)
        preprocessed_data = np.array(preprocessed_data)
        # Normalization
        if need_stan:
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

    def _H2GPreprocess(self, need_stan = False):
        '''
        Preprocessing data for analyzing hunting to grazing rate.
        Processed data have following features:  
            - The closest distance between Pacman and dots.
            - Distance between Pacman and ghost current locations.
            - Two remaining scared time.
            - Distance between Pacman and ghost future locations.
        
        :param need_stan: Whther need standardiztion (x-mean)/std.
        :return: Preprocessed data matrix.
            - h2g_training_data: Features in training set for hunt2graze analyzing.
            - h2g_training_label: Labels in training set for hunt2graze analyzing.
            - h2g_testing_data: Features in testing set for hunt2graze analyzing
            - h2g_testing_label: Labels in testing set for hunt2graze analyzing
        '''
        # Extract useful features
        preprocessed_data = []
        for index, each in enumerate(self.hunting_data):
            # Extract features
            temp = [
                [int(i) for i in each[11].strip('()').split(',')], # ghost 1 location
                [int(i) for i in each[12].strip('()').split(',')], # ghost 2 location
                [int(i) for i in each[10].strip('()').split(',')], # Pacman location
                float(each[2]) if each[2] != '' else 0,  # ghost 1 remained scared time
                float(each[3]) if each[3] != '' else 0,  # ghost 2 remained scared time
                each[13],  # ghost 1 moving direction
                each[14],  # ghost 2 moving direction
                each[4], # distance between Pacman and ghost 1
                each[5], # distance between Pacman and ghost 2
                each[6] # distance between Pacman and closest dot
            ]
            # Processing features
            ghosts_loc = temp[0:2]
            move_dir = temp[5:7]
            pacman_loc = temp[2]
            remained_time = temp[3:5]
            ghosts_distance = temp[7:9]
            dot_distance = temp[9]
            future_loc = estimateGhostLocation(self.locs_df, ghosts_loc, move_dir, remained_time, pacman_loc)
            future_loc = np.array([np.array(each).squeeze() for each in future_loc])
            future_distance = computeLocDis(self.map_distance, pacman_loc, future_loc)
            ghosts_distance = [float(a) for a in ghosts_distance]
            processed_each = [float(dot_distance)]
            processed_each.extend(ghosts_distance)
            processed_each.extend(remained_time)
            processed_each.extend(future_distance)
            preprocessed_data.append(processed_each)
        preprocessed_data = np.array(preprocessed_data)
        # The preprocessed data has the following 7 features:
        #   [0] The closest distance between Pacman and dots
        #   [1] The distance between Pacman and the current ghost 1
        #   [2] The distance between Pacman and the current ghost 2
        #   [3] Remaining scared time of ghost 1
        #   [4] Remaining scared time of ghost 2
        #   [5] The distance between Pacman and the future ghost 1
        #   [6] The distance between Pacman and the future ghost 2

        if need_stan:
            for col_index in range(preprocessed_data.shape[1]):
                preprocessed_data[:, col_index] = (preprocessed_data[:, col_index] - np.nanmean(
                    preprocessed_data[:, col_index])) / np.nanstd(preprocessed_data[:, col_index])
        # Split into training and testing sets(60% for training and 40% for testing) for each mode
        training_ratio = 0.6
        sample_num = len(preprocessed_data) - 1
        training_num = int(training_ratio * sample_num)
        shuffled_index = np.arange(0, sample_num)
        np.random.shuffle(shuffled_index)
        training_index = shuffled_index[:training_num]
        testing_index = shuffled_index[training_num:]
        # TODO: ===========================================
        with open('testing_index.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(testing_index.reshape(-1,1))
        # TODO: ===========================================
        h2g_training_data = preprocessed_data[training_index]
        h2g_training_label = self.hunting_label[training_index, :]
        h2g_testing_data = preprocessed_data[testing_index, :]
        h2g_testing_label = self.hunting_label[testing_index, :]
        return (h2g_training_data, h2g_training_label, h2g_testing_data, h2g_testing_label)


    # ======================================
    #           ANALYZING G2H
    # ======================================
    def G2HAnalyzeMLP(self, in_dim, batch_size = 5, lr = 1e-2, enable_cuda = False, need_train = True, model_file = ''):
        '''
        Grazing to hunting rate analysis with MLP.
        :param in_dim: Input dimension.
        :param batch_size: The size of training batch (default = 5).
        :param lr: Learning rate (default = 1e-2).
        :param enable_cuda: Whether enable CUDA (default = False).
        :param need_train: Whether need train (default = True).
        :param model_file: The filename of trained model. Required if need_train = False.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2HPreprocess()
        print('...Finished preprocessing!\n')
        model = Graze2Hunt(in_dim, batch_size, lr, enable_cuda)
        # Loss before training
        pred_label = model.testMLP(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        print('Loss before training {}'.format(estimation_loss))
        # Train the model
        if need_train:
            print('=' * 30)
            print('Train model for analyzing Graze2Hunt:\n', model.network)
            batch_loss = model.trainMLP(training_data, training_label)
            sio.savemat('MLP_train_batch_loss_batch{}.mat'.format(batch_size),
                        {'batch_loss': np.array([tensor2np(each) for each in batch_loss]), 'batch_size': batch_size})
            model.saveMLPModel('save_m/G2H_model_batch{}.pt'.format(batch_size))
        else:
            if '' == model_file:
                raise ValueError('The filename of the trained model is not defined!')
            print('=' * 30)
            print('Load model for analyzing Graze2Hunt:\n', model_file)
            model = Graze2Hunt(in_dim, batch_size, lr, enable_cuda)
            model.loadMLPModel(model_file)
        # Testing the model after training
        print('\n', '=' * 30)
        print('Testing...')
        pred_label = model.testMLP(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_label)
        auc = AUC(pred_label, testing_label)
        print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))

    def G2HAnalyzeLogistic(self):
        '''
        Grazing to hunting rate analysis with the logistic regression.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2HPreprocess()
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_label = 1 - testing_label # To satisfy the prediction of logistic regression TODO: more explanation
        # Train a Logistic Model
        model = LogisticRegression(
            penalty = 'elasticnet',
            random_state = 0,
            solver = 'saga',
            l1_ratio = 0.5,
            fit_intercept = True)
        model.fit(training_data, training_ind_label)
        # Testing
        pred_label = model.predict_proba(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_label)
        auc = AUC(pred_label, testing_label)
        print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        print(
            model.coef_,
            model.intercept_
        )
        # Save tranferring rate
        with open('Logistic_grazing2hunting_rate (ENet).csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))

    def G2HAnalyzeDTree(self):
        '''
        Grazing to hunting rate analysis with the decision tree.
        :return: VOID
        '''
        # In decision tree, don't have to do standardization
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2HPreprocess(need_stan = False)
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_ind_label = np.array([list(each).index(0) for each in testing_label])
        # To satisfy the prediction of logistic regression.
        # Because after preprocessing, class 0 stands for grazing, thus, need reversed.
        testing_label = 1 - testing_label
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion = 'entropy',
                                       random_state = 0,
                                       max_depth = 4)
        trained_tree = model.fit(training_data, training_ind_label)
        # Testing
        pred_label = trained_tree.predict_proba(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = np.sum(trained_tree.predict(testing_data) == testing_ind_label) / len(testing_ind_label)
        auc = AUC(pred_label, testing_label)
        print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        # Store tree features as txt file
        print('Feature Importances:', trained_tree.feature_importances_)
        tree_structure =  export_text(trained_tree, feature_names = ['D_g1', 'D_g2', 'D_d'])
        with open('G2H_tree_structure.txt', 'w') as file:
            file.write(tree_structure)
        print('Decision Rule:\n', tree_structure)
        # Store hunting rate
        with open('DTree_grazing2hunting_rate.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))
        # Plot the trained tree
        node_data = export_graphviz(trained_tree,
                                    out_file = None,
                                    feature_names=['D_g1', 'D_g2', 'D_d'],
                                    class_names=['Hunting', 'Grazing'],
                                    filled = True,
                                    proportion = True)
        graph = graphviz.Source(node_data)
        graph.render('G2H_trained_tree_structure', view = False)


    # ======================================
    #           ANALYZING H2G
    # ======================================
    def H2GAnalyzeDeterministic(self):
        '''
        Hunting to grazing analysis with the deterministic function.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._H2GPreprocess()
        print('...Finished preprocessing!\n')
        all_data = np.vstack((training_data, testing_data))
        all_label = np.vstack((training_label, testing_label))
        m = Hunt2Graze()
        # Use the deterministic model for analysis
        pred_label = m.deterministicModel(all_data, all_label)
        correct_rate = correctRate(pred_label, all_label)
        print('Classification correct rate {}'.format(correct_rate))

    def H2GAnalyzeLogistic(self):
        '''
        Hunting to grazing analysis with the logistic regression.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._H2GPreprocess()
        print('...Finished preprocessing!\n')
        # Train the logistic model
        m = Hunt2Graze()
        trained_model = m.trainLogistic(training_data, training_label)
        pred_label = m.testingLogistic(testing_data)
        testing_label = 1 - testing_label
        estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_label)
        auc = AUC(pred_label, testing_label)
        print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        print(
            trained_model.coef_,
            trained_model.intercept_)
        # Save tranferring rate
        with open('Logistic_hunting2grazing_rate (ENet).csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1, 1)))))
        # TODO: print out fail trials
        fail_index = []
        for index in range(pred_label.shape[0]):
            if np.all(np.round(pred_label[index,:]) == testing_label[index,:]):
                continue
            else:
                fail_index.append(index)
        with open('fail_testing_index.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.index(fail_index).reshape(-1,1))

    def H2GAnalyzeDTree(self):
        '''
        Hunting to grazing analysis with the decision tree.
        :return: 
        '''
        #TODO: features as a parameter; put in the H2G class; a general framework for all the analysis method
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._H2GPreprocess()
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_ind_label = np.array([list(each).index(0) for each in testing_label])
        # To satisfy the prediction of logistic regression.
        # Because after preprocessing, class 0 stands for grazing, thus, need reversed.
        testing_label = 1 - testing_label
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion='entropy',
                                       random_state=0,
                                       max_depth=8)
        trained_tree = model.fit(training_data, training_ind_label)
        # Testing
        pred_label = trained_tree.predict_proba(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = np.sum(trained_tree.predict(testing_data) == testing_ind_label) / len(testing_ind_label)
        auc = AUC(pred_label, testing_label)
        print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        print('Feature Importances:', trained_tree.feature_importances_)
        #TODO: store tree structures



if __name__ == '__main__':
    # # For restricting CPU usage
    # torch.set_num_threads(4)
    # torch.set_num_interop_threads(4)

    feature_filename = 'data/extract_feature.csv'
    label_filename = 'data/all_labels.csv'
    mode_filename = 'data/all_modes.csv'
    a = Analyzer(feature_filename, label_filename, mode_filename)

    # # Analyze grazing to hunting
    # a.G2HAnalyzeMLP(3, batch_size = 1, need_train= False, model_file='save_m/G2H_model_batch1(cr-0.972).pt') # With MLP: f(D)
    # a.G2HAnalyzeMLP(3, batch_size = 1, need_train= True) # With MLP: f(D)
    # a.G2HAnalyzeLogistic() # Train with logistic regression
    # a.G2HAnalyzeDTree() # Train with decision classification tree

    # # Analyze hunting to grazeing
    # a.H2GAnalyzeDeterministic()  # with deterministic model
    a.H2GAnalyzeLogistic() # with logistic regression
    # a.H2GAnalyzeDTree() # with decision tree