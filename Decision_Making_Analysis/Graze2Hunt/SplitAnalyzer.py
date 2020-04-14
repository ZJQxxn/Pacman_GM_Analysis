'''
Description:
    Analyze the transitive probability between hunting and grazing mode. 
    Hunting ghost 1 and hunting ghost 2 is considered as two different modes.  

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
import seaborn as sns
import graphviz
import random

from Hunt2GrazeModel import Hunt2Graze
from Graze2HuntModel import Graze2Hunt
from evaluation import binaryClassError, correctRate, AUC
from util import tensor2np, oneHot
from util import estimateGhostLocation, future_position, computeLocDis



class SplitAnalyzer:
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
        self.locs_df = pd.read_csv("../common_data/dij_distance_map.csv")
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
        self.hunting1_index = []
        self.hunting2_index = []
        self.grazing_index = []
        for index in range(len(self.data)-1):
            # data = self.data[index]the last
            # Discard the data of escaping mode, with escaping label, and at the last time step of a trial
            if 'escaping' != self.modes[index][3] \
                    and 'escaping' != self.modes[index+1][3] \
                    and self.data[index + 1][18] != "0":
                if 'hunting1' == self.modes[index][3]:
                    self.hunting1_index.append(index)
                elif 'hunting2' == self.modes[index][3]:
                    self.hunting2_index.append(index)
                elif 'grazing' == self.modes[index][3]:
                    self.grazing_index.append(index)
                else:
                    raise ValueError('Undefined Pacman mode! Check you modes file.')
        self.hunting1_data = self.data[self.hunting1_index, :]
        self.hunting2_data = self.data[self.hunting2_index, :]
        self.grazing_data = self.data[self.grazing_index, :]
        # Only use first three of labels, because the forth is a string
        self.hunting1_label = np.array(self.labels[self.hunting1_index, :3], dtype = np.float)
        self.hunting2_label = np.array(self.labels[self.hunting2_index, :3], dtype=np.float)
        self.grazing_label = np.array(self.labels[self.grazing_index, :3], dtype = np.float)

    # ======================================
    #           PREPROCESSING
    # ======================================
    def _G2H1Preprocess(self, need_stan = True):
        '''
        Preprocessing data for analyzing grazing to hunting 1.
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
        # Extract useful features; only pick out those data with label hunting 1.
        preprocessed_data = []
        no_hunt2_label = [] # labels of data with labels other than 'hunting2'
        for index, each in enumerate(self.grazing_data):
            # Discard data with 'hunting2' label
            if 1 == self.grazing_label[index][2]:
                continue
            pacman_loc = [int(i) for i in each[10].strip('()').split(',')]
            ghost_loc = each[7]
            if ghost_loc != '':
                ghost_loc = ghost_loc.replace('[(', '').replace('(', '').replace(')]', '').replace(')', '').split(',')
                ghost_loc = [[int(float(ghost_loc[index])), int(float(ghost_loc[index + 1]))] for index in range(len(ghost_loc) - 1)]
            else:
                ghost_loc = []
            #TODO: distance
            temp = [
                # float(each[4]), # Distance betweeen Pacman and ghost 1
                float(each[5]), # Distance between Pacman and ghost 2
                # float(each[6]), # Distance between Pacman and the closest dot
                float(each[4]) if 1 == self.grazing_label[index][1] else np.min([float(each[4]), float(each[5])]) # combined distance
            ]
            preprocessed_data.append(temp)
            no_hunt2_label.append(self.grazing_label[index, :])
        preprocessed_data = np.array(preprocessed_data)
        no_hunt2_label = np.array(no_hunt2_label)
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
        g2h1_training_data = preprocessed_data[training_index, :]
        g2h1_testing_data = preprocessed_data[testing_index, :]
        # Convert label to a 2-d array because we only have two classes here; [1, 0] for grazing and [0,1] for hunting1.
        g2h1_training_label = no_hunt2_label[training_index, :2]
        g2h1_testing_label = no_hunt2_label[testing_index, :2]
        return (g2h1_training_data, g2h1_training_label, g2h1_testing_data, g2h1_testing_label)

    def _G2H2Preprocess(self, need_stan=True):
        '''
        Preprocessing data for analyzing grazing to hunting 2.
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
        # Extract useful features; only pick out those data with label hunting 2.
        preprocessed_data = []
        no_hunt1_label = []  # labels of data with labels other than 'hunting1'
        for index, each in enumerate(self.grazing_data):
            # Discard data with 'hunting2' label
            if 1 == self.grazing_label[index][1]:
                continue
            pacman_loc = [int(i) for i in each[10].strip('()').split(',')]
            ghost_loc = each[7]
            if ghost_loc != '':
                ghost_loc = ghost_loc.replace('[(', '').replace('(', '').replace(')]', '').replace(')', '').split(',')
                ghost_loc = [[int(float(ghost_loc[index])), int(float(ghost_loc[index + 1]))] for index in
                             range(len(ghost_loc) - 1)]
            else:
                ghost_loc = []
            # TODO: distance
            temp = [
                float(each[4]), # Distance betweeen Pacman and ghost 1
                # float(each[5]), # Distance between Pacman and ghost 2
                # float(each[6]),  # Distance between Pacman and the closest dot
                float(each[5]) if 1 == self.grazing_label[index][2] else np.min([float(each[4]), float(each[5])])# combined distance
            ]
            preprocessed_data.append(temp)
            no_hunt1_label.append(self.grazing_label[index, :])
        preprocessed_data = np.array(preprocessed_data)
        no_hunt1_label = np.array(no_hunt1_label)
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
        g2h2_training_data = preprocessed_data[training_index, :]
        g2h2_testing_data = preprocessed_data[testing_index, :]
        # Convert label to a 2-d array because we only have two classes here; [1, 0] for grazing and [0,1] for hunting2.
        g2h2_training_label = np.array(
            [
                [1.,0.] if np.all(each == np.array([1.,0.,0.])) else [0,1]
                for each in no_hunt1_label[training_index, :]
            ]
        )
        g2h2_testing_label = np.array(
            [
                [1.,0.] if np.all(each == np.array([1.,0.,0.])) else [0,1]
                for each in no_hunt1_label[testing_index, :]
            ]
        )
        return (g2h2_training_data, g2h2_training_label, g2h2_testing_data, g2h2_testing_label)

    def _G2IntegrateHPreprocess(self, need_stan = True):
        '''
        Preprocessing data for analyzing grazing to hunting 1.
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
        # Extract useful features; only pick out those data with label hunting 1.
        preprocessed_data = []
        hunt1_index = [] # label index of hunting 1
        hunt2_index = [] # label index of hunting 2
        for index, each in enumerate(self.grazing_data):
            hunt1_flag = False
            hunt2_flag = False
            # Discard data with 'hunting2' label
            if 1 == self.grazing_label[index][2]:
                hunt2_index.append(index)
                hunt1_flag = True
            elif 1 == self.grazing_label[index][1]:
                hunt1_index.append(index)
                hunt2_flag = True
            pacman_loc = [int(i) for i in each[10].strip('()').split(',')]
            ghost_loc = each[7]
            if ghost_loc != '':
                ghost_loc = ghost_loc.replace('[(', '').replace('(', '').replace(')]', '').replace(')', '').split(',')
                ghost_loc = [[int(float(ghost_loc[index])), int(float(ghost_loc[index + 1]))] for index in range(len(ghost_loc) - 1)]
            else:
                ghost_loc = []

            integrated_dist = None
            if not hunt1_flag and not hunt2_flag:
                integrated_dist = np.mean([float(each[4]), float(each[5])])
            elif hunt1_flag and not hunt2_flag:
                integrated_dist = float(each[4])
            elif not hunt1_flag and hunt2_flag:
                integrated_dist = float(each[5])
            else:
                integrated_dist = np.mean([float(each[4]), float(each[5])])
            temp = [
                # float(each[4]), # Distance betweeen Pacman and ghost 1
                # float(each[5]), # Distance between Pacman and ghost 2
                float(each[6]), # Distance between Pacman and the closest dot
                integrated_dist
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
        # testing_index = shuffled_index[training_num:]
        testing_index = np.arange(0, sample_num)
        g2h_training_data = preprocessed_data[training_index, :]
        g2h_testing_data = preprocessed_data[testing_index, :]
        # Convert label to a 2-d array because we only have two classes here; [1, 0] for grazing and [0,1] for hunting.
        g2h_training_label = np.array(
            [
                [1., 0.] if np.all(each == np.array([1., 0., 0.])) else [0, 1]
                for each in self.grazing_label[training_index, :]
            ]
        )
        g2h_testing_label = np.array(
            [
                [1., 0.] if np.all(each == np.array([1., 0., 0.])) else [0, 1]
                for each in self.grazing_label[testing_index, :]
            ]
        )
        return (g2h_training_data, g2h_training_label, g2h_testing_data, g2h_testing_label)


    # ======================================
    #           ANALYZING G2H
    # ======================================
    def G2H1AnalyzeLogistic(self):
        '''
        Grazing to hunting rate analysis with the logistic regression.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2H1Preprocess()
        print('...Finished preprocessing!\n')
        # TODO: discard hunting distance and check whether another ghost will affect the performance
        # np.where(2 == np.array([list(each).index(1) for each in np.vstack((training_label, testing_label))]))
        # Label: 0 for hunting 1 and 1 for grazing
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
        with open('Logistic_grazing2hunting1_rate (ENet).csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))

    def G2H1AnalyzeDTree(self):
        '''
        Grazing to hunting rate analysis with the decision tree.
        :return: VOID
        '''
        # In decision tree, don't have to do standardization
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2H1Preprocess(need_stan = False)
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_ind_label = np.array([list(each).index(0) for each in testing_label])
        # To satisfy the prediction of logistic regression.
        # Because after preprocessing, class 0 stands for grazing, thus, need reversed.
        testing_label = 1 - testing_label
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion = 'gini',
                                       # random_state = 0,
                                       max_depth = 3)
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
        tree_structure =  export_text(trained_tree, feature_names = ['D_2', 'D_C'])
        with open('G2H1_tree_structure.txt', 'w') as file:
            file.write(tree_structure)
        print('Decision Rule:\n', tree_structure)
        # Store hunting rate
        with open('DTree_grazing2hunting1_rate.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))
        # Plot the trained tree
        node_data = export_graphviz(trained_tree,
                                    out_file = None,
                                    feature_names=['D_2', 'D_C'],
                                    class_names=['Hunting 1', 'Grazing'],
                                    filled = True,
                                    proportion = True)
        graph = graphviz.Source(node_data)
        graph.render('G2H1_trained_tree_structure', view = False)

    def G2H2AnalyzeLogistic(self):
        '''
        Grazing to hunting rate analysis with the logistic regression.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2H2Preprocess()
        print('...Finished preprocessing!\n')
        # np.where(2 == np.array([list(each).index(1) for each in np.vstack((training_label, testing_label))]))
        # Label: 0 for hunting 2 and 1 for grazing
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
        with open('Logistic_grazing2hunting2_rate (ENet).csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))

    def G2H2AnalyzeDTree(self):
        '''
        Grazing to hunting rate analysis with the decision tree.
        :return: VOID
        '''
        # In decision tree, don't have to do standardization
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2H2Preprocess(need_stan = False)
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_ind_label = np.array([list(each).index(0) for each in testing_label])
        # To satisfy the prediction of logistic regression.
        # Because after preprocessing, class 0 stands for grazing, thus, need reversed.
        testing_label = 1 - testing_label
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion = 'gini',
                                       # random_state = 0,
                                       max_depth = 3)
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
        tree_structure =  export_text(trained_tree, feature_names = ['D_1','D_C'])
        with open('G2H2_tree_structure.txt', 'w') as file:
            file.write(tree_structure)
        print('Decision Rule:\n', tree_structure)
        # Store hunting rate
        with open('DTree_grazing2hunting2_rate.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))
        # Plot the trained tree
        node_data = export_graphviz(trained_tree,
                                    out_file = None,
                                    feature_names=['D_1','D_C'],
                                    class_names=['Hunting 2', 'Grazing'],
                                    filled = True,
                                    proportion = True)
        graph = graphviz.Source(node_data)
        graph.render('G2H2_trained_tree_structure', view = False)

    def G2IntegrateHAnalyzeLogistic(self):
        '''
                Grazing to hunting rate analysis with the logistic regression.
                :return: VOID
                '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2IntegrateHPreprocess()
        print('...Finished preprocessing!\n')
        # np.where(2 == np.array([list(each).index(1) for each in np.vstack((training_label, testing_label))]))
        # Label: 0 for hunting  and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_label = 1 - testing_label  # To satisfy the prediction of logistic regression TODO: more explanation
        # Train a Logistic Model
        model = LogisticRegression(
            penalty='l2',
            solver='lbfgs',
            fit_intercept=True)
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
        with open('Logistic_grazing2Integratehunting_rate (ENet).csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1, 1)))))

    def G2IntegrateHAnalyzeDTree(self):
        '''
        Grazing to hunting rate analysis with the decision tree.
        :return: VOID
        '''
        # In decision tree, don't have to do standardization
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2IntegrateHPreprocess(need_stan = False)
        temp_training = []
        temp_testing = []
        # for index in range(training_data.shape[0]):
        #     temp_training.append(training_data[index,2:])
        # for index in range(testing_data.shape[0]):
        #     temp_testing.append(testing_data[index, 2:])
        # training_data = np.array(temp_training)
        # testing_data = np.array(temp_testing)
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_ind_label = np.array([list(each).index(0) for each in testing_label])
        # To satisfy the prediction of logistic regression.
        # Because after preprocessing, class 0 stands for grazing, thus, need reversed.
        testing_label = 1 - testing_label
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion = 'gini',
                                       # random_state = 0,
                                       max_depth = 3)
        trained_tree = model.fit(training_data, training_ind_label)
        # Testing
        pred_label = trained_tree.predict_proba(testing_data)
        pred_ind_label = trained_tree.predict(testing_data)
        estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = np.sum(trained_tree.predict(testing_data) == testing_ind_label) / len(testing_ind_label)
        auc = AUC(pred_label, testing_label)
        print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        # Store tree features as txt file
        print('Feature Importances:', trained_tree.feature_importances_)
        tree_structure =  export_text(trained_tree, feature_names = ['D_d','D_C'])
        with open('G2HIntegrate_tree_structure.txt', 'w') as file:
            file.write(tree_structure)
        print('Decision Rule:\n', tree_structure)
        # Store hunting rate
        with open('DTree_grazing2hunting_integrate__rate.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1,1)))))
        # Plot the trained tree
        node_data = export_graphviz(trained_tree,
                                    out_file = None,
                                    feature_names=['D_d','D_C'],
                                    class_names=['Hunting', 'Grazing'],
                                    filled = True,
                                    proportion = True)
        graph = graphviz.Source(node_data)
        # graph.render('G2HIntegrate_trained_tree_structure', view = False)
        # Collect data
        testing_data = pd.DataFrame(testing_data)
        testing_data.columns = ['closest_dot_dist', 'combined_dist']
        pred_ind_label = pd.DataFrame(pred_ind_label)
        testing_ind_label = pd.DataFrame(testing_ind_label)
        is_correct = (pred_ind_label == testing_ind_label)
        is_correct.columns = ['is_correct']
        is_hunting = pd.DataFrame(1 - testing_ind_label.values)
        is_hunting.columns = ['is_hunting']
        testing_result = pd.concat([testing_data, is_correct, is_hunting], axis=1)
        # The overall hunting rate and huntign correct rate
        print('Hunting rate:{}'.format(len(np.where(is_hunting.is_hunting == 1)[0]) / testing_data.shape[0]))
        cr_index = np.where(is_correct.is_correct == True)
        hunt_index = np.where(is_hunting.is_hunting == True)
        inter = np.intersect1d(cr_index, hunt_index)
        print('Hunting correct rate:{}'.format(len(inter) / len(np.where(is_hunting.is_hunting == 1)[0])))
        # Plot estimation accuracy heat map
        # plt.figure(figsize=(15,10))
        ax = sns.heatmap(
            testing_result.pivot_table(
                index="closest_dot_dist",
                columns="combined_dist",
                values="is_correct",
                aggfunc=lambda x: sum(x) / len(x),
            )[
            ::-1
            ],
            cmap="coolwarm",
            square=True,
            cbar_kws={'shrink': 0.5}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        bottom, top = plt.gca().get_ylim()
        plt.title('G2H Correct Rate', fontsize=20)
        plt.xlabel("Combined Ghosts Distance", fontsize=20)
        plt.xticks(fontsize=15)
        plt.ylabel("Nearest Dot Distance", fontsize=20)
        plt.yticks(plt.yticks()[0][::3], plt.yticks()[1][::3], fontsize=15)
        plt.gca().set_ylim(bottom + 0.5, top - 0.5)
        plt.show()
        # Plot transfer rate heat map
        plt.clf()
        # plt.figure(figsize=(15, 10))
        ax = sns.heatmap(
            testing_result.pivot_table(
                index="closest_dot_dist",
                columns="combined_dist",
                values="is_hunting",
                aggfunc=lambda x: sum(x) / len(x),
            )[
            ::-1
            ],
            cmap="coolwarm",
            square=True,
            cbar_kws={'shrink': 0.5}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=15)
        bottom, top = plt.gca().get_ylim()
        plt.title('G2H Transfer Rate', fontsize=20)
        plt.xlabel("Combined Ghosts Distance", fontsize=20)
        plt.xticks(fontsize=15)
        plt.ylabel("Nearest Dot Distance", fontsize=20)
        plt.yticks(plt.yticks()[0][::3], plt.yticks()[1][::3], fontsize=15)
        plt.gca().set_ylim(bottom + 0.5, top - 0.5)
        plt.show()


if __name__ == '__main__':
    # # For restricting CPU usage
    # torch.set_num_threads(4)
    # torch.set_num_interop_threads(4)

    feature_filename = 'extracted_data/extract_feature.csv'
    label_filename = 'extracted_data/split_all_labels.csv'
    mode_filename = 'extracted_data/split_all_modes.csv'
    a = SplitAnalyzer(feature_filename, label_filename, mode_filename)
    # # Analyze grazing to hunting

    # a.G2H1AnalyzeLogistic()
    # a.G2H2AnalyzeLogistic()
    # a.G2IntegrateHAnalyzeLogistic()

    # a.G2H1AnalyzeDTree()
    # a.G2H2AnalyzeDTree()
    a.G2IntegrateHAnalyzeDTree()