'''
Description:
    Analyze the transitive probability from grazing to hunting. Using data that first grazing and then hunting. 

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


class LessGHAnalyzer:
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
        random.seed(a=None)
        self.data = []
        self.labels =[]
        self.modes = []
        # Read features
        with open(feature_file, 'r') as file:
            reader = csv.DictReader(file)
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


    # ======================================
    #           PREPROCESSING
    # ======================================
    def _G2HPreprocess(self, need_stan=False):
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
        for index, each in enumerate(self.data):
            combined_dist = None
            ghost1_dist = float(each['distance1'])
            ghost2_dist = float(each['distance2'])
            dot_distance = float(each['rwd_pac_distance'])
            # Compute the combined distance
            this_label = self.labels[index, 3]
            if 'hunting1' == this_label:
                combined_dist = ghost1_dist
            elif 'hunting2' == this_label:
                combined_dist = ghost2_dist
            elif 'hunting_all' == this_label:
                #TODO: min or mean? (min is better)
                combined_dist = np.mean([ghost1_dist, ghost2_dist])
                # combined_dist = np.min([ghost1_dist, ghost2_dist])
            else:
                # TODO: min or mean? (min is better)
                combined_dist = np.mean([ghost1_dist, ghost2_dist])
                # combined_dist = np.min([ghost1_dist, ghost2_dist])

            temp = [
                combined_dist,
                dot_distance
            ]
            preprocessed_data.append(temp)
        preprocessed_data = np.array(preprocessed_data)
        # Labels
        preprocessed_labels = []
        for index, each in enumerate(self.labels):
            if 'hunting' in each[3]:
                preprocessed_labels.append([0, 1])
            else:
                preprocessed_labels.append([1, 0])
        preprocessed_labels = np.array(preprocessed_labels)
        # Standardization
        if need_stan:
            for col_index in range(preprocessed_data.shape[1]):
                preprocessed_data[:, col_index] = (preprocessed_data[:, col_index] - np.nanmean(
                    preprocessed_data[:, col_index])) / np.nanstd(preprocessed_data[:, col_index])
        # Split into training and testing sets(60% for training and 40% for testing) for each mode
        training_ratio = 0.8
        sample_num = preprocessed_data.shape[0] - 1
        training_num = int(training_ratio * sample_num)
        shuffled_index = np.arange(0, sample_num)
        np.random.shuffle(shuffled_index)
        training_index = shuffled_index[:training_num]
        testing_index = shuffled_index[training_num:]
        g2h_training_data = preprocessed_data[training_index, :]
        g2h_training_label = preprocessed_labels[training_index, :]
        g2h_testing_data = preprocessed_data[testing_index, :]
        g2h_testing_label = preprocessed_labels[testing_index, :]
        return (g2h_training_data, g2h_training_label, g2h_testing_data, g2h_testing_label)

    # ======================================
    #           ANALYZING G2H
    # ======================================
    def G2HAnalyzeLogistic(self):
        '''
        Grazing to hunting rate analysis with the logistic regression.
        :return: VOID
        '''
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2HPreprocess(need_stan = False)
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_label = 1 - testing_label  # To satisfy the prediction of logistic regression TODO: more explanation
        # Train a Logistic Model
        model = LogisticRegression(
            penalty='elasticnet',
            random_state=0,
            solver='saga',
            l1_ratio=0.5,
            fit_intercept=True)
        model.fit(training_data, training_ind_label)
        # Testing
        pred_label = model.predict_proba(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_label)
        auc = AUC(pred_label, testing_label)
        # print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        print(
            model.coef_,
            model.intercept_
        )
        # Save tranferring rate
        with open('Logistic_selected_grazing2hunting_rate (ENet).csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1, 1)))))

    def G2HAnalyzeDTree(self):
        '''
        Grazing to hunting rate analysis with the decision tree.
        :return: VOID
        '''
        # In decision tree, don't have to do standardization
        print('Start preprocessing...')
        training_data, training_label, testing_data, testing_label = self._G2HPreprocess(need_stan=False)
        print('...Finished preprocessing!\n')
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        testing_ind_label = np.array([list(each).index(0) for each in testing_label])
        # To satisfy the prediction of logistic regression.
        # Because after preprocessing, class 0 stands for grazing, thus, need reversed.
        testing_label = 1 - testing_label
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion='gini',
                                       max_depth=3)
        trained_tree = model.fit(training_data, training_ind_label)
        # Testing
        pred_label = trained_tree.predict_proba(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = np.sum(trained_tree.predict(testing_data) == testing_ind_label) / len(testing_ind_label)
        auc = AUC(pred_label, testing_label)
        # print('BCE Loss after training {}'.format(estimation_loss))
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        # Store tree features as txt file
        print('Feature Importances:', trained_tree.feature_importances_)
        tree_structure = export_text(trained_tree, feature_names=['D_C', 'D_d'])
        # with open('G2H_tree_structure.txt', 'w') as file:
        #     file.write(tree_structure)
        print('Decision Rule:\n', tree_structure)
        # Store hunting rate
        with open('DTree_selected_grazing2hunting_rate.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(np.hstack((testing_data, pred_label[:, 0].reshape((-1, 1)))))
        # Plot the trained tree
        node_data = export_graphviz(trained_tree,
                                    out_file=None,
                                    feature_names=['D_C', 'D_d'],
                                    class_names=['Hunting', 'Grazing'],
                                    filled=True,
                                    proportion=True)
        graph = graphviz.Source(node_data)
        graph.render('G2H_selected_trained_tree_structure', view=False)




if __name__ == '__main__':
    feature_filename = 'extracted_data/less_G2H_feature.csv'
    label_filename = 'extracted_data/less_G2H_label.csv'
    mode_filename = 'extracted_data/less_G2H_mode.csv'
    a = LessGHAnalyzer(feature_filename, label_filename, mode_filename)

    a.G2HAnalyzeLogistic()
    # a.G2HAnalyzeDTree()