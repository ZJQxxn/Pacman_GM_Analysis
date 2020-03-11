''''
Description:
    Model for analyzing the transfer probability from hunting model to the grazing model.

Author:
    Jiaqi Zhang <zjqseu@gmail.com >

Date:
    2020/3/5
'''

import numpy as np
from sklearn.linear_model import LogisticRegression
from evaluation import correctRate


class Hunt2Graze:
    '''
    Description:
    
    Variables:
    
    Functions:
    '''

    def __init__(self):
        pass

    def deterministicModel(self, data, label):
        '''
        The deterministic model for analyzing the transferation.
        :param data: All the data features, with shape of (number of samples, number of features).
        :param label: All the labels, with shape of (number of samples,)
        :return: Predictions with the shape of (number of samples, 2).
        '''
        pred_label = []
        for index, each in enumerate(data):
            if index % 500 == 0 and index is not 0:
                print('Finished {} samples...'.format(index))
                correct_rate = correctRate(pred_label, label[:index])
                print('Classification correct rate for first {} samples is {}'.format(index + 1, correct_rate))
            remained_time = each[3:5]
            pursuing_time = each[5:7]
            if np.all(pursuing_time > remained_time):
                cur_label = [0, 1]  # Switch to grazing mode
            else:
                cur_label = [1, 0]  # Remained hunting mode
            pred_label.append(cur_label)
        return np.array(pred_label, dtype = np.int)

    def trainLogistic(self, training_data,training_label):
        '''
        Train with the logistic model.
        :param training_data: Training data with shape of (number of samples, number of features)
        :param training_label: Training label with shape of (number of samples, 2).
        :return: Trained logistic model (sklearn.linear_model.LogisticRegression).
        '''
        # Label: 0 for hunting and 1 for grazing
        training_ind_label = np.array([list(each).index(0) for each in training_label])
        # Train a Logistic Model
        self.logstic_model = LogisticRegression(
                penalty='elasticnet',
                random_state=0,
                solver='saga',
                l1_ratio=0.5,
                fit_intercept=True,
                max_iter = 1000
        )
        self.logstic_model.fit(training_data, training_ind_label)
        return self.logstic_model

    def testingLogistic(self, testing_data):
        '''
        Testing the trained logistic model.
        :param testing_data: Testing data.
        :return: Prediction with shape of (number of samples, 2)
        '''
        pred_label = self.logstic_model.predict_proba(testing_data)
        return pred_label


