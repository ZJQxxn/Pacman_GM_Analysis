''''
Description:
    The movement estimation for all the normal data. 

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/17
'''
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree.export import export_text, export_graphviz

from Estimator import Estimator
from Estimator import oneHot
from Estimator import locs_df

class AllNormalEstimatior(Estimator):

    def __init__(self, filename):
        super(AllNormalEstimatior, self).__init__(filename)
        self.class_list = ['up', 'down', 'left', 'right', 'stay']
        self.labels = self.data.pacman_dir.fillna('stay')
        self.labels = self.labels.map(lambda x : oneHot(x, self.class_list)).values


    def _extractLocalFeature(self):
        '''
        Extract local features.
        :return: Local features (pandas.DataFrame)
        '''
        # local_bean_num = localBeanNum(self.data)
        print("Finished extracting the number of local beans.")
        self.local_features = self.data.loc[:, ['distance1', 'distance2']].values
        # Split into training and testing sets(60% for training and 40% for testing) for each mode
        training_ratio = 0.6
        sample_num = self.local_features.shape[0]
        training_num = int(training_ratio * sample_num)
        shuffled_index = np.arange(0, sample_num)
        np.random.shuffle(shuffled_index)
        training_index = shuffled_index[:training_num]
        testing_index = shuffled_index[training_num:]
        local_training_data = self.local_features[training_index, :]
        local_training_label = self.labels[training_index]
        local_testing_data = self.local_features[testing_index, :]
        local_testing_label = self.labels[testing_index]
        return local_training_data, local_training_label, local_testing_data, local_testing_label

    def _extractGlobalFeature(self):
        '''
        Extract global features.
        :return: Global features (pandas.DataFrame)
        '''
        # TODO: for now, choose raw data and make no preprocessing
        self.global_features = self.data.loc[:, ['distance1', 'distance2']].values
        # Split into training and testing sets(60% for training and 40% for testing) for each mode
        training_ratio = 0.6
        sample_num = self.global_features.shape[0]
        training_num = int(training_ratio * sample_num)
        shuffled_index = np.arange(0, sample_num)
        np.random.shuffle(shuffled_index)
        training_index = shuffled_index[:training_num]
        testing_index = shuffled_index[training_num:]
        global_training_data = self.global_features[training_index, :]
        global_training_label = self.labels[training_index]
        global_testing_data = self.global_features[testing_index, :]
        global_testing_label = self.labels[testing_index]
        return global_training_data, global_training_label, global_testing_data, global_testing_label

    def localEstimationLogistic(self):
        '''
        The estimation process with local features.
        :return: Correct rate (float).
        '''
        print('='*20, 'Local Estimation','='*20)
        print('Start preprocessing...')
        train_data, train_label, testing_data, testing_label = self._extractLocalFeature()
        print('...Finished preprocessing!\n')
        train_ind_label = np.array([list(each).index(1) for each in train_label])
        testing_label = np.array([list(each) for each in testing_label], dtype = np.int)
        # Train a Logistic Model
        model = LogisticRegression(
            penalty='elasticnet',
            random_state=0,
            solver='saga',
            l1_ratio=0.5,
            fit_intercept=True,
            multi_class='multinomial')
        model.fit(train_data, train_ind_label)
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

    def localEstimationDTree(self):
        print('=' * 20, 'Local Estimation', '=' * 20)
        print('Start preprocessing...')
        train_data, train_label, testing_data, testing_label = self._extractGlobalFeature()
        print('...Finished preprocessing!\n')
        train_ind_label = np.array([list(each).index(1) for each in train_label])
        testing_ind_label = np.array([list(each).index(1) for each in testing_label])
        testing_label = np.array([list(each) for each in testing_label], dtype = np.int)
        # testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion='entropy',
                                       random_state=0,
                                       max_depth=4)
        trained_tree = model.fit(train_data, train_ind_label)
        # Testing
        pred_label = trained_tree.predict_proba(testing_data)
        correct_rate = np.sum(trained_tree.predict(testing_data) == testing_ind_label) / len(testing_ind_label)
        auc = AUC(pred_label, testing_label)
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        # Store tree features as txt file
        print('Feature Importances:', trained_tree.feature_importances_)

    def globalEstimationLogistic(self):
        '''
        The estimation process with global features.
        :return: Correct rate (float).
        '''
        print('=' * 20, 'Global Estimation', '=' * 20)
        print('Start preprocessing...')
        train_data, train_label, testing_data, testing_label = self._extractLocalFeature()
        print('...Finished preprocessing!\n')
        train_ind_label = np.array([list(each).index(1) for each in train_label])
        testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train a Logistic Model
        model = LogisticRegression(
            penalty='elasticnet',
            random_state=0,
            solver='saga',
            l1_ratio=0.5,
            fit_intercept=True,
            multi_class='multinomial')
        model.fit(train_data, train_ind_label)
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

    def globalEstimationDTree(self):
        print('=' * 20, 'Global Estimation', '=' * 20)
        print('Start preprocessing...')
        train_data, train_label, testing_data, testing_label = self._extractGlobalFeature()
        print('...Finished preprocessing!\n')
        train_ind_label = np.array([list(each).index(1) for each in train_label])
        testing_ind_label = np.array([list(each).index(1) for each in testing_label])
        testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion='entropy',
                                       random_state=0,
                                       max_depth=4)
        trained_tree = model.fit(train_data, train_ind_label)
        # Testing
        pred_label = trained_tree.predict_proba(testing_data)
        correct_rate = np.sum(trained_tree.predict(testing_data) == testing_ind_label) / len(testing_ind_label)
        auc = AUC(pred_label, testing_label)
        print('Classification correct rate {}'.format(correct_rate))
        print('AUC {}'.format(auc))
        # Store tree features as txt file
        print('Feature Importances:', trained_tree.feature_importances_)

    def plotRes(self):
        '''
        Plot the estimations on the map.
        :return: VOID
        '''
        pass


if __name__ == '__main__':
    estimator = AllNormalEstimatior('./extracted_data/normal_all_data.csv')
    print('Size of the data', estimator.data.shape)

    estimator._extractLocalFeature()

    # # Estimation with local features
    # estimator.localEstimationLogistic()
    # estimator.localEstimationDTree()

    # Estimation with global features
    # estimator.globalEstimationLogistic()
    # estimator.globalEstimationDTree()