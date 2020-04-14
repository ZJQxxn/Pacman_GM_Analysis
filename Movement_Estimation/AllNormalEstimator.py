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
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree.export import export_text, export_graphviz
import sys

sys.path.append('./')
from Estimator import Estimator
from evaluation import AUC, correctRate
from EstimationUtils import oneHot

class AllNormalEstimatior(Estimator):

    def __init__(self, all_feature_file, local_feature_file, global_feature_file, eval_list):
        print("Start reading data...")
        super(AllNormalEstimatior, self).__init__(all_feature_file, local_feature_file, global_feature_file, eval_list)
        # Select only useful features; convert directions to vectors
        self.dir_list = ["up", "down", "left", "right"] #TODO: what about "same"?
        self.local_features = self.local_features[
            ["local_bean_num_left",
             "local_bean_num_right",
             "local_bean_num_up",
             "local_bean_num_down",
             "local_ghost1_dir",
             "local_ghost2_dir",
             # "local_nearest_energizer_dir",
             "pacmanPos"]
        ]
        #TODO: how to deal with nan? Currently, set to an empty vector
        self.local_features.local_ghost1_dir= self.local_features.local_ghost1_dir.apply(
            lambda x: oneHot(x, self.dir_list))
        self.local_features.local_ghost2_dir= self.local_features.local_ghost2_dir.apply(
            lambda x: oneHot(x, self.dir_list)
        )
        #TODO: need dir?
        # self.local_features.local_nearest_energizer_dir= self.local_features.local_nearest_energizer_dir.apply(
        #     lambda x: oneHot(x, self.dir_list)
        # )
        self.local_features.pacmanPos = self.local_features.pacmanPos.apply(lambda x: eval(x))
        self.local_features = self.local_features.assign(
            pos_x = self.local_features.pacmanPos.apply(lambda x: x[0]),
            pos_y = self.local_features.pacmanPos.apply(lambda x: x[1])
        )
        self.local_features = self.local_features.drop(columns = ["pacmanPos"])
        self.global_features = self.global_features[
            ["left_count",
             "right_count",
             "up_count",
             "down_count",
             "ghost1_global_dir",
             "ghost2_global_dir",
             "global_energizer_dir",
             "pacmanPos"]
        ]
        self.global_features.ghost1_global_dir = self.global_features.ghost1_global_dir.apply(
            lambda x: oneHot(x, self.dir_list)
        )
        self.global_features.ghost2_global_dir = self.global_features.ghost2_global_dir.apply(
            lambda x: oneHot(x, self.dir_list)
        )
        self.global_features.global_energizer_dir = self.global_features.global_energizer_dir.apply(
            lambda x: oneHot(x, self.dir_list)
        )
        self.global_features.pacmanPos = self.global_features.pacmanPos.apply(lambda x: eval(x))
        self.global_features = self.global_features.assign(
            pos_x=self.global_features.pacmanPos.apply(lambda x: x[0]),
            pos_y=self.global_features.pacmanPos.apply(lambda x: x[1])
        )
        self.global_features = self.global_features.drop(columns=["pacmanPos"])
        # Indices of training and testing data
        training_ratio = 0.8
        sample_num = self.labels.shape[0]
        training_num = int(training_ratio * sample_num)
        shuffled_index = np.arange(0, sample_num)
        np.random.shuffle(shuffled_index)
        training_index = shuffled_index[:training_num]
        testing_index = shuffled_index[training_num:]
        # Split local features into training and testing set
        self.local_train_features = self.local_features.iloc[training_index, :]
        self.local_train_labels = self.labels.iloc[training_index]
        self.local_test_features = self.local_features.iloc[testing_index, :]
        self.local_test_labels = self.labels.iloc[testing_index]
        # Split global features into training and testing set
        self.global_train_features = self.global_features.iloc[training_index, :]
        self.global_train_labels = self.labels.iloc[training_index]
        self.global_test_features = self.global_features.iloc[testing_index, :]
        self.global_test_labels = self.labels.iloc[testing_index]

    def _df2Vec(self, df_data):
        '''
        Concate all the columns into a vector.
        :param df_data: Data with the type pf pandas.DataFrame
        :return: A 2-d numpy.ndarray data.
        '''
        columns_name = df_data.columns.values
        res =  df_data[[columns_name[0]]].values
        for each_col in columns_name[1:]:
            res = np.hstack(
                (res,
                 df_data[[each_col]].values if not isinstance(df_data[[each_col]].values[0,0], np.ndarray) # scalars
                 else [each for each in df_data[[each_col]].values[:,0]]# vectors
                 )
            )
        return res

    def localEstimationLogistic(self):
        '''
        The estimation process with local features.
        :return: Correct rate (float).
        '''
        print('='*20, 'Local Estimation (Logistic)','='*20)
        train_data = self._df2Vec(self.local_train_features)
        testing_data = self._df2Vec(self.local_test_features)
        train_ind_label = self.local_train_labels.apply(lambda x: list(x).index(1)).values
        testing_ind_label = self.local_test_labels.apply(lambda x: list(x).index(1)).values
        # Train a Logistic Model
        model = LogisticRegression(
            penalty='l1',
            tol = 1e-5,
            solver='saga',
            fit_intercept=True,
            multi_class='multinomial')
        model.fit(train_data, train_ind_label)
        # Testing
        pred_label_prob = model.predict_proba(testing_data)
        pred_label = model.predict(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_ind_label)
        print('Classification correct rate {}'.format(correct_rate))

    def localEstimationDTree(self):
        print('=' * 20, 'Local Estimation (DTree)', '=' * 20)
        train_data = self._df2Vec(self.local_train_features)
        testing_data = self._df2Vec(self.local_test_features)
        train_ind_label = self.local_train_labels.apply(lambda x: list(x).index(1)).values
        testing_ind_label = self.local_test_labels.apply(lambda x: list(x).index(1)).values
        # testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=10)
        trained_tree = model.fit(train_data, train_ind_label)
        # Testing
        pred_label_prob = model.predict_proba(testing_data)
        pred_label = model.predict(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_ind_label)
        print('Classification correct rate {}'.format(correct_rate))

    def localEstimationSVM(self):
        print('=' * 20, 'Local Estimation (SVM)', '=' * 20)
        train_data = self._df2Vec(self.local_train_features)
        testing_data = self._df2Vec(self.local_test_features)
        train_ind_label = self.local_train_labels.apply(lambda x: list(x).index(1)).values
        testing_ind_label = self.local_test_labels.apply(lambda x: list(x).index(1)).values
        # testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train the decision classification tree
        model = SVC(gamma = "auto", decision_function_shape = "ovo")
        model.fit(train_data, train_ind_label)
        # Testing
        # pred_label_prob = model.predict_proba(testing_data)
        pred_label = model.predict(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_ind_label)
        print('Classification correct rate {}'.format(correct_rate))

    def globalEstimationLogistic(self):
        '''
        The estimation process with global features.
        :return: Correct rate (float).
        '''
        print('=' * 20, 'Global Estimation (Logistic)', '=' * 20)
        train_data = self._df2Vec(self.global_train_features)
        testing_data = self._df2Vec(self.global_test_features)
        train_ind_label = self.global_train_labels.apply(lambda x: list(x).index(1)).values
        testing_ind_label = self.global_test_labels.apply(lambda x: list(x).index(1)).values
        # Train a Logistic Model
        model = LogisticRegression(
            penalty='l1',
            tol=1e-5,
            solver='saga',
            fit_intercept=True,
            multi_class='multinomial')
        model.fit(train_data, train_ind_label)
        # Testing
        pred_label_prob = model.predict_proba(testing_data)
        pred_label = model.predict(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_ind_label)
        print('Classification correct rate {}'.format(correct_rate))

    def globalEstimationDTree(self):
        print('=' * 20, 'Global Estimation (DTree)', '=' * 20)
        train_data = self._df2Vec(self.global_train_features)
        testing_data = self._df2Vec(self.global_test_features)
        train_ind_label = self.global_train_labels.apply(lambda x: list(x).index(1)).values
        testing_ind_label = self.global_test_labels.apply(lambda x: list(x).index(1)).values
        # testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train the decision classification tree
        model = DecisionTreeClassifier(criterion='entropy',
                                       max_depth=10)
        trained_tree = model.fit(train_data, train_ind_label)
        # Testing
        pred_label_prob = model.predict_proba(testing_data)
        pred_label = model.predict(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_ind_label)
        print('Classification correct rate {}'.format(correct_rate))

    def globalEstimationSVM(self):
        print('=' * 20, 'Global Estimation (SVM)', '=' * 20)
        train_data = self._df2Vec(self.global_train_features)
        testing_data = self._df2Vec(self.global_test_features)
        train_ind_label = self.global_train_labels.apply(lambda x: list(x).index(1)).values
        testing_ind_label = self.global_test_labels.apply(lambda x: list(x).index(1)).values
        # testing_label = np.array([list(each) for each in testing_label], dtype=np.int)
        # Train the decision classification tree
        model = SVC(gamma = "auto", decision_function_shape = "ovo")
        model.fit(train_data, train_ind_label)
        # Testing
        pred_label = model.predict(testing_data)
        # estimation_loss = binaryClassError(pred_label, testing_label)
        correct_rate = correctRate(pred_label, testing_ind_label)
        print('Classification correct rate {}'.format(correct_rate))

    def plotRes(self):
        '''
        Plot the estimations on the map.
        :return: VOID
        '''
        pass


if __name__ == '__main__':
    eval_list = {
        "global":["ghost1_global_dir", "ghost2_global_dir", "global_energizer_dir"],
        "local": ["local_ghost1_dir", "local_ghost2_dir", "local_nearest_energizer_dir"]
    }
    estimator = AllNormalEstimatior(
        './extracted_data/normal_all_data.csv',
        './extracted_data/normal_local_features.csv',
        './extracted_data/normal_global_features.csv',
        eval_list)
    print('Size of local features', estimator.local_features.shape)
    print('Size of global features', estimator.global_features.shape)

    # Estimation with local features
    estimator.localEstimationLogistic()
    estimator.localEstimationDTree()
    # estimator.localEstimationSVM()

    # Estimation with global features
    estimator.globalEstimationLogistic()
    estimator.globalEstimationDTree()
    # estimator.globalEstimationSVM()