'''
Description:
    The abstract class for estimator of all conditions.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/17
'''
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import sys

sys.path.append('./')
from EstimationUtils import oneHot

class Estimator(ABC):
    '''
    Description:
        The abstract class for movement estimators in all cases. 
        
    Variables:
        data (pandas.DataFrame): All the features.
    
    Functions:
        __init__: Read the data from csv file.
        _extractLocalFeature (abstract): Extract local features. Should be the same for all estimators.
         _extractGlobalFeature (abstract): Extract global features. Should be the same for all estimators.
         estimation (abstract): The estimation process.
         plotRes (abstract): Plot the estimation on the map.
    '''

    def __init__(self, all_feature_file, local_feature_file, global_feature_file, eval_list):
        '''
        The class initialization. Read data from the corresponding csv file.
        :param filename: CSV filename.
        '''
        super(Estimator, self).__init__()
        self.local_features = pd.read_csv(local_feature_file)
        self.global_features = pd.read_csv(global_feature_file)
        for c in eval_list["local"]:
            self.local_features[c] = self.local_features[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
        for c in eval_list["global"]:
            self.global_features[c] = self.global_features[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
        self.labels = pd.read_csv(all_feature_file).merge(
            self.local_features,
            on = ["file", "index"],
            how = "right"
        )[["pacman_dir"]]
        self.labels = self.labels.fillna('stay')
        self.class_list = ['up', 'down', 'left', 'right', 'stay']
        self.labels = self.labels.apply(lambda x : oneHot(x.values.item(), self.class_list), axis = 1)
        print("Finished initialization!")

    # @abstractmethod
    # def localEstimation(self):
    #     '''
    #     The estimation process with local features.
    #     :return: Correct rate (float).
    #     '''
    #     pass
    #
    # @abstractmethod
    # def globalEstimation(self):
    #     '''
    #     The estimation process with global features.
    #     :return: Correct rate (float).
    #     '''
    #     pass

    @abstractmethod
    def plotRes(self):
        '''
        Plot the estimations on the map.
        :return: VOID
        '''
        pass

