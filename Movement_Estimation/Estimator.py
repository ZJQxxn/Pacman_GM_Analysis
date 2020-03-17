'''
Description:
    The abstract class for estimator of all conditions.

uthor:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/17
'''
from abc import ABC, abstractmethod
import pandas as pd


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

    def __init__(self, filename):
        '''
        The class initialization. Read data from the corresponding csv file.
        :param filename: CSV filename.
        '''
        super(Estimator, self).__init__()
        self.data = pd.read_csv(filename)

    @abstractmethod
    def _extractLocalFeature(self):
        '''
        Extract local features.
        :return: Local features (pandas.DataFrame)
        '''
        pass

    @abstractmethod
    def _extractGlobalFeature(self):
        '''
        Extract global features.
        :return: Global features (pandas.DataFrame)
        '''
        pass

    @abstractmethod
    def localEstimation(self):
        '''
        The estimation process with local features.
        :return: Correct rate (float).
        '''
        pass

    @abstractmethod
    def globalEstimation(self):
        '''
        The estimation process with global features.
        :return: Correct rate (float).
        '''
        pass

    @abstractmethod
    def plotRes(self):
        '''
        Plot the estimations on the map.
        :return: VOID
        '''
        pass

