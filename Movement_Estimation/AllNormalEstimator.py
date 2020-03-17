''''
Description:
    The movement estimation for all the normal data. 

uthor:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/17
'''
import pandas as pd

from Estimator import Estimator

class AllNormalEstimatior(Estimator):

    def __init__(self, filename):
        super(AllNormalEstimatior, self).__init__(filename)
        # TODO: determine labels (the movement direction)

    def _extractLocalFeature(self):
        '''
        Extract local features.
        :return: Local features (pandas.DataFrame)
        '''
        #TODO: for now, choose raw data and make no processing
        self.local_features = self.data.loc[:, ['distance1', 'distance2']]
        return self.local_features



    def _extractGlobalFeature(self):
        '''
        Extract global features.
        :return: Global features (pandas.DataFrame)
        '''
        # TODO: for now, choose raw data and make no processing
        self.global_features = self.data.loc[:, ['distance1', 'distance2']]
        return self.global_features


    def localEstimation(self):
        '''
        The estimation process with local features.
        :return: Correct rate (float).
        '''
        local_features = self._extractLocalFeature()
        # TODO: estimation with multi-class logistic


    def globalEstimation(self):
        '''
        The estimation process with global features.
        :return: Correct rate (float).
        '''
        global_features = self._extractGlobalFeature()
        # TODO: estimation with multi-class logistic

    def plotRes(self):
        '''
        Plot the estimations on the map.
        :return: VOID
        '''
        pass


if __name__ == '__main__':
    estimator = AllNormalEstimatior('./extracted_data/normal_all_data.csv')
    print('Size of the data', estimator.data.shape)