'''
Description:
    Analyze the transitive probability from grazing to hunting. Using pre-hunt data. 

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/28
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PreHuntGHAnalyzer:
    '''
    Description:

    Variables:

    Functions:
    '''

    def __init__(self, feature_file):
        '''
        Initialization.
        :param feature_file: Filename for features. 
        '''
        self.data = pd.read_csv(feature_file)

    def analyze(self):
        print()
        self.data = self.data.assign(
            mean_ghost_distance = self.data.loc[:,['mean_ghost1_distance', 'mean_ghost2_distance']].mean(axis = 1)
        ).drop(columns = ['mean_ghost1_distance', 'mean_ghost2_distance'])
        print('Finished preprecessing.')
        # Plot the histograms
        p = sns.distplot(self.data.mean_ghost_distance,kde=False, color="#1BA3F9")
        # plt.fill_between([34, max(p.get_xticks())], 0, max(p.get_yticks()), color = '#f26755', alpha = 0.6)
        plt.fill_between([34, 40], 0, 14, color = '#f26755', alpha = 0.2, hatch = '\\', edgecolor = 'w')
        plt.xlabel('Average Ghosts Distance', fontsize = 30)
        plt.xticks(fontsize = 20)
        plt.ylabel('count', fontsize = 30)
        plt.yticks(fontsize=20)
        plt.show()

        plt.clf()
        sns.distplot(self.data.mean_rwd_distance, kde=False, color="#1BA3F9")
        plt.xlabel('Nearest Beans Distance', fontsize=30)
        plt.xticks(fontsize=20)
        plt.ylabel('count', fontsize=30)
        plt.yticks(fontsize=20)
        plt.show()


if __name__ == '__main__':
    feature_filename = 'extracted_data/prehunt_G2H_feature.csv'
    a = PreHuntGHAnalyzer(feature_filename)
    a.analyze()