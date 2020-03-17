'''
Description:
    Select out data for movement estimation.

uthor:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/17
'''

import pandas as pd


def selectAllNormalData(feature_filename):
    all_data = pd.read_csv(feature_filename)
    # Discard data with scared ghosts ()
    normal_index = all_data.index[(all_data.status_h1 == 0) & (all_data.status_h2 == 0)]
    normal_data = all_data.iloc[normal_index.values, 1:]
    normal_data.to_csv('extracted_data/normal_all_data.csv')
    print('Size of all the normal data:', normal_data.shape)

def selectTJuntionData(feature_filename):
    pass

def selectEndGamaData(feature_filename):
    pass

def selectEnergizerData(feature_filename):
    pass


if __name__ == '__main__':
    all_data_filename = '../common_data/df_total_GM.csv'
    selectAllNormalData(all_data_filename)