'''
Description:
    Extract data for moving status analysis.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    Apr. 12 2020
'''

import pandas as pd
import numpy as np

def extractData(feature_filename):
    all_data = pd.read_csv(feature_filename)
    # Find where the Pacman stay
    all_data.pacman_dir = all_data.pacman_dir.shift(-1)


if __name__ == '__main__':
    feature_filename = '../common_data/df_total_new.csv'
    extractData(feature_filename)