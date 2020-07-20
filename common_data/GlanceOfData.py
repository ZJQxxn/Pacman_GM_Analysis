'''
Description:
    Take a glance about the data.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    20 July 2020
'''

import h5py
import pandas as pd
import pickle

if __name__ == '__main__':
    with open("labeled_df_toynew.pkl", "rb") as file:
        data = pickle.load(file)
        data = data[data.file == "1-1-Omega-15-Jul-2019.csv"]
        print()