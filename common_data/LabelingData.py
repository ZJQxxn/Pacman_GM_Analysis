'''
Description:
    Labeling the data.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    25 July 2020
'''


import pickle
import pandas as pd
import numpy as np


def _isGlobal():
    pass

def _isLocal():
    pass

def _isEvade():
    pass

def _isSuicide():
    pass

def _isOptimistic():
    pass

def _isPessimistic():
    pass


def labeling(df_total):
    pass




if __name__ == '__main__':
    # Configurations
    data_filename = "labeled_df_toynew.pkl"
    # Read in the complet data
    with open(data_filename, "rb") as file:
        df_total = pickle.load(file)
    print("=" * 20)
    print("Finished reading.")