'''
Description:
    Extract global graze data.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 22 2020
'''

import pandas as pd
import numpy as np
import pickle

# Read in the data
all_data = pd.read_csv("../common_data/df_total_new.csv")
all_data[all_data.file == "10-2-Omega-16-Jul-2019.csv"].to_csv("extracted_data/test_data.csv")