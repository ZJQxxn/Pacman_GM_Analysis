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
with open("../common_data/df_total_with_reward.pkl", 'rb') as file:
    all_data = pickle.load(file)

with open("extracted_data/test_data.pkl", 'wb') as file:
    pickle.dump(all_data[all_data.file == "10-1-Omega-16-Jul-2019.csv"], file, pickle.HIGHEST_PROTOCOL)
