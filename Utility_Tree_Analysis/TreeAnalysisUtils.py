'''
Description:
    Utility functions for the utility tree analysis.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    Apr. 21 2020
'''

import pandas as pd
import numpy as np

adjacent_data = pd.read_csv("extracted_data/adjacent_map.csv")
for each in ['pos', 'left', 'right', 'up', 'down']:
    adjacent_data[[each]] = adjacent_data[[each]].apply(lambda x : eval(x) if not isinstance(x, float) else np.nan)