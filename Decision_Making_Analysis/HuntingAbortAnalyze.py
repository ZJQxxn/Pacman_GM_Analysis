'''
Description:
    Check all the time step that convert from hunting to grazing. Find out the common characteristics.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/03/13
'''

import numpy as np
import csv
from DataHelper import whichMode

data = []
with open('data/df_total_GM.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        features = [
                    row['index'],
                    row['file'], # File name (different trials)
                    row['status_g'], # Whether grazing (1 for true, 0 for false)
                    row['status_h1'], # Whether hunting ghost 1 (1 for true, 0 for false)
                    row['status_h2'], # Whether hunting ghost 2 (1 for true, 0 for false)
                    row['Step'] # No. of this time step
        ]
        data.append(features)

required_data = []
selection_index = [0,1,5]
for index in range(len(data) - 1):
    this_mode = whichMode(data[index][2], data[index][3], data[index][4])
    next_mode = whichMode(data[index+1][2], data[index+1][3], data[index+1][4])

    if 'escaping' != this_mode[2] \
            and 'escaping' != next_mode[2] \
            and data[index + 1][4] != "0":
        if 'hunting1' == this_mode[3] or 'hunting2' == this_mode[3]:
            if 'grazing' == next_mode[3]:
                required_data.append(np.array(data[index])[selection_index])

with open('hunting_abort_index.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(required_data)