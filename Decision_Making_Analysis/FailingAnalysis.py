'''
Description:
    Check all the time step that our model fails to detect hunting2grazing transferation.

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

testing_index = []
with open('save_m/testing_index.csv', 'r') as file:
    reader  = csv.reader(file)
    for each in reader:
        testing_index.append(int(float(each[0])))

fail_index = []
with open('save_m/fail_testing_index.csv', 'r') as file:
    reader = csv.reader(file)
    for each in reader:
        fail_index.append(int(float(each[0])))

testing_data = np.array(data)[testing_index]
fail_data = testing_data[fail_index]

with open('save_m/H2G_fail_data.csv', 'w') as file:
    writer = csv.writer(file)
    writer.writerows(fail_data)