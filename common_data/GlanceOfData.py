'''
Description:
    Take a glance about the data.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    20 July 2020
'''

import pandas as pd
import pickle

if __name__ == '__main__':
    with open("status/planned_hunting_status.pkl", "rb") as file:
        data = pickle.load(file)
        # data = data[data.file == "1-1-Omega-15-Jul-2019.csv"]
        data = data[data.at_cross][["pacmanPos", "file", "index", "global_estimation", "local_estimation", "random_estimation", "next_pacman_dir_fill"]]
        # data.to_csv("local_estimation_scan.csv")
        print()