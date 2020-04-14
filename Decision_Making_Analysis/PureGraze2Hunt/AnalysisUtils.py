'''
Description:
    Some utility functions for the movement estimation.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    Mar. 28 2020
'''
import numpy as np
import pandas as pd
from shapely.geometry import Polygon, Point

locs_df = pd.read_csv("../../common_data/dij_distance_map.csv")
locs_df.pos1, locs_df.pos2, locs_df.path = (
    locs_df.pos1.apply(eval),
    locs_df.pos2.apply(eval),
    locs_df.path.apply(eval)
)

