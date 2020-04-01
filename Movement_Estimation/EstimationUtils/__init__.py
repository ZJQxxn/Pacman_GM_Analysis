import numpy as np
import networkx as nx
import pandas as pd
from shapely.geometry import Polygon, Point

from .utils import *


# Totally 11 regions
polys = [
    [[2, 5], [2, 12], [7, 12], [7, 5]],
    [[8, 5], [8, 14], [14, 14], [14, 5]],
    [[27, 5], [22, 5], [22, 12], [27, 12]],
    [[21, 5], [15, 5], [15, 14], [21, 14]],
    [[2, 13], [7, 13],[7,15],[9, 15], [9, 23], [2, 23]],
    [[20, 15], [20, 23], [29, 23], [29, 13],[22,13],[22,15]],
    [[10, 15], [10, 23], [19, 23], [19, 15]],
    [[2, 24], [2, 34], [7, 34], [7, 24]],
    [[8, 24], [8, 34], [14, 34], [14, 24]],
    [[27, 24], [22, 24], [22, 34], [27, 34]],
    [[21, 24], [15, 24], [15, 34], [21, 34]]
]
poly_ext = [Polygon(p).buffer(0.001) for p in polys]

# The direction relationship between regions
region_relation = {
    1: {"up": np.nan, "down":[5], "left": np.nan, "right": [2]},
    2: {"up": np.nan, "down":[4, 5], "left": [1], "right": [3]},
    3: {"up": np.nan, "down":[5, 6], "left": [2], "right": [4]},
    4: {"up": np.nan, "down": [6], "left": [23], "right": np.nan},
    5: {"up": [1, 2], "down": [8, 9], "left": np.nan, "right": [2, 6, 9]},
    6: {"up": [2, 3], "down": [9, 10], "left": [5], "right": [7]},
    7: {"up": [3, 4], "down": [10, 11], "left": [3, 6, 10], "right": np.nan},
    8: {"up": [5], "down": np.nan, "left": np.nan, "right": [9]},
    9: {"up": [5, 6], "down": np.nan, "left": [8], "right": [10]},
    10: {"up": [6, 7], "down": np.nan, "left": [9], "right": [11]},
    11: {"up": [7], "down": np.nan, "left": [10], "right": np.nan}
}

def determine_centroid(poly_list, pos):
    '''
    Given a position 2-tuple, determine the centroid for eits region.  
    :param poly_list: A list of "shapely.geometry.Polygon" denotes each region. 
    :param pos: The position with shape of a 2-tuple.
    :return: A list contains the centroid point and region index (1-11).
    '''
    for index, p in enumerate(poly_list):
        if p.contains(Point(pos)):
            return (round(p.centroid.xy[0][-1], 1), round(p.centroid.xy[1][-1], 1))

def determine_region(poly_list, pos):
    '''
    Given a position 2-tuple, determine the region index for the position.  
    :param poly_list: A list of "shapely.geometry.Polygon" denotes each region. 
    :param pos: The position with shape of a 2-tuple.
    :return: A list contains the centroid point and region index (1-11).
    '''
    for index, p in enumerate(poly_list):
        if p.contains(Point(pos)):
            return index + 1


map_info = pd.read_csv("../common_data/map_info_brian.csv")
map_info['pos'] = map_info['pos'].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
map_info["pos_global"] = [
    determine_centroid(poly_ext, pos)for pos in map_info.pos.values
]
map_info["region_index"] = [
    determine_region(poly_ext, pos) for pos in map_info.pos.values
]
# map_info = map_info[["pos", "pos_global", "region_index"]]


# ====================================================================
locs_df = pd.read_csv("../common_data/dij_distance_map.csv")
locs_df.pos1, locs_df.pos2, locs_df.path = (
    locs_df.pos1.apply(eval),
    locs_df.pos2.apply(eval),
    locs_df.path.apply(eval)
)

#======================================================================
d = dict(
    zip(
        map_info.pos,
        list(
            zip(
                *[
                    tuple_list(
                        map_info[
                            ["Next" + str(i) + "Pos1", "Next" + str(i) + "Pos2"]
                        ].values
                    )
                    for i in range(1, 5)
                ]
            )
        ),
    )
)


#======================================================================
G = nx.DiGraph()
G.add_nodes_from(d.keys())
for k, v in d.items():
    G.add_edges_from(([(k, t) for t in v if t != (0, 0)]))


#======================================================================
forbidden_pos = list(map(lambda x: (x, 18), list(range(7)) + list(range(23, 30))))
def global_pos(pos):
    if not isinstance(pos, float) and pos not in forbidden_pos and pos is not None:
        return (
            pd.Series(pd.cut([pos[0]], xedges)).replace(dictx).values[0][0],
            pd.Series(pd.cut([pos[1]], yedges)).replace(dicty).values[0][0],
        )
    else:
        return np.nan