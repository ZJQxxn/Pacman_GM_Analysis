'''
Description:
    Some utility functions for the prehunt analysis.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date: 
    Apr. 3 2020
'''
from shapely.geometry import Polygon, Point
import pandas as pd
import numpy as np
import networkx as nx
import skimage.graph


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

# The number of cross in each region
region_cross_num = {
    1: 6,
    2: 8,
    3: 8,
    4: 6,
    5: 3,
    6: 9,
    7: 3,
    8: 6,
    9: 9,
    10: 9,
    11: 6
}

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


def determine_centroid(poly_list, pos):
    for p in poly_list:
        if p.contains(Point(pos)):
            return (p.centroid.xy[0][-1], p.centroid.xy[1][-1])


# Distance between two locations on the map
locs_df = pd.read_csv("../common_data/dij_distance_map.csv")
locs_df = locs_df[["pos1", "pos2", "dis"]]



# The map
#TODO: need explanations
def tuple_list(l):
    return [tuple(a) for a in l]

map_info = pd.read_csv("../common_data/map_info_brian.csv")
map_info['pos'] = map_info['pos'].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
map_info["pos_global"] = [
    determine_centroid(poly_ext, pos) for pos in map_info.pos.values
]
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

G = nx.DiGraph()
G.add_nodes_from(d.keys())
for k, v in d.items():
    G.add_edges_from(([(k, t) for t in v if t != (0, 0)]))

T, F = True, False
array = np.asarray(
    map_info.pivot_table(columns="Pos1", index="Pos2")
    .iswall.reindex(range(map_info.Pos2.max() + 1))
    .replace({1: F, np.nan: F, 0: T})
)
array = np.concatenate((array, np.array([[False] * 30])))
costs = np.where(array, 1, 1000)


def switch(start, end):
    return start[::-1], end[::-1]

def dijkstra_distance(start, end):
    global costs
    start, end = switch(start, end)
    path, cost = skimage.graph.route_through_array(
        costs, start, end, fully_connected=False
    )
    path = [i[::-1] for i in path]
    return path

# Cross position
#TODO: what is cross?
cross_pos = map_info[map_info.NextNum >= 3].pos.values
cross_pos = list(
    set(cross_pos)
    - set(
        [
            i
            for i in cross_pos
            if i[0] >= 11 and i[0] <= 18 and i[1] >= 16 and i[1] <= 20
        ]
    )
)
