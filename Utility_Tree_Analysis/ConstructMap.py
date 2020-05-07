'''
Description:
    Construct a JSON to store the map direction information.
       
Author: 
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    Apr. 21 2020
'''
import json
import pandas as pd
import numpy as np

# Read in data
map_pos = pd.read_csv("../common_data/map_info_brian.csv")[["Pos1", "Pos2", "iswall"]]
map_pos = map_pos.assign(
    pos = map_pos.apply(lambda x: (x.Pos1, x.Pos2), axis = 1)
)
# map_pos.pos = map_pos.pos.apply(lambda x: eval(x))
map_pos = map_pos.iloc[:-4, :]

# Construct a map using matrix
up_bound = 5
low_bound = 33
left_bound = 2
right_bound = 27
is_wall = map_pos.iswall.values.reshape((low_bound - up_bound + 1, right_bound - left_bound + 1)) # indicate whether a position is wall
positions = map_pos.pos.values.reshape((low_bound - up_bound + 1, right_bound - left_bound + 1)) # the coordinates

# Reset bounds
# right_bound = right_bound - left_bound + 1
# left_bound = 0
# low_bound = low_bound - up_bound + 1
# up_bound = 0
right_bound = right_bound - left_bound + 1
left_bound = 0
low_bound = low_bound - up_bound + 1
up_bound = 0

# Find adjacent position of all the map positions
adjacent_dict = {}
for y in range(positions.shape[0]):
    for x in range(positions.shape[1]):
        cur_pos = positions[y, x]
        cur_is_wall = is_wall[y, x]
        if cur_pos not in adjacent_dict and not cur_is_wall:
            adjacent_dict[cur_pos] = {}
        # adjacent positions
        if not cur_is_wall:
            adjacent_dict[cur_pos]["left"] = tuple(positions[y, x - 1]) if x - 1 >= left_bound and not is_wall[y, x - 1] else None
            adjacent_dict[cur_pos]["right"] = tuple(positions[y, x + 1]) if x + 1 < right_bound and not is_wall[y, x + 1]  else None
            adjacent_dict[cur_pos]["up"] = tuple(positions[y - 1, x]) if y - 1 >= up_bound and not is_wall[y - 1, x]  else None
            adjacent_dict[cur_pos]["down"] = tuple(positions[y + 1, x]) if y + 1 < low_bound and not is_wall[y + 1, x]  else None

# Add the tunnel (connect the leftmost and the rightmost)
adjacent_dict[(1,18)] = {"left":(27,18), "right":(2,18),"up":None, "down":None}
adjacent_dict[(2,18)]["left"]  = (1,18)
adjacent_dict[(27,18)]["right"]  = (1,18)


# str_adjacent_dict = {}
# for key in adjacent_dict.keys():
#     str_adjacent_dict[str(key)] = {}
#     for each in adjacent_dict[key]:
#         if adjacent_dict[key][each] != None:
#             str_adjacent_dict[str(key)][each] = (int(adjacent_dict[key][each][0]), int(adjacent_dict[key][each][1]))
#         else:
#             str_adjacent_dict[str(key)][each] = None
# with open("extracted_data/adjacent_map.json", 'w') as file:
#     json.dump(str_adjacent_dict, file)

# Convert to pd.Dataframe and save
adjacent_df = pd.DataFrame(adjacent_dict.items(), columns = ["pos", "adjacent"])
adjacent_df = adjacent_df.assign(
    left = adjacent_df.adjacent.apply(lambda x : x['left']),
    right = adjacent_df.adjacent.apply(lambda x: x['right']),
    up = adjacent_df.adjacent.apply(lambda x: x['up']),
    down = adjacent_df.adjacent.apply(lambda x: x['down'])
).drop(columns = ['adjacent'])
adjacent_df.to_csv("extracted_data/adjacent_map.csv")