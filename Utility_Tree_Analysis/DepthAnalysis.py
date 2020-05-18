'''
Description:
    Analyze the behaviors of utility tree with different depth.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    May. 9 2020
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json
import graphviz

from PathTreeConstructor import PathTree
from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount
# ===========================================
#              FUNCTIOMS
# ===========================================

def t2p(tile_pos):
    '''
    Convert tile position to the heatmap pixel position.
    :param tile_pos: Tile position of 2-tuple.
    :return: Heatmap position of 2-tuple.
    '''
    return [tile_pos[0]+0.5, tile_pos[1]+0.5] if len(tile_pos) > 0 else []


# ===========================================
#          PRECOMPUTED VARIABLES
# ===========================================

# Read map data
map_info = pd.read_csv("../common_data/map_info_brian.csv")
map_info['pos'] = map_info['pos'].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
# Formalize map data
T, F = True, False
array = np.asarray(
    map_info.pivot_table(columns="Pos1", index="Pos2")
    .iswall.reindex(range(map_info.Pos2.max() + 1))
    .replace({1: F, np.nan: F, 0: T})
)
array = np.concatenate((array, np.array([[False] * 30])))


# ===========================================
#             SET GAME STATUS
#
#   game_status_1 : many beans in local and an energizer far away; normal ghosts
#   game_status_2: many beans in local and two energizers far away; one energizer can catch ghosts; normal ghosts
#   game_status_3: many beans in local and one energizers far away; scared ghosts
# ===========================================

with open("game_status_3.json", 'r') as file:
    game_status = json.load(file)
adjacent_data = readAdjacentMap("extracted_data/adjacent_map.csv")
locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
reward_amount = readRewardAmount()
print("Finished reading data.")

status = "scared"
for depth in [5, 15, 20]:
    # Reset game status
    pacman_pos = tuple(game_status["pacman_pos"])
    ghost_pos = [tuple(each) for each in game_status["ghost_pos"]]
    ghost_status = game_status["ghost_status"]
    energizer_pos = [tuple(each) for each in game_status["energizer_pos"]]
    bean_pos = [tuple(each) for each in game_status["bean_pos"]]
    # ===========================================
    #             PATH ESTIMATION
    # ===========================================
    path_tree = PathTree(
        adjacent_data,
        locs_df,
        reward_amount,
        pacman_pos,
        energizer_pos.copy(),
        bean_pos.copy(),
        ghost_pos,
        np.nan,
        np.nan,
        ghost_status,
        depth=depth,
        ghost_attractive_thr=20,
        ghost_repulsive_thr=20,
        fruit_attractive_thr=20
    )
    root, highest_utility, best_path = path_tree.construct()
    print("\n Estimated path:", best_path)
    best_path = [t2p(each[0]) for each in best_path]

    # from anytree.exporter import DotExporter
    # dot_file = DotExporter(
    #     root
    # ).to_dotfile("tree.dot")
    # ===========================================
    #             CONVERT POSITION
    # ===========================================
    pacman_pos = t2p(pacman_pos)
    ghost_pos = np.array([t2p(each) for each in ghost_pos])
    energizer_pos = np.array([t2p(each) for each in energizer_pos])
    bean_pos = np.array([t2p(each) for each in bean_pos])

    # ===========================================
    #               PLOTTING
    # ===========================================
    plt.figure(figsize=(10, 10))
    plt.title("Estimated Trajectory (depth = {}, {} ghosts)".format(depth, status), fontsize=20)
    # plt.title("Game Map", fontsize = 20)
    # fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(array, cbar=False, cmap="bone", linewidth=0.1)

    # Plot the Pacman
    plt.scatter(
        pacman_pos[0], pacman_pos[1], color="green", marker="^", s=210, label="Pacman"
    )
    # Plot ghosts
    if "normal" == status:
        plt.scatter(
            ghost_pos[0, 0], ghost_pos[0, 1], color="yellow", marker="s", s=210, label="yellow ghost"
        )
        plt.scatter(
            ghost_pos[1, 0], ghost_pos[1, 1], color="red", marker="s", s=210, label="red ghost"
        )
    else:
        plt.scatter(
            ghost_pos[0][0], ghost_pos[0][1], color="blue", marker="s", s=210, label="scared ghost"
        )
    # Plot energizers
    plt.scatter(
        energizer_pos[:, 0], energizer_pos[:, 1], color="purple", marker="o", s=230, label="energizer"
    )
    # Plot beans
    plt.scatter(
        bean_pos[:, 0], bean_pos[:, 1], color="black", marker="o", s=100, label="bean"
    )
    # Plot the estimated path
    hunt_flag = False
    best_path.insert(0, tuple(pacman_pos))
    for index in range(len(best_path) - 1):
        cur_pos = best_path[index]
        next_pos = best_path[index + 1]
        # Pacman is grazing
        if tuple(cur_pos) in [tuple(each) for each in energizer_pos] or "scared" == status:
            hunt_flag = True

        if hunt_flag:
            plt.arrow(
                cur_pos[0],
                cur_pos[1],
                next_pos[0] - cur_pos[0],
                next_pos[1] - cur_pos[1],
                linewidth=1.5,
                head_width=0.3,
                head_length=0.3,
                fc="purple",
                ec="purple"
            )
        else:
            plt.arrow(
                cur_pos[0],
                cur_pos[1],
                next_pos[0] - cur_pos[0],
                next_pos[1] - cur_pos[1],
                linewidth=1.5,
                head_width=0.3,
                head_length=0.3,
                fc="orange",
                ec="orange"
            )

    plt.legend(fontsize=12, ncol=2)
    plt.savefig("estimated_path_depth={}_ghost_{}.pdf".format(depth, status))
    # plt.savefig("game_status_ghost_{}.pdf".format(status))
    # plt.show()
