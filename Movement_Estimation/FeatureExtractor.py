''''
Description:
    Utility tools for the movement estimationa. 

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/19
'''
import numpy as np
import pandas as pd
import sys

sys.path.append('./')
from EstimationUtils import *


# =====================================
#          Local Features
# =====================================
def localBeanNum(all_data):
    # Find all the paths with length of 5
    path_data = all_data[["file", "index", "pacmanPos", "beans"]].merge(
        locs_df.loc[locs_df.dis == 5, ["pos1", "path"]],
        left_on="pacmanPos",
        right_on="pos1",
        how="left",
    )
    # Construct path with length no more than 5
    path_data = (
        path_data.assign(
            pos2 = path_data.path.apply(lambda x: x[0][1]),  # TODO: why not [0][-1]?
            pos1 = path_data.path.apply(lambda x: x[0][0]),
        )
            .merge(
            locs_df[["pos2", "pos1", "relative_dir"]].rename(
                columns={"relative_dir": "local_feature_dir"}
            ),
            on=["pos2", "pos1"],
            how="left",
        )
            .drop(columns=["pos1", "pos2"])
    )
    # Find the number of beans on each path
    path_data["beans_in_local"] = path_data.apply(
        lambda x: sum([len(set(p) & set(x.beans)) / 5 for p in x.path]), 1
    )
    # Compute the number of beans for each direction
    local_bean_num = (
        path_data.groupby(["file", "index", "pacmanPos", "local_feature_dir"])
        .apply(lambda x: x.beans_in_local.sum() / x.path.map(len).sum())
        .rename("beans_in_local")
        .reset_index()
        .pivot_table(
            index="local_feature_dir",
            columns=["file", "index", "pacmanPos"],
            values="beans_in_local",
            aggfunc="max" # maximal number of beans on paths of each direction
        ).T.fillna(-1)
    )
    local_bean_num.columns = ["local_bean_num_" + each.strip("['").strip("']") for each in local_bean_num.columns.values]
    # Match with the index
    all_data[["pacmanPos", "index", "file"]].merge(
        local_bean_num.reset_index(), on=["pacmanPos", "index", "file"], how="left"
    )
    return local_bean_num


def localGhostDir(all_data):
    # The direction of ghost 1 with respect to the Pacman
    ghost1_dir_wrt_pacman = (
        all_data[["file", "index", "pacmanPos", "ghost1Pos"]].merge(
        locs_df[["pos1", "pos2", "relative_dir"]],
        left_on = ["pacmanPos", "ghost1Pos"],
        right_on = ["pos1", "pos2"],
        how = "left"
        )
            .drop(columns = ["pos1", "pos2", "pacmanPos", "ghost1Pos"])
            .rename(columns = {"relative_dir":"local_ghost1_dir"})
    )
    ghost1_dir_wrt_pacman.local_ghost1_dir.apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    # The direction of ghost 2 with respect to the Pacman
    ghost2_dir_wrt_pacman = (
        all_data[["file", "index", "pacmanPos", "ghost2Pos"]].merge(
            locs_df[["pos1", "pos2", "relative_dir"]],
            left_on=["pacmanPos", "ghost2Pos"],
            right_on=["pos1", "pos2"],
            how="left"
        )
            .drop(columns=["pos1", "pos2", "pacmanPos", "ghost2Pos"])
            .rename(columns={"relative_dir": "local_ghost2_dir"})
    )
    ghost2_dir_wrt_pacman.local_ghost2_dir.apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    # Combine into one dataframe
    ghosts_dir = (
        ghost1_dir_wrt_pacman.merge(
            ghost2_dir_wrt_pacman,
            on = ["file", "index"],
            how = "left"
        )
    )
    return ghosts_dir


def localEnergizerDir(all_data):
    # Find the positions that are located 5 steps away from the Pacman
    path_data = all_data[["file", "index", "pacmanPos", "energizers"]].merge(
        locs_df.loc[locs_df.dis <= 5, ["pos1", "pos2", "relative_dir", "dis"]],
        left_on="pacmanPos",
        right_on="pos1",
        how="left"
    ).drop(columns = ["pos1"]).rename(columns = {"pos2": "near_pos"})
    # Determin whethr the near position is an energizer
    all_near_energizer = path_data.assign(
        is_energizer=path_data.apply(
            lambda x: x.near_pos in x.energizers if not isinstance(x.energizers, float) else False,
            axis=1)
    )
    all_near_energizer = (
        all_near_energizer.loc[all_near_energizer.is_energizer == True, :]
            .drop(columns = ["energizers", "near_pos", "is_energizer"])
    )
    # Find out the direction of the nearest energizer
    nearest_energizer = (
        all_near_energizer.groupby(["file", "index", "pacmanPos"])
            .apply(lambda x: x.relative_dir.iloc[np.argmin(x.dis.values)])
            .reset_index()
    ).rename(columns = {0:"local_nearest_energizer_dir"})
    energizer_dir = all_data[["file", "index"]].merge(
        nearest_energizer,
        on = ["file", "index"],
        how = "left"
    )
    return energizer_dir


# =====================================
#          Global Features
# =====================================
def _countBeans(each_data, region_info):
    each_data.left_count = 0 if not isinstance(each_data.left_region, float) else -1
    each_data.right_count = 0 if not isinstance(each_data.right_region, float) else -1
    each_data.up_count = 0 if not isinstance(each_data.up_region, float) else -1
    each_data.down_count = 0 if not isinstance(each_data.down_region, float) else -1
    for bean in each_data.beans:
        # TODO: we need to compute the size of each region (length of steps)
        bean_region = int(region_info[region_info.pos == bean].region_index.values.item())
        # Determine at which region direction the bean locates
        if not isinstance(each_data.left_region, float) and bean_region in each_data.left_region:
            each_data.left_count += 1
        elif not isinstance(each_data.right_region, float) and bean_region in each_data.right_region:
            each_data.right_count += 1
        elif not isinstance(each_data.up_region, float) and bean_region in each_data.up_region:
            each_data.up_count += 1
        elif not isinstance(each_data.down_region, float) and bean_region in each_data.down_region:
            each_data.down_count += 1
        else:
            # The same region; not considered for now.
            pass
    return  each_data


def globalBeanNum(all_data):
    # Find all the relative regions of the Pacman
    # all_data = all_data.iloc[:500,:] #TODO: for saving time
    region_info = map_info[["pos", "region_index"]]
    pacman_region = all_data[["file", "index", "pacmanPos", "beans"]].merge(
        region_info[["pos", "region_index"]],
        left_on="pacmanPos",
        right_on="pos",
        how="left",
    ).drop(columns = ["pos"]).rename(columns = {"region_index": "pacman_region"})
    pacman_region = pacman_region.assign(
        # The near regions
        left_region = pacman_region.pacman_region.apply(lambda x: region_relation[x]["left"] if not pd.isna(x) else np.nan),
        right_region=pacman_region.pacman_region.apply(lambda x: region_relation[x]["right"] if not pd.isna(x) else np.nan),
        up_region=pacman_region.pacman_region.apply(lambda x: region_relation[x]["up"] if not pd.isna(x) else np.nan),
        down_region=pacman_region.pacman_region.apply(lambda x: region_relation[x]["down"] if not pd.isna(x) else np.nan),
        # The number of beans in each near region
        left_count = np.nan,
        right_count = np.nan,
        up_count = np.nan,
        down_count = np.nan
    )
    # Count the number of beans in the near regions
    print("Counting the number of beans in near regions...")
    near_region_bean_count = pacman_region.apply(lambda x: _countBeans(x, region_info), axis = 1)
    near_region_bean_count = near_region_bean_count[['file', "index", "pacmanPos","pacman_region",
                                                     "left_count", "right_count", "up_count", "down_count"]]
    return near_region_bean_count


def globalGhostDir(all_data):
    # Determine the region of ghosts
    region_info = map_info[["pos", "region_index", "pos_global"]]
    ghost_1_region = (
        all_data[['file', "index", "ghost1Pos", "pacmanPos"]].merge(
            region_info,
            left_on = ["ghost1Pos"],
            right_on = ["pos"],
            how = "left"
        )
            .drop(columns = ["ghost1Pos", "pos"])
            .rename(columns = {"region_index": "ghost1_region_index", "pos_global": "ghost1_pos_global"})
    )
    ghost_2_region = (
        all_data[['file', "index", "ghost2Pos", "pacmanPos"]].merge(
            region_info,
            left_on=["ghost2Pos"],
            right_on=["pos"],
            how="left"
        )
            .drop(columns = ["ghost2Pos", "pos"])
            .rename(columns = {"region_index": "ghost2_region_index", "pos_global": "ghost2_pos_global"})
    )
    # Determine the region of the Pacman
    pacman_region = (
        all_data[['file', "index", "pacmanPos"]].merge(
            region_info,
            left_on=["pacmanPos"],
            right_on=["pos"],
            how="left"
        )
            .drop(columns = ["pos"])
            .rename(columns = {"region_index": "pacman_region_index", "pos_global": "pacman_pos_global"})
    )
    # Compute the direction of two ghost regions respect to the Pacman region
    #TODO: use the exact Pacman position instread of region position?
    integrate_data = pacman_region.merge(
        ghost_1_region,
        on = ["file", "index", "pacmanPos"],
        how = "left"
    ).merge(
        ghost_2_region,
        on=["file", "index", "pacmanPos"],
        how="left"
    )
    ghosts_dir = integrate_data.assign(
        ghost1_global_dir = integrate_data[["pacman_pos_global", "ghost1_pos_global"]].apply(
            lambda x: relative_dir(x.ghost1_pos_global, x.pacman_pos_global)
            if not pd.isna(x.ghost1_pos_global) and not pd.isna(x.pacman_pos_global) else np.nan,
            axis = 1),
        ghost2_global_dir=integrate_data[["pacman_pos_global", "ghost2_pos_global"]].apply(
            lambda x: relative_dir(x.ghost2_pos_global, x.pacman_pos_global)
            if not pd.isna(x.ghost2_pos_global) and not pd.isna(x.pacman_pos_global) else np.nan,
            axis=1)
    )[["file", "index", "pacmanPos", "ghost1_global_dir", "ghost2_global_dir"]]
    return ghosts_dir


def globalNearestEnergizerDir(all_data):
    # Find the nearest energizer
    path_data = all_data[["file", "index", "pacmanPos", "energizers"]].merge(
        locs_df[["pos1", "pos2", "dis"]],
        left_on="pacmanPos",
        right_on="pos1",
        how="left"
    ).drop(columns=["pos1"]).rename(columns={"pos2": "destination_pos"})
    # Take out distance of energizers
    all_energizers = path_data.assign(
        is_energizer=path_data.apply(
            lambda x: x.destination_pos in x.energizers if not isinstance(x.energizers, float) else False,
            axis=1)
    )
    all_energizers = (
        all_energizers.loc[all_energizers.is_energizer == True, :]
            .drop(columns=["energizers", "is_energizer"])
            .rename(columns = {"destination_pos": "energizer_pos"})
    )
    # Find out the nearest energizer
    nearest_energizer = (
        all_energizers.groupby(["file", "index", "pacmanPos"])
            .apply(lambda x: x.energizer_pos.iloc[np.argmin(x.dis.values)])
            .reset_index()
    ).rename(columns={0: "nearest_energizer_pos"})
    # Determine the region of the nearest energizer
    region_info = map_info[["pos", "region_index", "pos_global"]]
    nearest_energizer_region_pos = (
        nearest_energizer.merge(
            region_info,
            left_on = ["nearest_energizer_pos"],
            right_on = ["pos"],
            how = "left"
        )
            .drop(columns = ["pos", "region_index", "nearest_energizer_pos"])
            .rename(columns = {"pos_global": "nearest_energizer_region_pos"})
    )
    # Determine the region of the Pacman
    pacman_region_pos = (
        nearest_energizer.merge(
            region_info,
            left_on=["pacmanPos"],
            right_on=["pos"],
            how="left"
        )
            .drop(columns=["pos", "region_index", "nearest_energizer_pos"])
            .rename(columns={"pos_global": "pacman_region_pos"})
    )
    # Compute the direction of the region of nearest energizer
    integrate_data = (
        nearest_energizer_region_pos.merge(
            pacman_region_pos,
            on=["file", "index", "pacmanPos"],
            how="left"

        )[["file", "index", "pacman_region_pos", "nearest_energizer_region_pos"]]
    )
    nearest_energizer_dir = (
        integrate_data.assign(
            global_energizer_dir = integrate_data.apply(
                lambda x: relative_dir(x.nearest_energizer_region_pos, x.pacman_region_pos)
                if not pd.isna(x.nearest_energizer_region_pos) and not pd.isna(x.pacman_region_pos) else np.nan,
                axis = 1
            )
        )[["file", "index", "global_energizer_dir"]]
    )
    nearest_energizer_dir = all_data[["file", "index"]].merge(
        nearest_energizer_dir,
        on = ["file", "index"],
        how = "left"
    )
    return nearest_energizer_dir


def globalAllEnergizerDir(all_data):
    #TODO: how to represent all the energizers
    # Determin the region of all the energizers
    region_info = map_info[["pos", "region_index", "pos_global"]]
    # Determine the energizer direction


# =====================================
#         Read and Eval Data
# =====================================
def readData(filename):
    data = pd.read_csv(filename)
    for c in [
        "ghost1Pos",
        "ghost2Pos",
        "pacmanPos",
        "previousPos",
        "possible_dirs",
        "before_last",
        "after_first",
        "pos",
        "next_eat_rwd",
        "nearbean_dir",
        "energizers",
        "nearrwdPos",
        "ghost1_wrt_pacman",
        "beans"
    ]:
        data[c] = data[c].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    return data


# =====================================
# Extract Features for Different Conditions
# =====================================
def extractNormalFeatures(feature_file):
    print("=" * 20, "ALL NORMAL", "=" * 20)
    # Extract features for all normal data
    print('Start reading all normal data...')
    all_normal_data = readData(feature_file)
    print('Size of the data', all_normal_data.shape)
    # Extract features for all normal data
    # # Extract local features
    local_bean_num = localBeanNum(all_normal_data)
    print("Finished bean num...")
    local_ghost_dir = localGhostDir(all_normal_data)
    print("Finished ghost dir...")
    local_energizer_dir = localEnergizerDir(all_normal_data)
    print("Finished energizer dir...")
    all_local_feature = local_bean_num.merge(
        local_ghost_dir,
        on = ["file", "index"],
        how = "left"
    ).merge(
        local_energizer_dir,
        on = ["file", "index"]
    )
    all_local_feature.to_csv('./extracted_data/normal_local_features.csv')
    print("Finished saving local features!")
    # Extract global features
    global_bean_num = globalBeanNum(all_normal_data)
    print("Finished bean num...")
    global_ghost_dir = globalGhostDir(all_normal_data)
    print("Finished ghost dir...")
    global_energizer_dir = globalNearestEnergizerDir(all_normal_data)
    print("Finished energizer dir...")
    all_global_feature = global_bean_num.merge(
        global_ghost_dir,
        on=["file", "index"],
        how="left"
    ).merge(
        global_energizer_dir,
        on=["file", "index"]
    )
    all_global_feature.to_csv('./extracted_data/normal_global_features.csv')
    print("Finished saving global features!")


def extractEndGameFeatures(feature_file):
    print("=" * 20, "END GAME", "=" * 20)
    # Extract features for end-game data
    print('Start reading end-game data...')
    end_game_data = readData(feature_file)
    print('Size of the data', end_game_data.shape)
    # Extract local features
    local_bean_num = localBeanNum(end_game_data)
    print("Finished bean num...")
    local_ghost_dir = localGhostDir(end_game_data)
    print("Finished ghost dir...")
    #TODO: no energizers
    # local_energizer_dir = localEnergizerDir(end_game_data)
    # print("Finished energizer dir...")
    all_local_feature = local_bean_num.merge(
        local_ghost_dir,
        on=["file", "index"],
        how="left"
    )
    #     .merge(
    #     local_energizer_dir,
    #     on=["file", "index"]
    # )
    all_local_feature.to_csv('./extracted_data/end_game_local_features.csv')
    print("Finished saving local features!")
    # Extract global features
    global_bean_num = globalBeanNum(end_game_data)
    print("Finished bean num...")
    global_ghost_dir = globalGhostDir(end_game_data)
    print("Finished ghost dir...")
    # global_energizer_dir = globalNearestEnergizerDir(end_game_data)
    # print("Finished energizer dir...")
    all_global_feature = global_bean_num.merge(
        global_ghost_dir,
        on=["file", "index"],
        how="left"
    )
    #     .merge(
    #     global_energizer_dir,
    #     on=["file", "index"]
    # )
    all_global_feature.to_csv('./extracted_data/end_game_global_features.csv')


def extractTJunctionFeatures(feature_file):
    print("=" * 20, "T JUNCTION", "=" * 20)
    # Extract features for end-game data
    print('Start reading T-junction data...')
    T_junction_data = readData(feature_file)
    print('Size of the data', T_junction_data.shape)
    # # Extract local features
    local_bean_num = localBeanNum(T_junction_data)
    print("Finished bean num...")
    local_ghost_dir = localGhostDir(T_junction_data)
    print("Finished ghost dir...")
    local_energizer_dir = localEnergizerDir(T_junction_data)
    print("Finished energizer dir...")
    all_local_feature = local_bean_num.merge(
        local_ghost_dir,
        on=["file", "index"],
        how="left"
    ).merge(
        local_energizer_dir,
        on=["file", "index"]
    )
    all_local_feature.to_csv('./extracted_data/T_junction_local_features.csv')
    print("Finished saving local features!")
    # Extract global features
    global_bean_num = globalBeanNum(T_junction_data)
    print("Finished bean num...")
    global_ghost_dir = globalGhostDir(T_junction_data)
    print("Finished ghost dir...")
    global_energizer_dir = globalNearestEnergizerDir(T_junction_data)
    print("Finished energizer dir...")
    all_global_feature = global_bean_num.merge(
        global_ghost_dir,
        on=["file", "index"],
        how="left"
    ).merge(
        global_energizer_dir,
        on=["file", "index"]
    )
    all_global_feature.to_csv('./extracted_data/T_junction_global_features.csv')


def extractEnergizerFeatures(feature_file):
    print("=" * 20, "ENERGIZER", "=" * 20)
    # Extract features for end-game data
    print('Start reading energizer data...')
    energizer_data = readData(feature_file)
    print('Size of the data', energizer_data.shape)
    # # Extract local features
    local_bean_num = localBeanNum(energizer_data)
    print("Finished bean num...")
    local_ghost_dir = localGhostDir(energizer_data)
    print("Finished ghost dir...")
    local_energizer_dir = localEnergizerDir(energizer_data)
    print("Finished energizer dir...")
    all_local_feature = local_bean_num.merge(
        local_ghost_dir,
        on=["file", "index"],
        how="left"
    ).merge(
        local_energizer_dir,
        on=["file", "index"]
    )
    all_local_feature.to_csv('./extracted_data/energizer_local_features.csv')
    print("Finished saving local features!")
    # Extract global features
    global_bean_num = globalBeanNum(energizer_data)
    print("Finished bean num...")
    global_ghost_dir = globalGhostDir(energizer_data)
    print("Finished ghost dir...")
    global_energizer_dir = globalNearestEnergizerDir(energizer_data)
    print("Finished energizer dir...")
    all_global_feature = global_bean_num.merge(
        global_ghost_dir,
        on=["file", "index"],
        how="left"
    ).merge(
        global_energizer_dir,
        on=["file", "index"]
    )
    all_global_feature.to_csv('./extracted_data/energizer_global_features.csv')



# MAIN FUNCTION
if __name__ == '__main__':
    print("Finished all the initialization!")
    # extractNormalFeatures('./extracted_data/normal_all_data.csv')
    # extractEndGameFeatures('./extracted_data/end_game_data.csv')
    # extractTJunctionFeatures('./extracted_data/T_junction_data.csv')
    # extractEnergizerFeatures('./extracted_data/energizer_data.csv')