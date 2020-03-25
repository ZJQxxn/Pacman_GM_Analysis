'''
Description:
    Select out data for movement estimation.

uthor:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    2020/3/17
'''

import pandas as pd
import numpy as np
import json


def selectAllNormalData(feature_filename):
    all_data = pd.read_csv(feature_filename)
    # Discard data with scared ghosts
    normal_index = all_data.index[(all_data.ifscared1 == 1) & (all_data.ifscared2 == 1)]
    normal_data = all_data.iloc[normal_index.values, 1:]
    normal_data.to_csv('extracted_data/normal_all_data.csv')
    print('Size of all the normal data:', normal_data.shape)


def selectTJunctionData(feature_filename):
    all_data = pd.read_csv(feature_filename)
    map_info = pd.read_csv("../common_data/map_info_brian.csv")
    map_info['pos'] = map_info['pos'].apply(lambda x: eval(x) if not isinstance(x, float) else np.nan)
    # The position of T-junction
    t_junction_pos = (
        map_info[
            ((map_info.Pos1 == 2) | (map_info.Pos1 == 27))
            & ((map_info.Pos2 <= 12) | (map_info.Pos2 >= 24))
            ].pos.values.tolist()
        + map_info[
            ((map_info.Pos1.between(2, 7)) | (map_info.Pos1.between(22, 27)))
            & (map_info.Pos2.isin([9, 30]))
            ].pos.values.tolist()
    )
    t_junction_data = (
        all_data[
            (all_data.ifscared1 == 1)
            & (all_data.ifscared2 == 1)
            & (all_data.pacmanPos.isin(t_junction_pos))
            #         & (~df_total.index.isin(end_game_reset))  # 去掉end game reset
            ].drop(columns=["Step_x", "Step_y"])
    )
    print()


def selectEndGameData(feature_filename):
    # TODO: check if is correct; reset game?
    all_data = pd.read_csv(feature_filename)
    with open("../common_data/ten_points_pac.json", "r") as f:
        ten_points_pac = json.load(f)
    ten_points_df = _from_dict2df(ten_points_pac)
    # Select end game points
    end_game_data = (
        all_data.reset_index()
            .merge(ten_points_df, on=["index", "file"])
            .set_index("level_0")
    )
    # Select data after restart
    suicide_data = pd.read_csv("../common_data/suicide_point.csv", delimiter=",")
    ss = pd.DataFrame(
        sorted(
            all_data.file.drop_duplicates().values,
            key=lambda x: [x.split("-")[0]] + x.split("-")[2:],
        )
    )
    ss[1] = ss[0].shift(-1)
    restart_index = ss[ss[0].isin(suicide_data.file.values + ".csv")][1].values
    end_game_data[
        (end_game_data.file.isin(restart_index)) & (end_game_data["index"] == 0)
        ].dropna(subset=["next_eat_rwd"])
    end_game_data.to_csv('extracted_data/end_game_data.csv')
    print('Finished extracting end-game data!')


def selectEnergizerData(feature_filename):
    # TODO: needs comments and explanations
    all_data = pd.read_csv(feature_filename)
    all_data = pd.concat(
        [
            all_data,
            all_data.groupby("file") \
                .apply(lambda x: x.next_eat_rwd.mask(~x.eat_energizer).fillna(method="bfill"))\
                .rename("next_eat_energizer")\
                .reset_index()\
                .drop(columns=["file"])\
                .set_index("level_1"),
        ],
        1,
    )
    locs_df = pd.read_csv("../common_data/dij_distance_map.csv")
    locs_df.pos1, locs_df.pos2, locs_df.path = (
        locs_df.pos1.apply(eval),
        locs_df.pos2.apply(eval),
        locs_df.path.apply(eval)
    )
    df_temp = all_data[["next_eat_energizer", "pacmanPos", "file", "index"]].merge(
        locs_df,
        left_on=["next_eat_energizer", "pacmanPos"],
        right_on=["pos2", "pos1"],
        how="left",
    )
    print('After preprocessing data!')
    # Select eat energizer data
    #TODO: float to integer? Why float
    energizer_data = (
        df_temp.groupby(["file", "next_eat_energizer"])
            .apply(
            lambda d: list(
                range(
                    d.dropna(subset=["dis"])
                        .dis.diff()
                        .where(lambda x: x > 0)
                        .dropna()
                        .index.max(),
                        d.dropna(subset=["dis"]).index.max() + 1,
                )
            )
            if not pd.isnull(
                d.dropna(subset=["dis"])
                    .dis.diff()
                    .where(lambda x: x > 0)
                    .dropna()
                    .index.max()
            )
            else list(
                range(
                    d.dropna(subset=["dis"]).index.min(),
                    d.dropna(subset=["dis"]).index.max(),
                )
            )
        )
            .reset_index()
    )
    energizer_data = (
        all_data.reset_index()
            .merge(energizer_data, on=["file", "next_eat_energizer"])
            .set_index("level_0")
    )
    print('After selecting eating energizer data!')
    df_model = energizer_data[
        energizer_data.apply(lambda x: x.name in x[0] and len(x[0]) > 15, 1)
    ]
    print()



def _from_dict2df(ten_points_pac):
    # TODO:need comments
    idx_l, item_l = [], []
    for idx, item in ten_points_pac.items():
        idx_l.extend([idx + ".csv"] * len(item))
        item_l.extend(item)
    return pd.DataFrame({"file": idx_l, "index": item_l})


if __name__ == '__main__':
    all_data_filename = '../common_data/df_total.csv'
    # selectAllNormalData(all_data_filename)
    # selectEndGameData(all_data_filename)
    # selectEnergizerData(all_data_filename)
    # selectTJunctionData(all_data_filename)