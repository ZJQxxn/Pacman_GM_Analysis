## 1. suicide_point.csv, ten_points_pac.json

### 这两个文件在提取 end-game 数据的时候会用到。这两个文件内数据的具体作用是什么？
提取end-game数据的相关代码如下：

```python
    all_data = pd.read_csv('df_total.csv')
    with open("ten_points_pac.json", "r") as f:
        ten_points_pac = json.load(f)
    ten_points_df = _from_dict2df(ten_points_pac)
    # Select end game points
    end_game_data = (
        all_data.reset_index()
            .merge(ten_points_df, on=["index", "file"])
            .set_index("level_0")
    )
    # Select data after restart
    suicide_data = pd.read_csv("suicide_point.csv", delimiter=",")
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
    end_game_data.to_csv('end_game_data.csv')
```

## 2. No T-junction data

### 没有满足T-junction条件的数据
提取T-junction数据的相关代码如下：

```python
all_data = pd.read_csv('df_total.csv')
    map_info = pd.read_csv("map_info_brian.csv")
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
```
用这段代码提取出的 t_junction_data 是空的DataFrame。