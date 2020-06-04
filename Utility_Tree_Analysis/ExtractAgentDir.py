import pandas as pd
import numpy as np
from MultiAgentInteractor import MultiAgentInteractor
from TreeAnalysisUtils import readLocDistance
import pickle

locs_df = readLocDistance("extracted_data/dij_distance_map.csv")
all_data = pd.read_csv("diary.csv")
multiagent = MultiAgentInteractor("config.json")

data_df = pd.DataFrame(
    columns=["global_step", "pacman_pos", "global_dir", "local_dir", "lazy_dir", "random_dir", "integrate_dir"]
)
# Simulated multi-agent
for index in range(all_data.shape[0]):
    each = all_data.iloc[index]
    cur_pos = eval(each.pacmanPos)
    energizer_data = eval(each.energizers)
    bean_data = eval(each.beans)
    # ghost_data = np.array([
    #     locs_df[cur_pos][eval(each.ghost1_pos)],
    #     locs_df[cur_pos][eval(each.ghost1_pos)]
    # ])
    ghost_data = np.array([
        eval(each.ghost1_pos),
        eval(each.ghost1_pos)
    ])
    ghost_status = each[["ghost1_status", "ghost2_status"]].values
    reward_type = int(each.fruit_type) if not np.isnan(each.fruit_type) else np.nan
    fruit_pos = eval(each.fruit_pos) if not isinstance(each.fruit_pos, float) else np.nan
    multiagent.resetStatus(
        cur_pos,
        energizer_data,
        bean_data,
        ghost_data,
        reward_type,
        fruit_pos,
        ghost_status
    )
    dir_prob, agent_estimation = multiagent.estimateDir()
    cur_dir = multiagent.dir_list[np.argmax(dir_prob)]
    multiagent.last_dir = cur_dir
    data_df.loc[index] = [
        index,
        cur_pos,
        "[{}]".format(','.join([str(s) for s in agent_estimation[0]])),
        "[{}]".format(','.join([str(s) for s in agent_estimation[1]])),
        "[{}]".format(','.join([str(s) for s in agent_estimation[2]])),
        "[{}]".format(','.join([str(s) for s in agent_estimation[3]])),
        "[{}]".format(','.join([str(s) for s in dir_prob]))
    ]
    if index % 1000 == 0:
        print("="*10, index, "="*10)


data_df.to_pickle("agent_dir.pkl")
