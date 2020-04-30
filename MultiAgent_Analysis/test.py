'''
Description:
    Test the multi-agent interactor.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    Apr. 30 2020
'''
import pickle
import numpy as np
from MultiAgentInteractor import MultiAgentInteractor

# Read testing data
with open("extracted_data/test_data.pkl", 'rb') as file:
    all_data = pickle.load(file)
# Construct the multi-agent interactor
multiagent = MultiAgentInteractor(agent_weight=[0.4, 0.3, 0.2, 0.1])
# Estimate moving directions for every time step
for index in range(15):
    # Extract game status and Pacman status
    each = all_data.iloc[index]
    cur_pos = each.pacmanPos #
    energizer_data = each.energizers
    bean_data = each.beans
    ghost_data = np.array([each.distance1, each.distance2])
    ghost_status = each[["ifscared1", "ifscared2"]].values
    reward_type = int(each.Reward)
    fruit_pos = each.fruitPos
    # Pass game status to the agent
    multiagent.resetStatus(cur_pos, energizer_data, bean_data, ghost_data, reward_type, fruit_pos, ghost_status)
    # Estimation
    dir_prob = multiagent.estimateDir()
    # Print out the estimation
    cur_dir = multiagent.dir_list[np.argmax(dir_prob)]
    if "left" == cur_dir:
        next_pos = [cur_pos[0] - 1, cur_pos[1]]
    elif "right" == cur_dir:
        next_pos = [cur_pos[0] + 1, cur_pos[1]]
    elif "up" == cur_dir:
        next_pos = [cur_pos[0], cur_pos[1] - 1]
    else:
        next_pos = [cur_pos[0], cur_pos[1] + 1]
    print(cur_dir, next_pos, dir_prob)
