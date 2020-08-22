'''
Description:
    Planned hunting agent.If the ghosts are scared or no energizer exists, the planned hunting agent degenerates to 
    random agent. Else, when ghosts are normal, the Pacman has chances to plan hunting by reaching out to the energizer 
    intentionally.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    21 Aug. 2020
'''

import numpy as np

class PlannedHuntingAgent:

    def __init__(self, adjacent_data, adjacent_path, locs_df, reward_amount, cur_pos, energizer_data, ghost_data, ghost_status):
        # Game status (energizer)
        self.cur_pos = cur_pos
        self.energizer_data = energizer_data
        self.ghost_data = ghost_data
        self.ghost_status = np.array(ghost_status)
        # Auxiliary data
        self.adjacent_data=  adjacent_data
        self.adjacent_path = adjacent_path
        self.locs_df = locs_df
        self.reward_amount = reward_amount
        # Adjacent position and available direction
        self.adjacent_pos = adjacent_data[self.cur_pos]
        self.available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.adjacent_pos = [self.adjacent_pos[each] for each in self.available_dir]


    def nextDir(self):
        # If ghosts are scared or no energizer exists, degenerate to random agent
        if np.all(self.ghost_status >= 3) or isinstance(self.energizer_data, float):
            # self.is_planned = False
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
            return choice
        # Else, the Pacman has chances to plan hunting
        P_G_distance = []  # distance between adjacent positions of Pacman and ghosts
        P_E_distance = []  # distance between adjacent positions of Pacman and energizers
        for each_adjacent_pos in self.adjacent_pos:
            temp_P_G_distance = []
            temp_P_E_distance = []
            # ghost distance
            for index, each_ghost_pos in enumerate(self.ghost_data):
                each_ghost_pos = tuple(each_ghost_pos)
                if 3 == self.ghost_status[index]: # in case one ghost is dead
                    continue
                if each_ghost_pos in self.locs_df[each_adjacent_pos]:
                    temp_P_G_distance.append(self.locs_df[each_adjacent_pos][each_ghost_pos])
                elif each_ghost_pos == each_adjacent_pos:
                    temp_P_G_distance.append(1.0)
                else:
                    print("Lost path : {} to {}".format(each_adjacent_pos, each_ghost_pos))
            P_G_distance.append(temp_P_G_distance)
            # energizer distance
            for each_energizer_pos in self.energizer_data:
                if each_energizer_pos in self.locs_df[each_adjacent_pos]:
                    temp_P_E_distance.append(self.locs_df[each_adjacent_pos][each_energizer_pos])
                elif each_energizer_pos == each_adjacent_pos:
                    temp_P_E_distance.append(1.0)
                else:
                    print("Lost path : {} to {}".format(each_adjacent_pos, each_energizer_pos))
            P_E_distance.append(temp_P_E_distance)
        P_G_distance = np.array(P_G_distance)
        P_E_distance = np.array(P_E_distance)
        # compute utility of each adjacent positions (i.e., each moving direction)
        dir_utility= []
        for adjacent_index in range(len(self.available_dir)):
            temp_utility = 0.0
            temp_P_G_distance = P_G_distance[adjacent_index]
            temp_P_E_distance = P_E_distance[adjacent_index]
            for P_G in temp_P_G_distance:
                for P_E in temp_P_E_distance:
                    temp_utility += (P_G - P_E)
            dir_utility.append(temp_utility)
        choice = np.argmax(dir_utility).item()
        choice = self.available_dir[choice]
        return choice


if __name__ == '__main__':
    import sys

    sys.path.append('./')
    from TreeAnalysisUtils import readAdjacentMap, readLocDistance, readRewardAmount, readAdjacentPath

    # Read data
    locs_df = readLocDistance("./extracted_data/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    adjacent_path = readAdjacentPath("./extracted_data/dij_distance_map.csv")
    reward_amount = readRewardAmount()
    print("Finished reading auxiliary data!")
    # Planned hunting agent
    cur_pos = (7, 16)
    ghost_data = [(21, 5), (22, 5)]
    ghost_status = [1, 1]
    energizer_data = [(19, 27)]

    agent = PlannedHuntingAgent(
        adjacent_data,
        adjacent_path,
        locs_df,
        reward_amount,
        cur_pos,
        energizer_data,
        ghost_data,
        ghost_status
    )
    choice = agent.nextDir()
    print("Position : ", agent.cur_pos)
    print("Available directions : ", agent.available_dir)
    print("Choice : ", choice)

