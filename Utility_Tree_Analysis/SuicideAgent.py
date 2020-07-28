'''
Description:
    Suicide agent.

Author:
    Jiaqi Zhang <zjqseu@gmail.com>

Date:
    25 July 2020
'''

import numpy as np


class SuicideAgent:

    def __init__(self, adjacent_data, adjacent_path, locs_df, cur_pos, ghost_pos, ghost_status, reward_pos, last_dir, random_seed = None):
        '''
        Initialization of suicide agent.
        :param adjacent_data: The adjacent tiles of each tile in the map. Should be a data with the type pf pandas.DataFrame.
        :param adjacent_path: 
        :param locs_df: 
        :param cur_pos: The current position of Pacman, should be a 2-tuple.
        :param ghost_pos: 
        :param ghost_status: 4/5 for scared ghosts, 3 for dead ghosts, and others for normal ghosts
        :param reward_pos: 
        :param last_dir: The moving direction of the last time step, should be a string from {``left'', ``right'', ``up'',`down''}.
        :param random_seed: The random seed.
        '''
        # Initialization
        np.random.seed(random_seed)
        self.adjaccent_path = adjacent_path # path from one location to another
        self.locs_df = locs_df # distance between locations
        self.cur_pos = cur_pos
        self.ghost_pos = ghost_pos
        self.ghost_status = ghost_status
        self.reward_pos = reward_pos
        self.reborn_pos = (14, 27) #TODO: specify the reborn position
        # Obtain available directions from the current location
        self.adjacent_pos = adjacent_data[self.cur_pos]
        self.available_dir = []
        for dir in ["left", "right", "up", "down"]:
            if None != self.adjacent_pos[dir] and not isinstance(self.adjacent_pos[dir], float):
                self.available_dir.append(dir)
        if 0 == len(self.available_dir) or 1 == len(self.available_dir):
            raise ValueError("The position {} has {} adjacent positions.".format(self.cur_pos, len(self.available_dir)))
        self.last_dir = last_dir  # moving direction for the last time step
        self.dir_list = ['left', 'right', 'up', 'down']
        # opposite direction; to evade from ghosts
        self.opposite_dir = {"left": "right", "right": "left", "up": "down", "down": "up"}


    def _relativeDir(self, cur_pos, destination):
        '''
        Determine the relative direction of the adjacent destination given the current location of Pacman.
        :param cur_pos: Current position of Pacman.
        :param destination: Location of destination.
        :return: Relative direction. "left"/"right"/"up"/"down"/None. Heere, None denoting that two positions are the same.
        '''
        if cur_pos[0] < destination[0]:
            return "right"
        elif cur_pos[0] > destination[0]:
            return "left"
        elif cur_pos[1] > destination[1]:
            return "up"
        elif cur_pos[1] > destination[1]:
            return "down"
        else:
            return None


    def nextDir(self):
        '''
        Estimate the moving direction. 
        :return: The moving direction {`left'', ``right'', ``up'', ``down''}.
        '''
        #TODO: if suicide, run to the closest ghost; If evade, run to the opposite directions of ghosts,
        #TODO: if not avaialble, randomly choose a direction
        is_scared = False
        is_suicide = False
        # Do not go back
        if self.last_dir is not None:
            if self.opposite_dir[self.last_dir] in self.available_dir:
                self.available_dir.remove(self.opposite_dir[self.last_dir])
        # If ghosts are scared, degenerate to the random agent
        if  np.any(np.array(self.ghost_status) in np.array([4, 5])):
            is_scared = True
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
            return choice, is_scared, is_suicide
        #Else if ghosts are scared
        P_G_distance = np.array([
            self.locs_df[self.cur_pos][each]
            for each in self.ghost_pos
        ])# distance between Pacman and ghosts
        P_R_distance = np.array([
            self.locs_df[self.cur_pos][each]
            for each in self.reward_pos
        ])  # distance between Pacman and rewards
        R_R_distance = np.array([
            self.locs_df[self.reborn_pos][each]
            for each in self.reward_pos
        ]) # distance between reborn point and rewards

        # # determine whether ghosts are closer than rewards
        # is_ghosts_closer = [np.all(each < P_R_distance) for each in P_G_distance] #TODO: do not need this

        # determine whether suicide is better
        is_suicide_better = [np.all(each < P_R_distance) for each in R_R_distance]
        closest_ghost_index = np.argmin(P_G_distance)
        # Suicide. Run to ghosts.
        if True in is_suicide_better:
            if True in is_suicide_better:
                is_suicide = True
                choice = self._relativeDir(
                    self.cur_pos,
                    self.adjaccent_path[
                        self.adjaccent_path.pos1 == self.cur_pos and self.adjaccent_path.pos2 == self.ghost_pos[
                            closest_ghost_index]
                        ].path.values[0])
                if choice is None:
                    choice = np.random.choice(range(len(self.available_dir)), 1).item()
                    choice = self.available_dir[choice]
        # Evade. Run away to the opposite direction
        elif self.last_dir is not None:
            cur_opposite_dir = self.opposite_dir[self.last_dir]
            if cur_opposite_dir in self.available_dir:
                choice = cur_opposite_dir
            else:
                choice = np.random.choice(range(len(self.available_dir)), 1).item()
                choice = self.available_dir[choice]
        else:
            choice = np.random.choice(range(len(self.available_dir)), 1).item()
            choice = self.available_dir[choice]
        return choice, is_scared, is_suicide

if __name__ == '__main__':
    import sys
    sys.path.append('./')
    from TreeAnalysisUtils import readAdjacentMap, readAdjacentPath, readLocDistance

    # Read data
    locs_df = readLocDistance("./extracted_data/dij_distance_map.csv")
    adjacent_data = readAdjacentMap("./extracted_data/adjacent_map.csv")
    adjacent_path = readAdjacentPath("./extracted_data/dij_distance_map.csv")

    # Suicide agent
    cur_pos = (22, 24)
    ghost_pos = [(21, 5), (22, 5)]
    ghost_status = [1, 1]
    reward_pos = [(13, 9)]
    last_dir = "right"
    agent = SuicideAgent(
        adjacent_data, adjacent_path, locs_df,
        cur_pos, ghost_pos, ghost_status, reward_pos, last_dir)
    choice, is_scared, is_suicide = agent.nextDir()
    print("Choice : ", choice)
    print("Is scared : ", is_scared)
    print("Is suicide : ", is_suicide)