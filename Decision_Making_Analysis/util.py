import numpy as np
import torch
import pandas as pd


def tensor2np(torch_tensor, cuda_enabled = False):
    '''
    Convert a Pytorch tensor to Numpy ndarray.
    :param torch_tensor: Pytorch tensor.
    :param cuda_enabled: Whether CUDA is available, either True or False (set to False by default).
    :return: A Numpy ndarray.
    '''
    if cuda_enabled:
        return torch_tensor.data.cpu().numpy()  # pull data from GPU to CPU
    else:
        return torch_tensor.data.numpy()

def np2tensor(nparray, cuda_enabled = False, gradient_required = True):
    '''
    Convert a Numpy ndarray to a Pytorch tensor.
    :param nparray: A numpy ndarray.
    :param block_size: Block size, should be a positive integer (set to 1 by default).
    :param cuda_enabled: Whether CUDA is available, either True or False (set to False by default).
    :param gradient_required: Whether automatic gradient computing is required, either True of False (set to True by default).
    :return: A Pytorch tensor with shape of (1, block_size, -1).
    '''
    if cuda_enabled:
        return torch.tensor(torch.tensor(nparray).cuda(), requires_grad=gradient_required)
    else:
        return torch.tensor(nparray).clone().detach().requires_grad_(gradient_required)

def oneHot(value, val_list):
    res = list(
        np.zeros((len(val_list,), ), dtype = np.int)
    )
    if value not in val_list:
        raise ValueError('Undefined value ``{}''! Can not convert the value to an one-hot vector.'.format(value))
    else:
        res[val_list.index(value)] = 1
    return res

def tuple_list(l):
    return [tuple(a) for a in l]

def maptodict(ghost_pos):
    # ghost_pos = pd.DataFrame(ghost_pos)
    map_info = pd.read_csv("data/map_info_brian.csv")
    map_info = map_info.assign(pacmanPos=tuple_list(map_info[["Pos1", "Pos2"]].values))
    map_info_mapping = {
        "up": "Next1Pos",
        "left": "Next2Pos",
        "down": "Next3Pos",
        "right": "Next4Pos",
    }
    d_dict = {}
    for d in ["up", "down", "right", "left"]:
        pos = tuple(
            map_info.loc[
                list(map_info.pacmanPos == tuple(ghost_pos)),
                [map_info_mapping[d] + "1", map_info_mapping[d] + "2"],
            ].values[0]
        )
        if pos != (0, 0):
            d_dict[d] = pos
    return d_dict

def estimateGhostLocation(locs_df, ghosts_loc, move_dir, remained_time, pacman_loc):
        '''
        Estimate the future locations of two ghosts.
        :param ghosts_loc: The current location of two ghosts, with shape of (2,2).
        :param move_dir: Moving direction of two ghosts, with shape of (2,). 
                        The direction should be selected from up/down/left/right.
        :param remained_time: Remained scaringtime for two ghosts.
        :param pacman_loc: The current location of Pacman.
        :return: The future locations of two ghosts, with shape of (2, 2).
        '''
        return [
            future_position(locs_df, ghosts_loc[0], move_dir[0], remained_time[0], pacman_loc),
            future_position(locs_df, ghosts_loc[1], move_dir[1], remained_time[1], pacman_loc),
        ]

def future_position(locs_df, ghost_pos, ghost_dir, t, pacman_pos):
    '''
    Infer the future location of one ghost.
    :param locs_df: Locs_df.
    :param ghost_pos: Current position of ghost, with shape of (2, ).
    :param ghost_dir: The moving firection of the ghost (str).
    :param t: Remaining scared time of the ghost (float).
    :param pacman_pos: The current location of Pacman, with shape of (2, )
    :return: Future ghost location at the end of the scared status.
    '''
    if t == 0:
        return ghost_pos
    history = [ghost_pos]
    for i in range(int(t // 2)):
        d_dict = {}
        for key, val in maptodict(ghost_pos).items():
            val = list(val)
            if val not in history:
                d_dict[key] = val
        if i == 0 and ghost_dir in d_dict.keys():
            ghost_pos = d_dict[ghost_dir]
        else:
            dict_df = pd.DataFrame.from_dict(d_dict, orient="index")
            dict_df["poss_pos"] = tuple_list(dict_df[[0, 1]].values)
            try:
                ghost_dir, ghost_pos = (
                    locs_df[(locs_df.pos1 == tuple(pacman_pos))]
                        .merge(dict_df.reset_index(), left_on="pos2", right_on="poss_pos")
                        .sort_values(by="dis")[["index", "poss_pos"]]
                        .values[-1]
                )
            except:
                return pacman_pos
        history.append(ghost_pos)
    return ghost_pos

def computeLocDis(map_distance, pacman_loc, other_loc):
    '''
    Compute the distance between Pacman and a list of location positions.
    :param map_distance: The distance betweeen each two points on the map.
    :param pacman_loc: The Pacman location.
    :param other_loc: A list of locations of other points.
    :return: A list of distance.
    '''
    if len(other_loc) == 0:
        return [0]
    pacman_loc = tuple(pacman_loc)
    other_loc = [tuple(each) for each in other_loc]
    distance = []
    for index in range(len(other_loc)):
        try:
            distance.append(map_distance[pacman_loc][other_loc[index]])
        except:
            # If the point doesn't exist, search around it
            offset = 1
            found = False
            around_loc = other_loc[index]
            while not found:
                if (around_loc[0] - offset,around_loc[1]) in map_distance[pacman_loc]:
                    other_loc[index] = (around_loc[0] - offset,around_loc[1])
                    found = True
                elif (around_loc[0] + offset,around_loc[1]) in map_distance[pacman_loc]:
                    other_loc[index] = (around_loc[0] - offset,around_loc[1])
                    found = True
                elif (around_loc[0], around_loc[1] - offset) in map_distance[pacman_loc]:
                    other_loc[index] = (around_loc[0], around_loc[1] - offset)
                    found = True
                elif (around_loc[0], around_loc[1] + offset) in map_distance[pacman_loc]:
                    other_loc[index] = (around_loc[0], around_loc[1] + offset)
                    found = True
                else:
                    offset += 1
            distance.append(map_distance[pacman_loc][other_loc[index]])
    return distance

