import numpy as np
import torch

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

import pandas as pd
import numpy as np

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