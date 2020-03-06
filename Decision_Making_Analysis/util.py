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