''''
Description:
    Model for analyzing the transfer probability from hunting model to the grazing model.

Author:
    Jiaqi Zhang < zjqseu @ gmail.com >

Date:
    2020 / 3 / 5
'''

import torch
import torch.nn as nn
import numpy as np
import copy

from util import np2tensor, tensor2np


class Hunt2Graze:

    def __init__(self):
        pass

    def _estimateGhostLocation(self):
        pass

    def _computePursuingTime(self):
        pass

    def training(self):
        # train with MLP (input are computed features, output is the label prob)
        pass