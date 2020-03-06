'''
Description:
    Evaluating the model performance.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/3/6
'''

import torch
import numpy as np
from util import np2tensor, tensor2np


def binaryClassError(pred_label, true_label):
    lossFunc = torch.nn.BCELoss()
    total_loss = 0
    sample_num= len(true_label)
    for i in range(sample_num):
        label = np2tensor(true_label[i, :], cuda_enabled=False, gradient_required=False)
        prediction = torch.tensor(pred_label[i])
        total_loss += lossFunc(prediction, label)
    return total_loss / sample_num


def AUC(pred_label, true_label):
    # TODO: AUC for the classfication result
    pass