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
from sklearn import metrics

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
    auc = metrics.roc_auc_score(true_label, pred_label)
    return auc

def correctRate(pred_label, true_label):
    pred_label = np.array(np.round(pred_label), dtype = np.int)
    true_label = np.array(true_label, dtype = np.int)
    total_count = len(true_label)
    correct_count = 0
    for index in range(total_count):
        if np.all(pred_label[index] == true_label[index]):
            correct_count += 1
    return correct_count / total_count