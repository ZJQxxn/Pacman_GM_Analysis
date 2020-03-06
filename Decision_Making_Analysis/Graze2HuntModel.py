'''
Description:
    Model for analyzing the transfer probability from grazing model to the hunting model.
     
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/3/5
'''

import torch
import numpy as np

from util import np2tensor, tensor2np
from MLP import MLP


class Graze2Hunt:

    def __init__(self, in_dim, batch_size = 5, lr = 1e-3, enable_cuda = False):
        # Initialize parameters
        self.enable_cuda = enable_cuda
        self.batch_size = batch_size
        self.trained = False
        # Initialize the MLP model
        self.network = MLP(in_dim, batch_size, lr, self.enable_cuda).double()
        self.network = self.network.cuda() if self.enable_cuda else self.network

    def train(self, train_set, train_label):
        '''
        Train the analyzer.
        :param train_set: ndarray with shape of (number of samples, number of features) 
        :param train_label: ndarray with shape of (number of samples, 2). Label denotes whether the next time step, 
                            the Pacman is the in hunting mode or not ([1,0] means still in grazing mode and [0,1] means 
                            in hunting mode). 
        :return: Loss for every batch:
            - batch_loss: A list of loss for each batch.
        '''
        batch_data = []
        batch_label = []
        batch_loss = []
        batch_count = 0 # Count the number of batches
        for index, input in enumerate(train_set):
            batch_data.append(input)
            batch_label.append(train_label[index, :])
            if 0 == (index + 1) % self.batch_size:
                batch_count += 1
                batch_data = np.array(batch_data, dtype = np.double)
                batch_label = np.array(batch_label, dtype = np.double)
                loss = self._trainBatch(batch_data, batch_label)
                batch_loss.append(loss)
                loss.backward()
                self.network.optimizer.step()
                # Collect info and clear the batch
                print("The average loss for the {}-th batch is {}".format(batch_count, loss / self.batch_size))
                batch_data, batch_label = [], []
        self.trained = True
        return batch_loss

    def _trainBatch(self, batch_data, batch_label):
        '''
        Train with the batch data.
        :param batch_data: A batch of data. ndarray with shape of (batch_size, number of features)
        :param batch_label: A batch of label. ndarray with shape of (batch_size, nyumber of  features)
        :return: Total loss for the batch.
        '''
        self.network.zero_grad()
        # Initialization for this batch
        total_loss = 0
        for i in range(self.batch_size):
            input = np2tensor(batch_data[i, :], cuda_enabled = False, gradient_required = False)
            label = np2tensor(batch_label[i, :], cuda_enabled = False, gradient_required = False)
            output = self.network(input)
            total_loss += self.network.lossFunc(output, label)
        return total_loss

    def test(self, testing_set):
        '''
        Validate the network on testing dataset.
        :param testing_set: Testing dataset with shape of (number of testing samples, number of features)
        :param testing_label: Lables of tsting data.
        :return: Average binary class entropy loss over testing data.
        '''
        pred_output = []
        test_num = len(testing_set)
        for i in range(test_num):
            input = np2tensor(testing_set[i, :], cuda_enabled=False, gradient_required=False)
            output = self.network(input)
            pred_output.append(tensor2np(output))
        return np.array(pred_output)

    def saveModel(self, filename):
        '''
            Save Pytorch network to a ``.pt'' file .
            :param filename: The filename.
            :return: VOID
        '''
        pars = self.network.state_dict()
        torch.save(pars, filename)

    def loadModel(self, filename):
        '''
            Load Pytorch network from ``.pt'' file.
            :param filename: Filename of .py file.
            :return: VOID
        '''
        pars = torch.load(filename, map_location=torch.device('cpu'))
        self.network.load_state_dict(pars)
        self.trained = True


if __name__ == '__main__':
    m = Graze2Hunt(5, batch_size=5)
    print(m.network)
    data = np.random.random((100, 5))
    label = np.vstack(
        (np.tile([[1],[0]], 50).T,
         np.tile([[0], [1]], 50).T)
    )
    m.train(data, label)