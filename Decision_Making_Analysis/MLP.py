'''
Description:
    A simple multilayer perceptron. 
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/3/5
'''
import torch
import torch.nn as nn

class MLP(nn.Module):
    '''
        Description:
            The MLP with the input layer, one hidden layer, and the output layer.
            Every neuron uses the ReLu function as the activation function.
            The output is a 2-d vector because we have two classes.
            Use cross entropy loss as the loss function.
    '''
    def __init__(self, in_dim, batch_size = 5, lr = 1e-4, enable_cuda = False):
        super(MLP, self).__init__()
        # Initialize the network and parameters
        self.in_dim = in_dim
        self.out_dim = 2 # Two classification classes
        self.hid_dim = 32
        self.batch_size = batch_size
        self.lr = lr
        self.enable_cuda = enable_cuda
        # The hidden layer is a linear layer
        self.hidden_layer = nn.Linear(self.in_dim, self.out_dim)
        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        # Loss functions
        self.lossFunc = nn.BCELoss()
        # Activationj function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=0)

    def forward(self, input):
        '''
        Feddforward.
        :param input: The input with shape of (batch_size, number of features) 
        :return: The output vector:
            - output: (2,) ndarray denotes the probability of this input belonging to each class
        '''
        output = self.hidden_layer(input)
        output = self.relu(output)
        output = self.sigmoid(output)
        output = self.softmax(output)
        return output



if __name__ == '__main__':
    m = MLP(5)
    print(m)