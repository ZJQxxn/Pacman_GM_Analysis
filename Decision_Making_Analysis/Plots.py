'''
Description:
    Plotting the batch loss for the MLP.
    
Author:
    Jiaqi Zhang <zjqseu@gmail.com>
    
Date:
    2020/3/6
'''
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

mat_file = sio.loadmat('MLP_train_batch_loss_batch1.mat')
# mat_file = sio.loadmat('train_batch_los.mat') # without ReLu

batch_loss = np.array(mat_file['batch_loss']).squeeze()[:500]
batch_size = mat_file['batch_size'][0][0]
# print(batch_loss / batch_size)

plt.title('Batch Binary Class Entropy Loss', fontsize = 30)
plt.plot(np.arange(len(batch_loss)), batch_loss / batch_size)
# plt.yticks(np.arange(0,1.1,0.2), fontsize = 30)
# plt.xticks(np.arange(len(batch_loss)), range(len(batch_loss)), fontsize = 30)
plt.yticks(fontsize = 30)
plt.xticks([], [], fontsize = 30)
plt.xlabel('Batch', fontsize = 30)
plt.ylabel('BCE Loss', fontsize = 30)
plt.show()