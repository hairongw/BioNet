import os 
import torch 
import pandas as pd
import matplotlib.pyplot as plt
from run import *

LR = 1e-4 
input_size = 280
output_sizes = [2]
batch_size_label = 128 
hidden_sizes = [2048, 2048] 
Epoch = 7 
l2_lambda = 0 
K = 5 
accs= [] 
accs_inclass=[]
Permute = 1 

pred_all = []
for perm in range(Permute):
  prediction = []
  for k in range(K):  
    print('Beginning run_finetune {}, Total runs {}...'.format(perm * K + k+1, K * Permute))
    acc, acc_inclass, running_loss, ft_model, MAE, results = run_finetune(LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, perm, batch_size_label, l2_lambda)
    accs.append(acc)
    accs_inclass.append(acc_inclass)
    prediction.append(results)
    print('MAE =', MAE)
    print(results.size())

    # # plot current run 
    # plt.figure(1)
    # running_loss = np.array(running_loss)[:,0]
    # plt.plot(running_loss, label='loss')

    # plt.figure(2)
    # acc = np.array(acc)
    # acc_inclass = np.array(acc_inclass)
    # plt.plot(acc[:,0], label='NeuN')
    # plt.legend()

    # plt.show()

    if k == 0:
      break 

  prediction = torch.cat(prediction, dim=-1) 
  
  pred_all.append(prediction)
