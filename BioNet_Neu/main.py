import os 
import torch 
import pandas as pd
import matplotlib.pyplot as plt
from run import *
# 1e-3 ,4, 32

LR = 1e-3
input_size = 280 
output_sizes = [2]
batch_size_label = 128 
hidden_sizes = [2048, 2048] 
Epoch = 8 
K = 1
accs= [] 
accs_inclass=[]
Permute = 1 

for perm in range(Permute):
  prediction = []
  for k in range(K):  
    print('Beginning run {}, Total runs {}...'.format(perm * K + k+1, K * Permute))
    acc, acc_inclass, running_loss, model, mae, results = run(LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, perm, batch_size_label)
    accs.append(acc)
    accs_inclass.append(acc_inclass)
    prediction.append(results)
    print('final mae', mae)
    print(results.size())
      
  prediction = torch.cat(prediction, dim=-1)

os.chdir(r"/content/drive/MyDrive/RecurrentGBM_soft_classification_all_features/") 
cwd = os.getcwd()
torch.save(model.state_dict(), 'pre_trained_model.pth')