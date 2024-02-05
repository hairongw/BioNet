import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from itertools import cycle
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
import pandas as pd
import math
import sklearn.preprocessing as sk
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from PIL import Image
from matplotlib import cm
import copy
import os
from os import listdir
import re
from torch.optim.lr_scheduler import StepLR

from scipy.stats import norm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/RecurrentGBM_soft classification_all features/")
cwd = os.getcwd()
print('working directory {}'.format(cwd))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Select the Runtime > "Change runtime type" menu to enable a GPU accelerator, ')
  print('and then re-execute this cell.')
else:
  print(gpu_info)


def train_split_neighbors(file, k, K, seed):
  # generate train-test split for labeled-data
  labeled_data = pd.read_csv(file, sep=',')
  n = labeled_data.shape[0]
  # random shuffle data
  np.random.seed(seed)
  shuffle_idx = np.random.permutation(n)
  labeled_data = labeled_data.iloc[shuffle_idx, :]
  labeled_data.reset_index(drop=True)

  os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/RecurrentGBM_soft classification_all features/")
  cwd = os.getcwd()
  neighbors = pd.read_csv('biopsies69_neighbors_3marker_all5_threshold0.5.csv', sep=',')

  slices = np.linspace(0, n, K+1, dtype=int)
  lb = slices[k]
  ub = slices[k+1]
  train_idx = list(range(lb)) + list(range(ub, n))

  train_labeled_data = labeled_data
  train_labeled_data = labeled_data.iloc[train_idx, :]   # train by all data if delete this line
  train_labeled_data.to_csv('train_labeled_data.csv', index=False)

  train_labeled_all = labeled_data
  train_labeled_all.to_csv('all_labeled_data.csv', index=False)

  train_labeled_AVID = train_labeled_data['AVID']
  df = neighbors.merge(train_labeled_AVID, how='inner', on='AVID')
  train_labeled_neighbors = pd.concat([df, train_labeled_data], axis = 0)
  train_labeled_neighbors.to_csv('train_labeled_neighbors.csv', index=False)

  all_labeled_AVID = train_labeled_all['AVID']
  df = neighbors.merge(all_labeled_AVID, how='inner', on='AVID')
  all_labeled_neighbors = pd.concat([df, train_labeled_all], axis = 0)
  all_labeled_neighbors.to_csv('all_labeled_neighbors.csv', index=False)

  test_labeled_data = labeled_data.iloc[lb:ub, :]
  test_labeled_data.to_csv('test_labeled_data.csv', index=False)

  test_labeled_AVID = test_labeled_data['AVID']
  df2 = neighbors.merge(test_labeled_AVID, how='inner', on='AVID')
  test_labeled_neighbors = pd.concat([df2, test_labeled_data], axis = 0)
  test_labeled_neighbors.to_csv('test_labeled_neighbors.csv', index=False)



# Define custom dataset
class BrainTumorDataset(Dataset):
    def __init__(self, file_list=['biopsies69_neighbors_3marker_all5_threshold0.5.csv'], weight_display=False): #biopsies_scaledNeuN --- acc only on biopsies
        self.patient_id = []
        self.continuous = []
        self.label = []
        self.location = []
        self.avid = []

        for i, csv_file in enumerate(file_list):
          data = pd.read_csv(csv_file, sep=',')
          size = data.shape[0]
          self.patient_id.append(np.array(data.iloc[:,0]))
          self.location.append(np.array(data.iloc[:,2:5]))
          self.features = np.array(data.iloc[:, -336:])
          continuous_feature_idx = list(range(0, 224)) + list(range(280, 336))
          self.continuous_feature = self.features[:, continuous_feature_idx]
          self.continuous.append(self.continuous_feature) #336

          self.label.append(np.array(data.iloc[:,6:8])) #scaled scores of SOX2 and CD68
          self.avid.append(np.array(data.iloc[:,1]))

        self.patient_id = np.concatenate(self.patient_id)
        self.location = np.concatenate(self.location)
        self.continuous = np.concatenate(self.continuous)
        self.label = np.concatenate(self.label)
        self.avid = np.concatenate(self.avid)

        num_A0 = sum(self.label[:,0]<0.5)
        num_A1 = sum(self.label[:,0]>=0.5)
        num_B0 = sum(self.label[:, 1]<0.50)
        num_B1 = sum(self.label[:, 1]>=0.5)
        total_size = self.label.shape[0]

        self.percentage = [num_A0/total_size, num_A1/total_size]




    def __len__(self):
        return len(self.patient_id)

    def __getitem__(self, idx):
        y1 = self.label[idx,0]
        y2 = self.label[idx,1]

        # col_idx = [i for i in range(0, 336)] # all contrasts
        col_idx = [i for i in range(0, 224)] + [i for i in range(280, 336)] 
        x = self.continuous[:,col_idx]
        x_continuous = x[idx,:]
        x_location = self.location[idx, 0:3]
        x_patient_id = self.patient_id[idx]
        x_avid = self.avid[idx]
        sample = {'SOX2': y1,'CD68': y2, 'x_continuous':x_continuous, 'x_location':x_location, 'patient_id':x_patient_id, 'AVID':x_avid}
        return sample


class SimpleMTL(nn.Module):
  def __init__(self, input_size, output_sizes, hidden_sizes):
    super(SimpleMTL, self).__init__()

    self.hidden_sizes = hidden_sizes.copy()
    # self.hidden_sizes.insert(0, input_size)
    self.towers = []
    self.tasks = len(output_sizes)
    # define tower and skip connection for each task
    for output_size in output_sizes:
      tower = []
      for j in range(len(self.hidden_sizes)-1):
          tower.append(nn.Dropout(p=1e-2)) #p = droprate, default = 0.5 should be smaller than 0.0001 since network is too small
          tower.append(nn.Linear(self.hidden_sizes[j], self.hidden_sizes[j+1]))
          tower.append(nn.ReLU())
      tower.append(nn.Linear(self.hidden_sizes[-1], output_size))
      self.towers.append(nn.Sequential(*tower))
    self.towers = nn.ModuleList(self.towers)

  def predict(self, x):
    with torch.no_grad():
      y = []
      logits = self.forward(x)
      for task in range(self.tasks):
        logit = logits[task]

        ytask = torch.zeros(len(logit))
        for i in range(len(logit)):
          if logit[i] > 0.5:
            ytask[i] = 1
          else:
            ytask[i] = 0

        y.append(ytask)
      return y


  def forward(self, x):
    outputs = []
    for tower in self.towers:
      outputs.append(tower(x))
    return outputs

def mtl_loss(x, y, criterion, model): 
    logits = model(x) 
    loss = 0
    for (i, logit) in enumerate(logits):
      loss += criterion(logit, y[i])

    return loss

def l_loss(logit, y): 
  loss_criterion = nn.L1Loss() #nn.L1Loss() #nn.MSELoss()
  wloss = loss_criterion(logit.view(-1), y)
  return wloss

def clean_loss(x_labeled, y, weight, model):
  loss = mtl_loss(x_labeled, y, l_loss, model) #+ consistency_loss(x_labeled, model, optimizer, step_size_adv, delta, perturb_steps, alpha_adv))/x_labeled.shape[0]
  return loss


def train(Epoch, train_labeled_loader, test_loader, model, eval, loss, optimizer, weight): # passing optional arguments as dictionary step_size=0.002, epsilon=0.02, perturb_steps=1, beta=1.0
  running_loss = []
  acc = []
  acc_inclass = []

  scheduler = StepLR(optimizer, step_size=8, gamma=0.1)
  model.train()
  for epoch in range(Epoch):
    for i, sample in enumerate(train_labeled_loader, 0):
      # print('epoch: {}, iteration: {}'.format(epoch, i))
      y1, y2, x_continuous = sample['SOX2'].float().to(device), sample['CD68'].float().to(device), sample['x_continuous'].float().to(device)
      x_labeled = x_continuous

      # zero grad
      optimizer.zero_grad()

      loss_all = loss(x_labeled, [y1, y2], weight, model)

      loss_all.backward()

      optimizer.step()

      # # check gradient norm
      # total_norm = 0
      # # for p in model.parameters():
      # #   param_norm = p.grad.data.norm(2)
      # #   total_norm += param_norm.item() ** 2
      # # total_norm = total_norm ** (1. / 2)

      running_loss.append([loss_all.item()])


    # evaluating on test set after after epoch
    ac, ac_inclass, mae, results = eval(test_loader, model, epoch+1, running_loss)
    acc.append(ac)
    acc_inclass.append(ac_inclass)
    # adjust learning rate
    scheduler.step()

  # print('unlabeled files used: {}'.format(used_files))
  return acc, acc_inclass, running_loss, mae, results


def eval(loader, model, epoch, running_loss):
  # Set the model to evaluation mode
  model.eval()
  with torch.no_grad():
    correct_t1 = torch.zeros(2)
    correct_t2 = torch.zeros(2)
    total_t1 = torch.zeros(2)
    total_t2 = torch.zeros(2)
    all_AE1 = torch.zeros(1).to(device)
    all_AE2 = torch.zeros(1).to(device)
    results = []


    for i, sample in enumerate(loader, 0):
        y_logit1, y_logit2, x_continuous = sample['SOX2'].float().to(device), sample['CD68'].float().to(device), sample['x_continuous'].float().to(device)
        y1 = torch.zeros(len(y_logit1))
        y2 = torch.zeros(len(y_logit2))

        for j in range(len(y_logit1)):
          if y_logit1[j] >= 0.5: 
            y1[j] = 1
          else:
            y1[j] = 0
        for j in range(len(y_logit2)):
          if y_logit2[j] >= 0.5: 
            y2[j] = 1
          else:
            y2[j] = 0

        x = x_continuous
        yhat1 = model.predict(x)[0]
        yhat2 = model.predict(x)[1]

        y_pred_logit1 = model(x)[0]
        y_pred_logit2 = model(x)[1]

        y_pred1 = y_pred_logit1[:,0]
        y_pred2 = y_pred_logit2[:,0]

        ct1, tt1 = confusion(yhat1, y1)
        correct_t1 += ct1
        total_t1 += tt1

        ct2, tt2 = confusion(yhat2, y2)
        correct_t2 += ct2
        total_t2 += tt2

        # output predicted scores of two tasks
        pred_results = torch.stack((y_logit1, y_pred1, y_logit2, y_pred2))
        results.append(pred_results)

        abs_batch1 = torch.abs(y_logit1 - y_pred1).to(device)
        AE_batch1 = torch.sum(abs_batch1).to(device)
        all_AE1 += AE_batch1

        abs_batch2 = torch.abs(y_logit2 - y_pred2).to(device)
        AE_batch2 = torch.sum(abs_batch2).to(device)
        all_AE2 += AE_batch2

    results = torch.cat(results, dim=-1)

    #total acc
    acc1 = correct_t1.sum()/total_t1.sum()
    acc2 = correct_t2.sum()/total_t2.sum()
    MAE1 = all_AE1/ total_t1.sum()
    MAE2 = all_AE2/ total_t2.sum()

    #inclass acc
    # print('SOX2',correct_t1, total_t1)
    acc1_inclass = (correct_t1/total_t1).numpy()
    # print('CD68',correct_t2, total_t2)
    acc2_inclass = (correct_t2/total_t2).numpy()
    # print('test -- Epoch: {}, Accs for SOX2: {:4f}/{}, CD68: {:4f}/{}, total/clean/reg/grad: {}'.format(epoch, acc1, acc1_inclass, acc2, acc2_inclass, running_loss[-1] if running_loss else 'initial'))
    # print('SOX2 MAE = ', MAE1, 'CD68 MAE = ', MAE2)
    return [acc1, acc2], [acc1_inclass, acc2_inclass], [MAE1, MAE2], results

def confusion(yhat, y):
  correct_zero = ((yhat == y) * (y == 0)).sum().item()
  correct_one = ((yhat == y) * (y == 1)).sum().item()
  total_zero = (y == 0).sum().item()
  total_one = (y==1).sum().item()
  return torch.Tensor([correct_zero, correct_one]), torch.Tensor([total_zero, total_one])



def run(LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, seed, batch_size_label):
  os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/personalized model SOX2 CD68/"+ patient_id + "/")
  cwd = os.getcwd()
  test_data = BrainTumorDataset(['biopsies69_3marker_5_5_threshold0.5_' + patient_id +'.csv']) 
  train_split_neighbors('biopsies69_3marker_5_5_threshold0.5_no'+ patient_id +'.csv', k, K, seed)
  train_labeled_data = BrainTumorDataset(['all_labeled_neighbors.csv'], weight_display=True)
  train_labeled_loader = torch.utils.data.DataLoader(train_labeled_data, batch_size=batch_size_label, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_label, shuffle=True, num_workers=2)
  os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/personalized model SOX2 CD68/"+ patient_id + "/")
  cwd = os.getcwd()
  np.random.seed(seed)

  percentage = train_labeled_data.percentage
  percentage = torch.tensor([math.exp(-0*percentage[0]), math.exp(-0*percentage[1])])
  weight = percentage

  model = SimpleMTL(input_size, output_sizes, hidden_sizes).to(device)
  pytorch_total_params = sum(p.numel() for p in model.parameters())
  # print('num params', pytorch_total_params)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
  # evaluate initial model
  # print('evaluating random initialization')
  acc = []
  acc_inclass = []
  ac, ac_inclass, a, b = eval(test_loader, model, 0, [])
  acc.append(ac)
  acc_inclass.append(ac_inclass)
  # training
  # print('begin train')
  accf, accf_inclass, running_loss, mae, results = train(Epoch, train_labeled_loader, test_loader, model, eval, clean_loss, optimizer, weight)
  acc = acc + accf
  acc_inclass = acc_inclass + accf_inclass
  return acc, acc_inclass, running_loss, model, mae, results

def reult_process(df_all):
    # calculate AUC, ACC, MAE for each combination of hyper parameters
    pred = pd.concat(df_all, ignore_index=True)
    for i in range(10):
      print(i)
      SOX2_true = pred.iloc[:,4*i+0]
      SOX2_pred = pred.iloc[:,4*i+1]
      CD68_true = pred.iloc[:,4*i+2]
      CD68_pred = pred.iloc[:,4*i+3]

      SOX2_pred_label = (SOX2_pred >= 0.5)
      CD68_pred_label = (CD68_pred >= 0.5)
      SOX2_true_label = (SOX2_true >= 0.5)
      CD68_true_label = (CD68_true >= 0.5)

      # ACC
      acc_SOX2 = accuracy_score(SOX2_true_label, SOX2_pred_label)
      acc_CD68 = accuracy_score(CD68_true_label, CD68_pred_label)
      # print(acc_SOX2, acc_CD68)

      # AUC
      fpr = dict()
      tpr = dict()
      fpr, tpr, _ = roc_curve(SOX2_true_label.to_numpy(), SOX2_pred.to_numpy())
      roc_auc_SOX2 = auc(fpr, tpr)
      fpr, tpr, _ = roc_curve(CD68_true_label.to_numpy(), CD68_pred.to_numpy())
      roc_auc_CD68 = auc(fpr, tpr)
      # print(roc_auc_SOX2, roc_auc_CD68)

      # MAE
      MAE_SOX2 = (SOX2_true - SOX2_pred).abs().mean()
      MAE_CD68 = (CD68_true - CD68_pred).abs().mean()
      # print('MAE', MAE_SOX2,MAE_CD68)

      roc_auc_SOX2 = round(roc_auc_SOX2, 3)
      roc_auc_CD68 = round(roc_auc_CD68, 3)
      acc_SOX2 = round(acc_SOX2, 3)
      acc_CD68 = round(acc_CD68, 3)
      MAE_SOX2 = round(MAE_SOX2, 3)
      MAE_CD68 = round(MAE_CD68, 3)

      print("AUC", roc_auc_SOX2, roc_auc_CD68,
            "ACC", acc_SOX2, acc_CD68,
            "MAE", MAE_SOX2, MAE_CD68)

    return pred, roc_auc_SOX2, roc_auc_CD68, acc_SOX2, acc_CD68, MAE_SOX2, MAE_CD68

os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/RecurrentGBM_soft classification_all features/")
cwd = os.getcwd()
df = pd.read_csv('biopsies69_3marker_5*5_threshold0.5.csv',sep=',')
patients = df["Patient"]
p = patients.drop_duplicates().to_list()
path = r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/NN Grid search updated/"
if not os.path.exists(path):
  os.makedirs(path)

# p_done = [] #["CU1275", "CU1269", "CU1324", "CU1265","CU1253", "CU1101", "CU1154"] #, "CU1324", "CU1275", "CU1265","CU1253", "CU1101", "CU1154"
# p_used = [e for e in p if e not in p_done]
# print(len(p_used))

# chunks = [p_used[x:x+4] for x in range(0, len(p_used), 4)]
# print(len(chunks), len(chunks[0]))

# starting = 0
# print(len(chunks[starting:]))

# for i in range(len(chunks[starting:])):
for i in range(1):
  # print(i+starting)
  pred_all = []
  for patient_id in ["CU1319"]: #chunks[i+starting]: #["CU1077"]:
    print(patient_id)

    os.chdir(r"/content/drive/MyDrive/Project 1: Recurrent GBM/5 contrasts AAE/personalized model SOX2 CD68/"+ patient_id + "/")
    cwd = os.getcwd()

    LR = 1e-3
    input_size = 280 
    output_sizes = [1, 1]
    batch_size_label = 128 
    hidden_sizes = [256, 256, 256, 256] 
    Epoch = 32 
    K = 5 
    accs= []
    accs_inclass=[]

    prediction = []
    for k in range(K):
      perm = 1
      acc, acc_inclass, running_loss, model, mae, results = run(LR, input_size, output_sizes, hidden_sizes, Epoch, k, K, perm, batch_size_label)
      accs.append(acc)
      accs_inclass.append(acc_inclass)
      prediction.append(results)

      plt.figure(1)
      running_loss = np.array(running_loss)[:,0]
      plt.plot(running_loss, label='loss')

      plt.figure(2)
      acc = np.array(acc)
      acc_inclass = np.array(acc_inclass)
      plt.plot(acc[:,0], label='SOX2')
      plt.plot(acc[:,1], label='CD68')
      plt.legend()

      plt.show()

      if k == 0:
        break

    prediction = torch.cat(prediction, dim=-1)
    a1 = torch.transpose(prediction,0,1).cpu().numpy()
    import pandas as pd
    from functools import reduce
    DF1 = pd.DataFrame(a1)
    DF1.columns = ['SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred','SOX2 score', 'SOX2 pred', 'CD68 score', 'CD68 pred']
    pred_all.append(DF1)

  pred, roc_auc_SOX2, roc_auc_CD68, acc_SOX2, acc_CD68, MAE_SOX2, MAE_CD68 = reult_process(pred_all)
  mean_AUC = (roc_auc_SOX2 + roc_auc_CD68)/2
  file_name = str(mean_AUC) + "_" + str(LR) + "_" + str(batch_size_label) + "_" + str(hidden_sizes) + "_" + str(Epoch) + "_part" + str(patient_id) + ".csv"
  pred.to_csv(path + file_name, index=False)
  print("----- finish one cluster ----- ", i)
